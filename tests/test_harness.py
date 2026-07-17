"""Harness-level tests.

Groups:
  1. The constant-key exog column dispatch (32-column cap) and its named
     failure beyond the cap.
  2. v2 end-to-end acceptance: the PUBLIC sarimax_fit / sarimax_forecast /
     sarimax_summary / sarimax_residuals path against the fixtures_v2 golden
     references (trend, concentrated scale, missing values,
     simple_differencing=False). Fit acceptance mirrors tests/test_estimate.py:
     a fixture passes as
       - "params":  constrained AND unconstrained params match statsmodels at
                    1e-6 abs, loglik agrees at rel <= 1e-8;
       - "ll_won":  our loglik beats the fixture params' loglik BOTH evaluated
                    through the same v2 engine (dll > 1e-8) -- re-baselining;
       - "ll_tie":  |dll| <= 1e-8 with params within 1e-3.
     In all paths our loglik must not be materially worse than statsmodels'
     reported optimum (>= sm - 1e-8). Forecasts are compared at rel tolerance
     (mean 1e-6, se 1e-5) only when the "params" path was taken; otherwise the
     optima differ so only sanity (finite, lo < yhat < hi) is asserted.
     AIC/BIC: k counts sigma2 even when concentrated and the BIC n is
     nobs_effective = n_eff - burn (both pinned by fixtures_v2 fitted_meta).
  3. v1 public-path regression: arma_1_0_1 (a v1 fixture) fit through the NEW
     public path with default flags; same acceptance rule against the v1
     fixture -- proves the engine swap kept public v1 behavior.
"""
import math
import os

import duckdb
import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIXDIR = os.path.join(HERE, "fixtures")
FIXDIR_V2 = os.path.join(HERE, "fixtures_v2")

SQL_FILES = ["sql/00_linalg.sql", "sql/01_prep.sql", "sql/02_ssm.sql",
             "sql/03_filter.sql", "sql/04_estimate.sql", "sql/05_forecast.sql",
             "sql/06_harness.sql"]

# fixtures_v2 cases run through the public fit (nodiff_sarimax_011_011_12 is
# deliberately not fit here: k_states = 27 makes the full BFGS a many-minute
# run; its engine-level agreement is covered by the layer suites)
V2_FIT_CASES = ["nodiff_arima_111", "trend_ct_arma_100", "conc_arma_101",
                "missing_arma_101", "kitchen_sink"]

FIT_V2 = {}          # fixture -> dict(path=..., model table df, ...)


@pytest.fixture(scope="module")
def con():
    c = duckdb.connect()
    c.execute("SET lambda_syntax = 'DISABLE_SINGLE_ARROW'")
    for f in SQL_FILES:
        with open(os.path.join(ROOT, f), encoding="utf-8") as fh:
            c.execute(fh.read())
    return c


# ---------------------------------------------------------------------------
# 1. exog column dispatch
# ---------------------------------------------------------------------------

def make_wide_table(c, ncols, n=20, name="_wide"):
    rng = np.random.default_rng(ncols)
    cols = ", ".join(f"x{j} DOUBLE" for j in range(1, ncols + 1))
    c.execute(f"CREATE OR REPLACE TABLE {name} (t BIGINT, y DOUBLE, {cols})")
    vals = rng.standard_normal((n, ncols + 1))
    ph = ", ".join(["?"] * (ncols + 2))
    c.executemany(f"INSERT INTO {name} VALUES ({ph})",
                  [(i + 1, *map(float, vals[i])) for i in range(n)])
    return [f"x{j}" for j in range(1, ncols + 1)], vals[:, 1:]


@pytest.mark.parametrize("r", [1, 13, 32])
def test_exog_extraction_up_to_32(con, r):
    names, vals = make_wide_table(con, r)
    lst = "[" + ", ".join(f"'{n}'" for n in names) + "]"
    got = con.execute(
        f"SELECT t, j, x FROM _sarimax_exog_of('_wide', {lst}, 'y', 't') "
        f"ORDER BY j, t").df()
    assert len(got) == 20 * r
    for j in range(1, r + 1):
        np.testing.assert_array_equal(
            got[got.j == j]["x"].to_numpy(), vals[:, j - 1],
            err_msg=f"column {j} extracted wrong values")


def test_exog_extraction_errors_beyond_32(con):
    names, _ = make_wide_table(con, 33)
    lst = "[" + ", ".join(f"'{n}'" for n in names) + "]"
    with pytest.raises(duckdb.Error, match="at most 32 exogenous columns"):
        # aggregate over x so the projection is actually evaluated
        # (count(*) would never touch the CASE and the lazy error can't fire)
        con.execute(
            f"SELECT sum(x) FROM _sarimax_exog_of('_wide', {lst}, 'y', 't')")


def test_exog_extraction_empty_list(con):
    make_wide_table(con, 1)
    got = con.execute(
        "SELECT count(*) FROM _sarimax_exog_of('_wide', []::VARCHAR[], 'y', 't')"
    ).fetchone()[0]
    assert got == 0


# ---------------------------------------------------------------------------
# 2. v2 end-to-end against fixtures_v2
# ---------------------------------------------------------------------------

def load_v2(fx, name):
    return pd.read_parquet(os.path.join(FIXDIR_V2, fx, name + ".parquet"))


def v2_trend_arg(fx):
    degs = load_v2(fx, "trend")["degree"].tolist()
    return {(): "n", (0,): "c", (1,): "t", (0, 1): "ct"}[tuple(degs)]


def setup_v2_user_tables(c, fx, prefix="_hv"):
    """User-facing tables for the public macros: <prefix>_dat (t, y, x1..xr)
    in-sample with NULL y at the fixture's missing points, <prefix>_fut the
    future exog rows. Returns (spec row, exog col names)."""
    spec = load_v2(fx, "spec").iloc[0]
    n, r = int(spec["n"]), int(spec["r"])
    ser = load_v2(fx, "series").sort_values("t")
    cols = [f"x{j}" for j in range(1, r + 1)]
    coldefs = "".join(f", {cname} DOUBLE" for cname in cols)
    c.execute(f"CREATE OR REPLACE TABLE {prefix}_dat (t BIGINT, y DOUBLE{coldefs})")
    c.execute(f"CREATE OR REPLACE TABLE {prefix}_fut (t BIGINT{coldefs})")
    if r:
        exog = load_v2(fx, "exog")
        wide = exog.pivot(index="t", columns="j", values="x").sort_index()
        xall = wide.to_numpy()
    for _, row in ser.iterrows():
        t = int(row["t"])
        # parquet stores missing y as NaN doubles; the engine's missing-value
        # convention is SQL NULL, so convert here
        yv = None if pd.isna(row["y"]) else float(row["y"])
        xv = [float(v) for v in xall[t - 1]] if r else []
        c.execute(f"INSERT INTO {prefix}_dat VALUES ({', '.join(['?'] * (2 + r))})",
                  [t, yv] + xv)
    if r:
        for t in range(n + 1, xall.shape[0] + 1):
            c.execute(f"INSERT INTO {prefix}_fut VALUES ({', '.join(['?'] * (1 + r))})",
                      [t] + [float(v) for v in xall[t - 1]])
    return spec, cols


def engine_ll_at(c, fx, spec, params, prefix="_hll"):
    """Evaluate a constrained parameter vector through the v2 engine's
    loglikelihood on the fixture's (raw, sdiff=0) data."""
    p, d, q, P, Q, s, r = (int(spec[k]) for k in
                           ["p", "d", "q", "bigp", "bigq", "s", "r"])
    D = int(spec["bigd"])
    ktrend, conc, n = int(spec["ktrend"]), bool(spec["conc"]), int(spec["n"])
    ser = load_v2(fx, "series").sort_values("t")
    c.execute(f"CREATE OR REPLACE TABLE {prefix}_y (t BIGINT, y DOUBLE)")
    c.executemany(f"INSERT INTO {prefix}_y VALUES (?, ?)",
                  [(int(t), None if pd.isna(v) else float(v))
                   for t, v in zip(ser["t"], ser["y"])])
    c.execute(f"CREATE OR REPLACE TABLE {prefix}_x (t BIGINT, j INT, x DOUBLE)")
    if r:
        exog = load_v2(fx, "exog")
        ins = exog[exog.t <= n]
        c.executemany(f"INSERT INTO {prefix}_x VALUES (?, ?, ?)",
                      [(int(a), int(b), float(v))
                       for a, b, v in zip(ins["t"], ins["j"], ins["x"])])
    degs = load_v2(fx, "trend")
    c.execute(f"CREATE OR REPLACE TABLE {prefix}_degs (idx BIGINT, degree BIGINT)")
    if len(degs):
        c.executemany(f"INSERT INTO {prefix}_degs VALUES (?, ?)",
                      [(int(a), int(b)) for a, b in zip(degs["idx"], degs["degree"])])
    c.execute(f"CREATE OR REPLACE TABLE {prefix}_pr (probe_id BIGINT, params DOUBLE[])")
    c.execute(f"INSERT INTO {prefix}_pr VALUES (1, ?)", [list(map(float, params))])
    c.execute(f"""CREATE OR REPLACE TABLE {prefix}_sys AS
                  SELECT * FROM _sarimax_systems_v2('{prefix}_pr', {r}, {p}, {q},
                      {P}, {Q}, {max(s, 1)}, {d}, {D}, {ktrend}, {conc})""")
    c.execute(f"""CREATE OR REPLACE TABLE {prefix}_obs AS
                  SELECT * FROM _sarimax_obs_adj_v2('{prefix}_y', '{prefix}_x',
                      '{prefix}_pr', {r}, {ktrend}, '{prefix}_degs')""")
    return float(c.execute(
        f"""SELECT loglik FROM _sarimax_loglik_v2('{prefix}_obs', '{prefix}_sys',
            {conc})""").fetchone()[0])


@pytest.mark.parametrize("fx", V2_FIT_CASES)
def test_v2_public_fit(con, fx):
    spec, cols = setup_v2_user_tables(con, fx)
    p, d, q, P, D, Q, s = (int(spec[k]) for k in
                           ["p", "d", "q", "bigp", "bigd", "bigq", "s"])
    conc = bool(spec["conc"])
    lst = "[" + ", ".join(f"'{cname}'" for cname in cols) + "]" if cols else "[]::VARCHAR[]"
    con.execute(f"""CREATE OR REPLACE TABLE _hv_m_{fx} AS
        SELECT * FROM sarimax_fit('_hv_dat', 'y', {p}, {d}, {q},
            sp := {P}, sd := {D}, sq := {Q}, s := {max(s, 1)},
            exog_cols := {lst}, t_col := 't',
            trend := '{v2_trend_arg(fx)}', concentrate := {str(conc).lower()},
            simple_differencing := false)""")
    m = con.execute(f"SELECT * FROM _hv_m_{fx}").df()

    fitted = load_v2(fx, "fitted").sort_values("k")
    meta = load_v2(fx, "fitted_meta").iloc[0]
    kp = int(spec["k_params"])

    got_p = m[m.kind == "param"].sort_values("idx")
    got_u = m[m.kind == "param_unc"].sort_values("idx")
    mm = {(r_.kind, r_.name): r_.value for r_ in
          m[m.kind == "meta"].itertuples()}

    # parameter naming and count (statsmodels param_names; sigma2 absent
    # when concentrated)
    assert list(got_p["name"]) == list(fitted["name"]), \
        f"{fx}: param names {list(got_p['name'])} != {list(fitted['name'])}"
    assert len(got_p) == kp

    our_ll = mm[("meta", "loglik")]
    sm_ll = float(meta["loglik"])
    assert mm[("meta", "converged")] == 1.0, f"{fx}: fit did not converge"
    assert np.isfinite(our_ll), f"{fx}: non-finite loglik"
    assert our_ll - sm_ll >= -1e-8, \
        f"{fx}: our optimum is WORSE than statsmodels by {sm_ll - our_ll:.3e}"

    dcon = float(np.abs(got_p["value"].to_numpy()
                        - fitted["constrained"].to_numpy()).max())
    dunc = float(np.abs(got_u["value"].to_numpy()
                        - fitted["unconstrained"].to_numpy()).max())
    if dcon <= 1e-6 and dunc <= 1e-6:
        path = "params"
        assert abs(our_ll - sm_ll) <= 1e-8 * max(1.0, abs(sm_ll)), \
            f"{fx}: params match but loglik differs"
    else:
        # re-baselining: both optima evaluated through the SAME v2 engine
        sm_ll_eng = engine_ll_at(con, fx, spec, fitted["constrained"].tolist())
        dll = our_ll - sm_ll_eng
        if dll > 1e-8:
            path = "ll_won"
        elif abs(dll) <= 1e-8 and dcon <= 1e-3:
            path = "ll_tie"
        else:
            pytest.fail(f"{fx}: params off (dcon {dcon:.3e}, dunc {dunc:.3e}) "
                        f"and loglik not better (dll {dll:.3e})")

    # AIC/BIC identities: k counts sigma2 even when concentrated; BIC's n is
    # nobs_effective = n_eff - burn (fixture-pinned statsmodels behavior)
    k_eff = kp + (1 if conc else 0)
    nobs_eff = float(spec["n_eff"]) - float(spec["burn"])
    assert math.isclose(mm[("meta", "aic")], 2.0 * k_eff - 2.0 * our_ll,
                        rel_tol=1e-12), f"{fx}: aic formula"
    assert math.isclose(mm[("meta", "bic")],
                        k_eff * math.log(nobs_eff) - 2.0 * our_ll,
                        rel_tol=1e-12), f"{fx}: bic formula"
    if path == "params":
        assert abs(mm[("meta", "aic")] - float(meta["aic"])) <= 1e-6 * max(
            1.0, abs(float(meta["aic"]))), f"{fx}: aic vs fixture"
        assert abs(mm[("meta", "bic")] - float(meta["bic"])) <= 1e-6 * max(
            1.0, abs(float(meta["bic"]))), f"{fx}: bic vs fixture"
    assert mm[("meta", "sigma2")] > 0.0, f"{fx}: sigma2 not positive"

    # spec rows carry the v2 flags
    sp_rows = dict(zip(m[m.kind == "spec"]["name"], m[m.kind == "spec"]["value"]))
    assert sp_rows["sdiff"] == 0.0 and sp_rows["conc"] == float(conc)
    assert sp_rows["ktrend"] == float(spec["ktrend"])
    assert sp_rows["burn"] == float(spec["burn"])
    # trend rows carry the degrees
    tr = m[m.kind == "trend"].sort_values("idx")
    assert tr["value"].tolist() == [float(v) for v in
                                    load_v2(fx, "trend")["degree"]]
    # no anchors stored in nodiff mode
    assert (m.kind == "anchor").sum() == 0

    FIT_V2[fx] = dict(path=path, dcon=dcon, our_ll=our_ll, sm_ll=sm_ll)


@pytest.mark.parametrize("fx", V2_FIT_CASES)
def test_v2_public_forecast(con, fx):
    if fx not in FIT_V2:
        pytest.skip("fit test did not run/pass")
    spec, cols = setup_v2_user_tables(con, fx)
    fcfix = load_v2(fx, "forecast").sort_values("h")
    H = len(fcfix)
    lst = "[" + ", ".join(f"'{cname}'" for cname in cols) + "]" if cols else "[]::VARCHAR[]"
    nd = ", newdata := '_hv_fut'" if cols else ""
    fc = con.execute(f"""SELECT * FROM sarimax_forecast('_hv_m_{fx}', '_hv_dat', 'y',
        {H}{nd}, exog_cols := {lst}, t_col := 't')""").df().sort_values("h")
    assert len(fc) == H
    assert np.isfinite(fc[["yhat", "se", "lo", "hi"]].to_numpy()).all(), \
        f"{fx}: non-finite forecast output"
    assert ((fc["lo"] < fc["yhat"]) & (fc["yhat"] < fc["hi"])).all(), \
        f"{fx}: broken interval ordering"
    if FIT_V2[fx]["path"] == "params":
        relm = (np.abs(fc["yhat"].to_numpy() - fcfix["mean_orig"].to_numpy())
                / np.abs(fcfix["mean_orig"].to_numpy())).max()
        rels = (np.abs(fc["se"].to_numpy() - fcfix["se_orig"].to_numpy())
                / np.abs(fcfix["se_orig"].to_numpy())).max()
        assert relm <= 1e-6, f"{fx}: forecast mean rel err {relm:.3e}"
        assert rels <= 1e-5, f"{fx}: forecast se rel err {rels:.3e}"
    # sdiff = 0: the model scale IS the original scale, so the fc_orig stage
    # must be an exact identity
    np.testing.assert_array_equal(fc["yhat"].to_numpy(),
                                  fc["yhat_diff"].to_numpy(),
                                  err_msg=f"{fx}: nodiff mean identity")
    np.testing.assert_array_equal(fc["se"].to_numpy(), fc["se_diff"].to_numpy(),
                                  err_msg=f"{fx}: nodiff se identity")


@pytest.mark.parametrize("fx", V2_FIT_CASES)
def test_v2_summary_and_residuals(con, fx):
    if fx not in FIT_V2:
        pytest.skip("fit test did not run/pass")
    spec, cols = setup_v2_user_tables(con, fx)
    kp = int(spec["k_params"])
    sm = con.execute(f"SELECT * FROM sarimax_summary('_hv_m_{fx}', '_hv_dat', 'y')").df()
    assert len(sm) == kp, f"{fx}: summary rows {len(sm)} != k_params {kp}"
    assert np.isfinite(sm["std_error"].to_numpy()).all()

    lst = "[" + ", ".join(f"'{cname}'" for cname in cols) + "]" if cols else "[]::VARCHAR[]"
    res = con.execute(f"""SELECT * FROM sarimax_residuals('_hv_m_{fx}', '_hv_dat', 'y',
        exog_cols := {lst}, t_col := 't') ORDER BY t""").df()
    ser = load_v2(fx, "series").sort_values("t")
    miss_t = set(int(t) for t, v in zip(ser["t"], ser["y"]) if pd.isna(v))
    got_null = set(int(t) for t, v in zip(res["t"], res["std_resid"])
                   if v is None or (isinstance(v, float) and np.isnan(v)))
    assert got_null == miss_t, \
        f"{fx}: std_resid NULL at {sorted(got_null)} != missing t {sorted(miss_t)}"
    assert len(res) == int(spec["n_eff"])
    # ljung-box on the non-null residuals must be finite
    lb = con.execute(f"""SELECT * FROM sarimax_ljungbox('_hv_m_{fx}', '_hv_dat', 'y', 5,
        exog_cols := {lst}, t_col := 't')""").df()
    assert len(lb) == 5 and np.isfinite(lb["stat"].to_numpy()).all()
    ev = con.execute(f"""SELECT * FROM sarimax_evaluate('_hv_m_{fx}', '_hv_dat', 'y',
        exog_cols := {lst}, t_col := 't')""").df().iloc[0]
    assert np.isfinite(ev["aic"]) and ev["sigma2"] > 0.0


def test_v2_fit_summary_report():
    if not FIT_V2:
        pytest.skip("no v2 fits ran in this session (deselected?)")
    print("\n--- v2 public-fit outcomes ------------------------------------")
    for fx, rr in FIT_V2.items():
        print(f"{fx:22s} {rr['path']:8s} dcon {rr['dcon']:9.3e} "
              f"ll(ours)-ll(sm) {rr['our_ll'] - rr['sm_ll']:12.3e}")
    print("----------------------------------------------------------------")


# ---------------------------------------------------------------------------
# 3. v1 public-path regression through the new engine
# ---------------------------------------------------------------------------

def test_v1_public_path_regression(con):
    """arma_1_0_1 through the NEW public path with default flags: acceptance
    identical to tests/test_estimate.py against the v1 fixture."""
    fx = "arma_1_0_1"
    ser = pd.read_parquet(os.path.join(FIXDIR, fx, "series.parquet")).sort_values("t")
    con.execute("CREATE OR REPLACE TABLE _hv1_dat (t BIGINT, y DOUBLE)")
    con.executemany("INSERT INTO _hv1_dat VALUES (?, ?)",
                    [(int(t), float(v)) for t, v in zip(ser["t"], ser["y"])])
    con.execute("""CREATE OR REPLACE TABLE _hv1_m AS
        SELECT * FROM sarimax_fit('_hv1_dat', 'y', 1, 0, 1, t_col := 't')""")
    m = con.execute("SELECT * FROM _hv1_m").df()
    fitted = pd.read_parquet(os.path.join(FIXDIR, fx, "fitted.parquet")).sort_values("k")
    sm_ll = float(pd.read_parquet(
        os.path.join(FIXDIR, fx, "fitted_meta.parquet"))["loglik"].iloc[0])

    got_p = m[m.kind == "param"].sort_values("idx")["value"].to_numpy()
    got_u = m[m.kind == "param_unc"].sort_values("idx")["value"].to_numpy()
    our_ll = float(m[(m.kind == "meta") & (m.name == "loglik")]["value"].iloc[0])
    assert m[(m.kind == "meta") & (m.name == "converged")]["value"].iloc[0] == 1.0
    assert np.isfinite(our_ll)
    dll = our_ll - sm_ll
    assert dll >= -1e-8, f"v1 regression: worse than statsmodels by {-dll:.3e}"
    dcon = float(np.abs(got_p - fitted["constrained"].to_numpy()).max())
    dunc = float(np.abs(got_u - fitted["unconstrained"].to_numpy()).max())
    if dcon <= 1e-6 and dunc <= 1e-6:
        assert abs(dll) <= 1e-8 * max(1.0, abs(sm_ll))
    elif dll > 1e-8:
        pass                                    # our optimum won (re-baseline)
    else:
        assert abs(dll) <= 1e-8 and dcon <= 1e-3, \
            f"v1 regression: params off (dcon {dcon:.3e}) and ll not better " \
            f"(dll {dll:.3e})"
    # default-flag spec rows: sdiff = 1, conc = 0, ktrend = 0, burn = 0
    sp = dict(zip(m[m.kind == "spec"]["name"], m[m.kind == "spec"]["value"]))
    assert (sp["sdiff"], sp["conc"], sp["ktrend"], sp["burn"]) == (1.0, 0.0, 0.0, 0.0)
    # param names unchanged from v1 (incl. sigma2 last)
    assert list(m[m.kind == "param"].sort_values("idx")["name"]) == \
        list(fitted["name"])
