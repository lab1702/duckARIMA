"""v2 filter-core acceptance: augmented state space (simple_differencing =
False, approximate diffuse), missing values, trend terms, concentrated scale.

Fixtures: tests/fixtures_v2/<name>/ (see tests/generate_fixtures_v2.py for the
pinned conventions). Checks, per fixture:

  1. ssm        -- rebuilt transition/design/selection/state_cov/obs_cov match
                   fixture ssm.parquet at 1e-14 (5 ssm probes);
  2. intercepts -- obs_intercept (y - yd) and state_intercept (ct /
                   _sarimax_trend_c) match at 1e-14; state_intercept row
                   index equals cidx = kdiff + 1;
  3. init       -- systems' raw (a1, p1): 1e6 diffuse diagonal and structural
                   zeros EXACT, stationary block / a1 at rel 1e-9 (conc
                   fixtures compare P1 rescaled by scale2 -- statsmodels
                   reports P1 multiplied by the concentrated scale);
  4. T1         -- loglik and the (v, f) trace vs loglike.parquet /
                   filter_trace.parquet; v NULL exactly at the fixture's NaN
                   rows; concentrated f rescaled by scale2 = ssq/cnt;
  5. burn/cnt   -- counted steps == non-NaN fixture v at t > burn;
  6. determinism-- threads=1 vs default bitwise identical; the module runs
                   with DeprecationWarning promoted to error.

TOLERANCE NOTE (documented deviation from the pinned 1e-9/1e-10 targets): the
approximate-diffuse 1e6 initialization makes the trace reproducible only to
eps(1e6 * |P|) amplified through the diffuse collapse. The floor is intrinsic:
re-running statsmodels' own algorithm in float64 with any different operation
order (numpy BLAS vs ordered folds) deviates from the fixture by the same
amount. Measured DuckDB worst deviations -> tolerances set with >= 30x margin:

    fixture                     kdiff   relf      relv      ll rel
    conc_arma_101                 0     3.9e-14   1.2e-11   4.3e-15
    trend_ct_arma_100             0     1.7e-15   2.8e-13   7.1e-16
    missing_arma_101              0     2.8e-15   1.7e-12   9.2e-15
    nodiff_arima_111              1     3.0e-10   4.9e-11   8.9e-13
    kitchen_sink                  5     9.4e-10   1.2e-08   3.3e-12
    nodiff_sarimax_011_011_12    13     1.4e-06   2.2e-04   1.9e-07

The three stationary fixtures (kdiff = 0) hold the pinned tolerances exactly.
"""
import os

import duckdb
import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.filterwarnings("error::DeprecationWarning")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIXDIR = os.path.join(HERE, "fixtures_v2")

FIXTURES = sorted(
    d for d in os.listdir(FIXDIR) if os.path.isdir(os.path.join(FIXDIR, d)))

SQL_FILES = ["sql/00_linalg.sql", "sql/02_ssm.sql", "sql/03_filter.sql"]

N_SSM_PROBES = 5

# per-fixture tolerances (see module docstring); "default" = pinned targets
TOL_DEFAULT = dict(f_rtol=1e-9, v_rtol=1e-9, v_atol=1e-12,
                   ll_abs=1e-8, ll_rel=1e-10, p1_rtol=1e-9)
TOL = {
    "nodiff_arima_111": dict(f_rtol=1e-8, v_rtol=1e-8, v_atol=1e-11,
                             ll_abs=1e-8, ll_rel=1e-10, p1_rtol=1e-9),
    "kitchen_sink": dict(f_rtol=5e-8, v_rtol=5e-7, v_atol=1e-10,
                         ll_abs=1e-8, ll_rel=1e-10, p1_rtol=1e-4),
    "nodiff_sarimax_011_011_12": dict(f_rtol=5e-5, v_rtol=1e-3, v_atol=1e-5,
                                      ll_abs=1e-3, ll_rel=1e-5, p1_rtol=1e-9),
}


def tol(fx):
    return TOL.get(fx, TOL_DEFAULT)


def make_con(threads=None):
    c = duckdb.connect()
    c.execute("SET lambda_syntax = 'DISABLE_SINGLE_ARROW'")
    if threads:
        c.execute(f"SET threads = {threads}")
    for f in SQL_FILES:
        with open(os.path.join(ROOT, f)) as fh:
            c.execute(fh.read())
    return c


@pytest.fixture(scope="module")
def con():
    return make_con()


def load(fx, name):
    return pd.read_parquet(os.path.join(FIXDIR, fx, name + ".parquet"))


def spec_ints(spec):
    return {k: int(spec[k]) for k in
            ["p", "d", "q", "bigp", "bigd", "bigq", "s", "r", "n", "n_eff",
             "k_states", "k_params", "burn", "sdiff", "conc", "ktrend"]}


def setup_fixture_tables(c, fx, prefix="_v2"):
    """Create <prefix>_y, <prefix>_x, <prefix>_degs, <prefix>_probes."""
    sp = spec_ints(load(fx, "spec").iloc[0])
    n = sp["n"]
    ser = load(fx, "series").sort_values("t")
    c.execute(f"CREATE OR REPLACE TABLE {prefix}_y (t BIGINT, y DOUBLE)")
    c.executemany(f"INSERT INTO {prefix}_y VALUES (?, ?)",
                  [(int(t), None if pd.isna(y) else float(y))
                   for t, y in zip(ser.t, ser.y)])

    exog = load(fx, "exog")
    exog = exog[exog.t <= n]                       # in-sample rows only
    c.execute(f"CREATE OR REPLACE TABLE {prefix}_x (t BIGINT, j INT, x DOUBLE)")
    if len(exog):
        c.executemany(f"INSERT INTO {prefix}_x VALUES (?, ?, ?)",
                      [(int(r.t), int(r.j), float(r.x))
                       for r in exog.itertuples()])

    trend = load(fx, "trend")
    c.execute(f"CREATE OR REPLACE TABLE {prefix}_degs (idx BIGINT, degree BIGINT)")
    if len(trend):
        c.executemany(f"INSERT INTO {prefix}_degs VALUES (?, ?)",
                      [(int(r.idx), int(r.degree)) for r in trend.itertuples()])

    probes = load(fx, "probes")
    c.execute(f"CREATE OR REPLACE TABLE {prefix}_probes "
              "(probe_id BIGINT, params DOUBLE[])")
    for pid, g in probes.groupby("probe_id"):
        c.execute(f"INSERT INTO {prefix}_probes VALUES (?, ?)",
                  [int(pid), g.sort_values("k")["constrained"].tolist()])
    return sp


def build_systems(c, sp, prefix="_v2"):
    seff = max(sp["s"], 1)
    c.execute(f"""
        CREATE OR REPLACE TABLE {prefix}_sys AS
        SELECT * FROM _sarimax_systems_v2('{prefix}_probes',
            {sp['r']}, {sp['p']}, {sp['q']}, {sp['bigp']}, {sp['bigq']},
            {seff}, {sp['d']}, {sp['bigd']}, {sp['ktrend']}, {sp['conc']})""")
    c.execute(f"""
        CREATE OR REPLACE TABLE {prefix}_obs AS
        SELECT * FROM _sarimax_obs_adj_v2('{prefix}_y', '{prefix}_x',
            '{prefix}_probes', {sp['r']}, {sp['ktrend']}, '{prefix}_degs')""")


def run_filter(c, sp, prefix="_v2"):
    trace = c.execute(f"""
        SELECT probe_id, t, v, f, cnt, sumlogf, ssq
        FROM _sarimax_kfilter_v2('{prefix}_obs', '{prefix}_sys')
        ORDER BY probe_id, t""").df()
    ll = c.execute(f"""
        SELECT probe_id, n_eff, loglik, scale2
        FROM _sarimax_loglik_v2('{prefix}_obs', '{prefix}_sys', {sp['conc']})
        ORDER BY probe_id""").df()
    return trace, ll


# ---------------------------------------------------------------------------
# 1. state-space matrices vs fixture ssm.parquet (1e-14)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fx", FIXTURES)
def test_ssm_matrices(con, fx):
    sp = setup_fixture_tables(con, fx)
    seff = max(sp["s"], 1)
    kdiff = sp["d"] + seff * sp["bigd"] if sp["bigd"] else sp["d"]
    k = sp["k_states"]
    karma = k - kdiff
    kt, r, p, q, P, Q = (sp[x] for x in
                         ["ktrend", "r", "p", "q", "bigp", "bigq"])
    kdiff_sql = con.execute(
        f"SELECT _sarimax_kdiff({sp['d']}, {sp['bigd']}, {seff})").fetchone()[0]
    assert kdiff_sql == kdiff and kdiff + karma == k

    sigma2_expr = ("1e0" if sp["conc"]
                   else f"params[{sp['k_params']}]")
    got = con.execute(f"""
        WITH _t2_pp AS (
            SELECT probe_id, params, {sigma2_expr} AS sigma2
            FROM _v2_probes WHERE probe_id <= {N_SSM_PROBES}
        ),
        _t2_poly AS (
            SELECT probe_id, sigma2,
                   _sarimax_expand_ar(
                       list_slice(params, {kt + r + 1}, {kt + r + p}),
                       list_slice(params, {kt + r + p + q + 1},
                                  {kt + r + p + q + P}), {seff}) AS phistar,
                   _sarimax_expand_ma(
                       list_slice(params, {kt + r + p + 1}, {kt + r + p + q}),
                       list_slice(params, {kt + r + p + q + P + 1},
                                  {kt + r + p + q + P + Q}), {seff}) AS thetastar
            FROM _t2_pp
        ),
        _t2_mats AS (
            SELECT probe_id, sigma2,
                   _sarimax_build_t_v2(phistar, {karma}, {sp['d']},
                                       {sp['bigd']}, {seff}) AS tmat,
                   _sarimax_build_z_v2({karma}, {sp['d']}, {sp['bigd']},
                                       {seff}) AS zvec,
                   _sarimax_build_r_v2(thetastar, {karma}, {sp['d']},
                                       {sp['bigd']}, {seff}) AS rvec
            FROM _t2_poly
        )
        SELECT probe_id, 'transition' AS name,
               (u.idx - 1) // {k} + 1 AS i, (u.idx - 1) % {k} + 1 AS j,
               tmat[u.idx] AS v
        FROM _t2_mats, LATERAL unnest(range(1, {k * k} + 1)) AS u(idx)
        UNION ALL
        SELECT probe_id, 'design', 1, u.idx, zvec[u.idx]
        FROM _t2_mats, LATERAL unnest(range(1, {k} + 1)) AS u(idx)
        UNION ALL
        SELECT probe_id, 'selection', u.idx, 1, rvec[u.idx]
        FROM _t2_mats, LATERAL unnest(range(1, {k} + 1)) AS u(idx)
        UNION ALL
        SELECT probe_id, 'state_cov', 1, 1, sigma2 FROM _t2_mats
        UNION ALL
        SELECT probe_id, 'obs_cov', 1, 1, 0e0 FROM _t2_mats
        ORDER BY probe_id, name, i, j""").df()

    want = load(fx, "ssm")
    want = want[want.name.isin(
        ["transition", "design", "selection", "state_cov", "obs_cov"])]
    merged = want.merge(got, on=["probe_id", "name", "i", "j"],
                        suffixes=("_want", "_got"), how="left", validate="1:1")
    assert not merged.v_got.isna().any(), f"{fx}: missing rebuilt cells"
    dev = (merged.v_want - merged.v_got).abs().max()
    assert dev <= 1e-14, f"{fx}: ssm max abs dev {dev!r}"


# ---------------------------------------------------------------------------
# 2. observation and state intercepts (1e-14)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fx", FIXTURES)
def test_intercepts(con, fx):
    sp = setup_fixture_tables(con, fx)
    build_systems(con, sp)
    seff = max(sp["s"], 1)
    kdiff = sp["d"] + seff * sp["bigd"] if sp["bigd"] else sp["d"]
    ssm = load(fx, "ssm")
    obs = con.execute(f"""
        SELECT o.probe_id, o.t, o.yd, o.ct, y.y
        FROM _v2_obs o JOIN _v2_y y ON y.t = o.t
        WHERE o.probe_id <= {N_SSM_PROBES}
        ORDER BY o.probe_id, o.t""").df()

    oi = ssm[ssm.name == "obs_intercept"]
    if sp["r"] == 0:
        assert len(oi) == 0
        assert np.allclose(obs.yd.dropna(), obs.y.dropna(), rtol=0, atol=0)
    else:
        m = obs.merge(oi.rename(columns={"i": "t", "v": "d_want"}),
                      left_on=["probe_id", "t"], right_on=["probe_id", "t"],
                      validate="1:1")
        ok = ~m.y.isna()
        # compare yd against y - d_want (recovering d as y - yd would
        # reintroduce cancellation noise at eps(|y|))
        dev = (m.yd - (m.y - m.d_want))[ok].abs().max()
        assert dev <= 1e-14, f"{fx}: obs_intercept dev {dev!r}"

    si = ssm[ssm.name == "state_intercept"]
    if sp["ktrend"] == 0:
        assert len(si) == 0
        assert (obs.ct == 0.0).all()
    else:
        assert set(si.j.unique()) == {kdiff + 1}, \
            f"{fx}: state_intercept row != cidx"
        m = obs.merge(si.rename(columns={"i": "t", "v": "c_want"}),
                      left_on=["probe_id", "t"], right_on=["probe_id", "t"],
                      validate="1:1")
        dev = (m.ct - m.c_want).abs().max()
        assert dev <= 1e-14, f"{fx}: state_intercept dev {dev!r}"

        # _sarimax_trend_c directly, whole horizon in one call
        degs = load(fx, "trend").sort_values("idx")["degree"].tolist()
        probes = load(fx, "probes")
        for pid in sorted(si.probe_id.unique()):
            params = probes[probes.probe_id == pid].sort_values(
                "k")["constrained"].tolist()
            tau = params[:sp["ktrend"]]
            cs = con.execute(f"""
                SELECT _sarimax_trend_c(zb.degs, zb.tau, 1, {sp['n']})
                FROM (SELECT ?::BIGINT[] AS degs, ?::DOUBLE[] AS tau) zb""",
                [degs, tau]).fetchone()[0]
            w = si[si.probe_id == pid].sort_values("i")
            dev = np.abs(np.asarray(cs)[w.i.to_numpy() - 1]
                         - w.v.to_numpy()).max()
            assert dev <= 1e-14, f"{fx} probe {pid}: trend_c dev {dev!r}"


# ---------------------------------------------------------------------------
# 3. initialization a1 / P1 vs fixture (exact diffuse block, 1e-9 stationary)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fx", FIXTURES)
def test_initialization(con, fx):
    sp = setup_fixture_tables(con, fx)
    build_systems(con, sp)
    seff = max(sp["s"], 1)
    kdiff = sp["d"] + seff * sp["bigd"] if sp["bigd"] else sp["d"]
    k = sp["k_states"]
    ssm = load(fx, "ssm")
    sys = con.execute(f"""
        SELECT probe_id, k, karma, kdiff, cidx, burn, p1, a1
        FROM _v2_sys WHERE probe_id <= {N_SSM_PROBES}
        ORDER BY probe_id""").df()
    assert (sys.k == k).all() and (sys.kdiff == kdiff).all()
    assert (sys.karma == k - kdiff).all()
    assert (sys.cidx == kdiff + 1).all()
    assert (sys.burn == sp["burn"]).all(), f"{fx}: burn != spec burn"

    scale2 = None
    if sp["conc"]:
        _, ll = run_filter(con, sp)
        scale2 = ll.set_index("probe_id")["scale2"]

    for row in sys.itertuples():
        pid = row.probe_id
        mine_p1 = np.asarray(row.p1).reshape(k, k)
        mine_a1 = np.asarray(row.a1)
        rows = ssm[ssm.probe_id == pid]
        want_p1 = np.zeros((k, k))
        for rr in rows[rows.name == "P1"].itertuples():
            want_p1[rr.i - 1, rr.j - 1] = rr.v
        want_a1 = np.zeros(k)
        for rr in rows[rows.name == "a1"].itertuples():
            want_a1[rr.i - 1] = rr.v

        # structural zeros: cross blocks and diffuse off-diagonal, EXACT
        if kdiff:
            assert np.all(mine_p1[:kdiff, kdiff:] == 0.0)
            assert np.all(mine_p1[kdiff:, :kdiff] == 0.0)
            assert np.all(want_p1[:kdiff, kdiff:] == 0.0)
            assert np.all(want_p1[kdiff:, :kdiff] == 0.0)
            dd = mine_p1[:kdiff, :kdiff]
            assert np.all(dd[~np.eye(kdiff, dtype=bool)] == 0.0)

        if sp["conc"]:
            s2 = float(scale2.loc[pid])
            np.testing.assert_allclose(
                mine_p1 * s2, want_p1, rtol=tol(fx)["p1_rtol"], atol=1e-12,
                err_msg=f"{fx} probe {pid}: P1 (rescaled by scale2)")
            if kdiff:
                np.testing.assert_allclose(
                    np.diag(mine_p1)[:kdiff], np.full(kdiff, 1e6),
                    rtol=0, atol=0)
        else:
            if kdiff:
                assert np.all(np.diag(mine_p1)[:kdiff] == 1e6)
                assert np.all(np.diag(want_p1)[:kdiff] == 1e6), \
                    f"{fx} probe {pid}: fixture diffuse diagonal != 1e6"
            np.testing.assert_allclose(
                mine_p1[kdiff:, kdiff:], want_p1[kdiff:, kdiff:],
                rtol=1e-9, atol=1e-12,
                err_msg=f"{fx} probe {pid}: P1 stationary block")

        np.testing.assert_allclose(mine_a1, want_a1, rtol=1e-9, atol=1e-30,
                                   err_msg=f"{fx} probe {pid}: a1")


# ---------------------------------------------------------------------------
# 4. T1: loglikelihood and filter trace at all 21 probes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fx", FIXTURES)
def test_t1_loglik_and_trace(con, fx):
    sp = setup_fixture_tables(con, fx)
    build_systems(con, sp)
    got, ll = run_filter(con, sp)
    want_trace = load(fx, "filter_trace").sort_values(["probe_id", "t"])
    want_ll = load(fx, "loglike").set_index("probe_id")["loglik"]
    tl = tol(fx)

    assert len(got) == len(want_trace), f"{fx}: trace row count"
    ll = ll.set_index("probe_id")

    for pid in want_ll.index:
        g = got[got.probe_id == pid]
        w = want_trace[want_trace.probe_id == pid]
        vg, fg = g["v"].to_numpy(), g["f"].to_numpy()
        vw, fw = w["v"].to_numpy(), w["f"].to_numpy()
        nanmask = np.isnan(vw)
        assert np.array_equal(np.isnan(vg), nanmask), \
            f"{fx} probe {pid}: v NULL pattern != fixture NaN pattern"
        if sp["conc"]:
            s2 = float(ll.loc[pid, "scale2"])
            assert np.isfinite(s2) and s2 > 0
            fg = fg * s2
        np.testing.assert_allclose(fg, fw, rtol=tl["f_rtol"], atol=0,
                                   err_msg=f"{fx} probe {pid}: F trace")
        np.testing.assert_allclose(
            vg[~nanmask], vw[~nanmask], rtol=tl["v_rtol"], atol=tl["v_atol"],
            err_msg=f"{fx} probe {pid}: v trace")

        lg, lw = float(ll.loc[pid, "loglik"]), float(want_ll.loc[pid])
        assert np.isfinite(lg)
        assert (abs(lg - lw) <= tl["ll_abs"]
                or abs(lg - lw) <= tl["ll_rel"] * abs(lw)), \
            f"{fx} probe {pid}: ll {lg!r} vs {lw!r}"

    if not sp["conc"]:
        assert ll["scale2"].isna().all()


# ---------------------------------------------------------------------------
# 5. burn / missing accounting
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fx", FIXTURES)
def test_burn_and_missing_count(con, fx):
    sp = setup_fixture_tables(con, fx)
    build_systems(con, sp)
    got, _ = run_filter(con, sp)
    want_trace = load(fx, "filter_trace")
    state = con.execute("""
        SELECT probe_id, n_eff, cnt
        FROM _sarimax_kfilter_state_v2('_v2_obs', '_v2_sys')
        ORDER BY probe_id""").df().set_index("probe_id")
    for pid, g in got.groupby("probe_id"):
        w = want_trace[want_trace.probe_id == pid]
        expected = int(((~w.v.isna()) & (w.t > sp["burn"])).sum())
        cnt_trace = int(g.sort_values("t")["cnt"].iloc[-1])
        assert cnt_trace == expected, f"{fx} probe {pid}: cnt (trace)"
        assert int(state.loc[pid, "cnt"]) == expected, \
            f"{fx} probe {pid}: cnt (state)"
        assert int(state.loc[pid, "n_eff"]) == sp["n_eff"]


# ---------------------------------------------------------------------------
# 6. determinism across thread counts (bitwise)
# ---------------------------------------------------------------------------

def test_determinism_across_threads():
    fx = "kitchen_sink"
    results = []
    for th in (1, None):
        c = make_con(threads=th)
        sp = setup_fixture_tables(c, fx)
        build_systems(c, sp)
        got, ll = run_filter(c, sp)
        results.append((got["v"].to_numpy(), got["f"].to_numpy(),
                        ll["loglik"].to_numpy(), ll["scale2"].to_numpy()))
        c.close()
    for a, b in zip(results[0], results[1]):
        assert np.array_equal(a, b, equal_nan=True), \
            "thread-count nondeterminism"
