"""Layer 4 acceptance: estimation (spec 5.4, milestones M4/M5, tiers T2/T3-bse).

Test groups:
  1. Kernel agreement: the scalar loglikelihood kernel `_sarimax_ll_x`
     (doubling-init P1) reproduces statsmodels' loglike at EVERY fixture probe
     at T1 tolerance (abs <= 1e-8 or rel <= 1e-10), and agrees with the
     recursive-CTE filter of sql/03_filter.sql (vec-trick P1) at
     abs <= 1e-9 or rel <= 1e-11 on a probe subset.
  2. T2 fit: `_sarimax_bfgs` on every fixture. converged must be true. The
     comparison implements the spec's re-baselining diagnostic (section 3 +
     10): a fixture passes through exactly one of
       - "params":  parameters match statsmodels at 1e-6 abs (1e-5 for the
                    near_nonstationary / near_noninvertible fixtures) in both
                    spaces, and loglik agrees at rel <= 1e-8;
       - "ll_won":  our loglik exceeds statsmodels' by > 1e-8 with our
                    params stationary (our optimum won -- statsmodels'
                    optimizer stopped short; both lls evaluated through the
                    SAME T1-validated function);
       - "ll_tie":  the two optima are equivalent at the spec's loglik
                    resolution (|delta ll| <= 1e-8, spec section 10:
                    "if our optimum has ll >= statsmodels' minus 1e-8, the
                    fixture's parameter row is re-baselined to our optimum"),
                    with our params stationary and within 1e-3 of
                    statsmodels' (sanity bound).
     The path each fixture took is recorded and printed by the summary test.
     In ALL paths our loglik must not be materially worse: ll >= sm - 1e-8.
  3. bse at the FIXTURE theta-hat (comparing the standard-error machinery at
     statsmodels' own optimum, independent of optimizer endpoint) vs the
     fixture 'bse' column at rel <= 1e-3.
  4. Start values: finite x0 strictly inside the region on every fixture
     (untransform finite = the stationarity/invertibility test), round-trip
     transform(x0) == params0.
  5. Determinism: the full fit is bitwise identical at threads=1 vs default.
  6. Timing: per-fixture wall times are recorded and printed (not asserted),
     except the airline fixture full fit which must complete in < 10 minutes
     (spec section 7).
"""
import os
import time

import duckdb
import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIXDIR = os.path.join(HERE, "fixtures")

FIXTURES = sorted(
    d for d in os.listdir(FIXDIR) if os.path.isdir(os.path.join(FIXDIR, d)))

BOUNDARY_FIXTURES = {"near_nonstationary", "near_noninvertible"}

SQL_FILES = ["sql/00_linalg.sql", "sql/02_ssm.sql", "sql/03_filter.sql",
             "sql/04_estimate.sql"]

# module-level records for the summary test
FIT_RESULTS = {}     # fixture -> dict(path=..., time=..., ...)


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


def diff_series(z, d, D, s):
    """EXACTLY the fixture generator's differencing: d ordinary diffs first,
    then D seasonal lag-s diffs."""
    z = np.asarray(z, dtype=float)
    for _ in range(d):
        z = np.diff(z)
    for _ in range(D):
        z = z[s:] - z[:-s]
    return z


def setup_fixture_tables(c, fx, prefix="_es"):
    """Create <prefix>_w and <prefix>_exd (differenced series/exog) plus
    <prefix>_probes; returns the spec row."""
    spec = load(fx, "spec").iloc[0]
    d, D, s, n, r = (int(spec[k]) for k in ["d", "bigd", "s", "n", "r"])
    y = load(fx, "series").sort_values("t")["y"].to_numpy()
    w = diff_series(y, d, D, s)

    c.execute(f"CREATE OR REPLACE TABLE {prefix}_w (t BIGINT, w DOUBLE)")
    c.executemany(f"INSERT INTO {prefix}_w VALUES (?, ?)",
                  [(t + 1, float(w[t])) for t in range(len(w))])

    c.execute(f"CREATE OR REPLACE TABLE {prefix}_exd (t BIGINT, j INT, x DOUBLE)")
    if r:
        exog = load(fx, "exog")
        for j in range(1, r + 1):
            xj = exog[(exog.j == j) & (exog.t <= n)].sort_values("t")["x"].to_numpy()
            xd = diff_series(xj, d, D, s)
            c.executemany(f"INSERT INTO {prefix}_exd VALUES (?, ?, ?)",
                          [(t + 1, j, float(xd[t])) for t in range(len(xd))])

    probes = load(fx, "probes")
    c.execute(f"CREATE OR REPLACE TABLE {prefix}_probes "
              f"(probe_id BIGINT, unc DOUBLE[], con DOUBLE[])")
    for pid, g in probes.groupby("probe_id"):
        g = g.sort_values("k")
        c.execute(f"INSERT INTO {prefix}_probes VALUES (?, ?, ?)",
                  [int(pid), g["unconstrained"].tolist(), g["constrained"].tolist()])
    return spec


def blocks_of(spec):
    r, p, q, P, Q, s = (int(spec[k]) for k in ["r", "p", "q", "bigp", "bigq", "s"])
    return r, p, q, P, Q, max(s, 1)


DATA_CTE = """
    WITH _dat AS (
        SELECT (SELECT list(w ORDER BY t) FROM {pfx}_w) AS zwl,
               coalesce((SELECT list(zxr ORDER BY t) FROM (
                   SELECT t, list(x ORDER BY j) AS zxr FROM {pfx}_exd GROUP BY t)),
                        []::DOUBLE[][]) AS zxm
    )
"""


# ---------------------------------------------------------------------------
# 1. kernel agreement
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fx", FIXTURES)
def test_kernel_vs_statsmodels(con, fx):
    """_sarimax_ll_x at every probe (column 'unconstrained') vs loglike.parquet
    at T1 tolerance -- the doubling P1 init must not cost accuracy."""
    spec = setup_fixture_tables(con, fx)
    r, p, q, P, Q, s = blocks_of(spec)
    got = con.execute(DATA_CTE.format(pfx="_es") + f"""
        SELECT pr.probe_id,
               _sarimax_ll_x(pr.unc, zd.zwl, zd.zxm, {r}, {p}, {q}, {P}, {Q}, {s}) AS ll
        FROM _es_probes pr, _dat zd
        ORDER BY pr.probe_id""").df()
    want = load(fx, "loglike").sort_values("probe_id")
    assert len(got) == len(want)
    for gv, wv, pid in zip(got["ll"], want["loglik"], want["probe_id"]):
        ok = abs(gv - wv) <= 1e-8 or abs(gv - wv) <= 1e-10 * abs(wv)
        assert ok, f"{fx} probe {pid}: kernel ll {gv!r} vs statsmodels {wv!r}"


KERNEL_FILTER_PROBES = [1, 7, 14, 18, 23, 26]


@pytest.mark.parametrize("fx", FIXTURES)
def test_kernel_vs_recursive_filter(con, fx):
    """_sarimax_ll_x (doubling P1) vs _sarimax_loglik (the recursive filter,
    vec-trick P1) on a probe subset: agreement <= 1e-9 abs or 1e-11 rel.
    This bounds the documented P1-initialization deviation."""
    spec = setup_fixture_tables(con, fx, prefix="_ek")
    r, p, q, P, Q, s = blocks_of(spec)
    pl = ", ".join(str(x) for x in KERNEL_FILTER_PROBES)
    con.execute(f"""CREATE OR REPLACE TABLE _ek_sub AS
                    SELECT probe_id, con AS params FROM _ek_probes
                    WHERE probe_id IN ({pl})""")
    con.execute(f"""CREATE OR REPLACE TABLE _ek_sys AS
                    SELECT * FROM _sarimax_systems('_ek_sub', {r}, {p}, {q}, {P}, {Q}, {s})""")
    con.execute("""CREATE OR REPLACE TABLE _ek_obs AS
                   SELECT * FROM _sarimax_obs_adj('_ek_w', '_ek_exd', '_ek_sub')""")
    filt = con.execute("""SELECT probe_id, loglik
                          FROM _sarimax_loglik('_ek_obs', '_ek_sys')
                          ORDER BY probe_id""").df()
    kern = con.execute(DATA_CTE.format(pfx="_ek") + f"""
        SELECT pr.probe_id,
               _sarimax_ll_x(pr.unc, zd.zwl, zd.zxm, {r}, {p}, {q}, {P}, {Q}, {s}) AS ll
        FROM _ek_probes pr, _dat zd
        WHERE pr.probe_id IN ({pl})
        ORDER BY pr.probe_id""").df()
    for pid, lf, lk in zip(filt["probe_id"], filt["loglik"], kern["ll"]):
        ok = abs(lk - lf) <= 1e-9 or abs(lk - lf) <= 1e-11 * abs(lf)
        assert ok, f"{fx} probe {pid}: kernel {lk!r} vs filter {lf!r}"


# ---------------------------------------------------------------------------
# 2. T2 fit (with the spec's re-baselining diagnostic)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fx", FIXTURES)
def test_t2_fit(con, fx):
    spec = setup_fixture_tables(con, fx)
    r, p, q, P, Q, s = blocks_of(spec)
    tol = 1e-5 if fx in BOUNDARY_FIXTURES else 1e-6

    t0 = time.perf_counter()
    row = con.execute(f"""SELECT * FROM _sarimax_bfgs('_es_w', '_es_exd',
                              {r}, {p}, {q}, {P}, {Q}, {s})""").df().iloc[0]
    dt = time.perf_counter() - t0

    params = np.asarray(row["params"], dtype=float)
    x_opt = np.asarray(row["x_opt"], dtype=float)
    our_ll = float(row["loglik"])

    fitted = load(fx, "fitted").sort_values("k")
    sm_con = fitted["constrained"].to_numpy()
    sm_unc = fitted["unconstrained"].to_numpy()
    sm_ll = float(load(fx, "fitted_meta")["loglik"].iloc[0])

    dcon = float(np.abs(params - sm_con).max())
    dunc = float(np.abs(x_opt - sm_unc).max())
    dll = our_ll - sm_ll
    stationary = bool(np.isfinite(x_opt).all() and np.isfinite(params).all())

    assert bool(row["converged"]), f"{fx}: fit did not converge (status flags: " \
        f"iters={row['iterations']} restarted={row['restarted']} lsf={row['ls_failures']})"
    assert np.isfinite(our_ll), f"{fx}: non-finite loglik"
    assert stationary, f"{fx}: non-finite parameters"
    # never materially worse than statsmodels (spec section 10)
    assert dll >= -1e-8, f"{fx}: our optimum is WORSE than statsmodels by {-dll:.3e}"

    if dcon <= tol and dunc <= tol:
        path = "params"
        assert abs(dll) <= 1e-8 * max(1.0, abs(sm_ll)), \
            f"{fx}: params match but loglik differs (rel {abs(dll)/abs(sm_ll):.2e})"
    elif dll > 1e-8:
        path = "ll_won"          # our optimum won; spec 10 re-baselining applies
    elif abs(dll) <= 1e-8 and dcon <= 1e-3:
        path = "ll_tie"          # optima equivalent at the spec's ll resolution
    else:
        pytest.fail(f"{fx}: params off (dcon {dcon:.3e}, dunc {dunc:.3e}) and "
                    f"loglik not better (dll {dll:.3e})")

    FIT_RESULTS[fx] = dict(path=path, time=dt, dll=dll, dcon=dcon, dunc=dunc,
                           iters=int(row["iterations"]), our_ll=our_ll, sm_ll=sm_ll,
                           params=params)
    if fx == "airline":
        assert dt < 600.0, f"airline full fit took {dt:.0f}s (spec cap: 10 minutes)"


# ---------------------------------------------------------------------------
# 3. standard errors
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fx", FIXTURES)
def test_bse(con, fx):
    """bse at statsmodels' theta-hat (the fixture's own optimum, so the
    comparison isolates the Hessian machinery) vs fixture bse, rel <= 1e-3."""
    spec = setup_fixture_tables(con, fx, prefix="_eb")
    r, p, q, P, Q, s = blocks_of(spec)
    fitted = load(fx, "fitted").sort_values("k")
    con.execute("CREATE OR REPLACE TABLE _eb_th (params DOUBLE[])")
    con.execute("INSERT INTO _eb_th VALUES (?)", [fitted["constrained"].tolist()])
    got = con.execute(f"""
        SELECT bse FROM _sarimax_bse(
            (SELECT params FROM _eb_th),
            (SELECT list(w ORDER BY t) FROM _eb_w),
            coalesce((SELECT list(zxr ORDER BY t) FROM (
                SELECT t, list(x ORDER BY j) AS zxr FROM _eb_exd GROUP BY t)),
                     []::DOUBLE[][]),
            {r}, {p}, {q}, {P}, {Q}, {s})""").fetchone()[0]
    got = np.asarray(got, dtype=float)
    ref = fitted["bse"].to_numpy()
    assert np.isfinite(got).all(), f"{fx}: non-finite bse {got}"
    rel = np.abs(got - ref) / np.abs(ref)
    assert (rel <= 1e-3).all(), \
        f"{fx}: bse rel error {rel.max():.3e}\n got {got}\n ref {ref}"


# ---------------------------------------------------------------------------
# 4. starting values
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fx", FIXTURES)
def test_start_params(con, fx):
    spec = setup_fixture_tables(con, fx, prefix="_ep")
    r, p, q, P, Q, s = blocks_of(spec)
    x0, p0 = con.execute(f"""SELECT x0, params0
                             FROM _sarimax_start_params('_ep_w', '_ep_exd',
                                 {r}, {p}, {q}, {P}, {Q}, {s})""").fetchone()
    x0, p0 = np.asarray(x0, dtype=float), np.asarray(p0, dtype=float)
    kp = r + int(spec["p"]) + int(spec["q"]) + int(spec["bigp"]) + int(spec["bigq"]) + 1
    assert len(x0) == kp and len(p0) == kp
    assert np.isfinite(x0).all(), f"{fx}: x0 not finite (outside region): {x0}"
    assert np.isfinite(p0).all(), f"{fx}: params0 not finite: {p0}"
    assert p0[-1] > 0, f"{fx}: sigma2_0 not positive"
    # round trip: transform(x0) == params0 (they were built as a pair)
    rt = con.execute(f"SELECT _sarimax_transform_params(?, {r}, {p}, {q}, {P}, {Q})",
                     [x0.tolist()]).fetchone()[0]
    np.testing.assert_allclose(rt, p0, rtol=1e-10, atol=1e-12,
                               err_msg=f"{fx}: transform(x0) != params0")


# ---------------------------------------------------------------------------
# 5. determinism
# ---------------------------------------------------------------------------

def test_fit_determinism_across_threads():
    """Full fit on arma_1_0_1 at threads=1 vs default: bitwise identical."""
    results = []
    for th in (1, None):
        c = make_con(threads=th)
        spec = setup_fixture_tables(c, "arma_1_0_1", prefix="_ed")
        r, p, q, P, Q, s = blocks_of(spec)
        row = c.execute(f"""SELECT * FROM _sarimax_bfgs('_ed_w', '_ed_exd',
                                {r}, {p}, {q}, {P}, {Q}, {s})""").df().iloc[0]
        results.append((np.asarray(row["params"], dtype=float),
                        np.asarray(row["x_opt"], dtype=float),
                        float(row["loglik"]), int(row["iterations"])))
        c.close()
    (p1, x1, l1, i1), (p2, x2, l2, i2) = results
    assert i1 == i2, f"iteration count differs: {i1} vs {i2}"
    assert np.array_equal(x1, x2), f"x_opt not bitwise identical:\n{x1}\n{x2}"
    assert np.array_equal(p1, p2), f"params not bitwise identical:\n{p1}\n{p2}"
    assert l1 == l2, f"loglik not bitwise identical: {l1!r} vs {l2!r}"


# ---------------------------------------------------------------------------
# 6. timing / outcome summary (runs last: pytest executes in file order)
# ---------------------------------------------------------------------------

def test_summary_report():
    """Prints per-fixture T2 outcome and wall time. Only asserts that every
    fixture actually produced a fit result (the airline < 10 min gate is
    asserted inside test_t2_fit)."""
    if not FIT_RESULTS:
        pytest.skip("no T2 fits ran in this session (deselected?)")
    missing = [fx for fx in FIXTURES if fx not in FIT_RESULTS]
    assert not missing, f"fixtures without fit results (t2 failed?): {missing}"
    print("\n--- T2 fit outcomes -------------------------------------------")
    print(f"{'fixture':22s} {'path':8s} {'time':>7s} {'iters':>5s} "
          f"{'ll(ours)-ll(sm)':>16s} {'max|dparam|':>12s}")
    for fx in FIXTURES:
        rr = FIT_RESULTS[fx]
        print(f"{fx:22s} {rr['path']:8s} {rr['time']:6.1f}s {rr['iters']:5d} "
              f"{rr['dll']:16.3e} {rr['dcon']:12.3e}")
    print("----------------------------------------------------------------")
