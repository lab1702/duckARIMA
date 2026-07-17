"""Layer 3 acceptance: Tier-1 of spec section 3 -- the heart of the project.

At every fixture probe point:
  * loglikelihood agrees with statsmodels loglike() at abs <= 1e-8 or
    rel <= 1e-10, whichever is looser;
  * the per-timestep innovations v_t and variances F_t agree at rel <= 1e-9.

Plus the M2x differential check (exog affects only the innovation, never F),
a bitwise threads=1 vs default determinism check, and a timing smoke test.
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

SQL_FILES = ["sql/00_linalg.sql", "sql/02_ssm.sql", "sql/03_filter.sql"]


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
    z = np.asarray(z, dtype=float)
    for _ in range(d):
        z = np.diff(z)
    for _ in range(D):
        z = z[s:] - z[:-s]
    return z


def setup_fixture_tables(c, fx, prefix="_t1"):
    """Create <prefix>_w, <prefix>_exd, <prefix>_probes for a fixture; returns spec."""
    spec = load(fx, "spec").iloc[0]
    d, D, s, n, r = (int(spec[c2]) for c2 in ["d", "bigd", "s", "n", "r"])
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
    c.execute(f"CREATE OR REPLACE TABLE {prefix}_probes (probe_id BIGINT, params DOUBLE[])")
    for pid, g in probes.groupby("probe_id"):
        params = g.sort_values("k")["constrained"].tolist()
        c.execute(f"INSERT INTO {prefix}_probes VALUES (?, ?)", [int(pid), params])
    return spec


def run_filter(c, spec, prefix="_t1"):
    r, p, q, P, Q, s = (int(spec[c2]) for c2 in ["r", "p", "q", "bigp", "bigq", "s"])
    seff = max(s, 1)
    c.execute(f"""
        CREATE OR REPLACE TABLE {prefix}_sys AS
        SELECT * FROM _sarimax_systems('{prefix}_probes', {r}, {p}, {q}, {P}, {Q}, {seff})""")
    c.execute(f"""
        CREATE OR REPLACE TABLE {prefix}_obs AS
        SELECT * FROM _sarimax_obs_adj('{prefix}_w', '{prefix}_exd', '{prefix}_probes')""")
    return c.execute(f"""
        SELECT probe_id, t, v, f, ll_acc
        FROM _sarimax_kfilter('{prefix}_obs', '{prefix}_sys')
        ORDER BY probe_id, t""").df()


@pytest.mark.parametrize("fx", FIXTURES)
def test_t1_loglik_and_trace(con, fx):
    spec = setup_fixture_tables(con, fx)
    got = run_filter(con, spec)
    want_trace = load(fx, "filter_trace").sort_values(["probe_id", "t"])
    want_ll = load(fx, "loglike").set_index("probe_id")["loglik"]

    assert len(got) == len(want_trace), f"{fx}: row count mismatch"

    np.testing.assert_allclose(got["v"].to_numpy(), want_trace["v"].to_numpy(),
                               rtol=1e-9, atol=1e-12, err_msg=f"{fx}: v trace")
    np.testing.assert_allclose(got["f"].to_numpy(), want_trace["f"].to_numpy(),
                               rtol=1e-9, atol=0, err_msg=f"{fx}: F trace")

    ll_got = got.groupby("probe_id")["ll_acc"].last()
    for pid in want_ll.index:
        w, g = want_ll.loc[pid], ll_got.loc[pid]
        assert g == pytest.approx(w, rel=1e-10, abs=1e-8), \
            f"{fx} probe {pid}: ll {g!r} vs {w!r}"


@pytest.mark.parametrize("fx", [f for f in FIXTURES])
def test_m2x_differential_exog(con, fx):
    """Probes sharing the ARMA block but differing in beta must yield bitwise-
    identical F_t (exog enters only the innovation)."""
    spec = load(fx, "spec").iloc[0]
    r = int(spec["r"])
    if r == 0:
        pytest.skip("no exog")
    setup_fixture_tables(con, fx, prefix="_dx")
    probes = load(fx, "probes")
    base = probes[probes.probe_id == 1].sort_values("k")["constrained"].tolist()
    shifted = list(base)
    for j in range(r):
        shifted[j] = shifted[j] + 1.7
    con.execute("CREATE OR REPLACE TABLE _dx_probes (probe_id BIGINT, params DOUBLE[])")
    con.execute("INSERT INTO _dx_probes VALUES (1, ?), (2, ?)", [base, shifted])
    got = run_filter(con, spec, prefix="_dx")
    f1 = got[got.probe_id == 1]["f"].to_numpy()
    f2 = got[got.probe_id == 2]["f"].to_numpy()
    assert np.array_equal(f1, f2), f"{fx}: F must be invariant to beta"
    v1 = got[got.probe_id == 1]["v"].to_numpy()
    v2 = got[got.probe_id == 2]["v"].to_numpy()
    assert not np.allclose(v1, v2), f"{fx}: v must depend on beta"


def test_determinism_across_threads():
    """Bitwise-identical loglikelihoods at threads=1 vs default threading."""
    fx = "sarima_111_111_12"
    results = []
    for th in (1, None):
        c = make_con(threads=th)
        spec = setup_fixture_tables(c, fx)
        got = run_filter(c, spec)
        results.append(got.groupby("probe_id")["ll_acc"].last().to_numpy())
        c.close()
    assert np.array_equal(results[0], results[1]), "thread-count nondeterminism"


def test_timing_single_loglik():
    """Spec section 7: one loglikelihood at n = 500, k = 14 in < 2 s (5x headroom)."""
    c = make_con()
    rng = np.random.default_rng(7)
    w = rng.standard_normal(500)
    c.execute("CREATE OR REPLACE TABLE _pf_w (t BIGINT, w DOUBLE)")
    c.executemany("INSERT INTO _pf_w VALUES (?, ?)",
                  [(i + 1, float(w[i])) for i in range(500)])
    c.execute("CREATE OR REPLACE TABLE _pf_exd (t BIGINT, j INT, x DOUBLE)")
    # airline-shaped spec: (0,1,1)(0,1,1)_12 -> k = 14
    params = [0.3, -0.4, 1.1]
    c.execute("CREATE OR REPLACE TABLE _pf_probes (probe_id BIGINT, params DOUBLE[])")
    c.execute("INSERT INTO _pf_probes VALUES (1, ?)", [params])
    c.execute("""CREATE OR REPLACE TABLE _pf_sys AS
                 SELECT * FROM _sarimax_systems('_pf_probes', 0, 0, 1, 0, 1, 12)""")
    c.execute("""CREATE OR REPLACE TABLE _pf_obs AS
                 SELECT * FROM _sarimax_obs_adj('_pf_w', '_pf_exd', '_pf_probes')""")
    t0 = time.perf_counter()
    ll = c.execute("SELECT loglik FROM _sarimax_loglik('_pf_obs', '_pf_sys')").fetchone()[0]
    dt = time.perf_counter() - t0
    assert np.isfinite(ll)
    assert dt < 10.0, f"single loglik took {dt:.2f}s (target 2s, hard cap 10s)"
    print(f"single loglik n=500 k=14: {dt*1000:.0f} ms")
