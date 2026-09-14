"""Gradient-probe sharing correctness checks.

Shared covariance and mean passes must reproduce the full likelihood exactly,
including perturbed head parameters, fitted trajectories, and thread-count
independence. The large-state batch checks every probe against the full kernel.
Performance measurements belong in profiling tools, not test assertions.
"""
import os

import duckdb
import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIXDIR = os.path.join(HERE, "fixtures_v2")

FIXTURES = sorted(
    d for d in os.listdir(FIXDIR) if os.path.isdir(os.path.join(FIXDIR, d)))

SQL_FILES = ["sql/00_linalg.sql", "sql/02_ssm.sql", "sql/03_filter.sql",
             "sql/04_estimate.sql"]

# ---------------------------------------------------------------------------
# Pre-sharing fit results retained as exact trajectory regressions.
OLD_FIT_LL_EXOG8 = -408.5134559762526
OLD_FIT_ITERS_EXOG8 = 20
OLD_FIT_LL_K14 = -94.2334412857046
OLD_FIT_ITERS_K14 = 12


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


def blocks_of(spec):
    r, p, q, P, Q, s, d, sd, kt, conc = (int(spec[k]) for k in
        ["r", "p", "q", "bigp", "bigq", "s", "d", "bigd", "ktrend", "conc"])
    return r, p, q, P, Q, max(s, 1), d, sd, kt, bool(conc)


def args_of(spec):
    r, p, q, P, Q, s, d, sd, kt, conc = blocks_of(spec)
    return f"{r}, {p}, {q}, {P}, {Q}, {s}, {d}, {sd}, {kt}, {conc}"


def setup_fixture_data(c, fx):
    """Materialize the fixture's (ylist, xmat, degs) into a one-row table
    _gs_dat (lists bound as TABLE columns: the scalar-subquery CTE form
    re-evaluates per lambda element -- section 2 header trap)."""
    spec = load(fx, "spec").iloc[0]
    n, r = int(spec["n"]), int(spec["r"])
    ser = load(fx, "series").sort_values("t")
    c.execute("CREATE OR REPLACE TABLE _gs_y (t BIGINT, y DOUBLE)")
    c.executemany("INSERT INTO _gs_y VALUES (?, ?)",
                  [(int(t), None if pd.isna(v) else float(v))
                   for t, v in zip(ser["t"], ser["y"])])
    c.execute("CREATE OR REPLACE TABLE _gs_x (t BIGINT, j INT, x DOUBLE)")
    if r:
        ex = load(fx, "exog")
        ex = ex[ex.t <= n]
        c.executemany("INSERT INTO _gs_x VALUES (?, ?, ?)",
                      [(int(a), int(b), float(v))
                       for a, b, v in zip(ex["t"], ex["j"], ex["x"])])
    degs = load(fx, "trend").sort_values("idx")["degree"].tolist()
    c.execute("CREATE OR REPLACE TABLE _gs_degs (idx BIGINT, degree BIGINT)")
    if degs:
        c.executemany("INSERT INTO _gs_degs VALUES (?, ?)",
                      [(i + 1, int(dv)) for i, dv in enumerate(degs)])
    c.execute("""CREATE OR REPLACE TABLE _gs_dat AS
        SELECT (SELECT list(y ORDER BY t) FROM _gs_y) AS zyl,
               coalesce((SELECT list(zxr ORDER BY t) FROM (
                   SELECT t, list(x ORDER BY j) AS zxr FROM _gs_x GROUP BY t)),
                        []::DOUBLE[][]) AS zxm,
               (SELECT coalesce(list(degree ORDER BY idx), []::BIGINT[])
                FROM _gs_degs) AS zdg""")
    return spec


def full_ll(c, args, params):
    return c.execute(f"""
        SELECT (_sarimax_ll_c_v2(?, zyl, zxm, zdg, {args})).ll,
               (_sarimax_ll_c_v2(?, zyl, zxm, zdg, {args})).scale2
        FROM _gs_dat""", [list(params), list(params)]).fetchone()


def split_ll(c, args, r, kt, base_params, probe_params):
    """gains from base_params; ydlist/clist from probe_params' head (built
    with the exact fold shapes of _sarimax_ll_c_v2's zyd / zcl)."""
    q = f"""
    WITH _g1 AS (SELECT zyl, zxm, zdg, ?::DOUBLE[] AS zcb, ?::DOUBLE[] AS zcp
                 FROM _gs_dat),
    _g2 AS (SELECT zyl, zxm, zdg, zcp,
                   _sarimax_kf_gains_v2(zcb, zyl, zxm, zdg, {args}) AS zgn
            FROM _g1),
    _g3 AS (SELECT zgn,
                   list_transform(range(1, len(zyl) + 1), lambda zt:
                       zyl[zt] - list_reduce(list_prepend(0e0,
                           list_transform(range(1, {r} + 1), lambda zj:
                               zxm[zt][zj] * zcp[{kt} + zj])),
                           lambda za, zb: za + zb)) AS zydl,
                   _sarimax_trend_c(zdg, list_slice(zcp, 1, {kt}),
                                    1, len(zyl) + 1) AS zcll
            FROM _g2)
    SELECT (_sarimax_ll_mean_v2(zgn, zydl, zcll)).ll,
           (_sarimax_ll_mean_v2(zgn, zydl, zcll)).scale2
    FROM _g3"""
    return c.execute(q, [list(base_params), list(probe_params)]).fetchone()


def assert_pair_equal(got, want, ctx):
    for name, g, w in zip(("ll", "scale2"), got, want):
        if w is None:
            assert g is None, f"{ctx}: {name} split={g!r} full=None"
        else:
            assert g == w, f"{ctx}: {name} NOT bitwise: split={g!r} full={w!r}"


# ---------------------------------------------------------------------------
# 1. kernel-split bitwise identity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fx", FIXTURES)
def test_kernel_split_bitwise(con, fx):
    spec = setup_fixture_data(con, fx)
    r, p, q, P, Q, s, d, sd, kt, conc = blocks_of(spec)
    args = args_of(spec)
    rtot = kt + r
    probes = load(fx, "probes")
    pids = sorted(probes.probe_id.unique())[:5]
    for pid in pids:
        g = probes[probes.probe_id == pid].sort_values("k")
        con_p = g["constrained"].to_numpy(dtype=float)
        want = full_ll(con, args, con_p)
        got = split_ll(con, args, r, kt, con_p, con_p)
        assert_pair_equal(got, want, f"{fx} probe {pid} same-vector")
        # the true gradient-probe case: gains from the BASE vector, mean at
        # a head-perturbed vector (first and last head coordinate, +-1e-5)
        if rtot > 0 and pid in pids[:3]:
            for ci in sorted({0, rtot - 1}):
                for sgn in (+1.0, -1.0):
                    pp = con_p.copy()
                    pp[ci] += sgn * 1e-5
                    want = full_ll(con, args, pp)
                    got = split_ll(con, args, r, kt, con_p, pp)
                    assert_pair_equal(
                        got, want,
                        f"{fx} probe {pid} head {ci} {sgn:+.0f}1e-5")


# ---------------------------------------------------------------------------
# 2. fit bitwise-trajectory guards (public-path synthetic fits)
# ---------------------------------------------------------------------------

def _load_fit_tables(c, y, X, pfx):
    n = len(y)
    r = X.shape[1] if X is not None else 0
    c.execute(f"CREATE OR REPLACE TABLE {pfx}_y (t BIGINT, y DOUBLE)")
    c.executemany(f"INSERT INTO {pfx}_y VALUES (?, ?)",
                  [(t + 1, float(v)) for t, v in enumerate(y)])
    c.execute(f"CREATE OR REPLACE TABLE {pfx}_x (t BIGINT, j INT, x DOUBLE)")
    if r:
        c.executemany(f"INSERT INTO {pfx}_x VALUES (?, ?, ?)",
                      [(t + 1, j + 1, float(X[t, j]))
                       for t in range(n) for j in range(r)])
    c.execute(f"CREATE OR REPLACE TABLE {pfx}_degs (idx BIGINT, degree BIGINT)")


def bench_a_data():
    rng = np.random.default_rng(20260717)
    n, r = 300, 8
    X = rng.standard_normal((n, r))
    X = (X - X.mean(0)) / X.std(0)
    beta = rng.uniform(-1.0, 1.0, r)
    phi, theta = 0.6, 0.3
    e = rng.standard_normal(n)
    u = np.zeros(n)
    for t in range(n):
        u[t] = (phi * u[t-1] if t else 0.0) + e[t] + (theta * e[t-1] if t else 0.0)
    return X @ beta + u, X


def bench_b_data():
    rng = np.random.default_rng(4242)
    n, r = 131, 2
    X = rng.standard_normal((n, r))
    X = (X - X.mean(0)) / X.std(0)
    beta = np.array([0.5, -0.3])
    th, Th = 0.4, 0.3
    e = rng.standard_normal(n + 13) * 0.5
    u = e[13:] + th * e[12:-1] + Th * e[1:-12] + th * Th * e[0:-13]
    return X @ beta + u, X


def test_fit_trajectory_exog8(con):
    """Preserve the exact fit trajectory for eight regressors below the sharing gate."""
    y, X = bench_a_data()
    _load_fit_tables(con, y, X, "_ga")
    row = con.execute("""SELECT * FROM _sarimax_bfgs_v2('_ga_y', '_ga_x', '_ga_degs',
                             8, 1, 1, 0, 0, 1, 0, 0, 0, false)""").df().iloc[0]
    assert float(row["loglik"]) == OLD_FIT_LL_EXOG8, \
        f"trajectory changed: ll {row['loglik']!r} vs {OLD_FIT_LL_EXOG8!r}"
    assert int(row["iterations"]) == OLD_FIT_ITERS_EXOG8
    assert bool(row["converged"])


def test_fit_trajectory_k14(con):
    """Preserve the exact seasonal fit trajectory below the sharing gate."""
    y, X = bench_b_data()
    _load_fit_tables(con, y, X, "_gb")
    row = con.execute("""SELECT * FROM _sarimax_bfgs_v2('_gb_y', '_gb_x', '_gb_degs',
                             2, 0, 1, 0, 1, 12, 0, 0, 0, false)""").df().iloc[0]
    assert float(row["loglik"]) == OLD_FIT_LL_K14, \
        f"trajectory changed: ll {row['loglik']!r} vs {OLD_FIT_LL_K14!r}"
    assert int(row["iterations"]) == OLD_FIT_ITERS_K14
    assert bool(row["converged"])


# ---------------------------------------------------------------------------
# 3. gradient-batch equivalence where sharing applies (k = 27)
# ---------------------------------------------------------------------------

def test_gradient_batch_equivalence_k27(con):
    """Shared gains + four mean + six full probes must equal ten full probes."""
    spec = setup_fixture_data(con, "nodiff_sarimax_011_011_12")
    args = args_of(spec)
    probes = load("nodiff_sarimax_011_011_12", "probes")
    cp = probes[probes.probe_id == 1].sort_values("k")["constrained"].tolist()
    con.execute("CREATE OR REPLACE TABLE _gk AS "
                "SELECT zyl, zxm, zdg, ?::DOUBLE[] AS zcp FROM _gs_dat", [cp])

    old_sql = f"""
        SELECT zidx, (_sarimax_ll_c_v2(
                   list_transform(zcpc, lambda zxe, zxi:
                       CASE WHEN zxi = (zidx + 1) // 2
                            THEN zxe + (CASE WHEN zidx % 2 = 1 THEN 1e-5 ELSE -1e-5 END)
                            ELSE zxe END),
                   zwlc, zxmc, zdgc, {args})).ll AS zll
        FROM (SELECT zg.zcp AS zcpc, zg.zyl AS zwlc, zg.zxm AS zxmc,
                     zg.zdg AS zdgc, zu.zidx
              FROM _gk zg, unnest(range(1, 11)) AS zu(zidx))"""
    new_sql = f"""
        SELECT zidx,
               CASE WHEN zidx <= 4
                    THEN (_sarimax_ll_mean_v2(zgnc,
                        list_transform(range(1, len(zwlc) + 1), lambda zt9:
                            zwlc[zt9] - list_reduce(list_prepend(0e0,
                                list_transform(range(1, 3), lambda zj9:
                                    zxmc[zt9][zj9]
                                    * (CASE WHEN zj9 = (zidx + 1) // 2
                                            THEN zcpc[zj9] + (CASE WHEN zidx % 2 = 1
                                                                   THEN 1e-5 ELSE -1e-5 END)
                                            ELSE zcpc[zj9] END))),
                                lambda za9, zb9: za9 + zb9)),
                        list_transform(range(1, len(zwlc) + 2), lambda zz9: 0e0))).ll
                    ELSE (_sarimax_ll_c_v2(
                        list_transform(zcpc, lambda zxe, zxi:
                            CASE WHEN zxi = (zidx + 1) // 2
                                 THEN zxe + (CASE WHEN zidx % 2 = 1 THEN 1e-5 ELSE -1e-5 END)
                                 ELSE zxe END),
                        zwlc, zxmc, zdgc, {args})).ll END AS zll
        FROM (SELECT zg.zcp AS zcpc, zg.zyl AS zwlc, zg.zxm AS zxmc,
                     zg.zdg AS zdgc, zgl.zgn AS zgnc, zu.zidx
              FROM _gk zg
              CROSS JOIN LATERAL (
                  SELECT max(_sarimax_kf_gains_v2(zcpg, zylg, zxmg, zdgg,
                                                  {args})) AS zgn
                  FROM (SELECT zg.zcp AS zcpg, zg.zyl AS zylg,
                               zg.zxm AS zxmg, zg.zdg AS zdgg
                        FROM unnest([1]) AS zu9(zn9))
              ) zgl
              CROSS JOIN unnest(range(1, 11)) AS zu(zidx))"""

    con.execute(f"PREPARE _gs_qold AS {old_sql}")
    con.execute(f"PREPARE _gs_qnew AS {new_sql}")

    old_rows = dict(con.execute("EXECUTE _gs_qold").fetchall())
    new_rows = dict(con.execute("EXECUTE _gs_qnew").fetchall())
    for zidx in range(1, 11):
        assert new_rows[zidx] == old_rows[zidx], \
            f"probe {zidx}: {new_rows[zidx]!r} != {old_rows[zidx]!r}"


# ---------------------------------------------------------------------------
# 4. determinism across thread counts
# ---------------------------------------------------------------------------

def test_split_determinism_threads():
    """gains + mean bitwise at threads=1 vs default on kitchen_sink (trend +
    exog + missing + conc + kdiff>0 -- every code path)."""
    results = []
    for th in (1, None):
        c = make_con(threads=th)
        spec = setup_fixture_data(c, "kitchen_sink")
        r, p, q, P, Q, s, d, sd, kt, conc = blocks_of(spec)
        args = args_of(spec)
        probes = load("kitchen_sink", "probes")
        cp = probes[probes.probe_id == 1].sort_values("k")["constrained"].to_numpy(dtype=float)
        pp = cp.copy()
        pp[0] += 1e-5
        results.append(split_ll(c, args, r, kt, cp, pp))
        c.close()
    assert results[0] == results[1], \
        f"split kernel not thread-deterministic: {results[0]!r} vs {results[1]!r}"
