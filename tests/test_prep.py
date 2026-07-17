"""Acceptance tests for duckARIMA Layer 1 (sql/01_prep.sql) -- series preparation.

Everything is validated against NumPy/pandas references computed inline:

* Differencing: sequential ``np.diff`` d times, then seasonal ``x[s:] - x[:-s]``
  D times (the pinned application order) -- compared EXACTLY (bit-identical
  floating point, since the SQL performs the same sequential lag-subtracts).
* Anchors: trailing values of each pre-stage intermediate series -- exact.
* Round-trip (the crown jewel): diff the train segment, compute anchors, feed
  the TRUE future differenced values (from differencing the full series)
  through ``_sarimax_undiff_forecast`` and recover the true future
  original-scale values to rel <= 1e-12, over the full (d, D, s) grid.
* Integration weights: unit impulses pushed through numpy cumsum /
  seasonal-cumsum -- exact (the weights are small integers).
* Rank check: elimination pivots against a hand-rolled numpy partial-pivoting
  elimination.
* ACF/PACF: statsmodels conventions -- acf(adjusted=False) (denominator n) and
  pacf(method='ywm') (Yule-Walker with mle/n autocovariance). statsmodels is
  used as the reference when importable; otherwise (as in the pinned test
  venv, which does not carry statsmodels) the references are a hand-rolled
  numpy ACF and per-lag Yule-Walker Toeplitz solves, which are algebraically
  identical to statsmodels' ywm path.  Tolerance 1e-10.

The whole suite runs twice: at default threads and with SET threads = 1.
DuckDB deprecation warnings are escalated to errors, and the SQL file is
loaded under SET lambda_syntax = 'DISABLE_SINGLE_ARROW', so any arrow lambda
or deprecated construct fails the run rather than warning.
"""

import subprocess
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

try:  # optional reference; the numpy fallback is exercised when absent
    from statsmodels.tsa.stattools import acf as sm_acf, pacf as sm_pacf

    HAVE_STATSMODELS = True
except ImportError:
    HAVE_STATSMODELS = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREP_SQL = PROJECT_ROOT / "sql" / "01_prep.sql"

pytestmark = pytest.mark.filterwarnings("error::DeprecationWarning")

# (d, D, s) grid mandated by the acceptance criteria; s is irrelevant when
# D = 0 (asserted separately), so a single s is kept there to avoid duplicates.
GRID = [
    (d, D, s)
    for d in (0, 1, 2)
    for D in (0, 1, 2)
    for s in ((4, 12) if D > 0 else (4,))
]

WEIGHT_CASES = [
    (1, 0, 4), (1, 0, 12), (2, 0, 4), (2, 0, 12),
    (0, 1, 4), (1, 1, 4), (1, 1, 12), (2, 1, 12),
    (0, 2, 4), (1, 2, 4),
]


# ---------------------------------------------------------------------------
# fixtures and helpers
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module", params=["threads_default", "threads_1"])
def con(request):
    c = duckdb.connect()
    c.execute("SET lambda_syntax = 'DISABLE_SINGLE_ARROW';")
    if request.param == "threads_1":
        c.execute("SET threads = 1;")
    c.execute(PREP_SQL.read_text(encoding="utf-8"))
    yield c
    c.close()


def load_series(con, name, y, t=None):
    y = np.asarray(y, dtype=np.float64)
    if t is None:
        t = np.arange(1, len(y) + 1, dtype=np.int64)
    df = pd.DataFrame({"t": np.asarray(t, dtype=np.int64), "y": y})
    con.register("_pydf", df)
    con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM _pydf")
    con.unregister("_pydf")


def load_exog(con, name, X):
    """X is (n, r); loads long form (t, j, x) with j-major blocks."""
    X = np.asarray(X, dtype=np.float64)
    n, r = X.shape
    df = pd.DataFrame({
        "t": np.tile(np.arange(1, n + 1, dtype=np.int64), r),
        "j": np.repeat(np.arange(1, r + 1, dtype=np.int32), n),
        "x": X.T.reshape(-1),
    })
    con.register("_pydf", df)
    con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM _pydf")
    con.unregister("_pydf")


def gen_series(rng, n, s):
    """Level + trend + seasonal + random walk + noise; bounded away from 0 so
    a pure relative tolerance is meaningful."""
    t = np.arange(1, n + 1, dtype=np.float64)
    return (100.0 + 0.3 * t + 8.0 * np.sin(2.0 * np.pi * t / max(s, 4))
            + 0.7 * np.cumsum(rng.standard_normal(n))
            + rng.standard_normal(n))


def ref_diff(y, d, D, s):
    """Pinned convention: ordinary differencing d times FIRST, then seasonal
    differencing (lag s) D times; each stage a sequential lag-subtract."""
    w = np.asarray(y, dtype=np.float64)
    for _ in range(d):
        w = w[1:] - w[:-1]          # identical FP ops to np.diff(w)
    for _ in range(D):
        w = w[s:] - w[:-s]
    return w


def ref_anchors(y, d, D, s):
    """Trailing pre-stage values per the anchor contract; returns the exact
    expected (stage, idx, value) rows in order."""
    rows = []
    cur = np.asarray(y, dtype=np.float64)
    for i in range(1, d + 1):
        rows.append((i, 1, cur[-1]))
        cur = cur[1:] - cur[:-1]
    for k in range(1, D + 1):
        for idx in range(1, s + 1):
            rows.append((d + k, idx, cur[len(cur) - s + idx - 1]))
        cur = cur[s:] - cur[:-s]
    return rows


def ref_weights(d, D, s, H):
    """C[h-1, l-1]: impulse at horizon l pushed through seasonal-cumsum D
    times then cumsum d times (inversion order = reverse application order)."""
    C = np.zeros((H, H))
    for l in range(1, H + 1):
        v = np.zeros(H)
        v[l - 1] = 1.0
        for _ in range(D):
            for i in range(s, H):
                v[i] += v[i - s]
        for _ in range(d):
            v = np.cumsum(v)
        C[:, l - 1] = v
    return C


def ref_pivlist(A):
    """|pivot| per step of Gaussian elimination with partial (row) pivoting,
    first-occurrence argmax, elimination skipped on an exactly zero pivot --
    the same elementwise operations as _sarimax_prep_pivlist."""
    A = np.array(A, dtype=np.float64, copy=True)
    n = A.shape[0]
    piv = []
    for k in range(n):
        p = k + int(np.argmax(np.abs(A[k:, k])))
        if p != k:
            A[[k, p]] = A[[p, k]]
        piv.append(abs(A[k, k]))
        if A[k, k] != 0.0:
            for i in range(k + 1, n):
                f = A[i, k] / A[k, k]
                A[i, :] = A[i, :] - f * A[k, :]
    return np.array(piv)


def ref_acf(y, nlags):
    """Mean-corrected, denominator n at every lag (statsmodels adjusted=False)."""
    y = np.asarray(y, dtype=np.float64)
    dev = y - y.mean()
    c0 = np.sum(dev * dev)
    out = [1.0]
    for k in range(1, nlags + 1):
        out.append(np.sum(dev[k:] * dev[:-k]) / c0)
    return np.array(out)


def ref_pacf(y, nlags):
    """Per-lag Yule-Walker with the n-denominator (mle) autocovariance --
    statsmodels pacf(method='ywm'); solved independently of Durbin-Levinson
    via dense Toeplitz systems so the reference is a genuinely different
    algorithm from the SQL."""
    r = ref_acf(y, nlags)
    out = [1.0]
    for k in range(1, nlags + 1):
        R = np.array([[r[abs(i - j)] for j in range(k)] for i in range(k)])
        phi = np.linalg.solve(R, r[1:k + 1])
        out.append(phi[-1])
    return np.array(out)


def fetch(con, query):
    return con.execute(query).fetchnumpy()


def seed_for(*parts):
    return 90210 + sum(p * m for p, m in zip(parts, (1, 17, 289, 4913)))


# ---------------------------------------------------------------------------
# 1. differencing vs numpy (exact) across the grid
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("d,D,s", GRID)
def test_diff_matches_numpy_exactly(con, d, D, s):
    rng = np.random.default_rng(seed_for(d, D, s, 1))
    n = int(rng.integers(80, 201))
    y = gen_series(rng, n, s)
    load_series(con, "_t_ser", y)
    res = fetch(con, f"SELECT t, w FROM _sarimax_diff('_t_ser', 't', 'y', {d}, {D}, {s}) ORDER BY t")
    ref = ref_diff(y, d, D, s)
    assert len(ref) == n - d - D * s
    assert np.array_equal(res["t"], np.arange(1, len(ref) + 1))
    assert np.array_equal(res["w"], ref), "differencing must be bit-identical to sequential np.diff"


def test_diff_identity_when_no_differencing(con):
    rng = np.random.default_rng(seed_for(0, 0, 0, 2))
    y = gen_series(rng, 100, 4)
    load_series(con, "_t_ser", y)
    res = fetch(con, "SELECT t, w FROM _sarimax_diff('_t_ser', 't', 'y', 0, 0, 4) ORDER BY t")
    assert np.array_equal(res["t"], np.arange(1, 101))
    assert np.array_equal(res["w"], y)


def test_diff_s_irrelevant_when_D_zero(con):
    rng = np.random.default_rng(seed_for(2, 0, 4, 3))
    y = gen_series(rng, 90, 4)
    load_series(con, "_t_ser", y)
    a = fetch(con, "SELECT w FROM _sarimax_diff('_t_ser', 't', 'y', 2, 0, 4) ORDER BY t")
    b = fetch(con, "SELECT w FROM _sarimax_diff('_t_ser', 't', 'y', 2, 0, 12) ORDER BY t")
    assert np.array_equal(a["w"], b["w"])


def test_diff_arbitrary_column_names(con):
    rng = np.random.default_rng(seed_for(1, 1, 4, 4))
    y = gen_series(rng, 60, 4)
    df = pd.DataFrame({"period": np.arange(1, 61, dtype=np.int64), "sales": y})
    con.register("_pydf", df)
    con.execute("CREATE OR REPLACE TABLE _t_named AS SELECT * FROM _pydf")
    con.unregister("_pydf")
    res = fetch(con, "SELECT w FROM _sarimax_diff('_t_named', 'period', 'sales', 1, 1, 4) ORDER BY t")
    assert np.array_equal(res["w"], ref_diff(y, 1, 1, 4))


def test_diff_validation_failures(con):
    y = gen_series(np.random.default_rng(1), 30, 4)
    load_series(con, "_t_ser", y)
    with pytest.raises(duckdb.Error, match="d must be in 0\\.\\.4"):
        con.execute("SELECT * FROM _sarimax_diff('_t_ser', 't', 'y', 5, 0, 4)")
    with pytest.raises(duckdb.Error, match="D must be in 0\\.\\.3"):
        con.execute("SELECT * FROM _sarimax_diff('_t_ser', 't', 'y', 0, 4, 4)")
    with pytest.raises(duckdb.Error, match="s must be >= 2 when D > 0"):
        con.execute("SELECT * FROM _sarimax_diff('_t_ser', 't', 'y', 0, 1, 1)")
    load_series(con, "_t_short", y[:20])  # need d + D*s + 1 = 27 > 20
    with pytest.raises(duckdb.Error, match="series too short"):
        con.execute("SELECT * FROM _sarimax_diff('_t_short', 't', 'y', 2, 2, 12)")
    # gap in t
    load_series(con, "_t_gap", y[:20], t=np.concatenate([np.arange(1, 11), np.arange(12, 22)]))
    with pytest.raises(duckdb.Error, match="dense 1\\.\\.n"):
        con.execute("SELECT * FROM _sarimax_diff('_t_gap', 't', 'y', 1, 0, 4)")
    # NULL y
    con.execute("CREATE OR REPLACE TABLE _t_null AS SELECT t, CASE WHEN t = 5 THEN NULL ELSE y END AS y FROM _t_ser")
    with pytest.raises(duckdb.Error, match="NULL"):
        con.execute("SELECT * FROM _sarimax_diff('_t_null', 't', 'y', 1, 0, 4)")
    con.execute("CREATE OR REPLACE TABLE _t_empty (t BIGINT, y DOUBLE)")
    with pytest.raises(duckdb.Error, match="empty"):
        con.execute("SELECT * FROM _sarimax_diff('_t_empty', 't', 'y', 0, 0, 4)")


# ---------------------------------------------------------------------------
# 2. exog lockstep differencing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("d,D,s", [(0, 0, 4), (1, 0, 4), (0, 1, 4), (1, 1, 4), (2, 2, 12), (2, 1, 4)])
def test_diff_exog_lockstep_with_series_path(con, d, D, s):
    rng = np.random.default_rng(seed_for(d, D, s, 5))
    n, r = 130, 3
    X = np.column_stack([gen_series(rng, n, s) for _ in range(r)])
    load_exog(con, "_t_ex", X)
    res = con.execute(f"SELECT t, j, x FROM _sarimax_diff_exog('_t_ex', {d}, {D}, {s}) ORDER BY j, t").fetchnumpy()
    n_eff = n - d - D * s
    assert np.array_equal(res["j"], np.repeat(np.arange(1, r + 1), n_eff))
    assert np.array_equal(res["t"], np.tile(np.arange(1, n_eff + 1), r))
    for j in range(1, r + 1):
        # series path on the same column must agree exactly
        load_series(con, "_t_exj", X[:, j - 1])
        via_series = fetch(con, f"SELECT w FROM _sarimax_diff('_t_exj', 't', 'y', {d}, {D}, {s}) ORDER BY t")
        got = res["x"][res["j"] == j]
        assert np.array_equal(got, via_series["w"])
        assert np.array_equal(got, ref_diff(X[:, j - 1], d, D, s))


def test_diff_exog_zero_rows(con):
    con.execute("CREATE OR REPLACE TABLE _t_ex0 (t BIGINT, j INT, x DOUBLE)")
    res = con.execute("SELECT * FROM _sarimax_diff_exog('_t_ex0', 1, 1, 4)").fetchall()
    assert res == []


# ---------------------------------------------------------------------------
# 3. anchors: exact contents per the contract
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("d,D,s", GRID)
def test_anchor_contents_exact(con, d, D, s):
    rng = np.random.default_rng(seed_for(d, D, s, 6))
    n = int(rng.integers(80, 201))
    y = gen_series(rng, n, s)
    load_series(con, "_t_ser", y)
    rows = con.execute(
        f"SELECT stage, idx, value FROM _sarimax_diff_anchors('_t_ser', 't', 'y', {d}, {D}, {s}) ORDER BY stage, idx"
    ).fetchall()
    expected = ref_anchors(y, d, D, s)
    assert len(rows) == d + D * s
    assert [(r[0], r[1]) for r in rows] == [(e[0], e[1]) for e in expected]
    got_vals = np.array([r[2] for r in rows])
    exp_vals = np.array([e[2] for e in expected])
    assert np.array_equal(got_vals, exp_vals), "anchor values must be exact trailing pre-stage values"


# ---------------------------------------------------------------------------
# 4. THE round-trip: diff train -> anchors -> integrate true future
#    differenced values -> recover true future original-scale values
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("d,D,s", GRID)
def test_undiff_forecast_roundtrip(con, d, D, s):
    rng = np.random.default_rng(seed_for(d, D, s, 7))
    n = int(rng.integers(80, 201))
    H = 36
    y = gen_series(rng, n + H, s)
    load_series(con, "_t_full", y)
    load_series(con, "_t_train", y[:n])
    n_eff = n - d - D * s
    # true future values on the differenced scale, from differencing the FULL series
    con.execute(
        f"""CREATE OR REPLACE TABLE _t_fc AS
            SELECT (st.t - {n_eff})::BIGINT AS h, st.w
            FROM _sarimax_diff('_t_full', 't', 'y', {d}, {D}, {s}) st
            WHERE st.t > {n_eff}"""
    )
    assert con.execute("SELECT count(*) FROM _t_fc").fetchone()[0] == H
    con.execute(
        f"""CREATE OR REPLACE TABLE _t_anch AS
            SELECT * FROM _sarimax_diff_anchors('_t_train', 't', 'y', {d}, {D}, {s})"""
    )
    res = fetch(con, f"SELECT h, yhat FROM _sarimax_undiff_forecast('_t_fc', '_t_anch', {d}, {D}, {s}) ORDER BY h")
    assert np.array_equal(res["h"], np.arange(1, H + 1))
    np.testing.assert_allclose(res["yhat"], y[n:], rtol=1e-12, atol=0.0)


def test_undiff_forecast_identity_when_no_differencing(con):
    rng = np.random.default_rng(seed_for(0, 0, 4, 8))
    w = rng.standard_normal(12)
    df = pd.DataFrame({"h": np.arange(1, 13, dtype=np.int64), "w": w})
    con.register("_pydf", df)
    con.execute("CREATE OR REPLACE TABLE _t_fc AS SELECT * FROM _pydf")
    con.unregister("_pydf")
    con.execute("CREATE OR REPLACE TABLE _t_anch (stage INT, idx INT, value DOUBLE)")
    res = fetch(con, "SELECT yhat FROM _sarimax_undiff_forecast('_t_fc', '_t_anch', 0, 0, 4) ORDER BY h")
    assert np.array_equal(res["yhat"], w)


def test_undiff_forecast_validation_failures(con):
    df = pd.DataFrame({"h": np.arange(1, 7, dtype=np.int64), "w": np.ones(6)})
    con.register("_pydf", df)
    con.execute("CREATE OR REPLACE TABLE _t_fc AS SELECT * FROM _pydf")
    con.unregister("_pydf")
    y = gen_series(np.random.default_rng(2), 40, 4)
    load_series(con, "_t_ser", y)
    con.execute("CREATE OR REPLACE TABLE _t_anch AS SELECT * FROM _sarimax_diff_anchors('_t_ser', 't', 'y', 1, 1, 4)")
    # sanity: this configuration works
    assert len(con.execute("SELECT * FROM _sarimax_undiff_forecast('_t_fc', '_t_anch', 1, 1, 4)").fetchall()) == 6
    # missing seasonal anchor row
    con.execute("CREATE OR REPLACE TABLE _t_anch2 AS SELECT * FROM _t_anch WHERE NOT (stage = 2 AND idx = 3)")
    with pytest.raises(duckdb.Error, match="does not cover seasonal stages"):
        con.execute("SELECT * FROM _sarimax_undiff_forecast('_t_fc', '_t_anch2', 1, 1, 4)")
    # missing ordinary anchor row
    con.execute("CREATE OR REPLACE TABLE _t_anch3 AS SELECT * FROM _t_anch WHERE stage != 1")
    with pytest.raises(duckdb.Error, match="does not cover ordinary stages"):
        con.execute("SELECT * FROM _sarimax_undiff_forecast('_t_fc', '_t_anch3', 1, 1, 4)")
    # duplicate anchor row
    con.execute("CREATE OR REPLACE TABLE _t_anch4 AS SELECT * FROM _t_anch UNION ALL SELECT * FROM _t_anch LIMIT 6")
    with pytest.raises(duckdb.Error, match="duplicate"):
        con.execute("SELECT * FROM _sarimax_undiff_forecast('_t_fc', '_t_anch4', 1, 1, 4)")
    # non-dense h
    con.execute("CREATE OR REPLACE TABLE _t_fc2 AS SELECT * FROM _t_fc WHERE h != 3")
    with pytest.raises(duckdb.Error, match="dense 1\\.\\.H"):
        con.execute("SELECT * FROM _sarimax_undiff_forecast('_t_fc2', '_t_anch', 1, 1, 4)")
    # empty forecast table
    con.execute("CREATE OR REPLACE TABLE _t_fc3 (h BIGINT, w DOUBLE)")
    with pytest.raises(duckdb.Error, match="empty"):
        con.execute("SELECT * FROM _sarimax_undiff_forecast('_t_fc3', '_t_anch', 1, 1, 4)")


# ---------------------------------------------------------------------------
# 5. integration weight matrix vs impulse-response reference
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("d,D,s", WEIGHT_CASES)
def test_undiff_weights_vs_impulse_reference(con, d, D, s):
    H = 36
    res = con.execute(
        f"SELECT h, l, c FROM _sarimax_undiff_weights({d}, {D}, {s}, {H}) ORDER BY h, l"
    ).fetchnumpy()
    # full lower triangle emitted
    assert len(res["h"]) == H * (H + 1) // 2
    C = ref_weights(d, D, s, H)
    for h, l, c in zip(res["h"], res["l"], res["c"]):
        assert l <= h
        assert c == C[h - 1, l - 1], f"weight mismatch at h={h}, l={l}: {c} != {C[h - 1, l - 1]}"


def test_undiff_weights_identity_case(con):
    res = con.execute("SELECT h, l, c FROM _sarimax_undiff_weights(0, 0, 4, 5) ORDER BY h, l").fetchnumpy()
    C = ref_weights(0, 0, 4, 5)  # identity: c[h,l] = 1 iff l = h
    for h, l, c in zip(res["h"], res["l"], res["c"]):
        assert c == C[h - 1, l - 1] == (1.0 if h == l else 0.0)


def test_undiff_weights_link_forecast_decomposition(con):
    """undiff(fc) == undiff(zeros) + C @ w -- ties items 4 and 5 together."""
    d, D, s, H = 1, 1, 4, 24
    rng = np.random.default_rng(seed_for(d, D, s, 9))
    y = gen_series(rng, 60, s)
    w = rng.standard_normal(H)
    load_series(con, "_t_ser", y)
    con.execute(f"CREATE OR REPLACE TABLE _t_anch AS SELECT * FROM _sarimax_diff_anchors('_t_ser', 't', 'y', {d}, {D}, {s})")
    for name, vals in (("_t_fc", w), ("_t_fc0", np.zeros(H))):
        df = pd.DataFrame({"h": np.arange(1, H + 1, dtype=np.int64), "w": vals})
        con.register("_pydf", df)
        con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM _pydf")
        con.unregister("_pydf")
    full = fetch(con, f"SELECT yhat FROM _sarimax_undiff_forecast('_t_fc', '_t_anch', {d}, {D}, {s}) ORDER BY h")["yhat"]
    base = fetch(con, f"SELECT yhat FROM _sarimax_undiff_forecast('_t_fc0', '_t_anch', {d}, {D}, {s}) ORDER BY h")["yhat"]
    res = con.execute(f"SELECT h, l, c FROM _sarimax_undiff_weights({d}, {D}, {s}, {H})").fetchnumpy()
    C = np.zeros((H, H))
    C[res["h"] - 1, res["l"] - 1] = res["c"]
    np.testing.assert_allclose(full, base + C @ w, rtol=1e-9, atol=1e-9)


# ---------------------------------------------------------------------------
# 6. exog validation failure modes
# ---------------------------------------------------------------------------

def test_validate_exog_ok(con):
    rng = np.random.default_rng(seed_for(1, 2, 3, 10))
    X = rng.standard_normal((50, 2))
    load_exog(con, "_t_ex", X)
    rows = con.execute("SELECT j, n_rows, ok FROM _sarimax_validate_exog('_t_ex', 50) ORDER BY j").fetchall()
    assert rows == [(1, 50, True), (2, 50, True)]


def test_validate_exog_ignores_rows_beyond_n_expected(con):
    rng = np.random.default_rng(seed_for(1, 2, 3, 11))
    X = rng.standard_normal((60, 2))
    load_exog(con, "_t_ex", X)  # covers t = 1..60 (e.g. future exog through n+H)
    rows = con.execute("SELECT j, n_rows FROM _sarimax_validate_exog('_t_ex', 50) ORDER BY j").fetchall()
    assert rows == [(1, 50), (2, 50)]


def test_validate_exog_zero_rows_is_valid(con):
    con.execute("CREATE OR REPLACE TABLE _t_ex0 (t BIGINT, j INT, x DOUBLE)")
    assert con.execute("SELECT * FROM _sarimax_validate_exog('_t_ex0', 50)").fetchall() == []


def test_validate_exog_null_named(con):
    rng = np.random.default_rng(seed_for(1, 2, 3, 12))
    X = rng.standard_normal((60, 2))
    load_exog(con, "_t_ex", X)
    con.execute("CREATE OR REPLACE TABLE _t_exn AS "
                "SELECT t, j, CASE WHEN j = 2 AND t BETWEEN 50 AND 52 THEN NULL ELSE x END AS x FROM _t_ex")
    with pytest.raises(duckdb.Error, match=r"NULL x for exog column j = 2 at t in 50\.\.52"):
        con.execute("SELECT * FROM _sarimax_validate_exog('_t_exn', 60)")


def test_validate_exog_gap_named(con):
    rng = np.random.default_rng(seed_for(1, 2, 3, 13))
    X = rng.standard_normal((60, 2))
    load_exog(con, "_t_ex", X)
    con.execute("CREATE OR REPLACE TABLE _t_exg AS SELECT * FROM _t_ex WHERE NOT (j = 1 AND t = 30)")
    with pytest.raises(duckdb.Error, match=r"exog column j = 1 does not cover t in 30\.\.30"):
        con.execute("SELECT * FROM _sarimax_validate_exog('_t_exg', 60)")
    # coverage shorter than n_expected
    with pytest.raises(duckdb.Error, match=r"does not cover t in 61\.\.70"):
        con.execute("SELECT * FROM _sarimax_validate_exog('_t_ex', 70)")


def test_validate_exog_duplicate_named(con):
    rng = np.random.default_rng(seed_for(1, 2, 3, 14))
    X = rng.standard_normal((40, 2))
    load_exog(con, "_t_ex", X)
    con.execute("CREATE OR REPLACE TABLE _t_exd AS "
                "SELECT * FROM _t_ex UNION ALL SELECT t, j, x FROM _t_ex WHERE j = 2 AND t = 7")
    with pytest.raises(duckdb.Error, match=r"duplicate rows for exog column j = 2 at t in 7\.\.7"):
        con.execute("SELECT * FROM _sarimax_validate_exog('_t_exd', 40)")


def test_validate_exog_j_not_dense(con):
    rng = np.random.default_rng(seed_for(1, 2, 3, 15))
    X = rng.standard_normal((40, 2))
    load_exog(con, "_t_ex", X)
    con.execute("CREATE OR REPLACE TABLE _t_exj AS SELECT t, CASE WHEN j = 2 THEN 3 ELSE j END AS j, x FROM _t_ex")
    with pytest.raises(duckdb.Error, match=r"dense 1\.\.r"):
        con.execute("SELECT * FROM _sarimax_validate_exog('_t_exj', 40)")


# ---------------------------------------------------------------------------
# 7. rank check
# ---------------------------------------------------------------------------

def test_rank_check_ok_and_pivots_match_numpy(con):
    rng = np.random.default_rng(seed_for(4, 4, 4, 16))
    n, r = 120, 4
    X = rng.standard_normal((n, r))
    load_exog(con, "_t_ex", X)
    rows = con.execute("SELECT j, piv, pivmin, trace, ok FROM _sarimax_rank_check('_t_ex') ORDER BY j").fetchnumpy()
    G = X.T @ X
    ref = ref_pivlist(G)
    assert np.array_equal(rows["j"], np.arange(1, r + 1))
    assert bool(rows["ok"].all())
    # Gram entries come from a (bulk) SQL SUM; tiny summation-order differences
    # propagate through the elimination, hence a tolerance rather than equality.
    np.testing.assert_allclose(rows["piv"], ref, rtol=1e-9)
    np.testing.assert_allclose(rows["pivmin"], ref.min(), rtol=1e-9)
    np.testing.assert_allclose(rows["trace"], np.trace(G), rtol=1e-12)


def test_prep_pivots_scalar_macro_vs_numpy(con):
    rng = np.random.default_rng(seed_for(5, 5, 5, 17))
    for nn in (1, 2, 3, 5, 6):
        A = rng.standard_normal((nn, nn))
        A = A @ A.T + 0.1 * np.eye(nn)  # symmetric, like a Gram matrix
        flat = ", ".join(f"{float(v)!r}::DOUBLE" for v in A.reshape(-1))
        got = con.execute(f"SELECT _sarimax_prep_pivlist([{flat}], {nn}) AS pl, "
                          f"_sarimax_prep_pivots([{flat}], {nn}) AS pmin").fetchone()
        ref = ref_pivlist(A)
        np.testing.assert_allclose(np.array(got[0]), ref, rtol=1e-14)
        np.testing.assert_allclose(got[1], ref.min(), rtol=1e-14)


def test_rank_check_constant_column_differenced_to_zero(con):
    """The spec's canonical failure: a constant exog column differences to
    zero and must be rejected here, with the column named."""
    rng = np.random.default_rng(seed_for(6, 6, 6, 18))
    n = 80
    X = np.column_stack([gen_series(rng, n, 4), np.full(n, 7.5), rng.standard_normal(n)])
    load_exog(con, "_t_ex", X)
    con.execute("CREATE OR REPLACE TABLE _t_exdiff AS SELECT * FROM _sarimax_diff_exog('_t_ex', 1, 0, 4)")
    with pytest.raises(duckdb.Error, match=r"rank-deficient.*column j = 2"):
        con.execute("SELECT * FROM _sarimax_rank_check('_t_exdiff')")


def test_rank_check_near_collinear_pair(con):
    rng = np.random.default_rng(seed_for(7, 7, 7, 19))
    n = 100
    x1 = rng.standard_normal(n)
    x2 = x1 + 1e-13 * rng.standard_normal(n)
    load_exog(con, "_t_ex", np.column_stack([x1, x2]))
    with pytest.raises(duckdb.Error, match=r"rank-deficient.*column j = 2"):
        con.execute("SELECT * FROM _sarimax_rank_check('_t_ex')")


def test_rank_check_all_zero(con):
    load_exog(con, "_t_ex", np.zeros((30, 2)))
    with pytest.raises(duckdb.Error, match="identically zero"):
        con.execute("SELECT * FROM _sarimax_rank_check('_t_ex')")


def test_rank_check_single_column_ok(con):
    rng = np.random.default_rng(seed_for(8, 8, 8, 20))
    load_exog(con, "_t_ex", rng.standard_normal((50, 1)))
    rows = con.execute("SELECT j, ok FROM _sarimax_rank_check('_t_ex')").fetchall()
    assert rows == [(1, True)]


# ---------------------------------------------------------------------------
# 8. lag matrix (long form)
# ---------------------------------------------------------------------------

def test_lags_long_form(con):
    rng = np.random.default_rng(seed_for(9, 9, 9, 21))
    n, m = 40, 5
    y = rng.standard_normal(n)
    load_series(con, "_t_ser", y)
    res = con.execute(f"SELECT t, lag, value FROM _sarimax_lags('_t_ser', 't', 'y', {m}) ORDER BY t, lag").fetchnumpy()
    # rows only where the lag exists
    assert len(res["t"]) == (m + 1) * n - m * (m + 1) // 2
    for t, lag, value in zip(res["t"], res["lag"], res["value"]):
        assert t - lag >= 1
        assert value == y[t - lag - 1]
    # complete rows for t > m: exactly m+1 lags each
    counts = pd.Series(res["t"]).value_counts()
    assert (counts[counts.index > m] == m + 1).all()


def test_lags_m_zero_and_validation(con):
    y = np.arange(10, dtype=np.float64) + 1.0
    load_series(con, "_t_ser", y)
    res = con.execute("SELECT t, lag, value FROM _sarimax_lags('_t_ser', 't', 'y', 0) ORDER BY t").fetchnumpy()
    assert np.array_equal(res["value"], y) and set(res["lag"]) == {0}
    with pytest.raises(duckdb.Error, match="m must be >= 0"):
        con.execute("SELECT * FROM _sarimax_lags('_t_ser', 't', 'y', -1)")


# ---------------------------------------------------------------------------
# 9. ACF / PACF diagnostics
# ---------------------------------------------------------------------------

def _ar_series(rng, n):
    """AR(2)-ish data so the PACF has genuine structure."""
    e = rng.standard_normal(n + 50)
    y = np.zeros(n + 50)
    for i in range(2, n + 50):
        y[i] = 0.6 * y[i - 1] - 0.25 * y[i - 2] + e[i]
    return y[50:]


def test_acf_matches_reference(con):
    rng = np.random.default_rng(seed_for(10, 10, 10, 22))
    n, nlags = 180, 24
    y = _ar_series(rng, n)
    load_series(con, "_t_ser", y)
    res = fetch(con, f"SELECT lag, acf FROM _sarimax_acf('_t_ser', 't', 'y', {nlags}) ORDER BY lag")
    assert np.array_equal(res["lag"], np.arange(nlags + 1))
    if HAVE_STATSMODELS:
        ref = sm_acf(y, nlags=nlags, adjusted=False, fft=False)
    else:
        ref = ref_acf(y, nlags)
    np.testing.assert_allclose(res["acf"], ref, rtol=0.0, atol=1e-10)
    assert res["acf"][0] == 1.0


def test_pacf_matches_reference(con):
    rng = np.random.default_rng(seed_for(11, 11, 11, 23))
    n, nlags = 180, 20
    y = _ar_series(rng, n)
    load_series(con, "_t_ser", y)
    res = fetch(con, f"SELECT lag, pacf FROM _sarimax_pacf('_t_ser', 't', 'y', {nlags}) ORDER BY lag")
    assert np.array_equal(res["lag"], np.arange(nlags + 1))
    if HAVE_STATSMODELS:
        ref = sm_pacf(y, nlags=nlags, method="ywm")
    else:
        ref = ref_pacf(y, nlags)
    np.testing.assert_allclose(res["pacf"], ref, rtol=0.0, atol=1e-10)
    assert res["pacf"][0] == 1.0


def test_acf_validation(con):
    y = np.arange(20, dtype=np.float64)
    load_series(con, "_t_ser", y)
    with pytest.raises(duckdb.Error, match="nlags must be < n"):
        con.execute("SELECT * FROM _sarimax_acf('_t_ser', 't', 'y', 20)")
    with pytest.raises(duckdb.Error, match="nlags must be >= 0"):
        con.execute("SELECT * FROM _sarimax_acf('_t_ser', 't', 'y', -1)")
    load_series(con, "_t_const", np.full(20, 3.0))
    with pytest.raises(duckdb.Error, match="constant"):
        con.execute("SELECT * FROM _sarimax_acf('_t_const', 't', 'y', 5)")


def test_pacf_nlags_zero(con):
    y = np.sin(np.arange(30, dtype=np.float64))
    load_series(con, "_t_ser", y)
    rows = con.execute("SELECT lag, pacf FROM _sarimax_pacf('_t_ser', 't', 'y', 0)").fetchall()
    assert rows == [(0, 1.0)]


# ---------------------------------------------------------------------------
# 10. CLI load: clean, warning-free
# ---------------------------------------------------------------------------

def test_cli_load_clean():
    proc = subprocess.run(
        ["duckdb", "-c", ".read sql/01_prep.sql"],
        cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, f"duckdb CLI load failed: {proc.stderr}"
    combined = (proc.stdout + proc.stderr).lower()
    assert "deprecat" not in combined
    assert "warning" not in combined
    assert proc.stderr.strip() == ""
