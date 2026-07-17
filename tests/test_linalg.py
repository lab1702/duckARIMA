"""Acceptance tests for duckARIMA Layer 0 (sql/00_linalg.sql), spec 5.0.

Run:  .venv/Scripts/python.exe -m pytest tests/test_linalg.py -q

Requires the committed fixtures in tests/fixtures_linalg/ (regenerate --
deliberately, never silently -- with tests/gen_linalg_fixtures.py).

Every DuckDB connection runs with SET lambda_syntax = 'DISABLE_SINGLE_ARROW'
(spec 4.2), and the whole module escalates Python warnings to errors, so any
DuckDB deprecation warning fails the run.

Solver tolerance policy (see the fixture generator's docstring for the full
argument): two backward-stable solvers may legitimately disagree by
~0.1 * cond * eps per element relative to max |x|, so the flat 1e-10 band vs
numpy.linalg.solve is enforced for the well-conditioned and pivot systems
(cond <= 1e3, where the observed gap is <= ~2e-14) and a condition-scaled
band max(1e-10, 4 * eps * cond) for the mildly ill-conditioned systems
(cond 1e6..1e9). The sharp correctness instrument at every condition number
is the BITWISE comparison against the fixture's float64 Gauss-Jordan control,
which mirrors the SQL algorithm operation for operation.
"""
import pathlib
import time
import warnings

import duckdb
import numpy as np
import pyarrow.parquet as pq
import pytest

pytestmark = pytest.mark.filterwarnings("error")

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIXDIR = ROOT / "tests" / "fixtures_linalg"
MACROS = (ROOT / "sql" / "00_linalg.sql").read_text()
EPS = float(np.finfo(np.float64).eps)

# ---------------------------------------------------------------------------
# fixture data (NumPy side), loaded once at import for parametrization
# ---------------------------------------------------------------------------


def _read_dict(name):
    return pq.read_table(FIXDIR / f"{name}.parquet").to_pydict()


def _read_mats(name, key_name):
    t = _read_dict(name)
    keys = np.asarray(t[key_name])
    ii = np.asarray(t["i"])
    jj = np.asarray(t["j"])
    vv = np.asarray(t["v"])
    out = {}
    for k in np.unique(keys):
        sel = keys == k
        M = np.zeros((ii[sel].max(), jj[sel].max()))
        M[ii[sel] - 1, jj[sel] - 1] = vv[sel]
        out[int(k)] = M
    return out


SOLVE_META = {int(s): {"kind": k, "n": int(n), "nrhs": int(r), "cond": float(c),
                       "pivmin_gj": float(p), "ok_gj": bool(o), "has_x": bool(h)}
              for s, k, n, r, c, p, o, h in zip(*_read_dict("solve_meta").values())}
SOLVE_A = _read_mats("solve_a", "sys_id")
SOLVE_B = _read_mats("solve_b", "sys_id")
SOLVE_X_NP = _read_mats("solve_x_np", "sys_id")
SOLVE_X_GJ = _read_mats("solve_x_gj", "sys_id")
SOLVE_IDS = sorted(SOLVE_META)
NONSING_IDS = [s for s in SOLVE_IDS if SOLVE_META[s]["has_x"]]
SINGULAR_IDS = [s for s in SOLVE_IDS if not SOLVE_META[s]["has_x"]]

MM_META = {int(c): {"m": int(m), "k": int(k), "n": int(n), "c": float(sc)}
           for c, m, k, n, sc in zip(*_read_dict("mm_meta").values())}
MM_A = _read_mats("mm_a", "case_id")
MM_B = _read_mats("mm_b", "case_id")
MM_C = _read_mats("mm_c", "case_id")
MM_D = _read_mats("mm_d", "case_id")
MM_AT = _read_mats("mm_at", "case_id")
MM_APD = _read_mats("mm_apd", "case_id")
MM_SCALED = _read_mats("mm_scaled", "case_id")
_vec_raw = _read_dict("mm_vec")
MM_VEC = {}
for cid, idx, v in zip(_vec_raw["case_id"], _vec_raw["idx"], _vec_raw["v"]):
    MM_VEC.setdefault(int(cid), []).append((int(idx), float(v)))
MM_VEC = {c: np.array([v for _, v in sorted(rows)]) for c, rows in MM_VEC.items()}
MM_IDS = sorted(MM_META)
MM_30 = next(c for c, s in MM_META.items() if (s["m"], s["k"], s["n"]) == (30, 30, 30))

KRON_META = {int(c): {"ma": int(ma), "na": int(na), "mb": int(mb), "nb": int(nb),
                      "has_result": bool(h)}
             for c, ma, na, mb, nb, h in zip(*_read_dict("kron_meta").values())}
KRON_A = _read_mats("kron_a", "case_id")
KRON_B = _read_mats("kron_b", "case_id")
KRON_C = _read_mats("kron_c", "case_id")
KRON_IDS = [c for c in sorted(KRON_META) if KRON_META[c]["has_result"]]
KRON_PLAN = next(c for c in sorted(KRON_META) if not KRON_META[c]["has_result"])

LYAP_META = {int(c): {"k": int(k), "rho": float(r)}
             for c, k, r in zip(*_read_dict("lyap_meta").values())}
LYAP_T = _read_mats("lyap_t", "case_id")
LYAP_RQR = _read_mats("lyap_rqr", "case_id")
LYAP_P = _read_mats("lyap_p", "case_id")
LYAP_IDS = sorted(LYAP_META)


def solve_band(sid):
    meta = SOLVE_META[sid]
    if meta["kind"] == "ill":
        return max(1e-10, 4.0 * EPS * meta["cond"])
    return 1e-10


# ---------------------------------------------------------------------------
# DuckDB connections and one full solver/multiply pass per thread mode
# ---------------------------------------------------------------------------


def make_con(threads=None):
    con = duckdb.connect()
    con.execute("SET lambda_syntax = 'DISABLE_SINGLE_ARROW';")
    if threads is not None:
        con.execute(f"SET threads = {threads};")
    con.execute(MACROS)
    for name in ["solve_meta", "solve_a", "solve_b", "mm_a", "mm_b",
                 "kron_a", "kron_b", "lyap_t", "lyap_rqr"]:
        path = (FIXDIR / f"{name}.parquet").as_posix()
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM read_parquet('{path}')")
    return con


def stage_system(con, sid):
    con.execute(f"CREATE OR REPLACE TABLE fix_cur_a AS "
                f"SELECT i, j, v FROM solve_a WHERE sys_id = {sid}")
    con.execute(f"CREATE OR REPLACE TABLE fix_cur_b AS "
                f"SELECT i, j, v FROM solve_b WHERE sys_id = {sid}")


LIST_SOLVE_SQL = """
WITH _aug AS (
  SELECT list(v ORDER BY i, j) AS a FROM (
    SELECT i, j, v FROM fix_cur_a
    UNION ALL
    SELECT i, j + {n} AS j, v FROM fix_cur_b
  )
)
SELECT _sarimax_solve_list(a, {n}, {nrhs}) AS r FROM _aug
"""


def run_list_solve(con, sid):
    meta = SOLVE_META[sid]
    stage_system(con, sid)
    r = con.execute(LIST_SOLVE_SQL.format(n=meta["n"], nrhs=meta["nrhs"])).fetchone()[0]
    return (np.array(r["x"], dtype=np.float64).reshape(meta["n"], meta["nrhs"]),
            float(r["pivmin"]), bool(r["ok"]))


def run_rsolve(con, sid):
    meta = SOLVE_META[sid]
    stage_system(con, sid)
    rows = con.execute("SELECT i, j, v, pivmin, ok "
                       "FROM _sarimax_rsolve('fix_cur_a', 'fix_cur_b') "
                       "ORDER BY i, j").fetchall()
    assert len(rows) == meta["n"] * meta["nrhs"]
    x = np.array([r[2] for r in rows], dtype=np.float64).reshape(meta["n"], meta["nrhs"])
    return x, float(rows[0][3]), bool(rows[0][4])


def run_mmul(con, cid):
    s = MM_META[cid]
    r = con.execute(f"""
        WITH za AS (SELECT list(v ORDER BY i, j) AS al FROM mm_a WHERE case_id = {cid}),
             zb AS (SELECT list(v ORDER BY i, j) AS bl FROM mm_b WHERE case_id = {cid})
        SELECT _sarimax_mmul(al, bl, {s['m']}, {s['k']}, {s['n']}) AS r FROM za, zb
    """).fetchone()[0]
    return np.array(r, dtype=np.float64).reshape(s["m"], s["n"])


def run_rmul(con, cid):
    s = MM_META[cid]
    con.execute(f"CREATE OR REPLACE TABLE fix_mm_a AS "
                f"SELECT i, j, v FROM mm_a WHERE case_id = {cid}")
    con.execute(f"CREATE OR REPLACE TABLE fix_mm_b AS "
                f"SELECT i, j, v FROM mm_b WHERE case_id = {cid}")
    rows = con.execute("SELECT i, j, v FROM _sarimax_rmul('fix_mm_a', 'fix_mm_b') "
                       "ORDER BY i, j").fetchall()
    assert len(rows) == s["m"] * s["n"]
    return np.array([r[2] for r in rows], dtype=np.float64).reshape(s["m"], s["n"])


def run_full_pass(con):
    """Everything the determinism requirement compares across thread counts:
    both solvers on every system, and both multiplies on every case."""
    out = {"list": {}, "rsolve": {}, "mmul": {}, "rmul": {}}
    for sid in SOLVE_IDS:
        out["list"][sid] = run_list_solve(con, sid)
        out["rsolve"][sid] = run_rsolve(con, sid)
    for cid in MM_IDS:
        out["mmul"][cid] = run_mmul(con, cid)
        out["rmul"][cid] = run_rmul(con, cid)
    return out


@pytest.fixture(scope="session")
def con_default():
    con = make_con(None)
    yield con
    con.close()


@pytest.fixture(scope="session")
def pass_default(con_default):
    return run_full_pass(con_default)


@pytest.fixture(scope="session")
def pass_threads1():
    con = make_con(1)
    try:
        return run_full_pass(con)
    finally:
        con.close()


# ---------------------------------------------------------------------------
# linear systems: both solvers vs numpy.linalg.solve and vs the bitwise control
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sid", NONSING_IDS)
def test_solve_list_vs_numpy(pass_default, sid):
    x, pivmin, ok = pass_default["list"][sid]
    x_np = SOLVE_X_NP[sid]
    assert ok, f"solver reported failure, pivmin={pivmin}"
    rel = np.max(np.abs(x - x_np)) / np.max(np.abs(x_np))
    assert rel <= solve_band(sid), f"rel diff {rel:.3e} > {solve_band(sid):.3e}"


@pytest.mark.parametrize("sid", NONSING_IDS)
def test_solve_list_bitwise_vs_control(pass_default, sid):
    x, pivmin, ok = pass_default["list"][sid]
    assert x.tobytes() == SOLVE_X_GJ[sid].tobytes(), \
        "list solver diverged from the same-algorithm float64 control"
    assert pivmin == SOLVE_META[sid]["pivmin_gj"]
    assert ok == SOLVE_META[sid]["ok_gj"]


@pytest.mark.parametrize("sid", NONSING_IDS)
def test_rsolve_vs_numpy(pass_default, sid):
    x, pivmin, ok = pass_default["rsolve"][sid]
    x_np = SOLVE_X_NP[sid]
    assert ok, f"solver reported failure, pivmin={pivmin}"
    rel = np.max(np.abs(x - x_np)) / np.max(np.abs(x_np))
    assert rel <= solve_band(sid), f"rel diff {rel:.3e} > {solve_band(sid):.3e}"


@pytest.mark.parametrize("sid", NONSING_IDS)
def test_rsolve_bitwise_vs_control(pass_default, sid):
    x, pivmin, ok = pass_default["rsolve"][sid]
    assert x.tobytes() == SOLVE_X_GJ[sid].tobytes(), \
        "relational solver diverged from the same-algorithm float64 control"
    assert pivmin == SOLVE_META[sid]["pivmin_gj"]
    assert ok == SOLVE_META[sid]["ok_gj"]


def test_pivoting_proof(pass_default):
    """[[1e-18, 1], [1, 1]] is well conditioned (~2.6) but destroys unpivoted
    elimination; correct answers here prove the pivot/swap path works."""
    (sid,) = [s for s in SOLVE_IDS if SOLVE_META[s]["kind"] == "pivot"]
    for solver in ("list", "rsolve"):
        x, pivmin, ok = pass_default[solver][sid]
        assert ok and pivmin == 1.0
        rel = np.max(np.abs(x - SOLVE_X_NP[sid])) / np.max(np.abs(SOLVE_X_NP[sid]))
        assert rel <= 1e-10


@pytest.mark.parametrize("sid", SINGULAR_IDS)
def test_singular_reports_failure(pass_default, sid):
    for solver in ("list", "rsolve"):
        x, pivmin, ok = pass_default[solver][sid]
        assert not ok, f"{solver}: singular system must return ok = false"
        assert pivmin < 1e-12


# ---------------------------------------------------------------------------
# determinism: threads=1 vs default must be BITWISE identical (spec 4.2)
# ---------------------------------------------------------------------------


def test_determinism_bitwise_across_threads(pass_default, pass_threads1):
    for group in ("list", "rsolve"):
        for sid in SOLVE_IDS:
            xd, pd, okd = pass_default[group][sid]
            x1, p1, ok1 = pass_threads1[group][sid]
            assert xd.tobytes() == x1.tobytes(), f"{group} sys {sid}: thread count changed bits"
            assert pd == p1 and okd == ok1
    for group in ("mmul", "rmul"):
        for cid in MM_IDS:
            assert pass_default[group][cid].tobytes() == pass_threads1[group][cid].tobytes(), \
                f"{group} case {cid}: thread count changed bits"


# ---------------------------------------------------------------------------
# multiply / transpose / add / scale / Kronecker vs NumPy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cid", MM_IDS)
def test_mmul_vs_numpy(pass_default, cid):
    C = pass_default["mmul"][cid]
    rel = np.max(np.abs(C - MM_C[cid])) / max(np.max(np.abs(MM_C[cid])), 1e-300)
    assert rel <= 1e-13


@pytest.mark.parametrize("cid", MM_IDS)
def test_rmul_vs_numpy_and_matches_mmul(pass_default, cid):
    C = pass_default["rmul"][cid]
    rel = np.max(np.abs(C - MM_C[cid])) / max(np.max(np.abs(MM_C[cid])), 1e-300)
    assert rel <= 1e-13
    # same ascending-inner-index fold in both encodings => identical bits
    assert C.tobytes() == pass_default["mmul"][cid].tobytes()


@pytest.mark.parametrize("cid", MM_IDS)
def test_mtrans_exact(con_default, cid):
    s = MM_META[cid]
    r = con_default.execute(f"""
        WITH za AS (SELECT list(v ORDER BY i, j) AS al FROM mm_a WHERE case_id = {cid})
        SELECT _sarimax_mtrans(al, {s['m']}, {s['k']}) AS r FROM za
    """).fetchone()[0]
    got = np.array(r).reshape(s["k"], s["m"])
    assert got.tobytes() == MM_AT[cid].tobytes()


@pytest.mark.parametrize("cid", MM_IDS)
def test_madd_msub_mscale_exact(con_default, cid):
    s = MM_META[cid]
    add, sub, scl = con_default.execute(f"""
        WITH za AS (SELECT list(v ORDER BY i, j) AS al FROM mm_a WHERE case_id = {cid}),
             zd AS (SELECT list(v ORDER BY i, j) AS dl
                    FROM read_parquet('{(FIXDIR / "mm_d.parquet").as_posix()}')
                    WHERE case_id = {cid})
        SELECT _sarimax_madd(al, dl), _sarimax_msub(al, dl),
               _sarimax_mscale(al, {s['c']!r}::DOUBLE)
        FROM za, zd
    """).fetchone()
    m, k = s["m"], s["k"]
    assert np.array(add).reshape(m, k).tobytes() == MM_APD[cid].tobytes()
    assert np.array(sub).reshape(m, k).tobytes() == (MM_A[cid] - MM_D[cid]).tobytes()
    assert np.array(scl).reshape(m, k).tobytes() == MM_SCALED[cid].tobytes()


def test_msym_exact(con_default):
    cid = MM_30
    r = con_default.execute(f"""
        WITH za AS (SELECT list(v ORDER BY i, j) AS al FROM mm_a WHERE case_id = {cid})
        SELECT _sarimax_msym(al, 30) AS r FROM za
    """).fetchone()[0]
    ref = (MM_A[cid] + MM_A[cid].T) * 0.5
    assert np.array(r).reshape(30, 30).tobytes() == ref.tobytes()


def test_mget_exact(con_default):
    cid = next(c for c, s in MM_META.items() if (s["m"], s["k"]) == (2, 3))
    for (i, j) in [(1, 1), (1, 3), (2, 2), (2, 3)]:
        v = con_default.execute(f"""
            WITH za AS (SELECT list(v ORDER BY i, j) AS al FROM mm_a WHERE case_id = {cid})
            SELECT _sarimax_mget(al, {i}, {j}, 3) FROM za
        """).fetchone()[0]
        assert v == MM_A[cid][i - 1, j - 1]


def test_meye_mzeros_values_and_types(con_default):
    eye, t_eye, z, t_z = con_default.execute(
        "SELECT _sarimax_meye(4), typeof(_sarimax_meye(4)), "
        "_sarimax_mzeros(2, 3), typeof(_sarimax_mzeros(2, 3))").fetchone()
    assert np.array(eye).reshape(4, 4).tobytes() == np.eye(4).tobytes()
    assert np.array(z).tobytes() == np.zeros(6).tobytes()
    # DECIMAL-literal leak guard (spec 12): these must be DOUBLE[], never DECIMAL[]
    assert t_eye == "DOUBLE[]" and t_z == "DOUBLE[]"


@pytest.mark.parametrize("cid", KRON_IDS)
def test_mkron_exact(con_default, cid):
    s = KRON_META[cid]
    r = con_default.execute(f"""
        WITH za AS (SELECT list(v ORDER BY i, j) AS al FROM kron_a WHERE case_id = {cid}),
             zb AS (SELECT list(v ORDER BY i, j) AS bl FROM kron_b WHERE case_id = {cid})
        SELECT _sarimax_mkron(al, bl, {s['ma']}, {s['na']}, {s['mb']}, {s['nb']}) AS r
        FROM za, zb
    """).fetchone()[0]
    got = np.array(r).reshape(s["ma"] * s["mb"], s["na"] * s["nb"])
    assert got.tobytes() == KRON_C[cid].tobytes()


@pytest.mark.parametrize("cid", KRON_IDS)
def test_rkron_exact(con_default, cid):
    s = KRON_META[cid]
    con_default.execute(f"CREATE OR REPLACE TABLE fix_kr_a AS "
                        f"SELECT i, j, v FROM kron_a WHERE case_id = {cid}")
    con_default.execute(f"CREATE OR REPLACE TABLE fix_kr_b AS "
                        f"SELECT i, j, v FROM kron_b WHERE case_id = {cid}")
    rows = con_default.execute("SELECT i, j, v FROM _sarimax_rkron('fix_kr_a', 'fix_kr_b') "
                               "ORDER BY i, j").fetchall()
    got = np.array([r[2] for r in rows]).reshape(s["ma"] * s["mb"], s["na"] * s["nb"])
    assert got.tobytes() == KRON_C[cid].tobytes()


# ---------------------------------------------------------------------------
# relational transpose / add / scale / constructors
# ---------------------------------------------------------------------------


def test_rtrans_radd_rscale_exact(con_default):
    cid = MM_30
    con_default.execute(f"CREATE OR REPLACE TABLE fix_r_a AS "
                        f"SELECT i, j, v FROM mm_a WHERE case_id = {cid}")
    con_default.execute(f"CREATE OR REPLACE TABLE fix_r_d AS SELECT * FROM "
                        f"read_parquet('{(FIXDIR / 'mm_d.parquet').as_posix()}') "
                        f"WHERE case_id = {cid}")
    tr = con_default.execute("SELECT i, j, v FROM _sarimax_rtrans('fix_r_a') "
                             "ORDER BY i, j").fetchall()
    got = np.array([r[2] for r in tr]).reshape(30, 30)
    assert got.tobytes() == MM_AT[cid].tobytes()
    ad = con_default.execute("SELECT i, j, v FROM _sarimax_radd('fix_r_a', 'fix_r_d') "
                             "ORDER BY i, j").fetchall()
    got = np.array([r[2] for r in ad]).reshape(30, 30)
    assert got.tobytes() == MM_APD[cid].tobytes()
    c = MM_META[cid]["c"]
    sc = con_default.execute(f"SELECT i, j, v FROM _sarimax_rscale('fix_r_a', {c!r}::DOUBLE) "
                             "ORDER BY i, j").fetchall()
    got = np.array([r[2] for r in sc]).reshape(30, 30)
    assert got.tobytes() == MM_SCALED[cid].tobytes()


def test_reye_rzeros(con_default):
    rows = con_default.execute("SELECT i, j, v FROM _sarimax_reye(3) ORDER BY i, j").fetchall()
    assert np.array([r[2] for r in rows]).reshape(3, 3).tobytes() == np.eye(3).tobytes()
    desc = {r[0]: r[1] for r in
            con_default.execute("DESCRIBE SELECT * FROM _sarimax_reye(3)").fetchall()}
    assert desc == {"i": "INTEGER", "j": "INTEGER", "v": "DOUBLE"}
    rows = con_default.execute("SELECT i, j, v FROM _sarimax_rzeros(2, 3) ORDER BY i, j").fetchall()
    assert len(rows) == 6 and all(r[2] == 0.0 for r in rows)


# ---------------------------------------------------------------------------
# vec / unvec and the encoding round-trips (exact)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cid", MM_IDS)
def test_mvec_matches_numpy_and_roundtrips(con_default, cid):
    s = MM_META[cid]
    vec, rt = con_default.execute(f"""
        WITH za AS (SELECT list(v ORDER BY i, j) AS al FROM mm_a WHERE case_id = {cid})
        SELECT _sarimax_mvec(al, {s['m']}, {s['k']}),
               _sarimax_munvec(_sarimax_mvec(al, {s['m']}, {s['k']}), {s['m']}, {s['k']})
        FROM za
    """).fetchone()
    # classic column-stacking vec == numpy flatten(order='F'), bit for bit
    assert np.array(vec).tobytes() == MM_VEC[cid].tobytes()
    # unvec(vec(A)) == A, bit for bit
    assert np.array(rt).tobytes() == MM_A[cid].ravel().tobytes()


def test_mat_to_list_list_to_mat_roundtrip(con_default):
    cid = MM_30
    con_default.execute(f"CREATE OR REPLACE TABLE fix_rt_m AS "
                        f"SELECT i, j, v FROM mm_a WHERE case_id = {cid}")
    lst, m, n = con_default.execute(
        "SELECT v, m, n FROM _sarimax_mat_to_list('fix_rt_m')").fetchone()
    assert (m, n) == (30, 30)
    assert np.array(lst).tobytes() == MM_A[cid].ravel().tobytes()  # row-major, exact
    con_default.execute("CREATE OR REPLACE TABLE fix_rt_l AS "
                        "SELECT * FROM _sarimax_mat_to_list('fix_rt_m')")
    back = con_default.execute("""
        SELECT i, j, v FROM _sarimax_list_to_mat(
          (SELECT v FROM fix_rt_l), (SELECT m FROM fix_rt_l), (SELECT n FROM fix_rt_l))
        ORDER BY i, j
    """).fetchall()
    orig = con_default.execute("SELECT i, j, v FROM fix_rt_m ORDER BY i, j").fetchall()
    assert back == orig  # exact tuple equality: indices and doubles


# ---------------------------------------------------------------------------
# inversion
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sid", [s for s in NONSING_IDS
                                 if SOLVE_META[s]["kind"] == "well"
                                 and SOLVE_META[s]["n"] in (8, 16, 32)][:3])
def test_inv_list(con_default, sid):
    n = SOLVE_META[sid]["n"]
    r = con_default.execute(f"""
        WITH za AS (SELECT list(v ORDER BY i, j) AS al FROM solve_a WHERE sys_id = {sid})
        SELECT _sarimax_inv_list(al, {n}) AS r FROM za
    """).fetchone()[0]
    assert r["ok"]
    inv = np.array(r["x"]).reshape(n, n)
    ref = np.linalg.inv(SOLVE_A[sid])
    assert np.max(np.abs(inv - ref)) / np.max(np.abs(ref)) <= 1e-10
    assert np.max(np.abs(SOLVE_A[sid] @ inv - np.eye(n))) <= 1e-11


def test_inv_list_singular(con_default):
    (sid,) = SINGULAR_IDS
    r = con_default.execute(f"""
        WITH za AS (SELECT list(v ORDER BY i, j) AS al FROM solve_a WHERE sys_id = {sid})
        SELECT _sarimax_inv_list(al, 4) AS r FROM za
    """).fetchone()[0]
    assert not r["ok"]


# ---------------------------------------------------------------------------
# discrete Lyapunov solver
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cid", LYAP_IDS)
def test_lyap(con_default, cid):
    k = LYAP_META[cid]["k"]
    r = con_default.execute(f"""
        WITH zt AS (SELECT list(v ORDER BY i, j) AS tl FROM lyap_t WHERE case_id = {cid}),
             zq AS (SELECT list(v ORDER BY i, j) AS ql FROM lyap_rqr WHERE case_id = {cid})
        SELECT _sarimax_lyap(tl, ql, {k}) AS p FROM zt, zq
    """).fetchone()[0]
    P = np.array(r).reshape(k, k)
    ref = LYAP_P[cid]
    rel = np.max(np.abs(P - ref)) / np.max(np.abs(ref))
    assert rel <= 1e-10, f"lyap k={k}: rel {rel:.3e} vs numpy reference"
    # substitute back: P - T P T' - RQR' (spec 6 validation)
    T, RQR = LYAP_T[cid], LYAP_RQR[cid]
    resid = np.max(np.abs(P - T @ P @ T.T - RQR))
    assert resid <= 1e-9 * np.max(np.abs(P)), f"lyap k={k}: residual {resid:.3e}"


# ---------------------------------------------------------------------------
# plan-shape regression tests (spec 4.3): hash joins, no nested loops
# ---------------------------------------------------------------------------


def _plan(con, sql):
    return "\n".join(str(r[1]) for r in con.execute("EXPLAIN " + sql).fetchall())


def test_plan_rmul_hash_join(con_default):
    con_default.execute(f"CREATE OR REPLACE TABLE fix_plan_a AS "
                        f"SELECT i, j, v FROM mm_a WHERE case_id = {MM_30}")
    con_default.execute(f"CREATE OR REPLACE TABLE fix_plan_b AS "
                        f"SELECT i, j, v FROM mm_b WHERE case_id = {MM_30}")
    plan = _plan(con_default, "SELECT * FROM _sarimax_rmul('fix_plan_a', 'fix_plan_b')")
    assert "HASH_JOIN" in plan
    assert "NESTED_LOOP" not in plan
    assert "BLOCKWISE" not in plan


def test_plan_rkron_hash_join(con_default):
    # the |A| x |B| output grid is inherently a cross product, but both matrix
    # lookups are phrased as equi-joins on computed grid keys and must hash
    con_default.execute(f"CREATE OR REPLACE TABLE fix_plan_ka AS "
                        f"SELECT i, j, v FROM kron_a WHERE case_id = {KRON_PLAN}")
    con_default.execute(f"CREATE OR REPLACE TABLE fix_plan_kb AS "
                        f"SELECT i, j, v FROM kron_b WHERE case_id = {KRON_PLAN}")
    plan = _plan(con_default, "SELECT * FROM _sarimax_rkron('fix_plan_ka', 'fix_plan_kb')")
    assert "HASH_JOIN" in plan
    assert "NESTED_LOOP" not in plan
    assert "BLOCKWISE" not in plan


# ---------------------------------------------------------------------------
# timing smoke test (spec 7: generous 5x headroom)
# ---------------------------------------------------------------------------


def test_timing_solve_196(con_default):
    (sid,) = [s for s in SOLVE_IDS
              if SOLVE_META[s]["n"] == 196 and SOLVE_META[s]["kind"] == "well"]
    meta = SOLVE_META[sid]
    stage_system(con_default, sid)
    t0 = time.perf_counter()
    r = con_default.execute(LIST_SOLVE_SQL.format(n=meta["n"], nrhs=meta["nrhs"])).fetchone()[0]
    dt_list = time.perf_counter() - t0
    assert r["ok"]
    t0 = time.perf_counter()
    rows = con_default.execute("SELECT max(abs(v)) AS mx, bool_and(ok) AS ok "
                               "FROM _sarimax_rsolve('fix_cur_a', 'fix_cur_b')").fetchone()
    dt_rel = time.perf_counter() - t0
    assert rows[1]
    assert dt_list < 10.0, f"196x196 list solve took {dt_list:.2f}s"
    assert dt_rel < 10.0, f"196x196 relational solve took {dt_rel:.2f}s"


# ---------------------------------------------------------------------------
# types, syntax discipline, and warning hygiene
# ---------------------------------------------------------------------------


def test_solve_struct_type(con_default):
    t = con_default.execute(
        "SELECT typeof(_sarimax_solve_list([2e0, 1e0, 5e0, 1e0, 3e0, 1e1], 2, 1))"
    ).fetchone()[0]
    assert t == "STRUCT(x DOUBLE[], pivmin DOUBLE, ok BOOLEAN)"
    t = con_default.execute("SELECT typeof(_sarimax_mmul([1e0], [1e0], 1, 1, 1))").fetchone()[0]
    assert t == "DOUBLE[]"
    t = con_default.execute("SELECT typeof(_sarimax_lyap([5e-1], [3e0], 1))").fetchone()[0]
    assert t == "DOUBLE[]"
    desc = {r[0]: r[1] for r in con_default.execute(
        "DESCRIBE SELECT * FROM _sarimax_rsolve('fix_cur_a', 'fix_cur_b')").fetchall()}
    assert desc == {"i": "INTEGER", "j": "INTEGER", "v": "DOUBLE",
                    "pivmin": "DOUBLE", "ok": "BOOLEAN"}


def test_arrow_lambda_rejected(con_default):
    """The session setting must hard-fail deprecated arrow lambdas (spec 4.2)."""
    with pytest.raises(duckdb.Error):
        con_default.execute("SELECT list_transform([1, 2], x -> x + 1)")


def test_no_warnings_full_surface():
    """Load the macro file and touch every macro; any warning (deprecation or
    otherwise) raised through the Python client fails here explicitly."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        con = make_con(None)
        con.execute("""
            SELECT _sarimax_mget([1e0, 2e0], 1, 2, 2),
                   _sarimax_meye(3), _sarimax_mzeros(2, 2),
                   _sarimax_madd([1e0], [2e0]), _sarimax_msub([1e0], [2e0]),
                   _sarimax_mscale([1e0], 2e0),
                   _sarimax_mtrans([1e0, 2e0], 1, 2),
                   _sarimax_msym([1e0, 2e0, 3e0, 4e0], 2),
                   _sarimax_mmul([1e0, 2e0], [3e0, 4e0], 1, 2, 1),
                   _sarimax_mkron([1e0], [2e0], 1, 1, 1, 1),
                   _sarimax_mvec([1e0, 2e0], 1, 2),
                   _sarimax_munvec([1e0, 2e0], 1, 2),
                   _sarimax_maug([1e0], [2e0], 1, 1),
                   _sarimax_solve_list([2e0, 4e0], 1, 1),
                   _sarimax_inv_list([2e0], 1),
                   _sarimax_lyap([5e-1], [3e0], 1)
        """).fetchall()
        con.execute("CREATE TABLE warm_a AS SELECT 1::INT AS i, 1::INT AS j, 2e0 AS v")
        con.execute("SELECT * FROM _sarimax_reye(2)").fetchall()
        con.execute("SELECT * FROM _sarimax_rzeros(2, 2)").fetchall()
        con.execute("SELECT * FROM _sarimax_rtrans('warm_a')").fetchall()
        con.execute("SELECT * FROM _sarimax_rscale('warm_a', 2e0)").fetchall()
        con.execute("SELECT * FROM _sarimax_radd('warm_a', 'warm_a')").fetchall()
        con.execute("SELECT * FROM _sarimax_rmul('warm_a', 'warm_a')").fetchall()
        con.execute("SELECT * FROM _sarimax_rkron('warm_a', 'warm_a')").fetchall()
        con.execute("SELECT * FROM _sarimax_mat_to_list('warm_a')").fetchall()
        con.execute("SELECT * FROM _sarimax_list_to_mat([1e0], 1, 1)").fetchall()
        con.execute("SELECT * FROM _sarimax_rsolve('warm_a', 'warm_a')").fetchall()
        con.close()
    assert rec == [], f"warnings emitted: {[str(w.message) for w in rec]}"
