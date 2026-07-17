#!/usr/bin/env python
"""Fixture generator for duckARIMA Layer 0 (sql/00_linalg.sql).

Writes deterministic (fixed-seed) NumPy float64 reference fixtures as Parquet
to tests/fixtures_linalg/ -- Parquet rather than CSV so doubles round-trip
losslessly as binary IEEE 754 values (spec section 8). Run once, offline:

    .venv/Scripts/python.exe tests/gen_linalg_fixtures.py

Fixtures are committed and never regenerated silently (spec section 10/12).

Contents
--------
solve_*   : 50 random linear systems A X = B (sizes spanning {1,2,3,4,8,16,
            32,64} plus one each of {128,196,256}; 38 well-conditioned with
            cond <= 1e3, 12 mildly ill-conditioned with cond log-spaced
            1e6..1e9, all built via SVD with controlled spectrum), plus a
            'pivot' system ([[1e-18, 1], [1, 1]]: harmless condition number,
            lethal to elimination without partial pivoting) and an exactly
            'singular' system (integer entries, one row an exact sum of two
            others). For every non-singular system two references are stored:
              solve_x_np : numpy.linalg.solve (LAPACK) solutions
              solve_x_gj : a float64 Gauss-Jordan-with-partial-pivoting
                           CONTROL that mirrors the SQL solvers operation for
                           operation and rounding for rounding -- the SQL
                           result must match it BITWISE.
            The control exists because two backward-stable solvers may
            legitimately disagree by ~0.1 * cond * eps per element (relative
            to max |x|): at cond 1e9 that is ~2e-8, so a flat 1e-10 agreement
            with LAPACK is unattainable for ANY implementation on the
            ill-conditioned systems. Tests therefore assert <= 1e-10 vs numpy
            for the well-conditioned systems and a condition-scaled band
            max(1e-10, 4*eps*cond) for the ill-conditioned ones, while the
            bitwise control comparison pins the implementation exactly at all
            condition numbers.
mm_*      : multiply/transpose/add/scale fixtures (A, B, A@B, D, A', A+D,
            c*A, and vec(A) = A.flatten(order='F')).
kron_*    : Kronecker fixtures (A, B, np.kron(A, B)); the 30x30 pair is for
            the plan-shape test only (no stored result).
lyap_*    : 10 discrete-Lyapunov cases P = T P T' + R Q R' with spectral
            radius(T) < 0.95, k in {1,2,3,4,8,14,16}; reference computed the
            same scipy-free way the SQL does it: solve
            (I_{k^2} - kron(T, T)) vec(P) = vec(RQR') with column-stacking vec.
"""
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SEED = 20260716
OUT = Path(__file__).resolve().parent / "fixtures_linalg"


# ---------------------------------------------------------------------------
# Gauss-Jordan with partial pivoting: float64 control implementation.
# Mirrors _sarimax_solve_list / _sarimax_rsolve exactly:
#   * pivot = argmax |M(r, k)|, r = k..n, FIRST row on ties (np.argmax);
#   * swap, normalize pivot row (v / piv), eliminate every other row as
#     v - colk * (pivot_row_v / piv)  -- normalized row computed first;
#   * a pivot below 1e-300 skips the step (no swap, matrix unchanged);
#   * pivmin = smallest selected |pivot|; ok = pivmin > 1e-12.
# Every operation is an elementwise IEEE-754 double op in the same order as
# the SQL, so results are expected to agree BITWISE.
# ---------------------------------------------------------------------------
def gauss_jordan_pp(A, B):
    n = A.shape[0]
    M = np.hstack([A, B]).astype(np.float64)
    pivmin = np.float64(1e308)
    for k in range(n):
        p = k + int(np.argmax(np.abs(M[k:, k])))
        pivval = M[p, k]
        pivabs = abs(pivval)
        pivmin = min(pivmin, pivabs)
        if pivabs < 1e-300:
            continue
        if p != k:
            M[[k, p], :] = M[[p, k], :]
        norm_row = M[k, :] / pivval          # normalized pivot row, once
        colk = M[:, k].copy()                # multipliers, pre-update
        M = M - np.outer(colk, norm_row)     # v - colk * norm_row
        M[k, :] = norm_row
    return M[:, n:], float(pivmin), bool(pivmin > 1e-12)


def controlled_matrix(rng, n, cond):
    """Random n x n matrix with SVD-controlled spectrum (cond >= 1)."""
    if n == 1:
        return np.array([[rng.standard_normal() + np.sign(rng.standard_normal()) * 1.0]])
    U, _ = np.linalg.qr(rng.standard_normal((n, n)))
    V, _ = np.linalg.qr(rng.standard_normal((n, n)))
    s = np.logspace(0.0, -np.log10(cond), n)
    return (U * s) @ V.T


def long_rows(store, key_name, key, M):
    """Append matrix M in dense long form (key, i, j, v) to store."""
    m, n = M.shape
    ii, jj = np.indices((m, n))
    store["key"].extend([key] * (m * n))
    store["i"].extend((ii + 1).ravel().tolist())
    store["j"].extend((jj + 1).ravel().tolist())
    store["v"].extend(M.ravel().tolist())


def write_long(path, store, key_name):
    tbl = pa.table({
        key_name: pa.array(store["key"], pa.int32()),
        "i": pa.array(store["i"], pa.int32()),
        "j": pa.array(store["j"], pa.int32()),
        "v": pa.array(store["v"], pa.float64()),
    })
    pq.write_table(tbl, path)


def new_store():
    return {"key": [], "i": [], "j": [], "v": []}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # ------------------------------------------------------------------ solves
    specs = []                                   # (kind, n, cond)
    for rep in range(5):
        for n in [2, 3, 4, 8, 16, 32, 64]:
            specs.append(("well", n, 10.0 ** rng.uniform(0.5, 3.0)))
    specs[0] = ("well", 1, 10.0)                 # cover the n = 1 edge case
    for n in [128, 196, 256]:
        specs.append(("well", n, 10.0 ** rng.uniform(1.0, 2.0)))
    ill_sizes = [4, 8, 16, 32, 64, 8, 16, 32, 64, 16, 32, 64]
    for n, cond in zip(ill_sizes, np.logspace(6.0, 9.0, len(ill_sizes))):
        specs.append(("ill", n, float(cond)))
    assert len(specs) == 50

    a_store, b_store = new_store(), new_store()
    xnp_store, xgj_store = new_store(), new_store()
    meta = {"sys_id": [], "kind": [], "n": [], "nrhs": [], "cond": [],
            "pivmin_gj": [], "ok_gj": [], "has_x": []}

    def emit_system(sid, kind, A, B, cond):
        n, nrhs = A.shape[0], B.shape[1]
        X_gj, pivmin, ok = gauss_jordan_pp(A, B)
        singular = kind == "singular"
        long_rows(a_store, "sys_id", sid, A)
        long_rows(b_store, "sys_id", sid, B)
        if not singular:
            X_np = np.linalg.solve(A, B)
            long_rows(xnp_store, "sys_id", sid, X_np)
            long_rows(xgj_store, "sys_id", sid, X_gj)
            d = np.max(np.abs(X_np - X_gj)) / np.max(np.abs(X_np))
            band = max(1e-10, 4.0 * np.finfo(np.float64).eps * cond)
            assert d <= band, f"sys {sid} ({kind}, n={n}, cond={cond:.1e}): gj-vs-np {d:.2e} > band {band:.2e}"
            print(f"  solve sys {sid:2d} kind={kind:8s} n={n:3d} nrhs={nrhs} "
                  f"cond={cond:9.2e} gj-vs-np={d:9.2e} pivmin={pivmin:.2e}")
        else:
            print(f"  solve sys {sid:2d} kind={kind:8s} n={n:3d} nrhs={nrhs} "
                  f"pivmin={pivmin:.2e} ok={ok}")
            assert not ok
        meta["sys_id"].append(sid)
        meta["kind"].append(kind)
        meta["n"].append(n)
        meta["nrhs"].append(nrhs)
        meta["cond"].append(float(cond))
        meta["pivmin_gj"].append(pivmin)
        meta["ok_gj"].append(ok)
        meta["has_x"].append(not singular)

    sid = 0
    for kind, n, cond in specs:
        nrhs = [1, 2, 3][sid % 3]
        A = controlled_matrix(rng, n, cond)
        B = rng.standard_normal((n, nrhs))
        emit_system(sid, kind, A, B, cond)
        sid += 1

    # pivot-proof system: fine condition number (~2.6), but elimination
    # WITHOUT row pivoting hits the 1e-18 leading pivot and loses ~16 digits.
    A = np.array([[1e-18, 1.0], [1.0, 1.0]])
    B = np.array([[1.0], [2.0]])
    emit_system(sid, "pivot", A, B, float(np.linalg.cond(A)))
    sid += 1

    # exactly singular system: integer entries, row 3 = row 1 + row 2 exactly
    # (small-integer addition is exact in float64, so the rank deficiency is
    # exact, not approximate).
    A = rng.integers(-8, 9, size=(4, 4)).astype(np.float64)
    A[2, :] = A[0, :] + A[1, :]
    B = rng.integers(-8, 9, size=(4, 1)).astype(np.float64)
    emit_system(sid, "singular", A, B, float("inf"))
    sid += 1

    write_long(OUT / "solve_a.parquet", a_store, "sys_id")
    write_long(OUT / "solve_b.parquet", b_store, "sys_id")
    write_long(OUT / "solve_x_np.parquet", xnp_store, "sys_id")
    write_long(OUT / "solve_x_gj.parquet", xgj_store, "sys_id")
    pq.write_table(pa.table({
        "sys_id": pa.array(meta["sys_id"], pa.int32()),
        "kind": pa.array(meta["kind"], pa.string()),
        "n": pa.array(meta["n"], pa.int32()),
        "nrhs": pa.array(meta["nrhs"], pa.int32()),
        "cond": pa.array(meta["cond"], pa.float64()),
        "pivmin_gj": pa.array(meta["pivmin_gj"], pa.float64()),
        "ok_gj": pa.array(meta["ok_gj"], pa.bool_()),
        "has_x": pa.array(meta["has_x"], pa.bool_()),
    }), OUT / "solve_meta.parquet")

    # -------------------------------------------------- multiply / basic ops
    mm_specs = [(1, 1, 1), (2, 3, 4), (5, 5, 5), (8, 3, 7),
                (16, 16, 16), (30, 30, 30), (32, 40, 24), (64, 64, 64)]
    stores = {name: new_store() for name in
              ["mm_a", "mm_b", "mm_c", "mm_d", "mm_at", "mm_apd", "mm_scaled"]}
    vec_store = {"case_id": [], "idx": [], "v": []}
    mm_meta = {"case_id": [], "m": [], "k": [], "n": [], "c": []}
    for cid, (m, k, n) in enumerate(mm_specs):
        A = rng.standard_normal((m, k))
        B = rng.standard_normal((k, n))
        D = rng.standard_normal((m, k))
        c = float(rng.standard_normal())
        C = A @ B
        long_rows(stores["mm_a"], "case_id", cid, A)
        long_rows(stores["mm_b"], "case_id", cid, B)
        long_rows(stores["mm_c"], "case_id", cid, C)
        long_rows(stores["mm_d"], "case_id", cid, D)
        long_rows(stores["mm_at"], "case_id", cid, A.T)
        long_rows(stores["mm_apd"], "case_id", cid, A + D)
        long_rows(stores["mm_scaled"], "case_id", cid, A * c)
        vecA = A.flatten(order="F")
        vec_store["case_id"].extend([cid] * vecA.size)
        vec_store["idx"].extend(range(1, vecA.size + 1))
        vec_store["v"].extend(vecA.tolist())
        mm_meta["case_id"].append(cid)
        mm_meta["m"].append(m)
        mm_meta["k"].append(k)
        mm_meta["n"].append(n)
        mm_meta["c"].append(c)
        # confirm the ordered-fold inner sum stays inside the 1e-13 test band
        C_fold = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                s = np.float64(0.0)
                for kk in range(k):
                    s = s + A[i, kk] * B[kk, j]
                C_fold[i, j] = s
        d = np.max(np.abs(C_fold - C)) / np.max(np.abs(C))
        print(f"  mm case {cid} ({m}x{k})({k}x{n}): fold-vs-numpy {d:.2e}")
        assert d <= 1e-13
    for name, st in stores.items():
        write_long(OUT / f"{name}.parquet", st, "case_id")
    pq.write_table(pa.table({
        "case_id": pa.array(mm_meta["case_id"], pa.int32()),
        "m": pa.array(mm_meta["m"], pa.int32()),
        "k": pa.array(mm_meta["k"], pa.int32()),
        "n": pa.array(mm_meta["n"], pa.int32()),
        "c": pa.array(mm_meta["c"], pa.float64()),
    }), OUT / "mm_meta.parquet")
    pq.write_table(pa.table({
        "case_id": pa.array(vec_store["case_id"], pa.int32()),
        "idx": pa.array(vec_store["idx"], pa.int32()),
        "v": pa.array(vec_store["v"], pa.float64()),
    }), OUT / "mm_vec.parquet")

    # -------------------------------------------------------------- Kronecker
    kron_specs = [(1, 1, 1, 1), (2, 3, 3, 2), (4, 4, 3, 3), (3, 2, 2, 5),
                  (8, 8, 6, 7), (30, 30, 30, 30)]
    ka, kb, kc = new_store(), new_store(), new_store()
    kron_meta = {"case_id": [], "ma": [], "na": [], "mb": [], "nb": [], "has_result": []}
    for cid, (ma, na, mb, nb) in enumerate(kron_specs):
        A = rng.standard_normal((ma, na))
        B = rng.standard_normal((mb, nb))
        has_result = ma * na * mb * nb <= 10000   # 30x30 pair: plan test only
        long_rows(ka, "case_id", cid, A)
        long_rows(kb, "case_id", cid, B)
        if has_result:
            long_rows(kc, "case_id", cid, np.kron(A, B))
        kron_meta["case_id"].append(cid)
        kron_meta["ma"].append(ma)
        kron_meta["na"].append(na)
        kron_meta["mb"].append(mb)
        kron_meta["nb"].append(nb)
        kron_meta["has_result"].append(has_result)
    write_long(OUT / "kron_a.parquet", ka, "case_id")
    write_long(OUT / "kron_b.parquet", kb, "case_id")
    write_long(OUT / "kron_c.parquet", kc, "case_id")
    pq.write_table(pa.table({
        "case_id": pa.array(kron_meta["case_id"], pa.int32()),
        "ma": pa.array(kron_meta["ma"], pa.int32()),
        "na": pa.array(kron_meta["na"], pa.int32()),
        "mb": pa.array(kron_meta["mb"], pa.int32()),
        "nb": pa.array(kron_meta["nb"], pa.int32()),
        "has_result": pa.array(kron_meta["has_result"], pa.bool_()),
    }), OUT / "kron_meta.parquet")

    # --------------------------------------------------------------- Lyapunov
    ks = [1, 2, 3, 4, 8, 8, 14, 14, 16, 16]
    lt, lq, lp = new_store(), new_store(), new_store()
    lyap_meta = {"case_id": [], "k": [], "rho": []}
    for cid, k in enumerate(ks):
        M = rng.standard_normal((k, k))
        sr = max(np.abs(np.linalg.eigvals(M))) if k > 1 else abs(M[0, 0])
        rho = float(rng.uniform(0.3, 0.9))
        T = M * (rho / sr)
        W = rng.standard_normal((k, k))
        RQR = W @ W.T
        # reference: exactly the algorithm the SQL implements (vec trick,
        # column-stacking vec), solved by numpy -- no scipy involved.
        vecP = np.linalg.solve(np.eye(k * k) - np.kron(T, T), RQR.flatten(order="F"))
        P = vecP.reshape((k, k), order="F")
        resid = np.max(np.abs(P - T @ P @ T.T - RQR))
        print(f"  lyap case {cid} k={k:2d} rho={rho:.3f} "
              f"resid={resid:.2e} maxP={np.max(np.abs(P)):.2e}")
        assert resid <= 1e-9 * np.max(np.abs(P))
        long_rows(lt, "case_id", cid, T)
        long_rows(lq, "case_id", cid, RQR)
        long_rows(lp, "case_id", cid, P)
        lyap_meta["case_id"].append(cid)
        lyap_meta["k"].append(k)
        lyap_meta["rho"].append(rho)
    write_long(OUT / "lyap_t.parquet", lt, "case_id")
    write_long(OUT / "lyap_rqr.parquet", lq, "case_id")
    write_long(OUT / "lyap_p.parquet", lp, "case_id")
    pq.write_table(pa.table({
        "case_id": pa.array(lyap_meta["case_id"], pa.int32()),
        "k": pa.array(lyap_meta["k"], pa.int32()),
        "rho": pa.array(lyap_meta["rho"], pa.float64()),
    }), OUT / "lyap_meta.parquet")

    print(f"fixtures written to {OUT}")


if __name__ == "__main__":
    main()
