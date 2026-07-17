-- ============================================================================
-- duckARIMA Layer 0: relational linear algebra as pure DuckDB (>= 1.5.4) SQL
-- macros. No extensions, no UDFs. Spec: sarima-duckdb-sql-spec.md sections
-- 4.1, 4.2, 4.3, 5.0, 6.
--
-- Two matrix encodings, with exact round-trip conversions between them:
--
--   A. Flattened-list encoding (the hot path, used inside the Kalman filter
--      recursion of Layer 3): an m-by-n matrix is a row-major flattened
--      DOUBLE[] with dims passed alongside; element (i, j), 1-based, lives at
--      list index (i-1)*n + j. Vectors are m-by-1 (or 1-by-n) matrices.
--
--   B. Relational encoding: a table (i INT, j INT, v DOUBLE), 1-based, DENSE
--      (structural zeros stored -- the dims are recovered as max(i), max(j),
--      so a missing zero row would silently truncate the matrix). A leading
--      name VARCHAR column, per spec 4.1, is permitted and ignored: macros
--      only touch i, j, v, and every relational macro's input table must
--      contain exactly one matrix. Table macros take TABLE NAMES AS STRINGS,
--      resolved via query_table() (duckLM's regression_macros.sql pattern),
--      so they work on any table or view visible in the connection.
--
-- Determinism (spec 4.2): every summation here has a defined order. Inner
-- products fold left-to-right over an explicitly ordered list --
-- list_reduce(list(x ORDER BY inner_index), lambda za, zb: za + zb) in the
-- relational multiply, and list_reduce over a list built by ascending
-- range() in the flattened multiply. list_sum / SUM() over parallel scans
-- appear nowhere on these paths. Elimination pivoting breaks |v| ties by
-- the SMALLEST row index (matching NumPy argmax's first-max), so the whole
-- solve is bit-reproducible at any thread count.
--
-- Lambdas: Python-style ('lambda zx: ...') exclusively; the test harness runs
-- with SET lambda_syntax = 'DISABLE_SINGLE_ARROW', and the arrow form is
-- removed in DuckDB 2.0. Every lambda variable in this file starts with 'z';
-- that prefix is reserved -- callers must not use z-prefixed lambda variables
-- or column references inside expressions passed as macro arguments, or the
-- macro's own lambda variables would capture them (macro expansion is
-- textual). Internal CTE names are prefixed _sarimax_ for the same reason:
-- query_table() resolves table-name strings against CTEs in scope, so a
-- generic CTE name would shadow a user table of that name (duckLM documents
-- this trap).
--
-- Literal discipline (spec 6): bare decimals like 1.0 parse as DECIMAL in
-- DuckDB, and DECIMAL leaking into a fold anchor or list constructor fixes
-- the wrong type for every later iteration. All numeric literals feeding
-- arithmetic are written in scientific notation (1e0, 0e0, 5e-1, 1e-12,
-- 1e-300, 1e308), which parses as DOUBLE.
--
-- Composition caveat (performance, not correctness): macro expansion is
-- textual, so passing a list-VALUED macro call as an argument that the
-- receiving macro references INSIDE a lambda re-evaluates the whole inner
-- expression per element (there is no guaranteed hoisting of loop-invariant
-- subexpressions out of lambda bodies). Compose by binding intermediates
-- once via the single-element list_transform trick used in _sarimax_lyap
-- below -- (list_transform([expr], lambda zx: ...use zx...))[1] -- or pass
-- materialized columns. Arguments that are only indexed OUTSIDE a lambda
-- (e.g. _sarimax_solve_list's aaug, which appears only in the fold seed) are
-- evaluated once and are safe to compose freely.
-- ============================================================================


-- ----------------------------------------------------------------------------
-- A. Flattened-list encoding (scalar macros)
-- ----------------------------------------------------------------------------

-- Element (i, j) of the row-major flattened m-by-n matrix a (m is not needed
-- for the address, only n). 1-based everywhere.
CREATE OR REPLACE MACRO _sarimax_mget(a, i, j, n) AS (
  a[(i - 1) * n + j]
);

-- n-by-n identity, flattened. Diagonal when row = col, i.e. when
-- (idx-1) // n = (idx-1) % n.
CREATE OR REPLACE MACRO _sarimax_meye(n) AS (
  list_transform(range(1, n * n + 1), lambda zidx:
    CASE WHEN (zidx - 1) // n = (zidx - 1) % n THEN 1e0 ELSE 0e0 END)
);

-- m-by-n zero matrix, flattened.
CREATE OR REPLACE MACRO _sarimax_mzeros(m, n) AS (
  list_transform(range(1, m * n + 1), lambda zidx: 0e0)
);

-- Elementwise a + b (same shape; dims not needed). The two-argument
-- list_transform lambda receives (element, 1-based index).
CREATE OR REPLACE MACRO _sarimax_madd(a, b) AS (
  list_transform(a, lambda zv, zi: zv + b[zi])
);

-- Elementwise a - b.
CREATE OR REPLACE MACRO _sarimax_msub(a, b) AS (
  list_transform(a, lambda zv, zi: zv - b[zi])
);

-- Scalar multiple c * a.
CREATE OR REPLACE MACRO _sarimax_mscale(a, c) AS (
  list_transform(a, lambda zv: zv * c)
);

-- Transpose of the m-by-n matrix a: an n-by-m flattened matrix. Output
-- element (i2, j2) = a(j2, i2); with i2 = (idx-1)//m + 1, j2 = (idx-1)%m + 1,
-- the source address is (j2-1)*n + i2.
CREATE OR REPLACE MACRO _sarimax_mtrans(a, m, n) AS (
  list_transform(range(1, n * m + 1), lambda zidx:
    a[((zidx - 1) % m) * n + ((zidx - 1) // m + 1)])
);

-- Symmetrize the k-by-k matrix p as (p + p') / 2 (spec 5.3 uses this each
-- filter step to stop covariance asymmetry drift). The transpose partner of
-- flat index idx is (j-1)*k + i = ((idx-1)%k)*k + (idx-1)//k + 1.
-- Multiplying by 5e-1 is exact (power of two), identical to dividing by 2.
CREATE OR REPLACE MACRO _sarimax_msym(p, k) AS (
  list_transform(range(1, k * k + 1), lambda zidx:
    (p[zidx] + p[((zidx - 1) % k) * k + ((zidx - 1) // k + 1)]) * 5e-1)
);

-- Matrix product of a (m-by-k) and b (k-by-n), flattened row-major on both
-- sides. The inner sum over kk = 1..k runs ASCENDING: the terms are built by
-- list_transform over range(1, k+1) (which generates in order) and folded
-- left-to-right by list_reduce -- the spec-4.2 deterministic-summation form.
-- Output element (i, j): i-1 = (idx-1)//n, j = (idx-1)%n + 1;
-- a(i, kk) = a[(i-1)*k + kk], b(kk, j) = b[(kk-1)*n + j].
CREATE OR REPLACE MACRO _sarimax_mmul(a, b, m, k, n) AS (
  list_transform(range(1, m * n + 1), lambda zidx:
    list_reduce(
      list_transform(range(1, k + 1), lambda zkk:
        a[((zidx - 1) // n) * k + zkk] * b[(zkk - 1) * n + ((zidx - 1) % n + 1)]),
      lambda zx, zy: zx + zy))
);

-- Kronecker product of a (ma-by-na) and b (mb-by-nb): an (ma*mb)-by-(na*nb)
-- flattened matrix with K[(ia-1)*mb+ib, (ja-1)*nb+jb] = A[ia,ja]*B[ib,jb].
-- With W = na*nb, i-1 = (idx-1)//W and j-1 = (idx-1)%W:
--   A element: a[((i-1)//mb)*na + ((j-1)//nb + 1)]
--   B element: b[((i-1)%mb)*nb + ((j-1)%nb + 1)]
-- One float multiply per element -- bitwise identical to numpy.kron.
CREATE OR REPLACE MACRO _sarimax_mkron(a, b, ma, na, mb, nb) AS (
  list_transform(range(1, ma * mb * na * nb + 1), lambda zidx:
    a[(((zidx - 1) // (na * nb)) // mb) * na + (((zidx - 1) % (na * nb)) // nb + 1)]
    * b[(((zidx - 1) // (na * nb)) % mb) * nb + (((zidx - 1) % (na * nb)) % nb + 1)])
);

-- Classic COLUMN-STACKING vec of the row-major flattened m-by-n matrix a:
-- output index idx holds A(i, j) with j = (idx-1)//m + 1, i = (idx-1)%m + 1
-- (columns first), read from row-major address (i-1)*n + j. This is exactly
-- the row-major flattening of A', so vec(A) == mtrans(A) elementwise -- kept
-- as its own macro because the vec/Kronecker orientation of the Lyapunov
-- solve below depends on it and the name documents the intent.
CREATE OR REPLACE MACRO _sarimax_mvec(a, m, n) AS (
  list_transform(range(1, m * n + 1), lambda zidx:
    a[((zidx - 1) % m) * n + ((zidx - 1) // m + 1)])
);

-- Inverse of _sarimax_mvec: column-stacked vector v of length m*n back to the
-- row-major flattened m-by-n matrix. Output (i, j) at idx reads v[(j-1)*m+i].
CREATE OR REPLACE MACRO _sarimax_munvec(v, m, n) AS (
  list_transform(range(1, m * n + 1), lambda zidx:
    v[((zidx - 1) % n) * m + ((zidx - 1) // n + 1)])
);

-- [a | b]: augment the n-by-n matrix a with the n-by-nrhs matrix b into the
-- n-by-(n+nrhs) row-major flattened matrix _sarimax_solve_list expects.
-- NOTE (composition caveat above): a and b are indexed inside the lambda, so
-- pass materialized lists or bound values, not nested macro expressions.
CREATE OR REPLACE MACRO _sarimax_maug(a, b, n, nrhs) AS (
  list_transform(range(1, n * (n + nrhs) + 1), lambda zidx:
    CASE WHEN (zidx - 1) % (n + nrhs) + 1 <= n
         THEN a[((zidx - 1) // (n + nrhs)) * n + ((zidx - 1) % (n + nrhs) + 1)]
         ELSE b[((zidx - 1) // (n + nrhs)) * nrhs + ((zidx - 1) % (n + nrhs) + 1 - n)]
    END)
);

-- Gauss-Jordan elimination WITH PARTIAL PIVOTING on an augmented
-- n-by-(n+nrhs) row-major flattened matrix aaug = [A | B], as a single scalar
-- expression: a list_reduce fold over pivot index k = 1..n (duckLM's
-- __reg_matinv is the precedent for the fold shape; this adds the pivot-row
-- selection and swap that __reg_matinv, built for SPD matrices, omits).
-- A fold rather than a recursive CTE so it can run per-row inside other
-- queries -- e.g. batched probe evaluation in Layer 4. Handles n up to 256
-- (the vec-trick Lyapunov solve at state dimension k = 16).
--
-- Per step k:
--   1. pivot row  = argmax |M(r, k)| over r = k..n, first row on ties
--      (bitwise-matching numpy argmax semantics);
--   2. rows k and pivot row swap (logically, via an index remap);
--   3. the pivot row is normalized: M(k, :) <- M(k, :) / piv;
--   4. every other row r eliminates: M(r, :) <- M(r, :) - M(r, k) * Mnorm(k, :).
--      The grouping v - colk * (prow_v / piv) makes column k of every
--      non-pivot row EXACTLY zero (piv/piv == 1e0 exactly), and matches the
--      fixture generator's float64 control implementation operation for
--      operation, so results are bit-identical to it.
-- If the selected pivot magnitude is below 1e-300 the step is skipped (no
-- swap, matrix carried unchanged) instead of dividing by ~0; pivmin then
-- records ~0 and ok comes back false. list_reduce cannot take a struct
-- initial value over a BIGINT list, so the accumulator rides as the first
-- element of the folded struct list (duckLM's seed trick).
--
-- Returns STRUCT(x DOUBLE[]      -- n*nrhs row-major flattened solution
--                                --  (garbage when NOT ok),
--                pivmin DOUBLE,  -- smallest |pivot| encountered,
--                ok BOOLEAN      -- pivmin > 1e-12 (spec 5.0 threshold)).
CREATE OR REPLACE MACRO _sarimax_solve_list(aaug, n, nrhs) AS (
  (list_transform([
    list_reduce(
      [struct_pack(zkk := 0::BIGINT, mm := aaug, pm := 1e308)]
        || list_transform(range(1, n + 1), lambda zsi:
             struct_pack(zkk := zsi, mm := []::DOUBLE[], pm := 0e0)),
      lambda zacc, zstep:
        -- bind the selected pivot (magnitude, negated row, value) once
        (list_transform([
           list_reduce(
             list_transform(range(zstep.zkk, n + 1), lambda zr:
               struct_pack(zav := abs(zacc.mm[(zr - 1) * (n + nrhs) + zstep.zkk]),
                           zni := -zr,
                           zval := zacc.mm[(zr - 1) * (n + nrhs) + zstep.zkk])),
             lambda zpa, zpb: CASE WHEN zpb.zav > zpa.zav THEN zpb ELSE zpa END)
         ], lambda zp:
           -- bind the normalized pivot row (post-swap row k) once
           (list_transform([
              CASE WHEN zp.zav < 1e-300 THEN []::DOUBLE[]
                   ELSE list_transform(range(1, n + nrhs + 1), lambda zc:
                          zacc.mm[(-zp.zni - 1) * (n + nrhs) + zc] / zp.zval)
              END
            ], lambda znorm:
              struct_pack(
                zkk := zstep.zkk,
                mm := CASE WHEN zp.zav < 1e-300 THEN zacc.mm
                      ELSE list_transform(range(1, n * (n + nrhs) + 1), lambda zidx:
                        CASE WHEN (zidx - 1) // (n + nrhs) + 1 = zstep.zkk
                          -- (post-swap) pivot row: the normalized row
                          THEN znorm[(zidx - 1) % (n + nrhs) + 1]
                          -- other rows, reading through the swap remap:
                          -- source row = k when this is the old pivot row's
                          -- new home, else the row itself
                          ELSE zacc.mm[((CASE WHEN (zidx - 1) // (n + nrhs) + 1 = -zp.zni
                                              THEN zstep.zkk
                                              ELSE (zidx - 1) // (n + nrhs) + 1 END) - 1) * (n + nrhs)
                                       + ((zidx - 1) % (n + nrhs) + 1)]
                               - zacc.mm[((CASE WHEN (zidx - 1) // (n + nrhs) + 1 = -zp.zni
                                                THEN zstep.zkk
                                                ELSE (zidx - 1) // (n + nrhs) + 1 END) - 1) * (n + nrhs)
                                         + zstep.zkk]
                                 * znorm[(zidx - 1) % (n + nrhs) + 1]
                        END)
                      END,
                pm := least(zacc.pm, zp.zav))))[1]
        ))[1]
    )
  ], lambda zf:
    struct_pack(
      x := list_transform(range(1, n * nrhs + 1), lambda zxi:
             zf.mm[((zxi - 1) // nrhs) * (n + nrhs) + n + ((zxi - 1) % nrhs + 1)]),
      pivmin := zf.pm,
      ok := zf.pm > 1e-12)))[1]
);

-- Inverse of the n-by-n flattened matrix a, via the same elimination against
-- an identity augmentation [a | I]. The augmented matrix is built directly in
-- one pass (composing _sarimax_maug(_sarimax_meye(...)) here would
-- re-evaluate the identity constructor per element -- composition caveat).
-- Same STRUCT(x, pivmin, ok) return shape; x is the flattened n-by-n inverse.
CREATE OR REPLACE MACRO _sarimax_inv_list(a, n) AS (
  _sarimax_solve_list(
    list_transform(range(1, n * 2 * n + 1), lambda zidx:
      CASE WHEN (zidx - 1) % (2 * n) + 1 <= n
           THEN a[((zidx - 1) // (2 * n)) * n + ((zidx - 1) % (2 * n) + 1)]
           ELSE CASE WHEN (zidx - 1) % (2 * n) + 1 - n = (zidx - 1) // (2 * n) + 1
                     THEN 1e0 ELSE 0e0 END
      END),
    n, n)
);

-- Discrete Lyapunov solver: P = T P T' + RQR' for the k-by-k transition T
-- and k-by-k RQR' (both row-major flattened), via the vec trick (spec 5.3):
--   vec(P) = (I_{k^2} - T (x) T)^{-1} vec(RQR')
-- with COLUMN-STACKING vec (for which vec(A B C) = (C' (x) A) vec(B), so
-- vec(T P T') = (T (x) T) vec(P)) and _sarimax_mkron's row-major Kronecker
-- layout. Returns the row-major flattened k-by-k P (not symmetrized; the
-- caller may _sarimax_msym it). Intermediates are bound exactly once via the
-- single-element list_transform trick -- see the composition caveat: naive
-- nesting would re-evaluate the k^2-by-k^2 Kronecker product per element.
-- The k^2-by-(k^2+1) augmented system [I - T(x)T | vec(RQR')] is built in one
-- pass: row r (0-based) = (idx-1)//(k^2+1), col c (1-based) = (idx-1)%(k^2+1)+1;
-- left block gets I(r+1, c) - kron[r*k^2 + c], right column gets vec(RQR')[r+1].
-- Validation (residual substitution, spec 6) lives in the acceptance tests.
CREATE OR REPLACE MACRO _sarimax_lyap(t, r_qr, k) AS (
  (list_transform([
     struct_pack(zkr := _sarimax_mkron(t, t, k, k, k, k),
                 zvr := _sarimax_mvec(r_qr, k, k))
   ], lambda zl1:
     (list_transform([
        _sarimax_solve_list(
          list_transform(range(1, (k * k) * (k * k + 1) + 1), lambda zidx:
            CASE WHEN (zidx - 1) % (k * k + 1) + 1 <= k * k
                 THEN (CASE WHEN (zidx - 1) // (k * k + 1) = (zidx - 1) % (k * k + 1)
                            THEN 1e0 ELSE 0e0 END)
                      - zl1.zkr[((zidx - 1) // (k * k + 1)) * (k * k)
                                + ((zidx - 1) % (k * k + 1) + 1)]
                 ELSE zl1.zvr[(zidx - 1) // (k * k + 1) + 1]
            END),
          k * k, 1)
      ], lambda zl2:
        _sarimax_munvec(zl2.x, k, k)))[1]
  ))[1]
);


-- ----------------------------------------------------------------------------
-- B. Relational encoding (table macros; table names as strings)
-- ----------------------------------------------------------------------------

-- n-by-n identity as a dense (i, j, v) table.
CREATE OR REPLACE MACRO _sarimax_reye(n) AS TABLE
SELECT zr.zi::INT AS i, zc.zj::INT AS j,
       CASE WHEN zr.zi = zc.zj THEN 1e0 ELSE 0e0 END AS v
FROM (SELECT unnest(range(1, n + 1)) AS zi) zr,
     (SELECT unnest(range(1, n + 1)) AS zj) zc;

-- m-by-n zero matrix as a dense (i, j, v) table.
CREATE OR REPLACE MACRO _sarimax_rzeros(m, n) AS TABLE
SELECT zr.zi::INT AS i, zc.zj::INT AS j, 0e0 AS v
FROM (SELECT unnest(range(1, m + 1)) AS zi) zr,
     (SELECT unnest(range(1, n + 1)) AS zj) zc;

-- Transpose: swap the indices.
CREATE OR REPLACE MACRO _sarimax_rtrans(a_tbl) AS TABLE
SELECT j::INT AS i, i::INT AS j, v::DOUBLE AS v FROM query_table(a_tbl);

-- Scalar multiple c * A.
CREATE OR REPLACE MACRO _sarimax_rscale(a_tbl, c) AS TABLE
SELECT i::INT AS i, j::INT AS j, (v * c)::DOUBLE AS v FROM query_table(a_tbl);

-- Elementwise A + B (same shape, both dense). Equi-join on both indices.
CREATE OR REPLACE MACRO _sarimax_radd(a_tbl, b_tbl) AS TABLE
SELECT za.i::INT AS i, za.j::INT AS j, (za.v + zb.v)::DOUBLE AS v
FROM query_table(a_tbl) za
JOIN query_table(b_tbl) zb ON za.i = zb.i AND za.j = zb.j;

-- Matrix product A B: equi-join on the inner index (hash join -- asserted by
-- the plan-shape regression test per spec 4.3), GROUP BY the outer indices,
-- and the spec-4.2 ordered summation: list(term ORDER BY inner index ASC)
-- folded left-to-right. The inner index is unique within a group, so the
-- ordered list -- and therefore the fold -- is fully deterministic at any
-- thread count.
CREATE OR REPLACE MACRO _sarimax_rmul(a_tbl, b_tbl) AS TABLE
SELECT za.i::INT AS i, zb.j::INT AS j,
       list_reduce(list(za.v * zb.v ORDER BY za.j ASC), lambda zx, zy: zx + zy)::DOUBLE AS v
FROM query_table(a_tbl) za
JOIN query_table(b_tbl) zb ON za.j = zb.i
GROUP BY za.i, zb.j;

-- Kronecker product A (x) B as index arithmetic. Rather than a bare cross
-- join of A and B (which the planner can only run as CROSS_PRODUCT), the full
-- output index grid is generated first and then equi-joined to A and to B on
-- COMPUTED keys -- (gi-1)//mb+1 etc. are pure functions of the grid side, so
-- both joins plan as HASH_JOINs (spec 4.3; asserted in the tests). The grid
-- itself is the unavoidable |A|x|B| cross product of two index ranges.
-- Dims come from max(i)/max(j) -- correct because the encoding is dense.
CREATE OR REPLACE MACRO _sarimax_rkron(a_tbl, b_tbl) AS TABLE
WITH _sarimax_rk_dims AS (
  SELECT za.ma, za.na, zb.mb, zb.nb
  FROM (SELECT max(i)::BIGINT AS ma, max(j)::BIGINT AS na FROM query_table(a_tbl)) za,
       (SELECT max(i)::BIGINT AS mb, max(j)::BIGINT AS nb FROM query_table(b_tbl)) zb
),
_sarimax_rk_gi AS (
  SELECT unnest(range(1, zd.ma * zd.mb + 1)) AS gi, zd.mb, zd.nb
  FROM _sarimax_rk_dims zd
),
_sarimax_rk_gj AS (
  SELECT unnest(range(1, zd.na * zd.nb + 1)) AS gj FROM _sarimax_rk_dims zd
),
_sarimax_rk_grid AS (
  SELECT zgi.gi, zgj.gj, zgi.mb, zgi.nb FROM _sarimax_rk_gi zgi, _sarimax_rk_gj zgj
)
SELECT zg.gi::INT AS i, zg.gj::INT AS j, (za.v * zb.v)::DOUBLE AS v
FROM _sarimax_rk_grid zg
JOIN query_table(a_tbl) za
  ON za.i = (zg.gi - 1) // zg.mb + 1 AND za.j = (zg.gj - 1) // zg.nb + 1
JOIN query_table(b_tbl) zb
  ON zb.i = (zg.gi - 1) % zg.mb + 1 AND zb.j = (zg.gj - 1) % zg.nb + 1;

-- Relational -> flattened-list conversion: one row (v DOUBLE[], m INT, n INT)
-- with v row-major (ORDER BY i, j is exactly row-major order; keys are unique
-- so the ordered list aggregate is deterministic). Exact: pure reindexing.
CREATE OR REPLACE MACRO _sarimax_mat_to_list(tbl) AS TABLE
SELECT list(v::DOUBLE ORDER BY i ASC, j ASC) AS v,
       max(i)::INT AS m, max(j)::INT AS n
FROM query_table(tbl);

-- Flattened-list -> relational conversion: table (i, j, v) from the row-major
-- flattened m-by-n list. Exact inverse of _sarimax_mat_to_list.
CREATE OR REPLACE MACRO _sarimax_list_to_mat(lst, m, n) AS TABLE
SELECT ((zidx - 1) // n + 1)::INT AS i,
       ((zidx - 1) % n + 1)::INT AS j,
       (lst)[zidx]::DOUBLE AS v
FROM (SELECT unnest(range(1, m * n + 1)) AS zidx);

-- Relational Gauss-Jordan solve of A X = B with partial pivoting, as a
-- RECURSIVE CTE over pivot column k (the spec-5.0-pinned form for this
-- encoding), using USING KEY (i, j) with UNION ALL: each iteration rewrites
-- every cell of the augmented matrix in place of the previous iteration's
-- row, so the working set stays at one matrix rather than accumulating n of
-- them. Handles 256-by-256. Arithmetic, pivot choice (max |v| at-or-below k,
-- first row on ties via lexicographic struct max on (|v|, -i)), swap, and the
-- skip-tiny-pivot guard are operation-for-operation identical to
-- _sarimax_solve_list, so the two encodings produce bit-identical solutions.
--
-- The recursive member may reference the CTE only through one scan, so the
-- per-step cross-row lookups are window functions over that single scan:
--   zp    (OVER ())               = the pivot (|v|, -row, value) struct;
--   zcolk (PARTITION BY i)        = v(i, k), each row's multiplier;
--   zprv  (PARTITION BY j)        = v(prow, j), the (post-swap) pivot row.
-- max()/min() windows are exact (no floating-point accumulation), so thread
-- count cannot perturb the result.
--
-- B's columns ride along as augmented columns n+1 .. n+nrhs. Returns the
-- solution block relabeled to (i, j, v), each row also carrying
-- (pivmin DOUBLE, ok BOOLEAN) -- constant across rows -- so a failed
-- elimination surfaces in-band: ok = false (and x is garbage), never an
-- error, matching the list solver's contract.
CREATE OR REPLACE MACRO _sarimax_rsolve(a_tbl, b_tbl) AS TABLE
WITH RECURSIVE
_sarimax_rs_n AS (SELECT max(i)::INT AS n FROM query_table(a_tbl)),
_sarimax_rs_aug AS (
  SELECT i::INT AS i, j::INT AS j, v::DOUBLE AS v FROM query_table(a_tbl)
  UNION ALL
  SELECT zb.i::INT, (zb.j + zd.n)::INT, zb.v::DOUBLE
  FROM query_table(b_tbl) zb, _sarimax_rs_n zd
),
_sarimax_rs_elim(i, j, v, k, pivmin) USING KEY (i, j) AS (
  SELECT i, j, v, 0::INT AS k, 1e308::DOUBLE AS pivmin FROM _sarimax_rs_aug
  UNION ALL
  SELECT CASE WHEN z3.zav < 1e-300 THEN z3.i            -- tiny pivot: no swap,
              WHEN z3.i = z3.zprow THEN z3.zk           -- carry unchanged
              WHEN z3.i = z3.zk THEN z3.zprow
              ELSE z3.i END AS i,
         z3.j,
         CASE WHEN z3.zav < 1e-300 THEN z3.v
              WHEN z3.i = z3.zprow THEN z3.v / z3.zpv   -- post-swap pivot row
              ELSE z3.v - z3.zcolk * (z3.zprv / z3.zpv) -- eliminate
         END AS v,
         z3.zk AS k,
         least(z3.pivmin, z3.zav) AS pivmin
  FROM (
    SELECT z2.*,
           max(CASE WHEN z2.j = z2.zk THEN z2.v END) OVER (PARTITION BY z2.i) AS zcolk,
           max(CASE WHEN z2.i = z2.zprow THEN z2.v END) OVER (PARTITION BY z2.j) AS zprv
    FROM (
      SELECT z1.*,
             (-((z1.zp).zni))::INT AS zprow,
             (z1.zp).zav AS zav,
             (z1.zp).zval AS zpv
      FROM (
        SELECT z0.*, (z0.k + 1)::INT AS zk,
               max(CASE WHEN z0.j = z0.k + 1 AND z0.i >= z0.k + 1
                        THEN struct_pack(zav := abs(z0.v), zni := -z0.i, zval := z0.v)
                   END) OVER () AS zp
        FROM _sarimax_rs_elim z0, _sarimax_rs_n zn
        WHERE z0.k < zn.n
      ) z1
    ) z2
  ) z3
)
SELECT ze.i AS i, (ze.j - zn.n)::INT AS j, ze.v AS v,
       zpm.pivmin AS pivmin, zpm.pivmin > 1e-12 AS ok
FROM _sarimax_rs_elim ze, _sarimax_rs_n zn,
     (SELECT min(pivmin) AS pivmin FROM _sarimax_rs_elim) zpm
WHERE ze.j > zn.n;
