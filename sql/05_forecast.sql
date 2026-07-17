-- ============================================================================
-- duckARIMA Layer 5: forecasting (spec sections 3 T3, 4.2, 5.5, 6).
--
-- Multi-step SARIMAX forecasting in pure DuckDB SQL, from the final PREDICTED
-- Kalman state (a_{n+1}, P_{n+1}) produced by Layer 3's _sarimax_kfilter_state.
--
-- DIFFERENCED SCALE (per probe; a_1 := a_{n+1}, P_1 := P_{n+1}):
--   for h = 1..H:  mean_diff[h] = d_fut[h] + a_h[1];  var_diff[h] = P_h[1,1];
--                  a_{h+1} = T a_h;   P_{h+1} = T P_h T' + RQR'
--   (no symmetrization inside the horizon recursion -- matches the reference
--   construction in tests/generate_fixtures.py exactly).
--   d_fut[h] = x~_{n+h}' beta: the DIFFERENCED future exog dot the beta block
--   (ordered-j fold); identically 0 when r = 0.
--
-- FORECAST-ERROR CROSS-COVARIANCES (needed for the original scale):
--   Omega[i,j] = Cov(w-forecast-error i, w-forecast-error j), with
--   Omega[i,i] = var_diff[i] and, for j >= i,  Omega[j,i] = Omega[i,j]
--     = (Z T^{j-i}) P_i Z',   Z = e_1'.
--   Computed row-wise with gvec_0 = e_1' (row), gvec_m = gvec_{m-1} T:
--     Omega[h,j] = dot(gvec_{h-j}, column 1 of P_j)   for j <= h.
--
-- ORIGINAL SCALE (Layer 1 anchor/weights contract, see sql/01_prep.sql):
--   mean_orig : the per-stage cumulative-sum inversion of
--               _sarimax_undiff_forecast, reimplemented inline with an extra
--               PARTITION BY probe_id (the Layer 1 macro handles ONE series;
--               tests prove exact agreement on the single-probe case).
--   var_orig[h] = sum_{i<=h} sum_{j<=h} c[h,i] c[h,j] Omega[i,j]
--               with c from _sarimax_undiff_weights(d, sd, s, hmax); the
--               double sum is an ordered fold, i ascending then j ascending.
--   se = sqrt(var) on both scales.
--
-- FUTURE-EXOG CONTRACT (spec 5.5): the caller supplies ORIGINAL-scale exog for
-- t = 1..n (in-sample) ++ t = n+1..n+H (future); the differenced future exog
-- is obtained by lockstep-differencing the CONCATENATED original exog
-- (_sarimax_diff_exog on the full-coverage table) and taking rows t > n_eff.
-- Missing future coverage fails loudly with the missing t range named.
-- Future exog are treated as known constants: they shift the point forecast
-- and contribute nothing to the forecast variance (statsmodels' convention).
--
-- MACROS (all internal, `_sarimax_` prefix; interfaces pinned for Layer 6)
--   _sarimax_fc_dfut(exog_diff_full_tbl, probes_tbl, n_eff, hmax) AS TABLE
--       -> (probe_id BIGINT, h BIGINT, d DOUBLE)
--       exog_diff_full_tbl: (t, j, x) lockstep-differenced FULL exog covering
--       t = 1..n_eff+H (output of _sarimax_diff_exog on in-sample ++ future
--       original exog). Takes rows t = n_eff+1..n_eff+hmax and folds
--       d = sum_j x[t,j] * params[j] (beta = params[1..r], ordered-j fold).
--       Validates future coverage per present j (errors with the missing t
--       range named; j-density across columns is Layer 1's
--       _sarimax_validate_exog contract). Zero-row exog -> zero rows (callers
--       treat absent rows as d = 0, the r = 0 case).
--   _sarimax_fc_diff(state_tbl, sys_tbl, dfut_tbl, hmax) AS TABLE
--       -> (probe_id BIGINT, h BIGINT, mean_diff DOUBLE, var_diff DOUBLE,
--           omega DOUBLE[])
--       state_tbl: output of _sarimax_kfilter_state (probe_id, n_eff, a, p,
--       loglik); sys_tbl: output of _sarimax_systems (probe_id, k, tmat,
--       tmat_t, rqr, p1); dfut_tbl: (probe_id, h, d), at most one row per
--       (probe_id, h), missing rows meaning d = 0. The h-recursion runs ONCE
--       per probe as a recursive CTE over h (deterministic: one row per
--       (probe_id, h), every summation an ordered fold).
--       OMEGA ENCODING: omega has length h and holds row h of the
--       lower triangle of Omega: omega[j] = Omega[h, j] = Cov(w-forecast-
--       errors at horizons h and j), j = 1..h. The full h x h block for any
--       horizon follows by symmetry from rows 1..h. omega[h] == var_diff.
--   _sarimax_fc_diff_v2(state_tbl, sys_tbl, dfut_tbl, hmax) AS TABLE
--       -> same schema as _sarimax_fc_diff. The v2 variant: sys_tbl is the
--       _sarimax_systems_v2 output (has cidx) and dfut_tbl carries an extra
--       ct column (trend state intercept; see the macro header for the
--       empirically-pinned timing). _sarimax_fc_diff itself is untouched.
--   _sarimax_fc_orig(fcdiff_tbl, anchors_tbl, d, sd, s, hmax) AS TABLE
--       -> (probe_id BIGINT, h BIGINT, mean_orig DOUBLE, var_orig DOUBLE)
--       fcdiff_tbl: output of _sarimax_fc_diff (h dense 1..hmax per probe,
--       omega rows length h); anchors_tbl: output of _sarimax_diff_anchors
--       on the RAW series (probe-independent).
--   _sarimax_forecast_run(w_tbl, exog_diff_full_tbl, probes_tbl, anchors_tbl,
--                         r, p, q, bigp, bigq, s, d, sd, hmax) AS TABLE
--       -> (probe_id BIGINT, h BIGINT, mean_diff DOUBLE, se_diff DOUBLE,
--           mean_orig DOUBLE, se_orig DOUBLE)
--       Convenience chain: systems -> obs (in-sample rows only: w_tbl covers
--       t = 1..n_eff, so future exog rows never reach the filter) ->
--       kfilter_state -> dfut -> fc_diff -> fc_orig. n_eff is derived as
--       max(t) of w_tbl. This is what Layer 6 calls.
--
-- NOTE on parameter names (DuckDB macro params are case-insensitive and
-- shadow same-name column refs): the spec's P/Q/D/H are spelled bigp, bigq,
-- sd, hmax, as everywhere in this codebase.
--
-- DETERMINISM (spec 4.2): the horizon recursion is a recursive CTE producing
-- exactly one row per (probe_id, h); all reductions are ordered left-to-right
-- list folds (matrix products via Layer 0's _sarimax_mmul, ascending inner
-- index); the mean inversion uses window sums with explicit ORDER BY over
-- unique keys. All literals feeding DOUBLE arithmetic are DOUBLE (1e0/0e0).
--
-- Requires: sql/00_linalg.sql, sql/01_prep.sql, sql/02_ssm.sql,
--           sql/03_filter.sql.
-- ============================================================================


-- ----------------------------------------------------------------------------
-- _sarimax_fc_dfut: future observation intercepts d_fut[h] = x~_{n+h}' beta.
-- See header for the contract. beta = params[1..r] of each probe.
-- ----------------------------------------------------------------------------
CREATE OR REPLACE MACRO _sarimax_fc_dfut(exog_diff_full_tbl, probes_tbl, n_eff, hmax) AS TABLE
WITH
_sarimax_fd_nh AS (
    -- bind the scalar args ONCE (macro args are textually re-expanded at every
    -- use site; an expression arg referenced inside a lambda would re-evaluate
    -- per element -- so everything below reads the ne/hm columns instead)
    SELECT (n_eff)::BIGINT AS ne, (hmax)::BIGINT AS hm
),
_sarimax_fd_ex AS MATERIALIZED (
    SELECT t::BIGINT AS t, j::INT AS j, x::DOUBLE AS x
    FROM query_table(exog_diff_full_tbl)
),
_sarimax_fd_win AS MATERIALIZED (
    SELECT e.t, e.j, e.x
    FROM _sarimax_fd_ex e, _sarimax_fd_nh nh
    WHERE e.t > nh.ne AND e.t <= nh.ne + nh.hm
),
_sarimax_fd_missing AS (
    SELECT dj.j, gt.t
    FROM (SELECT DISTINCT j FROM _sarimax_fd_ex) dj, _sarimax_fd_nh nh,
         LATERAL unnest(range(nh.ne + 1, nh.ne + nh.hm + 1)) AS gt(t)
    WHERE NOT EXISTS (SELECT 1 FROM _sarimax_fd_win w2 WHERE w2.j = dj.j AND w2.t = gt.t)
),
_sarimax_fd_miss1 AS (
    SELECT j, min(t) AS t_min, max(t) AS t_max
    FROM _sarimax_fd_missing
    GROUP BY j ORDER BY j LIMIT 1
),
_sarimax_fd_null AS (
    SELECT j, min(t) AS t_min, max(t) AS t_max
    FROM _sarimax_fd_win
    WHERE x IS NULL
    GROUP BY j ORDER BY j LIMIT 1
),
_sarimax_fd_chk AS (
    SELECT CASE
             WHEN nh.hm < 1
               THEN error('_sarimax_fc_dfut: H must be >= 1, got ' || nh.hm)
             WHEN nh.ne < 1
               THEN error('_sarimax_fc_dfut: n_eff must be >= 1, got ' || nh.ne)
             WHEN EXISTS (SELECT 1 FROM _sarimax_fd_miss1)
               THEN error('_sarimax_fc_dfut: future exog coverage is incomplete: exog column j = '
                          || (SELECT j FROM _sarimax_fd_miss1) || ' is missing t in '
                          || (SELECT t_min FROM _sarimax_fd_miss1) || '..'
                          || (SELECT t_max FROM _sarimax_fd_miss1)
                          || ' (required future coverage: t = ' || (nh.ne + 1) || '..'
                          || (nh.ne + nh.hm) || ' on the differenced time axis)')
             WHEN EXISTS (SELECT 1 FROM _sarimax_fd_null)
               THEN error('_sarimax_fc_dfut: NULL future exog for column j = '
                          || (SELECT j FROM _sarimax_fd_null) || ' at t in '
                          || (SELECT t_min FROM _sarimax_fd_null) || '..'
                          || (SELECT t_max FROM _sarimax_fd_null))
             ELSE true
           END AS ok
    FROM _sarimax_fd_nh nh
)
SELECT pr.probe_id, (w.t - nh.ne)::BIGINT AS h,
       list_reduce(
           list_prepend(0e0, list(w.x * pr.params[w.j] ORDER BY w.j)),
           lambda zacc, zxb: zacc + zxb) AS d
FROM query_table(probes_tbl) pr
CROSS JOIN _sarimax_fd_win w
CROSS JOIN _sarimax_fd_nh nh
CROSS JOIN _sarimax_fd_chk ck
WHERE ck.ok
GROUP BY pr.probe_id, w.t, nh.ne
ORDER BY 1, 2;


-- ----------------------------------------------------------------------------
-- _sarimax_fc_diff: differenced-scale forecasts + Omega rows, one horizon
-- recursion per probe. See header for the omega encoding.
-- Carried recursion payload per (probe_id, h):
--   a, p           : a_h (k), P_h (k*k row-major)
--   gflat (h*k)    : gvec_0..gvec_{h-1} concatenated, gvec_m at m*k+1 .. m*k+k
--   pzflat (h*k)   : column 1 of P_1..P_h concatenated, P_j's at (j-1)*k+1 ..
-- ----------------------------------------------------------------------------
CREATE OR REPLACE MACRO _sarimax_fc_diff(state_tbl, sys_tbl, dfut_tbl, hmax) AS TABLE
WITH RECURSIVE
_sarimax_fcd_chk AS (
    SELECT CASE WHEN (hmax) < 1
                THEN error('_sarimax_fc_diff: H must be >= 1, got ' || (hmax))
                ELSE true END AS ok
),
-- densified per-(probe, h) intercept, built WITHOUT any outer join (a
-- non-inner join on a CTE subquery in this context is a DuckDB "Cannot
-- perform non-inner join on subquery" error when the dfut table arrives as a
-- caller CTE rather than a physical table): explicit zero base rows are
-- union-ed with the actual rows and summed, which equals coalesce(d, 0)
_sarimax_fcd_df AS MATERIALIZED (
    SELECT probe_id, h, sum(d) AS d
    FROM (
        SELECT st.probe_id, zu.zh AS h, 0e0 AS d
        FROM query_table(state_tbl) st
        CROSS JOIN (SELECT unnest(range(1, (hmax)::BIGINT + 1)) AS zh) zu
        UNION ALL
        SELECT df.probe_id, df.h, df.d
        FROM query_table(dfut_tbl) df
        WHERE df.h <= (hmax)
    )
    GROUP BY probe_id, h
),
_sarimax_fcd AS (
    SELECT st.probe_id,
           1::BIGINT AS h,
           sy.k AS k,
           df.d + st.a[1] AS mean_diff,
           st.p[1] AS var_diff,
           [st.p[1]]::DOUBLE[] AS omega,
           st.a AS a,
           st.p AS p,
           list_transform(range(1, sy.k + 1), lambda zi:
               CASE WHEN zi = 1 THEN 1e0 ELSE 0e0 END) AS gflat,
           list_transform(range(1, sy.k + 1), lambda zi:
               st.p[(zi - 1) * sy.k + 1]) AS pzflat
    FROM query_table(state_tbl) st
    JOIN query_table(sys_tbl) sy ON sy.probe_id = st.probe_id
    JOIN _sarimax_fcd_df df ON df.probe_id = st.probe_id AND df.h = 1
    CROSS JOIN _sarimax_fcd_chk ck
    WHERE ck.ok
    UNION ALL
    SELECT probe_id,
           h,
           k,
           dnew + anew[1] AS mean_diff,
           pnew[1] AS var_diff,
           -- omega[zj] = Omega[h, zj] = dot(gvec_{h-zj}, column 1 of P_zj)
           list_transform(range(1, h + 1), lambda zj:
               list_reduce(
                   list_transform(range(1, k + 1), lambda zi:
                       gflat2[(h - zj) * k + zi] * pzflat2[(zj - 1) * k + zi]),
                   lambda za, zb: za + zb)) AS omega,
           anew AS a,
           pnew AS p,
           gflat2 AS gflat,
           pzflat2 AS pzflat
    FROM (
        -- middle level: extend the gvec / P-column accumulators
        SELECT probe_id, h, k, dnew, anew, pnew,
               gflat || gnew AS gflat2,
               pzflat || list_transform(range(1, k + 1), lambda zi:
                   pnew[(zi - 1) * k + 1]) AS pzflat2
        FROM (
            -- inner level: one state-recursion step (intermediates bound as
            -- columns so nothing is re-expanded inside downstream lambdas)
            SELECT fc.probe_id,
                   fc.h + 1 AS h,
                   fc.k AS k,
                   df.d AS dnew,
                   _sarimax_mmul(sy.tmat, fc.a, fc.k, fc.k, 1) AS anew,
                   _sarimax_madd(
                       _sarimax_mmul(
                           _sarimax_mmul(sy.tmat, fc.p, fc.k, fc.k, fc.k),
                           sy.tmat_t, fc.k, fc.k, fc.k),
                       sy.rqr) AS pnew,
                   -- gvec_h = gvec_{h-1} T (row vector times T, ascending fold)
                   list_transform(range(1, fc.k + 1), lambda zj:
                       list_reduce(
                           list_transform(range(1, fc.k + 1), lambda zi:
                               fc.gflat[(fc.h - 1) * fc.k + zi]
                               * sy.tmat[(zi - 1) * fc.k + zj]),
                           lambda za, zb: za + zb)) AS gnew,
                   fc.gflat AS gflat,
                   fc.pzflat AS pzflat
            FROM _sarimax_fcd fc
            JOIN query_table(sys_tbl) sy ON sy.probe_id = fc.probe_id
            JOIN _sarimax_fcd_df df
                   ON df.probe_id = fc.probe_id AND df.h = fc.h + 1
            WHERE fc.h < (hmax)
        )
    )
)
SELECT probe_id, h, mean_diff, var_diff, omega
FROM _sarimax_fcd
ORDER BY 1, 2;


-- ----------------------------------------------------------------------------
-- _sarimax_fc_orig: original-scale forecasts. Mean by the per-stage windowed
-- cumulative-sum inversion of _sarimax_undiff_forecast (sql/01_prep.sql),
-- reimplemented with PARTITION BY probe_id (the Layer 1 macro handles one
-- series; exact agreement on the single-probe case is test-enforced); variance
-- by the integration-weights double fold over the Omega rows.
-- ----------------------------------------------------------------------------
CREATE OR REPLACE MACRO _sarimax_fc_orig(fcdiff_tbl, anchors_tbl, d, sd, s, hmax) AS TABLE
WITH
_sarimax_fo_fc AS MATERIALIZED (
    SELECT probe_id, h::BIGINT AS h, mean_diff::DOUBLE AS v, omega::DOUBLE[] AS omega
    FROM query_table(fcdiff_tbl)
),
_sarimax_fo_an AS MATERIALIZED (
    SELECT stage::INT AS stage, idx::INT AS idx, value::DOUBLE AS value
    FROM query_table(anchors_tbl)
),
-- scalar args bound as columns: an argument that is a scalar subquery may not
-- appear inside a non-inner join condition ("Cannot perform non-inner join on
-- subquery"), so the stage CTEs below reference these columns instead
_sarimax_fo_args AS MATERIALIZED (
    SELECT (d)::INT AS zd, (sd)::INT AS zsd, (s)::BIGINT AS zs
),
_sarimax_fo_chk AS (
    SELECT CASE
             WHEN d < 0 OR d > 4 THEN error('_sarimax_fc_orig: d must be in 0..4, got ' || d)
             WHEN sd < 0 OR sd > 3 THEN error('_sarimax_fc_orig: D must be in 0..3, got ' || sd)
             WHEN s < 1 THEN error('_sarimax_fc_orig: s must be >= 1, got ' || s)
             WHEN sd > 0 AND s < 2
               THEN error('_sarimax_fc_orig: s must be >= 2 when D > 0, got s = ' || s)
             WHEN (hmax) < 1 THEN error('_sarimax_fc_orig: H must be >= 1, got ' || (hmax))
             WHEN count(*) = 0 THEN error('_sarimax_fc_orig: forecast table is empty')
             WHEN count(v) < count(*)
               THEN error('_sarimax_fc_orig: forecast table contains NULL means')
             WHEN EXISTS (SELECT 1
                          FROM (SELECT probe_id, count(*) AS cnt, min(h) AS h_min,
                                       max(h) AS h_max, count(DISTINCT h) AS cntd
                                FROM _sarimax_fo_fc GROUP BY probe_id)
                          WHERE h_min != 1 OR h_max != (hmax) OR cnt != (hmax) OR cntd != cnt)
               THEN error('_sarimax_fc_orig: forecast horizons must be dense 1..' || (hmax)
                          || ' for every probe')
             WHEN EXISTS (SELECT 1 FROM _sarimax_fo_fc WHERE omega IS NULL OR len(omega) != h)
               THEN error('_sarimax_fc_orig: omega row at horizon h must have length h '
                          || '(the _sarimax_fc_diff encoding)')
             WHEN (SELECT count(*) FROM _sarimax_fo_an
                   WHERE stage BETWEEN 1 AND d AND idx = 1 AND value IS NOT NULL) != d
               THEN error('_sarimax_fc_orig: anchors table does not cover ordinary stages 1..' || d
                          || ' (idx = 1 each, non-NULL)')
             WHEN (SELECT count(*) FROM _sarimax_fo_an
                   WHERE stage > d AND stage <= d + sd AND idx BETWEEN 1 AND s
                     AND value IS NOT NULL) != sd * s
               THEN error('_sarimax_fc_orig: anchors table does not cover seasonal stages '
                          || (d + 1) || '..' || (d + sd) || ' (idx = 1..' || s || ' each, non-NULL)')
             ELSE true
           END AS ok
    FROM _sarimax_fo_fc
),
-- ---- mean: seasonal inversions, most recently applied stage first ----------
_sarimax_fo_s1 AS (
    SELECT f.probe_id, f.h,
           CASE WHEN za.zsd >= 1
                THEN a.value + sum(f.v) OVER (PARTITION BY f.probe_id, (f.h - 1) % za.zs ORDER BY f.h
                                              ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
                ELSE f.v END AS v
    FROM _sarimax_fo_fc f
    CROSS JOIN _sarimax_fo_args za
    LEFT JOIN _sarimax_fo_an a
      ON za.zsd >= 1 AND a.stage = za.zd + za.zsd AND a.idx = ((f.h - 1) % za.zs) + 1
),
_sarimax_fo_s2 AS (
    SELECT f.probe_id, f.h,
           CASE WHEN za.zsd >= 2
                THEN a.value + sum(f.v) OVER (PARTITION BY f.probe_id, (f.h - 1) % za.zs ORDER BY f.h
                                              ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
                ELSE f.v END AS v
    FROM _sarimax_fo_s1 f
    CROSS JOIN _sarimax_fo_args za
    LEFT JOIN _sarimax_fo_an a
      ON za.zsd >= 2 AND a.stage = za.zd + za.zsd - 1 AND a.idx = ((f.h - 1) % za.zs) + 1
),
_sarimax_fo_s3 AS (
    SELECT f.probe_id, f.h,
           CASE WHEN za.zsd >= 3
                THEN a.value + sum(f.v) OVER (PARTITION BY f.probe_id, (f.h - 1) % za.zs ORDER BY f.h
                                              ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
                ELSE f.v END AS v
    FROM _sarimax_fo_s2 f
    CROSS JOIN _sarimax_fo_args za
    LEFT JOIN _sarimax_fo_an a
      ON za.zsd >= 3 AND a.stage = za.zd + za.zsd - 2 AND a.idx = ((f.h - 1) % za.zs) + 1
),
-- ---- mean: ordinary inversions, most recently applied stage first ----------
_sarimax_fo_o1 AS (
    SELECT f.probe_id, f.h,
           CASE WHEN za.zd >= 1
                THEN a.value + sum(f.v) OVER (PARTITION BY f.probe_id ORDER BY f.h
                                              ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
                ELSE f.v END AS v
    FROM _sarimax_fo_s3 f
    CROSS JOIN _sarimax_fo_args za
    LEFT JOIN _sarimax_fo_an a ON za.zd >= 1 AND a.stage = za.zd AND a.idx = 1
),
_sarimax_fo_o2 AS (
    SELECT f.probe_id, f.h,
           CASE WHEN za.zd >= 2
                THEN a.value + sum(f.v) OVER (PARTITION BY f.probe_id ORDER BY f.h
                                              ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
                ELSE f.v END AS v
    FROM _sarimax_fo_o1 f
    CROSS JOIN _sarimax_fo_args za
    LEFT JOIN _sarimax_fo_an a ON za.zd >= 2 AND a.stage = za.zd - 1 AND a.idx = 1
),
_sarimax_fo_o3 AS (
    SELECT f.probe_id, f.h,
           CASE WHEN za.zd >= 3
                THEN a.value + sum(f.v) OVER (PARTITION BY f.probe_id ORDER BY f.h
                                              ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
                ELSE f.v END AS v
    FROM _sarimax_fo_o2 f
    CROSS JOIN _sarimax_fo_args za
    LEFT JOIN _sarimax_fo_an a ON za.zd >= 3 AND a.stage = za.zd - 2 AND a.idx = 1
),
_sarimax_fo_o4 AS (
    SELECT f.probe_id, f.h,
           CASE WHEN za.zd >= 4
                THEN a.value + sum(f.v) OVER (PARTITION BY f.probe_id ORDER BY f.h
                                              ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
                ELSE f.v END AS v
    FROM _sarimax_fo_o3 f
    CROSS JOIN _sarimax_fo_args za
    LEFT JOIN _sarimax_fo_an a ON za.zd >= 4 AND a.stage = za.zd - 3 AND a.idx = 1
),
-- ---- variance: integration-weights double fold over the Omega rows ---------
_sarimax_fo_wl AS (
    -- full lower triangle of c, flattened ordered (h, l):
    -- cl[h*(h-1)/2 + l] = c[h, l], 1 <= l <= h
    SELECT list(c ORDER BY h, l) AS cl
    FROM _sarimax_undiff_weights(d, sd, s, hmax)
),
_sarimax_fo_tri AS (
    -- full lower triangle of Omega per probe, flattened ordered by h:
    -- tri[i*(i-1)/2 + j] = Omega[i, j], 1 <= j <= i
    SELECT probe_id, flatten(list(omega ORDER BY h)) AS tri
    FROM _sarimax_fo_fc
    GROUP BY probe_id
),
_sarimax_fo_var AS (
    -- var_orig[h] = sum_{i<=h} sum_{j<=h} c[h,i] c[h,j] Omega[i,j],
    -- ordered fold: i ascending outer, j ascending inner (spec 4.2)
    SELECT f.probe_id, f.h,
           list_reduce(
               list_prepend(0e0, list_transform(range(1, f.h + 1), lambda zi:
                   list_reduce(
                       list_prepend(0e0, list_transform(range(1, f.h + 1), lambda zj:
                           w.cl[(f.h * (f.h - 1)) // 2 + zi]
                           * w.cl[(f.h * (f.h - 1)) // 2 + zj]
                           * CASE WHEN zj <= zi
                                  THEN tr.tri[(zi * (zi - 1)) // 2 + zj]
                                  ELSE tr.tri[(zj * (zj - 1)) // 2 + zi] END)),
                       lambda za, zb: za + zb))),
               lambda za2, zb2: za2 + zb2) AS var_orig
    FROM _sarimax_fo_fc f
    JOIN _sarimax_fo_tri tr ON tr.probe_id = f.probe_id
    CROSS JOIN _sarimax_fo_wl w
)
SELECT m.probe_id, m.h, m.v AS mean_orig, va.var_orig
FROM _sarimax_fo_o4 m
JOIN _sarimax_fo_var va ON va.probe_id = m.probe_id AND va.h = m.h
CROSS JOIN _sarimax_fo_chk ck
WHERE ck.ok
ORDER BY 1, 2;


-- ----------------------------------------------------------------------------
-- _sarimax_fc_diff_v2: the v2 horizon recursion -- identical omega/variance
-- machinery to _sarimax_fc_diff (which stays untouched for the v1 tests), plus
-- the trend STATE INTERCEPT of the v2 engine:
--
--   a_{h+1}[i] = (T a_h)[i] + (i = cidx ? ct_{h+1} : 0)
--
-- sys_tbl is the _sarimax_systems_v2 output (has cidx, the 1-based state row
-- that receives the trend intercept, and kdiff); dfut_tbl is
-- (probe_id, h, d, ct) with missing rows meaning d = 0 AND ct = 0.
--
-- CT TIMING CONVENTION (empirically pinned against the v2 trend fixtures'
-- forecast.parquet; see sql/06_harness.sql where the rows are built):
-- dfut.ct at row h is the state intercept CONSUMED FORMING the state used
-- for mean_h. The alignment to model time depends on the filter basis
-- (sql/03_filter.sql section 2, "the shifted-basis trick"):
--   kdiff = 0  (unshifted): the h = 1 state is a_{n+1}, whose intercept
--     c_{n_model} the filter already applied -- row 1's ct is never read,
--     and advancing h -> h+1 adds row (h+1)'s ct = c_{n_model+h}. Callers
--     fill ct[h] = c at model-time n_model + h - 1, i.e.
--     _sarimax_trend_c(degs, tau, n_model, H)[h].
--   kdiff > 0  (shifted basis): the stored filter state EXCLUDES its own
--     pending intercept c_{n_model+1} (applied as 0 past the sample end), so
--     THIS macro adds row 1's ct to the base state at cidx before the
--     recursion; the state used for mean_h consumed c at model-time
--     n_model + h. Callers fill ct[h] =
--     _sarimax_trend_c(degs, tau, n_model + 1, H)[h].
--
-- P is unaffected by the intercept, so var_diff / omega are computed exactly
-- as in v1. With ct identically 0 and cidx arbitrary this macro is
-- row-for-row identical to _sarimax_fc_diff (test-enforced).
-- ----------------------------------------------------------------------------
CREATE OR REPLACE MACRO _sarimax_fc_diff_v2(state_tbl, sys_tbl, dfut_tbl, hmax) AS TABLE
WITH RECURSIVE
_sarimax_fcd2_chk AS (
    SELECT CASE WHEN (hmax) < 1
                THEN error('_sarimax_fc_diff_v2: H must be >= 1, got ' || (hmax))
                ELSE true END AS ok
),
-- densified per-(probe, h) intercepts (both d and ct), same zero-base UNION
-- trick as _sarimax_fc_diff (no outer join on a caller CTE)
_sarimax_fcd2_df AS MATERIALIZED (
    SELECT probe_id, h, sum(d) AS d, sum(ct) AS ct
    FROM (
        SELECT st.probe_id, zu.zh AS h, 0e0 AS d, 0e0 AS ct
        FROM query_table(state_tbl) st
        CROSS JOIN (SELECT unnest(range(1, (hmax)::BIGINT + 1)) AS zh) zu
        UNION ALL
        SELECT df.probe_id, df.h, df.d, df.ct
        FROM query_table(dfut_tbl) df
        WHERE df.h <= (hmax)
    )
    GROUP BY probe_id, h
),
_sarimax_fcd2 AS (
    SELECT st.probe_id,
           1::BIGINT AS h,
           sy.k AS k,
           sy.cidx AS cidx,
           df.d + st.a[1] AS mean_diff,
           st.p[1] AS var_diff,
           [st.p[1]]::DOUBLE[] AS omega,
           -- shifted basis (kdiff > 0): restore the pending intercept the
           -- filter left out of the stored state (cidx > 1 there, so
           -- mean_diff at h = 1 is unaffected)
           list_transform(range(1, sy.k + 1), lambda zi:
               st.a[zi] + CASE WHEN sy.kdiff > 0 AND zi = sy.cidx
                               THEN df.ct ELSE 0e0 END) AS a,
           st.p AS p,
           list_transform(range(1, sy.k + 1), lambda zi:
               CASE WHEN zi = 1 THEN 1e0 ELSE 0e0 END) AS gflat,
           list_transform(range(1, sy.k + 1), lambda zi:
               st.p[(zi - 1) * sy.k + 1]) AS pzflat
    FROM query_table(state_tbl) st
    JOIN query_table(sys_tbl) sy ON sy.probe_id = st.probe_id
    JOIN _sarimax_fcd2_df df ON df.probe_id = st.probe_id AND df.h = 1
    CROSS JOIN _sarimax_fcd2_chk ck
    WHERE ck.ok
    UNION ALL
    SELECT probe_id,
           h,
           k,
           cidx,
           dnew + anew[1] AS mean_diff,
           pnew[1] AS var_diff,
           -- omega[zj] = Omega[h, zj] = dot(gvec_{h-zj}, column 1 of P_zj)
           list_transform(range(1, h + 1), lambda zj:
               list_reduce(
                   list_transform(range(1, k + 1), lambda zi:
                       gflat2[(h - zj) * k + zi] * pzflat2[(zj - 1) * k + zi]),
                   lambda za, zb: za + zb)) AS omega,
           anew AS a,
           pnew AS p,
           gflat2 AS gflat,
           pzflat2 AS pzflat
    FROM (
        -- outer level: apply the state intercept to the propagated mean
        -- (ta bound as a column below so the k^2 product is not re-expanded)
        SELECT probe_id, h, k, cidx, dnew, pnew, gflat2, pzflat2,
               list_transform(range(1, k + 1), lambda zi:
                   ta[zi] + CASE WHEN zi = cidx THEN ctnew ELSE 0e0 END) AS anew
        FROM (
            -- middle level: extend the gvec / P-column accumulators
            SELECT probe_id, h, k, cidx, dnew, ctnew, ta, pnew,
                   gflat || gnew AS gflat2,
                   pzflat || list_transform(range(1, k + 1), lambda zi:
                       pnew[(zi - 1) * k + 1]) AS pzflat2
            FROM (
                -- inner level: one state-recursion step (intermediates bound
                -- as columns so nothing re-expands inside downstream lambdas)
                SELECT fc.probe_id,
                       fc.h + 1 AS h,
                       fc.k AS k,
                       fc.cidx AS cidx,
                       df.d AS dnew,
                       df.ct AS ctnew,
                       _sarimax_mmul(sy.tmat, fc.a, fc.k, fc.k, 1) AS ta,
                       _sarimax_madd(
                           _sarimax_mmul(
                               _sarimax_mmul(sy.tmat, fc.p, fc.k, fc.k, fc.k),
                               sy.tmat_t, fc.k, fc.k, fc.k),
                           sy.rqr) AS pnew,
                       -- gvec_h = gvec_{h-1} T (row vector times T, ascending fold)
                       list_transform(range(1, fc.k + 1), lambda zj:
                           list_reduce(
                               list_transform(range(1, fc.k + 1), lambda zi:
                                   fc.gflat[(fc.h - 1) * fc.k + zi]
                                   * sy.tmat[(zi - 1) * fc.k + zj]),
                               lambda za, zb: za + zb)) AS gnew,
                       fc.gflat AS gflat,
                       fc.pzflat AS pzflat
                FROM _sarimax_fcd2 fc
                JOIN query_table(sys_tbl) sy ON sy.probe_id = fc.probe_id
                JOIN _sarimax_fcd2_df df
                       ON df.probe_id = fc.probe_id AND df.h = fc.h + 1
                WHERE fc.h < (hmax)
            )
        )
    )
)
SELECT probe_id, h, mean_diff, var_diff, omega
FROM _sarimax_fcd2
ORDER BY 1, 2;


-- ----------------------------------------------------------------------------
-- _sarimax_forecast_run: the full chain on both scales. w_tbl (t, w) is the
-- differenced IN-SAMPLE series t = 1..n_eff; exog_diff_full_tbl (t, j, x) the
-- lockstep-differenced in-sample ++ future exog t = 1..n_eff+H (zero rows if
-- r = 0); probes_tbl (probe_id, params); anchors_tbl the _sarimax_diff_anchors
-- output for the raw series. Output ordered (probe_id, h).
-- ----------------------------------------------------------------------------
CREATE OR REPLACE MACRO _sarimax_forecast_run(w_tbl, exog_diff_full_tbl, probes_tbl, anchors_tbl,
                                              r, p, q, bigp, bigq, s, d, sd, hmax) AS TABLE
WITH
_sarimax_fr_sys AS MATERIALIZED (
    SELECT * FROM _sarimax_systems(probes_tbl, r, p, q, bigp, bigq, s)
),
_sarimax_fr_obs AS MATERIALIZED (
    -- w_tbl only covers t = 1..n_eff, so only in-sample rows reach the filter
    SELECT * FROM _sarimax_obs_adj(w_tbl, exog_diff_full_tbl, probes_tbl)
),
_sarimax_fr_state AS MATERIALIZED (
    SELECT * FROM _sarimax_kfilter_state('_sarimax_fr_obs', '_sarimax_fr_sys')
),
_sarimax_fr_dfut AS MATERIALIZED (
    SELECT * FROM _sarimax_fc_dfut(exog_diff_full_tbl, probes_tbl,
                                   (SELECT max(t) FROM query_table(w_tbl)), hmax)
),
_sarimax_fr_fcd AS MATERIALIZED (
    SELECT * FROM _sarimax_fc_diff('_sarimax_fr_state', '_sarimax_fr_sys',
                                   '_sarimax_fr_dfut', hmax)
),
_sarimax_fr_fco AS MATERIALIZED (
    SELECT * FROM _sarimax_fc_orig('_sarimax_fr_fcd', anchors_tbl, d, sd, s, hmax)
)
SELECT fd.probe_id, fd.h, fd.mean_diff, sqrt(fd.var_diff) AS se_diff,
       fo.mean_orig, sqrt(fo.var_orig) AS se_orig
FROM _sarimax_fr_fcd fd
JOIN _sarimax_fr_fco fo ON fo.probe_id = fd.probe_id AND fo.h = fd.h
ORDER BY 1, 2;
