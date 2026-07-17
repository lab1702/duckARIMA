-- ============================================================================
-- duckARIMA Layer 1 -- series preparation (spec sections 4.1, 4.2, 5.1, 5.5).
-- Pure DuckDB (>= 1.5.4) SQL macros; load with:  .read sql/01_prep.sql
-- No extensions, no UDFs. All lambdas are Python-style (`lambda x: ...`) and
-- the library is clean under  SET lambda_syntax = 'DISABLE_SINGLE_ARROW'.
--
-- DATA MODEL
--   series : (t BIGINT, y DOUBLE), t dense and gap-free from 1
--   exog   : (t BIGINT, j INT, x DOUBLE) long form, j = 1..r
--   Table names are passed as strings and resolved via query_table(); column
--   names are passed as strings where a macro works on arbitrary user tables.
--   Names (tables, columns) beginning with `_sarimax_` are reserved.
--
-- DIFFERENCING CONVENTION (pinned)
--   Given (d, D, s), NON-SEASONAL differencing is applied d times FIRST, then
--   seasonal differencing at lag s is applied D times.  The operators commute
--   in exact arithmetic; in floating point the subtraction order matters, so
--   exactly this order is pinned -- it matches statsmodels'
--   simple_differencing=True path (np.diff applied d times, then seasonal
--   differencing D times).  Each stage is a sequential lag-subtract on the
--   previous stage's output (NOT a binomial-expansion linear combination), so
--   results are bit-identical to repeated np.diff / x[s:] - x[:-s].
--   The differenced series is re-densified to t = 1..n_eff,
--   n_eff = n - d - D*s.  Supported orders: 0 <= d <= 4, 0 <= D <= 3.
--
-- STAGE NUMBERING AND THE ANCHOR CONTRACT (consumed by Layer 5)
--   Stages are numbered 1..(d+D) in APPLICATION order:
--     stage i,       1 <= i <= d      : the i-th ordinary (lag-1) difference
--     stage d+k,     1 <= k <= D      : the k-th seasonal (lag-s) difference
--   Let series_0 = y and series_i = the series after stage i was applied.
--   _sarimax_diff_anchors returns (stage INT, idx INT, value DOUBLE):
--   for each stage i it stores the TRAILING values of series_(i-1) -- the
--   series as it existed immediately BEFORE stage i was applied:
--     ordinary stage i   : 1 row,  idx = 1,     value = series_(i-1)[n]
--     seasonal stage d+k : s rows, idx = 1..s,  value = series_(d+k-1)[n-s+idx]
--   (idx increases with time; idx = s is the most recent value; n is the
--   ORIGINAL series length -- every intermediate series here is kept on the
--   original time axis, defined for t = (lags consumed so far)+1 .. n.)
--   These trailing values are exactly what h-step forecast integration needs:
--   inversion proceeds stage (d+D) down to stage 1 (reverse of application),
--     seasonal stage:  v[h] = w[h] + (h <= s ? anchor[idx = h] : v[h-s])
--                      == anchor[((h-1) % s) + 1] + seasonal-cumsum of w
--     ordinary stage:  v[h] = w[h] + (h = 1 ? anchor[idx = 1] : v[h-1])
--                      == anchor[1] + cumsum of w.
--   Leading in-sample values are NOT stored: the forecast-integration path is
--   the load-bearing contract (in-sample values on the original scale are the
--   observed y themselves).
--
-- MACROS (all internal, `_sarimax_` prefix; signatures frozen for Layer 5)
--   NOTE on parameter names: DuckDB macro parameters are case-insensitive and
--   shadow unqualified column references, so the spec's `D` is spelled `sd`
--   (seasonal differencing order) and the horizon `H` is spelled `hmax`.
--   Callers pass arguments positionally; the semantics are unchanged.
--   _sarimax_diff(tbl, tcol, ycol, d, sd, s)          -> (t BIGINT, w DOUBLE)
--   _sarimax_diff_exog(tbl, d, sd, s)                 -> (t BIGINT, j INT, x DOUBLE)
--        lockstep differencing of long-form exog, partitioned by j;
--        zero-row input -> zero-row output.  tbl must have columns (t, j, x).
--   _sarimax_diff_anchors(tbl, tcol, ycol, d, sd, s)  -> (stage INT, idx INT, value DOUBLE)
--   _sarimax_undiff_forecast(fc_tbl, anchors_tbl, d, sd, s)
--                                                    -> (h BIGINT, yhat DOUBLE)
--        fc_tbl must have columns (h, w), h dense 1..H on the differenced
--        scale; anchors_tbl has the anchor schema above.
--   _sarimax_undiff_weights(d, sd, s, hmax)              -> (h INT, l INT, c DOUBLE)
--        integration weights: y[n+h] = sum_{l=1..h} c[h,l] * w[n+l] + (anchor
--        term, = _sarimax_undiff_forecast of an all-zero forecast).  Rows are
--        emitted for the full lower triangle 1 <= l <= h (zeros included).
--        These are the truncated coefficients of (1-L)^-d (1-L^s)^-D and map
--        the differenced-scale forecast-error covariance to original-scale
--        forecast variances in Layer 5.
--   _sarimax_validate_exog(tbl, n_expected)          -> (j INT, n_rows BIGINT, ok BOOLEAN)
--        errors (naming j and the offending t range) unless every j covers
--        t = 1..n_expected exactly once with no NULLs; j must be dense 1..r.
--        Rows with t outside 1..n_expected are ignored.  Zero rows (r = 0) is
--        valid and returns zero rows.
--   _sarimax_rank_check(tbl)                         -> (j INT, piv DOUBLE, pivmin DOUBLE, trace DOUBLE, ok BOOLEAN)
--        (the per-step pivot column is named `piv` because PIVOT is a
--        reserved DuckDB keyword)
--        tbl is DIFFERENCED long-form exog (t, j, x), assumed validated.
--        Builds the r x r Gram matrix X'X by grouped sums, runs Gaussian
--        elimination with partial (row) pivoting on it, and errors when the
--        smallest |pivot| < 1e-10 * trace(X'X).  Column named in the error:
--        the FIRST elimination step whose pivot falls below the threshold.
--        With row-only pivoting, step k eliminates column k, so this is the
--        first column numerically linearly dependent on columns 1..k-1 (a
--        constant column, differencing to zero, is named directly).
--   _sarimax_lags(tbl, tcol, ycol, m)                -> (t BIGINT, lag INT, value DOUBLE)
--        LONG-form lag matrix, lag = 0..m, value = y[t-lag]; rows emitted only
--        where the lagged value exists (t - lag >= 1).  Hannan-Rissanen keeps
--        t > m for complete rows and pivots by lag.  (Wide form is impossible
--        with a variable column count in a macro; long form is what the OLS
--        normal-equation assembly wants anyway.)
--   _sarimax_acf(tbl, tcol, ycol, nlags)             -> (lag INT, acf DOUBLE)
--        sample ACF, mean-corrected, denominator n at every lag (statsmodels
--        acf(adjusted=False) convention); acf[0] = 1.
--   _sarimax_pacf(tbl, tcol, ycol, nlags)            -> (lag INT, pacf DOUBLE)
--        PACF via Durbin-Levinson on the denominator-n ACF == statsmodels
--        pacf(method='ywm') (Yule-Walker, mle/n acov); pacf[0] = 1.
--   Private helpers: _sarimax_stage_all, _sarimax_prep_pivstep,
--   _sarimax_prep_pivlist, _sarimax_prep_pivots(a, n) -> DOUBLE (smallest
--   |pivot| of partial-pivoting elimination on a row-major flattened n x n
--   matrix; self-contained, no dependency on 00_linalg.sql).
--
-- DETERMINISM (spec 4.2): scalar reductions fold ordered lists left-to-right
-- with list_reduce; running sums use window functions with explicit ORDER BY.
-- The Gram-matrix sums in _sarimax_rank_check are bulk validation aggregates
-- (never feed likelihood arithmetic) and use plain SUM by design.
-- All literals feeding DOUBLE arithmetic are written as DOUBLE (1e0, 0e0, or
-- an explicit cast); integer arithmetic on t/d/D/s indices is intentional.
-- ============================================================================


-- ----------------------------------------------------------------------------
-- _sarimax_stage_all (private): every intermediate differencing stage, kept on
-- the ORIGINAL time axis t = 1..n.  Output columns:
--   a0 = original series; a1..a4 = after ordinary stages 1..4 (ai carries
--   a(i-1) forward when d < i, so a4 is always the series after ALL d
--   ordinary stages); b0 = a4; b1..b3 = after seasonal stages 1..3 (same
--   carry-forward, so b3 is always the fully differenced series).
-- Rows where a stage's lag reaches before t = 1 hold NULL for that stage.
-- Also validates: 0 <= d <= 4, 0 <= D <= 3, s >= 1 (>= 2 when D > 0), series
-- non-empty, no NULLs, t dense 1..n, and n >= d + D*s + 1.
-- ----------------------------------------------------------------------------
CREATE OR REPLACE MACRO _sarimax_stage_all(tbl, tcol, ycol, d, sd, s) AS TABLE
WITH
_sarimax_sa_in AS MATERIALIZED (
    SELECT struct_extract(_sarimax_r, tcol)::BIGINT AS t,
           struct_extract(_sarimax_r, ycol)::DOUBLE AS a0
    FROM (SELECT _sarimax_q AS _sarimax_r FROM query_table(tbl) AS _sarimax_q)
),
_sarimax_sa_chk AS (
    SELECT CASE
             WHEN d < 0 OR d > 4 THEN error('_sarimax_diff: d must be in 0..4, got ' || d)
             WHEN sd < 0 OR sd > 3 THEN error('_sarimax_diff: D must be in 0..3, got ' || sd)
             WHEN s < 1 THEN error('_sarimax_diff: s must be >= 1, got ' || s)
             WHEN sd > 0 AND s < 2 THEN error('_sarimax_diff: s must be >= 2 when D > 0, got s = ' || s)
             WHEN count(*) = 0 THEN error('_sarimax_diff: input series is empty')
             WHEN count(a0) < count(*) THEN error('_sarimax_diff: series contains NULL values')
             WHEN min(t) != 1 OR max(t) != count(*) OR count(DISTINCT t) != count(*)
               THEN error('_sarimax_diff: series time index must be dense 1..n (found min t = '
                          || min(t) || ', max t = ' || max(t) || ', rows = ' || count(*) || ')')
             WHEN count(*) < d + sd * s + 1
               THEN error('_sarimax_diff: series too short: n = ' || count(*)
                          || ' but d + D*s + 1 = ' || (d + sd * s + 1) || ' observations are required')
             ELSE true
           END AS ok
    FROM _sarimax_sa_in
),
_sarimax_sa1 AS (
    SELECT t, a0,
           CASE WHEN d >= 1 THEN a0 - lag(a0) OVER (ORDER BY t) ELSE a0 END AS a1
    FROM _sarimax_sa_in
),
_sarimax_sa2 AS (
    SELECT t, a0, a1,
           CASE WHEN d >= 2 THEN a1 - lag(a1) OVER (ORDER BY t) ELSE a1 END AS a2
    FROM _sarimax_sa1
),
_sarimax_sa3 AS (
    SELECT t, a0, a1, a2,
           CASE WHEN d >= 3 THEN a2 - lag(a2) OVER (ORDER BY t) ELSE a2 END AS a3
    FROM _sarimax_sa2
),
_sarimax_sa4 AS (
    SELECT t, a0, a1, a2, a3,
           CASE WHEN d >= 4 THEN a3 - lag(a3) OVER (ORDER BY t) ELSE a3 END AS a4
    FROM _sarimax_sa3
),
_sarimax_sb1 AS (
    SELECT t, a0, a1, a2, a3, a4,
           CASE WHEN sd >= 1 THEN a4 - lag(a4, s) OVER (ORDER BY t) ELSE a4 END AS b1
    FROM _sarimax_sa4
),
_sarimax_sb2 AS (
    SELECT t, a0, a1, a2, a3, a4, b1,
           CASE WHEN sd >= 2 THEN b1 - lag(b1, s) OVER (ORDER BY t) ELSE b1 END AS b2
    FROM _sarimax_sb1
),
_sarimax_sb3 AS (
    SELECT t, a0, a1, a2, a3, a4, b1, b2,
           CASE WHEN sd >= 3 THEN b2 - lag(b2, s) OVER (ORDER BY t) ELSE b2 END AS b3
    FROM _sarimax_sb2
)
SELECT f.t, f.a0, f.a1, f.a2, f.a3, f.a4, f.a4 AS b0, f.b1, f.b2, f.b3
FROM _sarimax_sb3 f, _sarimax_sa_chk c
WHERE c.ok;


-- ----------------------------------------------------------------------------
-- _sarimax_diff: the differenced series, re-densified to t = 1..n_eff.
-- Identity when d = D = 0.
-- ----------------------------------------------------------------------------
CREATE OR REPLACE MACRO _sarimax_diff(tbl, tcol, ycol, d, sd, s) AS TABLE
SELECT (st.t - (d + sd * s))::BIGINT AS t, st.b3 AS w
FROM _sarimax_stage_all(tbl, tcol, ycol, d, sd, s) st
WHERE st.t > d + sd * s
ORDER BY 1;


-- ----------------------------------------------------------------------------
-- _sarimax_diff_exog: identical lockstep differencing of long-form exog
-- (t, j, x), partitioned by j.  Assumes each j is dense t = 1..n (run
-- _sarimax_validate_exog first); zero-row input yields zero-row output.
-- ----------------------------------------------------------------------------
CREATE OR REPLACE MACRO _sarimax_diff_exog(tbl, d, sd, s) AS TABLE
WITH
_sarimax_de_chk AS (
    SELECT CASE
             WHEN d < 0 OR d > 4 THEN error('_sarimax_diff_exog: d must be in 0..4, got ' || d)
             WHEN sd < 0 OR sd > 3 THEN error('_sarimax_diff_exog: D must be in 0..3, got ' || sd)
             WHEN s < 1 THEN error('_sarimax_diff_exog: s must be >= 1, got ' || s)
             WHEN sd > 0 AND s < 2 THEN error('_sarimax_diff_exog: s must be >= 2 when D > 0, got s = ' || s)
             ELSE true
           END AS ok
),
_sarimax_de_in AS (
    SELECT t::BIGINT AS t, j::INT AS j, x::DOUBLE AS x
    FROM query_table(tbl)
),
_sarimax_de_a1 AS (
    SELECT t, j, CASE WHEN d >= 1 THEN x - lag(x) OVER (PARTITION BY j ORDER BY t) ELSE x END AS x
    FROM _sarimax_de_in
),
_sarimax_de_a2 AS (
    SELECT t, j, CASE WHEN d >= 2 THEN x - lag(x) OVER (PARTITION BY j ORDER BY t) ELSE x END AS x
    FROM _sarimax_de_a1
),
_sarimax_de_a3 AS (
    SELECT t, j, CASE WHEN d >= 3 THEN x - lag(x) OVER (PARTITION BY j ORDER BY t) ELSE x END AS x
    FROM _sarimax_de_a2
),
_sarimax_de_a4 AS (
    SELECT t, j, CASE WHEN d >= 4 THEN x - lag(x) OVER (PARTITION BY j ORDER BY t) ELSE x END AS x
    FROM _sarimax_de_a3
),
_sarimax_de_b1 AS (
    SELECT t, j, CASE WHEN sd >= 1 THEN x - lag(x, s) OVER (PARTITION BY j ORDER BY t) ELSE x END AS x
    FROM _sarimax_de_a4
),
_sarimax_de_b2 AS (
    SELECT t, j, CASE WHEN sd >= 2 THEN x - lag(x, s) OVER (PARTITION BY j ORDER BY t) ELSE x END AS x
    FROM _sarimax_de_b1
),
_sarimax_de_b3 AS (
    SELECT t, j, CASE WHEN sd >= 3 THEN x - lag(x, s) OVER (PARTITION BY j ORDER BY t) ELSE x END AS x
    FROM _sarimax_de_b2
)
SELECT (f.t - (d + sd * s))::BIGINT AS t, f.j, f.x
FROM _sarimax_de_b3 f, _sarimax_de_chk c
WHERE c.ok AND f.t > d + sd * s
ORDER BY 2, 1;


-- ----------------------------------------------------------------------------
-- _sarimax_diff_anchors: the trailing pre-stage values needed to invert the
-- differencing for h-step forecast integration.  See the anchor contract in
-- the header.  Empty result when d = D = 0.
-- ----------------------------------------------------------------------------
CREATE OR REPLACE MACRO _sarimax_diff_anchors(tbl, tcol, ycol, d, sd, s) AS TABLE
WITH
_sarimax_da_st AS MATERIALIZED (
    SELECT * FROM _sarimax_stage_all(tbl, tcol, ycol, d, sd, s)
),
_sarimax_da_n AS (
    SELECT max(t) AS n FROM _sarimax_da_st
)
SELECT stage, idx, value
FROM (
    -- ordinary stages i = 1..d: last value of series_(i-1) = a(i-1) at t = n
    SELECT g.i::INT AS stage, 1::INT AS idx,
           CASE g.i WHEN 1 THEN st.a0 WHEN 2 THEN st.a1 WHEN 3 THEN st.a2 WHEN 4 THEN st.a3 END AS value
    FROM _sarimax_da_st st, _sarimax_da_n nn, range(1, d + 1) g(i)
    WHERE st.t = nn.n
    UNION ALL
    -- seasonal stages d+k, k = 1..D: last s values of series_(d+k-1) = b(k-1)
    SELECT (d + g.k)::INT AS stage, (st.t - (nn.n - s))::INT AS idx,
           CASE g.k WHEN 1 THEN st.b0 WHEN 2 THEN st.b1 WHEN 3 THEN st.b2 END AS value
    FROM _sarimax_da_st st, _sarimax_da_n nn, range(1, sd + 1) g(k)
    WHERE st.t > nn.n - s
)
ORDER BY stage, idx;


-- ----------------------------------------------------------------------------
-- _sarimax_undiff_forecast: integrate h-step forecasts from the differenced
-- scale back to the original scale.  fc_tbl: (h, w) with h dense 1..H;
-- anchors_tbl: (stage, idx, value) per the anchor contract.  Stages are
-- inverted in REVERSE application order: seasonal stages (d+D .. d+1) first,
-- then ordinary stages (d .. 1).  Each inversion is a windowed cumulative sum
-- (per season phase for seasonal stages) anchored on the stored trailing
-- values -- no recursion.
-- ----------------------------------------------------------------------------
CREATE OR REPLACE MACRO _sarimax_undiff_forecast(fc_tbl, anchors_tbl, d, sd, s) AS TABLE
WITH
_sarimax_uf_fc AS MATERIALIZED (
    SELECT h::BIGINT AS h, w::DOUBLE AS v FROM query_table(fc_tbl)
),
_sarimax_uf_an AS MATERIALIZED (
    SELECT stage::INT AS stage, idx::INT AS idx, value::DOUBLE AS value
    FROM query_table(anchors_tbl)
),
_sarimax_uf_chk AS (
    SELECT CASE
             WHEN d < 0 OR d > 4 THEN error('_sarimax_undiff_forecast: d must be in 0..4, got ' || d)
             WHEN sd < 0 OR sd > 3 THEN error('_sarimax_undiff_forecast: D must be in 0..3, got ' || sd)
             WHEN s < 1 THEN error('_sarimax_undiff_forecast: s must be >= 1, got ' || s)
             WHEN sd > 0 AND s < 2 THEN error('_sarimax_undiff_forecast: s must be >= 2 when D > 0, got s = ' || s)
             WHEN count(*) = 0 THEN error('_sarimax_undiff_forecast: forecast table is empty')
             WHEN count(v) < count(*) THEN error('_sarimax_undiff_forecast: forecast table contains NULL values')
             WHEN min(h) != 1 OR max(h) != count(*) OR count(DISTINCT h) != count(*)
               THEN error('_sarimax_undiff_forecast: forecast horizons must be dense 1..H (found min h = '
                          || min(h) || ', max h = ' || max(h) || ', rows = ' || count(*) || ')')
             WHEN (SELECT count(*)
                   FROM (SELECT stage, idx FROM _sarimax_uf_an GROUP BY stage, idx HAVING count(*) > 1)) > 0
               THEN error('_sarimax_undiff_forecast: anchors table contains duplicate (stage, idx) rows')
             WHEN (SELECT count(*) FROM _sarimax_uf_an
                   WHERE stage BETWEEN 1 AND d AND idx = 1 AND value IS NOT NULL) != d
               THEN error('_sarimax_undiff_forecast: anchors table does not cover ordinary stages 1..' || d
                          || ' (idx = 1 each, non-NULL)')
             WHEN (SELECT count(*) FROM _sarimax_uf_an
                   WHERE stage > d AND stage <= d + sd AND idx BETWEEN 1 AND s AND value IS NOT NULL) != sd * s
               THEN error('_sarimax_undiff_forecast: anchors table does not cover seasonal stages '
                          || (d + 1) || '..' || (d + sd) || ' (idx = 1..' || s || ' each, non-NULL)')
             ELSE true
           END AS ok
    FROM _sarimax_uf_fc
),
-- seasonal inversions, most recently applied stage first
_sarimax_uf_s1 AS (
    SELECT f.h,
           CASE WHEN sd >= 1
                THEN a.value + sum(f.v) OVER (PARTITION BY (f.h - 1) % s ORDER BY f.h
                                              ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
                ELSE f.v END AS v
    FROM _sarimax_uf_fc f
    LEFT JOIN _sarimax_uf_an a
      ON sd >= 1 AND a.stage = d + sd AND a.idx = ((f.h - 1) % s) + 1
),
_sarimax_uf_s2 AS (
    SELECT f.h,
           CASE WHEN sd >= 2
                THEN a.value + sum(f.v) OVER (PARTITION BY (f.h - 1) % s ORDER BY f.h
                                              ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
                ELSE f.v END AS v
    FROM _sarimax_uf_s1 f
    LEFT JOIN _sarimax_uf_an a
      ON sd >= 2 AND a.stage = d + sd - 1 AND a.idx = ((f.h - 1) % s) + 1
),
_sarimax_uf_s3 AS (
    SELECT f.h,
           CASE WHEN sd >= 3
                THEN a.value + sum(f.v) OVER (PARTITION BY (f.h - 1) % s ORDER BY f.h
                                              ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
                ELSE f.v END AS v
    FROM _sarimax_uf_s2 f
    LEFT JOIN _sarimax_uf_an a
      ON sd >= 3 AND a.stage = d + sd - 2 AND a.idx = ((f.h - 1) % s) + 1
),
-- ordinary inversions, most recently applied stage first
_sarimax_uf_o1 AS (
    SELECT f.h,
           CASE WHEN d >= 1
                THEN a.value + sum(f.v) OVER (ORDER BY f.h ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
                ELSE f.v END AS v
    FROM _sarimax_uf_s3 f
    LEFT JOIN _sarimax_uf_an a ON d >= 1 AND a.stage = d AND a.idx = 1
),
_sarimax_uf_o2 AS (
    SELECT f.h,
           CASE WHEN d >= 2
                THEN a.value + sum(f.v) OVER (ORDER BY f.h ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
                ELSE f.v END AS v
    FROM _sarimax_uf_o1 f
    LEFT JOIN _sarimax_uf_an a ON d >= 2 AND a.stage = d - 1 AND a.idx = 1
),
_sarimax_uf_o3 AS (
    SELECT f.h,
           CASE WHEN d >= 3
                THEN a.value + sum(f.v) OVER (ORDER BY f.h ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
                ELSE f.v END AS v
    FROM _sarimax_uf_o2 f
    LEFT JOIN _sarimax_uf_an a ON d >= 3 AND a.stage = d - 2 AND a.idx = 1
),
_sarimax_uf_o4 AS (
    SELECT f.h,
           CASE WHEN d >= 4
                THEN a.value + sum(f.v) OVER (ORDER BY f.h ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
                ELSE f.v END AS v
    FROM _sarimax_uf_o3 f
    LEFT JOIN _sarimax_uf_an a ON d >= 4 AND a.stage = d - 3 AND a.idx = 1
)
SELECT f.h, f.v AS yhat
FROM _sarimax_uf_o4 f, _sarimax_uf_chk ck
WHERE ck.ok
ORDER BY 1;


-- ----------------------------------------------------------------------------
-- _sarimax_undiff_weights: c[h,l] such that
--   y[n+h] = sum_{l=1..h} c[h,l] * w[n+l] + (deterministic anchor term).
-- Computed by pushing a unit impulse at horizon l through the SAME inversion
-- machinery as _sarimax_undiff_forecast with all anchors zero (the impulse
-- response of the integration recursion).  Values are exact small integers
-- (binomial-with-repetition coefficients) represented as DOUBLE.
-- ----------------------------------------------------------------------------
CREATE OR REPLACE MACRO _sarimax_undiff_weights(d, sd, s, hmax) AS TABLE
WITH
_sarimax_uw_chk AS (
    SELECT CASE
             WHEN d < 0 OR d > 4 THEN error('_sarimax_undiff_weights: d must be in 0..4, got ' || d)
             WHEN sd < 0 OR sd > 3 THEN error('_sarimax_undiff_weights: D must be in 0..3, got ' || sd)
             WHEN s < 1 THEN error('_sarimax_undiff_weights: s must be >= 1, got ' || s)
             WHEN sd > 0 AND s < 2 THEN error('_sarimax_undiff_weights: s must be >= 2 when D > 0, got s = ' || s)
             WHEN hmax < 1 THEN error('_sarimax_undiff_weights: H must be >= 1, got ' || hmax)
             ELSE true
           END AS ok
),
_sarimax_uw0 AS (
    SELECT gh.h, gl.l, CASE WHEN gh.h = gl.l THEN 1e0 ELSE 0e0 END AS v
    FROM range(1, hmax + 1) gh(h), range(1, hmax + 1) gl(l)
    WHERE gl.l <= gh.h
),
_sarimax_uw_s1 AS (
    SELECT h, l,
           CASE WHEN sd >= 1
                THEN sum(v) OVER (PARTITION BY l, (h - 1) % s ORDER BY h
                                  ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
                ELSE v END AS v
    FROM _sarimax_uw0
),
_sarimax_uw_s2 AS (
    SELECT h, l,
           CASE WHEN sd >= 2
                THEN sum(v) OVER (PARTITION BY l, (h - 1) % s ORDER BY h
                                  ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
                ELSE v END AS v
    FROM _sarimax_uw_s1
),
_sarimax_uw_s3 AS (
    SELECT h, l,
           CASE WHEN sd >= 3
                THEN sum(v) OVER (PARTITION BY l, (h - 1) % s ORDER BY h
                                  ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
                ELSE v END AS v
    FROM _sarimax_uw_s2
),
_sarimax_uw_o1 AS (
    SELECT h, l,
           CASE WHEN d >= 1
                THEN sum(v) OVER (PARTITION BY l ORDER BY h ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
                ELSE v END AS v
    FROM _sarimax_uw_s3
),
_sarimax_uw_o2 AS (
    SELECT h, l,
           CASE WHEN d >= 2
                THEN sum(v) OVER (PARTITION BY l ORDER BY h ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
                ELSE v END AS v
    FROM _sarimax_uw_o1
),
_sarimax_uw_o3 AS (
    SELECT h, l,
           CASE WHEN d >= 3
                THEN sum(v) OVER (PARTITION BY l ORDER BY h ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
                ELSE v END AS v
    FROM _sarimax_uw_o2
),
_sarimax_uw_o4 AS (
    SELECT h, l,
           CASE WHEN d >= 4
                THEN sum(v) OVER (PARTITION BY l ORDER BY h ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
                ELSE v END AS v
    FROM _sarimax_uw_o3
)
SELECT f.h::INT AS h, f.l::INT AS l, f.v AS c
FROM _sarimax_uw_o4 f, _sarimax_uw_chk ck
WHERE ck.ok
ORDER BY 1, 2;


-- ----------------------------------------------------------------------------
-- _sarimax_validate_exog: every j must cover t = 1..n_expected exactly once
-- with no NULLs; j must be dense 1..r.  Errors name the offending j and t
-- range.  Rows with t outside 1..n_expected are ignored (future exog rows are
-- validated by a second call with n_expected = n + H).
-- ----------------------------------------------------------------------------
CREATE OR REPLACE MACRO _sarimax_validate_exog(tbl, n_expected) AS TABLE
WITH
_sarimax_ve_in AS MATERIALIZED (
    SELECT t::BIGINT AS t, j::INT AS j, x::DOUBLE AS x
    FROM query_table(tbl)
),
_sarimax_ve_dup AS (
    SELECT j, min(t) AS t_min, max(t) AS t_max
    FROM (SELECT j, t FROM _sarimax_ve_in WHERE t BETWEEN 1 AND n_expected
          GROUP BY j, t HAVING count(*) > 1)
    GROUP BY j ORDER BY j LIMIT 1
),
_sarimax_ve_null AS (
    SELECT j, min(t) AS t_min, max(t) AS t_max
    FROM _sarimax_ve_in
    WHERE x IS NULL AND t BETWEEN 1 AND n_expected
    GROUP BY j ORDER BY j LIMIT 1
),
_sarimax_ve_miss AS (
    SELECT dj.j, min(g.t) AS t_min, max(g.t) AS t_max
    FROM (SELECT DISTINCT j FROM _sarimax_ve_in) dj
    CROSS JOIN range(1, n_expected + 1) g(t)
    WHERE NOT EXISTS (SELECT 1 FROM _sarimax_ve_in e WHERE e.j = dj.j AND e.t = g.t)
    GROUP BY dj.j ORDER BY dj.j LIMIT 1
),
_sarimax_ve_jrange AS (
    SELECT min(j) AS j_min, max(j) AS j_max, count(DISTINCT j) AS j_cnt
    FROM _sarimax_ve_in
),
_sarimax_ve_chk AS (
    SELECT CASE
             WHEN n_expected < 1
               THEN error('_sarimax_validate_exog: n_expected must be >= 1, got ' || n_expected)
             WHEN (SELECT count(*) FROM _sarimax_ve_in) = 0
               THEN true  -- r = 0: no regressors, trivially valid
             WHEN (SELECT j_min FROM _sarimax_ve_jrange) < 1
               THEN error('_sarimax_validate_exog: exog column index j must be >= 1, found j = '
                          || (SELECT j_min FROM _sarimax_ve_jrange))
             WHEN (SELECT j_cnt FROM _sarimax_ve_jrange) != (SELECT j_max FROM _sarimax_ve_jrange)
               THEN error('_sarimax_validate_exog: exog column indices must be dense 1..r, found '
                          || (SELECT j_cnt FROM _sarimax_ve_jrange) || ' distinct j but max j = '
                          || (SELECT j_max FROM _sarimax_ve_jrange))
             WHEN EXISTS (SELECT 1 FROM _sarimax_ve_dup)
               THEN error('_sarimax_validate_exog: duplicate rows for exog column j = '
                          || (SELECT j FROM _sarimax_ve_dup) || ' at t in '
                          || (SELECT t_min FROM _sarimax_ve_dup) || '..' || (SELECT t_max FROM _sarimax_ve_dup))
             WHEN EXISTS (SELECT 1 FROM _sarimax_ve_null)
               THEN error('_sarimax_validate_exog: NULL x for exog column j = '
                          || (SELECT j FROM _sarimax_ve_null) || ' at t in '
                          || (SELECT t_min FROM _sarimax_ve_null) || '..' || (SELECT t_max FROM _sarimax_ve_null))
             WHEN EXISTS (SELECT 1 FROM _sarimax_ve_miss)
               THEN error('_sarimax_validate_exog: exog column j = '
                          || (SELECT j FROM _sarimax_ve_miss) || ' does not cover t in '
                          || (SELECT t_min FROM _sarimax_ve_miss) || '..' || (SELECT t_max FROM _sarimax_ve_miss)
                          || ' (required coverage: t = 1..' || n_expected || ')')
             ELSE true
           END AS ok
)
SELECT p.j, p.n_rows, true AS ok
FROM (SELECT j, count(*) AS n_rows
      FROM _sarimax_ve_in WHERE t BETWEEN 1 AND n_expected GROUP BY j) p,
     _sarimax_ve_chk ck
WHERE ck.ok
ORDER BY 1;


-- ----------------------------------------------------------------------------
-- _sarimax_prep_pivstep (private): one step of Gaussian elimination with
-- partial pivoting on a row-major flattened n x n matrix M (DOUBLE[]).
-- k = current elimination step, p = chosen pivot row (p >= k).  Swaps rows k
-- and p, then eliminates column k from rows below k.  Elimination is skipped
-- when the pivot is exactly zero (the recorded pivot already flags failure).
-- ----------------------------------------------------------------------------
CREATE OR REPLACE MACRO _sarimax_prep_pivstep(M, n, k, p) AS (
    list_transform(M, lambda zv, zpos:
        CASE WHEN ((zpos - 1) // n) + 1 <= k
             -- rows 1..k: apply the row swap only
             THEN M[((CASE WHEN ((zpos - 1) // n) + 1 = k THEN p
                           WHEN ((zpos - 1) // n) + 1 = p THEN k
                           ELSE ((zpos - 1) // n) + 1 END) - 1) * n
                     + (zpos - ((zpos - 1) // n) * n)]
             -- rows k+1..n: swap, then subtract (M[row,k]/pivot) * pivot row
             ELSE M[((CASE WHEN ((zpos - 1) // n) + 1 = p THEN k
                           ELSE ((zpos - 1) // n) + 1 END) - 1) * n
                     + (zpos - ((zpos - 1) // n) * n)]
                  - CASE WHEN M[(p - 1) * n + k] = 0e0 THEN 0e0
                         ELSE (M[((CASE WHEN ((zpos - 1) // n) + 1 = p THEN k
                                        ELSE ((zpos - 1) // n) + 1 END) - 1) * n + k]
                               / M[(p - 1) * n + k])
                              * M[(p - 1) * n + (zpos - ((zpos - 1) // n) * n)]
                    END
        END)
);


-- ----------------------------------------------------------------------------
-- _sarimax_prep_pivlist (private): |pivot| at each elimination step 1..n of
-- Gaussian elimination with partial (row) pivoting on a row-major flattened
-- n x n matrix, as a DOUBLE[] in step order.  The fold is sequential
-- (list_reduce), the pivot row is the FIRST row attaining the maximum |value|
-- at or below the diagonal (ties -> lowest row index, matching np.argmax).
-- ----------------------------------------------------------------------------
CREATE OR REPLACE MACRO _sarimax_prep_pivlist(a, n) AS (
    list_reduce(
        [ struct_pack(k := 0::BIGINT, M := a::DOUBLE[], piv := []::DOUBLE[]) ]
        || list_transform(range(1, n + 1), lambda zk:
               struct_pack(k := zk, M := []::DOUBLE[], piv := []::DOUBLE[])),
        lambda zacc, zel:
            list_transform(
                [ list_reduce(
                      list_transform(range(zel.k, n + 1), lambda zi:
                          struct_pack(i := zi, v := abs(zacc.M[(zi - 1) * n + zel.k]))),
                      lambda za, zb: CASE WHEN zb.v > za.v THEN zb ELSE za END
                  ).i ],
                lambda zp: struct_pack(
                    k := zel.k,
                    M := _sarimax_prep_pivstep(zacc.M, n, zel.k, zp),
                    piv := zacc.piv || [abs(zacc.M[(zp - 1) * n + zel.k])]
                )
            )[1]
    ).piv
);


-- ----------------------------------------------------------------------------
-- _sarimax_prep_pivots: smallest |pivot| from partial-pivoting Gaussian
-- elimination on a row-major flattened n x n matrix (DOUBLE[] of length n*n).
-- Self-contained; does not depend on sql/00_linalg.sql.
-- ----------------------------------------------------------------------------
CREATE OR REPLACE MACRO _sarimax_prep_pivots(a, n) AS (
    list_reduce(_sarimax_prep_pivlist(a, n), lambda za, zb: least(za, zb))
);


-- ----------------------------------------------------------------------------
-- _sarimax_rank_check: rank check on the DIFFERENCED long-form exog design
-- (t, j, x) via elimination pivots on the Gram matrix X'X.  See header for
-- the named-column semantics.  r = 0 (empty input) returns zero rows.
-- ----------------------------------------------------------------------------
CREATE OR REPLACE MACRO _sarimax_rank_check(tbl) AS TABLE
WITH
_sarimax_rc_in AS MATERIALIZED (
    SELECT t::BIGINT AS t, j::INT AS j, x::DOUBLE AS x
    FROM query_table(tbl)
),
_sarimax_rc_r AS (
    SELECT coalesce(max(j), 0)::INT AS r FROM _sarimax_rc_in
),
-- Gram matrix X'X by grouped sums (bulk validation aggregate; never feeds
-- likelihood arithmetic, so plain SUM is permitted here by spec 4.2).
_sarimax_rc_g AS MATERIALIZED (
    SELECT xa.j AS ga, xb.j AS gb, sum(xa.x * xb.x) AS gv
    FROM _sarimax_rc_in xa
    JOIN _sarimax_rc_in xb ON xb.t = xa.t
    GROUP BY xa.j, xb.j
),
_sarimax_rc_flat AS (
    SELECT list(gv ORDER BY ga, gb) AS aflat, count(*) AS ncells
    FROM _sarimax_rc_g
),
_sarimax_rc_res AS (
    SELECT rr.r,
           _sarimax_prep_pivlist(ff.aflat, rr.r) AS pl,
           list_reduce(list_transform(range(1, rr.r + 1), lambda zi:
                           ff.aflat[(zi - 1) * rr.r + zi]),
                       lambda za, zb: za + zb) AS tr,
           list_reduce(_sarimax_prep_pivlist(ff.aflat, rr.r),
                       lambda za, zb: least(za, zb)) AS pivmin,
           coalesce(list_reduce(
               list_transform(_sarimax_prep_pivlist(ff.aflat, rr.r), lambda zv, zi:
                   CASE WHEN zv < 1e-10 * list_reduce(list_transform(range(1, rr.r + 1), lambda zj:
                                                          ff.aflat[(zj - 1) * rr.r + zj]),
                                                      lambda za2, zb2: za2 + zb2)
                        THEN zi ELSE 2147483647::BIGINT END),
               lambda za, zb: least(za, zb)), 2147483647::BIGINT) AS badj
    FROM _sarimax_rc_flat ff, _sarimax_rc_r rr
    WHERE rr.r > 0
      AND CASE WHEN ff.ncells != rr.r * rr.r
               THEN error('_sarimax_rank_check: exog columns are not aligned across t ('
                          || ff.ncells || ' Gram cells for r = ' || rr.r
                          || '); run _sarimax_validate_exog first')
               ELSE true END
),
_sarimax_rc_chk AS (
    SELECT CASE
             WHEN EXISTS (SELECT 1 FROM _sarimax_rc_in WHERE x IS NULL)
               THEN error('_sarimax_rank_check: NULL x values in exog; run _sarimax_validate_exog first')
             WHEN EXISTS (SELECT 1 FROM _sarimax_rc_res WHERE NOT isfinite(pivmin))
               THEN error('_sarimax_rank_check: non-finite values in elimination (check exog for inf/NaN)')
             WHEN EXISTS (SELECT 1 FROM _sarimax_rc_res WHERE tr <= 0e0)
               THEN error('_sarimax_rank_check: all differenced exog columns are identically zero (trace(X''X) = 0)')
             WHEN EXISTS (SELECT 1 FROM _sarimax_rc_res WHERE pivmin < 1e-10 * tr)
               THEN error('_sarimax_rank_check: differenced exog design matrix is numerically rank-deficient: '
                          || 'elimination breaks down at column j = ' || (SELECT badj FROM _sarimax_rc_res)
                          || ' (|pivot| = ' || (SELECT pl[badj] FROM _sarimax_rc_res)
                          || ' < 1e-10 * trace(X''X) = ' || (SELECT 1e-10 * tr FROM _sarimax_rc_res)
                          || '); column ' || (SELECT badj FROM _sarimax_rc_res)
                          || ' is numerically linearly dependent on the columns before it')
             ELSE true
           END AS ok
)
SELECT unnest(list_transform(range(1, res.r + 1), lambda zi: zi::INT)) AS j,
       unnest(res.pl) AS piv,
       res.pivmin,
       res.tr AS trace,
       true AS ok
FROM _sarimax_rc_res res, _sarimax_rc_chk ck
WHERE ck.ok
ORDER BY 1;


-- ----------------------------------------------------------------------------
-- _sarimax_lags: LONG-form lag matrix for the Hannan-Rissanen OLS assembly.
-- (t, lag, value) with value = y[t-lag], lag = 0..m; rows emitted only where
-- the lagged observation exists (t - lag >= 1).  Filter t > m downstream for
-- complete rows.
-- ----------------------------------------------------------------------------
CREATE OR REPLACE MACRO _sarimax_lags(tbl, tcol, ycol, m) AS TABLE
WITH
_sarimax_lg_in AS MATERIALIZED (
    SELECT struct_extract(_sarimax_r, tcol)::BIGINT AS t,
           struct_extract(_sarimax_r, ycol)::DOUBLE AS y
    FROM (SELECT _sarimax_q AS _sarimax_r FROM query_table(tbl) AS _sarimax_q)
),
_sarimax_lg_chk AS (
    SELECT CASE WHEN m < 0 THEN error('_sarimax_lags: m must be >= 0, got ' || m)
                ELSE true END AS ok
)
SELECT a.t, g.k::INT AS lag, b.y AS value
FROM _sarimax_lg_in a
CROSS JOIN range(0, m + 1) g(k)
JOIN _sarimax_lg_in b ON b.t = a.t - g.k
CROSS JOIN _sarimax_lg_chk ck
WHERE ck.ok
ORDER BY 1, 2;


-- ----------------------------------------------------------------------------
-- _sarimax_acf: sample ACF, mean-corrected, denominator n at every lag
-- (statsmodels acf(adjusted=False) convention): acf[k] =
-- sum_{t=k+1..n}(y[t]-ybar)(y[t-k]-ybar) / sum_{t=1..n}(y[t]-ybar)^2.
-- Diagnostics only.  Sums are ordered left-to-right folds (deterministic).
-- t serves only to order the observations.
-- ----------------------------------------------------------------------------
CREATE OR REPLACE MACRO _sarimax_acf(tbl, tcol, ycol, nlags) AS TABLE
WITH
_sarimax_ac_in AS (
    SELECT struct_extract(_sarimax_r, tcol)::BIGINT AS t,
           struct_extract(_sarimax_r, ycol)::DOUBLE AS y
    FROM (SELECT _sarimax_q AS _sarimax_r FROM query_table(tbl) AS _sarimax_q)
),
_sarimax_ac_pack AS (
    SELECT list(y ORDER BY t) AS yl, count(*)::BIGINT AS n,
           count(*) - count(y) AS n_null
    FROM _sarimax_ac_in
),
_sarimax_ac_chk AS (
    SELECT CASE
             WHEN nlags < 0 THEN error('_sarimax_acf: nlags must be >= 0, got ' || nlags)
             WHEN n = 0 THEN error('_sarimax_acf: input series is empty')
             WHEN n_null > 0 THEN error('_sarimax_acf: series contains NULL values')
             WHEN nlags >= n THEN error('_sarimax_acf: nlags must be < n (nlags = ' || nlags || ', n = ' || n || ')')
             ELSE true
           END AS ok
    FROM _sarimax_ac_pack
),
_sarimax_ac_mu AS (
    SELECT yl, n, list_reduce(yl, lambda za, zb: za + zb) / n AS mu
    FROM _sarimax_ac_pack
),
_sarimax_ac_c0 AS (
    SELECT yl, n, mu,
           list_reduce(list_transform(yl, lambda zv: (zv - mu) * (zv - mu)),
                       lambda za, zb: za + zb) AS c0
    FROM _sarimax_ac_mu
),
_sarimax_ac_chk2 AS (
    SELECT CASE WHEN c0 <= 0e0 THEN error('_sarimax_acf: series is constant (zero variance)')
                ELSE true END AS ok
    FROM _sarimax_ac_c0
)
SELECT g.k::INT AS lag,
       CASE WHEN g.k = 0 THEN 1e0
            ELSE list_reduce(
                     list_transform(range(g.k + 1, st.n + 1), lambda zt:
                         (st.yl[zt] - st.mu) * (st.yl[zt - g.k] - st.mu)),
                     lambda za, zb: za + zb) / st.c0
       END AS acf
FROM _sarimax_ac_c0 st, range(0, nlags + 1) g(k), _sarimax_ac_chk c1, _sarimax_ac_chk2 c2
WHERE c1.ok AND c2.ok
ORDER BY 1;


-- ----------------------------------------------------------------------------
-- _sarimax_pacf: PACF via the Durbin-Levinson recursion on the denominator-n
-- ACF above.  Identical (to solver arithmetic) to statsmodels
-- pacf(method='ywm'), i.e. per-lag Yule-Walker with the mle (n-denominator)
-- autocovariance.  pacf[0] = 1.  Diagnostics only.
-- ----------------------------------------------------------------------------
CREATE OR REPLACE MACRO _sarimax_pacf(tbl, tcol, ycol, nlags) AS TABLE
WITH
_sarimax_pc_rho AS (
    SELECT list(acf ORDER BY lag) AS rho
    FROM _sarimax_acf(tbl, tcol, ycol, nlags)
),
_sarimax_pc_dl AS (
    -- rho[1] = acf(0) = 1, rho[k+1] = acf(k).  Fold carries phi = phi_{k,1..k}
    -- and pk = [pacf(1)..pacf(k)]:
    --   a_k = (rho_k - sum_j phi_{k-1,j} rho_{k-j}) / (1 - sum_j phi_{k-1,j} rho_j)
    --   phi_{k,j} = phi_{k-1,j} - a_k * phi_{k-1,k-j};  phi_{k,k} = pacf(k) = a_k
    SELECT list_reduce(
               [ struct_pack(k := 0::BIGINT, phi := []::DOUBLE[], pk := []::DOUBLE[]) ]
               || list_transform(range(1, nlags + 1), lambda zk:
                      struct_pack(k := zk, phi := []::DOUBLE[], pk := []::DOUBLE[])),
               lambda zacc, zel:
                   list_transform(
                       [ CASE WHEN zel.k = 1 THEN rho[2]
                              ELSE (rho[zel.k + 1]
                                    - list_reduce(
                                          list_transform(range(1, zel.k), lambda zj:
                                              zacc.phi[zj] * rho[zel.k - zj + 1]),
                                          lambda za, zb: za + zb))
                                   / (1e0
                                    - list_reduce(
                                          list_transform(range(1, zel.k), lambda zj:
                                              zacc.phi[zj] * rho[zj + 1]),
                                          lambda za, zb: za + zb))
                         END ],
                       lambda zaa: struct_pack(
                           k := zel.k,
                           phi := list_transform(zacc.phi, lambda zp, zj:
                                      zp - zaa * zacc.phi[zel.k - zj]) || [zaa],
                           pk := zacc.pk || [zaa]
                       )
                   )[1]
           ).pk AS pkl
    FROM _sarimax_pc_rho
)
SELECT g.k::INT AS lag,
       CASE WHEN g.k = 0 THEN 1e0 ELSE dl.pkl[g.k] END AS pacf
FROM _sarimax_pc_dl dl, range(0, nlags + 1) g(k)
ORDER BY 1;
