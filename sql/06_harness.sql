-- ============================================================================
-- duckARIMA Layer 6: public interface and harness (spec sections 5.5, 11).
--
-- Public macros (duckLM-style; table names as strings via query_table):
--   sarimax_fit(data, y_col, p, d, q, sp := 0, sd := 0, sq := 0, s := 1,
--               exog_cols := []::VARCHAR[], t_col := NULL,
--               trend := 'n', concentrate := false,
--               simple_differencing := true)             -> model table
--   sarimax_forecast(model, data, y_col, h, newdata := NULL, t_col := NULL,
--                    level := 0.95)                      -> forecasts
--   sarimax_summary(model, data, y_col)                  -> coefficient table
--   sarimax_evaluate(model, data, y_col, t_col := NULL)  -> fit metrics row
--   sarimax_residuals(model, data, y_col, t_col := NULL) -> innovations
--   sarimax_ljungbox(model, data, y_col, nlags, t_col := NULL)
--
-- NOTE on the seasonal argument names: the spec writes P :=, D :=, Q :=, but
-- DuckDB macro parameters are case-insensitive, so p/P, d/D, q/Q would
-- collide. The seasonal orders are therefore sp, sd, sq (documented in
-- GUIDE.md; positional order matches the spec).
--
-- V2 FIT OPTIONS (all fits route through the v2 engine of sql/02..04, which
-- reproduces v1 arithmetic exactly at kdiff = 0, ktrend = 0, conc = false on
-- a complete series):
--   trend               'n' | 'c' | 't' | 'ct' -> polynomial degrees
--                       [] | [0] | [1] | [0,1]; parameter names 'intercept'
--                       (degree 0) and 'drift' (degree 1), ordered FIRST.
--   concentrate         concentrate sigma2 out of the likelihood; sigma2 then
--                       appears only in meta (engine scale2), not as a param.
--   simple_differencing true  -> difference y/exog up front (v1 behavior;
--                                trend timeline = DIFFERENCED timeline);
--                       false -> filter the RAW series with the differencing
--                                states inside the state vector (statsmodels
--                                simple_differencing=False: no data loss,
--                                burn-in of d + s*D, diffuse-ish init).
--   Missing y (NULL) is allowed (not all-missing); exog must stay complete.
--
-- ROUTING NOTE: simple_differencing/concentrate/trend arrive as macro
-- LITERALS, so the CASE/WHERE gates they feed constant-fold at plan time --
-- the untaken branch of each UNION ALL pair is pruned and its error() checks
-- never run.
--
-- Model table schema (kind VARCHAR, name VARCHAR, idx INT, value DOUBLE,
-- value_list DOUBLE[]):
--   'param'     one row per parameter, canonical order via idx, constrained
--               (trend first, then exog betas, ar/ma/seasonal; sigma2 last
--               UNLESS concentrate -- then no sigma2 param row at all)
--   'param_unc' the same in unconstrained (optimizer) space
--   'bse'       standard errors (numerical Hessian), aligned with 'param' idx
--   'spec'      p, d, q, sp, sd, sq, s, r, n, n_eff, sdiff, conc, ktrend, burn
--   'meta'      loglik, aic, bic, converged, iterations, grad_norm, restarted,
--               sigma2 (= engine scale2 when concentrated).
--               AIC = 2k - 2ll, BIC = k*ln(n_eff - burn) - 2ll where k ALWAYS
--               counts sigma2 (k = ktrend + r + p + q + sp + sq + 1, even when
--               concentrated) and n_eff - burn = statsmodels' nobs_effective
--               -- both pinned empirically against fixtures_v2 fitted_meta.
--   'trend'     one row per trend term: idx, value = polynomial degree
--   'anchor'    name 'endog:<stage>' idx/value rows (Layer 1 anchor contract),
--               and 'exog:<j>' trailing original-scale exog values;
--               ONLY when sdiff = 1 (none stored when sdiff = 0)
--   'exog_col'  idx j -> name of the j-th exogenous column
--   'state'     name 'a' / 'P': final predicted state mean and vec(P) in
--               value_list (UNIT scale when concentrated); name 'k': state
--               dimension in value
--
-- Requires sql/00 .. sql/05.
-- ============================================================================

-- ---- distribution helpers ------------------------------------------------------
-- Regularized incomplete gamma: series for x < a+1, Lentz continued fraction
-- otherwise; fixed 200-term folds (deterministic, converged to ~1e-15 long
-- before). P(a,x) + Q(a,x) = 1.

CREATE OR REPLACE MACRO _sarimax_gser(za, zx) AS (
    (list_reduce(
        [struct_pack(ap := za, del := 1e0 / za, tot := 1e0 / za)]
        || list_transform(range(1, 201), lambda zi:
               struct_pack(ap := 0e0, del := 0e0, tot := 0e0)),
        lambda zacc, ze:
            struct_pack(ap := zacc.ap + 1e0,
                        del := zacc.del * zx / (zacc.ap + 1e0),
                        tot := zacc.tot + zacc.del * zx / (zacc.ap + 1e0)))
    ).tot * exp(-zx + za * ln(zx) - lgamma(za))
);

CREATE OR REPLACE MACRO _sarimax_gcf(za, zx) AS (
    (list_reduce(
        [struct_pack(bb := zx + 1e0 - za, cc := 1e300, dd := 1e0 / (zx + 1e0 - za),
                     hh := 1e0 / (zx + 1e0 - za), ii := 0e0)]
        || list_transform(range(1, 201), lambda zi:
               struct_pack(bb := 0e0, cc := 0e0, dd := 0e0, hh := 0e0, ii := 0e0)),
        lambda zacc, ze:
            (list_transform(
                [struct_pack(
                    zan := -(zacc.ii + 1e0) * (zacc.ii + 1e0 - za),
                    zb2 := zacc.bb + 2e0)],
                lambda zt: struct_pack(
                    bb := zt.zb2,
                    cc := CASE WHEN abs(zt.zb2 + zt.zan / zacc.cc) < 1e-300
                               THEN 1e-300 ELSE zt.zb2 + zt.zan / zacc.cc END,
                    dd := 1e0 / (CASE WHEN abs(zt.zb2 + zt.zan * zacc.dd) < 1e-300
                                      THEN 1e-300 ELSE zt.zb2 + zt.zan * zacc.dd END),
                    hh := zacc.hh
                          * (1e0 / (CASE WHEN abs(zt.zb2 + zt.zan * zacc.dd) < 1e-300
                                         THEN 1e-300 ELSE zt.zb2 + zt.zan * zacc.dd END))
                          * (CASE WHEN abs(zt.zb2 + zt.zan / zacc.cc) < 1e-300
                                  THEN 1e-300 ELSE zt.zb2 + zt.zan / zacc.cc END),
                    ii := zacc.ii + 1e0))
            )[1])
    ).hh * exp(-zx + za * ln(zx) - lgamma(za))
);

-- Lower regularized incomplete gamma P(a, x).
CREATE OR REPLACE MACRO _sarimax_gammp(za, zx) AS (
    CASE WHEN zx <= 0e0 THEN 0e0
         WHEN zx < za + 1e0 THEN _sarimax_gser(za, zx)
         ELSE 1e0 - _sarimax_gcf(za, zx) END
);

-- Upper regularized incomplete gamma Q(a, x).
CREATE OR REPLACE MACRO _sarimax_gammq(za, zx) AS (
    CASE WHEN zx <= 0e0 THEN 1e0
         WHEN zx < za + 1e0 THEN 1e0 - _sarimax_gser(za, zx)
         ELSE _sarimax_gcf(za, zx) END
);

-- Chi-square survival function (upper tail), df > 0.
CREATE OR REPLACE MACRO _sarimax_chi2_sf(zx, zdf) AS (
    _sarimax_gammq(zdf / 2e0, zx / 2e0)
);

-- Standard normal CDF via Phi(z) = 0.5 * erfc(|z|/sqrt(2)), erfc through Q(1/2, z^2/2).
CREATE OR REPLACE MACRO _sarimax_norm_cdf(zz) AS (
    CASE WHEN zz < 0e0 THEN 0.5e0 * _sarimax_gammq(0.5e0, zz * zz / 2e0)
         ELSE 1e0 - 0.5e0 * _sarimax_gammq(0.5e0, zz * zz / 2e0) END
);

-- Acklam's inverse-normal rational approximation, one Halley refinement.
CREATE OR REPLACE MACRO _sarimax_norm_ppf_raw(zp) AS (
    CASE
    WHEN zp < 0.02425e0 THEN
        (list_transform([sqrt(-2e0 * ln(zp))], lambda zq:
            (((((-7.784894002430293e-03 * zq - 3.223964580411365e-01) * zq
                - 2.400758277161838e+00) * zq - 2.549732539343734e+00) * zq
                + 4.374664141464968e+00) * zq + 2.938163982698783e+00)
            / ((((7.784695709041462e-03 * zq + 3.224671290700398e-01) * zq
                + 2.445134137142996e+00) * zq + 3.754408661907416e+00) * zq + 1e0)))[1]
    WHEN zp > 1e0 - 0.02425e0 THEN
        -(list_transform([sqrt(-2e0 * ln(1e0 - zp))], lambda zq:
            (((((-7.784894002430293e-03 * zq - 3.223964580411365e-01) * zq
                - 2.400758277161838e+00) * zq - 2.549732539343734e+00) * zq
                + 4.374664141464968e+00) * zq + 2.938163982698783e+00)
            / ((((7.784695709041462e-03 * zq + 3.224671290700398e-01) * zq
                + 2.445134137142996e+00) * zq + 3.754408661907416e+00) * zq + 1e0)))[1]
    ELSE
        (list_transform([(zp - 0.5e0) * (zp - 0.5e0)], lambda zr:
            (((((-3.969683028665376e+01 * zr + 2.209460984245205e+02) * zr
                - 2.759285104469687e+02) * zr + 1.383577518672690e+02) * zr
                - 3.066479806614716e+01) * zr + 2.506628277459239e+00) * (zp - 0.5e0)
            / (((((-5.447609879822406e+01 * zr + 1.615858368580409e+02) * zr
                - 1.556989798598866e+02) * zr + 6.680131188771972e+01) * zr
                - 1.328068155288572e+01) * zr + 1e0)))[1]
    END
);

-- One Halley refinement step for the inverse normal.
CREATE OR REPLACE MACRO _sarimax_norm_ppf_step(zx0, zp) AS (
    (list_transform(
        [(_sarimax_norm_cdf(zx0) - zp) * sqrt(2e0 * pi()) * exp(zx0 * zx0 / 2e0)],
        lambda zu: zx0 - zu / (1e0 + zx0 * zu / 2e0)))[1]
);

CREATE OR REPLACE MACRO _sarimax_norm_ppf(zp) AS (
    (list_transform([_sarimax_norm_ppf_raw(zp)], lambda zx0:
        (list_transform([_sarimax_norm_ppf_step(zx0, zp)], lambda zx1:
            _sarimax_norm_ppf_step(zx1, zp)))[1]))[1]
);

-- ---- input validation (spec 5.5 named failures) --------------------------------

CREATE OR REPLACE MACRO _sarimax_validate_spec(p, d, q, sp, sd, sq, s) AS (
    CASE
    WHEN p < 0 OR d < 0 OR q < 0 OR sp < 0 OR sd < 0 OR sq < 0
        THEN error('sarimax: all of p, d, q, sp(P), sd(D), sq(Q) must be >= 0')
    WHEN (sp > 0 OR sd > 0 OR sq > 0) AND s < 2
        THEN error('sarimax: s must be >= 2 whenever any of sp(P), sd(D), sq(Q) is > 0')
    ELSE true END
);

-- ---- internal: series / exog extraction from a user table ----------------------

-- (t, y) with t = 1..n by t_col order (or natural order when t_col is NULL).
-- struct_extract requires a CONSTANT key, hence the coalesce guard: when
-- t_col is NULL the CASE never reads the extracted value, but the branch must
-- still bind against a valid constant column name.
CREATE OR REPLACE MACRO _sarimax_series_of(data, y_col, t_col) AS TABLE
SELECT row_number() OVER (
           ORDER BY CASE WHEN t_col IS NULL THEN 0e0
                         ELSE struct_extract(zd, coalesce(t_col, y_col))::DOUBLE END) AS t,
       struct_extract(zd, y_col)::DOUBLE AS y
FROM query_table(data) zd;

-- Long-form exog (t, j, x) from named columns; zero rows when the list is
-- empty. struct_extract keys must be constants, so the column dispatch is a
-- CASE over constant list positions -- which caps the supported number of
-- regressors at 32 (documented; raise by extending the CASE, see
-- tools/gen_exog_dispatch note in the repo history).
CREATE OR REPLACE MACRO _sarimax_exog_x(zd, exog_cols, y_col, zj) AS (
    CASE zj
        WHEN 1  THEN struct_extract(zd, coalesce(exog_cols[1]::VARCHAR,  exog_cols[1]::VARCHAR, y_col))::DOUBLE
        WHEN 2  THEN struct_extract(zd, coalesce(exog_cols[2]::VARCHAR,  exog_cols[1]::VARCHAR, y_col))::DOUBLE
        WHEN 3  THEN struct_extract(zd, coalesce(exog_cols[3]::VARCHAR,  exog_cols[1]::VARCHAR, y_col))::DOUBLE
        WHEN 4  THEN struct_extract(zd, coalesce(exog_cols[4]::VARCHAR,  exog_cols[1]::VARCHAR, y_col))::DOUBLE
        WHEN 5  THEN struct_extract(zd, coalesce(exog_cols[5]::VARCHAR,  exog_cols[1]::VARCHAR, y_col))::DOUBLE
        WHEN 6  THEN struct_extract(zd, coalesce(exog_cols[6]::VARCHAR,  exog_cols[1]::VARCHAR, y_col))::DOUBLE
        WHEN 7  THEN struct_extract(zd, coalesce(exog_cols[7]::VARCHAR,  exog_cols[1]::VARCHAR, y_col))::DOUBLE
        WHEN 8  THEN struct_extract(zd, coalesce(exog_cols[8]::VARCHAR,  exog_cols[1]::VARCHAR, y_col))::DOUBLE
        WHEN 9  THEN struct_extract(zd, coalesce(exog_cols[9]::VARCHAR,  exog_cols[1]::VARCHAR, y_col))::DOUBLE
        WHEN 10 THEN struct_extract(zd, coalesce(exog_cols[10]::VARCHAR,  exog_cols[1]::VARCHAR, y_col))::DOUBLE
        WHEN 11 THEN struct_extract(zd, coalesce(exog_cols[11]::VARCHAR,  exog_cols[1]::VARCHAR, y_col))::DOUBLE
        WHEN 12 THEN struct_extract(zd, coalesce(exog_cols[12]::VARCHAR,  exog_cols[1]::VARCHAR, y_col))::DOUBLE
        WHEN 13 THEN struct_extract(zd, coalesce(exog_cols[13]::VARCHAR,  exog_cols[1]::VARCHAR, y_col))::DOUBLE
        WHEN 14 THEN struct_extract(zd, coalesce(exog_cols[14]::VARCHAR,  exog_cols[1]::VARCHAR, y_col))::DOUBLE
        WHEN 15 THEN struct_extract(zd, coalesce(exog_cols[15]::VARCHAR,  exog_cols[1]::VARCHAR, y_col))::DOUBLE
        WHEN 16 THEN struct_extract(zd, coalesce(exog_cols[16]::VARCHAR,  exog_cols[1]::VARCHAR, y_col))::DOUBLE
        WHEN 17 THEN struct_extract(zd, coalesce(exog_cols[17]::VARCHAR,  exog_cols[1]::VARCHAR, y_col))::DOUBLE
        WHEN 18 THEN struct_extract(zd, coalesce(exog_cols[18]::VARCHAR,  exog_cols[1]::VARCHAR, y_col))::DOUBLE
        WHEN 19 THEN struct_extract(zd, coalesce(exog_cols[19]::VARCHAR,  exog_cols[1]::VARCHAR, y_col))::DOUBLE
        WHEN 20 THEN struct_extract(zd, coalesce(exog_cols[20]::VARCHAR,  exog_cols[1]::VARCHAR, y_col))::DOUBLE
        WHEN 21 THEN struct_extract(zd, coalesce(exog_cols[21]::VARCHAR,  exog_cols[1]::VARCHAR, y_col))::DOUBLE
        WHEN 22 THEN struct_extract(zd, coalesce(exog_cols[22]::VARCHAR,  exog_cols[1]::VARCHAR, y_col))::DOUBLE
        WHEN 23 THEN struct_extract(zd, coalesce(exog_cols[23]::VARCHAR,  exog_cols[1]::VARCHAR, y_col))::DOUBLE
        WHEN 24 THEN struct_extract(zd, coalesce(exog_cols[24]::VARCHAR,  exog_cols[1]::VARCHAR, y_col))::DOUBLE
        WHEN 25 THEN struct_extract(zd, coalesce(exog_cols[25]::VARCHAR,  exog_cols[1]::VARCHAR, y_col))::DOUBLE
        WHEN 26 THEN struct_extract(zd, coalesce(exog_cols[26]::VARCHAR,  exog_cols[1]::VARCHAR, y_col))::DOUBLE
        WHEN 27 THEN struct_extract(zd, coalesce(exog_cols[27]::VARCHAR,  exog_cols[1]::VARCHAR, y_col))::DOUBLE
        WHEN 28 THEN struct_extract(zd, coalesce(exog_cols[28]::VARCHAR,  exog_cols[1]::VARCHAR, y_col))::DOUBLE
        WHEN 29 THEN struct_extract(zd, coalesce(exog_cols[29]::VARCHAR,  exog_cols[1]::VARCHAR, y_col))::DOUBLE
        WHEN 30 THEN struct_extract(zd, coalesce(exog_cols[30]::VARCHAR,  exog_cols[1]::VARCHAR, y_col))::DOUBLE
        WHEN 31 THEN struct_extract(zd, coalesce(exog_cols[31]::VARCHAR,  exog_cols[1]::VARCHAR, y_col))::DOUBLE
        WHEN 32 THEN struct_extract(zd, coalesce(exog_cols[32]::VARCHAR,  exog_cols[1]::VARCHAR, y_col))::DOUBLE
        ELSE error('sarimax: at most 32 exogenous columns are supported')
    END
);

CREATE OR REPLACE MACRO _sarimax_exog_of(data, exog_cols, y_col, t_col) AS TABLE
SELECT zs.t, zc.j::INT AS j, _sarimax_exog_x(zs.zd, exog_cols, y_col, zc.j) AS x
FROM (
    SELECT row_number() OVER (
               ORDER BY CASE WHEN t_col IS NULL THEN 0e0
                             ELSE struct_extract(zd, coalesce(t_col, y_col))::DOUBLE END) AS t,
           zd
    FROM query_table(data) zd
) zs
CROSS JOIN (
    SELECT unnest(range(1, len(exog_cols) + 1)) AS j
) zc;

-- ---- dynamic-order differencing -------------------------------------------------
-- Layer 1's _sarimax_diff uses window lag(x, s), whose offset must be a
-- constant -- fine for sarimax_fit (orders arrive as literals) but not for
-- macros that read the orders back out of a model table (scalar subqueries).
-- These variants difference by explicit convolution with the coefficients of
-- (1-L)^d (1-L^s)^D: algebraically identical, floating-point-equal to ~1 ulp
-- (summation order differs from sequential differencing), and used ONLY on
-- the model-table-driven paths (residual diagnostics, forecasting).

-- Coefficient list of (1-L)^d (1-L^s)^D, constant term first.
CREATE OR REPLACE MACRO _sarimax_diff_poly(zd, zsd, zs) AS (
    _sarimax_polymul(
        list_reduce(
            [[1e0]] || list_transform(range(1, zd + 1), lambda zi: [[1e0]][1]),
            lambda zacc, ze: _sarimax_polymul(zacc, [1e0, -1e0])),
        list_reduce(
            [[1e0]] || list_transform(range(1, zsd + 1), lambda zi: [[1e0]][1]),
            lambda zacc, ze: _sarimax_polymul(
                zacc,
                list_append(list_prepend(1e0, list_transform(range(1, zs), lambda zz: 0e0)),
                            -1e0)))
    )
);

CREATE OR REPLACE MACRO _sarimax_diff_dyn(tbl, tcol, ycol, d, sd, s) AS TABLE
WITH _sarimax_dd_args AS (
    SELECT d::INT AS zd, sd::INT AS zsd, s::BIGINT AS zs
),
_sarimax_dd_list AS (
    SELECT za.zd + za.zsd * za.zs AS zoff,
           _sarimax_diff_poly(za.zd, za.zsd, greatest(za.zs, 1)) AS zc,
           (SELECT list(struct_extract(zrow, ycol)::DOUBLE
                        ORDER BY struct_extract(zrow, tcol)) FROM query_table(tbl) zrow) AS zy
    FROM _sarimax_dd_args za
)
SELECT zu.zt AS t,
       list_reduce(
           list_prepend(0e0, list_transform(range(1, len(zl.zc) + 1), lambda zi:
               zl.zc[zi] * zl.zy[zu.zt + zl.zoff - (zi - 1)])),
           lambda zacc, zx: zacc + zx) AS w
FROM _sarimax_dd_list zl,
     LATERAL unnest(range(1, len(zl.zy) - zl.zoff + 1)) AS zu(zt);

CREATE OR REPLACE MACRO _sarimax_diff_exog_dyn(tbl, d, sd, s) AS TABLE
WITH _sarimax_de_args AS (
    SELECT d::INT AS zd, sd::INT AS zsd, s::BIGINT AS zs
),
_sarimax_de_lists AS (
    SELECT ze.j,
           za.zd + za.zsd * za.zs AS zoff,
           _sarimax_diff_poly(za.zd, za.zsd, greatest(za.zs, 1)) AS zc,
           list(ze.x ORDER BY ze.t) AS zy
    FROM query_table(tbl) ze
    CROSS JOIN _sarimax_de_args za
    GROUP BY ze.j, za.zd, za.zsd, za.zs
)
SELECT zu.zt AS t, zl.j,
       list_reduce(
           list_prepend(0e0, list_transform(range(1, len(zl.zc) + 1), lambda zi:
               zl.zc[zi] * zl.zy[zu.zt + zl.zoff - (zi - 1)])),
           lambda zacc, zx: zacc + zx) AS x
FROM _sarimax_de_lists zl,
     LATERAL unnest(range(1, len(zl.zy) - zl.zoff + 1)) AS zu(zt);

-- ---- v2 helpers: trend arg, NULL-tolerant differencing ---------------------------

-- trend argument -> polynomial degrees (0-based, statsmodels convention).
-- PURE (no error() branch): this macro is textually expanded into engine
-- arguments that may land inside a TRY(), where volatile error() is rejected
-- by the binder. Validation lives in _sarimax_validate_trend, called from the
-- fit's gate CTE (which every output row reads) BEFORE anything else runs.
CREATE OR REPLACE MACRO _sarimax_trend_degrees(trend) AS (
    CASE trend
        WHEN 'c'  THEN [0]::BIGINT[]
        WHEN 't'  THEN [1]::BIGINT[]
        WHEN 'ct' THEN [0, 1]::BIGINT[]
        ELSE []::BIGINT[]
    END
);

CREATE OR REPLACE MACRO _sarimax_validate_trend(trend) AS (
    CASE WHEN trend IN ('n', 'c', 't', 'ct') THEN true
         ELSE error('sarimax: trend must be one of ''n'', ''c'', ''t'', ''ct''; got '
                    || coalesce('''' || trend::VARCHAR || '''', 'NULL')) END
);

-- statsmodels' names for the trend parameters by degree.
CREATE OR REPLACE MACRO _sarimax_trend_name(zdeg) AS (
    CASE zdeg::BIGINT WHEN 0 THEN 'intercept' WHEN 1 THEN 'drift'
         ELSE 'trend.' || zdeg::VARCHAR END
);

-- NULL-tolerant clone of Layer 1's _sarimax_stage_all: identical staged
-- (sequential lag-subtract) differencing arithmetic -- bit-identical to
-- np.diff / statsmodels simple_differencing=True on complete data -- but a
-- NULL y propagates through the lag subtraction instead of raising (missing
-- values are handled by the v2 filter). All other validation is kept.
CREATE OR REPLACE MACRO _sarimax_stage_all_nt(tbl, tcol, ycol, d, sd, s) AS TABLE
WITH
_sarimax_snt_in AS MATERIALIZED (
    SELECT struct_extract(_sarimax_r, tcol)::BIGINT AS t,
           struct_extract(_sarimax_r, ycol)::DOUBLE AS a0
    FROM (SELECT _sarimax_q AS _sarimax_r FROM query_table(tbl) AS _sarimax_q)
),
_sarimax_snt_chk AS (
    SELECT CASE
             WHEN d < 0 OR d > 4 THEN error('_sarimax_diff: d must be in 0..4, got ' || d)
             WHEN sd < 0 OR sd > 3 THEN error('_sarimax_diff: D must be in 0..3, got ' || sd)
             WHEN s < 1 THEN error('_sarimax_diff: s must be >= 1, got ' || s)
             WHEN sd > 0 AND s < 2 THEN error('_sarimax_diff: s must be >= 2 when D > 0, got s = ' || s)
             WHEN count(*) = 0 THEN error('_sarimax_diff: input series is empty')
             WHEN min(t) != 1 OR max(t) != count(*) OR count(DISTINCT t) != count(*)
               THEN error('_sarimax_diff: series time index must be dense 1..n (found min t = '
                          || min(t) || ', max t = ' || max(t) || ', rows = ' || count(*) || ')')
             WHEN count(*) < d + sd * s + 1
               THEN error('_sarimax_diff: series too short: n = ' || count(*)
                          || ' but d + D*s + 1 = ' || (d + sd * s + 1) || ' observations are required')
             ELSE true
           END AS ok
    FROM _sarimax_snt_in
),
_sarimax_snt_a1 AS (
    SELECT t, a0,
           CASE WHEN d >= 1 THEN a0 - lag(a0) OVER (ORDER BY t) ELSE a0 END AS a1
    FROM _sarimax_snt_in
),
_sarimax_snt_a2 AS (
    SELECT t, a0, a1,
           CASE WHEN d >= 2 THEN a1 - lag(a1) OVER (ORDER BY t) ELSE a1 END AS a2
    FROM _sarimax_snt_a1
),
_sarimax_snt_a3 AS (
    SELECT t, a0, a1, a2,
           CASE WHEN d >= 3 THEN a2 - lag(a2) OVER (ORDER BY t) ELSE a2 END AS a3
    FROM _sarimax_snt_a2
),
_sarimax_snt_a4 AS (
    SELECT t, a0, a1, a2, a3,
           CASE WHEN d >= 4 THEN a3 - lag(a3) OVER (ORDER BY t) ELSE a3 END AS a4
    FROM _sarimax_snt_a3
),
_sarimax_snt_b1 AS (
    SELECT t, a0, a1, a2, a3, a4,
           CASE WHEN sd >= 1 THEN a4 - lag(a4, s) OVER (ORDER BY t) ELSE a4 END AS b1
    FROM _sarimax_snt_a4
),
_sarimax_snt_b2 AS (
    SELECT t, a0, a1, a2, a3, a4, b1,
           CASE WHEN sd >= 2 THEN b1 - lag(b1, s) OVER (ORDER BY t) ELSE b1 END AS b2
    FROM _sarimax_snt_b1
),
_sarimax_snt_b3 AS (
    SELECT t, a0, a1, a2, a3, a4, b1, b2,
           CASE WHEN sd >= 3 THEN b2 - lag(b2, s) OVER (ORDER BY t) ELSE b2 END AS b3
    FROM _sarimax_snt_b2
)
SELECT f.t, f.a0, f.a1, f.a2, f.a3, f.a4, f.a4 AS b0, f.b1, f.b2, f.b3
FROM _sarimax_snt_b3 f, _sarimax_snt_chk c
WHERE c.ok;

-- NULL-tolerant _sarimax_diff (rows where a lag reaches a NULL carry NULL w).
CREATE OR REPLACE MACRO _sarimax_diff_nt(tbl, tcol, ycol, d, sd, s) AS TABLE
SELECT (st.t - (d + sd * s))::BIGINT AS t, st.b3 AS w
FROM _sarimax_stage_all_nt(tbl, tcol, ycol, d, sd, s) st
WHERE st.t > d + sd * s
ORDER BY 1;

-- NULL-tolerant _sarimax_diff_anchors. A NULL trailing value produces a NULL
-- anchor row, which _sarimax_fc_orig rejects loudly at FORECAST time (fitting
-- a model whose trailing observations are missing is fine; integrating a
-- forecast from a missing anchor is not).
CREATE OR REPLACE MACRO _sarimax_diff_anchors_nt(tbl, tcol, ycol, d, sd, s) AS TABLE
WITH
_sarimax_dnt_st AS MATERIALIZED (
    SELECT * FROM _sarimax_stage_all_nt(tbl, tcol, ycol, d, sd, s)
),
_sarimax_dnt_n AS (
    SELECT max(t) AS n FROM _sarimax_dnt_st
)
SELECT stage, idx, value
FROM (
    SELECT g.i::INT AS stage, 1::INT AS idx,
           CASE g.i WHEN 1 THEN st.a0 WHEN 2 THEN st.a1 WHEN 3 THEN st.a2 WHEN 4 THEN st.a3 END AS value
    FROM _sarimax_dnt_st st, _sarimax_dnt_n nn, range(1, d + 1) g(i)
    WHERE st.t = nn.n
    UNION ALL
    SELECT (d + g.k)::INT AS stage, (st.t - (nn.n - s))::INT AS idx,
           CASE g.k WHEN 1 THEN st.b0 WHEN 2 THEN st.b1 WHEN 3 THEN st.b2 END AS value
    FROM _sarimax_dnt_st st, _sarimax_dnt_n nn, range(1, sd + 1) g(k)
    WHERE st.t > nn.n - s
)
ORDER BY stage, idx;

-- ---- the fit --------------------------------------------------------------------

CREATE OR REPLACE MACRO sarimax_fit(data, y_col, p, d, q,
                                    sp := 0, sd := 0, sq := 0, s := 1,
                                    exog_cols := []::VARCHAR[], t_col := NULL,
                                    trend := 'n', concentrate := false,
                                    simple_differencing := true) AS TABLE
WITH _sarimax_f_series AS (
    SELECT t, y FROM _sarimax_series_of(data, y_col, t_col)
),
_sarimax_f_chk AS (
    SELECT _sarimax_validate_spec(p, d, q, sp, sd, sq, s)
           AND _sarimax_validate_trend(trend)
           AND (SELECT CASE
                         WHEN count(*) = 0 THEN error('sarimax: input series is empty')
                         WHEN count(y) = 0
                           THEN error('sarimax: series is all-missing (every y is NULL)')
                         ELSE true END
                FROM _sarimax_f_series) AS ok
),
_sarimax_f_exog AS (
    SELECT t, j, x FROM _sarimax_exog_of(data, exog_cols, y_col, t_col)
),
-- model-scale series: differenced up front when simple_differencing (the
-- NULL-tolerant staged variant: bit-identical to v1 on complete data, NULL y
-- propagates), the RAW series otherwise. Literal WHERE gates constant-fold.
_sarimax_f_y AS (
    SELECT t, w AS y FROM _sarimax_diff_nt('_sarimax_f_series', 't', 'y', d, sd, s)
    WHERE simple_differencing
    UNION ALL
    SELECT t, y FROM _sarimax_f_series
    WHERE NOT simple_differencing
),
_sarimax_f_exd AS (
    SELECT t, j, x FROM _sarimax_diff_exog('_sarimax_f_exog', d, sd, s)
    WHERE simple_differencing
    UNION ALL
    SELECT t, j, x FROM _sarimax_f_exog
    WHERE NOT simple_differencing
),
_sarimax_f_degs AS (
    SELECT unnest(range(1, len(_sarimax_trend_degrees(trend)) + 1))::BIGINT AS idx,
           unnest(_sarimax_trend_degrees(trend))::BIGINT AS degree
),
_sarimax_f_fit AS (
    SELECT * FROM _sarimax_bfgs_v2('_sarimax_f_y', '_sarimax_f_exd', '_sarimax_f_degs',
                                   len(exog_cols), p, q, sp, sq, greatest(s, 1),
                                   CASE WHEN simple_differencing THEN 0 ELSE d END,
                                   CASE WHEN simple_differencing THEN 0 ELSE sd END,
                                   len(_sarimax_trend_degrees(trend)), concentrate)
),
_sarimax_f_dims AS (
    SELECT len(exog_cols)::INT AS r,
           len(_sarimax_trend_degrees(trend))::INT AS ktrend,
           (SELECT count(*) FROM _sarimax_f_series)::INT AS n,
           (SELECT count(*) FROM _sarimax_f_y)::INT AS n_eff,
           (CASE WHEN simple_differencing THEN 0 ELSE d + greatest(s, 1) * sd END)::INT AS burn,
           -- reported parameter count (sigma2 dropped when concentrated) ...
           (len(_sarimax_trend_degrees(trend)) + len(exog_cols) + p + q + sp + sq
            + CASE WHEN concentrate THEN 0 ELSE 1 END)::INT AS k_params,
           -- ... but AIC/BIC ALWAYS count sigma2 (statsmodels' df_model does,
           -- concentrated or not; pinned against fixtures_v2 fitted_meta)
           (len(_sarimax_trend_degrees(trend)) + len(exog_cols) + p + q + sp + sq + 1)::INT
               AS k_aicbic
    FROM _sarimax_f_chk
    WHERE ok
),
_sarimax_f_probe AS (
    SELECT 1::BIGINT AS probe_id, params FROM _sarimax_f_fit
),
_sarimax_f_sys AS (
    SELECT * FROM _sarimax_systems_v2('_sarimax_f_probe', len(exog_cols), p, q, sp, sq,
                                      greatest(s, 1),
                                      CASE WHEN simple_differencing THEN 0 ELSE d END,
                                      CASE WHEN simple_differencing THEN 0 ELSE sd END,
                                      len(_sarimax_trend_degrees(trend)), concentrate)
),
_sarimax_f_obs AS (
    SELECT * FROM _sarimax_obs_adj_v2('_sarimax_f_y', '_sarimax_f_exd', '_sarimax_f_probe',
                                      len(exog_cols), len(_sarimax_trend_degrees(trend)),
                                      '_sarimax_f_degs')
),
_sarimax_f_state AS (
    SELECT * FROM _sarimax_kfilter_state_v2('_sarimax_f_obs', '_sarimax_f_sys')
),
_sarimax_f_lists AS (
    SELECT (SELECT list(y ORDER BY t) FROM _sarimax_f_y) AS ylist,
           (SELECT CASE WHEN len(exog_cols) = 0
                        THEN (SELECT list([]::DOUBLE[]) FROM _sarimax_f_y)
                        ELSE (SELECT list(zxr ORDER BY zt)
                              FROM (SELECT t AS zt, list(x ORDER BY j) AS zxr
                                    FROM _sarimax_f_exd GROUP BY t))
                   END) AS xmat,
           (SELECT coalesce(list(degree ORDER BY idx), []::BIGINT[])
            FROM _sarimax_f_degs) AS degs
),
_sarimax_f_bse AS (
    SELECT bse FROM _sarimax_bse_v2(
        (SELECT params FROM _sarimax_f_fit),
        (SELECT ylist FROM _sarimax_f_lists),
        (SELECT xmat FROM _sarimax_f_lists),
        (SELECT degs FROM _sarimax_f_lists),
        len(exog_cols), p, q, sp, sq, greatest(s, 1),
        CASE WHEN simple_differencing THEN 0 ELSE d END,
        CASE WHEN simple_differencing THEN 0 ELSE sd END,
        len(_sarimax_trend_degrees(trend)), concentrate)
),
_sarimax_f_names AS (
    SELECT list_transform(range(1, dm.k_params + 1), lambda zi:
        CASE
        WHEN zi <= dm.ktrend
            THEN _sarimax_trend_name((_sarimax_trend_degrees(trend))[zi])
        WHEN zi <= dm.ktrend + dm.r THEN exog_cols[zi - dm.ktrend]
        WHEN zi <= dm.ktrend + dm.r + p THEN 'ar.L' || (zi - dm.ktrend - dm.r)::VARCHAR
        WHEN zi <= dm.ktrend + dm.r + p + q
            THEN 'ma.L' || (zi - dm.ktrend - dm.r - p)::VARCHAR
        WHEN zi <= dm.ktrend + dm.r + p + q + sp
            THEN 'ar.S.L' || ((zi - dm.ktrend - dm.r - p - q) * s)::VARCHAR
        WHEN zi <= dm.ktrend + dm.r + p + q + sp + sq
            THEN 'ma.S.L' || ((zi - dm.ktrend - dm.r - p - q - sp) * s)::VARCHAR
        ELSE 'sigma2' END) AS pnames
    FROM _sarimax_f_dims dm
)
SELECT 'param' AS kind, nm.pnames[zu.zi] AS name, zu.zi::INT AS idx,
       ft.params[zu.zi] AS value, NULL::DOUBLE[] AS value_list
FROM _sarimax_f_fit ft, _sarimax_f_names nm, _sarimax_f_dims dm,
     LATERAL unnest(range(1, dm.k_params + 1)) AS zu(zi)
UNION ALL
SELECT 'param_unc', nm.pnames[zu.zi], zu.zi::INT, ft.x_opt[zu.zi], NULL
FROM _sarimax_f_fit ft, _sarimax_f_names nm, _sarimax_f_dims dm,
     LATERAL unnest(range(1, dm.k_params + 1)) AS zu(zi)
UNION ALL
SELECT 'bse', nm.pnames[zu.zi], zu.zi::INT, bs.bse[zu.zi], NULL
FROM _sarimax_f_bse bs, _sarimax_f_names nm, _sarimax_f_dims dm,
     LATERAL unnest(range(1, dm.k_params + 1)) AS zu(zi)
UNION ALL
SELECT 'spec', zsp.name, zsp.i::INT, zsp.v, NULL
FROM _sarimax_f_dims dm,
     LATERAL (SELECT unnest(['p','d','q','sp','sd','sq','s','r','n','n_eff',
                             'sdiff','conc','ktrend','burn']) AS name,
                     unnest(range(1, 15)) AS i,
                     unnest([p::DOUBLE, d::DOUBLE, q::DOUBLE, sp::DOUBLE, sd::DOUBLE,
                             sq::DOUBLE, s::DOUBLE, dm.r::DOUBLE, dm.n::DOUBLE,
                             dm.n_eff::DOUBLE,
                             CASE WHEN simple_differencing THEN 1e0 ELSE 0e0 END,
                             CASE WHEN concentrate THEN 1e0 ELSE 0e0 END,
                             dm.ktrend::DOUBLE, dm.burn::DOUBLE]) AS v) zsp
UNION ALL
SELECT 'meta', zm.name, zm.i::INT, zm.v, NULL
FROM _sarimax_f_fit ft, _sarimax_f_dims dm,
     LATERAL (SELECT unnest(['loglik','aic','bic','converged','iterations',
                             'grad_norm','restarted','sigma2']) AS name,
                     unnest(range(1, 9)) AS i,
                     unnest([ft.loglik,
                             2e0 * dm.k_aicbic - 2e0 * ft.loglik,
                             dm.k_aicbic * ln((dm.n_eff - dm.burn)::DOUBLE) - 2e0 * ft.loglik,
                             CASE WHEN ft.converged THEN 1e0 ELSE 0e0 END,
                             ft.iterations::DOUBLE,
                             ft.grad_norm,
                             CASE WHEN ft.restarted THEN 1e0 ELSE 0e0 END,
                             CASE WHEN concentrate THEN ft.scale2
                                  ELSE ft.params[dm.k_params] END]) AS v) zm
UNION ALL
SELECT 'trend', _sarimax_trend_name(dg.degree), dg.idx::INT, dg.degree::DOUBLE, NULL
FROM _sarimax_f_degs dg
UNION ALL
SELECT 'anchor', 'endog:' || za.stage::VARCHAR, za.idx::INT, za.value, NULL
FROM _sarimax_diff_anchors_nt('_sarimax_f_series', 't', 'y', d, sd, s) za
WHERE simple_differencing
UNION ALL
SELECT 'anchor', 'exog:' || ze.j::VARCHAR, ze.ai::INT, ze.x, NULL
FROM (
    SELECT j, x, row_number() OVER (PARTITION BY j ORDER BY t) AS ai
    FROM (
        SELECT j, t, x,
               row_number() OVER (PARTITION BY j ORDER BY t DESC) AS rdesc
        FROM _sarimax_f_exog
        WHERE t <= (SELECT n FROM _sarimax_f_dims)
    )
    WHERE rdesc <= d + sd * s
) ze
WHERE simple_differencing
UNION ALL
SELECT 'exog_col', exog_cols[zu.zi], zu.zi::INT, NULL, NULL
FROM _sarimax_f_dims dm, LATERAL unnest(range(1, dm.r + 1)) AS zu(zi)
UNION ALL
SELECT 'state', 'a', NULL, NULL, st.a FROM _sarimax_f_state st
UNION ALL
SELECT 'state', 'P', NULL, NULL, st.p FROM _sarimax_f_state st
UNION ALL
SELECT 'state', 'k', NULL, sy.k::DOUBLE, NULL
FROM (SELECT k FROM _sarimax_f_sys LIMIT 1) sy;

-- ---- model-table readers (internal) ---------------------------------------------

CREATE OR REPLACE MACRO _sarimax_m_spec(model, which) AS (
    (SELECT value FROM query_table(model) WHERE kind = 'spec' AND name = which)
);
CREATE OR REPLACE MACRO _sarimax_m_meta(model, which) AS (
    (SELECT value FROM query_table(model) WHERE kind = 'meta' AND name = which)
);
CREATE OR REPLACE MACRO _sarimax_m_params(model) AS (
    (SELECT list(value ORDER BY idx) FROM query_table(model) WHERE kind = 'param')
);
CREATE OR REPLACE MACRO _sarimax_m_bse(model) AS (
    (SELECT list(value ORDER BY idx) FROM query_table(model) WHERE kind = 'bse')
);

-- ---- summary ---------------------------------------------------------------------

CREATE OR REPLACE MACRO sarimax_summary(model, data, y_col) AS TABLE
SELECT pm.idx, pm.name,
       pm.value AS coefficient,
       bs.value AS std_error,
       pm.value / bs.value AS z_stat,
       2e0 * (1e0 - _sarimax_norm_cdf(abs(pm.value / bs.value))) AS p_value,
       pm.value - _sarimax_norm_ppf(0.975e0) * bs.value AS ci_lo,
       pm.value + _sarimax_norm_ppf(0.975e0) * bs.value AS ci_hi
FROM (SELECT * FROM query_table(model) WHERE kind = 'param') pm
JOIN (SELECT * FROM query_table(model) WHERE kind = 'bse') bs USING (idx)
ORDER BY pm.idx;

-- ---- residuals / evaluate / ljung-box ----------------------------------------------

-- Names of the exogenous columns must be re-supplied as a LITERAL list by the
-- caller (struct field access requires constant keys in DuckDB); they are
-- validated against the names stored in the model table.
CREATE OR REPLACE MACRO _sarimax_check_exog_names(model, exog_cols) AS (
    CASE WHEN (SELECT coalesce(list(name ORDER BY idx), [])
               FROM query_table(model) WHERE kind = 'exog_col') = exog_cols
         THEN true
         ELSE error('sarimax: exog_cols does not match the columns the model was fit with: '
                    || (SELECT coalesce(string_agg(name, ', ' ORDER BY idx), '(none)')
                        FROM query_table(model) WHERE kind = 'exog_col')) END
);

-- Recompute the innovation trace at theta-hat from the data + model table,
-- through the V2 engine (so trend / concentrated / missing-value / nodiff
-- models diagnose correctly; a v1-style model is the kdiff = ktrend = 0,
-- conc = 0 special case). v is NULL (hence std_resid NULL) at missing t;
-- F is still reported there. When the model is concentrated the reported F
-- (and hence std_resid) is rescaled by meta sigma2 -- statsmodels' reported
-- trace has F multiplied by the scale, the stored state runs at unit scale.
CREATE OR REPLACE MACRO _sarimax_retrace(model, data, y_col, exog_cols, t_col) AS TABLE
WITH _sarimax_rt_chk AS (
    SELECT _sarimax_check_exog_names(model, exog_cols) AS ok
),
-- spec values bound as columns once; effective INTEGRATION orders (d_eff,
-- sd_eff: applied to the data up front, zero when sdiff = 0) vs ENGINE orders
-- (d_eng, sd_eng: differencing states inside the filter, zero when sdiff = 1).
-- coalesce defaults keep model tables from before the v2 columns readable.
_sarimax_rt_dims AS (
    SELECT zz.*,
           CASE WHEN zz.sdiff = 1 THEN zz.d  ELSE 0 END AS d_eff,
           CASE WHEN zz.sdiff = 1 THEN zz.sd ELSE 0 END AS sd_eff,
           CASE WHEN zz.sdiff = 1 THEN 0 ELSE zz.d  END AS d_eng,
           CASE WHEN zz.sdiff = 1 THEN 0 ELSE zz.sd END AS sd_eng
    FROM (SELECT _sarimax_m_spec(model, 'r')::INT AS r,
                 _sarimax_m_spec(model, 'p')::INT AS p,
                 _sarimax_m_spec(model, 'q')::INT AS q,
                 _sarimax_m_spec(model, 'sp')::INT AS sp,
                 _sarimax_m_spec(model, 'sq')::INT AS sq,
                 greatest(_sarimax_m_spec(model, 's')::INT, 1) AS s,
                 _sarimax_m_spec(model, 'd')::INT AS d,
                 _sarimax_m_spec(model, 'sd')::INT AS sd,
                 coalesce(_sarimax_m_spec(model, 'sdiff'), 1e0)::INT AS sdiff,
                 coalesce(_sarimax_m_spec(model, 'conc'), 0e0)::INT AS conc,
                 coalesce(_sarimax_m_spec(model, 'ktrend'), 0e0)::INT AS ktrend) zz
),
_sarimax_rt_series AS (
    SELECT zs.t, zs.y FROM _sarimax_series_of(data, y_col, t_col) zs, _sarimax_rt_chk zc
    WHERE zc.ok
),
_sarimax_rt_exog AS (
    SELECT t, j, x FROM _sarimax_exog_of(data, exog_cols, y_col, t_col)
),
-- _sarimax_diff_dyn with d = D = 0 is an exact identity (0e0 + 1e0 * y) and
-- propagates NULL y, so ONE call serves both sdiff modes.
_sarimax_rt_y AS (
    SELECT t, w AS y FROM _sarimax_diff_dyn('_sarimax_rt_series', 't', 'y',
        (SELECT d_eff FROM _sarimax_rt_dims),
        (SELECT sd_eff FROM _sarimax_rt_dims),
        (SELECT s FROM _sarimax_rt_dims))
),
_sarimax_rt_exd AS (
    SELECT t, j, x FROM _sarimax_diff_exog_dyn('_sarimax_rt_exog',
        (SELECT d_eff FROM _sarimax_rt_dims),
        (SELECT sd_eff FROM _sarimax_rt_dims),
        (SELECT s FROM _sarimax_rt_dims))
),
_sarimax_rt_probe AS (
    SELECT 1::BIGINT AS probe_id, _sarimax_m_params(model) AS params
),
_sarimax_rt_degs AS (
    SELECT idx::BIGINT AS idx, value::BIGINT AS degree
    FROM query_table(model) WHERE kind = 'trend'
),
_sarimax_rt_sys AS (
    SELECT * FROM _sarimax_systems_v2('_sarimax_rt_probe',
        (SELECT r FROM _sarimax_rt_dims),
        (SELECT p FROM _sarimax_rt_dims),
        (SELECT q FROM _sarimax_rt_dims),
        (SELECT sp FROM _sarimax_rt_dims),
        (SELECT sq FROM _sarimax_rt_dims),
        (SELECT s FROM _sarimax_rt_dims),
        (SELECT d_eng FROM _sarimax_rt_dims),
        (SELECT sd_eng FROM _sarimax_rt_dims),
        (SELECT ktrend FROM _sarimax_rt_dims),
        (SELECT conc = 1 FROM _sarimax_rt_dims))
),
_sarimax_rt_obs AS (
    SELECT * FROM _sarimax_obs_adj_v2('_sarimax_rt_y', '_sarimax_rt_exd', '_sarimax_rt_probe',
        (SELECT r FROM _sarimax_rt_dims),
        (SELECT ktrend FROM _sarimax_rt_dims),
        '_sarimax_rt_degs')
),
_sarimax_rt_scale AS (
    SELECT CASE WHEN zd.conc = 1 THEN _sarimax_m_meta(model, 'sigma2') ELSE 1e0 END AS sc
    FROM _sarimax_rt_dims zd
)
SELECT zk.t, zk.v, zk.f * zs.sc AS f, zk.v / sqrt(zk.f * zs.sc) AS std_resid
FROM _sarimax_kfilter_v2('_sarimax_rt_obs', '_sarimax_rt_sys') zk
CROSS JOIN _sarimax_rt_scale zs;

CREATE OR REPLACE MACRO sarimax_residuals(model, data, y_col, exog_cols := []::VARCHAR[], t_col := NULL) AS TABLE
SELECT t, v, f, std_resid
FROM _sarimax_retrace(model, data, y_col, exog_cols, t_col)
ORDER BY t;

-- Ljung-Box Q on the standardized innovations at the exact lag set 1..nlags.
-- Computed on the NON-NULL std_resid only (missing-y timesteps produce NULL
-- innovations): the surviving residuals are compacted in time order and n is
-- their count -- drop-the-NaNs, the same thing one does before handing a
-- residual series with gaps to statsmodels' acorr_ljungbox.
CREATE OR REPLACE MACRO sarimax_ljungbox(model, data, y_col, nlags, exog_cols := []::VARCHAR[], t_col := NULL) AS TABLE
WITH _sarimax_lb_sr AS (
    SELECT row_number() OVER (ORDER BY t) AS t, std_resid
    FROM _sarimax_retrace(model, data, y_col, exog_cols, t_col)
    WHERE std_resid IS NOT NULL
),
_sarimax_lb_acf AS (
    SELECT lag, acf FROM _sarimax_acf('_sarimax_lb_sr', 't', 'std_resid', nlags)
),
_sarimax_lb_n AS (
    SELECT count(*)::DOUBLE AS n FROM _sarimax_lb_sr
),
_sarimax_lb_q AS (
    SELECT za.lag,
           zn.n * (zn.n + 2e0) * sum(zb.acf * zb.acf / (zn.n - zb.lag)) AS stat
    FROM _sarimax_lb_acf za
    JOIN _sarimax_lb_acf zb ON zb.lag BETWEEN 1 AND za.lag
    CROSS JOIN _sarimax_lb_n zn
    WHERE za.lag >= 1
    GROUP BY za.lag, zn.n
)
SELECT lag, stat, _sarimax_chi2_sf(stat, lag::DOUBLE) AS pvalue
FROM _sarimax_lb_q
ORDER BY lag;

-- sigma2 reported from meta (= the engine's concentrated scale2 when the
-- model was fit with concentrate := true). resid_finite_frac is computed over
-- the NON-NULL std_resid (missing-y timesteps are excluded, not penalized).
CREATE OR REPLACE MACRO sarimax_evaluate(model, data, y_col, exog_cols := []::VARCHAR[], t_col := NULL) AS TABLE
SELECT _sarimax_m_meta(model, 'loglik') AS loglik,
       _sarimax_m_meta(model, 'aic') AS aic,
       _sarimax_m_meta(model, 'bic') AS bic,
       _sarimax_m_meta(model, 'sigma2') AS sigma2,
       _sarimax_m_spec(model, 'n_eff') AS n_eff,
       (SELECT 1e0 - (count(*) FILTER (WHERE NOT isfinite(std_resid)))::DOUBLE
                     / count(std_resid)
        FROM _sarimax_retrace(model, data, y_col, exog_cols, t_col)) AS resid_finite_frac,
       _sarimax_m_meta(model, 'converged') AS converged;

-- ---- forecast -----------------------------------------------------------------------

-- ONE pipeline serves both fit modes:
--   sdiff = 1: the v1 path -- future exog = trailing anchors ++ newdata,
--     lockstep-differenced; the diff-scale recursion runs the (d = D = 0)
--     v2 system; _sarimax_fc_orig integrates with the model's d, D.
--   sdiff = 0: the model scale IS the original scale -- the integration
--     orders (d_eff, sd_eff) collapse to 0, making _sarimax_fc_orig an exact
--     identity (weights c[h,l] = delta, anchor checks vacuous, no anchors
--     stored or needed) and future exog enter RAW (identity differencing).
-- Trend models add the state intercept inside _sarimax_fc_diff_v2 (ct
-- timing pinned there; the trend_c window starts at n_eff when the engine
-- ran unshifted, n_eff + 1 in the shifted kdiff > 0 basis). Concentrated
-- models: the stored state P is at unit scale, so both variance columns are
-- multiplied by meta sigma2 before se / lo / hi.
CREATE OR REPLACE MACRO sarimax_forecast(model, data, y_col, h,
                                         newdata := NULL, exog_cols := []::VARCHAR[],
                                         t_col := NULL, level := 0.95) AS TABLE
WITH _sarimax_fc_chk AS (
    SELECT _sarimax_check_exog_names(model, exog_cols)
           AND CASE WHEN len(exog_cols) > 0 AND newdata IS NULL
                    THEN error('sarimax: the model has exogenous regressors; supply future values via newdata')
                    ELSE true END AS ok
),
_sarimax_fc_dims AS (
    SELECT zz.*,
           CASE WHEN zz.sdiff = 1 THEN zz.d  ELSE 0 END AS d_eff,
           CASE WHEN zz.sdiff = 1 THEN zz.sd ELSE 0 END AS sd_eff,
           CASE WHEN zz.sdiff = 1 THEN 0 ELSE zz.d  END AS d_eng,
           CASE WHEN zz.sdiff = 1 THEN 0 ELSE zz.sd END AS sd_eng
    FROM (SELECT _sarimax_m_spec(model, 'r')::INT AS r,
                 _sarimax_m_spec(model, 'p')::INT AS p,
                 _sarimax_m_spec(model, 'q')::INT AS q,
                 _sarimax_m_spec(model, 'sp')::INT AS sp,
                 _sarimax_m_spec(model, 'sd')::INT AS sd,
                 _sarimax_m_spec(model, 'sq')::INT AS sq,
                 _sarimax_m_spec(model, 's')::INT AS s,
                 _sarimax_m_spec(model, 'd')::INT AS d,
                 _sarimax_m_spec(model, 'n_eff')::BIGINT AS n_eff,
                 coalesce(_sarimax_m_spec(model, 'sdiff'), 1e0)::INT AS sdiff,
                 coalesce(_sarimax_m_spec(model, 'conc'), 0e0)::INT AS conc,
                 coalesce(_sarimax_m_spec(model, 'ktrend'), 0e0)::INT AS ktrend
          FROM _sarimax_fc_chk
          WHERE ok) zz
),
_sarimax_fc_probe AS MATERIALIZED (
    SELECT 1::BIGINT AS probe_id, _sarimax_m_params(model) AS params
),
_sarimax_fc_sys AS MATERIALIZED (
    SELECT * FROM _sarimax_systems_v2('_sarimax_fc_probe',
        (SELECT r FROM _sarimax_fc_dims),
        (SELECT p FROM _sarimax_fc_dims),
        (SELECT q FROM _sarimax_fc_dims),
        (SELECT sp FROM _sarimax_fc_dims),
        (SELECT sq FROM _sarimax_fc_dims),
        (SELECT greatest(s, 1) FROM _sarimax_fc_dims),
        (SELECT d_eng FROM _sarimax_fc_dims),
        (SELECT sd_eng FROM _sarimax_fc_dims),
        (SELECT ktrend FROM _sarimax_fc_dims),
        (SELECT conc = 1 FROM _sarimax_fc_dims))
),
_sarimax_fc_state AS MATERIALIZED (
    SELECT 1::BIGINT AS probe_id,
           (SELECT n_eff FROM _sarimax_fc_dims) AS n_eff,
           (SELECT value_list FROM query_table(model) WHERE kind = 'state' AND name = 'a') AS a,
           (SELECT value_list FROM query_table(model) WHERE kind = 'state' AND name = 'P') AS p,
           _sarimax_m_meta(model, 'loglik') AS loglik
),
-- future exog: trailing in-sample anchors ++ newdata on a shifted axis (the
-- shift = d_eff + sd_eff*s = 0 with no anchor rows when sdiff = 0, so the
-- newdata rows pass through raw), differenced by the EFFECTIVE orders; rows
-- re-densify to exactly h future rows on the model scale
_sarimax_fc_exfull AS (
    SELECT za.j, za.ai AS tt, za.value AS x, 0 AS grp
    FROM (SELECT split_part(name, ':', 2)::INT AS j, idx AS ai, value
          FROM query_table(model) WHERE kind = 'anchor' AND name LIKE 'exog:%') za
    UNION ALL
    SELECT zx.j,
           (SELECT max(zd2.d_eff + zd2.sd_eff * zd2.s) FROM _sarimax_fc_dims zd2) + zx.t,
           zx.x, 1
    FROM _sarimax_exog_of(coalesce(newdata, data), exog_cols, y_col, t_col) zx
),
_sarimax_fc_exfull2 AS (
    SELECT tt AS t, j, x FROM _sarimax_fc_exfull
),
_sarimax_fc_exd AS MATERIALIZED (
    SELECT zz.t, zz.j, zz.x
    FROM _sarimax_diff_exog_dyn('_sarimax_fc_exfull2',
        (SELECT d_eff FROM _sarimax_fc_dims),
        (SELECT sd_eff FROM _sarimax_fc_dims),
        (SELECT s FROM _sarimax_fc_dims)) zz
),
-- trend state intercepts for the horizon window, bound as columns FIRST (the
-- degs/tau expressions must not reach _sarimax_trend_c's lambdas as raw
-- subqueries). ct[zh] = the intercept consumed FORMING the state used at
-- horizon zh: c at model-time n_eff + zh - 1 in the unshifted (kdiff = 0)
-- filter basis, n_eff + zh in the shifted one (kdiff = d_eng + s*sd_eng > 0,
-- where the stored state excludes its pending intercept) -- see
-- _sarimax_fc_diff_v2's header.
_sarimax_fc_targs AS (
    SELECT zd.ktrend AS ktrend,
           (SELECT coalesce(list(value::BIGINT ORDER BY idx), []::BIGINT[])
            FROM query_table(model) WHERE kind = 'trend') AS degs,
           list_slice(zp.params, 1, zd.ktrend) AS tau,
           zd.n_eff + CASE WHEN zd.d_eng + zd.s * zd.sd_eng > 0
                           THEN 1 ELSE 0 END AS tstart
    FROM _sarimax_fc_dims zd, _sarimax_fc_probe zp
),
_sarimax_fc_ct AS (
    SELECT CASE WHEN ta.ktrend = 0
                THEN list_transform(range(1, (h)::BIGINT + 1), lambda zi: 0e0)
                ELSE _sarimax_trend_c(ta.degs, ta.tau, ta.tstart, (h)::BIGINT)
           END AS ctl
    FROM _sarimax_fc_targs ta
),
-- densified per-horizon observation intercept (exog dot beta; beta lives at
-- params[ktrend+1 .. ktrend+r]) and trend state intercept
-- (the horizon column is spelled zh inside this CTE: h is a macro ARGUMENT
-- of sarimax_forecast and would shadow a same-named GROUP BY column ref)
_sarimax_fc_dfut AS MATERIALIZED (
    SELECT probe_id, zh AS h, sum(zdv) AS d, sum(zct) AS ct
    FROM (
        SELECT zp.probe_id, zu.zh AS zh, 0e0 AS zdv, zc.ctl[zu.zh] AS zct
        FROM _sarimax_fc_probe zp, _sarimax_fc_ct zc,
             LATERAL unnest(range(1, (h)::BIGINT + 1)) AS zu(zh)
        UNION ALL
        SELECT zp.probe_id, zed.t AS zh,
               list_reduce(
                   list_prepend(0e0, list(zed.x * zp.params[zdm.ktrend + zed.j] ORDER BY zed.j)),
                   lambda zacc, zxb: zacc + zxb) AS zdv,
               0e0 AS zct
        FROM _sarimax_fc_exd zed, _sarimax_fc_probe zp, _sarimax_fc_dims zdm
        GROUP BY zp.probe_id, zed.t, zp.params, zdm.ktrend
    )
    GROUP BY probe_id, zh
),
_sarimax_fc_diffres AS MATERIALIZED (
    SELECT * FROM _sarimax_fc_diff_v2('_sarimax_fc_state', '_sarimax_fc_sys', '_sarimax_fc_dfut', h)
),
_sarimax_fc_anch AS MATERIALIZED (
    SELECT split_part(name, ':', 2)::INT AS stage, idx, value
    FROM query_table(model) WHERE kind = 'anchor' AND name LIKE 'endog:%'
),
_sarimax_fc_orig_res AS (
    SELECT * FROM _sarimax_fc_orig('_sarimax_fc_diffres', '_sarimax_fc_anch',
        (SELECT d_eff FROM _sarimax_fc_dims),
        (SELECT sd_eff FROM _sarimax_fc_dims),
        (SELECT s FROM _sarimax_fc_dims), h)
),
_sarimax_fc_scale AS (
    SELECT CASE WHEN zd.conc = 1 THEN _sarimax_m_meta(model, 'sigma2') ELSE 1e0 END AS sc
    FROM _sarimax_fc_dims zd
)
SELECT zdf.h,
       zor.mean_orig AS yhat,
       sqrt(zor.var_orig * zs.sc) AS se,
       zor.mean_orig - _sarimax_norm_ppf(0.5e0 + level / 2e0) * sqrt(zor.var_orig * zs.sc) AS lo,
       zor.mean_orig + _sarimax_norm_ppf(0.5e0 + level / 2e0) * sqrt(zor.var_orig * zs.sc) AS hi,
       zdf.mean_diff AS yhat_diff,
       sqrt(zdf.var_diff * zs.sc) AS se_diff
FROM _sarimax_fc_diffres zdf
JOIN _sarimax_fc_orig_res zor
  ON zor.probe_id = zdf.probe_id AND zor.h = zdf.h
CROSS JOIN _sarimax_fc_scale zs
ORDER BY zdf.h;

-- ---- grid runner ------------------------------------------------------------------

-- DuckDB cannot correlate table-macro arguments through a LATERAL join (the
-- macro stack misbinds), so the order-grid runner follows duckLM's
-- dummy_encode_sql precedent: it RETURNS THE SQL TEXT that fits every row of
-- the orders table and ranks by AIC; run it as a second step:
--     SELECT sarimax_grid_sql('sales', 'units', 'orders');   -- copy the text
-- or from the CLI:  .once grid.sql  /  SELECT ...;  /  .read grid.sql
-- The orders table must have columns (p, d, q, sp, sd, sq, s).
CREATE OR REPLACE MACRO sarimax_grid_sql(data, y_col, orders, t_col := NULL) AS (
    (SELECT string_agg(
        'SELECT ' || zo.p || ' AS p, ' || zo.d || ' AS d, ' || zo.q || ' AS q, '
                  || zo.sp || ' AS sp, ' || zo.sd || ' AS sd, ' || zo.sq || ' AS sq, '
                  || zo.s || ' AS s, '
                  || 'max(CASE WHEN name = ''loglik'' THEN value END) AS loglik, '
                  || 'max(CASE WHEN name = ''aic'' THEN value END) AS aic, '
                  || 'max(CASE WHEN name = ''bic'' THEN value END) AS bic, '
                  || 'max(CASE WHEN name = ''converged'' THEN value END) AS converged '
                  || 'FROM sarimax_fit(' || '''' || data || ''', ''' || y_col || ''', '
                  || zo.p || ', ' || zo.d || ', ' || zo.q
                  || ', sp := ' || zo.sp || ', sd := ' || zo.sd || ', sq := ' || zo.sq
                  || ', s := ' || zo.s
                  || CASE WHEN t_col IS NULL THEN '' ELSE ', t_col := ''' || t_col || '''' END
                  || ') WHERE kind = ''meta''',
        ' UNION ALL ' ORDER BY zo.p, zo.d, zo.q, zo.sp, zo.sd, zo.sq, zo.s)
     FROM query_table(orders) zo) || ' ORDER BY aic'
);
