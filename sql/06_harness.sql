-- ============================================================================
-- duckARIMA Layer 6: public interface and harness (spec sections 5.5, 11).
--
-- Public macros (duckLM-style; table names as strings via query_table):
--   sarimax_fit(data, y_col, p, d, q, sp := 0, sd := 0, sq := 0, s := 1,
--               exog_cols := []::VARCHAR[], t_col := NULL)          -> model table
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
-- Model table schema (kind VARCHAR, name VARCHAR, idx INT, value DOUBLE,
-- value_list DOUBLE[]):
--   'param'     one row per parameter, canonical order via idx, constrained
--   'param_unc' the same in unconstrained (optimizer) space
--   'bse'       standard errors (numerical Hessian), aligned with 'param' idx
--   'spec'      p, d, q, sp, sd, sq, s, r, n, n_eff, k_states
--   'meta'      loglik, aic, bic, converged, iterations, grad_norm, restarted
--   'anchor'    name 'endog:<stage>' idx/value rows (Layer 1 anchor contract),
--               and 'exog:<j>' trailing original-scale exog values
--   'exog_col'  idx j -> name of the j-th exogenous column
--   'state'     name 'a' / 'P': final predicted state mean and vec(P) in
--               value_list; name 'k': state dimension in value
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
-- regressors at 12 (documented; raise by extending the CASE).
CREATE OR REPLACE MACRO _sarimax_exog_x(zd, exog_cols, y_col, zj) AS (
    CASE zj
        WHEN 1  THEN struct_extract(zd, coalesce(exog_cols[1]::VARCHAR, exog_cols[1]::VARCHAR, y_col))::DOUBLE
        WHEN 2  THEN struct_extract(zd, coalesce(exog_cols[2]::VARCHAR, exog_cols[1]::VARCHAR, y_col))::DOUBLE
        WHEN 3  THEN struct_extract(zd, coalesce(exog_cols[3]::VARCHAR, exog_cols[1]::VARCHAR, y_col))::DOUBLE
        WHEN 4  THEN struct_extract(zd, coalesce(exog_cols[4]::VARCHAR, exog_cols[1]::VARCHAR, y_col))::DOUBLE
        WHEN 5  THEN struct_extract(zd, coalesce(exog_cols[5]::VARCHAR, exog_cols[1]::VARCHAR, y_col))::DOUBLE
        WHEN 6  THEN struct_extract(zd, coalesce(exog_cols[6]::VARCHAR, exog_cols[1]::VARCHAR, y_col))::DOUBLE
        WHEN 7  THEN struct_extract(zd, coalesce(exog_cols[7]::VARCHAR, exog_cols[1]::VARCHAR, y_col))::DOUBLE
        WHEN 8  THEN struct_extract(zd, coalesce(exog_cols[8]::VARCHAR, exog_cols[1]::VARCHAR, y_col))::DOUBLE
        WHEN 9  THEN struct_extract(zd, coalesce(exog_cols[9]::VARCHAR, exog_cols[1]::VARCHAR, y_col))::DOUBLE
        WHEN 10 THEN struct_extract(zd, coalesce(exog_cols[10]::VARCHAR, exog_cols[1]::VARCHAR, y_col))::DOUBLE
        WHEN 11 THEN struct_extract(zd, coalesce(exog_cols[11]::VARCHAR, exog_cols[1]::VARCHAR, y_col))::DOUBLE
        WHEN 12 THEN struct_extract(zd, coalesce(exog_cols[12]::VARCHAR, exog_cols[1]::VARCHAR, y_col))::DOUBLE
        ELSE error('sarimax: at most 12 exogenous columns are supported')
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

-- ---- the fit --------------------------------------------------------------------

CREATE OR REPLACE MACRO sarimax_fit(data, y_col, p, d, q,
                                    sp := 0, sd := 0, sq := 0, s := 1,
                                    exog_cols := []::VARCHAR[], t_col := NULL) AS TABLE
WITH _sarimax_f_chk AS (
    SELECT _sarimax_validate_spec(p, d, q, sp, sd, sq, s) AS ok
),
_sarimax_f_series AS (
    SELECT t, y FROM _sarimax_series_of(data, y_col, t_col)
),
_sarimax_f_exog AS (
    SELECT t, j, x FROM _sarimax_exog_of(data, exog_cols, y_col, t_col)
),
_sarimax_f_w AS (
    SELECT t, w FROM _sarimax_diff('_sarimax_f_series', 't', 'y', d, sd, s)
),
_sarimax_f_exd AS (
    SELECT t, j, x FROM _sarimax_diff_exog('_sarimax_f_exog', d, sd, s)
),
_sarimax_f_fit AS (
    SELECT * FROM _sarimax_bfgs('_sarimax_f_w', '_sarimax_f_exd',
                                len(exog_cols), p, q, sp, sq, greatest(s, 1))
),
_sarimax_f_dims AS (
    SELECT len(exog_cols)::INT AS r,
           (SELECT count(*) FROM _sarimax_f_series)::INT AS n,
           (SELECT count(*) FROM _sarimax_f_w)::INT AS n_eff,
           (len(exog_cols) + p + q + sp + sq + 1)::INT AS k_params
    FROM _sarimax_f_chk
    WHERE ok
),
_sarimax_f_probe AS (
    SELECT 1::BIGINT AS probe_id, params FROM _sarimax_f_fit
),
_sarimax_f_sys AS (
    SELECT * FROM _sarimax_systems('_sarimax_f_probe', len(exog_cols), p, q, sp, sq, greatest(s, 1))
),
_sarimax_f_obs AS (
    SELECT * FROM _sarimax_obs_adj('_sarimax_f_w', '_sarimax_f_exd', '_sarimax_f_probe')
),
_sarimax_f_state AS (
    SELECT * FROM _sarimax_kfilter_state('_sarimax_f_obs', '_sarimax_f_sys')
),
_sarimax_f_lists AS (
    SELECT (SELECT list(w ORDER BY t) FROM _sarimax_f_w) AS wlist,
           (SELECT CASE WHEN len(exog_cols) = 0
                        THEN (SELECT list([]::DOUBLE[]) FROM _sarimax_f_w)
                        ELSE (SELECT list(zxr ORDER BY zt)
                              FROM (SELECT t AS zt, list(x ORDER BY j) AS zxr
                                    FROM _sarimax_f_exd GROUP BY t))
                   END) AS xmat
),
_sarimax_f_bse AS (
    SELECT bse FROM _sarimax_bse(
        (SELECT params FROM _sarimax_f_fit),
        (SELECT wlist FROM _sarimax_f_lists),
        (SELECT xmat FROM _sarimax_f_lists),
        len(exog_cols), p, q, sp, sq, greatest(s, 1))
),
_sarimax_f_names AS (
    SELECT list_transform(range(1, dm.k_params + 1), lambda zi:
        CASE
        WHEN zi <= dm.r THEN exog_cols[zi]
        WHEN zi <= dm.r + p THEN 'ar.L' || (zi - dm.r)::VARCHAR
        WHEN zi <= dm.r + p + q THEN 'ma.L' || (zi - dm.r - p)::VARCHAR
        WHEN zi <= dm.r + p + q + sp THEN 'ar.S.L' || ((zi - dm.r - p - q) * s)::VARCHAR
        WHEN zi <= dm.r + p + q + sp + sq THEN 'ma.S.L' || ((zi - dm.r - p - q - sp) * s)::VARCHAR
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
     LATERAL (SELECT unnest(['p','d','q','sp','sd','sq','s','r','n','n_eff']) AS name,
                     unnest(range(1, 11)) AS i,
                     unnest([p::DOUBLE, d::DOUBLE, q::DOUBLE, sp::DOUBLE, sd::DOUBLE,
                             sq::DOUBLE, s::DOUBLE, dm.r::DOUBLE, dm.n::DOUBLE,
                             dm.n_eff::DOUBLE]) AS v) zsp
UNION ALL
SELECT 'meta', zm.name, zm.i::INT, zm.v, NULL
FROM _sarimax_f_fit ft, _sarimax_f_dims dm,
     LATERAL (SELECT unnest(['loglik','aic','bic','converged','iterations',
                             'grad_norm','restarted','sigma2']) AS name,
                     unnest(range(1, 9)) AS i,
                     unnest([ft.loglik,
                             2e0 * dm.k_params - 2e0 * ft.loglik,
                             dm.k_params * ln(dm.n_eff::DOUBLE) - 2e0 * ft.loglik,
                             CASE WHEN ft.converged THEN 1e0 ELSE 0e0 END,
                             ft.iterations::DOUBLE,
                             ft.grad_norm,
                             CASE WHEN ft.restarted THEN 1e0 ELSE 0e0 END,
                             ft.params[dm.k_params]]) AS v) zm
UNION ALL
SELECT 'anchor', 'endog:' || za.stage::VARCHAR, za.idx::INT, za.value, NULL
FROM _sarimax_diff_anchors('_sarimax_f_series', 't', 'y', d, sd, s) za
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

-- Recompute the innovation trace at theta-hat from the data + model table.
CREATE OR REPLACE MACRO _sarimax_retrace(model, data, y_col, exog_cols, t_col) AS TABLE
WITH _sarimax_rt_chk AS (
    SELECT _sarimax_check_exog_names(model, exog_cols) AS ok
),
_sarimax_rt_series AS (
    SELECT zs.t, zs.y FROM _sarimax_series_of(data, y_col, t_col) zs, _sarimax_rt_chk zc
    WHERE zc.ok
),
_sarimax_rt_exog AS (
    SELECT t, j, x FROM _sarimax_exog_of(data, exog_cols, y_col, t_col)
),
_sarimax_rt_w AS (
    SELECT t, w FROM _sarimax_diff_dyn('_sarimax_rt_series', 't', 'y',
        _sarimax_m_spec(model, 'd'),
        _sarimax_m_spec(model, 'sd'),
        _sarimax_m_spec(model, 's'))
),
_sarimax_rt_exd AS (
    SELECT t, j, x FROM _sarimax_diff_exog_dyn('_sarimax_rt_exog',
        _sarimax_m_spec(model, 'd'),
        _sarimax_m_spec(model, 'sd'),
        _sarimax_m_spec(model, 's'))
),
_sarimax_rt_probe AS (
    SELECT 1::BIGINT AS probe_id, _sarimax_m_params(model) AS params
),
_sarimax_rt_sys AS (
    SELECT * FROM _sarimax_systems('_sarimax_rt_probe',
        _sarimax_m_spec(model, 'r')::INT,
        _sarimax_m_spec(model, 'p')::INT,
        _sarimax_m_spec(model, 'q')::INT,
        _sarimax_m_spec(model, 'sp')::INT,
        _sarimax_m_spec(model, 'sq')::INT,
        greatest(_sarimax_m_spec(model, 's')::INT, 1))
),
_sarimax_rt_obs AS (
    SELECT * FROM _sarimax_obs_adj('_sarimax_rt_w', '_sarimax_rt_exd', '_sarimax_rt_probe')
)
SELECT t, v, f, v / sqrt(f) AS std_resid
FROM _sarimax_kfilter('_sarimax_rt_obs', '_sarimax_rt_sys');

CREATE OR REPLACE MACRO sarimax_residuals(model, data, y_col, exog_cols := []::VARCHAR[], t_col := NULL) AS TABLE
SELECT t, v, f, std_resid
FROM _sarimax_retrace(model, data, y_col, exog_cols, t_col)
ORDER BY t;

-- Ljung-Box Q on the standardized innovations at the exact lag set 1..nlags.
CREATE OR REPLACE MACRO sarimax_ljungbox(model, data, y_col, nlags, exog_cols := []::VARCHAR[], t_col := NULL) AS TABLE
WITH _sarimax_lb_sr AS (
    SELECT t, std_resid FROM _sarimax_retrace(model, data, y_col, exog_cols, t_col)
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

CREATE OR REPLACE MACRO sarimax_evaluate(model, data, y_col, exog_cols := []::VARCHAR[], t_col := NULL) AS TABLE
SELECT _sarimax_m_meta(model, 'loglik') AS loglik,
       _sarimax_m_meta(model, 'aic') AS aic,
       _sarimax_m_meta(model, 'bic') AS bic,
       _sarimax_m_meta(model, 'sigma2') AS sigma2,
       _sarimax_m_spec(model, 'n_eff') AS n_eff,
       (SELECT 1e0 - (count(*) FILTER (WHERE NOT isfinite(std_resid)))::DOUBLE / count(*)
        FROM _sarimax_retrace(model, data, y_col, exog_cols, t_col)) AS resid_finite_frac,
       _sarimax_m_meta(model, 'converged') AS converged;

-- ---- forecast -----------------------------------------------------------------------

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
    SELECT _sarimax_m_spec(model, 'r')::INT AS r,
           _sarimax_m_spec(model, 'p')::INT AS p,
           _sarimax_m_spec(model, 'q')::INT AS q,
           _sarimax_m_spec(model, 'sp')::INT AS sp,
           _sarimax_m_spec(model, 'sd')::INT AS sd,
           _sarimax_m_spec(model, 'sq')::INT AS sq,
           _sarimax_m_spec(model, 's')::INT AS s,
           _sarimax_m_spec(model, 'd')::INT AS d,
           _sarimax_m_spec(model, 'n_eff')::BIGINT AS n_eff
    FROM _sarimax_fc_chk
    WHERE ok
),
_sarimax_fc_probe AS MATERIALIZED (
    SELECT 1::BIGINT AS probe_id, _sarimax_m_params(model) AS params
),
_sarimax_fc_sys AS MATERIALIZED (
    SELECT * FROM _sarimax_systems('_sarimax_fc_probe',
        _sarimax_m_spec(model, 'r')::INT,
        _sarimax_m_spec(model, 'p')::INT,
        _sarimax_m_spec(model, 'q')::INT,
        _sarimax_m_spec(model, 'sp')::INT,
        _sarimax_m_spec(model, 'sq')::INT,
        greatest(_sarimax_m_spec(model, 's')::INT, 1))
),
_sarimax_fc_state AS MATERIALIZED (
    SELECT 1::BIGINT AS probe_id,
           (SELECT n_eff FROM _sarimax_fc_dims) AS n_eff,
           (SELECT value_list FROM query_table(model) WHERE kind = 'state' AND name = 'a') AS a,
           (SELECT value_list FROM query_table(model) WHERE kind = 'state' AND name = 'P') AS p,
           _sarimax_m_meta(model, 'loglik') AS loglik
),
-- future exog: trailing in-sample anchors ++ newdata, differenced; rows re-densify
-- to exactly h future differenced rows
_sarimax_fc_exfull AS (
    SELECT za.j, za.ai AS tt, za.value AS x, 0 AS grp
    FROM (SELECT split_part(name, ':', 2)::INT AS j, idx AS ai, value
          FROM query_table(model) WHERE kind = 'anchor' AND name LIKE 'exog:%') za
    UNION ALL
    SELECT zx.j,
           (SELECT max(zd2.d + zd2.sd * zd2.s) FROM _sarimax_fc_dims zd2) + zx.t,
           zx.x, 1
    FROM _sarimax_exog_of(coalesce(newdata, data), exog_cols, y_col, t_col) zx
),
_sarimax_fc_exfull2 AS (
    SELECT tt AS t, j, x FROM _sarimax_fc_exfull
),
_sarimax_fc_exd AS MATERIALIZED (
    SELECT zz.t, zz.j, zz.x
    FROM _sarimax_diff_exog_dyn('_sarimax_fc_exfull2',
        (SELECT d FROM _sarimax_fc_dims),
        (SELECT sd FROM _sarimax_fc_dims),
        (SELECT s FROM _sarimax_fc_dims)) zz
),
_sarimax_fc_dfut AS MATERIALIZED (
    SELECT 1::BIGINT AS probe_id, zed.t AS h,
           list_reduce(
               list_prepend(0e0, list(zed.x * zp.params[zed.j] ORDER BY zed.j)),
               lambda zacc, zxb: zacc + zxb) AS d
    FROM _sarimax_fc_exd zed, _sarimax_fc_probe zp
    GROUP BY zed.t, zp.params
),
_sarimax_fc_diffres AS MATERIALIZED (
    SELECT * FROM _sarimax_fc_diff('_sarimax_fc_state', '_sarimax_fc_sys', '_sarimax_fc_dfut', h)
),
_sarimax_fc_anch AS MATERIALIZED (
    SELECT split_part(name, ':', 2)::INT AS stage, idx, value
    FROM query_table(model) WHERE kind = 'anchor' AND name LIKE 'endog:%'
),
_sarimax_fc_orig_res AS (
    SELECT * FROM _sarimax_fc_orig('_sarimax_fc_diffres', '_sarimax_fc_anch',
        (SELECT d FROM _sarimax_fc_dims),
        (SELECT sd FROM _sarimax_fc_dims),
        (SELECT s FROM _sarimax_fc_dims), h)
)
SELECT zdf.h,
       zor.mean_orig AS yhat,
       sqrt(zor.var_orig) AS se,
       zor.mean_orig - _sarimax_norm_ppf(0.5e0 + level / 2e0) * sqrt(zor.var_orig) AS lo,
       zor.mean_orig + _sarimax_norm_ppf(0.5e0 + level / 2e0) * sqrt(zor.var_orig) AS hi,
       zdf.mean_diff AS yhat_diff,
       sqrt(zdf.var_diff) AS se_diff
FROM _sarimax_fc_diffres zdf
JOIN _sarimax_fc_orig_res zor
  ON zor.probe_id = zdf.probe_id AND zor.h = zdf.h
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
