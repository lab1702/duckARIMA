-- ============================================================================
-- duckARIMA Layer 4: estimation (spec section 5.4).
--
-- This file provides, in order:
--   1. the stationarity/invertibility parameter transform (Monahan/Jones PACF
--      bijection), replicating statsmodels' transform_params /
--      untransform_params exactly (a T1-class requirement);
--   2. starting values (OLS beta + Hannan-Rissanen)          [added later]
--   3. the BFGS optimizer                                     [added later]
--
-- Transform conventions, matched to statsmodels 0.14.6
-- (statsmodels.tsa.statespace.tools.constrain_stationary_univariate):
--   * per block (phi | theta | Phi | Theta), independently:
--       r_k = x_k / sqrt(1 + x_k^2)          (partial autocorrelations)
--       Durbin-Levinson: y_K[i] = y_{K-1}[i] + r_K * y_{K-1}[K-i], y_K[K] = r_K
--       AR blocks:  constrained = -y_n   (note the negation)
--       MA blocks:  constrained = +y_n
--   * beta block: identity (regression coefficients unconstrained)
--   * sigma2: constrained = x^2, untransformed = sqrt(sigma2)
--     (statsmodels' convention; the spec's log-space note loses to fixture
--     agreement, documented in GUIDE.md conventions)
-- ============================================================================

-- ---- one block: unconstrained -> stationary-region coefficients -------------

-- Durbin-Levinson fold given precomputed PACFs. Returns y_n (the raw recursion
-- output); callers negate for AR blocks.
CREATE OR REPLACE MACRO _sarimax_dl_forward(rv) AS (
    CASE WHEN len(rv) = 0 THEN []::DOUBLE[]
    ELSE list_reduce(
        list_prepend(
            [rv[1]],
            list_transform(range(2, len(rv) + 1),
                           lambda k2: [rv[k2]])),
        lambda acc, e:
            list_transform(range(1, len(acc) + 2), lambda i2:
                CASE WHEN i2 = len(acc) + 1 THEN e[1]
                     ELSE acc[i2] + e[1] * acc[len(acc) + 1 - i2] END))
    END
);

-- Unconstrained block -> constrained block. sgn = -1.0 for AR, +1.0 for MA.
CREATE OR REPLACE MACRO _sarimax_constrain_block(x, sgn) AS (
    list_transform(
        _sarimax_dl_forward(list_transform(x, lambda xi: xi / sqrt(1.0::DOUBLE + xi * xi))),
        lambda c: sgn * c)
);

-- ---- one block: constrained -> unconstrained ---------------------------------

-- Inverse Durbin-Levinson: peel one row per fold step (n steps in total),
-- collecting the diagonal r_K at each level; the accumulator carries
-- {rw, rs}. Input is y_n (callers pre-negate AR blocks). After n peels the
-- row is empty and rs holds r_n..r_1; reversed on the way out to r_1..r_n.
CREATE OR REPLACE MACRO _sarimax_dl_backward(yn) AS (
    CASE WHEN len(yn) = 0 THEN []::DOUBLE[]
    ELSE list_reverse((
        list_reduce(
            list_prepend(
                struct_pack(rw := yn, rs := []::DOUBLE[]),
                list_transform(range(1, len(yn) + 1), lambda z:
                    struct_pack(rw := []::DOUBLE[], rs := []::DOUBLE[]))),
            lambda acc, e:
                struct_pack(
                    rw := list_transform(range(1, len(acc.rw)), lambda i2:
                        (acc.rw[i2] - acc.rw[len(acc.rw)] * acc.rw[len(acc.rw) - i2])
                        / (1.0::DOUBLE - acc.rw[len(acc.rw)] * acc.rw[len(acc.rw)])),
                    rs := list_append(acc.rs, acc.rw[len(acc.rw)])))
        ).rs)
    END
);

-- Constrained block -> unconstrained block. sgn = -1.0 for AR, +1.0 for MA
-- (the same sign used to constrain; applied on the way in).
CREATE OR REPLACE MACRO _sarimax_unconstrain_block(c, sgn) AS (
    list_transform(
        _sarimax_dl_backward(list_transform(c, lambda ci: sgn * ci)),
        lambda rk: rk / sqrt(1.0::DOUBLE - rk * rk))
);

-- ---- full parameter vector ----------------------------------------------------

-- Unconstrained -> constrained, canonical order (beta, phi, theta, Phi, Theta,
-- sigma2). Mirrors statsmodels transform_params.
CREATE OR REPLACE MACRO _sarimax_transform_params(u, r, p, q, bigp, bigq) AS (
    list_slice(u, 1, r)
    || _sarimax_constrain_block(list_slice(u, r + 1, r + p), -1.0::DOUBLE)
    || _sarimax_constrain_block(list_slice(u, r + p + 1, r + p + q), 1.0::DOUBLE)
    || _sarimax_constrain_block(list_slice(u, r + p + q + 1, r + p + q + bigp), -1.0::DOUBLE)
    || _sarimax_constrain_block(list_slice(u, r + p + q + bigp + 1, r + p + q + bigp + bigq), 1.0::DOUBLE)
    || [u[r + p + q + bigp + bigq + 1] * u[r + p + q + bigp + bigq + 1]]
);

-- Constrained -> unconstrained. Mirrors statsmodels untransform_params.
CREATE OR REPLACE MACRO _sarimax_untransform_params(c, r, p, q, bigp, bigq) AS (
    list_slice(c, 1, r)
    || _sarimax_unconstrain_block(list_slice(c, r + 1, r + p), -1.0::DOUBLE)
    || _sarimax_unconstrain_block(list_slice(c, r + p + 1, r + p + q), 1.0::DOUBLE)
    || _sarimax_unconstrain_block(list_slice(c, r + p + q + 1, r + p + q + bigp), -1.0::DOUBLE)
    || _sarimax_unconstrain_block(list_slice(c, r + p + q + bigp + 1, r + p + q + bigp + bigq), 1.0::DOUBLE)
    || [sqrt(c[r + p + q + bigp + bigq + 1])]
);


-- ============================================================================
-- SECTION 2: estimation engine (appended; spec 5.4, milestones M4/M5).
--
--   2a. scalar loglikelihood kernel  _sarimax_ll_c / _sarimax_ll_x
--   2b. starting values              _sarimax_start_params
--   2c. BFGS optimizer               _sarimax_bfgs
--   2d. standard errors              _sarimax_bse
--
-- DOCUMENTED DEVIATION (P1 initialization inside the optimizer kernel): the
-- spec (5.3) pins the vec-trick Lyapunov solve for the FILTER's stationary
-- initialization, which sql/03_filter.sql honors. The scalar kernel below is
-- an internal evaluation path for the optimizer, where the k^2-by-k^2 linear
-- solve per probe is too slow; it instead computes P1 by 30 fixed DOUBLING
-- iterations (S <- S + A S A', A <- A A, starting from S = RQR', A = T),
-- which equals the exact stationary covariance to < 1e-12 for any spectral
-- radius <= 1 - 1e-6 (the truncation term is rho^(2^31)). Explosive probes
-- diverge to inf/NaN, the loglikelihood poisons to NULL, and the line search
-- rejects them -- exactly the TRY-style guard of spec 4.3. Kernel-vs-filter
-- agreement is asserted at 1e-9 abs / 1e-11 rel in tests/test_estimate.py,
-- and the optimum agrees with statsmodels at T2 tolerance.
--
-- All lambda variables in this section are z-prefixed (codebase reservation);
-- list-valued macro arguments referenced inside lambdas (wlist, xmat) must be
-- passed as materialized values/columns, never as expressions (trap: textual
-- macro expansion re-evaluates argument expressions per lambda element).
--
-- NEW TRAP DISCOVERED HERE (DuckDB 1.5.4): inside a correlated LATERAL
-- subquery, a lambda body may reference either correlated outer columns or
-- local columns -- but not resolve local ones once any correlated column is
-- referenced anywhere in the lambda ("Referenced table ... not found").
-- Workaround used throughout: pre-project every needed outer column into the
-- innermost derived table of the LATERAL (plain, non-lambda projections),
-- so lambdas only ever see local columns.
-- ============================================================================

-- ---- 2a. scalar loglikelihood kernel -----------------------------------------

-- Loglikelihood at a CONSTRAINED parameter vector cpar (canonical order:
-- beta, phi, theta, Phi, Theta, sigma2), over the differenced series wlist
-- (DOUBLE[], t order) and differenced exog rows xmat (DOUBLE[][]: n lists of
-- length r; pass []::DOUBLE[][] when r = 0). Reproduces sql/03_filter.sql's
-- recursive-member arithmetic operation for operation (same fold order, same
-- guards); P1 by doubling (see section header). Returns DOUBLE, NULL when
-- poisoned (F_t <= 0, or a non-finite loglikelihood from an explosive probe).
CREATE OR REPLACE MACRO _sarimax_ll_c(cpar, wlist, xmat, r, p, q, bigp, bigq, s) AS (
  (list_transform([struct_pack(
      zk := _sarimax_k_states(p, q, bigp, bigq, s),
      zn := len(wlist),
      zbeta := list_slice(cpar, 1, r),
      zphistar := _sarimax_expand_ar(list_slice(cpar, r + 1, r + p),
                                     list_slice(cpar, r + p + q + 1, r + p + q + bigp), s),
      zthetastar := _sarimax_expand_ma(list_slice(cpar, r + p + 1, r + p + q),
                                       list_slice(cpar, r + p + q + bigp + 1,
                                                  r + p + q + bigp + bigq), s),
      zsigma2 := cpar[r + p + q + bigp + bigq + 1])],
   lambda zl1:
     (list_transform([struct_pack(
         ztm := _sarimax_build_t(zl1.zphistar, zl1.zk),
         zrv := _sarimax_build_r(zl1.zthetastar, zl1.zk),
         -- yd_t = w_t - x~_t' beta, ordered-j fold seeded 0e0 (matches
         -- _sarimax_obs_adj's list_reduce(list_prepend(0.0, ...)) exactly)
         zyd := list_transform(range(1, zl1.zn + 1), lambda zt:
                    wlist[zt] - list_reduce(
                        list_prepend(0e0, list_transform(range(1, r + 1), lambda zj:
                            xmat[zt][zj] * (zl1.zbeta)[zj])),
                        lambda za, zb: za + zb)))],
      lambda zl2:
        (list_transform([struct_pack(
            ztt := _sarimax_mtrans(zl2.ztm, zl1.zk, zl1.zk),
            zrqr := _sarimax_build_rqr(zl2.zrv, zl1.zsigma2, zl1.zk))],
         lambda zl3:
           -- P1: 30 doubling iterations (bind the fold result BEFORE msym --
           -- a lambda-dependent capture is re-evaluated per element, so msym
           -- over the raw fold expression would run the fold 2k^2 times)
           (list_transform([(list_reduce(
                    [struct_pack(zsm := zl3.zrqr, zaa := zl2.ztm)]
                      || list_transform(range(1, 31), lambda zd2:
                           struct_pack(zsm := []::DOUBLE[], zaa := []::DOUBLE[])),
                    lambda zacc, zel:
                      (list_transform([_sarimax_mmul(zacc.zaa, zacc.zsm, zl1.zk, zl1.zk, zl1.zk)],
                       lambda zas:
                         (list_transform([_sarimax_mtrans(zacc.zaa, zl1.zk, zl1.zk)],
                          lambda zat2:
                            (list_transform([_sarimax_mmul(zas, zat2, zl1.zk, zl1.zk, zl1.zk)],
                             lambda zasat:
                               struct_pack(
                                 zsm := _sarimax_madd(zacc.zsm, zasat),
                                 zaa := _sarimax_mmul(zacc.zaa, zacc.zaa,
                                                      zl1.zk, zl1.zk, zl1.zk))))[1]
                         ))[1]
                      ))[1]
                 )).zsm],
            lambda zs30:
           (list_transform([_sarimax_msym(zs30, zl1.zk)],
            lambda zp1:
              -- Kalman fold in strict t order; accumulator and elements share
              -- one struct type, per-step yd rides in the spare zydv field
              (list_transform([(list_reduce(
                 [struct_pack(za2 := _sarimax_mzeros(zl1.zk, 1), zp2 := zp1,
                              zll := 0e0, zydv := 0e0)]
                   || list_transform(zl2.zyd, lambda zydt:
                        struct_pack(za2 := []::DOUBLE[], zp2 := []::DOUBLE[],
                                    zll := 0e0, zydv := zydt)),
                 lambda zacc, zel:
                   (list_transform([struct_pack(
                        zv := zel.zydv - (zacc.za2)[1],
                        zf := (zacc.zp2)[1],
                        ztp := _sarimax_mmul(zl2.ztm, zacc.zp2, zl1.zk, zl1.zk, zl1.zk),
                        zta := _sarimax_mmul(zl2.ztm, zacc.za2, zl1.zk, zl1.zk, 1))],
                    lambda zi1:
                      (list_transform([list_transform(range(1, zl1.zk + 1), lambda zi2:
                                           (zi1.ztp)[(zi2 - 1) * zl1.zk + 1])],
                       lambda ztpz:
                         -- bind TP T' and the gain outer product once each,
                         -- then the pre-symmetrization update once, so msym
                         -- sees only bound values (no per-element recompute)
                         (list_transform([struct_pack(
                              ztpt := _sarimax_mmul(zi1.ztp, zl3.ztt, zl1.zk, zl1.zk, zl1.zk),
                              zoutr := list_transform(range(1, zl1.zk * zl1.zk + 1), lambda zi4:
                                           ztpz[(zi4 - 1) // zl1.zk + 1]
                                           * ztpz[(zi4 - 1) % zl1.zk + 1] / zi1.zf))],
                          lambda zb1:
                            (list_transform([_sarimax_madd(
                                                 _sarimax_msub(zb1.ztpt, zb1.zoutr),
                                                 zl3.zrqr)],
                             lambda zpu:
                               struct_pack(
                                 za2 := list_transform(range(1, zl1.zk + 1), lambda zi3:
                                            (zi1.zta)[zi3] + ztpz[zi3] * zi1.zv / zi1.zf),
                                 zp2 := _sarimax_msym(zpu, zl1.zk),
                                 zll := zacc.zll + CASE WHEN zi1.zf > 0e0
                                                        THEN -0.5e0 * (ln(2e0 * pi())
                                                                       + ln(zi1.zf)
                                                                       + zi1.zv * zi1.zv / zi1.zf)
                                                        ELSE NULL END,
                                 zydv := 0e0)))[1]
                         ))[1]
                      ))[1]
                   ))[1]
              )).zll],
               lambda zllf:
                 CASE WHEN isfinite(zllf) THEN zllf ELSE NULL END))[1]
           ))[1]
           ))[1]
        ))[1]
     ))[1]
  ))[1]
);

-- Loglikelihood at an UNCONSTRAINED parameter vector (the optimizer's
-- objective): transform, then evaluate. NULL when poisoned.
CREATE OR REPLACE MACRO _sarimax_ll_x(xunc, wlist, xmat, r, p, q, bigp, bigq, s) AS (
  _sarimax_ll_c(_sarimax_transform_params(xunc, r, p, q, bigp, bigq),
                wlist, xmat, r, p, q, bigp, bigq, s)
);

-- ---- 2b. starting values (spec 5.4, two-stage) --------------------------------

-- One row (x0 DOUBLE[] unconstrained, params0 DOUBLE[] constrained).
--   1. beta0 by OLS of w on the differenced exog (normal equations); residuals
--      e_t = w_t - x~_t' beta0. Gram/moment sums use ordered folds, NOT plain
--      SUM: start values feed the optimizer whose bitwise thread-count
--      determinism is asserted by the acceptance suite, and a parallel SUM
--      over >1 morsel is not order-deterministic.
--   2. Hannan-Rissanen: long AR(m) OLS on e_t with
--      m = greatest(20, 2*greatest(p + s*P, q + s*Q)) capped at (n_eff-1)//2
--      (skipped when q + Q = 0: no MA block needs innovation estimates);
--      residuals epshat_t; then e_t regressed on
--      [e lags 1..p | e lags s..P*s | epshat lags 1..q | epshat lags s..Q*s].
--   3. sigma2_0 = mean squared stage-2 residual, guarded positive (fallback
--      mean(w^2), then 1e0).
--   4. Shrink-until-valid: the four ARMA blocks are jointly scaled by 0.5^z,
--      z = 0..60, first z whose untransform is entirely finite wins
--      (TRY() absorbs the sqrt-of-negative error DuckDB raises outside the
--      stationary/invertible region).
-- Named failure (spec 5.5): errors when n_eff is too small for the stage-2
-- regression, with the minimum n_eff stated.
CREATE OR REPLACE MACRO _sarimax_start_params(w_tbl, exog_diff_tbl, r, p, q, bigp, bigq, s) AS TABLE
WITH _sarimax_sp_w AS (
    SELECT count(*)::BIGINT AS zn FROM query_table(w_tbl)
),
_sarimax_sp_dims AS (
    SELECT zn,
           CASE WHEN q + bigq = 0 THEN 0::BIGINT
                ELSE least(greatest(20, 2 * greatest(p + s * bigp, q + s * bigq))::BIGINT,
                           (zn - 1) // 2) END AS zm,
           (p + q + bigp + bigq)::BIGINT AS zncoef2
    FROM _sarimax_sp_w
),
_sarimax_sp_dims3 AS (
    SELECT zn, zm, zncoef2,
           CASE WHEN zncoef2 > 0 AND zn < zt0raw + zncoef2
                THEN error(concat('duckARIMA start values: n_eff = ', zn,
                                  ' is too small for the Hannan-Rissanen stage-2 regression;',
                                  ' need n_eff >= ', zt0raw + zncoef2,
                                  ' (long-AR order m = ', zm, ')'))
                ELSE zt0raw END AS zt0
    FROM (
        SELECT zn, zm, zncoef2,
               greatest(p + s * bigp + 1,
                        CASE WHEN q + bigq > 0 THEN zm + 1 + q + s * bigq ELSE 1 END)::BIGINT AS zt0raw
        FROM _sarimax_sp_dims
    )
),
_sarimax_sp_beta AS (
    SELECT CASE WHEN r = 0 THEN []::DOUBLE[]
                WHEN NOT (zsol).ok
                THEN error('duckARIMA start values: singular exog Gram matrix in the beta0 OLS')
                ELSE (zsol).x END AS zbeta
    FROM (
        SELECT _sarimax_solve_list(zaug, r, 1) AS zsol
        FROM (
            SELECT list(zv ORDER BY zj1, zj2) AS zaug
            FROM (
                SELECT za.j::BIGINT AS zj1, zb.j::BIGINT AS zj2, list_reduce(list_prepend(0e0, list(za.x * zb.x ORDER BY za.t)),
                       lambda zfa, zfb: zfa + zfb) AS zv
                FROM query_table(exog_diff_tbl) za
                JOIN query_table(exog_diff_tbl) zb ON za.t = zb.t
                GROUP BY za.j, zb.j
                UNION ALL
                SELECT ze.j::BIGINT, (r + 1)::BIGINT, list_reduce(list_prepend(0e0, list(ze.x * zw.w ORDER BY ze.t)),
                       lambda zfa, zfb: zfa + zfb)
                FROM query_table(exog_diff_tbl) ze
                JOIN query_table(w_tbl) zw ON ze.t = zw.t
                GROUP BY ze.j
            )
        )
    )
),
_sarimax_sp_e AS (
    SELECT zw.t::BIGINT AS zt, zw.w - coalesce(zdi.zd, 0e0) AS zev
    FROM query_table(w_tbl) zw
    LEFT JOIN (
        SELECT ze.t, list_reduce(list_prepend(0e0, list(ze.x * zbc.zbeta[ze.j] ORDER BY ze.j)),
                                 lambda za3, zb3: za3 + zb3) AS zd
        FROM query_table(exog_diff_tbl) ze, _sarimax_sp_beta zbc
        GROUP BY ze.t, zbc.zbeta
    ) zdi ON zdi.t = zw.t
),
_sarimax_sp_d1 AS (       -- long-AR design: rows t = m+1..n_eff, lag l = 1..m
    SELECT ze1.zt, zlx.zl, ze2.zev AS zx
    FROM _sarimax_sp_e ze1
    CROSS JOIN _sarimax_sp_dims3 zd3
    CROSS JOIN LATERAL (SELECT zu.zl FROM unnest(range(1, zd3.zm + 1)) AS zu(zl)) zlx
    JOIN _sarimax_sp_e ze2 ON ze2.zt = ze1.zt - zlx.zl
    WHERE ze1.zt > zd3.zm
),
_sarimax_sp_a1 AS (
    SELECT CASE WHEN zd3.zm = 0 THEN []::DOUBLE[]
                WHEN NOT (zs1.zsol).ok
                THEN list_transform(range(1, zd3.zm + 1), lambda zz3: 0e0)
                ELSE (zs1.zsol).x END AS za1
    FROM _sarimax_sp_dims3 zd3,
         (SELECT _sarimax_solve_list(zg1.zaug, zdi.zm, 1) AS zsol
          FROM _sarimax_sp_dims3 zdi,
               (SELECT list(zv ORDER BY zj1, zj2) AS zaug
                FROM (
                    SELECT za.zl AS zj1, zb.zl AS zj2, list_reduce(list_prepend(0e0, list(za.zx * zb.zx ORDER BY za.zt)),
                       lambda zfa, zfb: zfa + zfb) AS zv
                    FROM _sarimax_sp_d1 za JOIN _sarimax_sp_d1 zb ON za.zt = zb.zt
                    GROUP BY za.zl, zb.zl
                    UNION ALL
                    SELECT za.zl, zmx.zm + 1, list_reduce(list_prepend(0e0, list(za.zx * ze.zev ORDER BY za.zt)),
                       lambda zfa, zfb: zfa + zfb)
                    FROM _sarimax_sp_d1 za
                    JOIN _sarimax_sp_e ze ON ze.zt = za.zt
                    CROSS JOIN _sarimax_sp_dims3 zmx
                    GROUP BY za.zl, zmx.zm
                )) zg1) zs1
),
_sarimax_sp_eps AS (      -- epshat_t for t = m+1..n_eff
    SELECT ze1.zt, ze1.zev - coalesce(zli.zsum, 0e0) AS zeps
    FROM _sarimax_sp_e ze1
    CROSS JOIN _sarimax_sp_dims3 zd3
    LEFT JOIN (
        SELECT za.zt, list_reduce(list_prepend(0e0, list(za.zx * za1c.za1[za.zl] ORDER BY za.zl)),
                                  lambda za4, zb4: za4 + zb4) AS zsum
        FROM _sarimax_sp_d1 za, _sarimax_sp_a1 za1c
        GROUP BY za.zt, za1c.za1
    ) zli ON zli.zt = ze1.zt
    WHERE ze1.zt > zd3.zm
),
_sarimax_sp_cols AS (     -- stage-2 column spec, canonical block layout
    SELECT zcx.zc,
           CASE WHEN zcx.zc <= p + bigp THEN 1 ELSE 2 END AS zsrc,
           CASE WHEN zcx.zc <= p THEN zcx.zc
                WHEN zcx.zc <= p + bigp THEN (zcx.zc - p) * s
                WHEN zcx.zc <= p + bigp + q THEN zcx.zc - p - bigp
                ELSE (zcx.zc - p - bigp - q) * s END AS zlag
    FROM (SELECT unnest(range(1, p + q + bigp + bigq + 1)) AS zc) zcx
),
_sarimax_sp_d2 AS (       -- stage-2 design: rows t = t0..n_eff
    SELECT zrow.zt, zc.zc, CASE WHEN zc.zsrc = 1 THEN zej.zev ELSE zpj.zeps END AS zx
    FROM (SELECT ze.zt FROM _sarimax_sp_e ze, _sarimax_sp_dims3 zd3 WHERE ze.zt >= zd3.zt0) zrow
    CROSS JOIN _sarimax_sp_cols zc
    LEFT JOIN _sarimax_sp_e zej ON zc.zsrc = 1 AND zej.zt = zrow.zt - zc.zlag
    LEFT JOIN _sarimax_sp_eps zpj ON zc.zsrc = 2 AND zpj.zt = zrow.zt - zc.zlag
),
_sarimax_sp_coef AS (     -- stage-2 coefficients in COLUMN order (phi,Phi,theta,Theta)
    SELECT CASE WHEN p + q + bigp + bigq = 0 THEN []::DOUBLE[]
                WHEN NOT (zs2.zsol).ok
                THEN list_transform(range(1, p + q + bigp + bigq + 1), lambda zz4: 0e0)
                ELSE (zs2.zsol).x END AS zcoef
    FROM (SELECT _sarimax_solve_list(
                     (SELECT list(zv ORDER BY zj1, zj2)
                      FROM (
                          SELECT za.zc::BIGINT AS zj1, zb.zc::BIGINT AS zj2, list_reduce(list_prepend(0e0, list(za.zx * zb.zx ORDER BY za.zt)),
                       lambda zfa, zfb: zfa + zfb) AS zv
                          FROM _sarimax_sp_d2 za JOIN _sarimax_sp_d2 zb ON za.zt = zb.zt
                          GROUP BY za.zc, zb.zc
                          UNION ALL
                          SELECT za.zc::BIGINT, (p + q + bigp + bigq + 1)::BIGINT, list_reduce(list_prepend(0e0, list(za.zx * ze.zev ORDER BY za.zt)),
                       lambda zfa, zfb: zfa + zfb)
                          FROM _sarimax_sp_d2 za JOIN _sarimax_sp_e ze ON ze.zt = za.zt
                          GROUP BY za.zc
                      )),
                     p + q + bigp + bigq, 1) AS zsol) zs2
),
_sarimax_sp_sig AS (
    SELECT CASE WHEN zsg.zs2v IS NOT NULL AND isfinite(zsg.zs2v) AND zsg.zs2v > 0e0 THEN zsg.zs2v
                WHEN zwv.zvw IS NOT NULL AND isfinite(zwv.zvw) AND zwv.zvw > 0e0 THEN zwv.zvw
                ELSE 1e0 END AS zsig2
    FROM (
        SELECT list_reduce(list_prepend(0e0, list(zres * zres ORDER BY zt)),
               lambda zfa, zfb: zfa + zfb) / greatest(count(*), 1) AS zs2v
        FROM (
            SELECT ze.zt, ze.zev - coalesce(zfld.zsum, 0e0) AS zres
            FROM _sarimax_sp_e ze
            JOIN _sarimax_sp_dims3 zd3 ON ze.zt >= zd3.zt0
            LEFT JOIN (
                SELECT za.zt, list_reduce(
                           list_prepend(0e0, list(za.zx * zcf.zcoef[za.zc] ORDER BY za.zc)),
                           lambda za5, zb5: za5 + zb5) AS zsum
                FROM _sarimax_sp_d2 za, _sarimax_sp_coef zcf
                GROUP BY za.zt, zcf.zcoef
            ) zfld ON zfld.zt = ze.zt
        )
    ) zsg,
    (SELECT list_reduce(list_prepend(0e0, list(w * w ORDER BY t)),
                lambda zfa, zfb: zfa + zfb) / greatest(count(*), 1) AS zvw FROM query_table(w_tbl)) zwv
),
_sarimax_sp_base AS (     -- remap stage-2 columns to canonical order, guard finite
    SELECT zbc.zbeta,
           list_transform(range(1, p + q + bigp + bigq + 1), lambda zc5:
               (list_transform([CASE WHEN zc5 <= p THEN zcf.zcoef[zc5]
                                     WHEN zc5 <= p + q THEN zcf.zcoef[p + bigp + (zc5 - p)]
                                     WHEN zc5 <= p + q + bigp THEN zcf.zcoef[p + (zc5 - p - q)]
                                     ELSE zcf.zcoef[p + bigp + q + (zc5 - p - q - bigp)] END],
                    lambda zv5: CASE WHEN coalesce(isfinite(zv5), false)
                                     THEN zv5 ELSE 0e0 END))[1]) AS zarma,
           zsg.zsig2
    FROM _sarimax_sp_beta zbc, _sarimax_sp_coef zcf, _sarimax_sp_sig zsg
),
_sarimax_sp_pick AS (     -- shrink-until-valid, smallest z wins
    SELECT zshrink, zcand, zx0c
    FROM (
        SELECT zz.zshrink,
               zb6.zbeta
                 || list_transform(zb6.zarma, lambda za6: za6 * power(5e-1, zz.zshrink))
                 || [zb6.zsig2] AS zcand,
               TRY(_sarimax_untransform_params(
                       zb6.zbeta
                         || list_transform(zb6.zarma, lambda za6: za6 * power(5e-1, zz.zshrink))
                         || [zb6.zsig2],
                       r, p, q, bigp, bigq)) AS zx0c
        FROM _sarimax_sp_base zb6, unnest(range(0, 61)) AS zz(zshrink)
    )
    WHERE zx0c IS NOT NULL
      AND len(list_filter(zx0c, lambda ze6: coalesce(isfinite(ze6), false))) = len(zx0c)
      AND len(zx0c) = r + p + q + bigp + bigq + 1
    ORDER BY zshrink
    LIMIT 1
)
SELECT CASE WHEN zp.zx0c IS NULL
            THEN error('duckARIMA start values: no stationary/invertible starting point found')
            ELSE zp.zx0c END AS x0,
       zp.zcand AS params0
FROM (SELECT 1 AS zone) zdum
LEFT JOIN _sarimax_sp_pick zp ON true;

-- ---- 2c. BFGS optimizer (spec 5.4 pinned constants) ---------------------------

-- Minimizes f(x) = -loglik over the unconstrained space as one recursive CTE
-- (USING KEY on a constant key: the last emitted row is the result). Pinned
-- constants (spec 5.4): H0 = I; Armijo c1 = 1e-4, alpha0 = 1, backtrack 0.5,
-- <= 30 backtracks (phase A evaluates {1, .5, .25} eagerly, phase B the
-- remaining 0.5^3..0.5^30 ONLY when phase A rejects all three -- the gated
-- unnest keeps it genuinely lazy); central differences with
-- h_i = 1e-7 * greatest(1, |x_i|); BFGS update skipped when
-- y's <= 1e-10 ||y|| ||s||; gradient tolerance 1e-9; 500 iterations; ONE
-- restart from x + hash-based deterministic perturbation in [-0.1, 0.1]
-- (H reset to I, iteration counter reset, run flagged restarted).
--
-- DOCUMENTED DEVIATION (noise-floor stall certificate): a central-difference
-- gradient of this loglikelihood carries irreducible rounding noise of order
-- eps * |ll| / h  (~5e-6 on these fixtures, measured), so the pinned 1e-9
-- test can never certify and a run would burn 2 x 500 iterations and report
-- non-convergence from a point that is BETTER than statsmodels' optimum.
-- Two additional certificates therefore also set status = converged:
--   (a) the line search exhausts all 31 candidates while ||g||inf <= 1e-4
--       (no descent is possible along -H g when g is pure FD noise), or
--   (b) an accepted step changes f by <= 32 * eps * greatest(1, |f|) while
--       ||g_new||inf <= 1e-4 (the objective is flat at evaluation noise).
-- Both stop AT the noise floor -- tighter than statsmodels' own lbfgs
-- stopping slack (its theta-hat shows FD gradients ~1e-3 on these fixtures).
--
-- Status codes: 1 converged, 2 max-iterations, 3 line-search/gradient failure.
CREATE OR REPLACE MACRO _sarimax_bfgs(w_tbl, exog_diff_tbl, r, p, q, bigp, bigq, s) AS TABLE
WITH RECURSIVE
_sarimax_bf_pc AS (
    SELECT (SELECT list(w ORDER BY t) FROM query_table(w_tbl)) AS zwl,
           coalesce((SELECT list(zxr ORDER BY t)
                     FROM (SELECT t, list(x ORDER BY j) AS zxr
                           FROM query_table(exog_diff_tbl) GROUP BY t)),
                    []::DOUBLE[][]) AS zxm,
           (r + p + q + bigp + bigq + 1)::BIGINT AS znp,
           zsp.x0 AS zx0
    FROM _sarimax_start_params(w_tbl, exog_diff_tbl, r, p, q, bigp, bigq, s) zsp
),
_sarimax_bf_it USING KEY (zkk) AS (
    -- ---------------- anchor: f and gradient at x0 ----------------
    SELECT 1::INT AS zkk, 0::INT AS ziter,
           za2.zx AS zx, za2.zfx AS zfx, za2.zg_new AS zgx,
           list_transform(range(1, za2.znp * za2.znp + 1), lambda zidx:
               CASE WHEN (zidx - 1) // za2.znp = (zidx - 1) % za2.znp THEN 1e0 ELSE 0e0 END) AS zhinv,
           CASE WHEN za2.zg_new IS NULL
                     OR len(list_filter(za2.zg_new, lambda ze: ze IS NULL)) > 0 THEN 3
                WHEN list_reduce(list_prepend(0e0, list_transform(za2.zg_new, lambda ze: abs(ze))),
                                 lambda za, zb: greatest(za, zb)) <= 1e-9 THEN 1
                ELSE 0 END AS zstatus,
           false AS zrestarted, 0::INT AS zlsf
    FROM (
        SELECT za1.*,
               list_transform(range(1, za1.znp + 1), lambda zi:
                   (za1.zfpm[2 * zi - 1] - za1.zfpm[2 * zi])
                   / (2e0 * 1e-7 * greatest(1e0, abs(za1.zx[zi])))) AS zg_new
        FROM (
            SELECT zpc.zx0 AS zx, zpc.znp,
                   0e0 - _sarimax_ll_x(zpc.zx0, zpc.zwl, zpc.zxm,
                                       r, p, q, bigp, bigq, s) AS zfx,
                   zag.zfpm
            FROM _sarimax_bf_pc zpc
            CROSS JOIN LATERAL (
                SELECT list(zval ORDER BY zidx) AS zfpm
                FROM (
                    SELECT zidx,
                           0e0 - _sarimax_ll_x(
                               list_transform(zxc, lambda zxe, zxi:
                                   CASE WHEN zxi = (zidx + 1) // 2
                                        THEN zxe + (CASE WHEN zidx % 2 = 1 THEN 1e0 ELSE -1e0 END)
                                                   * 1e-7 * greatest(1e0, abs(zxe))
                                        ELSE zxe END),
                               zwlc, zxmc, r, p, q, bigp, bigq, s) AS zval
                    FROM (SELECT zpc.zx0 AS zxc, zpc.zwl AS zwlc, zpc.zxm AS zxmc, zu.zidx
                          FROM unnest(range(1, 2 * zpc.znp + 1)) AS zu(zidx))
                )
            ) zag
        ) za1
    ) za2

    UNION ALL

    -- ---------------- one BFGS iteration ----------------
    SELECT zfin.zkk,
           CASE WHEN zfin.zrestart_ls OR zfin.zrestart2 THEN 0 ELSE zfin.zniter END AS ziter,
           CASE WHEN zfin.zterminal_ls OR zfin.zstall_ls OR zfin.zgterm THEN zfin.zx
                WHEN zfin.zrestart_ls THEN zfin.zx_eval
                WHEN zfin.zrestart2 THEN zfin.zx_r
                ELSE zfin.zx_new END AS zx,
           CASE WHEN zfin.zterminal_ls OR zfin.zstall_ls OR zfin.zgterm THEN zfin.zfx
                WHEN zfin.zrestart_ls THEN zfin.zfe
                WHEN zfin.zrestart2 THEN zfin.zf2
                ELSE zfin.zfe END AS zfx,
           CASE WHEN zfin.zterminal_ls OR zfin.zstall_ls OR zfin.zgterm THEN zfin.zgx
                WHEN zfin.zrestart_ls THEN zfin.zg_new
                WHEN zfin.zrestart2 THEN zfin.zg2
                ELSE zfin.zg_new END AS zgx,
           CASE WHEN zfin.zrestart_ls OR zfin.zrestart2
                THEN list_transform(range(1, zfin.znp * zfin.znp + 1), lambda zidx:
                         CASE WHEN (zidx - 1) // zfin.znp = (zidx - 1) % zfin.znp
                              THEN 1e0 ELSE 0e0 END)
                WHEN zfin.zstatus_new = 0 AND NOT zfin.zskip THEN zfin.zhup
                ELSE zfin.zhinv END AS zhinv,
           zfin.zstatus_new AS zstatus,
           (zfin.zrestarted OR zfin.zrestart_ls OR zfin.zrestart2) AS zrestarted,
           zfin.zlsf + CASE WHEN zfin.zls_fail OR (zfin.zgnull AND zfin.zls_ok)
                            THEN 1 ELSE 0 END AS zlsf
    FROM (
        SELECT zs9.*,
               CASE WHEN zs9.zterminal_ls THEN 3
                    WHEN zs9.zstall_ls THEN 1
                    WHEN zs9.zrestart_ls THEN CASE WHEN zs9.zgnull THEN 3 ELSE 0 END
                    WHEN zs9.zrestart2 THEN CASE WHEN zs9.zgnull2 THEN 3 ELSE 0 END
                    WHEN zs9.zgterm THEN 3
                    WHEN zs9.zconv OR zs9.zstall_f THEN 1
                    WHEN zs9.zniter >= 500 THEN 2
                    ELSE 0 END AS zstatus_new
        FROM (
            SELECT zs8.*,
                   CASE WHEN zs8.zskip0 THEN NULL ELSE
                       _sarimax_madd(
                           _sarimax_mmul(zs8.zhalf, zs8.zat, zs8.znp, zs8.znp, zs8.znp),
                           list_transform(range(1, zs8.znp * zs8.znp + 1), lambda zidx:
                               (1e0 / zs8.zsy) * zs8.zsv[(zidx - 1) // zs8.znp + 1]
                                               * zs8.zsv[(zidx - 1) % zs8.znp + 1])) END AS zhup,
                   zs8.zskip0 AS zskip
            FROM (
                SELECT zs7.*,
                       CASE WHEN zs7.zskip0 THEN NULL
                            ELSE _sarimax_mmul(zs7.zam, zs7.zhinv,
                                               zs7.znp, zs7.znp, zs7.znp) END AS zhalf
                FROM (
                    SELECT zs6.*,
                           CASE WHEN zs6.zskip0 THEN NULL ELSE
                               list_transform(range(1, zs6.znp * zs6.znp + 1), lambda zidx:
                                   (CASE WHEN (zidx - 1) // zs6.znp = (zidx - 1) % zs6.znp
                                         THEN 1e0 ELSE 0e0 END)
                                   - (1e0 / zs6.zsy) * zs6.zsv[(zidx - 1) // zs6.znp + 1]
                                                     * zs6.zyv[(zidx - 1) % zs6.znp + 1]) END AS zam,
                           CASE WHEN zs6.zskip0 THEN NULL ELSE
                               list_transform(range(1, zs6.znp * zs6.znp + 1), lambda zidx:
                                   (CASE WHEN (zidx - 1) // zs6.znp = (zidx - 1) % zs6.znp
                                         THEN 1e0 ELSE 0e0 END)
                                   - (1e0 / zs6.zsy) * zs6.zyv[(zidx - 1) // zs6.znp + 1]
                                                     * zs6.zsv[(zidx - 1) % zs6.znp + 1]) END AS zat
                    FROM (
                        SELECT zs5b.*,
                               CASE WHEN zs5b.zsy IS NULL
                                         OR NOT (zs5b.zsy > 1e-10 * sqrt(zs5b.zyy) * sqrt(zs5b.zss))
                                    THEN true ELSE false END AS zskip0
                        FROM (
                            SELECT zs5.*,
                                   list_reduce(list_prepend(0e0, list_transform(range(1, zs5.znp + 1),
                                       lambda zi: zs5.zyv[zi] * zs5.zsv[zi])),
                                       lambda za, zb: za + zb) AS zsy,
                                   list_reduce(list_prepend(0e0, list_transform(range(1, zs5.znp + 1),
                                       lambda zi: zs5.zyv[zi] * zs5.zyv[zi])),
                                       lambda za, zb: za + zb) AS zyy,
                                   list_reduce(list_prepend(0e0, list_transform(range(1, zs5.znp + 1),
                                       lambda zi: zs5.zsv[zi] * zs5.zsv[zi])),
                                       lambda za, zb: za + zb) AS zss
                            FROM (
                                SELECT zs4b.*,
                                       zs4b.zrestart2 AND (zs4b.zg2 IS NULL
                                           OR len(list_filter(zs4b.zg2, lambda ze: ze IS NULL)) > 0) AS zgnull2,
                                       list_transform(zs4b.zx_new, lambda zxe, zxi:
                                           zxe - zs4b.zx[zxi]) AS zsv,
                                       list_transform(zs4b.zg_new, lambda zge, zgi:
                                           zge - zs4b.zgx[zgi]) AS zyv
                                FROM (
                                    SELECT zs4a.*,
                                           CASE WHEN zs4a.zn2 = 0 THEN NULL ELSE
                                               list_transform(range(1, zs4a.znp + 1), lambda zi:
                                                   (zs4a.zfpm2[2 * zi - 1] - zs4a.zfpm2[2 * zi])
                                                   / (2e0 * 1e-7 * greatest(1e0, abs(zs4a.zx_r[zi]))))
                                           END AS zg2
                                    FROM (
                                        SELECT zs4.*, zbb2.zf2, zbb2.zfpm2, zbb2.zn2
                                        FROM (
                                            SELECT zs3g.*,
                                                   zs3g.zls_ok AND NOT zs3g.zrestarted
                                                     AND (zs3g.zgnull
                                                          OR (NOT (zs3g.zconv OR zs3g.zstall_f)
                                                              AND zs3g.zniter >= 500)) AS zrestart2,
                                                   zs3g.zls_ok AND zs3g.zgnull
                                                     AND zs3g.zrestarted AS zgterm,
                                                   list_transform(
                                                       CASE WHEN zs3g.zgnull THEN zs3g.zx
                                                            ELSE zs3g.zx_new END,
                                                       lambda zxe, zxi:
                                                           zxe + ((hash(zxi) % 2001) / 1e3 - 1e0)
                                                                 * 1e-1) AS zx_r
                                            FROM (
                                                SELECT zs3f.*,
                                                       (zs3f.zg_new IS NULL
                                                        OR len(list_filter(zs3f.zg_new,
                                                               lambda ze: ze IS NULL)) > 0) AS zgnull,
                                                       CASE WHEN zs3f.zg_new IS NOT NULL
                                                                 AND len(list_filter(zs3f.zg_new,
                                                                         lambda ze: ze IS NULL)) = 0
                                                                 AND list_reduce(
                                                                       list_prepend(0e0,
                                                                           list_transform(zs3f.zg_new,
                                                                               lambda ze: abs(ze))),
                                                                       lambda za, zb: greatest(za, zb))
                                                                     <= 1e-9
                                                            THEN true ELSE false END AS zconv,
                                                       -- stall certificate (b): flat objective at
                                                       -- noise-floor gradient (documented deviation)
                                                       CASE WHEN zs3f.zls_ok
                                                                 AND zs3f.zg_new IS NOT NULL
                                                                 AND len(list_filter(zs3f.zg_new,
                                                                         lambda ze: ze IS NULL)) = 0
                                                                 AND list_reduce(
                                                                       list_prepend(0e0,
                                                                           list_transform(zs3f.zg_new,
                                                                               lambda ze: abs(ze))),
                                                                       lambda za, zb: greatest(za, zb))
                                                                     <= 1e-4
                                                                 AND abs(zs3f.zfe - zs3f.zfx)
                                                                     <= 7.105427357601002e-15
                                                                        * greatest(1e0, abs(zs3f.zfx))
                                                            THEN true ELSE false END AS zstall_f
                                                FROM (
                                                    SELECT zs3x.*,
                                                           CASE WHEN zs3x.zfpm IS NULL THEN NULL ELSE
                                                               list_transform(range(1, zs3x.znp + 1),
                                                                   lambda zi:
                                                                       (zs3x.zfpm[2 * zi - 1]
                                                                        - zs3x.zfpm[2 * zi])
                                                                       / (2e0 * 1e-7 * greatest(1e0,
                                                                              abs(zs3x.zx_eval[zi]))))
                                                           END AS zg_new,
                                                           CASE WHEN zs3x.zls_ok THEN (zs3x.zbest).zf
                                                                ELSE zs3x.zfe_c END AS zfe
                                                    FROM (
                                                        SELECT zs3.*, zbb1.zfpm, zbb1.zfe_c
                                                        FROM (
                                                            SELECT zs2.*,
                                                                   CASE WHEN zs2.zls_ok
                                                                        THEN list_transform(zs2.zx,
                                                                                 lambda zxe, zxi:
                                                                                     zxe + (zs2.zbest).za
                                                                                           * zs2.zdir[zxi])
                                                                        WHEN NOT zs2.zstall_ls
                                                                             AND NOT zs2.zrestarted
                                                                        THEN list_transform(zs2.zx,
                                                                                 lambda zxe, zxi:
                                                                                     zxe + ((hash(zxi) % 2001)
                                                                                            / 1e3 - 1e0)
                                                                                           * 1e-1)
                                                                        ELSE NULL END AS zx_eval,
                                                                   CASE WHEN zs2.zls_ok
                                                                        THEN list_transform(zs2.zx,
                                                                                 lambda zxe, zxi:
                                                                                     zxe + (zs2.zbest).za
                                                                                           * zs2.zdir[zxi])
                                                                        END AS zx_new,
                                                                   (NOT zs2.zls_ok)
                                                                     AND NOT zs2.zstall_ls AS zls_fail,
                                                                   (NOT zs2.zls_ok) AND NOT zs2.zstall_ls
                                                                     AND NOT zs2.zrestarted AS zrestart_ls,
                                                                   (NOT zs2.zls_ok) AND NOT zs2.zstall_ls
                                                                     AND zs2.zrestarted AS zterminal_ls,
                                                                   zs2.ziter + 1 AS zniter
                                                            FROM (
                                                                SELECT zs1b.*,
                                                                       -- stall certificate (a): line
                                                                       -- search exhausted at noise-floor
                                                                       -- gradient (documented deviation)
                                                                       (zs1b.zbest IS NULL)
                                                                         AND zs1b.zgn_cur <= 1e-4 AS zstall_ls,
                                                                       zs1b.zbest IS NOT NULL AS zls_ok
                                                                FROM (
                                                                    SELECT zs1.*,
                                                                           coalesce(zs1.zbesta, zpb.zbestb) AS zbest
                                                                    FROM (
                                                                        SELECT zs0b.*, zpa.zbesta
                                                                        FROM (
                                                                            SELECT zs0.*,
                                                                                   list_reduce(list_prepend(0e0,
                                                                                       list_transform(range(1, zs0.znp + 1),
                                                                                           lambda zi:
                                                                                               zs0.zgx[zi] * zs0.zdir[zi])),
                                                                                       lambda za, zb: za + zb) AS zgd,
                                                                                   list_reduce(list_prepend(0e0,
                                                                                       list_transform(zs0.zgx,
                                                                                           lambda ze: abs(ze))),
                                                                                       lambda za, zb: greatest(za, zb)) AS zgn_cur
                                                                            FROM (
                                                                                SELECT zit.zkk, zit.ziter, zit.zx, zit.zfx,
                                                                                       zit.zgx, zit.zhinv, zit.zrestarted,
                                                                                       zit.zlsf, zpc.znp, zpc.zwl, zpc.zxm,
                                                                                       list_transform(range(1, zpc.znp + 1),
                                                                                           lambda zi:
                                                                                               -list_reduce(
                                                                                                   list_prepend(0e0,
                                                                                                       list_transform(range(1, zpc.znp + 1),
                                                                                                           lambda zj:
                                                                                                               zit.zhinv[(zi - 1) * zpc.znp + zj]
                                                                                                               * zit.zgx[zj])),
                                                                                                   lambda za, zb: za + zb)) AS zdir
                                                                                FROM _sarimax_bf_it zit, _sarimax_bf_pc zpc
                                                                                WHERE zit.zstatus = 0
                                                                            ) zs0
                                                                        ) zs0b
                                                                        CROSS JOIN LATERAL (
                                                                            SELECT max(CASE WHEN zfa IS NOT NULL
                                                                                             AND zfa <= zfxc + 1e-4 * zalpha * zgdc
                                                                                            THEN struct_pack(za := zalpha, zf := zfa)
                                                                                       END) AS zbesta
                                                                            FROM (
                                                                                SELECT zfxc, zgdc,
                                                                                       zalpha,
                                                                                       0e0 - _sarimax_ll_x(
                                                                                           list_transform(zxc, lambda zxe, zxi:
                                                                                               zxe + zalpha * zdirc[zxi]),
                                                                                           zwlc, zxmc,
                                                                                           r, p, q, bigp, bigq, s) AS zfa
                                                                                FROM (SELECT zs0b.zx AS zxc, zs0b.zdir AS zdirc,
                                                                                             zs0b.zfx AS zfxc, zs0b.zgd AS zgdc,
                                                                                             zs0b.zwl AS zwlc, zs0b.zxm AS zxmc,
                                                                                             zua.zalpha
                                                                                      FROM unnest([1e0, 5e-1, 25e-2]) AS zua(zalpha))
                                                                            )
                                                                        ) zpa
                                                                    ) zs1
                                                                    CROSS JOIN LATERAL (
                                                                        SELECT max(CASE WHEN zfa IS NOT NULL
                                                                                         AND zfa <= zfxc + 1e-4 * zalpha * zgdc
                                                                                        THEN struct_pack(za := zalpha, zf := zfa)
                                                                                   END) AS zbestb
                                                                        FROM (
                                                                            SELECT zfxc, zgdc, zalpha,
                                                                                   0e0 - _sarimax_ll_x(
                                                                                       list_transform(zxc, lambda zxe, zxi:
                                                                                           zxe + zalpha * zdirc[zxi]),
                                                                                       zwlc, zxmc,
                                                                                       r, p, q, bigp, bigq, s) AS zfa
                                                                            FROM (SELECT zs1.zx AS zxc, zs1.zdir AS zdirc,
                                                                                         zs1.zfx AS zfxc, zs1.zgd AS zgdc,
                                                                                         zs1.zwl AS zwlc, zs1.zxm AS zxmc,
                                                                                         zub.zalpha
                                                                                  FROM unnest(CASE WHEN zs1.zbesta IS NULL
                                                                                                   THEN list_transform(range(3, 31),
                                                                                                            lambda zev: power(5e-1, zev))
                                                                                                   ELSE []::DOUBLE[] END)
                                                                                       AS zub(zalpha))
                                                                        )
                                                                    ) zpb
                                                                ) zs1b
                                                            ) zs2
                                                        ) zs3
                                                        CROSS JOIN LATERAL (
                                                            -- batch 1: gradient (idx 1..2np) + center
                                                            -- f (idx 0, restart path only) at x_eval
                                                            SELECT list(zval ORDER BY zidx)
                                                                       FILTER (WHERE zidx > 0) AS zfpm,
                                                                   max(zval) FILTER (WHERE zidx = 0) AS zfe_c
                                                            FROM (
                                                                SELECT zidx,
                                                                       0e0 - _sarimax_ll_x(
                                                                           list_transform(zxc, lambda zxe, zxi:
                                                                               CASE WHEN zidx > 0
                                                                                         AND zxi = (zidx + 1) // 2
                                                                                    THEN zxe + (CASE WHEN zidx % 2 = 1
                                                                                                     THEN 1e0 ELSE -1e0 END)
                                                                                               * 1e-7
                                                                                               * greatest(1e0, abs(zxe))
                                                                                    ELSE zxe END),
                                                                           zwlc, zxmc,
                                                                           r, p, q, bigp, bigq, s) AS zval
                                                                FROM (SELECT zs3.zx_eval AS zxc,
                                                                             zs3.zwl AS zwlc, zs3.zxm AS zxmc,
                                                                             zu.zidx
                                                                      FROM unnest(CASE WHEN zs3.zterminal_ls
                                                                                            OR zs3.zstall_ls
                                                                                       THEN []::BIGINT[]
                                                                                       WHEN zs3.zrestart_ls
                                                                                       THEN range(0, 2 * zs3.znp + 1)
                                                                                       ELSE range(1, 2 * zs3.znp + 1)
                                                                                  END) AS zu(zidx))
                                                            )
                                                        ) zbb1
                                                    ) zs3x
                                                ) zs3f
                                            ) zs3g
                                        ) zs4
                                        CROSS JOIN LATERAL (
                                            -- batch 2: f + gradient at the perturbed restart point
                                            -- (gated: rows exist only when restart-2 triggers)
                                            SELECT max(zval) FILTER (WHERE zidx = 0) AS zf2,
                                                   list(zval ORDER BY zidx) FILTER (WHERE zidx > 0) AS zfpm2,
                                                   count(*) AS zn2
                                            FROM (
                                                SELECT zidx,
                                                       0e0 - _sarimax_ll_x(
                                                           list_transform(zxc, lambda zxe, zxi:
                                                               CASE WHEN zidx > 0 AND zxi = (zidx + 1) // 2
                                                                    THEN zxe + (CASE WHEN zidx % 2 = 1
                                                                                     THEN 1e0 ELSE -1e0 END)
                                                                               * 1e-7 * greatest(1e0, abs(zxe))
                                                                    ELSE zxe END),
                                                           zwlc, zxmc, r, p, q, bigp, bigq, s) AS zval
                                                FROM (SELECT zs4.zx_r AS zxc, zs4.zwl AS zwlc,
                                                             zs4.zxm AS zxmc, zu.zidx
                                                      FROM unnest(CASE WHEN zs4.zrestart2
                                                                       THEN range(0, 2 * zs4.znp + 1)
                                                                       ELSE []::BIGINT[] END) AS zu(zidx))
                                            )
                                        ) zbb2
                                    ) zs4a
                                ) zs4b
                            ) zs5
                        ) zs5b
                    ) zs6
                ) zs7
            ) zs8
        ) zs9
    ) zfin
)
SELECT zit.zx AS x_opt,
       _sarimax_transform_params(zit.zx, r, p, q, bigp, bigq) AS params,
       0e0 - zit.zfx AS loglik,
       zit.zstatus = 1 AS converged,
       zit.ziter AS iterations,
       list_reduce(list_prepend(0e0, list_transform(zit.zgx, lambda ze: abs(ze))),
                   lambda za, zb: greatest(za, zb)) AS grad_norm,
       zit.zrestarted AS restarted,
       zit.zlsf AS ls_failures
FROM _sarimax_bf_it zit;

-- ---- 2d. standard errors (numerical Hessian, spec 5.4 last paragraph) --------

-- One row (bse DOUBLE[]): standard errors of the CONSTRAINED parameters,
-- matching statsmodels' cov_params_approx flavor: central second differences
-- of the LOGLIKELIHOOD (not its negation) directly in the constrained space
-- at theta-hat, h_i = 1e-4 * greatest(0.1, |theta_i|); bse = sqrt of the
-- diagonal of inv(-H). All np*(np+1)/2 Hessian cells batch as rows of one
-- query (diagonal cells 2 evaluations, off-diagonal 4).
-- DOCUMENTED DEVIATION (step floor 0.1, not 1.0): with a floor of 1.0 the
-- sigma2 step on the log-scale airline fixtures (sigma2 ~ 1.3e-3) is a 7%
-- relative perturbation, whose truncation error pushes bse ~4e-3 away from
-- statsmodels -- beyond the T3 1e-3 gate (measured). The 0.1 floor matches
-- statsmodels' own numdiff._get_epsilon convention (EPS^(1/4) * max(|x|, 0.1)).
-- Boundary behavior (documented): if a +-h_i probe steps outside the
-- stationary region (NULL loglikelihood), that coordinate's step is halved,
-- at most twice, before the grid is evaluated (CASE laziness keeps the extra
-- probes unevaluated at interior points). params/wlist/xmat must be
-- materialized values (see section header).
CREATE OR REPLACE MACRO _sarimax_bse(params, wlist, xmat, r, p, q, bigp, bigq, s) AS TABLE
WITH _sarimax_bse_in0 AS (
    -- bind the raw arguments to columns FIRST (plain projections only), so
    -- callers may pass scalar subqueries: a subquery argument that reached a
    -- lambda body would be a binder error in DuckDB 1.5.4
    SELECT params AS zc, wlist AS zwl, xmat AS zxm,
           (r + p + q + bigp + bigq + 1)::BIGINT AS znp
),
_sarimax_bse_in AS (
    SELECT zc, zwl, zxm, znp,
           _sarimax_ll_c(zc, zwl, zxm, r, p, q, bigp, bigq, s) AS zf0
    FROM _sarimax_bse_in0
),
_sarimax_bse_h AS (       -- adaptive per-coordinate steps
    SELECT zin.zc, zin.zwl, zin.zxm, zin.znp, zin.zf0, zhh.zhl
    FROM _sarimax_bse_in zin
    CROSS JOIN LATERAL (
        SELECT list(zh ORDER BY zi) AS zhl
        FROM (
            SELECT zi,
                   (list_transform([1e-4 * greatest(1e-1, abs(zcc[zi]))], lambda zh0:
                        CASE WHEN _sarimax_ll_c(list_transform(zcc, lambda zv2, zi2:
                                      CASE WHEN zi2 = zi THEN zv2 + zh0 ELSE zv2 END),
                                      zwlc, zxmc, r, p, q, bigp, bigq, s) IS NOT NULL
                                  AND _sarimax_ll_c(list_transform(zcc, lambda zv2, zi2:
                                          CASE WHEN zi2 = zi THEN zv2 - zh0 ELSE zv2 END),
                                          zwlc, zxmc, r, p, q, bigp, bigq, s) IS NOT NULL
                             THEN zh0
                             WHEN _sarimax_ll_c(list_transform(zcc, lambda zv2, zi2:
                                      CASE WHEN zi2 = zi THEN zv2 + zh0 * 5e-1 ELSE zv2 END),
                                      zwlc, zxmc, r, p, q, bigp, bigq, s) IS NOT NULL
                                  AND _sarimax_ll_c(list_transform(zcc, lambda zv2, zi2:
                                          CASE WHEN zi2 = zi THEN zv2 - zh0 * 5e-1 ELSE zv2 END),
                                          zwlc, zxmc, r, p, q, bigp, bigq, s) IS NOT NULL
                             THEN zh0 * 5e-1
                             ELSE zh0 * 25e-2 END))[1] AS zh
            FROM (SELECT zin.zc AS zcc, zin.zwl AS zwlc, zin.zxm AS zxmc, zu.zi
                  FROM unnest(range(1, zin.znp + 1)) AS zu(zi))
        )
    ) zhh
),
_sarimax_bse_tri AS (     -- upper-triangle H cells, ordered (i, j), i <= j
    SELECT zh2.zc, zh2.zwl, zh2.zxm, zh2.znp, zh2.zhl, zgr.ztri
    FROM _sarimax_bse_h zh2
    CROSS JOIN LATERAL (
        SELECT list(zhv ORDER BY zi, zj) AS ztri
        FROM (
            SELECT zi, zj,
                   CASE WHEN zi = zj
                        THEN (list_reduce(list_prepend(0e0, list(zfv ORDER BY zs1, zs2)),
                                          lambda za7, zb7: za7 + zb7)
                              - 2e0 * zf0c) / (zhlc[zi] * zhlc[zi])
                        ELSE list_reduce(list_prepend(0e0,
                                 list(zs1 * zs2 * zfv ORDER BY zs1, zs2)),
                                 lambda za7, zb7: za7 + zb7)
                             / (4e0 * zhlc[zi] * zhlc[zj]) END AS zhv
            FROM (
                SELECT zi, zj, zs1, zs2, zf0c, zhlc,
                       _sarimax_ll_c(list_transform(zcc, lambda zv3, zk3:
                               zv3 + CASE WHEN zk3 = zi THEN zs1 * zhlc[zi] ELSE 0e0 END
                                   + CASE WHEN zk3 = zj AND zj <> zi
                                          THEN zs2 * zhlc[zj] ELSE 0e0 END),
                           zwlc, zxmc, r, p, q, bigp, bigq, s) AS zfv
                FROM (
                    SELECT zh2.zc AS zcc, zh2.zwl AS zwlc, zh2.zxm AS zxmc,
                           zh2.zhl AS zhlc, zh2.zf0 AS zf0c,
                           zu1.zi, zu2.zj, (zsg.zst).zs1 AS zs1, (zsg.zst).zs2 AS zs2
                    FROM unnest(range(1, zh2.znp + 1)) AS zu1(zi)
                    CROSS JOIN unnest(range(1, zh2.znp + 1)) AS zu2(zj)
                    CROSS JOIN LATERAL unnest(
                        CASE WHEN zu2.zj = zu1.zi
                             THEN [struct_pack(zs1 := 1e0, zs2 := 0e0),
                                   struct_pack(zs1 := -1e0, zs2 := 0e0)]
                             ELSE [struct_pack(zs1 := 1e0, zs2 := 1e0),
                                   struct_pack(zs1 := 1e0, zs2 := -1e0),
                                   struct_pack(zs1 := -1e0, zs2 := 1e0),
                                   struct_pack(zs1 := -1e0, zs2 := -1e0)] END)
                        AS zsg(zst)
                    WHERE zu2.zj >= zu1.zi
                )
            )
            GROUP BY zi, zj, zf0c, zhlc
        )
    ) zgr
)
SELECT (list_transform([_sarimax_inv_list(zng.znegh, zng.znp)], lambda zinv:
           list_transform(range(1, zng.znp + 1), lambda zi6:
               (list_transform([(zinv.x)[(zi6 - 1) * zng.znp + zi6]], lambda zdv:
                    CASE WHEN (zinv.ok) AND zdv > 0e0 THEN sqrt(zdv) ELSE NULL END))[1])))[1] AS bse
FROM (
    SELECT znp,
           list_transform(range(1, znp * znp + 1), lambda zidx:
               -(ztri[(least((zidx - 1) // znp + 1, (zidx - 1) % znp + 1) - 1)
                      * (znp + 1)
                      - least((zidx - 1) // znp + 1, (zidx - 1) % znp + 1)
                        * (least((zidx - 1) // znp + 1, (zidx - 1) % znp + 1) - 1) // 2
                      + (greatest((zidx - 1) // znp + 1, (zidx - 1) % znp + 1)
                         - least((zidx - 1) // znp + 1, (zidx - 1) % znp + 1) + 1)])) AS znegh
    FROM _sarimax_bse_tri
) zng;
