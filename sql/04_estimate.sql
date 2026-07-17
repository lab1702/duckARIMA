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


-- ============================================================================
-- SECTION 3: v2 estimation engine (appended; simple_differencing = FALSE,
-- missing values, trend terms, concentrated scale). Section 1/2 macros are
-- FROZEN -- the v1 acceptance suite keeps running against them unchanged.
--
--   3a. conc-aware parameter transform   _sarimax_transform_params_v2 /
--                                        _sarimax_untransform_params_v2
--   3b. scalar loglikelihood kernel      _sarimax_ll_c_v2 / _sarimax_ll_x_v2
--   3c. starting values                  _sarimax_start_params_v2
--   3d. BFGS optimizer                   _sarimax_bfgs_v2
--   3e. standard errors                  _sarimax_bse_v2
--
-- v2 canonical parameter order: tau_1..tau_ktrend, beta_1..beta_r, phi,
-- theta, Phi, Theta, sigma2 -- the sigma2 slot EXISTS ONLY when NOT
-- concentrated. The transform is the v1 PACF bijection on the four ARMA
-- blocks, identity on the leading ktrend + r entries, and square/sqrt on the
-- sigma2 slot when present (verified against fixtures_v2 probes.parquet).
--
-- THE KERNEL'S STATE REPRESENTATION (one-step-shifted; derivation): the
-- augmented system of sql/02_ssm.sql section 2 has design Z equal to ROW 1 of
-- the transition T, and its selection R has R[1] = 0 whenever kdiff > 0.
-- Writing the statsmodels recursion alpha_{t+1} = T alpha_t + c_t e_cidx
-- + R eta_t, y_t = Z alpha_t, and defining gamma_t := alpha_{t+1}, gives the
-- IDENTITY y_t = (T alpha_t)[1] = gamma_t[1] - (c_t e_cidx + R eta_t)[1]
-- = gamma_t[1] (both intercept and noise rows vanish at index 1 for
-- kdiff > 0). gamma therefore follows a state-space model with the SAME T and
-- R, design e_1', intercept c_{t+1} e_cidx consumed at step t, and exact
-- initial moments E[gamma_1] = T a1 + c_1 e_cidx,
-- Var[gamma_1] = T P1 T' + RQR'. The kernel below filters gamma: v_t =
-- yd_t - a[1], F_t = P[1,1], gain = col1(T P)/F -- ALGEBRAICALLY IDENTICAL
-- innovations to statsmodels' general-Z filter (fixture-validated), with the
-- v1 kernel's exact fold shape. When kdiff = 0 the design is natively e_1'
-- and the kernel runs UNSHIFTED (plain v1 recursion + intercept c_t at state
-- 1, stationary a1/P1) -- the only branch points are the initial (a, P) pair
-- and the one-step intercept alignment (c_{t+1} vs c_t), both bound once.
--
-- P1 inside the kernel: same 30-iteration DOUBLING Lyapunov solve as the v1
-- kernel (documented deviation, section 2 header) for the ARMA block;
-- blockdiag(1e6 I_kdiff, .) matches _sarimax_p1_v2 exactly (fixture ssm
-- P1 rows are blockdiag with EXACT zero cross terms).
--
-- MEASURED NOISE FLOOR (documented, affects the T1 gate of ONE fixture): with
-- the approximate-diffuse 1e6 initialization, every filter update after the
-- diffuse states collapse carries catastrophic-cancellation noise of order
-- kappa * eps / F_t RELATIVE per step. On nodiff_sarimax_011_011_12
-- (log-airline scale, F_t ~ 2e-3 post-burn, 131 post-burn steps) a
-- 40-digit-Decimal reference filter puts statsmodels' own loglike 4.0e-5 from
-- exact arithmetic at probe 1 (this kernel: 9.5e-6 from exact -- CLOSER than
-- the reference implementation) and 1.5e-7 at theta-hat. No independent
-- float64 implementation can match statsmodels' reported loglike beyond that
-- floor on this fixture; tests/test_estimate_v2.py gates it at rel <= 5e-7
-- (all other fixtures: the pinned abs <= 1e-8 / rel <= 1e-10).
--
-- Lambda discipline: identical to section 2 (z-prefixed lambda vars; ylist /
-- xmat / degs / cpar must be materialized values or plain columns -- textual
-- macro expansion re-evaluates argument expressions at every reference).
-- ============================================================================

-- ---- 3a. conc-aware parameter transform ----------------------------------------

-- Unconstrained -> constrained. rtot = ktrend + r identity entries; NO sigma2
-- slot when conc (the v1 macro unconditionally appends u[last]^2).
CREATE OR REPLACE MACRO _sarimax_transform_params_v2(u, rtot, p, q, bigp, bigq, conc) AS (
    list_slice(u, 1, rtot)
    || _sarimax_constrain_block(list_slice(u, rtot + 1, rtot + p), -1.0::DOUBLE)
    || _sarimax_constrain_block(list_slice(u, rtot + p + 1, rtot + p + q), 1.0::DOUBLE)
    || _sarimax_constrain_block(list_slice(u, rtot + p + q + 1, rtot + p + q + bigp), -1.0::DOUBLE)
    || _sarimax_constrain_block(list_slice(u, rtot + p + q + bigp + 1, rtot + p + q + bigp + bigq), 1.0::DOUBLE)
    || CASE WHEN conc THEN []::DOUBLE[]
            ELSE [u[rtot + p + q + bigp + bigq + 1] * u[rtot + p + q + bigp + bigq + 1]] END
);

-- Constrained -> unconstrained (sqrt on sigma2 only when the slot exists).
CREATE OR REPLACE MACRO _sarimax_untransform_params_v2(c, rtot, p, q, bigp, bigq, conc) AS (
    list_slice(c, 1, rtot)
    || _sarimax_unconstrain_block(list_slice(c, rtot + 1, rtot + p), -1.0::DOUBLE)
    || _sarimax_unconstrain_block(list_slice(c, rtot + p + 1, rtot + p + q), 1.0::DOUBLE)
    || _sarimax_unconstrain_block(list_slice(c, rtot + p + q + 1, rtot + p + q + bigp), -1.0::DOUBLE)
    || _sarimax_unconstrain_block(list_slice(c, rtot + p + q + bigp + 1, rtot + p + q + bigp + bigq), 1.0::DOUBLE)
    || CASE WHEN conc THEN []::DOUBLE[]
            ELSE [sqrt(c[rtot + p + q + bigp + bigq + 1])] END
);

-- ---- 3b. scalar loglikelihood kernel -------------------------------------------

-- Loglikelihood + scale at a CONSTRAINED v2 parameter vector. ylist is the
-- MODEL-scale series (raw y when kdiff > 0; pre-differenced when the caller
-- runs simple_differencing with d = sd = 0) with SQL NULL at missing t; xmat
-- the UNdifferenced exog rows (n lists of length r; [] or n empty lists when
-- r = 0); degs the BIGINT[] 0-based trend degrees (length ktrend). Returns
-- STRUCT(ll DOUBLE, scale2 DOUBLE); ll is NULL when poisoned (F_t <= 0 on an
-- accumulated step, or non-finite ll from an explosive probe); scale2 is
-- ssq/cnt when conc (the concentrated scale) and the sigma2 parameter
-- otherwise. Loglikelihood sums NON-MISSING t > burn, burn = kdiff.
CREATE OR REPLACE MACRO _sarimax_ll_c_v2(cpar, ylist, xmat, degs,
                                         r, p, q, bigp, bigq, s, d, sd, ktrend, conc) AS (
  (list_transform([struct_pack(
      zka := _sarimax_k_states(p, q, bigp, bigq, s),
      zkd := _sarimax_kdiff(d, sd, s),
      zn := len(ylist),
      ztau := list_slice(cpar, 1, ktrend),
      zbeta := list_slice(cpar, ktrend + 1, ktrend + r),
      zphistar := _sarimax_expand_ar(
          list_slice(cpar, ktrend + r + 1, ktrend + r + p),
          list_slice(cpar, ktrend + r + p + q + 1, ktrend + r + p + q + bigp), s),
      zthetastar := _sarimax_expand_ma(
          list_slice(cpar, ktrend + r + p + 1, ktrend + r + p + q),
          list_slice(cpar, ktrend + r + p + q + bigp + 1,
                     ktrend + r + p + q + bigp + bigq), s),
      zsigma2 := CASE WHEN conc THEN 1e0
                      ELSE cpar[ktrend + r + p + q + bigp + bigq + 1] END)],
   lambda zl1:
     (list_transform([struct_pack(
         zk := zl1.zkd + zl1.zka,
         ztm := _sarimax_build_t_v2(zl1.zphistar, zl1.zka, d, sd, s),
         zta := _sarimax_build_t(zl1.zphistar, zl1.zka),
         zrv := _sarimax_build_r_v2(zl1.zthetastar, zl1.zka, d, sd, s),
         zrva := _sarimax_build_r(zl1.zthetastar, zl1.zka),
         -- yd_t = y_t - x_t' beta (UNdifferenced exog; ordered-j fold seeded
         -- 0e0; NULL y propagates, r = 0 never indexes xmat)
         zyd := list_transform(range(1, zl1.zn + 1), lambda zt:
                    ylist[zt] - list_reduce(
                        list_prepend(0e0, list_transform(range(1, r + 1), lambda zj:
                            xmat[zt][zj] * (zl1.zbeta)[zj])),
                        lambda za, zb: za + zb)),
         -- c_t for t = 1..n+1 (n+1 so the kdiff>0 shifted alignment below can
         -- consume c_{t+1} at step t; empty degs folds to zeros)
         zcl := _sarimax_trend_c(degs, zl1.ztau, 1, zl1.zn + 1))],
      lambda zl2:
        (list_transform([struct_pack(
            ztt := _sarimax_mtrans(zl2.ztm, zl2.zk, zl2.zk),
            zrqr := _sarimax_build_rqr(zl2.zrv, zl1.zsigma2, zl2.zk),
            zrqra := _sarimax_build_rqr(zl2.zrva, zl1.zsigma2, zl1.zka),
            za1u := _sarimax_a1_v2(zl2.zta, zl1.zka, d, sd, s, (zl2.zcl)[1]))],
         lambda zl3:
           -- ARMA-block P1 by 30 doubling iterations (bound before msym)
           (list_transform([(list_reduce(
                    [struct_pack(zsm := zl3.zrqra, zaa := zl2.zta)]
                      || list_transform(range(1, 31), lambda zd2:
                           struct_pack(zsm := []::DOUBLE[], zaa := []::DOUBLE[])),
                    lambda zacc, zel:
                      (list_transform([_sarimax_mmul(zacc.zaa, zacc.zsm,
                                                     zl1.zka, zl1.zka, zl1.zka)],
                       lambda zas:
                         (list_transform([_sarimax_mtrans(zacc.zaa, zl1.zka, zl1.zka)],
                          lambda zat2:
                            (list_transform([_sarimax_mmul(zas, zat2,
                                                           zl1.zka, zl1.zka, zl1.zka)],
                             lambda zasat:
                               struct_pack(
                                 zsm := _sarimax_madd(zacc.zsm, zasat),
                                 zaa := _sarimax_mmul(zacc.zaa, zacc.zaa,
                                                      zl1.zka, zl1.zka, zl1.zka))))[1]
                         ))[1]
                      ))[1]
                 )).zsm],
            lambda zs30:
           (list_transform([_sarimax_msym(zs30, zl1.zka)],
            lambda zsig:
              -- blockdiag(1e6 I_kdiff, Sigma), the UNSHIFTED P1
              (list_transform([list_transform(range(1, zl2.zk * zl2.zk + 1), lambda zidx:
                   CASE
                     WHEN (zidx - 1) // zl2.zk + 1 <= zl1.zkd
                          AND (zidx - 1) % zl2.zk + 1 <= zl1.zkd
                     THEN CASE WHEN (zidx - 1) // zl2.zk = (zidx - 1) % zl2.zk
                               THEN 1e6 ELSE 0e0 END
                     WHEN (zidx - 1) // zl2.zk + 1 > zl1.zkd
                          AND (zidx - 1) % zl2.zk + 1 > zl1.zkd
                     THEN zsig[((zidx - 1) // zl2.zk - zl1.zkd) * zl1.zka
                               + ((zidx - 1) % zl2.zk + 1 - zl1.zkd)]
                     ELSE 0e0 END)],
               lambda zp1b:
                 -- shift conjugation: Var[gamma_1] = T P1 T' + RQR,
                 -- E[gamma_1] = T a1 + c_1 e_cidx (kdiff > 0 only)
                 (list_transform([struct_pack(
                      ztp1 := _sarimax_mmul(zl2.ztm, zp1b, zl2.zk, zl2.zk, zl2.zk),
                      zta1 := _sarimax_mmul(zl2.ztm, zl3.za1u, zl2.zk, zl2.zk, 1))],
                  lambda zl7:
                    (list_transform([_sarimax_mmul(zl7.ztp1, zl3.ztt,
                                                   zl2.zk, zl2.zk, zl2.zk)],
                     lambda ztp1t:
                       -- msym mirrors _sarimax_systems_v2's p1f exactly (the
                       -- madd result is bound before msym sees it)
                       (list_transform([struct_pack(
                            zp1sh := (list_transform([_sarimax_madd(ztp1t, zl3.zrqr)],
                                          lambda zpm: _sarimax_msym(zpm, zl2.zk)))[1],
                            za1sh := list_transform(range(1, zl2.zk + 1), lambda zi5:
                                         (zl7.zta1)[zi5]
                                         + CASE WHEN zi5 = zl1.zkd + 1
                                                THEN (zl2.zcl)[1] ELSE 0e0 END))],
                        lambda zl9:
                          (list_transform([struct_pack(
                               zp1 := CASE WHEN zl1.zkd = 0 THEN zp1b
                                           ELSE zl9.zp1sh END,
                               za1 := CASE WHEN zl1.zkd = 0 THEN zl3.za1u
                                           ELSE zl9.za1sh END,
                               -- per-step intercept: c_t unshifted, c_{t+1} shifted
                               zcs := CASE WHEN zl1.zkd = 0
                                           THEN list_slice(zl2.zcl, 1, zl1.zn)
                                           ELSE list_slice(zl2.zcl, 2, zl1.zn + 1) END)],
                           lambda zl10:
                             -- Kalman fold in strict t order; accumulator and
                             -- elements share one struct type; per-step yd,
                             -- intercept and t ride in zydv/zct/zti
                             (list_transform([(list_reduce(
                                [struct_pack(za2 := zl10.za1, zp2 := zl10.zp1,
                                             zcnt := 0e0, zslf := 0e0, zssq := 0e0,
                                             zydv := 0e0, zct := 0e0, zti := 0::BIGINT)]
                                  || list_transform(range(1, zl1.zn + 1), lambda zt:
                                       struct_pack(za2 := []::DOUBLE[], zp2 := []::DOUBLE[],
                                                   zcnt := 0e0, zslf := 0e0, zssq := 0e0,
                                                   zydv := (zl2.zyd)[zt],
                                                   zct := (zl10.zcs)[zt], zti := zt)),
                                lambda zacc, zel:
                                  (list_transform([struct_pack(
                                       zv := zel.zydv - (zacc.za2)[1],
                                       zf := (zacc.zp2)[1],
                                       ztp := _sarimax_mmul(zl2.ztm, zacc.zp2,
                                                            zl2.zk, zl2.zk, zl2.zk),
                                       zta2 := _sarimax_mmul(zl2.ztm, zacc.za2,
                                                             zl2.zk, zl2.zk, 1))],
                                   lambda zi1:
                                     (list_transform([list_transform(range(1, zl2.zk + 1),
                                                          lambda zi2:
                                                              (zi1.ztp)[(zi2 - 1) * zl2.zk + 1])],
                                      lambda ztpz:
                                        (list_transform([struct_pack(
                                             ztpt := _sarimax_mmul(zi1.ztp, zl3.ztt,
                                                                   zl2.zk, zl2.zk, zl2.zk),
                                             zoutr := list_transform(
                                                 range(1, zl2.zk * zl2.zk + 1), lambda zi4:
                                                     ztpz[(zi4 - 1) // zl2.zk + 1]
                                                     * ztpz[(zi4 - 1) % zl2.zk + 1] / zi1.zf))],
                                         lambda zb1:
                                           (list_transform([_sarimax_msub(zb1.ztpt, zb1.zoutr)],
                                            lambda zmsb:
                                              (list_transform([CASE WHEN zi1.zv IS NULL
                                                                    THEN zb1.ztpt
                                                                    ELSE zmsb END],
                                               lambda zpre:
                                                 (list_transform([_sarimax_madd(zpre, zl3.zrqr)],
                                                  lambda zpu:
                                                    struct_pack(
                                                      za2 := list_transform(
                                                          range(1, zl2.zk + 1), lambda zi3:
                                                              (zi1.zta2)[zi3]
                                                              + (CASE WHEN zi1.zv IS NULL THEN 0e0
                                                                      ELSE ztpz[zi3] * zi1.zv / zi1.zf END)
                                                              + (CASE WHEN zi3 = zl1.zkd + 1
                                                                      THEN zel.zct ELSE 0e0 END)),
                                                      zp2 := _sarimax_msym(zpu, zl2.zk),
                                                      zcnt := zacc.zcnt
                                                          + CASE WHEN zi1.zv IS NOT NULL
                                                                      AND zel.zti > zl1.zkd
                                                                 THEN 1e0 ELSE 0e0 END,
                                                      zslf := zacc.zslf
                                                          + CASE WHEN zi1.zv IS NULL
                                                                      OR zel.zti <= zl1.zkd
                                                                 THEN 0e0
                                                                 WHEN zi1.zf > 0e0 THEN ln(zi1.zf)
                                                                 ELSE NULL END,
                                                      zssq := zacc.zssq
                                                          + CASE WHEN zi1.zv IS NULL
                                                                      OR zel.zti <= zl1.zkd
                                                                 THEN 0e0
                                                                 ELSE zi1.zv * zi1.zv / zi1.zf END,
                                                      zydv := 0e0, zct := 0e0,
                                                      zti := 0::BIGINT)))[1]
                                              ))[1]
                                           ))[1]
                                        ))[1]
                                     ))[1]
                                  ))[1]
                             ))],
                              lambda zfr:
                                (list_transform([CASE WHEN conc
                                     THEN CASE WHEN zfr.zcnt > 0e0 AND zfr.zssq > 0e0
                                               THEN -5e-1 * (zfr.zcnt * ln(2e0 * pi())
                                                             + zfr.zslf
                                                             + zfr.zcnt * ln(zfr.zssq / zfr.zcnt)
                                                             + zfr.zcnt)
                                               ELSE NULL END
                                     ELSE -5e-1 * (zfr.zcnt * ln(2e0 * pi())
                                                   + zfr.zslf + zfr.zssq) END],
                                 lambda zllr:
                                   struct_pack(
                                     ll := CASE WHEN zllr IS NOT NULL AND isfinite(zllr)
                                                THEN zllr ELSE NULL END,
                                     scale2 := CASE WHEN conc
                                                    THEN CASE WHEN zfr.zcnt > 0e0
                                                                   AND isfinite(zfr.zssq)
                                                              THEN zfr.zssq / zfr.zcnt
                                                              ELSE NULL END
                                                    ELSE cpar[ktrend + r + p + q
                                                              + bigp + bigq + 1] END)))[1]
                          ))[1]
                       ))[1]
                    ))[1]
                 ))[1]
              ))[1]
           ))[1]
           ))[1]
           ))[1]
        ))[1]
     ))[1]
  ))[1]
);

-- Loglikelihood struct at an UNCONSTRAINED v2 vector (the optimizer's
-- objective evaluates -(...).ll).
CREATE OR REPLACE MACRO _sarimax_ll_x_v2(xunc, ylist, xmat, degs,
                                         r, p, q, bigp, bigq, s, d, sd, ktrend, conc) AS (
  _sarimax_ll_c_v2(
      _sarimax_transform_params_v2(xunc, ktrend + r, p, q, bigp, bigq, conc),
      ylist, xmat, degs, r, p, q, bigp, bigq, s, d, sd, ktrend, conc)
);

-- ---- 3c. starting values (v2: differencing, missing values, trend, conc) -------

-- One row (x0 DOUBLE[] unconstrained, params0 DOUBLE[] constrained), v2
-- canonical order. Inputs: y_tbl (t, y) MODEL-scale series with NULLs at
-- missing t; exog_tbl (t, j, x) UNdifferenced long-form exog (rows with
-- t <= n are used; zero rows when r = 0); degs_tbl (idx, degree) the 0-based
-- trend polynomial degrees.
--
-- Strategy (start values must be finite and interior, not statsmodels-equal):
--   1. Difference y and the joint [trend-on-original-timeline | exog]
--      regressor block d ordinary + sd seasonal times via ONE convolution
--      with the lag polynomial (1-L)^d (1-L^s)^sd (NULL y propagates through
--      any window that touches a missing value).
--   2. OLS of w on the differenced regressors over COMPLETE CASES only.
--      Differencing annihilates trend columns of degree < d + sd-ish (e.g.
--      'c' under d = 1): columns with fold(x^2) <= 1e-12 are DROPPED and get
--      coefficient 0e0 instead of erroring (rank guard); a singular reduced
--      Gram also falls back to all-zero coefficients (BFGS recovers).
--   3. Hannan-Rissanen exactly like v1 on the complete-case residuals: rows
--      whose full lag window touches a missing residual are excluded from
--      both regressions (the v1 named minimum-length failure is kept, on the
--      differenced-timeline length n - kdiff).
--   4. sigma2_0 = mean squared stage-2 residual over the valid rows
--      (fallbacks: mean non-missing w^2, then 1e0); NO sigma2 slot when conc.
--   5. Shrink-until-valid via the conc-aware untransform (v1 semantics:
--      the four ARMA blocks jointly scaled by 0.5^z, tau/beta untouched).
CREATE OR REPLACE MACRO _sarimax_start_params_v2(y_tbl, exog_tbl, degs_tbl,
                                                 r, p, q, bigp, bigq, s, d, sd,
                                                 ktrend, conc) AS TABLE
WITH _sarimax_sq_args AS (
    SELECT (SELECT count(*) FROM query_table(y_tbl))::BIGINT AS zn,
           _sarimax_kdiff(d, sd, s) AS zkd,
           (ktrend + r)::BIGINT AS znreg
),
_sarimax_sq_dp AS (        -- (1-L)^d (1-L^s)^sd, constant term first
    SELECT list_reduce(
               list_prepend([1e0],
                   list_transform(range(1, d + 1), lambda zi: [1e0, -1e0])
                   || list_transform(range(1, sd + 1), lambda zi:
                          list_prepend(1e0, list_append(
                              list_transform(range(1, s), lambda zz: 0e0), -1e0)))),
               lambda za, zb: _sarimax_polymul(za, zb)) AS zdp
),
_sarimax_sq_dpr AS (       -- polynomial as rows (zj 1-based, zv)
    SELECT zu.zj::BIGINT AS zj, zdpp.zdp[zu.zj] AS zv
    FROM _sarimax_sq_dp zdpp, LATERAL unnest(range(1, len(zdpp.zdp) + 1)) AS zu(zj)
),
_sarimax_sq_w AS (         -- differenced series, zt = 1..n-kd (NULL propagates)
    SELECT (zw1.t - za.zkd)::BIGINT AS zt,
           list_reduce(list_prepend(0e0, list(zr.zv * zw2.y ORDER BY zr.zj)),
                       lambda zfa, zfb: zfa + zfb) AS zw
    FROM _sarimax_sq_args za
    CROSS JOIN query_table(y_tbl) zw1
    JOIN _sarimax_sq_dpr zr ON true
    JOIN query_table(y_tbl) zw2 ON zw2.t = zw1.t - zr.zj + 1
    WHERE zw1.t > za.zkd
    GROUP BY zw1.t, za.zkd
),
_sarimax_sq_base AS (      -- joint regressors, ORIGINAL timeline: trend | exog
    SELECT zu.zt::BIGINT AS zt, zdgs.idx::BIGINT AS zc,
           pow(zu.zt::DOUBLE, zdgs.degree) AS zx
    FROM _sarimax_sq_args za2
    CROSS JOIN LATERAL unnest(range(1, za2.zn + 1)) AS zu(zt)
    CROSS JOIN query_table(degs_tbl) zdgs
    UNION ALL
    SELECT ze.t::BIGINT, (ktrend + ze.j)::BIGINT, ze.x
    FROM query_table(exog_tbl) ze, _sarimax_sq_args za3
    WHERE ze.t <= za3.zn
),
_sarimax_sq_xd AS (        -- differenced regressors on the w timeline
    SELECT (zb.zt - za.zkd)::BIGINT AS zt, zb.zc,
           list_reduce(list_prepend(0e0, list(zr.zv * zb2.zx ORDER BY zr.zj)),
                       lambda zfa, zfb: zfa + zfb) AS zx
    FROM _sarimax_sq_args za
    CROSS JOIN _sarimax_sq_base zb
    JOIN _sarimax_sq_dpr zr ON true
    JOIN _sarimax_sq_base zb2 ON zb2.zc = zb.zc AND zb2.zt = zb.zt - zr.zj + 1
    WHERE zb.zt > za.zkd
    GROUP BY zb.zt, zb.zc, za.zkd
),
_sarimax_sq_cc AS (        -- complete-case rows for the OLS
    SELECT zt FROM _sarimax_sq_w WHERE zw IS NOT NULL
),
_sarimax_sq_keep AS (      -- rank guard: drop annihilated columns
    SELECT zx.zc,
           list_reduce(list_prepend(0e0, list(zx.zx * zx.zx ORDER BY zx.zt)),
                       lambda zfa, zfb: zfa + zfb) > 1e-12 AS zkeep
    FROM _sarimax_sq_xd zx
    JOIN _sarimax_sq_cc zk ON zk.zt = zx.zt
    GROUP BY zx.zc
),
_sarimax_sq_cmap AS (      -- kept columns -> dense 1..nk
    SELECT zc, row_number() OVER (ORDER BY zc) AS zcd
    FROM _sarimax_sq_keep WHERE zkeep
),
_sarimax_sq_nk AS (
    SELECT count(*)::BIGINT AS znk FROM _sarimax_sq_cmap
),
_sarimax_sq_bk AS (        -- OLS over kept columns, complete cases; zeros fallback
    SELECT CASE WHEN znkc.znk = 0 THEN []::DOUBLE[]
                WHEN NOT (zs.zsol).ok
                THEN list_transform(range(1, znkc.znk + 1), lambda zz2: 0e0)
                ELSE (zs.zsol).x END AS zbk
    FROM _sarimax_sq_nk znkc,
         (SELECT _sarimax_solve_list(zg.zaug, zni.znk, 1) AS zsol
          FROM _sarimax_sq_nk zni,
               (SELECT list(zv ORDER BY zj1, zj2) AS zaug
                FROM (
                    SELECT zma.zcd AS zj1, zmb.zcd AS zj2,
                           list_reduce(list_prepend(0e0, list(za.zx * zb.zx ORDER BY za.zt)),
                                       lambda zfa, zfb: zfa + zfb) AS zv
                    FROM _sarimax_sq_xd za
                    JOIN _sarimax_sq_cc zk1 ON zk1.zt = za.zt
                    JOIN _sarimax_sq_xd zb ON zb.zt = za.zt
                    JOIN _sarimax_sq_cmap zma ON zma.zc = za.zc
                    JOIN _sarimax_sq_cmap zmb ON zmb.zc = zb.zc
                    GROUP BY zma.zcd, zmb.zcd
                    UNION ALL
                    SELECT zma.zcd, znx.znk + 1,
                           list_reduce(list_prepend(0e0, list(za.zx * zw.zw ORDER BY za.zt)),
                                       lambda zfa, zfb: zfa + zfb)
                    FROM _sarimax_sq_xd za
                    JOIN _sarimax_sq_w zw ON zw.zt = za.zt AND zw.zw IS NOT NULL
                    JOIN _sarimax_sq_cmap zma ON zma.zc = za.zc
                    CROSS JOIN _sarimax_sq_nk znx
                    GROUP BY zma.zcd, znx.znk
                )) zg) zs
),
_sarimax_sq_bf AS (        -- full-length joint (tau | beta), zeros on dropped cols
    -- scalar-subquery form: MUST yield exactly one row even when znreg = 0
    -- (a GROUP BY over the empty column join would yield zero rows and
    -- silently empty every downstream CTE)
    SELECT coalesce((
        SELECT list(CASE WHEN zm.zcd IS NULL THEN 0e0
                         ELSE zbkc.zbk[zm.zcd] END ORDER BY zcs.zc)
        FROM _sarimax_sq_args za
        CROSS JOIN LATERAL unnest(range(1, za.znreg + 1)) AS zcs(zc)
        LEFT JOIN _sarimax_sq_cmap zm ON zm.zc = zcs.zc
        CROSS JOIN _sarimax_sq_bk zbkc
    ), []::DOUBLE[]) AS zbf
),
_sarimax_sq_e AS (         -- OLS residuals on the w timeline (NULL where w NULL)
    SELECT zw.zt, zw.zw - coalesce(zfit.zd, 0e0) AS zev
    FROM _sarimax_sq_w zw
    LEFT JOIN (
        SELECT zx.zt, list_reduce(list_prepend(0e0, list(zx.zx * zbf.zbf[zx.zc] ORDER BY zx.zc)),
                                  lambda za3, zb3: za3 + zb3) AS zd
        FROM _sarimax_sq_xd zx, _sarimax_sq_bf zbf
        GROUP BY zx.zt, zbf.zbf
    ) zfit ON zfit.zt = zw.zt
),
_sarimax_sq_dims AS (
    SELECT (za.zn - za.zkd)::BIGINT AS zne,
           CASE WHEN q + bigq = 0 THEN 0::BIGINT
                ELSE least(greatest(20, 2 * greatest(p + s * bigp, q + s * bigq))::BIGINT,
                           (za.zn - za.zkd - 1) // 2) END AS zm,
           (p + q + bigp + bigq)::BIGINT AS zncoef2
    FROM _sarimax_sq_args za
),
_sarimax_sq_dims3 AS (
    SELECT zne, zm, zncoef2,
           CASE WHEN zncoef2 > 0 AND zne < zt0raw + zncoef2
                THEN error(concat('duckARIMA v2 start values: n_eff = ', zne,
                                  ' (model scale) is too small for the Hannan-Rissanen',
                                  ' stage-2 regression; need n_eff >= ', zt0raw + zncoef2,
                                  ' (long-AR order m = ', zm, ')'))
                ELSE zt0raw END AS zt0
    FROM (
        SELECT zne, zm, zncoef2,
               greatest(p + s * bigp + 1,
                        CASE WHEN q + bigq > 0 THEN zm + 1 + q + s * bigq ELSE 1 END)::BIGINT AS zt0raw
        FROM _sarimax_sq_dims
    )
),
_sarimax_sq_d1ok AS (      -- long-AR rows whose FULL lag window is non-missing
    SELECT ze1.zt
    FROM _sarimax_sq_e ze1, _sarimax_sq_dims3 zd3
    WHERE ze1.zt > zd3.zm AND ze1.zev IS NOT NULL
      AND (SELECT count(*) FROM _sarimax_sq_e zi
           WHERE zi.zt >= ze1.zt - zd3.zm AND zi.zt < ze1.zt
             AND zi.zev IS NOT NULL) = zd3.zm
),
_sarimax_sq_d1 AS (        -- long-AR design over the valid rows
    SELECT ze1.zt, zlx.zl, ze2.zev AS zx
    FROM _sarimax_sq_e ze1
    JOIN _sarimax_sq_d1ok zok ON zok.zt = ze1.zt
    CROSS JOIN _sarimax_sq_dims3 zd3
    CROSS JOIN LATERAL (SELECT zu.zl FROM unnest(range(1, zd3.zm + 1)) AS zu(zl)) zlx
    JOIN _sarimax_sq_e ze2 ON ze2.zt = ze1.zt - zlx.zl
),
_sarimax_sq_a1 AS (        -- long-AR coefficients (zeros fallback)
    SELECT CASE WHEN zd3.zm = 0 THEN []::DOUBLE[]
                WHEN NOT (zs1.zsol).ok
                THEN list_transform(range(1, zd3.zm + 1), lambda zz3: 0e0)
                ELSE (zs1.zsol).x END AS za1
    FROM _sarimax_sq_dims3 zd3,
         (SELECT _sarimax_solve_list(zg1.zaug, zdi.zm, 1) AS zsol
          FROM _sarimax_sq_dims3 zdi,
               (SELECT list(zv ORDER BY zj1, zj2) AS zaug
                FROM (
                    SELECT za.zl AS zj1, zb.zl AS zj2,
                           list_reduce(list_prepend(0e0, list(za.zx * zb.zx ORDER BY za.zt)),
                                       lambda zfa, zfb: zfa + zfb) AS zv
                    FROM _sarimax_sq_d1 za JOIN _sarimax_sq_d1 zb ON za.zt = zb.zt
                    GROUP BY za.zl, zb.zl
                    UNION ALL
                    SELECT za.zl, zmx.zm + 1,
                           list_reduce(list_prepend(0e0, list(za.zx * ze.zev ORDER BY za.zt)),
                                       lambda zfa, zfb: zfa + zfb)
                    FROM _sarimax_sq_d1 za
                    JOIN _sarimax_sq_e ze ON ze.zt = za.zt
                    CROSS JOIN _sarimax_sq_dims3 zmx
                    GROUP BY za.zl, zmx.zm
                )) zg1) zs1
),
_sarimax_sq_eps AS (       -- epshat on the VALID long-AR rows only
    SELECT ze1.zt, ze1.zev - coalesce(zli.zsum, 0e0) AS zeps
    FROM _sarimax_sq_e ze1
    JOIN _sarimax_sq_d1ok zok ON zok.zt = ze1.zt
    LEFT JOIN (
        SELECT za.zt, list_reduce(
                   list_prepend(0e0, list(za.zx * za1c.za1[za.zl] ORDER BY za.zl)),
                   lambda za4, zb4: za4 + zb4) AS zsum
        FROM _sarimax_sq_d1 za, _sarimax_sq_a1 za1c
        GROUP BY za.zt, za1c.za1
    ) zli ON zli.zt = ze1.zt
),
_sarimax_sq_cols AS (      -- stage-2 column spec, canonical block layout
    SELECT zcx.zc,
           CASE WHEN zcx.zc <= p + bigp THEN 1 ELSE 2 END AS zsrc,
           CASE WHEN zcx.zc <= p THEN zcx.zc
                WHEN zcx.zc <= p + bigp THEN (zcx.zc - p) * s
                WHEN zcx.zc <= p + bigp + q THEN zcx.zc - p - bigp
                ELSE (zcx.zc - p - bigp - q) * s END AS zlag
    FROM (SELECT unnest(range(1, p + q + bigp + bigq + 1)) AS zc) zcx
),
_sarimax_sq_d2ok AS (      -- stage-2 rows with every referenced cell present
    SELECT zrow.zt
    FROM (SELECT ze.zt, ze.zev FROM _sarimax_sq_e ze, _sarimax_sq_dims3 zd3
          WHERE ze.zt >= zd3.zt0) zrow
    CROSS JOIN _sarimax_sq_cols zc
    LEFT JOIN _sarimax_sq_e zej ON zc.zsrc = 1 AND zej.zt = zrow.zt - zc.zlag
    LEFT JOIN _sarimax_sq_eps zpj ON zc.zsrc = 2 AND zpj.zt = zrow.zt - zc.zlag
    WHERE zrow.zev IS NOT NULL
    GROUP BY zrow.zt
    HAVING count(*) FILTER (WHERE (zc.zsrc = 1 AND zej.zev IS NULL)
                                  OR (zc.zsrc = 2 AND zpj.zeps IS NULL)) = 0
    UNION ALL
    SELECT ze.zt                    -- no ARMA columns: every non-missing row
    FROM _sarimax_sq_e ze, _sarimax_sq_dims3 zd3
    WHERE zd3.zncoef2 = 0 AND ze.zt >= zd3.zt0 AND ze.zev IS NOT NULL
),
_sarimax_sq_d2 AS (        -- stage-2 design over the valid rows
    SELECT zok.zt, zc.zc, CASE WHEN zc.zsrc = 1 THEN zej.zev ELSE zpj.zeps END AS zx
    FROM _sarimax_sq_d2ok zok
    CROSS JOIN _sarimax_sq_cols zc
    LEFT JOIN _sarimax_sq_e zej ON zc.zsrc = 1 AND zej.zt = zok.zt - zc.zlag
    LEFT JOIN _sarimax_sq_eps zpj ON zc.zsrc = 2 AND zpj.zt = zok.zt - zc.zlag
),
_sarimax_sq_coef AS (      -- stage-2 coefficients in COLUMN order (phi,Phi,theta,Theta)
    SELECT CASE WHEN p + q + bigp + bigq = 0 THEN []::DOUBLE[]
                WHEN NOT (zs2.zsol).ok
                THEN list_transform(range(1, p + q + bigp + bigq + 1), lambda zz4: 0e0)
                ELSE (zs2.zsol).x END AS zcoef
    FROM (SELECT _sarimax_solve_list(
                     (SELECT list(zv ORDER BY zj1, zj2)
                      FROM (
                          SELECT za.zc::BIGINT AS zj1, zb.zc::BIGINT AS zj2,
                                 list_reduce(list_prepend(0e0, list(za.zx * zb.zx ORDER BY za.zt)),
                                             lambda zfa, zfb: zfa + zfb) AS zv
                          FROM _sarimax_sq_d2 za JOIN _sarimax_sq_d2 zb ON za.zt = zb.zt
                          GROUP BY za.zc, zb.zc
                          UNION ALL
                          SELECT za.zc::BIGINT, (p + q + bigp + bigq + 1)::BIGINT,
                                 list_reduce(list_prepend(0e0, list(za.zx * ze.zev ORDER BY za.zt)),
                                             lambda zfa, zfb: zfa + zfb)
                          FROM _sarimax_sq_d2 za JOIN _sarimax_sq_e ze ON ze.zt = za.zt
                          GROUP BY za.zc
                      )),
                     p + q + bigp + bigq, 1) AS zsol) zs2
),
_sarimax_sq_sig AS (
    SELECT CASE WHEN zsg.zs2v IS NOT NULL AND isfinite(zsg.zs2v) AND zsg.zs2v > 0e0 THEN zsg.zs2v
                WHEN zwv.zvw IS NOT NULL AND isfinite(zwv.zvw) AND zwv.zvw > 0e0 THEN zwv.zvw
                ELSE 1e0 END AS zsig2
    FROM (
        SELECT list_reduce(list_prepend(0e0, list(zres * zres ORDER BY zt)),
                           lambda zfa, zfb: zfa + zfb) / greatest(count(*), 1) AS zs2v
        FROM (
            SELECT ze.zt, ze.zev - coalesce(zfld.zsum, 0e0) AS zres
            FROM _sarimax_sq_e ze
            JOIN _sarimax_sq_d2ok zok ON zok.zt = ze.zt
            LEFT JOIN (
                SELECT za.zt, list_reduce(
                           list_prepend(0e0, list(za.zx * zcf.zcoef[za.zc] ORDER BY za.zc)),
                           lambda za5, zb5: za5 + zb5) AS zsum
                FROM _sarimax_sq_d2 za, _sarimax_sq_coef zcf
                GROUP BY za.zt, zcf.zcoef
            ) zfld ON zfld.zt = ze.zt
        )
    ) zsg,
    (SELECT list_reduce(list_prepend(0e0, list(zw * zw ORDER BY zt) FILTER (zw IS NOT NULL)),
                        lambda zfa, zfb: zfa + zfb) / greatest(count(zw), 1) AS zvw
     FROM _sarimax_sq_w) zwv
),
_sarimax_sq_pbase AS (     -- remap stage-2 columns to canonical order, guard finite
    SELECT zbfc.zbf,
           list_transform(range(1, p + q + bigp + bigq + 1), lambda zc5:
               (list_transform([CASE WHEN zc5 <= p THEN zcf.zcoef[zc5]
                                     WHEN zc5 <= p + q THEN zcf.zcoef[p + bigp + (zc5 - p)]
                                     WHEN zc5 <= p + q + bigp THEN zcf.zcoef[p + (zc5 - p - q)]
                                     ELSE zcf.zcoef[p + bigp + q + (zc5 - p - q - bigp)] END],
                    lambda zv5: CASE WHEN coalesce(isfinite(zv5), false)
                                     THEN zv5 ELSE 0e0 END))[1]) AS zarma,
           zsg.zsig2
    FROM _sarimax_sq_bf zbfc, _sarimax_sq_coef zcf, _sarimax_sq_sig zsg
),
_sarimax_sq_pb2 AS (       -- rescale tau by the AR DC gain: the OLS estimates
    -- observation-scale trend coefficients, but tau drives the STATE
    -- intercept, whose stationary observation effect is tau / phi*(1) --
    -- so tau0 = lambda_OLS * (1 - sum phi)(1 - sum Phi). beta (exog) enters
    -- the observation equation directly and is NOT rescaled. Guarded to 1
    -- when the gain is tiny or non-finite.
    SELECT list_transform(range(1, len(zbf) + 1), lambda zc7:
               CASE WHEN zc7 <= ktrend THEN zbf[zc7] * zgadj ELSE zbf[zc7] END) AS zbf,
           zarma, zsig2
    FROM (
        SELECT zbf, zarma, zsig2,
               (list_transform([(1e0 - list_reduce(
                                     list_prepend(0e0, list_slice(zarma, 1, p)),
                                     lambda za7, zb7: za7 + zb7))
                                * (1e0 - list_reduce(
                                       list_prepend(0e0, list_slice(zarma, p + q + 1,
                                                                    p + q + bigp)),
                                       lambda za8, zb8: za8 + zb8))],
                   lambda zg: CASE WHEN coalesce(isfinite(zg), false) AND abs(zg) > 1e-6
                                   THEN zg ELSE 1e0 END))[1] AS zgadj
        FROM _sarimax_sq_pbase
    )
),
_sarimax_sq_pick AS (      -- shrink-until-valid, smallest z wins
    SELECT zshrink, zcand, zx0c
    FROM (
        SELECT zz.zshrink,
               zb6.zbf
                 || list_transform(zb6.zarma, lambda za6: za6 * power(5e-1, zz.zshrink))
                 || (CASE WHEN conc THEN []::DOUBLE[] ELSE [zb6.zsig2] END) AS zcand,
               TRY(_sarimax_untransform_params_v2(
                       zb6.zbf
                         || list_transform(zb6.zarma, lambda za6: za6 * power(5e-1, zz.zshrink))
                         || (CASE WHEN conc THEN []::DOUBLE[] ELSE [zb6.zsig2] END),
                       ktrend + r, p, q, bigp, bigq, conc)) AS zx0c
        FROM _sarimax_sq_pb2 zb6, unnest(range(0, 61)) AS zz(zshrink)
    )
    WHERE zx0c IS NOT NULL
      AND len(list_filter(zx0c, lambda ze6: coalesce(isfinite(ze6), false))) = len(zx0c)
      AND len(zx0c) = ktrend + r + p + q + bigp + bigq
                      + CASE WHEN conc THEN 0 ELSE 1 END
    ORDER BY zshrink
    LIMIT 1
)
SELECT CASE WHEN zp.zx0c IS NULL
            THEN error('duckARIMA v2 start values: no stationary/invertible starting point found')
            ELSE zp.zx0c END AS x0,
       zp.zcand AS params0
FROM (SELECT 1 AS zone) zdum
LEFT JOIN _sarimax_sq_pick zp ON true;

-- ============================================================================
-- SECTION 4: gradient-probe sharing (PERFORMANCE only -- the numeric
-- contract is BITWISE identity with _sarimax_ll_c_v2). Physically placed
-- BEFORE subsection 3d: DuckDB binds scalar-macro references inside table
-- macros at CREATE time, so these must exist before _sarimax_bfgs_v2 is
-- created (same load-order rule as the layer files themselves).
--
-- Motivation: in _sarimax_bfgs_v2 every gradient evaluation runs 2*np full
-- O(n*k^3) kernel passes, yet probes that perturb only a HEAD coordinate
-- (tau/beta, index <= ktrend + r -- identity-transformed, so the
-- unconstrained perturbation IS the constrained perturbation) leave the
-- ARMA+sigma2 slice unchanged: T, T', RQR and the filter anchor covariance
-- P1f are identical, hence the whole covariance recursion -- the F_t
-- sequence and the un-normalized gain vectors tpz_t = (T P_t) e_1 -- is THE
-- SAME for all such probes (the missing-step pattern comes from ylist NULLs,
-- which head perturbations never change). Only the O(n*k^2) mean recursion
-- differs, so it alone is re-run per head probe.
--
--   4a. _sarimax_kf_gains_v2 : one covariance-only pass -> shared gains
--   4b. _sarimax_ll_mean_v2  : mean-only pass over shared gains
--
-- BFGS INTEGRATION (measured engine law -- decides where routing pays):
-- DuckDB executes list-lambda folds VECTORIZED ACROSS ROWS, so a batched
-- kernel call site costs a row-count-INDEPENDENT expression-tree walk
-- (~0.25-0.35 s/site/evaluation for these kernels at n ~ 150-300) plus tiny
-- per-row value work (~1.6 ms/row at k = 2, ~80 ms at k = 14, ~540 ms at
-- k = 27; k = kdiff + karma). Sharing removes head-probe value work but ADDS
-- one covariance-site walk, so _sarimax_bfgs_v2 routes head probes through
-- the split ONLY under the CONSTANT gate ktrend + r >= 2 AND kdiff + karma
-- >= 20 (measured win ~1.2x full fit on the k = 27 fixture class; k <= 14
-- is a wash or a loss). Ungated models run the pre-SECTION-4 plan (the
-- rows-gated unnest sources make the gains sites zero-cost then). Three
-- engine hazards shaped the wiring (all measured, tests/test_gradshare.py):
--   * a projection column holding the gains struct gets RE-INLINED by the
--     optimizer into every consuming probe row after a fan-out -- the gains
--     therefore live behind LATERAL AGGREGATES (max() over a one-row
--     source), which are hard materialization boundaries;
--   * a correlated LATERAL whose inner query nests derived tables plans
--     catastrophically (~1.7 s/iteration even evaluating nothing) -- the
--     probe laterals keep the original single-pre-projection shape, with
--     ydlist/clist as inline args (_sarimax_ll_mean_v2 self-binds them);
--   * every big expression tree TEXTUALLY present in the recursive member
--     costs ~0.05-0.15 s/iteration even when never evaluated (the resident
--     tax) -- so the restart batch (fires <= once per fit; nothing to save)
--     and _sarimax_bse_v2 (optional per the plan; same wash economics as
--     the gradient at its np^2/2 Hessian cells) are NOT routed.
--
-- BITWISE-IDENTITY DISCIPLINE: every arithmetic expression below is copied
-- form-for-form from _sarimax_ll_c_v2 (`tpz[i]*v/f`, never `(tpz[i]/f)*v`;
-- same fold order; same CASE-guard shapes; every matrix op bound as its own
-- value before msym sees it). tests/test_gradshare.py asserts EXACT (==)
-- equality of (ll, scale2) against the full kernel on every fixture.
--
-- ANCHOR NOTE (documented deviation from the avec shortcut): the trend
-- anchor a1 is linear in c1 = sum(tau) in exact arithmetic, so
-- a1 = c1 * avec with avec the unit-c1 anchor direction -- but NOT in
-- float64: _sarimax_a1_v2 embeds c1 in the RHS of its Gauss-Jordan solve,
-- and fl(c1 * fl(1/x)) != fl(c1/x) in general. avec is still exported (it
-- IS the exact direction, useful diagnostically), but _sarimax_ll_mean_v2
-- re-runs the exact c1-scaled solve per probe (karma^3 once -- negligible)
-- so the anchor is bit-identical to the full kernel's.
--
-- Lambda discipline: identical to section 3 -- z-prefixed lambda vars; cpar
-- / ylist / xmat / degs must be materialized values or plain columns.
-- _sarimax_ll_mean_v2 additionally self-binds gains/ydlist/clist once, so an
-- expression argument costs one evaluation, never per-element -- but gains
-- should still be bound as a column so the covariance pass itself is shared
-- across probe ROWS. Missingness is read from ylist NULLs only (the v2
-- contract: exog carries no NULLs).
-- ============================================================================

-- ---- 4a. shared covariance pass -------------------------------------------------

-- One covariance-only filter pass at a CONSTRAINED v2 vector cpar. Arguments
-- exactly as _sarimax_ll_c_v2 (xmat/degs accepted for signature symmetry;
-- the covariance recursion needs neither). Returns a STRUCT:
--   tmat    DOUBLE[]  augmented transition T (k*k row-major)
--   tarma   DOUBLE[]  ARMA-block transition (karma*karma) -- anchor solves
--   k, karma, kdiff, cidx, burn                            -- dimensions
--   fs      DOUBLE[]  F_t = P_t[1,1], t = 1..n (recorded EVERY step, exactly
--                     as the full kernel computes it, missing steps included)
--   kmat    DOUBLE[]  tpz_t = (T P_t) e_1 flattened row-major (n x k)
--   sumlogf DOUBLE    sum of ln F_t over non-missing t > burn (NULL-poisoned
--                     when a counted F_t <= 0, like the full kernel)
--   cnt     BIGINT    number of counted steps
--   avec    DOUBLE[]  unit-c1 anchor direction (see ANCHOR NOTE above)
--   cflag   BOOLEAN   concentrated-scale flag
--   sig2    DOUBLE    the sigma2 parameter (NULL when concentrated)
CREATE OR REPLACE MACRO _sarimax_kf_gains_v2(cpar, ylist, xmat, degs,
                                             r, p, q, bigp, bigq, s, d, sd,
                                             ktrend, conc) AS (
  (list_transform([struct_pack(
      zka := _sarimax_k_states(p, q, bigp, bigq, s),
      zkd := _sarimax_kdiff(d, sd, s),
      zn := len(ylist),
      zphistar := _sarimax_expand_ar(
          list_slice(cpar, ktrend + r + 1, ktrend + r + p),
          list_slice(cpar, ktrend + r + p + q + 1, ktrend + r + p + q + bigp), s),
      zthetastar := _sarimax_expand_ma(
          list_slice(cpar, ktrend + r + p + 1, ktrend + r + p + q),
          list_slice(cpar, ktrend + r + p + q + bigp + 1,
                     ktrend + r + p + q + bigp + bigq), s),
      zsigma2 := CASE WHEN conc THEN 1e0
                      ELSE cpar[ktrend + r + p + q + bigp + bigq + 1] END,
      zsig2p := CASE WHEN conc THEN NULL::DOUBLE
                     ELSE cpar[ktrend + r + p + q + bigp + bigq + 1] END,
      -- the missing mask is the only mean-side input the covariance needs
      zmiss := list_transform(range(1, len(ylist) + 1), lambda zt:
                   ylist[zt] IS NULL))],
   lambda zl1:
     (list_transform([struct_pack(
         zk := zl1.zkd + zl1.zka,
         ztm := _sarimax_build_t_v2(zl1.zphistar, zl1.zka, d, sd, s),
         zta := _sarimax_build_t(zl1.zphistar, zl1.zka),
         zrv := _sarimax_build_r_v2(zl1.zthetastar, zl1.zka, d, sd, s),
         zrva := _sarimax_build_r(zl1.zthetastar, zl1.zka))],
      lambda zl2:
        (list_transform([struct_pack(
            ztt := _sarimax_mtrans(zl2.ztm, zl2.zk, zl2.zk),
            zrqr := _sarimax_build_rqr(zl2.zrv, zl1.zsigma2, zl2.zk),
            zrqra := _sarimax_build_rqr(zl2.zrva, zl1.zsigma2, zl1.zka))],
         lambda zl3:
           -- ARMA-block P1 by 30 doubling iterations (bound before msym) --
           -- verbatim from _sarimax_ll_c_v2
           (list_transform([(list_reduce(
                    [struct_pack(zsm := zl3.zrqra, zaa := zl2.zta)]
                      || list_transform(range(1, 31), lambda zd2:
                           struct_pack(zsm := []::DOUBLE[], zaa := []::DOUBLE[])),
                    lambda zacc, zel:
                      (list_transform([_sarimax_mmul(zacc.zaa, zacc.zsm,
                                                     zl1.zka, zl1.zka, zl1.zka)],
                       lambda zas:
                         (list_transform([_sarimax_mtrans(zacc.zaa, zl1.zka, zl1.zka)],
                          lambda zat2:
                            (list_transform([_sarimax_mmul(zas, zat2,
                                                           zl1.zka, zl1.zka, zl1.zka)],
                             lambda zasat:
                               struct_pack(
                                 zsm := _sarimax_madd(zacc.zsm, zasat),
                                 zaa := _sarimax_mmul(zacc.zaa, zacc.zaa,
                                                      zl1.zka, zl1.zka, zl1.zka))))[1]
                         ))[1]
                      ))[1]
                 )).zsm],
            lambda zs30:
           (list_transform([_sarimax_msym(zs30, zl1.zka)],
            lambda zsig:
              -- blockdiag(1e6 I_kdiff, Sigma), the UNSHIFTED P1
              (list_transform([list_transform(range(1, zl2.zk * zl2.zk + 1), lambda zidx:
                   CASE
                     WHEN (zidx - 1) // zl2.zk + 1 <= zl1.zkd
                          AND (zidx - 1) % zl2.zk + 1 <= zl1.zkd
                     THEN CASE WHEN (zidx - 1) // zl2.zk = (zidx - 1) % zl2.zk
                               THEN 1e6 ELSE 0e0 END
                     WHEN (zidx - 1) // zl2.zk + 1 > zl1.zkd
                          AND (zidx - 1) % zl2.zk + 1 > zl1.zkd
                     THEN zsig[((zidx - 1) // zl2.zk - zl1.zkd) * zl1.zka
                               + ((zidx - 1) % zl2.zk + 1 - zl1.zkd)]
                     ELSE 0e0 END)],
               lambda zp1b:
                 -- shift conjugation of the covariance: Var[gamma_1] =
                 -- T P1 T' + RQR (kdiff > 0), staged exactly as the kernel
                 (list_transform([_sarimax_mmul(zl2.ztm, zp1b, zl2.zk, zl2.zk, zl2.zk)],
                  lambda ztp1:
                    (list_transform([_sarimax_mmul(ztp1, zl3.ztt,
                                                   zl2.zk, zl2.zk, zl2.zk)],
                     lambda ztp1t:
                       (list_transform([struct_pack(
                            zp1 := CASE WHEN zl1.zkd = 0 THEN zp1b
                                        ELSE (list_transform([_sarimax_madd(ztp1t, zl3.zrqr)],
                                                  lambda zpm: _sarimax_msym(zpm, zl2.zk)))[1]
                                   END,
                            zw := _sarimax_a1_v2(zl2.zta, zl1.zka, d, sd, s, 1e0))],
                        lambda zl8:
                          (list_transform([_sarimax_mmul(zl2.ztm, zl8.zw,
                                                         zl2.zk, zl2.zk, 1)],
                           lambda ztw:
                             (list_transform([CASE WHEN zl1.zkd = 0 THEN zl8.zw
                                  ELSE list_transform(range(1, zl2.zk + 1), lambda zi5:
                                           ztw[zi5] + CASE WHEN zi5 = zl1.zkd + 1
                                                           THEN 1e0 ELSE 0e0 END) END],
                              lambda zavec:
                                -- covariance fold in strict t order; F_t and
                                -- tpz_t are recorded EVERY step; sumlogf/cnt
                                -- use the data-determined missing mask
                                (list_transform([(list_reduce(
                                   [struct_pack(zp2 := zl8.zp1,
                                                zfs := []::DOUBLE[], zkm := []::DOUBLE[],
                                                zslf := 0e0, zcnt := 0e0,
                                                zms := false, zti := 0::BIGINT)]
                                     || list_transform(range(1, zl1.zn + 1), lambda zt:
                                          struct_pack(zp2 := []::DOUBLE[],
                                                      zfs := []::DOUBLE[], zkm := []::DOUBLE[],
                                                      zslf := 0e0, zcnt := 0e0,
                                                      zms := (zl1.zmiss)[zt], zti := zt)),
                                   lambda zacc, zel:
                                     (list_transform([struct_pack(
                                          zf := (zacc.zp2)[1],
                                          ztp := _sarimax_mmul(zl2.ztm, zacc.zp2,
                                                               zl2.zk, zl2.zk, zl2.zk))],
                                      lambda zi1:
                                        (list_transform([list_transform(range(1, zl2.zk + 1),
                                                             lambda zi2:
                                                                 (zi1.ztp)[(zi2 - 1) * zl2.zk + 1])],
                                         lambda ztpz:
                                           (list_transform([struct_pack(
                                                ztpt := _sarimax_mmul(zi1.ztp, zl3.ztt,
                                                                      zl2.zk, zl2.zk, zl2.zk),
                                                zoutr := list_transform(
                                                    range(1, zl2.zk * zl2.zk + 1), lambda zi4:
                                                        ztpz[(zi4 - 1) // zl2.zk + 1]
                                                        * ztpz[(zi4 - 1) % zl2.zk + 1] / zi1.zf))],
                                            lambda zb1:
                                              (list_transform([_sarimax_msub(zb1.ztpt, zb1.zoutr)],
                                               lambda zmsb:
                                                 (list_transform([CASE WHEN zel.zms
                                                                       THEN zb1.ztpt
                                                                       ELSE zmsb END],
                                                  lambda zpre:
                                                    (list_transform([_sarimax_madd(zpre, zl3.zrqr)],
                                                     lambda zpu:
                                                       struct_pack(
                                                         zp2 := _sarimax_msym(zpu, zl2.zk),
                                                         zfs := list_append(zacc.zfs, zi1.zf),
                                                         zkm := zacc.zkm || ztpz,
                                                         zslf := zacc.zslf
                                                             + CASE WHEN zel.zms
                                                                         OR zel.zti <= zl1.zkd
                                                                    THEN 0e0
                                                                    WHEN zi1.zf > 0e0
                                                                    THEN ln(zi1.zf)
                                                                    ELSE NULL END,
                                                         zcnt := zacc.zcnt
                                                             + CASE WHEN NOT zel.zms
                                                                         AND zel.zti > zl1.zkd
                                                                    THEN 1e0 ELSE 0e0 END,
                                                         zms := false,
                                                         zti := 0::BIGINT)))[1]
                                                 ))[1]
                                              ))[1]
                                           ))[1]
                                        ))[1]
                                     ))[1]
                                ))],
                                 lambda zfr:
                                   struct_pack(
                                     tmat := zl2.ztm,
                                     tarma := zl2.zta,
                                     k := zl2.zk,
                                     karma := zl1.zka,
                                     kdiff := zl1.zkd,
                                     cidx := (zl1.zkd + 1)::BIGINT,
                                     burn := zl1.zkd::BIGINT,
                                     fs := zfr.zfs,
                                     kmat := zfr.zkm,
                                     sumlogf := zfr.zslf,
                                     cnt := zfr.zcnt::BIGINT,
                                     avec := zavec,
                                     cflag := conc,
                                     sig2 := zl1.zsig2p)))[1]
                             ))[1]
                          ))[1]
                       ))[1]
                    ))[1]
                 ))[1]
              ))[1]
           ))[1]
           ))[1]
        ))[1]
     ))[1]
  ))[1]
);

-- ---- 4b. mean-only pass over shared gains ---------------------------------------

-- Loglikelihood struct at a head-perturbed parameter point, given the shared
-- gains. ydlist = y_t - x_t' beta with the PROBE's beta (NULL at missing t,
-- built with the exact fold shape of _sarimax_ll_c_v2's zyd); clist = the
-- PROBE's trend values c_1..c_{n+1} (i.e. _sarimax_trend_c(degs, tau, 1,
-- n+1), matching the kernel's zcl -- the one-step offset for kdiff > 0 is
-- applied HERE). Returns STRUCT(ll, scale2), bitwise-identical to
-- _sarimax_ll_c_v2 at the same parameter vector whenever that vector's
-- ARMA+sigma2 slice equals the one gains was built from.
CREATE OR REPLACE MACRO _sarimax_ll_mean_v2(gains, ydlist, clist) AS (
  (list_transform([struct_pack(zg := gains, zyd := ydlist, zcl := clist)],
   lambda zb0:
     (list_transform([struct_pack(
          zk := (zb0.zg).k,
          zkd := (zb0.zg).kdiff,
          zka := (zb0.zg).karma,
          zn := len(zb0.zyd),
          zc1 := (zb0.zcl)[1])],
      lambda zl1:
        -- anchor: the EXACT c1-scaled solve (section header ANCHOR NOTE);
        -- _sarimax_a1_v2 only uses d + s*sd, so (kdiff, 0, 1) reproduces it
        (list_transform([_sarimax_a1_v2((zb0.zg).tarma, zl1.zka,
                                        zl1.zkd, 0, 1, zl1.zc1)],
         lambda za1u:
           (list_transform([_sarimax_mmul((zb0.zg).tmat, za1u, zl1.zk, zl1.zk, 1)],
            lambda zta1:
              (list_transform([struct_pack(
                   za1 := CASE WHEN zl1.zkd = 0 THEN za1u
                               ELSE list_transform(range(1, zl1.zk + 1), lambda zi5:
                                        zta1[zi5]
                                        + CASE WHEN zi5 = zl1.zkd + 1
                                               THEN zl1.zc1 ELSE 0e0 END) END,
                   -- per-step intercept: c_t unshifted, c_{t+1} shifted
                   zcs := CASE WHEN zl1.zkd = 0
                               THEN list_slice(zb0.zcl, 1, zl1.zn)
                               ELSE list_slice(zb0.zcl, 2, zl1.zn + 1) END)],
               lambda zl10:
                 -- mean fold in strict t order (v1-kernel fold shape; per-step
                 -- yd, intercept and t ride in zydv/zct/zti)
                 (list_transform([(list_reduce(
                     [struct_pack(za2 := zl10.za1, zcnt := 0e0, zssq := 0e0,
                                  zydv := 0e0, zct := 0e0, zti := 0::BIGINT)]
                       || list_transform(range(1, zl1.zn + 1), lambda zt:
                            struct_pack(za2 := []::DOUBLE[], zcnt := 0e0, zssq := 0e0,
                                        zydv := (zb0.zyd)[zt],
                                        zct := (zl10.zcs)[zt], zti := zt)),
                     lambda zacc, zel:
                       (list_transform([struct_pack(
                            zv := zel.zydv - (zacc.za2)[1],
                            zf := ((zb0.zg).fs)[zel.zti],
                            zta2 := _sarimax_mmul((zb0.zg).tmat, zacc.za2,
                                                  zl1.zk, zl1.zk, 1))],
                        lambda zi1:
                          struct_pack(
                            za2 := list_transform(range(1, zl1.zk + 1), lambda zi3:
                                       (zi1.zta2)[zi3]
                                       + (CASE WHEN zi1.zv IS NULL THEN 0e0
                                               ELSE ((zb0.zg).kmat)[(zel.zti - 1) * zl1.zk + zi3]
                                                    * zi1.zv / zi1.zf END)
                                       + (CASE WHEN zi3 = zl1.zkd + 1
                                               THEN zel.zct ELSE 0e0 END)),
                            zcnt := zacc.zcnt
                                + CASE WHEN zi1.zv IS NOT NULL AND zel.zti > zl1.zkd
                                       THEN 1e0 ELSE 0e0 END,
                            zssq := zacc.zssq
                                + CASE WHEN zi1.zv IS NULL OR zel.zti <= zl1.zkd
                                       THEN 0e0
                                       ELSE zi1.zv * zi1.zv / zi1.zf END,
                            zydv := 0e0, zct := 0e0, zti := 0::BIGINT)))[1]
                  ))],
                  lambda zfr:
                    (list_transform([CASE WHEN (zb0.zg).cflag
                         THEN CASE WHEN zfr.zcnt > 0e0 AND zfr.zssq > 0e0
                                   THEN -5e-1 * (zfr.zcnt * ln(2e0 * pi())
                                                 + (zb0.zg).sumlogf
                                                 + zfr.zcnt * ln(zfr.zssq / zfr.zcnt)
                                                 + zfr.zcnt)
                                   ELSE NULL END
                         ELSE -5e-1 * (zfr.zcnt * ln(2e0 * pi())
                                       + (zb0.zg).sumlogf + zfr.zssq) END],
                     lambda zllr:
                       struct_pack(
                         ll := CASE WHEN zllr IS NOT NULL AND isfinite(zllr)
                                    THEN zllr ELSE NULL END,
                         scale2 := CASE WHEN (zb0.zg).cflag
                                        THEN CASE WHEN zfr.zcnt > 0e0
                                                       AND isfinite(zfr.zssq)
                                                  THEN zfr.zssq / zfr.zcnt
                                                  ELSE NULL END
                                        ELSE (zb0.zg).sig2 END)))[1]
                 ))[1]
              ))[1]
           ))[1]
        ))[1]
     ))[1]
  ))[1]
);


-- ---- 3d. BFGS optimizer (v2 objective; v1 section 2c pinned constants) ---------

-- Structural mirror of _sarimax_bfgs (section 2c) -- same two-phase lazy
-- Armijo line search, central-difference gradients, BFGS update with the
-- y's-guard, the two documented noise-floor stall certificates, and the
-- single hash-perturbed restart -- with the objective
-- f(x) = -(_sarimax_ll_x_v2(...)).ll over the v2 unconstrained space
-- (np = ktrend + r + p + q + bigp + bigq + (conc ? 0 : 1)). ylist (column y,
-- with NULLs), UNdifferenced exog and the degs table are precomputed ONCE
-- into columns (zwl / zxm / zdg) and pre-projected into every innermost
-- derived table that evaluates the kernel (the correlated-LATERAL lambda
-- rule, section 2 header). Output adds scale2 (the concentrated scale at the
-- optimum when conc, else the fitted sigma2 parameter).
--
-- DOCUMENTED DEVIATION (kdiff-aware gradient step): h_i = 1e-7 *
-- max(1, |x_i|) exactly as v1 when kdiff = 0, but 1e-5 * max(1, |x_i|) when
-- kdiff > 0. The approximate-diffuse kappa = 1e6 initialization makes the
-- loglikelihood rough at the ~1e-7 ABSOLUTE level along ARMA/sigma2
-- coordinates on nodiff_sarimax_011_011_12 (measured: quadratic-fit
-- residuals over 1e-8-spaced samples); central differences at h = 1e-7 turn
-- that into gradient noise of ORDER ONE, which corrupts the BFGS curvature
-- pairs and stalls the line search at points whose true gradient is still
-- ~0.5 (verified: h = 1e-5 finite differences at the h = 1e-7 stall point
-- recover the true gradient, and a reference BFGS run using them descends a
-- further 1.9e-3 in ll, ending ABOVE statsmodels' reported optimum). At
-- h = 1e-5 the gradient noise is ~1e-2 and the truncation error stays below
-- ~1e-4 on every fixture coordinate.
--
-- DOCUMENTED DEVIATION (restarted-exhaustion certificate, kdiff > 0 only):
-- after the hash-perturbed restart has already happened, a SECOND
-- line-search exhaustion (31 rejected candidates down to alpha = 2^-30 at
-- two independent points) certifies convergence regardless of the reported
-- gradient, which at the diffuse noise floor is uninformative (see above);
-- the acceptance suite separately asserts the endpoint's ll >= statsmodels'
-- ll - 1e-8 through the T1-validated kernel, which is the substantive gate.
-- The FIRST exhaustion still restarts, preserving the second-opinion probe.
--
-- DOCUMENTED DEVIATION (stall-certificate gradient ceiling 1e-2, was 1e-4 in
-- v1): v2 surfaces mix near-collinear regressors (a trend term plus an exog
-- column that itself carries a trend, kitchen_sink) with extreme curvature
-- anisotropy -- the drift coordinate's second derivative is ~1e6, so the
-- smallest gradient any float64 optimizer can REACH there is about
-- sqrt(2 * f'' * eps|f|) ~ 4e-4: positioning the coordinate more finely
-- changes f by less than evaluation noise. Measured on kitchen_sink at this
-- engine's stall point (ll BETTER than statsmodels' reported optimum by
-- 1.4e-2, and within 1.8e-8 of a 20k-step Nelder-Mead valley floor):
-- scipy's own BFGS aborts there with "precision loss" at |g|inf = 8.6e-2,
-- and statsmodels' reported optimum carries |g|inf = 11.6 through the same
-- loglikelihood. The certificates still fire ONLY on line-search exhaustion
-- (31 rejected candidates down to alpha = 2^-30) or a
-- machine-epsilon-flat accepted step, and the acceptance suite separately
-- asserts ll >= statsmodels' ll - 1e-8 at the endpoint.
CREATE OR REPLACE MACRO _sarimax_bfgs_v2(y_tbl, exog_tbl, degs_tbl,
                                         r, p, q, bigp, bigq, s, d, sd,
                                         ktrend, conc) AS TABLE
WITH RECURSIVE
_sarimax_bf2_pc AS (
    SELECT (SELECT list(y ORDER BY t) FROM query_table(y_tbl)) AS zwl,
           coalesce((SELECT list(zxr ORDER BY t)
                     FROM (SELECT t, list(x ORDER BY j) AS zxr
                           FROM query_table(exog_tbl) GROUP BY t)),
                    []::DOUBLE[][]) AS zxm,
           (SELECT coalesce(list(degree ORDER BY idx), []::BIGINT[])
            FROM query_table(degs_tbl)) AS zdg,
           (ktrend + r + p + q + bigp + bigq
            + CASE WHEN conc THEN 0 ELSE 1 END)::BIGINT AS znp,
           zsp.x0 AS zx0
    FROM _sarimax_start_params_v2(y_tbl, exog_tbl, degs_tbl,
                                  r, p, q, bigp, bigq, s, d, sd, ktrend, conc) zsp
),
_sarimax_bf2_it USING KEY (zkk) AS (
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
                   / (2e0 * (CASE WHEN d + s * sd > 0 THEN 1e-5 ELSE 1e-7 END) * greatest(1e0, abs(za1.zx[zi])))) AS zg_new
        FROM (
            SELECT zpc.zx0 AS zx, zpc.znp,
                   CASE WHEN ktrend + r >= 2
                             AND _sarimax_kdiff(d, sd, s)
                                 + _sarimax_k_states(p, q, bigp, bigq, s) >= 20
                        THEN 0e0 - (_sarimax_ll_mean_v2(zgl0.zgn0,
                                 list_transform(range(1, len(zpc.zwl) + 1), lambda zt9:
                                     zpc.zwl[zt9] - list_reduce(
                                         list_prepend(0e0, list_transform(range(1, r + 1),
                                             lambda zj9: zpc.zxm[zt9][zj9] * zpc.zx0[ktrend + zj9])),
                                         lambda za9, zb9: za9 + zb9)),
                                 _sarimax_trend_c(zpc.zdg, list_slice(zpc.zx0, 1, ktrend),
                                                  1, len(zpc.zwl) + 1))).ll
                        ELSE 0e0 - (_sarimax_ll_x_v2(zpc.zx0, zpc.zwl, zpc.zxm,
                                        zpc.zdg, r, p, q, bigp, bigq, s, d, sd,
                                        ktrend, conc)).ll END AS zfx,
                   zag.zfpm
            FROM _sarimax_bf2_pc zpc
            CROSS JOIN LATERAL (
                -- SECTION 4 anchor gains: ONE shared covariance pass at x0.
                -- The aggregate is a hard materialization boundary (without
                -- it the optimizer re-inlines the pass into every probe row
                -- -- measured); the rows-gated unnest makes the whole site
                -- zero-cost when the constant SECTION-4 gate is off (an
                -- empty source never initializes the expression tree).
                SELECT max(_sarimax_kf_gains_v2(
                           _sarimax_transform_params_v2(zx0c, ktrend + r,
                                                        p, q, bigp, bigq, conc),
                           zwlc, zxmc, zdgc,
                           r, p, q, bigp, bigq, s, d, sd, ktrend, conc)) AS zgn0
                FROM (SELECT zpc.zx0 AS zx0c, zpc.zwl AS zwlc, zpc.zxm AS zxmc,
                             zpc.zdg AS zdgc
                      FROM unnest(CASE WHEN ktrend + r >= 2
                                       AND _sarimax_kdiff(d, sd, s)
                                           + _sarimax_k_states(p, q, bigp, bigq, s) >= 20
                                       THEN [1] ELSE []::BIGINT[] END) AS zu9(zn9))
            ) zgl0
            CROSS JOIN LATERAL (
                -- SECTION 4 routing: head probes (idx <= 2*(ktrend+r)) ride
                -- the mean path over zgn0's shared covariance pass (same
                -- constant gate); tail probes run the full kernel unchanged.
                -- ydlist/clist are inline args (_sarimax_ll_mean_v2 binds its
                -- arguments once per call); the perturbed head coordinate is
                -- expressed element-wise with the EXACT perturbation formula,
                -- so values are bitwise those of the full-kernel path.
                SELECT list(zval ORDER BY zidx) AS zfpm
                FROM (
                    SELECT zidx,
                           CASE WHEN ktrend + r >= 2
                                     AND _sarimax_kdiff(d, sd, s)
                                         + _sarimax_k_states(p, q, bigp, bigq, s) >= 20
                                     AND zidx <= 2 * (ktrend + r)
                                THEN 0e0 - (_sarimax_ll_mean_v2(
                                    zgnc,
                                    list_transform(range(1, len(zwlc) + 1), lambda zt9:
                                        zwlc[zt9] - list_reduce(
                                            list_prepend(0e0,
                                                list_transform(range(1, r + 1), lambda zj9:
                                                    zxmc[zt9][zj9]
                                                    * (CASE WHEN ktrend + zj9 = (zidx + 1) // 2
                                                            THEN zxc[ktrend + zj9]
                                                                 + (CASE WHEN zidx % 2 = 1
                                                                         THEN 1e0 ELSE -1e0 END)
                                                                 * (CASE WHEN d + s * sd > 0
                                                                         THEN 1e-5 ELSE 1e-7 END)
                                                                 * greatest(1e0, abs(zxc[ktrend + zj9]))
                                                            ELSE zxc[ktrend + zj9] END))),
                                            lambda za9, zb9: za9 + zb9)),
                                    _sarimax_trend_c(zdgc,
                                        list_transform(range(1, ktrend + 1), lambda zg9:
                                            CASE WHEN zg9 = (zidx + 1) // 2
                                                 THEN zxc[zg9]
                                                      + (CASE WHEN zidx % 2 = 1
                                                              THEN 1e0 ELSE -1e0 END)
                                                      * (CASE WHEN d + s * sd > 0
                                                              THEN 1e-5 ELSE 1e-7 END)
                                                      * greatest(1e0, abs(zxc[zg9]))
                                                 ELSE zxc[zg9] END),
                                        1, len(zwlc) + 1))).ll
                                ELSE 0e0 - (_sarimax_ll_x_v2(
                                    list_transform(zxc, lambda zxe, zxi:
                                        CASE WHEN zxi = (zidx + 1) // 2
                                             THEN zxe + (CASE WHEN zidx % 2 = 1 THEN 1e0 ELSE -1e0 END)
                                                        * (CASE WHEN d + s * sd > 0 THEN 1e-5 ELSE 1e-7 END) * greatest(1e0, abs(zxe))
                                             ELSE zxe END),
                                    zwlc, zxmc, zdgc, r, p, q, bigp, bigq, s, d, sd,
                                    ktrend, conc)).ll END AS zval
                    FROM (SELECT zpc.zx0 AS zxc, zpc.zwl AS zwlc, zpc.zxm AS zxmc,
                                 zpc.zdg AS zdgc, zgl0.zgn0 AS zgnc, zu.zidx
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
                                                   / (2e0 * (CASE WHEN d + s * sd > 0 THEN 1e-5 ELSE 1e-7 END) * greatest(1e0, abs(zs4a.zx_r[zi]))))
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
                                                       -- stall certificate (b); v2 ceiling
                                                       -- 1e-2 (header deviation note)
                                                       CASE WHEN zs3f.zls_ok
                                                                 AND zs3f.zg_new IS NOT NULL
                                                                 AND len(list_filter(zs3f.zg_new,
                                                                         lambda ze: ze IS NULL)) = 0
                                                                 AND list_reduce(
                                                                       list_prepend(0e0,
                                                                           list_transform(zs3f.zg_new,
                                                                               lambda ze: abs(ze))),
                                                                       lambda za, zb: greatest(za, zb))
                                                                     <= 1e-2
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
                                                                       / (2e0 * (CASE WHEN d + s * sd > 0 THEN 1e-5 ELSE 1e-7 END) * greatest(1e0,
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
                                                                   zs2.ziter + 1 AS zniter,
                                                                   zgl.zgne AS zgne
                                                            FROM (
                                                                SELECT zs1b.*,
                                                                       -- stall certificate (a); v2 ceiling
                                                                       -- 1e-2 (header deviation note)
                                                                       (zs1b.zbest IS NULL)
                                                                         AND (zs1b.zgn_cur <= 1e-2
                                                                              OR (d + s * sd > 0
                                                                                  AND zs1b.zrestarted)) AS zstall_ls,
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
                                                                                       zpc.zdg,
                                                                                       list_transform(range(1, zpc.znp + 1),
                                                                                           lambda zi:
                                                                                               -list_reduce(
                                                                                                   list_prepend(0e0,
                                                                                                       list_transform(range(1, zpc.znp + 1),
                                                                                                           lambda zj:
                                                                                                               zit.zhinv[(zi - 1) * zpc.znp + zj]
                                                                                                               * zit.zgx[zj])),
                                                                                                   lambda za, zb: za + zb)) AS zdir
                                                                                FROM _sarimax_bf2_it zit, _sarimax_bf2_pc zpc
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
                                                                                       0e0 - (_sarimax_ll_x_v2(
                                                                                           list_transform(zxc, lambda zxe, zxi:
                                                                                               zxe + zalpha * zdirc[zxi]),
                                                                                           zwlc, zxmc, zdgc,
                                                                                           r, p, q, bigp, bigq, s, d, sd,
                                                                                           ktrend, conc)).ll AS zfa
                                                                                FROM (SELECT zs0b.zx AS zxc, zs0b.zdir AS zdirc,
                                                                                             zs0b.zfx AS zfxc, zs0b.zgd AS zgdc,
                                                                                             zs0b.zwl AS zwlc, zs0b.zxm AS zxmc,
                                                                                             zs0b.zdg AS zdgc,
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
                                                                                   0e0 - (_sarimax_ll_x_v2(
                                                                                       list_transform(zxc, lambda zxe, zxi:
                                                                                           zxe + zalpha * zdirc[zxi]),
                                                                                       zwlc, zxmc, zdgc,
                                                                                       r, p, q, bigp, bigq, s, d, sd,
                                                                                       ktrend, conc)).ll AS zfa
                                                                            FROM (SELECT zs1.zx AS zxc, zs1.zdir AS zdirc,
                                                                                         zs1.zfx AS zfxc, zs1.zgd AS zgdc,
                                                                                         zs1.zwl AS zwlc, zs1.zxm AS zxmc,
                                                                                         zs1.zdg AS zdgc,
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
                                                            CROSS JOIN LATERAL (
                                                                -- SECTION 4 shared gains at x_eval: ONE
                                                                -- covariance pass per gradient point. The
                                                                -- aggregate is a hard materialization
                                                                -- boundary (without it the optimizer
                                                                -- re-inlines the pass into every head
                                                                -- probe row -- measured); the rows-gated
                                                                -- unnest makes the site zero-cost when the
                                                                -- constant SECTION-4 gate is off. The
                                                                -- x_eval expression is repeated verbatim
                                                                -- (deterministic, so values are bitwise
                                                                -- those of zx_eval; NULL exactly when
                                                                -- zx_eval is NULL, poisoning zgne
                                                                -- harmlessly on terminal iterations).
                                                                SELECT max(CASE WHEN zzok OR (NOT zzst AND NOT zzre)
                                                                                THEN _sarimax_kf_gains_v2(
                                                                                    _sarimax_transform_params_v2(
                                                                                        CASE WHEN zzok
                                                                                             THEN list_transform(zzx,
                                                                                                      lambda zxe, zxi:
                                                                                                          zxe + zza * zzdir[zxi])
                                                                                             ELSE list_transform(zzx,
                                                                                                      lambda zxe, zxi:
                                                                                                          zxe + ((hash(zxi) % 2001)
                                                                                                                 / 1e3 - 1e0)
                                                                                                                * 1e-1) END,
                                                                                        ktrend + r, p, q, bigp, bigq, conc),
                                                                                    zzwl, zzxm, zzdg,
                                                                                    r, p, q, bigp, bigq, s, d, sd,
                                                                                    ktrend, conc)
                                                                                END) AS zgne
                                                                FROM (SELECT zs2.zx AS zzx, (zs2.zbest).za AS zza,
                                                                             zs2.zdir AS zzdir, zs2.zls_ok AS zzok,
                                                                             zs2.zstall_ls AS zzst,
                                                                             zs2.zrestarted AS zzre,
                                                                             zs2.zwl AS zzwl, zs2.zxm AS zzxm,
                                                                             zs2.zdg AS zzdg
                                                                      FROM unnest(CASE WHEN ktrend + r >= 2
                                                                                       AND _sarimax_kdiff(d, sd, s)
                                                                                           + _sarimax_k_states(p, q, bigp, bigq, s) >= 20
                                                                                       THEN [1] ELSE []::BIGINT[] END)
                                                                           AS zu9(zn9))
                                                            ) zgl
                                                        ) zs3
                                                        CROSS JOIN LATERAL (
                                                            -- batch 1: gradient (idx 1..2np) + center
                                                            -- f (idx 0, restart path only) at x_eval.
                                                            -- SECTION 4 routing: head probes (and the
                                                            -- center) ride the mean path over the shared
                                                            -- zgne covariance pass; tail probes and
                                                            -- ungated models run the full kernel exactly
                                                            -- as before (see the anchor lateral note)
                                                            SELECT list(zval ORDER BY zidx)
                                                                       FILTER (WHERE zidx > 0) AS zfpm,
                                                                   max(zval) FILTER (WHERE zidx = 0) AS zfe_c
                                                            FROM (
                                                                SELECT zidx,
                                                                       CASE WHEN ktrend + r >= 2
                                                                                 AND _sarimax_kdiff(d, sd, s)
                                                                                     + _sarimax_k_states(p, q, bigp, bigq, s) >= 20
                                                                                 AND zidx <= 2 * (ktrend + r)
                                                                            THEN 0e0 - (_sarimax_ll_mean_v2(
                                                                                zgnc,
                                                                                list_transform(range(1, len(zwlc) + 1), lambda zt9:
                                                                                    zwlc[zt9] - list_reduce(
                                                                                        list_prepend(0e0,
                                                                                            list_transform(range(1, r + 1), lambda zj9:
                                                                                                zxmc[zt9][zj9]
                                                                                                * (CASE WHEN zidx > 0
                                                                                                             AND ktrend + zj9 = (zidx + 1) // 2
                                                                                                        THEN zxc[ktrend + zj9]
                                                                                                             + (CASE WHEN zidx % 2 = 1
                                                                                                                     THEN 1e0 ELSE -1e0 END)
                                                                                                             * (CASE WHEN d + s * sd > 0
                                                                                                                     THEN 1e-5 ELSE 1e-7 END)
                                                                                                             * greatest(1e0, abs(zxc[ktrend + zj9]))
                                                                                                        ELSE zxc[ktrend + zj9] END))),
                                                                                        lambda za9, zb9: za9 + zb9)),
                                                                                _sarimax_trend_c(zdgc,
                                                                                    list_transform(range(1, ktrend + 1), lambda zg9:
                                                                                        CASE WHEN zidx > 0
                                                                                                  AND zg9 = (zidx + 1) // 2
                                                                                             THEN zxc[zg9]
                                                                                                  + (CASE WHEN zidx % 2 = 1
                                                                                                          THEN 1e0 ELSE -1e0 END)
                                                                                                  * (CASE WHEN d + s * sd > 0
                                                                                                          THEN 1e-5 ELSE 1e-7 END)
                                                                                                  * greatest(1e0, abs(zxc[zg9]))
                                                                                             ELSE zxc[zg9] END),
                                                                                    1, len(zwlc) + 1))).ll
                                                                            ELSE 0e0 - (_sarimax_ll_x_v2(
                                                                                list_transform(zxc, lambda zxe, zxi:
                                                                                    CASE WHEN zidx > 0
                                                                                              AND zxi = (zidx + 1) // 2
                                                                                         THEN zxe + (CASE WHEN zidx % 2 = 1
                                                                                                          THEN 1e0 ELSE -1e0 END)
                                                                                                    * (CASE WHEN d + s * sd > 0 THEN 1e-5 ELSE 1e-7 END)
                                                                                                    * greatest(1e0, abs(zxe))
                                                                                         ELSE zxe END),
                                                                                zwlc, zxmc, zdgc,
                                                                                r, p, q, bigp, bigq, s, d, sd,
                                                                                ktrend, conc)).ll END AS zval
                                                                FROM (SELECT zs3.zx_eval AS zxc,
                                                                             zs3.zwl AS zwlc, zs3.zxm AS zxmc,
                                                                             zs3.zdg AS zdgc,
                                                                             zs3.zgne AS zgnc,
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
                                                       0e0 - (_sarimax_ll_x_v2(
                                                           list_transform(zxc, lambda zxe, zxi:
                                                               CASE WHEN zidx > 0 AND zxi = (zidx + 1) // 2
                                                                    THEN zxe + (CASE WHEN zidx % 2 = 1
                                                                                     THEN 1e0 ELSE -1e0 END)
                                                                               * (CASE WHEN d + s * sd > 0 THEN 1e-5 ELSE 1e-7 END) * greatest(1e0, abs(zxe))
                                                                    ELSE zxe END),
                                                           zwlc, zxmc, zdgc,
                                                           r, p, q, bigp, bigq, s, d, sd,
                                                           ktrend, conc)).ll AS zval
                                                FROM (SELECT zs4.zx_r AS zxc, zs4.zwl AS zwlc,
                                                             zs4.zxm AS zxmc, zs4.zdg AS zdgc, zu.zidx
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
       _sarimax_transform_params_v2(zit.zx, ktrend + r, p, q, bigp, bigq, conc) AS params,
       0e0 - zit.zfx AS loglik,
       (_sarimax_ll_x_v2(zit.zx, zpc.zwl, zpc.zxm, zpc.zdg,
                         r, p, q, bigp, bigq, s, d, sd, ktrend, conc)).scale2 AS scale2,
       zit.zstatus = 1 AS converged,
       zit.ziter AS iterations,
       list_reduce(list_prepend(0e0, list_transform(zit.zgx, lambda ze: abs(ze))),
                   lambda za, zb: greatest(za, zb)) AS grad_norm,
       zit.zrestarted AS restarted,
       zit.zlsf AS ls_failures
FROM _sarimax_bf2_it zit, _sarimax_bf2_pc zpc;

-- ---- 3e. standard errors (v2 objective; v1 section 2d semantics) ---------------

-- One row (bse DOUBLE[]): constrained-space central-difference Hessian of the
-- v2 LOGLIKELIHOOD at theta-hat, bse = sqrt(diag(inv(-H))) -- v1
-- _sarimax_bse's exact batching, laziness and boundary step-halving, with the
-- v2 kernel and np = ktrend + r + p + q + bigp + bigq + (conc ? 0 : 1).
-- params / ylist / xmat / degs must be materialized values (section 2 header).
--
-- DOCUMENTED DEVIATION (kdiff-aware step size): h_i = 1e-4 *
-- greatest(0.1, |theta_i|) exactly as v1 when kdiff = 0, but 1e-3 *
-- greatest(0.1, |theta_i|) when kdiff > 0. The approximate-diffuse 1e6
-- initialization leaves parameter-dependent quantization noise of ~1e-9 ..
-- 1e-7 in the loglikelihood (see section 3 header); a second difference
-- divides it by h^2, so the v1 step turns that noise into 1e-2-relative
-- Hessian errors on the diffuse fixtures. A measured sweep (1e-4 .. 1e-2,
-- tests/generate notes) puts the noise-vs-truncation optimum at 1e-3:
-- kitchen_sink improves 1.3e-2 -> 4.8e-4 rel, nodiff_arima_111 1.3e-4 ->
-- 1.5e-6, nodiff_sarimax_011_011_12 1.0e-1 -> 7.7e-3 (its floor; the
-- acceptance gate for that fixture is 2e-2, everything else 1e-3).
-- statsmodels sidesteps the noise with complex-step differentiation, which
-- has no SQL equivalent.
CREATE OR REPLACE MACRO _sarimax_bse_v2(params, ylist, xmat, degs,
                                        r, p, q, bigp, bigq, s, d, sd,
                                        ktrend, conc) AS TABLE
WITH _sarimax_bs2_in0 AS (
    SELECT params AS zc, ylist AS zwl, xmat AS zxm, degs AS zdg,
           (ktrend + r + p + q + bigp + bigq
            + CASE WHEN conc THEN 0 ELSE 1 END)::BIGINT AS znp
),
_sarimax_bs2_in AS (
    SELECT zc, zwl, zxm, zdg, znp,
           (_sarimax_ll_c_v2(zc, zwl, zxm, zdg, r, p, q, bigp, bigq, s, d, sd,
                             ktrend, conc)).ll AS zf0
    FROM _sarimax_bs2_in0
),
_sarimax_bs2_h AS (       -- adaptive per-coordinate steps
    SELECT zin.zc, zin.zwl, zin.zxm, zin.zdg, zin.znp, zin.zf0, zhh.zhl
    FROM _sarimax_bs2_in zin
    CROSS JOIN LATERAL (
        SELECT list(zh ORDER BY zi) AS zhl
        FROM (
            SELECT zi,
                   (list_transform([(CASE WHEN d + s * sd > 0 THEN 1e-3 ELSE 1e-4 END)
                                    * greatest(1e-1, abs(zcc[zi]))], lambda zh0:
                        CASE WHEN (_sarimax_ll_c_v2(list_transform(zcc, lambda zv2, zi2:
                                      CASE WHEN zi2 = zi THEN zv2 + zh0 ELSE zv2 END),
                                      zwlc, zxmc, zdgc, r, p, q, bigp, bigq, s, d, sd,
                                      ktrend, conc)).ll IS NOT NULL
                                  AND (_sarimax_ll_c_v2(list_transform(zcc, lambda zv2, zi2:
                                          CASE WHEN zi2 = zi THEN zv2 - zh0 ELSE zv2 END),
                                          zwlc, zxmc, zdgc, r, p, q, bigp, bigq, s, d, sd,
                                          ktrend, conc)).ll IS NOT NULL
                             THEN zh0
                             WHEN (_sarimax_ll_c_v2(list_transform(zcc, lambda zv2, zi2:
                                      CASE WHEN zi2 = zi THEN zv2 + zh0 * 5e-1 ELSE zv2 END),
                                      zwlc, zxmc, zdgc, r, p, q, bigp, bigq, s, d, sd,
                                      ktrend, conc)).ll IS NOT NULL
                                  AND (_sarimax_ll_c_v2(list_transform(zcc, lambda zv2, zi2:
                                          CASE WHEN zi2 = zi THEN zv2 - zh0 * 5e-1 ELSE zv2 END),
                                          zwlc, zxmc, zdgc, r, p, q, bigp, bigq, s, d, sd,
                                          ktrend, conc)).ll IS NOT NULL
                             THEN zh0 * 5e-1
                             ELSE zh0 * 25e-2 END))[1] AS zh
            FROM (SELECT zin.zc AS zcc, zin.zwl AS zwlc, zin.zxm AS zxmc,
                         zin.zdg AS zdgc, zu.zi
                  FROM unnest(range(1, zin.znp + 1)) AS zu(zi))
        )
    ) zhh
),
_sarimax_bs2_tri AS (     -- upper-triangle H cells, ordered (i, j), i <= j
    SELECT zh2.zc, zh2.zwl, zh2.zxm, zh2.zdg, zh2.znp, zh2.zhl, zgr.ztri
    FROM _sarimax_bs2_h zh2
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
                       (_sarimax_ll_c_v2(list_transform(zcc, lambda zv3, zk3:
                               zv3 + CASE WHEN zk3 = zi THEN zs1 * zhlc[zi] ELSE 0e0 END
                                   + CASE WHEN zk3 = zj AND zj <> zi
                                          THEN zs2 * zhlc[zj] ELSE 0e0 END),
                           zwlc, zxmc, zdgc, r, p, q, bigp, bigq, s, d, sd,
                           ktrend, conc)).ll AS zfv
                FROM (
                    SELECT zh2.zc AS zcc, zh2.zwl AS zwlc, zh2.zxm AS zxmc,
                           zh2.zdg AS zdgc,
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
    FROM _sarimax_bs2_tri
) zng;
