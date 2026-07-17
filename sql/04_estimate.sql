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
