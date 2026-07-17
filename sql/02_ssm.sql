-- ============================================================================
-- duckARIMA Layer 2: state-space construction (spec section 5.2).
--
-- Given the SARIMA spec (p,d,q)(P,D,Q)_s and a parameter vector in canonical
-- order (beta_1..beta_r, phi_1..phi_p, theta_1..theta_q, Phi_1..Phi_P,
-- Theta_1..Theta_Q, sigma2), materializes the Harvey-representation system:
--
--   transition T (k x k):   phi*_i in column 1 (rows 1..k, zero padded),
--                           superdiagonal identity T[i, i+1] = 1
--   selection  R (k x 1):   (1, theta*_1, ..., theta*_{k-1})' zero padded
--   design     Z (1 x k):   (1, 0, ..., 0)
--   obs var    H = 0,  state var Q = sigma2
--   k = max(p + s*P, q + s*Q + 1)
--
-- phi*(L) = (1 - sum phi_i L^i)(1 - sum Phi_i L^{is})   [AR: minus signs]
-- theta*(L) = (1 + sum theta_i L^i)(1 + sum Theta_i L^{is}) [MA: plus signs]
-- and T carries +phi*_i where phi*(L) = 1 - sum phi*_i L^i.
--
-- With exogenous regressors (mle_regression=True) the regression enters as a
-- time-varying observation intercept d_t = x~_t' beta over the DIFFERENCED
-- exog -- the only time-varying quantity; T/R/Z/Q are unaffected by r.
--
-- Matrices are produced in the flattened row-major DOUBLE[] encoding of Layer
-- 0 (element (i,j) of an m x n matrix at index (i-1)*n + j); a relational
-- (name,i,j,v) view is provided for the fixture acceptance tests.
--
-- Determinism: every summation here is an explicitly ordered left-to-right
-- fold (spec section 4.2); obs_intercept feeds T1 arithmetic.
-- ============================================================================

-- ---- polynomial machinery ---------------------------------------------------

-- Dense polynomial product; a and b are coefficient lists with the constant
-- term at index 1 (a = [a0, a1, ...] meaning a0 + a1 L + ...). Inner sum runs
-- ascending in i (defined summation order).
CREATE OR REPLACE MACRO _sarimax_polymul(a, b) AS (
    list_transform(
        range(1, len(a) + len(b)),
        lambda m: list_reduce(
            list_prepend(
                0.0::DOUBLE,
                list_transform(
                    range(greatest(1, m - len(b) + 1), least(len(a), m) + 1),
                    lambda i: a[i] * b[m - i + 1])),
            lambda acc, x: acc + x))
);

-- Dense seasonal lag polynomial 1 + sign * sum_i c_i L^{i*s} as a coefficient
-- list of length P*s + 1 (constant term first).
CREATE OR REPLACE MACRO _sarimax_seasonal_poly(coefs, s, sgn) AS (
    list_prepend(
        1.0::DOUBLE,
        flatten(list_transform(coefs, lambda c:
            list_append(
                list_transform(range(1, s::BIGINT), lambda z: 0.0::DOUBLE),
                sgn * c))))
);

-- Reduced-form AR tail: phi* with phi*(L) = 1 - sum_{i>=1} phi*_i L^i, as the
-- length-(p + s*P) list of +phi*_i (the values that fill T's first column).
CREATE OR REPLACE MACRO _sarimax_expand_ar(phi, bigphi, s) AS (
    list_transform(
        list_slice(
            _sarimax_polymul(
                list_prepend(1.0::DOUBLE, list_transform(phi, lambda c: -c)),
                _sarimax_seasonal_poly(bigphi, s, -1.0::DOUBLE)),
            2, len(phi) + s::BIGINT * len(bigphi) + 1),
        lambda c: -c)
);

-- Reduced-form MA tail: theta* with theta*(L) = 1 + sum_{i>=1} theta*_i L^i,
-- as the length-(q + s*Q) list of +theta*_i.
CREATE OR REPLACE MACRO _sarimax_expand_ma(theta, bigtheta, s) AS (
    list_slice(
        _sarimax_polymul(
            list_prepend(1.0::DOUBLE, theta),
            _sarimax_seasonal_poly(bigtheta, s, 1.0::DOUBLE)),
        2, len(theta) + s::BIGINT * len(bigtheta) + 1)
);

-- ---- parameter-vector slicing ------------------------------------------------

-- Canonical parameter order: beta (r), phi (p), theta (q), Phi (P), Theta (Q),
-- sigma2. Returns the blocks as a struct of DOUBLE[] plus sigma2.
CREATE OR REPLACE MACRO _sarimax_split_params(params, r, p, q, bigp, bigq) AS (
    struct_pack(
        beta     := list_slice(params, 1, r),
        phi      := list_slice(params, r + 1, r + p),
        theta    := list_slice(params, r + p + 1, r + p + q),
        bigphi   := list_slice(params, r + p + q + 1, r + p + q + bigp),
        bigtheta := list_slice(params, r + p + q + bigp + 1, r + p + q + bigp + bigq),
        sigma2   := params[r + p + q + bigp + bigq + 1])
);

-- State dimension k = max(p + s*P, q + s*Q + 1).
CREATE OR REPLACE MACRO _sarimax_k_states(p, q, bigp, bigq, s) AS (
    greatest(p + s * bigp, q + s * bigq + 1)::BIGINT
);

-- ---- system-matrix builders (flattened row-major) -----------------------------

-- Transition T (k x k): T[i,1] = phistar[i] (i <= len), T[i,i+1] = 1.
CREATE OR REPLACE MACRO _sarimax_build_t(phistar, k) AS (
    list_transform(range(1, k * k + 1), lambda idx:
        CASE WHEN (idx - 1) % k = 0 AND (idx - 1) // k + 1 <= len(phistar)
             THEN phistar[(idx - 1) // k + 1] ELSE 0.0::DOUBLE END
        + CASE WHEN (idx - 1) % k = (idx - 1) // k + 1
               THEN 1.0::DOUBLE ELSE 0.0::DOUBLE END)
);

-- Selection R (k x 1): (1, theta*_1, ..., theta*_{k-1})' zero padded.
CREATE OR REPLACE MACRO _sarimax_build_r(thetastar, k) AS (
    list_transform(range(1, k + 1), lambda i:
        CASE WHEN i = 1 THEN 1.0::DOUBLE
             WHEN i - 1 <= len(thetastar) THEN thetastar[i - 1]
             ELSE 0.0::DOUBLE END)
);

-- Design Z (1 x k): e_1'. (The filter exploits Z = e_1 directly; this builder
-- exists for the relational view and tests.)
CREATE OR REPLACE MACRO _sarimax_build_z(k) AS (
    list_transform(range(1, k + 1), lambda i:
        CASE WHEN i = 1 THEN 1.0::DOUBLE ELSE 0.0::DOUBLE END)
);

-- R * sigma2 * R' (k x k), the state-noise contribution RQR' used by the
-- filter recursion and the Lyapunov initialization.
CREATE OR REPLACE MACRO _sarimax_build_rqr(rvec, sigma2, k) AS (
    list_transform(range(1, k * k + 1), lambda idx:
        rvec[(idx - 1) // k + 1] * rvec[(idx - 1) % k + 1] * sigma2)
);

-- ---- observation intercept ----------------------------------------------------

-- d_t = x~_t' beta over the DIFFERENCED long-form exog (t, j, x); one grouped
-- join, ordered summation over j (feeds T1 arithmetic). Rows only for t
-- present in the exog table: r = 0 means zero rows and d_t == 0 implicitly.
CREATE OR REPLACE MACRO _sarimax_obs_intercept(exog_tbl, beta) AS TABLE
SELECT
    t,
    list_reduce(
        list_prepend(0.0::DOUBLE, list(x * beta[j] ORDER BY j)),
        lambda acc, xb: acc + xb) AS d
FROM query_table(exog_tbl)
GROUP BY t;

-- ---- relational view (fixture acceptance) --------------------------------------

-- All time-invariant system matrices for one parameter vector, in the
-- relational (name, i, j, v) encoding, matching the fixture ssm.parquet naming:
-- transition (k x k), selection (k x 1), design (1 x k), state_cov (1 x 1),
-- obs_cov (1 x 1).
CREATE OR REPLACE MACRO _sarimax_ssm_rel(params, r, p, q, bigp, bigq, s) AS TABLE
WITH _sarimax_parts AS (
    SELECT
        _sarimax_split_params(params, r, p, q, bigp, bigq) AS pp,
        _sarimax_k_states(p, q, bigp, bigq, s) AS k
),
_sarimax_mats AS (
    SELECT
        k,
        _sarimax_build_t(_sarimax_expand_ar(pp.phi, pp.bigphi, s), k) AS tmat,
        _sarimax_build_r(_sarimax_expand_ma(pp.theta, pp.bigtheta, s), k) AS rvec,
        _sarimax_build_z(k) AS zvec,
        pp.sigma2 AS sigma2
    FROM _sarimax_parts
)
SELECT 'transition' AS name,
       (u.idx - 1) // k + 1 AS i, (u.idx - 1) % k + 1 AS j, tmat[u.idx] AS v
FROM _sarimax_mats, LATERAL unnest(range(1, k * k + 1)) AS u(idx)
UNION ALL
SELECT 'selection', u.idx, 1, rvec[u.idx]
FROM _sarimax_mats, LATERAL unnest(range(1, k + 1)) AS u(idx)
UNION ALL
SELECT 'design', 1, u.idx, zvec[u.idx]
FROM _sarimax_mats, LATERAL unnest(range(1, k + 1)) AS u(idx)
UNION ALL
SELECT 'state_cov', 1, 1, sigma2 FROM _sarimax_mats
UNION ALL
SELECT 'obs_cov', 1, 1, 0.0::DOUBLE FROM _sarimax_mats;
