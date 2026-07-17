-- ============================================================================
-- duckARIMA Layer 3: Kalman filter and loglikelihood (spec section 5.3).
--
-- Standard covariance Kalman filter over the Harvey representation with H = 0,
-- Z = e_1', stationary initialization (a_1 = 0, P_1 from the discrete Lyapunov
-- equation via Layer 0's vec-trick solver). Predicted-state convention: a_t,
-- P_t entering step t are one-step-ahead predictions.
--
--   v_t = yd_t - a_t[1]                      (yd = w - d, intercept pre-folded)
--   F_t = P_t[1,1]
--   TP  = T P_t                              (computed once, reused)
--   K_t = TP e_1 / F_t
--   a_{t+1} = T a_t + K_t v_t
--   P_{t+1} = TP T' - (TP e_1)(TP e_1)'/F_t + RQR', symmetrized (P+P')/2
--
-- Loglikelihood accumulates over ALL t (loglikelihood_burn = 0):
--   ll = -1/2 sum_t (ln 2 pi + ln F_t + v_t^2 / F_t)
-- carried incrementally through the recursion, i.e. summed in strict t order
-- (spec 4.2 determinism). The reference is statsmodels' EXACT filter
-- (fixtures pin ssm.tolerance = 0: no convergence freezing of K and F).
--
-- Numeric guard (spec 4.3 / 6): F_t <= 0 poisons that probe's loglik term to
-- NULL, which propagates through the accumulator so the probe's final ll is
-- NULL -- a batched evaluation never aborts on one bad probe; callers that
-- require success fail loudly on NULL.
--
-- Batched multi-probe evaluation: everything is keyed by probe_id. Probes that
-- share the (phi, theta, Phi, Theta, sigma2) block -- e.g. central-difference
-- perturbations of beta -- share one system construction and Lyapunov solve
-- (the DISTINCT arma-block join below).
--
-- IMPLEMENTATION NOTE (the k^5 trap): DuckDB macro expansion is textual, so a
-- macro argument referenced inside a lambda is re-evaluated per element.
-- Nesting the covariance update as msym(madd(msub(mmul(...), ...), ...))
-- would re-expand the k^3 matrix product per output element. Every stage is
-- therefore bound as its OWN COLUMN of a nested derived table, so every macro
-- call below receives only plain column references.
--
-- Requires: sql/00_linalg.sql, sql/02_ssm.sql.
-- ============================================================================

-- ---- system construction per probe (shares work across identical ARMA blocks)

-- probes_tbl: (probe_id BIGINT, params DOUBLE[]) in canonical parameter order.
-- Returns (probe_id, k, tmat, tmat_t, rqr, p1): time-invariant system per
-- probe. Probes differing only in beta share one construction + Lyapunov solve.
CREATE OR REPLACE MACRO _sarimax_systems(probes_tbl, r, p, q, bigp, bigq, s) AS TABLE
WITH _sarimax_sys_args AS (
    -- bind scalar args to columns once: an argument that is a scalar subquery
    -- (e.g. read from a model table) may not land inside a lambda body, so
    -- everything downstream references these columns instead
    SELECT r::INT AS zr, p::INT AS zp, q::INT AS zq,
           bigp::INT AS zbp, bigq::INT AS zbq, s::BIGINT AS zs
),
_sarimax_sys_arma AS (
    SELECT DISTINCT
        za.zp, za.zq, za.zbp, za.zbq, za.zs,
        list_slice(params, za.zr + 1, za.zr + za.zp + za.zq + za.zbp + za.zbq + 1) AS armav
    FROM query_table(probes_tbl)
    CROSS JOIN _sarimax_sys_args za
),
_sarimax_sys_poly AS (
    SELECT
        armav,
        _sarimax_k_states(zp, zq, zbp, zbq, zs) AS k,
        _sarimax_expand_ar(list_slice(armav, 1, zp),
                           list_slice(armav, zp + zq + 1, zp + zq + zbp), zs) AS phistar,
        _sarimax_expand_ma(list_slice(armav, zp + 1, zp + zq),
                           list_slice(armav, zp + zq + zbp + 1, zp + zq + zbp + zbq), zs) AS thetastar,
        armav[zp + zq + zbp + zbq + 1] AS sigma2
    FROM _sarimax_sys_arma
),
_sarimax_sys_tr AS (
    SELECT armav, k, sigma2,
           _sarimax_build_t(phistar, k) AS tmat,
           _sarimax_build_r(thetastar, k) AS rvec
    FROM _sarimax_sys_poly
),
_sarimax_sys_rqr AS (
    SELECT armav, k, tmat,
           _sarimax_mtrans(tmat, k, k) AS tmat_t,
           _sarimax_build_rqr(rvec, sigma2, k) AS rqr
    FROM _sarimax_sys_tr
),
_sarimax_sys_full AS (
    SELECT armav, k, tmat, tmat_t, rqr,
           _sarimax_lyap(tmat, rqr, k) AS p1
    FROM _sarimax_sys_rqr
)
SELECT pr.probe_id, b.k, b.tmat, b.tmat_t, b.rqr, b.p1
FROM query_table(probes_tbl) pr
JOIN _sarimax_sys_full b
  ON b.armav = list_slice(pr.params, r + 1, r + p + q + bigp + bigq + 1);

-- ---- observation prep: fold the per-probe intercept into the series ---------

-- w_tbl: (t BIGINT, w DOUBLE) differenced series, t = 1..n_eff.
-- exog_diff_tbl: (t BIGINT, j INT, x DOUBLE) differenced exog (zero rows if r=0).
-- Returns (probe_id, t, yd) with yd = w_t - x~_t' beta, ordered-j summation.
CREATE OR REPLACE MACRO _sarimax_obs_adj(w_tbl, exog_diff_tbl, probes_tbl) AS TABLE
WITH _sarimax_oa_int AS (
    SELECT pr.probe_id, e.t,
           list_reduce(
               list_prepend(0.0::DOUBLE, list(e.x * pr.params[e.j] ORDER BY e.j)),
               lambda acc, xb: acc + xb) AS d
    FROM query_table(probes_tbl) pr
    CROSS JOIN query_table(exog_diff_tbl) e
    GROUP BY pr.probe_id, e.t
)
SELECT pr.probe_id, w.t, w.w - coalesce(di.d, 0.0::DOUBLE) AS yd
FROM query_table(probes_tbl) pr
CROSS JOIN query_table(w_tbl) w
LEFT JOIN _sarimax_oa_int di
       ON di.probe_id = pr.probe_id AND di.t = w.t;

-- ---- the filter ---------------------------------------------------------------

-- obs_tbl: (probe_id BIGINT, t BIGINT, yd DOUBLE), t dense 1..n_eff per probe.
-- sys_tbl: (probe_id BIGINT, k, tmat, tmat_t, rqr, p1).
-- Returns the full trace (probe_id, t, v, f, ll_acc) for t = 1..n_eff, where
-- ll_acc is the running loglikelihood through step t (strict t-order fold).
CREATE OR REPLACE MACRO _sarimax_kfilter(obs_tbl, sys_tbl) AS TABLE
WITH RECURSIVE _sarimax_kf USING KEY (probe_id, t) AS (
    SELECT s.probe_id,
           0::BIGINT AS t,
           list_transform(range(1, s.k + 1), lambda i: 0.0::DOUBLE) AS a,
           s.p1 AS p,
           NULL::DOUBLE AS v,
           NULL::DOUBLE AS f,
           0.0::DOUBLE AS ll_acc
    FROM query_table(sys_tbl) s
    UNION ALL
    SELECT probe_id, t,
           list_transform(range(1, k + 1), lambda i: ta[i] + tpz[i] * v / f) AS a,
           _sarimax_msym(prqr, k) AS p,
           v, f,
           ll_acc + CASE WHEN f > 0.0::DOUBLE
                         THEN -0.5::DOUBLE * (ln(2.0::DOUBLE * pi()) + ln(f) + v * v / f)
                         ELSE NULL END AS ll_acc
    FROM (
        SELECT probe_id, t, ll_acc, k, v, f, ta, tpz,
               _sarimax_madd(psub, rqr) AS prqr
        FROM (
            SELECT probe_id, t, ll_acc, k, rqr, v, f, ta, tpz,
                   _sarimax_msub(tpt, outerm) AS psub
            FROM (
                SELECT probe_id, t, ll_acc, k, rqr, v, f, ta, tpz,
                       _sarimax_mmul(tp, tmat_t, k, k, k) AS tpt,
                       list_transform(range(1, k * k + 1), lambda idx:
                           tpz[(idx - 1) // k + 1] * tpz[(idx - 1) % k + 1] / f) AS outerm
                FROM (
                    SELECT probe_id, t, ll_acc, k, tmat_t, rqr, v, f, tp,
                           list_transform(range(1, k + 1), lambda i: tp[(i - 1) * k + 1]) AS tpz,
                           ta
                    FROM (
                        SELECT kf.probe_id, kf.t + 1 AS t, kf.ll_acc,
                               s.k AS k, s.tmat_t, s.rqr,
                               o.yd - kf.a[1] AS v,
                               kf.p[1] AS f,
                               _sarimax_mmul(s.tmat, kf.p, s.k, s.k, s.k) AS tp,
                               _sarimax_mmul(s.tmat, kf.a, s.k, s.k, 1) AS ta
                        FROM _sarimax_kf kf
                        JOIN query_table(sys_tbl) s ON s.probe_id = kf.probe_id
                        JOIN query_table(obs_tbl) o ON o.probe_id = kf.probe_id AND o.t = kf.t + 1
                    )
                )
            )
        )
    )
)
SELECT probe_id, t, v, f, ll_acc
FROM _sarimax_kf
WHERE t >= 1;

-- Final state per probe (a_{n+1}, P_{n+1}) plus the loglikelihood -- the
-- compact variant used by estimation and forecasting (USING KEY on probe_id:
-- each iteration replaces the row, so only the final state is retained).
CREATE OR REPLACE MACRO _sarimax_kfilter_state(obs_tbl, sys_tbl) AS TABLE
WITH RECURSIVE _sarimax_kfs USING KEY (probe_id) AS (
    SELECT s.probe_id,
           0::BIGINT AS t,
           list_transform(range(1, s.k + 1), lambda i: 0.0::DOUBLE) AS a,
           s.p1 AS p,
           0.0::DOUBLE AS ll_acc
    FROM query_table(sys_tbl) s
    UNION ALL
    SELECT probe_id, t,
           list_transform(range(1, k + 1), lambda i: ta[i] + tpz[i] * v / f) AS a,
           _sarimax_msym(prqr, k) AS p,
           ll_acc + CASE WHEN f > 0.0::DOUBLE
                         THEN -0.5::DOUBLE * (ln(2.0::DOUBLE * pi()) + ln(f) + v * v / f)
                         ELSE NULL END AS ll_acc
    FROM (
        SELECT probe_id, t, ll_acc, k, v, f, ta, tpz,
               _sarimax_madd(psub, rqr) AS prqr
        FROM (
            SELECT probe_id, t, ll_acc, k, rqr, v, f, ta, tpz,
                   _sarimax_msub(tpt, outerm) AS psub
            FROM (
                SELECT probe_id, t, ll_acc, k, rqr, v, f, ta, tpz,
                       _sarimax_mmul(tp, tmat_t, k, k, k) AS tpt,
                       list_transform(range(1, k * k + 1), lambda idx:
                           tpz[(idx - 1) // k + 1] * tpz[(idx - 1) % k + 1] / f) AS outerm
                FROM (
                    SELECT probe_id, t, ll_acc, k, tmat_t, rqr, v, f, tp,
                           list_transform(range(1, k + 1), lambda i: tp[(i - 1) * k + 1]) AS tpz,
                           ta
                    FROM (
                        SELECT kfs.probe_id, kfs.t + 1 AS t, kfs.ll_acc,
                               s.k AS k, s.tmat_t, s.rqr,
                               o.yd - kfs.a[1] AS v,
                               kfs.p[1] AS f,
                               _sarimax_mmul(s.tmat, kfs.p, s.k, s.k, s.k) AS tp,
                               _sarimax_mmul(s.tmat, kfs.a, s.k, s.k, 1) AS ta
                        FROM _sarimax_kfs kfs
                        JOIN query_table(sys_tbl) s ON s.probe_id = kfs.probe_id
                        JOIN query_table(obs_tbl) o ON o.probe_id = kfs.probe_id AND o.t = kfs.t + 1
                    )
                )
            )
        )
    )
)
SELECT probe_id, t AS n_eff, a, p, ll_acc AS loglik
FROM _sarimax_kfs;

-- Loglikelihood per probe: ll_acc at the final step.
CREATE OR REPLACE MACRO _sarimax_loglik(obs_tbl, sys_tbl) AS TABLE
SELECT probe_id, n_eff, loglik
FROM _sarimax_kfilter_state(obs_tbl, sys_tbl);
