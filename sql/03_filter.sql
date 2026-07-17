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
-- (spec 4.2 determinism).
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
-- Requires: sql/00_linalg.sql, sql/02_ssm.sql.
-- ============================================================================

-- ---- per-step update, chained so intermediates are computed once ------------

-- Final assembly: v, f, TP = T*P, tpz = TP e_1 (un-normalized gain), ta = T*a
-- all precomputed by the stages below. K = tpz/f.
CREATE OR REPLACE MACRO _sarimax_kf_step3(tmat, rqr, k, v, f, tp, tpz, ta) AS (
    struct_pack(
        v := v,
        f := f,
        anew := list_transform(range(1, k + 1),
                               lambda i: ta[i] + tpz[i] * v / f),
        pnew := _sarimax_msym(
            _sarimax_madd(
                _sarimax_msub(
                    _sarimax_mmul(tp, _sarimax_mtrans(tmat, k, k), k, k, k),
                    list_transform(range(1, k * k + 1), lambda idx:
                        tpz[(idx - 1) // k + 1] * tpz[(idx - 1) % k + 1] / f)),
                rqr),
            k),
        term := CASE WHEN f > 0.0::DOUBLE
                     THEN -0.5::DOUBLE * (ln(2.0::DOUBLE * pi()) + ln(f) + v * v / f)
                     ELSE NULL END)
);

-- Stage 2: with v, f, TP in hand, derive TP e_1 (the un-normalized gain) and
-- T a, then assemble.
CREATE OR REPLACE MACRO _sarimax_kf_step2(a, tmat, rqr, k, v, f, tp) AS (
    _sarimax_kf_step3(
        tmat, rqr, k, v, f, tp,
        list_transform(range(1, k + 1), lambda i: tp[(i - 1) * k + 1]),
        _sarimax_mmul(tmat, a, k, k, 1))
);

-- Stage 1: innovation, innovation variance, and TP = T P (computed once).
CREATE OR REPLACE MACRO _sarimax_kf_step(a, p, tmat, rqr, yd, k) AS (
    _sarimax_kf_step2(
        a, tmat, rqr, k,
        yd - a[1],
        p[1],
        _sarimax_mmul(tmat, p, k, k, k))
);

-- ---- system construction per probe (shares work across identical ARMA blocks)

-- probes_tbl: (probe_id BIGINT, params DOUBLE[]) in canonical parameter order.
-- Returns (probe_id, k, tmat, rqr, p1): the time-invariant system per probe.
-- Probes differing only in beta share one construction + Lyapunov solve.
CREATE OR REPLACE MACRO _sarimax_systems(probes_tbl, r, p, q, bigp, bigq, s) AS TABLE
WITH _sarimax_sys_arma AS (
    SELECT DISTINCT
        list_slice(params, r + 1, r + p + q + bigp + bigq + 1) AS armav
    FROM query_table(probes_tbl)
),
_sarimax_sys_poly AS (
    SELECT
        armav,
        _sarimax_k_states(p, q, bigp, bigq, s) AS k,
        _sarimax_expand_ar(list_slice(armav, 1, p),
                           list_slice(armav, p + q + 1, p + q + bigp), s) AS phistar,
        _sarimax_expand_ma(list_slice(armav, p + 1, p + q),
                           list_slice(armav, p + q + bigp + 1, p + q + bigp + bigq), s) AS thetastar,
        armav[p + q + bigp + bigq + 1] AS sigma2
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
           _sarimax_build_rqr(rvec, sigma2, k) AS rqr
    FROM _sarimax_sys_tr
),
_sarimax_sys_full AS (
    SELECT armav, k, tmat, rqr,
           _sarimax_lyap(tmat, rqr, k) AS p1
    FROM _sarimax_sys_rqr
)
SELECT pr.probe_id, b.k, b.tmat, b.rqr, b.p1
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
-- sys_tbl: (probe_id BIGINT, k, tmat, rqr, p1).
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
    SELECT q.probe_id, q.t,
           (q.upd).anew AS a, (q.upd).pnew AS p,
           (q.upd).v AS v, (q.upd).f AS f,
           q.ll_acc + (q.upd).term AS ll_acc
    FROM (
        SELECT kf.probe_id, kf.t + 1 AS t, kf.ll_acc,
               _sarimax_kf_step(kf.a, kf.p, s.tmat, s.rqr, o.yd, s.k) AS upd
        FROM _sarimax_kf kf
        JOIN query_table(sys_tbl) s ON s.probe_id = kf.probe_id
        JOIN query_table(obs_tbl) o ON o.probe_id = kf.probe_id AND o.t = kf.t + 1
    ) q
)
SELECT probe_id, t, v, f, ll_acc
FROM _sarimax_kf
WHERE t >= 1;

-- Final state per probe (a_{n+1}, P_{n+1}) -- needed by forecasting.
CREATE OR REPLACE MACRO _sarimax_kfilter_state(obs_tbl, sys_tbl) AS TABLE
WITH RECURSIVE _sarimax_kfs USING KEY (probe_id) AS (
    SELECT s.probe_id,
           0::BIGINT AS t,
           list_transform(range(1, s.k + 1), lambda i: 0.0::DOUBLE) AS a,
           s.p1 AS p,
           0.0::DOUBLE AS ll_acc
    FROM query_table(sys_tbl) s
    UNION ALL
    SELECT q.probe_id, q.t, (q.upd).anew AS a, (q.upd).pnew AS p,
           q.ll_acc + (q.upd).term AS ll_acc
    FROM (
        SELECT kfs.probe_id, kfs.t + 1 AS t, kfs.ll_acc,
               _sarimax_kf_step(kfs.a, kfs.p, s.tmat, s.rqr, o.yd, s.k) AS upd
        FROM _sarimax_kfs kfs
        JOIN query_table(sys_tbl) s ON s.probe_id = kfs.probe_id
        JOIN query_table(obs_tbl) o ON o.probe_id = kfs.probe_id AND o.t = kfs.t + 1
    ) q
)
SELECT probe_id, t AS n_eff, a, p, ll_acc AS loglik
FROM _sarimax_kfs;

-- Loglikelihood per probe: ll_acc at the final step.
CREATE OR REPLACE MACRO _sarimax_loglik(obs_tbl, sys_tbl) AS TABLE
SELECT probe_id, n_eff, loglik
FROM _sarimax_kfilter_state(obs_tbl, sys_tbl);
