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


-- ============================================================================
-- SECTION 2 (v2, appended): filtering for the AUGMENTED system
-- (simple_differencing = FALSE) with missing values, trend terms and
-- concentrated scale. The v1 macros above are frozen.
--
-- THE SHIFTED-BASIS TRICK (why this filter can keep Z = e_1): the augmented
-- design Z is NOT e_1 (it has ones on every ordinary-diff state, on the last
-- state of each seasonal cycle block, and on the first ARMA state). But when
-- kdiff >= 1, row 1 of the augmented T equals Z exactly, R[1] = 0, and the
-- state intercept never enters row 1 -- so alpha_{t+1}[1] = Z alpha_t
-- deterministically. Filtering the SHIFTED state beta_t := alpha_{t+1} makes
-- the observation y_t = beta_t[1] (+ obs intercept), i.e. Z~ = e_1', with the
-- SAME T, R and RQR. The price is a shifted anchor and intercept timing:
--    anchor:  a~_1 = T a_1 + c_1 e_cidx,  P~_1 = msym(T P_1 T' + RQR)
--    step t adds c_{t+1} (not c_t) to row cidx; c_{n+1} is outside the obs
--    table and is applied as 0 -- the FINAL state returned by
--    _sarimax_kfilter_state_v2 therefore excludes the c_{n+1} contribution,
--    which a forecasting layer must add back (it needs c_{n+2}... anyway).
-- When kdiff = 0 the augmented system IS the v1 system (Z = e_1) and no shift
-- happens: anchor (a_1, P_1), step t adds c_t -- and because a_1, P_1 are the
-- stationary fixed points of the shift map, the two cases meet continuously.
-- statsmodels' predicted-state trace under the shift satisfies
-- v_t = yd_t - a~_t[1] and F_t = P~_t[1,1] EXACTLY (fixture-validated).
--
-- statsmodels-matching arithmetic note: with the approximate-diffuse 1e6
-- initialization the covariance recursion is ill-conditioned enough that only
-- same-order-of-operations agreement is achievable; agreement floors measured
-- against the fixtures are ~1e-14 (stationary), ~3e-10 (kdiff = 1), ~1e-9
-- (kdiff = 5), ~2e-6 (kdiff = 13) relative on F -- the noise is eps(1e6 * |P|)
-- amplified through the diffuse collapse, identical in kind to what any
-- BLAS-vs-fold reordering produces. See tests/test_filter_v2.py tolerances.
--
-- Loglikelihood bookkeeping (v2): the recursion carries three accumulators
-- instead of v1's ll_acc --
--   cnt      : number of counted steps (v not NULL and t > burn)
--   sumlogf  : sum of ln F_t over counted steps (NULL-poisoned when F <= 0)
--   ssq      : sum of v_t^2 / F_t over counted steps
-- so one filter pass serves both the plain and the CONCENTRATED loglik:
--   plain: ll = -1/2 (cnt ln 2pi + sumlogf + ssq)          (sigma2 in RQR)
--   conc:  ll = -1/2 (cnt ln 2pi + sumlogf + cnt ln(ssq/cnt) + cnt),
--          scale2 = ssq/cnt (the filter ran at sigma2 = 1)
--
-- Requires: sql/00_linalg.sql, sql/02_ssm.sql.
-- ============================================================================

-- ---- v2 initialization builders ------------------------------------------------
-- (These belong logically to Layer 2 but live here because they depend on
-- Layer 0 -- _sarimax_lyap / _sarimax_solve_list -- and sql/02_ssm.sql must
-- keep loading standalone for the v1 Layer-2 acceptance tests.)

-- Initial covariance P1 (k*k flattened): blockdiag(1e6 * I_kdiff,
-- lyapunov(T_arma, RQR_arma)) with EXACTLY zero cross blocks. tmat_arma and
-- rqr_arma are the karma-sized v1 blocks (flattened row-major).
CREATE OR REPLACE MACRO _sarimax_p1_v2(tmat_arma, rqr_arma, karma, d, sd, s) AS (
    (list_transform([_sarimax_lyap(tmat_arma, rqr_arma, karma)], lambda zlp:
        list_transform(
            range(1, (d + s * sd + karma) * (d + s * sd + karma) + 1),
            lambda zidx:
            CASE
              WHEN (zidx - 1) // (d + s * sd + karma) + 1 <= d + s * sd
                   AND (zidx - 1) % (d + s * sd + karma) + 1 <= d + s * sd
              THEN CASE WHEN (zidx - 1) // (d + s * sd + karma)
                             = (zidx - 1) % (d + s * sd + karma)
                        THEN 1e6 ELSE 0e0 END
              WHEN (zidx - 1) // (d + s * sd + karma) + 1 > d + s * sd
                   AND (zidx - 1) % (d + s * sd + karma) + 1 > d + s * sd
              THEN zlp[((zidx - 1) // (d + s * sd + karma) - (d + s * sd)) * karma
                       + ((zidx - 1) % (d + s * sd + karma) + 1 - (d + s * sd))]
              ELSE 0e0 END)))[1]
);

-- Initial state a1 (k-list): zeros(kdiff) ++ (I - T_arma)^-1 e_1 c_1 on the
-- ARMA block (all zeros when c1 = 0, i.e. no trend).
CREATE OR REPLACE MACRO _sarimax_a1_v2(tmat_arma, karma, d, sd, s, c1) AS (
    CASE WHEN c1 = 0e0 THEN _sarimax_mzeros(d + s * sd + karma, 1)
    ELSE _sarimax_mzeros(d + s * sd, 1) ||
         (_sarimax_solve_list(
             list_transform(range(1, karma * (karma + 1) + 1), lambda zidx:
                 CASE WHEN (zidx - 1) % (karma + 1) + 1 <= karma
                      THEN (CASE WHEN (zidx - 1) % (karma + 1) + 1
                                      = (zidx - 1) // (karma + 1) + 1
                                 THEN 1e0 ELSE 0e0 END)
                           - tmat_arma[((zidx - 1) // (karma + 1)) * karma
                                       + ((zidx - 1) % (karma + 1) + 1)]
                      ELSE CASE WHEN (zidx - 1) // (karma + 1) = 0
                                THEN c1 ELSE 0e0 END
                 END),
             karma, 1)).x
    END
);

-- ---- v2 system construction per probe ----------------------------------------

-- probes_tbl: (probe_id BIGINT, params DOUBLE[]) in v2 canonical order
-- [tau (ktrend), beta (r), phi, theta, Phi, Theta, (sigma2 unless conc)].
-- Returns one row per probe:
--   (probe_id, k, karma, kdiff, cidx, burn, tmat, tmat_t, rqr, p1, a1,
--    a1f, p1f)
-- where (p1, a1) are the RAW statsmodels initializations (fixture 'P1'/'a1')
-- and (p1f, a1f) are the shifted-basis filter anchors described above --
-- _sarimax_kfilter_v2 consumes a1f/p1f. Probes sharing the arma+sigma2 slice
-- share one heavy construction (poly expansion + Lyapunov + anchor cov);
-- only c1 = sum(tau) and hence a1/a1f are per-probe. conc = 1 -> the params
-- carry no sigma2 slot and the filter runs at sigma2 = 1e0.
CREATE OR REPLACE MACRO _sarimax_systems_v2(probes_tbl, r, p, q, bigp, bigq, s,
                                            d, sd, ktrend, conc) AS TABLE
WITH _sarimax_sv2_args AS (
    -- bind every scalar arg to a column once (scalar-subquery/lambda trap)
    SELECT r::BIGINT AS zr, p::BIGINT AS zp, q::BIGINT AS zq,
           bigp::BIGINT AS zbp, bigq::BIGINT AS zbq, s::BIGINT AS zs,
           d::BIGINT AS zd, sd::BIGINT AS zsd, ktrend::BIGINT AS zkt,
           conc::BIGINT AS zconc
),
_sarimax_sv2_arma AS (
    SELECT DISTINCT
        za.zr, za.zp, za.zq, za.zbp, za.zbq, za.zs, za.zd, za.zsd, za.zkt,
        list_slice(pr.params, za.zkt + za.zr + 1,
                   za.zkt + za.zr + za.zp + za.zq + za.zbp + za.zbq
                   + CASE WHEN za.zconc = 1 THEN 0 ELSE 1 END) AS armav,
        za.zconc
    FROM query_table(probes_tbl) pr
    CROSS JOIN _sarimax_sv2_args za
),
_sarimax_sv2_poly AS (
    SELECT zr, zkt, zd, zsd, zs, armav,
           _sarimax_k_states(zp, zq, zbp, zbq, zs) AS karma,
           _sarimax_kdiff(zd, zsd, zs) AS kdiff,
           _sarimax_expand_ar(
               list_slice(armav, 1, zp),
               list_slice(armav, zp + zq + 1, zp + zq + zbp), zs) AS phistar,
           _sarimax_expand_ma(
               list_slice(armav, zp + 1, zp + zq),
               list_slice(armav, zp + zq + zbp + 1, zp + zq + zbp + zbq),
               zs) AS thetastar,
           CASE WHEN zconc = 1 THEN 1e0
                ELSE armav[zp + zq + zbp + zbq + 1] END AS sigma2
    FROM _sarimax_sv2_arma
),
_sarimax_sv2_tr AS (
    SELECT zr, zkt, zd, zsd, zs, armav, karma, kdiff, kdiff + karma AS k,
           sigma2,
           _sarimax_build_t(phistar, karma) AS tmat_arma,
           _sarimax_build_r(thetastar, karma) AS rvec_arma,
           _sarimax_build_t_v2(phistar, karma, zd, zsd, zs) AS tmat,
           _sarimax_build_r_v2(thetastar, karma, zd, zsd, zs) AS rvec
    FROM _sarimax_sv2_poly
),
_sarimax_sv2_rqr AS (
    SELECT zr, zkt, zd, zsd, zs, armav, karma, kdiff, k, tmat, tmat_arma,
           _sarimax_mtrans(tmat, k, k) AS tmat_t,
           _sarimax_build_rqr(rvec, sigma2, k) AS rqr,
           _sarimax_build_rqr(rvec_arma, sigma2, karma) AS rqr_arma
    FROM _sarimax_sv2_tr
),
_sarimax_sv2_p1 AS (
    SELECT *, _sarimax_p1_v2(tmat_arma, rqr_arma, karma, zd, zsd, zs) AS p1
    FROM _sarimax_sv2_rqr
),
-- shifted-anchor covariance, staged one matrix op per derived table (k^5 trap)
_sarimax_sv2_pf1 AS (
    SELECT *, _sarimax_mmul(tmat, p1, k, k, k) AS tp1 FROM _sarimax_sv2_p1
),
_sarimax_sv2_pf2 AS (
    SELECT *, _sarimax_mmul(tp1, tmat_t, k, k, k) AS tp1t FROM _sarimax_sv2_pf1
),
_sarimax_sv2_pf3 AS (
    SELECT *, _sarimax_madd(tp1t, rqr) AS tp1tr FROM _sarimax_sv2_pf2
),
_sarimax_sv2_sys AS (
    SELECT zr, zkt, armav, karma, kdiff, k, tmat, tmat_arma, tmat_t, rqr, p1,
           zd, zsd, zs,
           CASE WHEN kdiff = 0 THEN p1
                ELSE _sarimax_msym(tp1tr, k) END AS p1f
    FROM _sarimax_sv2_pf3
),
_sarimax_sv2_probe AS (
    SELECT pr.probe_id, b.*,
           CASE WHEN b.zkt = 0 THEN 0e0
                ELSE list_reduce(
                    list_prepend(0e0, list_slice(pr.params, 1, b.zkt)),
                    lambda zacc, zx: zacc + zx) END AS c1
    FROM query_table(probes_tbl) pr
    JOIN _sarimax_sv2_sys b
      ON b.armav = list_slice(pr.params, b.zkt + b.zr + 1,
                              b.zkt + b.zr + len(b.armav))
),
_sarimax_sv2_a1 AS (
    SELECT *, _sarimax_a1_v2(tmat_arma, karma, zd, zsd, zs, c1) AS a1
    FROM _sarimax_sv2_probe
),
_sarimax_sv2_ta1 AS (
    SELECT *, _sarimax_mmul(tmat, a1, k, k, 1) AS ta1 FROM _sarimax_sv2_a1
)
SELECT probe_id, k, karma, kdiff, (kdiff + 1)::BIGINT AS cidx,
       kdiff::BIGINT AS burn, tmat, tmat_t, rqr, p1, a1,
       CASE WHEN kdiff = 0 THEN a1
            ELSE list_transform(range(1, k + 1), lambda zi:
                     ta1[zi] + CASE WHEN zi = kdiff + 1 THEN c1
                                    ELSE 0e0 END) END AS a1f,
       p1f
FROM _sarimax_sv2_ta1;

-- ---- v2 observation prep -------------------------------------------------------

-- y_tbl: (t BIGINT, y DOUBLE) UNdifferenced series, t = 1..n; y may be NULL
-- (missing). exog_tbl: (t, j, x) long-form UNdifferenced IN-SAMPLE exog (zero
-- rows when r = 0). degs_tbl: (idx BIGINT, degree BIGINT) trend polynomial
-- degrees (fixture trend.parquet; empty when no trend).
-- Returns (probe_id, t, yd, ct): yd = y_t - x_t' beta (NULL when y_t NULL),
-- ct = c_t = sum_g tau_g * t**deg_g (0e0 when ktrend = 0). Ordered folds.
CREATE OR REPLACE MACRO _sarimax_obs_adj_v2(y_tbl, exog_tbl, probes_tbl,
                                            r, ktrend, degs_tbl) AS TABLE
WITH _sarimax_oa2_args AS (
    SELECT r::BIGINT AS zr, ktrend::BIGINT AS zkt
),
_sarimax_oa2_degs AS (
    SELECT coalesce(list(degree::BIGINT ORDER BY idx), []::BIGINT[]) AS degs
    FROM query_table(degs_tbl)
),
_sarimax_oa2_pr AS (
    SELECT pr.probe_id,
           list_slice(pr.params, za.zkt + 1, za.zkt + za.zr) AS beta,
           list_slice(pr.params, 1, za.zkt) AS tau,
           dg.degs
    FROM query_table(probes_tbl) pr, _sarimax_oa2_args za, _sarimax_oa2_degs dg
),
_sarimax_oa2_int AS (
    SELECT pr.probe_id, e.t,
           list_reduce(
               list_prepend(0e0, list(e.x * pr.beta[e.j] ORDER BY e.j)),
               lambda zacc, zxb: zacc + zxb) AS d
    FROM _sarimax_oa2_pr pr
    CROSS JOIN query_table(exog_tbl) e
    GROUP BY pr.probe_id, e.t
)
SELECT pr.probe_id, w.t,
       w.y - coalesce(di.d, 0e0) AS yd,
       CASE WHEN len(pr.tau) = 0 THEN 0e0
            ELSE (_sarimax_trend_c(pr.degs, pr.tau, w.t, 1))[1] END AS ct
FROM _sarimax_oa2_pr pr
CROSS JOIN query_table(y_tbl) w
LEFT JOIN _sarimax_oa2_int di
       ON di.probe_id = pr.probe_id AND di.t = w.t;

-- ---- the v2 filter --------------------------------------------------------------

-- obs_tbl: (probe_id, t, yd, ct), t dense 1..n per probe (yd NULL = missing).
-- sys_tbl: output of _sarimax_systems_v2.
-- Full trace (probe_id, t, v, f, cnt, sumlogf, ssq); accumulators are the
-- running values through step t (strict t-order folds). Per-step semantics:
--   v = yd_t - a[1] (NULL when yd NULL); f = p[1] (always reported)
--   a' = T a + (v NULL ? 0 : (TP e_1) v / f) + c e_cidx
--   P' = msym((v NULL ? T P T' : T P T' - (TP e_1)(TP e_1)'/f) + RQR)
--   counted (v not NULL and t > burn): cnt += 1, sumlogf += ln f (NULL when
--   f <= 0, poisoning the accumulator), ssq += v^2/f.
-- The intercept c applied at step t is ct at row t + 1 when kdiff > 0 (the
-- shifted-basis timing; 0 beyond the sample) and ct at row t when kdiff = 0.
CREATE OR REPLACE MACRO _sarimax_kfilter_v2(obs_tbl, sys_tbl) AS TABLE
WITH RECURSIVE _sarimax_kf2 USING KEY (probe_id, t) AS (
    SELECT s.probe_id,
           0::BIGINT AS t,
           s.a1f AS a,
           s.p1f AS p,
           NULL::DOUBLE AS v,
           NULL::DOUBLE AS f,
           0::BIGINT AS cnt,
           0e0 AS sumlogf,
           0e0 AS ssq
    FROM query_table(sys_tbl) s
    UNION ALL
    SELECT probe_id, t,
           list_transform(range(1, k + 1), lambda zi:
               ta[zi]
               + (CASE WHEN v IS NULL THEN 0e0 ELSE tpz[zi] * v / f END)
               + (CASE WHEN zi = cidx THEN ct ELSE 0e0 END)) AS a,
           _sarimax_msym(prqr, k) AS p,
           v, f,
           cnt + CASE WHEN v IS NOT NULL AND t > burn
                      THEN 1::BIGINT ELSE 0::BIGINT END AS cnt,
           sumlogf + CASE WHEN v IS NOT NULL AND t > burn
                          THEN CASE WHEN f > 0e0 THEN ln(f) ELSE NULL END
                          ELSE 0e0 END AS sumlogf,
           ssq + CASE WHEN v IS NOT NULL AND t > burn
                      THEN v * v / f ELSE 0e0 END AS ssq
    FROM (
        SELECT probe_id, t, cnt, sumlogf, ssq, k, cidx, burn, ct, v, f, ta, tpz,
               _sarimax_madd(psel, rqr) AS prqr
        FROM (
            SELECT probe_id, t, cnt, sumlogf, ssq, k, cidx, burn, rqr, ct, v, f,
                   ta, tpz,
                   CASE WHEN v IS NULL THEN tpt
                        ELSE _sarimax_msub(tpt, outerm) END AS psel
            FROM (
                SELECT probe_id, t, cnt, sumlogf, ssq, k, cidx, burn, rqr, ct,
                       v, f, ta, tpz,
                       _sarimax_mmul(tp, tmat_t, k, k, k) AS tpt,
                       list_transform(range(1, k * k + 1), lambda zidx:
                           tpz[(zidx - 1) // k + 1]
                           * tpz[(zidx - 1) % k + 1] / f) AS outerm
                FROM (
                    SELECT probe_id, t, cnt, sumlogf, ssq, k, cidx, burn,
                           tmat_t, rqr, ct, v, f, tp,
                           list_transform(range(1, k + 1), lambda zi:
                               tp[(zi - 1) * k + 1]) AS tpz,
                           ta
                    FROM (
                        SELECT kf.probe_id, kf.t + 1 AS t, kf.cnt, kf.sumlogf,
                               kf.ssq, s.k AS k, s.cidx, s.burn, s.tmat_t,
                               s.rqr,
                               coalesce(oc.ct, 0e0) AS ct,
                               o.yd - kf.a[1] AS v,
                               kf.p[1] AS f,
                               _sarimax_mmul(s.tmat, kf.p, s.k, s.k, s.k) AS tp,
                               _sarimax_mmul(s.tmat, kf.a, s.k, s.k, 1) AS ta
                        FROM _sarimax_kf2 kf
                        JOIN query_table(sys_tbl) s
                          ON s.probe_id = kf.probe_id
                        JOIN query_table(obs_tbl) o
                          ON o.probe_id = kf.probe_id AND o.t = kf.t + 1
                        LEFT JOIN query_table(obs_tbl) oc
                          ON oc.probe_id = kf.probe_id
                         AND oc.t = kf.t + 1
                                   + CASE WHEN s.kdiff > 0 THEN 1 ELSE 0 END
                    )
                )
            )
        )
    )
)
SELECT probe_id, t, v, f, cnt, sumlogf, ssq
FROM _sarimax_kf2
WHERE t >= 1;

-- Compact variant (USING KEY probe_id: each iteration replaces the row, only
-- the final state survives). Same arithmetic as _sarimax_kfilter_v2. Returns
-- (probe_id, n_eff, a, p, cnt, sumlogf, ssq); note the shifted-basis caveat
-- above about the missing c_{n+1} term in the final a when kdiff > 0.
CREATE OR REPLACE MACRO _sarimax_kfilter_state_v2(obs_tbl, sys_tbl) AS TABLE
WITH RECURSIVE _sarimax_kfs2 USING KEY (probe_id) AS (
    SELECT s.probe_id,
           0::BIGINT AS t,
           s.a1f AS a,
           s.p1f AS p,
           0::BIGINT AS cnt,
           0e0 AS sumlogf,
           0e0 AS ssq
    FROM query_table(sys_tbl) s
    UNION ALL
    SELECT probe_id, t,
           list_transform(range(1, k + 1), lambda zi:
               ta[zi]
               + (CASE WHEN v IS NULL THEN 0e0 ELSE tpz[zi] * v / f END)
               + (CASE WHEN zi = cidx THEN ct ELSE 0e0 END)) AS a,
           _sarimax_msym(prqr, k) AS p,
           cnt + CASE WHEN v IS NOT NULL AND t > burn
                      THEN 1::BIGINT ELSE 0::BIGINT END AS cnt,
           sumlogf + CASE WHEN v IS NOT NULL AND t > burn
                          THEN CASE WHEN f > 0e0 THEN ln(f) ELSE NULL END
                          ELSE 0e0 END AS sumlogf,
           ssq + CASE WHEN v IS NOT NULL AND t > burn
                      THEN v * v / f ELSE 0e0 END AS ssq
    FROM (
        SELECT probe_id, t, cnt, sumlogf, ssq, k, cidx, burn, ct, v, f, ta, tpz,
               _sarimax_madd(psel, rqr) AS prqr
        FROM (
            SELECT probe_id, t, cnt, sumlogf, ssq, k, cidx, burn, rqr, ct, v, f,
                   ta, tpz,
                   CASE WHEN v IS NULL THEN tpt
                        ELSE _sarimax_msub(tpt, outerm) END AS psel
            FROM (
                SELECT probe_id, t, cnt, sumlogf, ssq, k, cidx, burn, rqr, ct,
                       v, f, ta, tpz,
                       _sarimax_mmul(tp, tmat_t, k, k, k) AS tpt,
                       list_transform(range(1, k * k + 1), lambda zidx:
                           tpz[(zidx - 1) // k + 1]
                           * tpz[(zidx - 1) % k + 1] / f) AS outerm
                FROM (
                    SELECT probe_id, t, cnt, sumlogf, ssq, k, cidx, burn,
                           tmat_t, rqr, ct, v, f, tp,
                           list_transform(range(1, k + 1), lambda zi:
                               tp[(zi - 1) * k + 1]) AS tpz,
                           ta
                    FROM (
                        SELECT kfs.probe_id, kfs.t + 1 AS t, kfs.cnt,
                               kfs.sumlogf, kfs.ssq, s.k AS k, s.cidx, s.burn,
                               s.tmat_t, s.rqr,
                               coalesce(oc.ct, 0e0) AS ct,
                               o.yd - kfs.a[1] AS v,
                               kfs.p[1] AS f,
                               _sarimax_mmul(s.tmat, kfs.p, s.k, s.k, s.k) AS tp,
                               _sarimax_mmul(s.tmat, kfs.a, s.k, s.k, 1) AS ta
                        FROM _sarimax_kfs2 kfs
                        JOIN query_table(sys_tbl) s
                          ON s.probe_id = kfs.probe_id
                        JOIN query_table(obs_tbl) o
                          ON o.probe_id = kfs.probe_id AND o.t = kfs.t + 1
                        LEFT JOIN query_table(obs_tbl) oc
                          ON oc.probe_id = kfs.probe_id
                         AND oc.t = kfs.t + 1
                                   + CASE WHEN s.kdiff > 0 THEN 1 ELSE 0 END
                    )
                )
            )
        )
    )
)
SELECT probe_id, t AS n_eff, a, p, cnt, sumlogf, ssq
FROM _sarimax_kfs2;

-- Loglikelihood per probe. conc: 0 = sigma2 lives in RQR (standard formula),
-- 1 = concentrated scale (filter ran at sigma2 = 1; scale2 = ssq/cnt).
-- A NULL-poisoned sumlogf (some counted F_t <= 0) propagates to loglik NULL.
CREATE OR REPLACE MACRO _sarimax_loglik_v2(obs_tbl, sys_tbl, conc) AS TABLE
WITH _sarimax_ll2_args AS (
    SELECT conc::BIGINT AS zconc
)
SELECT st.probe_id, st.n_eff,
       CASE WHEN za.zconc = 1
            THEN -5e-1 * (st.cnt * ln(2e0 * pi()) + st.sumlogf
                          + st.cnt * ln(st.ssq / st.cnt) + st.cnt)
            ELSE -5e-1 * (st.cnt * ln(2e0 * pi()) + st.sumlogf + st.ssq)
       END AS loglik,
       CASE WHEN za.zconc = 1 THEN st.ssq / st.cnt
            ELSE NULL::DOUBLE END AS scale2
FROM _sarimax_kfilter_state_v2(obs_tbl, sys_tbl) st, _sarimax_ll2_args za;
