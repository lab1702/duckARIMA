"""duckARIMA golden-fixture generator (spec section 8).

Runs OFFLINE, ONCE, against pinned statsmodels/NumPy versions (tests/requirements.txt)
and emits Parquet files per fixture under tests/fixtures/<name>/. Parquet (not CSV) so
doubles round-trip losslessly as binary IEEE 754.

Pinned model options for every fixture (spec section 8):
    simple_differencing=True, enforce_stationarity=True, enforce_invertibility=True,
    concentrate_scale=False, hamilton_representation=False, measurement_error=False,
    mle_regression=True, trend='n'

Fixture regeneration policy: NEVER silently (spec section 10/12). If a fixture looks
wrong, reproduce the discrepancy in a standalone statsmodels snippet first, document
the cause, then regenerate deliberately.

Files emitted per fixture directory:
    series.parquet        (t BIGINT, y DOUBLE)                     original scale, t=1..n
    exog.parquet          (t BIGINT, j INT, x DOUBLE)              original scale, t=1..n+36 (0 rows if r=0)
    spec.parquet          one row: p,d,q,P,D,Q,s,r,n,n_eff,k_states,k_params
    probes.parquet        (probe_id, k, name, unconstrained, constrained)
    loglike.parquet       (probe_id, loglik)                       statsmodels loglike(theta, transformed=True)
    filter_trace.parquet  (probe_id, t, v, f)                      one-step-ahead innovation and its variance
    ssm.parquet           (probe_id in 1..5, name, i, j, v)        T, R, Z, state_cov, obs_cov, obs_intercept(i=t)
    fitted.parquet        (k, name, constrained, unconstrained, bse)   bse from cov_params_approx (numerical Hessian)
    fitted_meta.parquet   (loglik, sigma2, aic, bic, n, n_eff, k_params, converged)
    forecast.parquet      (h, mean_diff, se_diff, mean_orig, se_orig) h=1..36
    state.parquet         (name in ('a','P'), i, j, v)             final predicted state a_{n+1}, P_{n+1} at theta-hat
    ljungbox.parquet      (lag, stat, pvalue)                      on standardized innovations at theta-hat

Conventions the SQL side must match (verified by tests/introspection against statsmodels 0.14.6):
  * Parameter order: beta_1..beta_r, ar.L*, ma.L*, ar.S.L*, ma.S.L*, sigma2.
  * transform_params: AR/MA blocks via PACF r = x/sqrt(1+x^2) + Durbin-Levinson; the AR block
    is the NEGATED recursion output, the MA block the recursion output as-is; sigma2 = x^2.
  * Differencing: non-seasonal d times FIRST, then seasonal D times (np.diff then lag-s).
  * Future exog for forecasting is supplied on the ORIGINAL scale; statsmodels differences it
    using trailing in-sample values as anchors.
  * statsmodels forecasts under simple_differencing=True are on the DIFFERENCED scale;
    original-scale mean/se are computed here by exact linear integration of the differenced
    forecasts (weights c_{h,l}) with the forecast-error cross-covariance
    Cov(w_{n+i}, w_{n+j}) = Z T^{j-i} P_{n+i} Z' (j >= i, H = 0).
"""
from __future__ import annotations

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima_process import arma_generate_sample
from statsmodels.tsa.statespace.sarimax import SARIMAX

HERE = os.path.dirname(os.path.abspath(__file__))
FIXDIR = os.path.join(HERE, "fixtures")
H = 36              # forecast horizon exported
N_PROBES = 26       # 25 random-ish + theta-hat (spec: "25+ ... including theta-hat")
N_SSM_PROBES = 5

PINNED_OPTIONS = dict(
    trend="n",
    simple_differencing=True,
    enforce_stationarity=True,
    enforce_invertibility=True,
    concentrate_scale=False,
    hamilton_representation=False,
    measurement_error=False,
    mle_regression=True,
)

# The classic Box-Jenkins airline passengers series (monthly totals, 1949-1960).
AIRLINE = np.array([
    112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118,
    115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140,
    145, 150, 178, 163, 172, 178, 199, 199, 184, 162, 146, 166,
    171, 180, 193, 181, 183, 218, 230, 242, 209, 191, 172, 194,
    196, 196, 236, 235, 229, 243, 264, 272, 237, 211, 180, 201,
    204, 188, 235, 227, 234, 264, 302, 293, 259, 229, 203, 229,
    242, 233, 267, 269, 270, 315, 364, 347, 312, 274, 237, 278,
    284, 277, 317, 313, 318, 374, 413, 405, 355, 306, 271, 306,
    315, 301, 356, 348, 355, 422, 465, 467, 404, 347, 305, 336,
    340, 318, 362, 348, 363, 435, 491, 505, 404, 359, 310, 337,
    360, 342, 406, 396, 420, 472, 548, 559, 463, 407, 362, 405,
    417, 391, 419, 461, 472, 535, 622, 606, 508, 461, 390, 432,
], dtype=float)


# --------------------------------------------------------------------------
# differencing / integration helpers (pinned order: d ordinary FIRST, then D seasonal)
# --------------------------------------------------------------------------

def diff_series(z: np.ndarray, d: int, D: int, s: int) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    for _ in range(d):
        z = np.diff(z)
    for _ in range(D):
        z = z[s:] - z[:-s]
    return z


def integrate_forecast(w_future: np.ndarray, y_insample: np.ndarray,
                       d: int, D: int, s: int) -> np.ndarray:
    """Invert the differencing for h-step forecasts: seasonal stages first (reverse
    of application order), then ordinary stages, each anchored on the trailing values
    of the corresponding intermediate in-sample series."""
    stages = []          # intermediate series BEFORE each stage was applied, in application order
    z = np.asarray(y_insample, dtype=float)
    for _ in range(d):
        stages.append(z)
        z = np.diff(z)
    for _ in range(D):
        stages.append(z)
        z = z[s:] - z[:-s]
    fc = np.asarray(w_future, dtype=float).copy()
    for k in range(len(stages) - 1, -1, -1):
        prev = stages[k]
        lag = s if k >= d else 1
        out = np.empty_like(fc)
        for h in range(len(fc)):
            back = out[h - lag] if h - lag >= 0 else prev[len(prev) + (h - lag)]
            out[h] = fc[h] + back
        fc = out
    return fc


def integration_weights(n: int, d: int, D: int, s: int, horizon: int) -> np.ndarray:
    """c[h, l] with y_{n+h} = sum_l c[h, l] w_{n+l} + anchor term; by unit impulses
    through the (linear) integration with zero in-sample series."""
    zero_hist = np.zeros(n)
    C = np.zeros((horizon, horizon))
    for l in range(horizon):
        e = np.zeros(horizon)
        e[l] = 1.0
        C[:, l] = integrate_forecast(e, zero_hist, d, D, s)
    return C


# --------------------------------------------------------------------------
# simulation helpers
# --------------------------------------------------------------------------

def expand_poly(coefs: np.ndarray, seas: np.ndarray, s: int, sign: float) -> np.ndarray:
    """Reduced-form lag polynomial 1 + sign*c_1 L + ... convolved with seasonal.
    For AR pass sign=-1 (1 - phi L)(1 - Phi L^s); for MA sign=+1."""
    a = np.zeros(len(coefs) + 1)
    a[0] = 1.0
    a[1:] = sign * np.asarray(coefs, dtype=float)
    b = np.zeros(s * len(seas) + 1)
    b[0] = 1.0
    for i, c in enumerate(np.asarray(seas, dtype=float)):
        b[s * (i + 1)] = sign * c
    return np.convolve(a, b)


def simulate_sarima(rng, n, order, seasonal_order, params, sigma2, burnin=300):
    """Simulate an original-scale series whose (d, D, s)-difference is the ARMA
    process with the given (phi, theta, Phi, Theta)."""
    p, d, q = order
    P, D, Q, s = seasonal_order
    phi, theta, Phi, Theta = params
    ar = expand_poly(np.asarray(phi), np.asarray(Phi), max(s, 1), -1.0)
    ma = expand_poly(np.asarray(theta), np.asarray(Theta), max(s, 1), +1.0)
    n_eff = n - d - D * s
    w = arma_generate_sample(ar, ma, nsample=n_eff + burnin,
                             scale=np.sqrt(sigma2), distrvs=rng.standard_normal)[burnin:]
    # integrate back up to the original scale with mild random anchor values
    z = w
    for _ in range(D):                       # invert seasonal stages first (reverse order)
        head = rng.normal(scale=np.sqrt(sigma2), size=s)
        out = np.concatenate([head, np.zeros(len(z))])
        for i in range(len(z)):
            out[s + i] = z[i] + out[i]
        z = out
    for _ in range(d):
        head = rng.normal(scale=np.sqrt(sigma2), size=1)
        z = np.concatenate([head, head + np.cumsum(z)])
    assert len(z) == n
    return z


# --------------------------------------------------------------------------
# fixture definition and export
# --------------------------------------------------------------------------

class Fixture:
    def __init__(self, name, y, order, seasonal_order, exog=None, exog_future=None,
                 seed=0, fit_kwargs=None):
        self.name = name
        self.y = np.asarray(y, dtype=float)
        self.order = order
        self.seasonal_order = seasonal_order
        self.exog = exog                    # (n, r) original scale or None
        self.exog_future = exog_future      # (H, r) original scale or None
        self.seed = seed
        self.fit_kwargs = fit_kwargs or {}


def make_probes(model, res, rng, r):
    """26 probe parameter vectors: 12 interior, 8 near-boundary, 5 varied interior,
    plus theta-hat last. Returned in UNCONSTRAINED space; constrained via
    model.transform_params."""
    kp = len(model.param_names)
    n_arma = kp - r - 1
    sig_unc_hat = np.sqrt(res.params[-1])
    beta_hat = res.params[:r] if r else np.zeros(0)
    probes = []
    for i in range(N_PROBES - 1):
        u = np.empty(kp)
        if r:
            u[:r] = beta_hat + rng.normal(scale=np.maximum(1.0, np.abs(beta_hat)))
        if i < 12:                                   # interior
            u[r:r + n_arma] = rng.normal(scale=0.6, size=n_arma)
        elif i < 20:                                 # near-boundary: |pacf| ~ 0.93..0.97
            u[r:r + n_arma] = rng.normal(scale=0.4, size=n_arma)
            nb = max(1, n_arma // 2)
            pick = rng.choice(n_arma, size=nb, replace=False)
            u[r + pick] = rng.choice([-1.0, 1.0], size=nb) * rng.uniform(2.5, 4.0, size=nb)
        elif i < 25:                                 # varied interior
            u[r:r + n_arma] = rng.normal(scale=1.0, size=n_arma)
        u[-1] = sig_unc_hat * np.exp(rng.normal(scale=0.25))
        probes.append(u)
    probes.append(model.untransform_params(res.params))
    return probes


def ssm_rows(model, params, probe_id):
    model.update(np.asarray(params), transformed=True)
    rows = []
    def emit(name, arr):
        arr = np.asarray(arr, dtype=float)
        assert arr.ndim == 2, (name, arr.shape)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                rows.append((probe_id, name, i + 1, j + 1, float(arr[i, j])))
    T = model["transition"];  T = T[:, :, 0] if T.ndim == 3 else T
    R = model["selection"];   R = R[:, :, 0] if R.ndim == 3 else R
    Z = model["design"];      Z = Z[:, :, 0] if Z.ndim == 3 else Z
    emit("transition", T)
    emit("selection", R.reshape(model.k_states, 1))
    emit("design", Z.reshape(1, model.k_states))
    emit("state_cov", model["state_cov"].reshape(1, 1))
    emit("obs_cov", model["obs_cov"].reshape(1, 1))
    oi = model["obs_intercept"]
    oi = np.asarray(oi, dtype=float)
    if oi.size == 1:                                # time-invariant zero (r = 0)
        pass                                        # nothing to emit; SQL treats absent as 0
    else:
        for t in range(oi.shape[1]):
            rows.append((probe_id, "obs_intercept", t + 1, 1, float(oi[0, t])))
    return rows


def build_fixture(fx: Fixture):
    print(f"--- {fx.name}")
    rng = np.random.default_rng(fx.seed + 1000)
    p, d, q = fx.order
    P, D, Q, s = fx.seasonal_order
    r = 0 if fx.exog is None else fx.exog.shape[1]
    n = len(fx.y)

    model = SARIMAX(fx.y, exog=fx.exog, order=fx.order,
                    seasonal_order=fx.seasonal_order, **PINNED_OPTIONS)
    assert model.ssm.loglikelihood_burn == 0, "stationary init expected (burn 0)"
    n_eff = model.nobs
    assert n_eff == n - d - D * s

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = model.fit(disp=0, maxiter=1000, method="lbfgs", **fx.fit_kwargs)
    converged = bool(res.mle_retvals.get("converged", False))
    print(f"    llf={res.llf:.6f} converged={converged} params={np.round(res.params, 4)}")
    if not converged:
        raise RuntimeError(f"{fx.name}: statsmodels fit did not converge")

    outdir = os.path.join(FIXDIR, fx.name)
    os.makedirs(outdir, exist_ok=True)

    def wr(fname, df):
        df.to_parquet(os.path.join(outdir, fname), engine="pyarrow", index=False)

    wr("series.parquet", pd.DataFrame({
        "t": np.arange(1, n + 1, dtype=np.int64), "y": fx.y}))

    if r:
        xall = np.vstack([fx.exog, fx.exog_future])
        tt, jj = np.meshgrid(np.arange(1, n + H + 1, dtype=np.int64),
                             np.arange(1, r + 1, dtype=np.int32), indexing="ij")
        wr("exog.parquet", pd.DataFrame({
            "t": tt.ravel(), "j": jj.ravel(), "x": xall.ravel()}))
    else:
        wr("exog.parquet", pd.DataFrame({
            "t": pd.Series([], dtype=np.int64),
            "j": pd.Series([], dtype=np.int32),
            "x": pd.Series([], dtype=float)}))

    kp = len(model.param_names)
    wr("spec.parquet", pd.DataFrame([dict(
        p=p, d=d, q=q, bigp=P, bigd=D, bigq=Q, s=s, r=r,
        n=n, n_eff=n_eff, k_states=model.k_states, k_params=kp)]).astype(np.int64))

    # ---- probes, loglike, filter traces ----------------------------------
    probes_unc = make_probes(model, res, rng, r)
    probe_rows, ll_rows, trace_rows, ssm_acc = [], [], [], []
    for pid, u in enumerate(probes_unc, start=1):
        c = model.transform_params(u)
        for k in range(kp):
            probe_rows.append((pid, k + 1, model.param_names[k], float(u[k]), float(c[k])))
        ll = model.loglike(c, transformed=True)
        assert np.isfinite(ll), (fx.name, pid, c)
        ll_rows.append((pid, float(ll)))
        fr = model.filter(c, transformed=True)
        v = fr.forecasts_error.ravel()
        F = fr.forecasts_error_cov.ravel()
        assert np.isclose(ll, -0.5 * np.sum(np.log(2 * np.pi) + np.log(F) + v * v / F),
                          rtol=0, atol=1e-8 * max(1.0, abs(ll)))
        for t in range(n_eff):
            trace_rows.append((pid, t + 1, float(v[t]), float(F[t])))
        if pid <= N_SSM_PROBES:
            ssm_acc.extend(ssm_rows(model, c, pid))

    wr("probes.parquet", pd.DataFrame(
        probe_rows, columns=["probe_id", "k", "name", "unconstrained", "constrained"]))
    wr("loglike.parquet", pd.DataFrame(ll_rows, columns=["probe_id", "loglik"]))
    wr("filter_trace.parquet", pd.DataFrame(
        trace_rows, columns=["probe_id", "t", "v", "f"]))
    wr("ssm.parquet", pd.DataFrame(
        ssm_acc, columns=["probe_id", "name", "i", "j", "v"]))

    # ---- fitted parameters / bse (numerical-Hessian flavor) ---------------
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bse = np.sqrt(np.diag(res.cov_params_approx))
    unc_hat = model.untransform_params(res.params)
    wr("fitted.parquet", pd.DataFrame({
        "k": np.arange(1, kp + 1, dtype=np.int64),
        "name": model.param_names,
        "constrained": res.params,
        "unconstrained": unc_hat,
        "bse": bse}))
    wr("fitted_meta.parquet", pd.DataFrame([dict(
        loglik=float(res.llf), sigma2=float(res.params[-1]),
        aic=float(res.aic), bic=float(res.bic),
        n=float(n), n_eff=float(n_eff), k_params=float(kp),
        converged=float(converged))]))

    # ---- forecasts: differenced scale (statsmodels) + original scale ------
    fr_hat = model.filter(res.params, transformed=True)
    if r:
        fc = fr_hat.get_forecast(H, exog=fx.exog_future)
    else:
        fc = fr_hat.get_forecast(H)
    mean_diff = np.asarray(fc.predicted_mean, dtype=float)
    se_diff = np.asarray(fc.se_mean, dtype=float)

    # exact original-scale integration
    Tm = model["transition"];  Tm = Tm[:, :, 0] if Tm.ndim == 3 else Tm
    Rm = model["selection"].reshape(model.k_states)
    Zm = model["design"].reshape(model.k_states)
    Qs = float(model["state_cov"].ravel()[0])
    RQR = np.outer(Rm, Rm) * Qs
    a = fr_hat.predicted_state[:, -1].copy()
    Pm = fr_hat.predicted_state_cov[:, :, -1].copy()
    # per-horizon predicted covariances and cross terms
    Ps = []
    for _ in range(H):
        Ps.append(Pm.copy())
        Pm = Tm @ Pm @ Tm.T + RQR
    Omega = np.zeros((H, H))
    for i in range(H):
        g = Zm.copy()
        for j in range(i, H):                       # Cov(w_{n+1+j}, w_{n+1+i}) = Z T^{j-i} P_i Z'
            Omega[j, i] = Omega[i, j] = g @ Ps[i] @ Zm
            g = g @ Tm
    assert np.allclose(np.sqrt(np.diag(Omega)), se_diff, rtol=1e-10)

    mean_orig = integrate_forecast(mean_diff, fx.y, d, D, s)
    C = integration_weights(n, d, D, s, H)
    se_orig = np.sqrt(np.einsum("hl,lm,hm->h", C, Omega, C))

    wr("forecast.parquet", pd.DataFrame({
        "h": np.arange(1, H + 1, dtype=np.int64),
        "mean_diff": mean_diff, "se_diff": se_diff,
        "mean_orig": mean_orig, "se_orig": se_orig}))

    # ---- final predicted state at theta-hat --------------------------------
    st_rows = [("a", i + 1, 1, float(fr_hat.predicted_state[i, -1]))
               for i in range(model.k_states)]
    Pn1 = fr_hat.predicted_state_cov[:, :, -1]
    st_rows += [("P", i + 1, j + 1, float(Pn1[i, j]))
                for i in range(model.k_states) for j in range(model.k_states)]
    wr("state.parquet", pd.DataFrame(st_rows, columns=["name", "i", "j", "v"]))

    # ---- Ljung-Box on standardized innovations at theta-hat ----------------
    std_innov = fr_hat.standardized_forecasts_error.ravel()
    lags = list(range(1, min(24, n_eff // 4) + 1))
    lb = acorr_ljungbox(std_innov, lags=lags, return_df=True)
    wr("ljungbox.parquet", pd.DataFrame({
        "lag": np.asarray(lags, dtype=np.int64),
        "stat": lb["lb_stat"].to_numpy(dtype=float),
        "pvalue": lb["lb_pvalue"].to_numpy(dtype=float)}))

    return dict(name=fx.name, p=p, d=d, q=q, P=P, D=D, Q=Q, s=s, r=r, n=n,
                n_eff=n_eff, k_states=model.k_states, k_params=kp,
                llf=float(res.llf))


# --------------------------------------------------------------------------
# the eleven fixtures (spec section 8)
# --------------------------------------------------------------------------

def build_all():
    fixtures = []

    rng = np.random.default_rng(101)
    y = simulate_sarima(rng, 400, (1, 0, 1), (0, 0, 0, 1), ([0.6], [0.3], [], []), 1.2)
    fixtures.append(Fixture("arma_1_0_1", y, (1, 0, 1), (0, 0, 0, 0), seed=101))

    rng = np.random.default_rng(102)
    y = simulate_sarima(rng, 300, (1, 1, 1), (0, 0, 0, 1), ([0.5], [-0.35], [], []), 0.8)
    fixtures.append(Fixture("arima_1_1_1", y, (1, 1, 1), (0, 0, 0, 0), seed=102))

    rng = np.random.default_rng(103)
    y = simulate_sarima(rng, 400, (2, 1, 2), (0, 0, 0, 1), ([0.5, -0.3], [0.4, 0.25], [], []), 1.0)
    fixtures.append(Fixture("arima_2_1_2", y, (2, 1, 2), (0, 0, 0, 0), seed=103))

    rng = np.random.default_rng(104)
    y = simulate_sarima(rng, 240, (1, 0, 0), (1, 0, 0, 4), ([0.5], [], [0.4], []), 1.0)
    fixtures.append(Fixture("sarima_100_100_4", y, (1, 0, 0), (1, 0, 0, 4), seed=104))

    fixtures.append(Fixture("airline", np.log(AIRLINE), (0, 1, 1), (0, 1, 1, 12), seed=105))

    rng = np.random.default_rng(106)
    y = simulate_sarima(rng, 300, (1, 1, 1), (1, 1, 1, 12),
                        ([0.4], [0.3], [0.35], [-0.4]), 1.0)
    fixtures.append(Fixture("sarima_111_111_12", y, (1, 1, 1), (1, 1, 1, 12), seed=106))

    rng = np.random.default_rng(107)
    y = simulate_sarima(rng, 360, (2, 1, 0), (0, 1, 1, 12),
                        ([0.45, -0.25], [], [], [-0.5]), 1.0)
    fixtures.append(Fixture("sarima_210_011_12", y, (2, 1, 0), (0, 1, 1, 12), seed=107))

    rng = np.random.default_rng(108)   # near-nonstationary: largest AR root modulus 0.98
    y = simulate_sarima(rng, 500, (1, 0, 1), (0, 0, 0, 1), ([0.98], [0.2], [], []), 1.0)
    fixtures.append(Fixture("near_nonstationary", y, (1, 0, 1), (0, 0, 0, 0), seed=108))

    rng = np.random.default_rng(109)   # near-noninvertible: theta = -0.95
    y = simulate_sarima(rng, 400, (0, 1, 1), (0, 0, 0, 1), ([], [-0.95], [], []), 1.0)
    fixtures.append(Fixture("near_noninvertible", y, (0, 1, 1), (0, 0, 0, 0), seed=109))

    # ARIMAX(1,1,1), r=2: smooth trend-like + pulse intervention
    rng = np.random.default_rng(110)
    n = 300
    t_all = np.arange(1, n + H + 1, dtype=float)
    x1 = np.sin(2 * np.pi * t_all / 100.0) + t_all / 150.0
    x2 = (t_all == 150).astype(float)
    X = np.column_stack([x1, x2])
    beta = np.array([1.5, -2.0])
    u = simulate_sarima(rng, n, (1, 1, 1), (0, 0, 0, 1), ([0.5], [0.3], [], []), 1.0)
    xd = np.column_stack([diff_series(X[:n, 0], 1, 0, 1), diff_series(X[:n, 1], 1, 0, 1)])
    w = diff_series(u, 1, 0, 1) + xd @ beta
    y = np.concatenate([[u[0]], u[0] + np.cumsum(w)])
    fixtures.append(Fixture("arimax_1_1_1", y, (1, 1, 1), (0, 0, 0, 0),
                            exog=X[:n], exog_future=X[n:], seed=110))

    # SARIMAX(0,1,1)(0,1,1)_12, r=2: airline model plus regressors
    rng = np.random.default_rng(111)
    n = len(AIRLINE)
    t_all = np.arange(1, n + H + 1, dtype=float)
    x1 = np.sin(2 * np.pi * t_all / 60.0) * 0.5 + t_all / 400.0
    x2 = ((t_all >= 90) & (t_all < 96)).astype(float)    # 6-month intervention window
    X = np.column_stack([x1, x2])
    beta = np.array([0.08, -0.05])
    y = np.log(AIRLINE) + X[:n] @ beta
    fixtures.append(Fixture("sarimax_011_011_12", y, (0, 1, 1), (0, 1, 1, 12),
                            exog=X[:n], exog_future=X[n:], seed=111))

    os.makedirs(FIXDIR, exist_ok=True)
    manifest = [build_fixture(fx) for fx in fixtures]
    mdf = pd.DataFrame(manifest)
    mdf.to_parquet(os.path.join(FIXDIR, "manifest.parquet"), engine="pyarrow", index=False)
    with open(os.path.join(FIXDIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n{len(manifest)} fixtures written to {FIXDIR}")


if __name__ == "__main__":
    import statsmodels
    print(f"statsmodels {statsmodels.__version__}, numpy {np.__version__}, "
          f"pandas {pd.__version__}, python {sys.version.split()[0]}")
    build_all()
