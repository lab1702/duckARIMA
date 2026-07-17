"""duckARIMA v2 golden-fixture generator: concentrated scale, trend terms,
missing values, and simple_differencing=False (approximate diffuse).

Writes to tests/fixtures_v2/<name>/ -- the v1 fixtures are NEVER touched.
Reuses the v1 generator's helpers and pinned options; same regeneration
policy (never silently; see tests/README.md).

Additional pinned conventions (verified by introspection, 2026-07-17):
  * trend enters via the STATE intercept: c_t = trend_data_t . tau added to
    the first ARMA-block state row (1-based index k_diff + 1, where
    k_diff = d + s*D when simple_differencing=False, else 0), with
    trend_data_t = [t**deg for deg in degrees], t = 1..n 1-based on the
    MODEL timeline. Timing: a_{t+1} = T a_t + K v_t + c_t; the stationary
    initialization solves a_1 = (I - T)^-1 c_1 on the ARMA block.
  * concentrate_scale=True: params exclude sigma2; the filter runs at
    sigma2 = 1; scale = sum_{t>burn, non-missing} v~^2/F~ / (nobs - burn);
    statsmodels' REPORTED trace has F multiplied by the scale, so the
    standard ll formula over non-missing t > burn applies to the reported
    trace as-is (mean of reported v^2/F over t>burn is exactly 1).
  * missing y: v_t reported NaN, F_t still reported; ll sums non-missing
    t > burn; AIC/BIC use nobs_effective = nobs - burn (NOT reduced by the
    missing count).
  * simple_differencing=False: k_states = d + s*D + k_arma;
    P1 = blockdiag(1e6 * I_{d+s*D}, lyapunov_arma) with zero cross terms;
    loglikelihood_burn = d + s*D; exog obs_intercept uses UNdifferenced x;
    forecasts are on the original scale directly (se = sqrt(Z P Z')).

Exported extras vs v1 fixtures: spec gains (burn, sdiff, conc, ktrend);
trend.parquet lists the polynomial degrees; ssm.parquet additionally holds
'state_intercept' rows (i = t, j = state row) and the exact initialization
'a1' (i, 1) and 'P1' (i, j) per ssm probe. forecast.parquet keeps the v1
schema with mean_diff/se_diff = raw statsmodels output = mean_orig/se_orig
(all v2 fixtures use simple_differencing=False, so the model scale IS the
original scale).
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_fixtures import (  # noqa: E402
    AIRLINE, H, N_SSM_PROBES, PINNED_OPTIONS, simulate_sarima, ssm_rows)

HERE = os.path.dirname(os.path.abspath(__file__))
FIXDIR_V2 = os.path.join(HERE, "fixtures_v2")
N_PROBES_V2 = 21


class FixtureV2:
    def __init__(self, name, y, order, seasonal_order, exog=None, exog_future=None,
                 seed=0, trend="n", concentrate=False, missing_idx=()):
        self.name, self.y = name, np.asarray(y, dtype=float)
        self.order, self.seasonal_order = order, seasonal_order
        self.exog, self.exog_future, self.seed = exog, exog_future, seed
        self.trend, self.conc = trend, concentrate
        self.missing_idx = list(missing_idx)


def make_probes_v2(model, res, rng, r, ktrend, conc):
    kp = len(model.param_names)
    n_arma = kp - r - ktrend - (0 if conc else 1)
    fitted = np.asarray(res.params, dtype=float)
    probes = []
    for i in range(N_PROBES_V2 - 1):
        u = np.empty(kp)
        head = ktrend + r
        if head:
            u[:head] = fitted[:head] + rng.normal(
                scale=np.maximum(0.5, np.abs(fitted[:head])))
        u[head:head + n_arma] = rng.normal(scale=0.6, size=n_arma)
        if i >= 14:                                   # a few near-boundary probes
            nb = max(1, n_arma // 2)
            pick = rng.choice(n_arma, size=nb, replace=False)
            u[head + pick] = rng.choice([-1.0, 1.0], size=nb) * rng.uniform(2.5, 3.5, size=nb)
        if not conc:
            u[-1] = np.sqrt(max(float(fitted[-1]), 1e-8)) * np.exp(rng.normal(scale=0.25))
        probes.append(u)
    probes.append(model.untransform_params(fitted))
    return probes


def build_fixture_v2(fx: FixtureV2):
    print(f"--- v2 {fx.name}")
    rng = np.random.default_rng(fx.seed + 5000)
    p, d, q = fx.order
    P, D, Q, s = fx.seasonal_order
    r = 0 if fx.exog is None else fx.exog.shape[1]
    n = len(fx.y)
    y = fx.y.copy()
    for i in fx.missing_idx:
        y[i] = np.nan

    opts = dict(PINNED_OPTIONS)
    opts["trend"] = fx.trend
    opts["simple_differencing"] = False
    opts["concentrate_scale"] = fx.conc
    model = SARIMAX(y, exog=fx.exog, order=fx.order,
                    seasonal_order=fx.seasonal_order, **opts)
    model.ssm.tolerance = 0.0
    burn = int(model.ssm.loglikelihood_burn)
    n_eff = model.nobs
    ktrend = int(model.k_trend)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = model.fit(disp=0, maxiter=1000, method="lbfgs")
    converged = bool(res.mle_retvals.get("converged", False))
    scale = float(res.filter_results.scale) if fx.conc else float(res.params[-1])
    print(f"    llf={res.llf:.6f} converged={converged} burn={burn} "
          f"k_states={model.k_states} params={np.round(np.asarray(res.params), 4)} "
          f"scale={scale:.6g}")
    if not converged:
        raise RuntimeError(f"{fx.name}: statsmodels fit did not converge")

    outdir = os.path.join(FIXDIR_V2, fx.name)
    os.makedirs(outdir, exist_ok=True)

    def wr(fname, df):
        df.to_parquet(os.path.join(outdir, fname), engine="pyarrow", index=False)

    wr("series.parquet", pd.DataFrame({
        "t": np.arange(1, n + 1, dtype=np.int64), "y": y}))
    if r:
        xall = np.vstack([fx.exog, fx.exog_future])
        tt, jj = np.meshgrid(np.arange(1, n + H + 1, dtype=np.int64),
                             np.arange(1, r + 1, dtype=np.int32), indexing="ij")
        wr("exog.parquet", pd.DataFrame({"t": tt.ravel(), "j": jj.ravel(),
                                         "x": xall.ravel()}))
    else:
        wr("exog.parquet", pd.DataFrame({
            "t": pd.Series([], dtype=np.int64), "j": pd.Series([], dtype=np.int32),
            "x": pd.Series([], dtype=float)}))

    kp = len(model.param_names)
    wr("spec.parquet", pd.DataFrame([dict(
        p=p, d=d, q=q, bigp=P, bigd=D, bigq=Q, s=s, r=r, n=n, n_eff=n_eff,
        k_states=int(model.k_states), k_params=kp, burn=burn,
        sdiff=0, conc=int(fx.conc), ktrend=ktrend)]).astype(np.int64))
    degrees = [int(i) for i, on in enumerate(np.asarray(model.polynomial_trend)) if on]
    wr("trend.parquet", pd.DataFrame({
        "idx": np.arange(1, len(degrees) + 1, dtype=np.int64),
        "degree": np.asarray(degrees, dtype=np.int64)}))

    probes_unc = make_probes_v2(model, res, rng, r, ktrend, fx.conc)
    probe_rows, ll_rows, trace_rows, ssm_acc = [], [], [], []
    for pid, u in enumerate(probes_unc, start=1):
        c = model.transform_params(u)
        for k in range(kp):
            probe_rows.append((pid, k + 1, model.param_names[k], float(u[k]), float(c[k])))
        ll = model.loglike(c, transformed=True)
        assert np.isfinite(ll), (fx.name, pid)
        ll_rows.append((pid, float(ll)))
        fr = model.filter(c, transformed=True)
        v = fr.forecasts_error.ravel()
        F = fr.forecasts_error_cov.ravel()
        mask = ~np.isnan(v)
        mask[:burn] = False
        assert np.isclose(
            ll, -0.5 * np.sum(np.log(2 * np.pi) + np.log(F[mask]) + v[mask] ** 2 / F[mask]),
            rtol=0, atol=1e-8 * max(1.0, abs(ll))), (fx.name, pid)
        for t in range(n_eff):
            trace_rows.append((pid, t + 1, float(v[t]), float(F[t])))
        if pid <= N_SSM_PROBES:
            ssm_acc.extend(ssm_rows(model, c, pid))
            model.update(np.asarray(c), transformed=True)
            si = np.atleast_2d(model["state_intercept"])
            if si.shape[1] > 1:
                for i in range(si.shape[0]):
                    for t in range(si.shape[1]):
                        if si[i, t] != 0.0:
                            ssm_acc.append((pid, "state_intercept", t + 1, i + 1,
                                            float(si[i, t])))
            frp = model.filter(np.asarray(c), transformed=True).filter_results
            a1 = frp.predicted_state[:, 0]
            P1 = frp.predicted_state_cov[:, :, 0]
            ssm_acc.extend((pid, "a1", i + 1, 1, float(a1[i]))
                           for i in range(model.k_states))
            ssm_acc.extend((pid, "P1", i + 1, j + 1, float(P1[i, j]))
                           for i in range(model.k_states) for j in range(model.k_states))

    wr("probes.parquet", pd.DataFrame(
        probe_rows, columns=["probe_id", "k", "name", "unconstrained", "constrained"]))
    wr("loglike.parquet", pd.DataFrame(ll_rows, columns=["probe_id", "loglik"]))
    wr("filter_trace.parquet", pd.DataFrame(
        trace_rows, columns=["probe_id", "t", "v", "f"]))
    wr("ssm.parquet", pd.DataFrame(ssm_acc, columns=["probe_id", "name", "i", "j", "v"]))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bse = np.sqrt(np.diag(res.cov_params_approx))
    wr("fitted.parquet", pd.DataFrame({
        "k": np.arange(1, kp + 1, dtype=np.int64),
        "name": model.param_names,
        "constrained": np.asarray(res.params, dtype=float),
        "unconstrained": model.untransform_params(np.asarray(res.params)),
        "bse": bse}))
    wr("fitted_meta.parquet", pd.DataFrame([dict(
        loglik=float(res.llf), sigma2=scale, aic=float(res.aic), bic=float(res.bic),
        n=float(n), n_eff=float(n_eff), k_params=float(kp), burn=float(burn),
        nobs_effective=float(res.nobs_effective), converged=float(converged))]))

    fr_hat = model.filter(np.asarray(res.params), transformed=True)
    fc = fr_hat.get_forecast(H, exog=fx.exog_future) if r else fr_hat.get_forecast(H)
    mean_raw = np.asarray(fc.predicted_mean, dtype=float)
    se_raw = np.asarray(fc.se_mean, dtype=float)
    wr("forecast.parquet", pd.DataFrame({
        "h": np.arange(1, H + 1, dtype=np.int64),
        "mean_diff": mean_raw, "se_diff": se_raw,
        "mean_orig": mean_raw, "se_orig": se_raw}))

    st_rows = [("a", i + 1, 1, float(fr_hat.filter_results.predicted_state[i, -1]))
               for i in range(model.k_states)]
    Pn1 = fr_hat.filter_results.predicted_state_cov[:, :, -1]
    st_rows += [("P", i + 1, j + 1, float(Pn1[i, j]))
                for i in range(model.k_states) for j in range(model.k_states)]
    wr("state.parquet", pd.DataFrame(st_rows, columns=["name", "i", "j", "v"]))
    return dict(name=fx.name, llf=float(res.llf), burn=burn,
                k_states=int(model.k_states), k_params=kp)


def build_all_v2():
    fixtures = []

    rng = np.random.default_rng(201)
    y = simulate_sarima(rng, 300, (1, 0, 1), (0, 0, 0, 1), ([0.6], [0.3], [], []), 1.4)
    fixtures.append(FixtureV2("conc_arma_101", y, (1, 0, 1), (0, 0, 0, 0),
                              seed=201, concentrate=True))

    rng = np.random.default_rng(202)
    u = simulate_sarima(rng, 250, (1, 0, 0), (0, 0, 0, 1), ([0.5], [], [], []), 1.0)
    t_idx = np.arange(1, 251, dtype=float)
    y = u + 0.3 * t_idx
    fixtures.append(FixtureV2("trend_ct_arma_100", y, (1, 0, 0), (0, 0, 0, 0),
                              seed=202, trend="ct"))

    rng = np.random.default_rng(203)
    y = simulate_sarima(rng, 300, (1, 0, 1), (0, 0, 0, 1), ([0.5], [-0.35], [], []), 0.8)
    miss = sorted(int(i) for i in rng.choice(np.arange(20, 280), size=15, replace=False))
    fixtures.append(FixtureV2("missing_arma_101", y, (1, 0, 1), (0, 0, 0, 0),
                              seed=203, missing_idx=miss))

    rng = np.random.default_rng(204)
    y = simulate_sarima(rng, 250, (1, 1, 1), (0, 0, 0, 1), ([0.5], [0.3], [], []), 1.0)
    fixtures.append(FixtureV2("nodiff_arima_111", y, (1, 1, 1), (0, 0, 0, 0), seed=204))

    rng = np.random.default_rng(205)
    n = len(AIRLINE)
    t_all = np.arange(1, n + H + 1, dtype=float)
    x1 = np.sin(2 * np.pi * t_all / 60.0) * 0.5 + t_all / 400.0
    x2 = ((t_all >= 90) & (t_all < 96)).astype(float)
    X = np.column_stack([x1, x2])
    y = np.log(AIRLINE) + X[:n] @ np.array([0.08, -0.05])
    fixtures.append(FixtureV2("nodiff_sarimax_011_011_12", y, (0, 1, 1), (0, 1, 1, 12),
                              exog=X[:n], exog_future=X[n:], seed=205))

    rng = np.random.default_rng(206)
    n = 200
    t_all = np.arange(1, n + H + 1, dtype=float)
    X = np.column_stack([np.sin(2 * np.pi * t_all / 40.0) + t_all / 150.0])
    u = simulate_sarima(rng, n, (1, 1, 1), (0, 1, 1, 4), ([0.4], [0.3], [], [-0.4]), 1.0)
    y = u + X[:n, 0] * 1.2 + 0.02 * t_all[:n]
    miss = sorted(int(i) for i in rng.choice(np.arange(15, 185), size=8, replace=False))
    fixtures.append(FixtureV2("kitchen_sink", y, (1, 1, 1), (0, 1, 1, 4),
                              exog=X[:n], exog_future=X[n:], seed=206, trend="ct",
                              concentrate=True, missing_idx=miss))

    os.makedirs(FIXDIR_V2, exist_ok=True)
    manifest = [build_fixture_v2(fx) for fx in fixtures]
    pd.DataFrame(manifest).to_parquet(
        os.path.join(FIXDIR_V2, "manifest.parquet"), engine="pyarrow", index=False)
    with open(os.path.join(FIXDIR_V2, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n{len(manifest)} v2 fixtures written to {FIXDIR_V2}")


if __name__ == "__main__":
    import statsmodels
    print(f"statsmodels {statsmodels.__version__}, numpy {np.__version__}, "
          f"pandas {pd.__version__}")
    build_all_v2()
