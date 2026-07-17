# duckARIMA user's guide

SARIMAX — seasonal ARIMA with exogenous regressors — implemented entirely in
DuckDB SQL macros and validated against statsmodels. This guide covers usage,
the statistical conventions the implementation pins, the model-table contract,
and edge-case behavior. For one-line signatures see CHEATSHEET.md.

## 1. Quick start

```sql
.read sarimax_macros.sql

-- a monthly series in a table `sales(month, units, promo, price)`
CREATE TABLE m AS SELECT * FROM sarimax_fit(
    'sales', 'units',
    0, 1, 1,                     -- p, d, q
    sd := 1, sq := 1, s := 12,   -- seasonal D, Q, season length
    exog_cols := ['promo', 'price'],
    t_col := 'month');

SELECT * FROM sarimax_summary('m', 'sales', 'units');

-- future_regressors(month, promo, price) must hold the next 12 rows
SELECT * FROM sarimax_forecast('m', 'sales', 'units', 12,
                               newdata := 'future_regressors',
                               exog_cols := ['promo', 'price'], t_col := 'month');
```

The model table `m` is self-contained: forecasting, summary, and evaluation
never refit. `sarimax_residuals`/`sarimax_ljungbox`/`sarimax_evaluate` re-run
only the (fast, deterministic) Kalman filter at the stored parameters.

## 2. The model

With w_t = Δ^d Δ_s^D y_t and x̃_t the identically differenced regressors:

φ(L) Φ(L^s) (w_t − x̃_t′β) = θ(L) Θ(L^s) ε_t,   ε_t ~ iid N(0, σ²)

- AR polynomials carry minus signs (1 − φ₁L − …), MA polynomials plus signs
  (1 + θ₁L + …) — **statsmodels' convention** (Box–Jenkins texts write MA with
  minus; flip signs when comparing to such tables).
- Estimation is exact Gaussian MLE via the Kalman filter on the Harvey
  companion state space, with stationary initialization (the discrete
  Lyapunov equation) of the ARMA block. With the default flags this is
  equivalent to statsmodels' `SARIMAX(..., simple_differencing=True,
  trend='n', mle_regression=True)`.
- Trend/constant terms are available via `trend := 'c' | 't' | 'ct'`
  (statsmodels' `intercept`/`drift` parameters, entering the transition
  equation as a state intercept); `concentrate := true` concentrates σ² out
  of the likelihood; `simple_differencing := false` estimates on the raw
  series with the differencing built into the state vector. See section
  "v1 vs v2 estimation modes" below.

## 3. Conventions pinned by this implementation

These are the decisions the spec left to "match statsmodels' observable
behavior", recorded here so nothing is implicit in code:

1. **Parameter order** (everywhere: model table `idx`, internal vectors):
   trend terms (`intercept` then `drift`, when requested), β₁..β_r (in
   `exog_cols` order), φ₁..φ_p, θ₁..θ_q, Φ₁..Φ_P, Θ₁..Θ_Q, σ². When
   `concentrate := true`, σ² is **not a parameter** (no `param` row) and is
   reported only in `meta.sigma2`.
2. **Differencing order**: d ordinary differences first, then D seasonal
   (lag s) differences — matching `np.diff` + seasonal loop under statsmodels'
   `simple_differencing=True`. Exog columns are differenced in lockstep.
3. **Unconstrained parameter space** (kind `param_unc`): the Monahan/Jones
   PACF bijection per polynomial block exactly as statsmodels implements it
   (PACF = x/√(1+x²), Durbin–Levinson; the AR block is the negated recursion
   output); β passes through unchanged; **σ² untransformed is √σ²**
   (statsmodels' square convention — not log — so unconstrained vectors are
   directly comparable to statsmodels').
4. **Loglikelihood**: summed over all timesteps (no burn-in), strict t-order
   accumulation. The reference is statsmodels' *exact* filter: fixtures pin
   `ssm.tolerance = 0`, disabling statsmodels' convergence freezing of the
   gain (that speed shortcut perturbs innovations by up to ~1e-6 near the
   invertibility boundary; this implementation always computes the exact
   recursion).
5. **Forecast variance and exog**: future exog values are treated as known
   constants — they shift the point forecast and contribute nothing to the
   interval; no parameter-uncertainty term for β̂ is added. Supply future exog
   on the **original scale**; they are differenced internally using the stored
   trailing in-sample values, exactly as statsmodels does.
6. **Differencing inversion**: point forecasts integrate the differenced-scale
   forecasts anchored on the stored trailing values; forecast variance maps
   through the integration weights applied to the full forecast-error
   covariance (cross-covariances included), so original-scale intervals are
   exact, not diagonal approximations.
7. **Standard errors**: central-difference numerical Hessian of the
   loglikelihood in the natural (constrained) parameter space at θ̂, inverted
   and negated — comparable to statsmodels' `cov_params_approx`.
8. **AIC/BIC**: AIC = 2k − 2ℓ, BIC = k·ln(n_eff − burn) − 2ℓ. k **always
   counts σ², even when the scale is concentrated** (statsmodels' `df_model`
   does the same — pinned empirically against the fixtures_v2 references),
   and the BIC sample size is statsmodels' `nobs_effective` = n_eff − burn,
   where n_eff is the model-timeline length (the differenced length under
   simple differencing, n otherwise) and burn = d + s·D when
   `simple_differencing := false`, else 0. NOTE: missing observations do
   NOT reduce n_eff (statsmodels' convention).
9. **Optimizer** (BFGS, Armijo backtracking, central-difference gradients) —
   constants pinned in the spec §5.4. `converged` in `meta` reports the
   gradient criterion; `restarted = 1` flags the one perturbed restart.
10. **Seasonal argument names**: `sp/sd/sq` stand in for P/D/Q because DuckDB
    macro parameters are case-insensitive.
11. **Re-supplying `exog_cols`**: the forecast/diagnostics macros take the
    regressor column names again as a literal list (struct-field access in
    DuckDB requires constant keys, so names stored in the model table cannot
    drive extraction); the list is validated against the stored names. At
    most 32 exogenous columns are supported.
12. **Grid search**: `sarimax_grid_sql(data, y_col, orders)` returns the SQL
    text that fits every row of the orders table and ranks by AIC (DuckDB
    cannot correlate table-macro arguments through a lateral join, so the
    grid runs as a generated statement).

## 4. The model table, column by column

`(kind VARCHAR, name VARCHAR, idx INT, value DOUBLE, value_list DOUBLE[])`

| kind | name | idx | value / value_list |
|---|---|---|---|
| `param` | parameter name (`intercept`/`drift` trend terms, `exog_cols` entries, `ar.L1…`, `ma.L1…`, `ar.S.L<s>…`, `ma.S.L<s>…`, `sigma2` — the latter absent when concentrated) | canonical position 1..k | constrained estimate |
| `param_unc` | same | same | unconstrained (optimizer-space) estimate |
| `bse` | same | same | standard error |
| `spec` | `p,d,q,sp,sd,sq,s,r,n,n_eff,sdiff,conc,ktrend,burn` | 1..14 | the value (`sdiff` = 1 iff fit with simple differencing; `conc` = 1 iff concentrated; `ktrend` = number of trend terms; `burn` = loglikelihood burn-in) |
| `meta` | `loglik,aic,bic,converged,iterations,grad_norm,restarted,sigma2` | 1..8 | the value (`sigma2` = the concentrated scale when `conc` = 1) |
| `trend` | `intercept` / `drift` | 1..ktrend | polynomial degree (0 / 1) |
| `anchor` | `endog:<stage>` | position within stage | trailing pre-differencing values (Layer-1 anchor contract: stages 1..d ordinary then d+1..d+D seasonal; ordinary stages store 1 value, seasonal stages s values). **Only when `sdiff` = 1** |
| `anchor` | `exog:<j>` | 1..(d+D·s) | trailing original-scale exog values used to difference future exog. **Only when `sdiff` = 1** |
| `exog_col` | the j-th exog column name | j | — |
| `state` | `a` / `P` | — | value_list: final predicted state mean / row-major vec of its covariance (P at **unit scale** when concentrated) |
| `state` | `k` | — | value: state dimension |

## 5. v1 vs v2 estimation modes

Every fit runs through one estimation engine; the flags select the model
formulation. The v1 behavior is exactly the default flag combination
(`trend := 'n', concentrate := false, simple_differencing := true` on a
complete series) — the engine reproduces the original v1 arithmetic in that
case, which is regression-tested.

**`simple_differencing := false`** changes the formulation, not just the
implementation:

- **No data loss.** Under simple differencing, d + s·D observations are
  consumed before the filter ever runs, and the likelihood is that of the
  differenced series (n_eff = n − d − s·D). With
  `simple_differencing := false` the raw series is filtered directly: the
  d + s·D differencing states live inside the state vector, n_eff = n, and
  the two likelihoods are **different functions** — expect slightly
  different estimates between the modes on the same data (both match
  statsmodels' corresponding option).
- **Burn-in.** The differencing states carry no stationary distribution;
  they are initialized approximately diffuse (variance κ = 1e6, statsmodels'
  `initialization='approximate_diffuse'` behavior on that block, with the
  stationary Lyapunov solution on the ARMA block and zero cross terms). The
  first burn = d + s·D innovations are therefore excluded from the
  loglikelihood (`loglikelihood_burn`), and AIC/BIC use
  nobs_effective = n − burn.
- **Forecasting needs no anchors.** The model scale IS the original scale:
  forecasts come straight from the state recursion (no integration step,
  `yhat = yhat_diff`), and future exog enter raw.

**Trend semantics per mode.** The trend polynomial c_t = Σ τ_j·t^(deg_j)
enters the transition equation as a state intercept, with t = 1..n_eff
1-based **on the model timeline**. Under `simple_differencing := true` the
model timeline is the DIFFERENCED series (statsmodels convention: `trend='c'`
with d = 1 is a constant in the differenced equation — i.e. a drift in y),
under `false` it is the raw series. Degree 0 is named `intercept`, degree 1
`drift`; with d > 0 statsmodels itself warns that trend + differencing can be
redundant — the same caution applies here.

**Concentrated scale (`concentrate := true`).** The filter runs at σ² = 1
and the scale is solved analytically: scale = Σ v²/F over non-missing,
post-burn steps divided by nobs_effective. One fewer optimizer dimension
(often noticeably faster and more robust), identical optimum in exact
arithmetic. Reporting: `meta.sigma2` holds the scale; no `sigma2` row in
`param`/`param_unc`/`bse`; `sarimax_summary` reports one row per actual
parameter; AIC/BIC still count σ² in k (statsmodels behavior); forecast
variances and reported innovation variances F are multiplied by the scale
(the stored state covariance stays at unit scale).

**Missing values.** NULL `y` values are allowed in any mode (not
all-missing; exog must stay complete — the existing exog validation is
unchanged). The filter skips the update at missing steps (innovation
undefined, state prediction continues), matching statsmodels: those steps
contribute nothing to the loglikelihood, and AIC/BIC's nobs_effective is NOT
reduced by the missing count. Diagnostics: `sarimax_residuals` reports
v = NULL, std_resid = NULL at missing t (F still reported);
`sarimax_ljungbox` computes on the non-NULL standardized residuals only
(compacted in time order); `sarimax_evaluate`'s resid_finite_frac counts
non-NULL residuals only. Under `simple_differencing := true` each missing y
poisons every differenced value whose lag window touches it (a NULL fans out
to up to (d+1)·(D+1) w-values), so prefer `simple_differencing := false`
for gappy data — and note that forecasting a simple-differencing fit whose
trailing d + s·D observations include a missing value fails loudly at
forecast time (the integration anchors would be NULL).

## 6. Named failure conditions

All failures raise immediately with a message naming the offender:

- invalid orders (negative; or seasonal orders with s < 2); invalid `trend`
  (the valid values are named);
- an empty series, or a series whose every y is NULL;
- exog coverage: NULLs, gaps, or missing t in-sample; missing future rows at
  forecast time (the missing t range is named);
- forecasting a `simple_differencing := true` model whose integration
  anchors are NULL (missing trailing observations);
- rank check: after differencing, the exog Gram matrix must pass a pivot
  threshold (1e-10 × trace) — a constant column under differencing is the
  canonical rejection, reported by column;
- series too short for the Hannan–Rissanen start-value regressions (the
  minimum length is stated in the error);
- non-finite loglikelihood at the optimum (poisoned fit) — reported rather
  than silently returned.

## 7. Practical notes

- **Standardize regressors** with wildly different scales; it conditions the
  β block of the Hessian (this mirrors the statsmodels recommendation).
- Determinism: results are identical at any thread count (`SET threads`),
  because every accumulation on the likelihood path is an ordered fold.
- Performance: one likelihood evaluation for n = 500, k = 14 runs in well
  under a second; a full airline-benchmark fit is minutes, not hours. State
  dimension k = max(p + s·P, q + s·Q + 1) drives cost — k ≤ 16 is the tested
  envelope.
- The library never creates or drops tables; every macro is a pure query over
  its inputs. Names beginning `_sarimax_` (macros, CTEs, columns) are
  reserved.
- Missing values: NULL y is supported (see section 5); NULL exog is not
  (statsmodels-parity choice).

## 8. Validating against statsmodels yourself

```python
import statsmodels.api as sm
m = sm.tsa.SARIMAX(y, exog=X, order=(p, d, q), seasonal_order=(P, D, Q, s),
                   trend='n', simple_differencing=True,
                   enforce_stationarity=True, enforce_invertibility=True)
m.ssm.tolerance = 0          # exact filter, no gain freezing
res = m.fit(method='lbfgs')
```

`res.params` aligns positionally with `SELECT value FROM m WHERE kind =
'param' ORDER BY idx`; `res.llf` with `meta.loglik`; forecasts compare on the
differenced scale via `yhat_diff`/`se_diff` and on the original scale via
`yhat`/`se` (statsmodels under `simple_differencing=True` reports the
differenced scale — integrate as in tests/generate_fixtures.py to compare
original-scale numbers).

The v2 flags map one-to-one: `trend := 'ct'` ↔ `trend='ct'`,
`concentrate := true` ↔ `concentrate_scale=True`, `simple_differencing :=
false` ↔ `simple_differencing=False` (then `res.llf` includes statsmodels'
`loglikelihood_burn` handling automatically, and statsmodels' forecasts are
already on the original scale — compare directly with `yhat`/`se`). For
missing data pass a `y` with NaNs to statsmodels and NULLs here.
