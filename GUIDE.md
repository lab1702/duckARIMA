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
  Lyapunov equation), equivalent to statsmodels' `SARIMAX(...,
  simple_differencing=True, trend='n', mle_regression=True)`.
- There is no trend/constant code path: supply a constant column as a
  regressor if you need an intercept (only meaningful when d = sd = 0).

## 3. Conventions pinned by this implementation

These are the decisions the spec left to "match statsmodels' observable
behavior", recorded here so nothing is implicit in code:

1. **Parameter order** (everywhere: model table `idx`, internal vectors):
   β₁..β_r (in `exog_cols` order), φ₁..φ_p, θ₁..θ_q, Φ₁..Φ_P, Θ₁..Θ_Q, σ².
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
8. **AIC/BIC**: AIC = 2k − 2ℓ, BIC = k·ln(n_eff) − 2ℓ, k counts σ², and
   n_eff is the differenced-series length (the state-space nobs).
9. **Optimizer** (BFGS, Armijo backtracking, central-difference gradients) —
   constants pinned in the spec §5.4. `converged` in `meta` reports the
   gradient criterion; `restarted = 1` flags the one perturbed restart.
10. **Seasonal argument names**: `sp/sd/sq` stand in for P/D/Q because DuckDB
    macro parameters are case-insensitive.
11. **Re-supplying `exog_cols`**: the forecast/diagnostics macros take the
    regressor column names again as a literal list (struct-field access in
    DuckDB requires constant keys, so names stored in the model table cannot
    drive extraction); the list is validated against the stored names. At
    most 12 exogenous columns are supported.
12. **Grid search**: `sarimax_grid_sql(data, y_col, orders)` returns the SQL
    text that fits every row of the orders table and ranks by AIC (DuckDB
    cannot correlate table-macro arguments through a lateral join, so the
    grid runs as a generated statement).

## 4. The model table, column by column

`(kind VARCHAR, name VARCHAR, idx INT, value DOUBLE, value_list DOUBLE[])`

| kind | name | idx | value / value_list |
|---|---|---|---|
| `param` | parameter name (`exog_cols` entries, `ar.L1…`, `ma.L1…`, `ar.S.L<s>…`, `ma.S.L<s>…`, `sigma2`) | canonical position 1..k | constrained estimate |
| `param_unc` | same | same | unconstrained (optimizer-space) estimate |
| `bse` | same | same | standard error |
| `spec` | `p,d,q,sp,sd,sq,s,r,n,n_eff` | 1..10 | the value |
| `meta` | `loglik,aic,bic,converged,iterations,grad_norm,restarted,sigma2` | 1..8 | the value |
| `anchor` | `endog:<stage>` | position within stage | trailing pre-differencing values (Layer-1 anchor contract: stages 1..d ordinary then d+1..d+D seasonal; ordinary stages store 1 value, seasonal stages s values) |
| `anchor` | `exog:<j>` | 1..(d+D·s) | trailing original-scale exog values used to difference future exog |
| `exog_col` | the j-th exog column name | j | — |
| `state` | `a` / `P` | — | value_list: final predicted state mean / row-major vec of its covariance |
| `state` | `k` | — | value: state dimension |

## 5. Named failure conditions

All failures raise immediately with a message naming the offender:

- invalid orders (negative; or seasonal orders with s < 2);
- exog coverage: NULLs, gaps, or missing t in-sample; missing future rows at
  forecast time (the missing t range is named);
- rank check: after differencing, the exog Gram matrix must pass a pivot
  threshold (1e-10 × trace) — a constant column under differencing is the
  canonical rejection, reported by column;
- series too short for the Hannan–Rissanen start-value regressions (the
  minimum length is stated in the error);
- non-finite loglikelihood at the optimum (poisoned fit) — reported rather
  than silently returned.

## 6. Practical notes

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
- Missing values in y or exog are not supported in v1 (statsmodels-parity
  choice; see spec §2).

## 7. Validating against statsmodels yourself

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
