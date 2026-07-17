# duckARIMA cheatsheet

Load once per session: `.read sarimax_macros.sql`

All macros take **table names as strings** (schema-qualified names work).
Seasonal orders are `sp`/`sd`/`sq` (= textbook P/D/Q; DuckDB macro parameters
are case-insensitive so P/p would collide). `s` is the season length.

---

### sarimax_fit(data, y_col, p, d, q, sp := 0, sd := 0, sq := 0, s := 1, exog_cols := [], t_col := NULL, trend := 'n', concentrate := false, simple_differencing := true)

Fit a SARIMAX(p,d,q)(sp,sd,sq)_s by maximum likelihood. Returns the **model
table** — materialize it: `CREATE TABLE m AS SELECT * FROM sarimax_fit(...)`.

- `exog_cols` — list of column names in `data` used as regressors (β after
  the trend terms in the parameter order). A constant column works as an
  intercept only when d = sd = 0 with `simple_differencing := true` (it
  differences to zero otherwise and is rejected by name) — prefer
  `trend := 'c'`.
- `t_col` — column defining time order; NULL = natural row order.
- `trend` — `'n'` (none, default), `'c'` (intercept), `'t'` (linear drift),
  `'ct'` (both). Parameter names follow statsmodels: `intercept` (degree 0),
  `drift` (degree 1), ordered FIRST. Under `simple_differencing := true`
  the trend applies to the **differenced** series (statsmodels convention);
  under `false` it applies to the raw series.
- `concentrate` — concentrate sigma2 out of the likelihood (one fewer
  optimizer dimension). No `sigma2` param row; the scale is in
  `meta.sigma2`.
- `simple_differencing` — `true` (default) differences y and exog up front,
  losing d + s·sd observations (v1 behavior); `false` keeps the full sample
  and filters the raw series with the differencing states in the state
  vector (statsmodels' default formulation; burn-in of d + s·sd steps in
  the loglikelihood).
- Missing `y` (NULL) is allowed — filtered over, not rejected. Exog must be
  complete.
- Model table schema: `(kind, name, idx, value, value_list)` with kinds
  `param`, `param_unc`, `bse`, `spec`, `meta`, `trend`, `anchor`,
  `exog_col`, `state`.

### sarimax_forecast(model, data, y_col, h, newdata := NULL, exog_cols := [], t_col := NULL, level := 0.95)

h-step-ahead forecasts on the **original scale**, no refit.
Returns `(h, yhat, se, lo, hi, yhat_diff, se_diff)`.

- `newdata` — table holding the **original-scale** future exog values
  (same column names), required iff the model has regressors; must cover all
  h rows. Future exog are treated as known constants (statsmodels convention).
- `exog_cols` — re-supply the regressor column names as a literal list
  (DuckDB struct-field access needs constant keys, so names cannot be read
  back out of the model table); validated against the names stored at fit time.

### sarimax_summary(model, data, y_col)

`(idx, name, coefficient, std_error, z_stat, p_value, ci_lo, ci_hi)` —
one row per parameter incl. sigma2; SEs from the numerical Hessian.

### sarimax_evaluate(model, data, y_col, exog_cols := [], t_col := NULL)

One row: `(loglik, aic, bic, sigma2, n_eff, resid_finite_frac, converged)`.
AIC = 2k − 2ℓ, BIC = k·ln(n_eff − burn) − 2ℓ with k counting sigma2 (even
when concentrated) and n_eff the model-timeline length (differenced length
when simple_differencing, n otherwise); sigma2 is the concentrated scale
when the model was fit with `concentrate := true`.

### sarimax_residuals(model, data, y_col, exog_cols := [], t_col := NULL)

`(t, v, f, std_resid)` — one-step-ahead innovations v_t, innovation variances
F_t, and standardized innovations v_t/√F_t at the fitted parameters.
v and std_resid are NULL at missing-y timesteps (F is still reported);
for concentrated models F is reported at the fitted scale.

### sarimax_ljungbox(model, data, y_col, nlags, exog_cols := [], t_col := NULL)

`(lag, stat, pvalue)` for lags 1..nlags on the standardized innovations
(chi-square upper tail, df = lag). Computed on the NON-NULL residuals only
(missing-y steps are dropped and the series compacted).

### sarimax_grid_sql(data, y_col, orders, t_col := NULL)

Returns the **SQL text** that fits every row of `orders(p, d, q, sp, sd, sq, s)`
and ranks by AIC; run the returned string as a second step (DuckDB cannot
correlate table-macro arguments through a lateral join; duckLM's
`dummy_encode_sql` precedent).

---

Notes: diagnostics/forecast macros need `exog_cols` re-supplied for exog
models; at most 32 exogenous columns (constant-key dispatch cap).

Everything prefixed `_sarimax_` is internal. Column and table names beginning
with `_sarimax_` are reserved.
