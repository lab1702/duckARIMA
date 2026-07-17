# duckARIMA

Seasonal ARIMA with exogenous regressors (SARIMAX) — estimation, inference,
and forecasting — as **pure DuckDB SQL table macros**. No extensions, no
UDFs, no driver required: everything runs inside a stock DuckDB ≥ 1.5.4
session, from the CLI or any client.

Validated against statsmodels' `SARIMAX` on committed golden fixtures at
honest tolerances: the Kalman-filter **loglikelihood reproduces statsmodels
to near machine precision at fixed parameters** (abs ≤ 1e-8 / rel ≤ 1e-10,
with per-timestep innovations at rel ≤ 1e-9); **fitted parameters and
forecasts agree to optimizer tolerance** (parameters abs ≤ 1e-6, forecasts
rel ≤ 1e-6, forecast standard errors rel ≤ 1e-5, parameter standard errors
rel ≤ 1e-3). The argmax of a likelihood is only defined to optimizer
tolerance — in statsmodels as much as here — so no blanket
"machine precision" claim is made for the fit itself.

## Setup

```sql
.read sarimax_macros.sql
```

That's it — one file of `CREATE OR REPLACE MACRO` statements, loaded once per
session.

## Quick taste

```sql
-- monthly sales with two regressors; SARIMA(0,1,1)(0,1,1)_12 + exog
CREATE TABLE m AS SELECT * FROM sarimax_fit('sales', 'units', 0, 1, 1,
                                            sd := 1, sq := 1, s := 12,
                                            exog_cols := ['promo', 'price']);

SELECT * FROM sarimax_summary('m', 'sales', 'units');          -- coefficients, SEs, z, p, CIs
SELECT * FROM sarimax_forecast('m', 'sales', 'units', 12,
                               newdata := 'future_regressors',
                               exog_cols := ['promo', 'price']); -- h, yhat, se, lo, hi
SELECT * FROM sarimax_evaluate('m', 'sales', 'units');          -- loglik, AIC, BIC, sigma2
```

Note: the seasonal orders are `sp` (seasonal AR), `sd` (seasonal
differencing), `sq` (seasonal MA) — DuckDB macro parameters are
case-insensitive, so the textbook `P/D/Q` names would collide with `p/d/q`.
Positionally everything matches `SARIMAX(order=(p,d,q),
seasonal_order=(sp,sd,sq,s))`.

## Files

| File | What |
|---|---|
| `sarimax_macros.sql` | the entire library, generated from `sql/00..06` by `tools/build_macros.py` |
| `CHEATSHEET.md` | one-page signature reference for every public macro |
| `GUIDE.md` | user's guide: examples, statistical conventions, contracts, edge cases |
| `sql/` | layered development sources (linalg → prep → state space → filter → estimation → forecast → harness) |
| `tests/` | pytest validation path (statsmodels golden fixtures) + `tests/smoke.sql` pure-SQL path |

## Validation

Two independent paths (see `tests/README.md`): `pytest tests/ -q` runs the
tiered statsmodels comparison on 11 committed fixtures (ARMA/ARIMA/SARIMA/
ARIMAX/SARIMAX, including near-boundary cases and the airline benchmark), and
`duckdb < tests/smoke.sql` fits and forecasts with no Python at all.

MIT licensed.
