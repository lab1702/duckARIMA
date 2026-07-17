# duckARIMA test suite

Two independent validation paths, mirroring duckLM (spec section 11).

## 1. The pytest path (statsmodels-validated tiers)

```
python -m venv .venv && .venv/Scripts/pip install -r tests/requirements.txt
.venv/Scripts/python -m pytest tests/ -q
```

| File | Layer under test | Contract |
|---|---|---|
| `test_linalg.py` | `sql/00_linalg.sql` | solvers vs NumPy (bitwise vs same-algorithm control), Kronecker/multiply/Lyapunov, plan-shape (hash join), thread determinism |
| `test_prep.py` | `sql/01_prep.sql` | differencing bit-identical to np.diff, exact undiff round-trips, anchors, integration weights, validation failures |
| `test_ssm.py` | `sql/02_ssm.sql` | system matrices == statsmodels `ssm` to 1e-14, incl. obs_intercept |
| `test_transform.py` | `sql/04_estimate.sql` (transform) | == statsmodels `transform_params`/`untransform_params` to 1e-10; round-trip 1e-12 |
| `test_filter.py` | `sql/03_filter.sql` | **Tier 1**: loglik abs<=1e-8/rel<=1e-10 and per-step (v, F) rel<=1e-9 at 26 probes x 11 fixtures; exog differential check; thread determinism; timing |
| `test_estimate.py` | `sql/04_estimate.sql` | **Tier 2**: fitted params abs<=1e-6 (1e-5 boundary fixtures), loglik rel<=1e-8, bse rel<=1e-3 |
| `test_forecast.py` | `sql/05_forecast.sql` | **Tier 3**: forecasts both scales rel<=1e-6, standard errors rel<=1e-5, h=1..36 |
| `test_assembly.py` | `sarimax_macros.sql` | shipped file == tools/build_macros.py output; loads clean; public macros callable |
| `test_harness.py` | `sql/06_harness.sql` | exog column dispatch up to the 32-column cap, named failure beyond it; **v2 end-to-end**: public fit/forecast/summary/residuals/ljungbox vs `fixtures_v2` (trend, concentrated scale, missing y, simple_differencing=False) under the Tier-2 re-baselining acceptance; v1 public-path regression through the new engine (default flags == v1 behavior) |
| `test_out_of_core.py` | relational fit path | native time-key ordering, scalar-vs-relational likelihood and Hessian parity paths, explicit unique-time/NULL-exog validation, public `out_of_core` fit without Hessian, a 20 MB keyed-filter regression, and profiled external spill where the input-sized `LIST` path fails |
| `test_live.py` | all layers | **live cross-check**: references recomputed against statsmodels in-process on fixed-seed data at fixed parameters (transform, ssm, filter trace, loglik, forecasts both scales) — guards against environment drift independently of the frozen fixtures |

## 2. The pure-SQL smoke path (no Python)

```
duckdb < tests/smoke.sql
```

Fits a small ARIMAX on inline data, forecasts, asserts convergence, finite
loglikelihood, interval ordering — then refits the same data with
`simple_differencing := false, trend := 'ct', concentrate := true` and
asserts the v2 invariants (no sigma2 param row, trend params lead, ordered
intervals). Guards the "no driver required" claim.

## Fixtures

`tests/fixtures/<name>/*.parquet` are golden references generated ONCE by
`tests/generate_fixtures.py` against the pinned environment in
`tests/requirements.txt` (statsmodels 0.14.6, NumPy 2.5.1). Parquet, not CSV,
so doubles round-trip losslessly. Schemas are documented in the generator's
docstring.

`tests/fixtures_v2/<name>/*.parquet` are the v2-feature references
(concentrated scale, trend terms, missing values, simple_differencing=False;
6 fixtures incl. the `kitchen_sink` combination), generated ONCE by
`tests/generate_fixtures_v2.py` under the same pinned environment and
regeneration policy. The v1 fixtures were not touched. Extra schema vs v1:
spec gains (burn, sdiff, conc, ktrend); a `trend.parquet` lists polynomial
degrees; `ssm.parquet` adds state_intercept / a1 / P1 rows; `fitted_meta`
gains (n, n_eff, k_params, burn, nobs_effective).

Pinned reference options: `simple_differencing=True, trend='n',
mle_regression=True, enforce_stationarity=True, enforce_invertibility=True,
concentrate_scale=False, hamilton_representation=False,
measurement_error=False`, and `ssm.tolerance = 0` (Kalman convergence
freezing disabled — statsmodels' default 1e-19 freeze perturbs per-step
innovations by up to ~1e-6 near the invertibility boundary, which is
incompatible with the Tier-1 1e-9 per-step contract; the exact filter is the
reference).

**Regeneration policy (spec sections 10, 12): never silently.** If a fixture
looks wrong: (1) reproduce the discrepancy in a standalone statsmodels
snippet, (2) document the cause here / in the commit message, (3) only then
regenerate deliberately. Regenerations so far:

* 2026-07-16 — `ssm_rows` exported the selection vector transposed
  (generator bug; values unchanged, orientation fixed).
* 2026-07-16 — pinned `ssm.tolerance = 0` (see above; cause reproduced on
  `arma_1_0_1` probe 23: statsmodels froze K/F at t=180 and its v_t then
  drifted 1.6e-6 from the exact recursion).

Both regenerations predate the first tagged release; the fixtures have been
stable since.
