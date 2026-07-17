-- duckARIMA pure-SQL smoke test (no Python, no driver): run from the project root as
--     duckdb < tests/smoke.sql
-- Fits a small ARIMAX on inline data, forecasts, and asserts sane invariants.
-- Any failed assertion raises via error(); success prints SMOKE OK rows.
--
-- NOTE: committed in deliberately FAILING form before the library exists
-- (spec section 12: both test paths run red from day one).

SET lambda_syntax = 'DISABLE_SINGLE_ARROW';

.read sarimax_macros.sql

-- ---- inline data: AR(1)-ish trending series with two regressors -------------
CREATE OR REPLACE TABLE smoke_data AS
SELECT
    t,
    (10.0::DOUBLE + 0.05::DOUBLE * t
        + 2.0::DOUBLE  * sin(2.0::DOUBLE * pi() * t / 25.0::DOUBLE)
        + 1.5::DOUBLE  * x1
        - 2.0::DOUBLE  * x2
        + 0.8::DOUBLE  * sin(t * 12.9898::DOUBLE) * cos(t * 78.233::DOUBLE) * 3.0::DOUBLE
    ) AS y,
    x1,
    x2
FROM (
    SELECT
        t,
        sin(2.0::DOUBLE * pi() * t / 40.0::DOUBLE) + t / 120.0::DOUBLE AS x1,
        CASE WHEN t BETWEEN 40 AND 45 THEN 1.0::DOUBLE ELSE 0.0::DOUBLE END AS x2
    FROM generate_series(1, 96) AS g(t)
);

-- future exog for the forecast horizon (original scale, t continues past the sample)
CREATE OR REPLACE TABLE smoke_future AS
SELECT
    t,
    sin(2.0::DOUBLE * pi() * t / 40.0::DOUBLE) + t / 120.0::DOUBLE AS x1,
    0.0::DOUBLE AS x2
FROM generate_series(97, 108) AS g(t);

-- ---- fit ---------------------------------------------------------------------
CREATE OR REPLACE TABLE smoke_model AS
SELECT * FROM sarimax_fit('smoke_data', 'y', 1, 1, 1, exog_cols := ['x1', 'x2']);

-- invariant: converged, finite loglik
SELECT CASE
    WHEN (SELECT value FROM smoke_model WHERE kind = 'meta' AND name = 'converged') <> 1.0::DOUBLE
    THEN error('smoke: fit did not converge')
    ELSE 'SMOKE OK: converged' END AS check_1;

SELECT CASE
    WHEN NOT isfinite((SELECT value FROM smoke_model WHERE kind = 'meta' AND name = 'loglik'))
    THEN error('smoke: loglik not finite')
    ELSE 'SMOKE OK: finite loglik' END AS check_2;

-- invariant: sigma2 strictly positive
SELECT CASE
    WHEN (SELECT value FROM smoke_model WHERE kind = 'param' AND name = 'sigma2') <= 0.0::DOUBLE
    THEN error('smoke: sigma2 not positive')
    ELSE 'SMOKE OK: sigma2 > 0' END AS check_3;

-- ---- forecast ------------------------------------------------------------------
CREATE OR REPLACE TABLE smoke_fc AS
SELECT * FROM sarimax_forecast('smoke_model', 'smoke_data', 'y', 12, newdata := 'smoke_future');

SELECT CASE
    WHEN (SELECT count(*) FROM smoke_fc) <> 12
    THEN error('smoke: expected 12 forecast rows')
    ELSE 'SMOKE OK: 12 forecast rows' END AS check_4;

-- invariant: interval ordering lo < mean < hi, finite everywhere, widening se
SELECT CASE
    WHEN (SELECT count(*) FROM smoke_fc
          WHERE NOT (isfinite(yhat) AND isfinite(se) AND lo < yhat AND yhat < hi)) > 0
    THEN error('smoke: broken forecast interval ordering')
    ELSE 'SMOKE OK: interval ordering' END AS check_5;

SELECT CASE
    WHEN (SELECT se FROM smoke_fc WHERE h = 1) > (SELECT se FROM smoke_fc WHERE h = 12)
    THEN error('smoke: forecast se should not shrink with horizon for an I(1) model')
    ELSE 'SMOKE OK: se growth' END AS check_6;

-- ---- summary / evaluate ---------------------------------------------------------
SELECT CASE
    WHEN (SELECT count(*) FROM sarimax_summary('smoke_model', 'smoke_data', 'y')) < 4
    THEN error('smoke: summary should report one row per parameter')
    ELSE 'SMOKE OK: summary rows' END AS check_7;

SELECT CASE
    WHEN NOT isfinite((SELECT aic FROM sarimax_evaluate('smoke_model', 'smoke_data', 'y')))
    THEN error('smoke: aic not finite')
    ELSE 'SMOKE OK: evaluate' END AS check_8;

SELECT 'SMOKE TEST PASSED' AS result;
