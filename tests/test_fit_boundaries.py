"""Sample-size validation and exact white-noise fits across target units."""
from pathlib import Path

import duckdb
import numpy as np
import pytest


@pytest.fixture(scope="module")
def con():
    c = duckdb.connect()
    c.execute("SET threads=1")
    c.execute((Path(__file__).resolve().parents[1] / "sarimax_macros.sql").read_text())
    yield c
    c.close()


@pytest.mark.parametrize("out_of_core", [False, True])
@pytest.mark.parametrize("concentrate", [False, True])
@pytest.mark.parametrize("missing", [False, True])
def test_saturated_regression_rejected(con, out_of_core, concentrate, missing):
    con.execute("""
        CREATE OR REPLACE TABLE boundary_data AS
        SELECT t, CASE WHEN t>2 THEN NULL ELSE sin(t) END AS y,
               CASE WHEN t=1 THEN 1e0 ELSE 0e0 END AS x1,
               CASE WHEN t=2 THEN 1e0 ELSE 0e0 END AS x2
        FROM range(1,?) r(t)
    """, [6 if missing else 3])
    with pytest.raises(duckdb.Error, match="too few usable observations.*at least 3"):
        con.execute("""
            SELECT * FROM sarimax_fit('boundary_data','y',0,0,0,
                exog_cols := ['x1','x2'], t_col := 't', compute_bse := false,
                out_of_core := ?, concentrate := ?)
        """, [out_of_core, concentrate]).fetchall()


@pytest.mark.parametrize("out_of_core", [False, True])
@pytest.mark.parametrize("scale", [1., 1e-6, 1e-10])
@pytest.mark.parametrize("missing", [False, True])
def test_white_noise_retains_analytic_optimum(con, out_of_core, scale, missing):
    y = scale * np.sin(np.arange(1., 31.))
    if missing:
        y[[2, 11]] = np.nan
    con.execute("CREATE OR REPLACE TABLE boundary_wn(t BIGINT, y DOUBLE)")
    con.executemany("INSERT INTO boundary_wn VALUES (?,?)",
                    [(i, None if np.isnan(v) else float(v)) for i, v in enumerate(y, 1)])
    expected = np.nanmean(y*y)
    con.execute("""
        CREATE OR REPLACE TABLE boundary_model AS
        SELECT * FROM sarimax_fit('boundary_wn','y',0,0,0,
            t_col := 't', compute_bse := false, out_of_core := ?)
    """, [out_of_core])
    meta = dict(con.execute("SELECT name,value FROM boundary_model WHERE kind='meta'").fetchall())
    np.testing.assert_allclose(meta['sigma2'], expected, rtol=1e-12, atol=0)
    assert meta['converged'] == 1
    assert meta['restarted'] == 0
    assert meta['grad_norm'] == 0
    n = np.count_nonzero(np.isfinite(y))
    np.testing.assert_allclose(meta['loglik'], -.5*n*(np.log(2*np.pi*expected)+1), rtol=1e-12)
    fc = con.execute("""
        SELECT yhat,se FROM sarimax_forecast('boundary_model','boundary_wn','y',3,t_col:='t')
    """).fetchnumpy()
    np.testing.assert_array_equal(fc['yhat'], np.zeros(3))
    np.testing.assert_allclose(fc['se'], np.sqrt(expected), rtol=1e-12, atol=0)


@pytest.mark.parametrize("out_of_core", [False, True])
@pytest.mark.parametrize("scale", [1e-3, 1e-6, 1e-10])
def test_small_variance_has_finite_standard_error(con, out_of_core, scale):
    con.execute("""
        CREATE OR REPLACE TABLE boundary_se AS
        SELECT t, ?*sin(t) AS y FROM range(1,31) r(t)
    """, [scale])
    con.execute("""
        CREATE OR REPLACE TABLE boundary_se_model AS
        SELECT * FROM sarimax_fit('boundary_se','y',0,0,0,
            t_col := 't', out_of_core := ?)
    """, [out_of_core])
    variance = con.execute("SELECT avg(y*y) FROM boundary_se").fetchone()[0]
    se = con.execute("""
        SELECT std_error FROM sarimax_summary('boundary_se_model','boundary_se','y')
    """).fetchone()[0]
    assert se is not None and np.isfinite(se)
    np.testing.assert_allclose(se, variance*np.sqrt(2/30), rtol=1e-3, atol=0)


@pytest.mark.parametrize("out_of_core", [False, True])
def test_restart_preserves_best_visited_likelihood(con, out_of_core):
    y = 1e-10*(2+np.sin(np.arange(1.,31.)))
    con.execute("CREATE OR REPLACE TABLE boundary_restart(t BIGINT,y DOUBLE)")
    con.executemany("INSERT INTO boundary_restart VALUES (?,?)", list(enumerate(y, 1)))
    # The default OLS initializer is already the analytic optimum. The
    # relational initializer uses zero mean and the raw second moment.
    initial_mean = 0 if out_of_core else y.mean()
    initial_variance = np.mean((y-initial_mean)**2)
    initial_ll = -.5*len(y)*(np.log(2*np.pi*initial_variance)+1)
    con.execute("""
        CREATE OR REPLACE TABLE boundary_restart_model AS
        SELECT * FROM sarimax_fit('boundary_restart','y',0,0,0,trend := 'c',
            t_col := 't', compute_bse := false, out_of_core := ?)
    """, [out_of_core])
    meta = dict(con.execute("SELECT name,value FROM boundary_restart_model WHERE kind='meta'").fetchall())
    assert meta['loglik'] >= initial_ll-1e-9
    params = dict(con.execute("SELECT name,value FROM boundary_restart_model WHERE kind='param'").fetchall())
    actual_ll = -.5*np.sum(np.log(2*np.pi*params['sigma2'])
                           +(y-params['intercept'])**2/params['sigma2'])
    np.testing.assert_allclose(meta['loglik'], actual_ll, rtol=1e-12)
    if not out_of_core:
        np.testing.assert_allclose(params['intercept'], initial_mean, rtol=1e-12, atol=0)
        np.testing.assert_allclose(params['sigma2'], initial_variance, rtol=1e-12, atol=0)
        assert meta['converged'] == 0  # restoring a point is not a convergence certificate
