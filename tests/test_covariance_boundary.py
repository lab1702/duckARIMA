"""Stationary initialization remains valid near the transform's unit-root boundary."""
from pathlib import Path

import duckdb
import numpy as np
import pytest


@pytest.fixture(scope='module')
def con():
    c = duckdb.connect()
    c.execute('SET threads=1')
    c.execute((Path(__file__).resolve().parents[1] / 'sarimax_macros.sql').read_text())
    yield c
    c.close()


@pytest.mark.parametrize('phi', [.99999999, .9999999998, -.9999999998])
def test_near_unit_initial_likelihood(con, phi):
    sigma2, y = 28., [1e6]
    variance = sigma2 / (1 - phi * phi)
    expected = -.5 * (np.log(2 * np.pi) + np.log(variance) + y[0]**2 / variance)
    v1 = con.execute('SELECT _sarimax_ll_c(?,?,[]::DOUBLE[][],0,1,0,0,0,1)', [[phi, sigma2], y]).fetchone()[0]
    args = [[phi, sigma2], y]
    v2 = con.execute('SELECT _sarimax_ll_c_v2(?,?,[]::DOUBLE[][],[]::BIGINT[],0,1,0,0,0,1,0,0,0,false)', args).fetchone()[0]
    np.testing.assert_allclose([v1, v2['ll']], expected, atol=1e-9, rtol=1e-12)
    # Shared covariance and mean passes must follow the same fallback.
    shared = con.execute('''WITH gains AS MATERIALIZED (
        SELECT _sarimax_kf_gains_v2(?,?,[]::DOUBLE[][],[]::BIGINT[],0,1,0,0,0,1,0,0,0,false) AS g)
        SELECT _sarimax_ll_mean_v2(g,?,[0e0,0e0]) FROM gains''', args + [y]).fetchone()[0]
    assert shared == v2


def test_public_near_unit_fit_agrees_with_retrace(con):
    con.execute('CREATE OR REPLACE TABLE boundary_obs AS SELECT t,1e6+sin(t) AS y FROM range(1,41) q(t)')
    con.execute("CREATE OR REPLACE TABLE boundary_model AS SELECT * FROM sarimax_fit('boundary_obs','y',1,0,0,t_col:='t',compute_bse:=false)")
    fitted = con.execute("SELECT value FROM boundary_model WHERE kind='meta' AND name='loglik'").fetchone()[0]
    retraced = con.execute("SELECT sum(-.5*(ln(2*pi())+ln(f)+v*v/f)) FROM sarimax_residuals('boundary_model','boundary_obs','y',t_col:='t')").fetchone()[0]
    assert np.isfinite(fitted)
    assert fitted == pytest.approx(retraced, rel=1e-10, abs=1e-8)


def test_singular_stationary_initialization_is_not_accepted(con):
    result = con.execute('SELECT _sarimax_ll_c_v2([1e0,1e0],[1e0,2e0],[]::DOUBLE[][],[]::BIGINT[],0,1,0,0,0,1,0,0,0,false)').fetchone()[0]
    assert result['ll'] is None
