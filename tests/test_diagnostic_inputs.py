"""Diagnostics must enforce the same nonmissing-regressor contract as fitting."""
from pathlib import Path

import duckdb
import pytest


@pytest.fixture(scope='module', params=[(0, True), (1, True), (1, False)])
def model(request):
    d, sdiff = request.param
    c = duckdb.connect()
    c.execute('SET threads=1')
    c.execute((Path(__file__).resolve().parents[1] / 'sarimax_macros.sql').read_text())
    c.execute('CREATE TABLE diagnostic_obs AS SELECT t,sin(.4*t) AS x,2*sin(.4*t)+cos(1.7*t) AS y FROM range(1,41) q(t)')
    c.execute("CREATE TABLE diagnostic_model AS SELECT * FROM sarimax_fit('diagnostic_obs','y',0,?,0,exog_cols:=['x'],t_col:='t',simple_differencing:=?,compute_bse:=false)", [d, sdiff])
    yield c
    c.close()


@pytest.mark.parametrize('function', ['sarimax_residuals', 'sarimax_ljungbox', 'sarimax_evaluate'])
@pytest.mark.parametrize('null_t', [1, 10, 40])
def test_diagnostics_reject_missing_regressor(model, function, null_t):
    model.execute('CREATE OR REPLACE TABLE diagnostic_bad AS SELECT t,y,CASE WHEN t=? THEN NULL ELSE x END AS x FROM diagnostic_obs', [null_t])
    lag_arg = ',3' if function == 'sarimax_ljungbox' else ''
    query = f"SELECT * FROM {function}('diagnostic_model','diagnostic_bad','y'{lag_arg},exog_cols:=['x'],t_col:='t')"
    with pytest.raises(duckdb.Error, match='exog contains NULL'):
        model.execute(query).fetchall()
