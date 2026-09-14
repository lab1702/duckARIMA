"""The implicit timeline must preserve the input scan, without an all-ties sort."""
from pathlib import Path

import duckdb
import numpy as np
import pytest


@pytest.fixture(scope='module')
def con():
    c = duckdb.connect()
    c.execute((Path(__file__).resolve().parents[1] / 'sarimax_macros.sql').read_text())
    yield c
    c.close()


@pytest.mark.parametrize('threads', [1, 4])
@pytest.mark.parametrize('n', [40, 5000])
@pytest.mark.parametrize('direction', ['ASC', 'DESC'])
def test_natural_target_and_regressor_order(con, threads, n, direction):
    con.execute(f'SET threads={threads}')
    con.execute(f'CREATE OR REPLACE TABLE timeline AS SELECT i AS original_t, '
                f'i::DOUBLE AS y,2*i::DOUBLE AS x1,-i::DOUBLE AS x2 '
                f'FROM range(1,{n+1}) r(i) ORDER BY i {direction}')
    expected = list(range(1, n + 1))
    if direction == 'DESC':
        expected.reverse()
    target = con.execute("SELECT y FROM _sarimax_series_of('timeline','y',NULL) ORDER BY t").fetchnumpy()['y']
    np.testing.assert_array_equal(target, expected)
    exog = con.execute("SELECT j,x FROM _sarimax_exog_of('timeline',['x1','x2'],'y',NULL) ORDER BY j,t").fetchnumpy()['x']
    np.testing.assert_array_equal(exog, np.concatenate([2*np.array(expected), -np.array(expected)]))
    explicit = con.execute("SELECT y FROM _sarimax_series_of('timeline','y','original_t') ORDER BY t").fetchnumpy()['y']
    np.testing.assert_array_equal(explicit, range(1, n + 1))


@pytest.mark.parametrize('simple_differencing', [False, True])
def test_public_fit_and_forecast_preserve_implicit_timeline(con, simple_differencing):
    con.execute('SET threads=1')
    y = np.random.default_rng(11).normal(size=40).cumsum()
    con.execute('CREATE OR REPLACE TABLE walk(t BIGINT,y DOUBLE)')
    con.executemany('INSERT INTO walk VALUES (?,?)', list(enumerate(y, 1)))
    outputs = []
    for time_arg in ['', ",t_col:='t'"]:
        con.execute("CREATE OR REPLACE TABLE walk_model AS SELECT * FROM "
                    "sarimax_fit('walk','y',0,1,0,concentrate:=true,compute_bse:=false,"
                    f"simple_differencing:={simple_differencing}{time_arg})")
        outputs.append((con.execute("SELECT name,value FROM walk_model WHERE kind='meta' ORDER BY name").fetchall(),
                        con.execute("SELECT * FROM sarimax_forecast('walk_model','walk','y',4) ORDER BY h").fetchall()))
    assert outputs[0] == outputs[1]
