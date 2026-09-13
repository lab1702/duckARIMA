"""Sparse transition products preserve the dense kernel's ordered arithmetic."""
from pathlib import Path

import duckdb
import numpy as np
import pytest


@pytest.fixture(scope="module")
def con():
    c = duckdb.connect()
    root = Path(__file__).resolve().parents[1]
    for name in ['00_linalg', '02_ssm', '03_filter', '04_estimate']:
        c.execute((root / 'sql' / (name + '.sql')).read_text())
    yield c
    c.close()


@pytest.mark.parametrize('k', [4, 20, 27])
@pytest.mark.parametrize('invalid', [None, 'null', 'inf', 'nan'])
def test_transition_products_match_dense(con, k, invalid):
    rng = np.random.default_rng(k)
    tm = rng.normal(size=(k,k))
    tm[rng.random((k,k)) < .85] = 0
    tm[0,:] = 0  # an empty support row must still produce zero
    b = rng.normal(size=(k,k))
    for which in (['b'] if invalid is None else ['tm', 'b']):
        vals = {'tm': tm.ravel().tolist(), 'b': b.ravel().tolist()}
        if invalid:
            vals[which][k+2] = {'null': None, 'inf': float('inf'), 'nan': float('nan')}[invalid]
        con.execute('CREATE OR REPLACE TABLE mats AS SELECT ?::DOUBLE[] AS tm, ?::DOUBLE[] AS b',
                    [vals['tm'], vals['b']])
        row = con.execute(f'''WITH z AS MATERIALIZED (
            SELECT *, _sarimax_transition_rows(tm,{k}) AS nz FROM mats)
            SELECT _sarimax_transition_left(tm,nz,b,{k},{k}),
                   _sarimax_mmul(tm,b,{k},{k},{k}),
                   _sarimax_transition_right(tm,nz,b,{k}),
                   _sarimax_mmul(b,_sarimax_mtrans(tm,{k},{k}),{k},{k},{k}),
                   _sarimax_transition_left(tm,nz,list_slice(b,1,{k}),{k},1),
                   _sarimax_mmul(tm,list_slice(b,1,{k}),{k},{k},1)
            FROM z''').fetchone()
        for got,want in zip(row[::2],row[1::2]):
            assert [v is None for v in got] == [v is None for v in want]
            np.testing.assert_array_equal(np.asarray(got,dtype=float),np.asarray(want,dtype=float))


@pytest.mark.parametrize('unconstrained', [False, True])
def test_scalar_dispatch_does_not_execute_relational_branch(con, unconstrained):
    con.execute("CREATE OR REPLACE VIEW forbidden_y AS SELECT 1::BIGINT AS t, error('relational branch executed')::DOUBLE AS y")
    con.execute("CREATE OR REPLACE TABLE empty_x (t BIGINT,j INT,x DOUBLE)")
    con.execute("CREATE OR REPLACE TABLE empty_degrees (idx BIGINT,degree BIGINT)")
    macro = '_sarimax_ll_x_eval_v2' if unconstrained else '_sarimax_ll_c_eval_v2'
    query = f"""SELECT {macro}([1e0],[1e0,2e0,3e0],[]::DOUBLE[][],[]::BIGINT[],
        'forbidden_y','empty_x','empty_degrees',0,0,0,0,0,1,0,0,0,false,{{mode}})"""
    got = con.execute(query.format(mode='false')).fetchone()[0]
    assert np.isfinite(got['ll'])
    assert got['scale2'] == 1
    with pytest.raises(duckdb.Error, match='relational branch executed'):
        con.execute(query.format(mode='true')).fetchall()
