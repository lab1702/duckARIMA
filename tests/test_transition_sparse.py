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


@pytest.mark.parametrize('threads', [1, 4])
def test_relational_dispatch_matches_dense_for_mixed_systems(con, threads):
    """Both recursive branches must retain every probe and its exact trace/state."""
    previous_threads = con.execute("SELECT current_setting('threads')").fetchone()[0]
    try:
        con.execute(f'SET threads = {threads}')
        con.execute("CREATE OR REPLACE TABLE mixed_probes AS "
                    "SELECT 1 AS probe_id, [0.3,0.2,1.0]::DOUBLE[] AS params")
        con.execute("""
            CREATE OR REPLACE TABLE mixed_sys AS
            SELECT * FROM _sarimax_systems_v2('mixed_probes',0,1,1,0,0,1,0,0,0,false)
            UNION ALL
            SELECT * REPLACE (2 AS probe_id) FROM
                _sarimax_systems_v2('mixed_probes',0,0,1,0,1,12,1,1,0,false)
            UNION ALL
            SELECT * REPLACE (3 AS probe_id) FROM
                _sarimax_systems_v2('mixed_probes',0,0,1,0,1,2,0,0,0,false)
            UNION ALL
            SELECT * REPLACE (4 AS probe_id) FROM
                _sarimax_systems_v2('mixed_probes',0,0,1,0,1,6,0,0,0,false)
            UNION ALL
            SELECT * REPLACE (5 AS probe_id) FROM
                _sarimax_systems_v2('mixed_probes',0,0,1,0,1,12,0,0,0,false)
            UNION ALL
            SELECT * REPLACE (
                6 AS probe_id,
                list_transform(tmat, lambda v, i: CASE WHEN i = 1 THEN 1e-12 ELSE v END) AS tmat,
                list_transform(tmat_t, lambda v, i: CASE WHEN i = 1 THEN 1e-12 ELSE v END) AS tmat_t)
            FROM _sarimax_systems_v2('mixed_probes',0,0,1,0,1,12,0,0,0,false)
        """)
        con.execute("""
            CREATE OR REPLACE TABLE mixed_obs AS
            SELECT probe_id, i AS t,
                   CASE WHEN i % 7 = 0 THEN NULL ELSE sin(i::DOUBLE) END AS yd,
                   0e0 AS ct
            FROM mixed_sys CROSS JOIN range(1, 31) r(i)
        """)
        for name in ['_sarimax_kfilter', '_sarimax_kfilter_state']:
            expected = con.execute(f"SELECT * FROM {name}_impl_v2("
                                   "'mixed_obs','mixed_sys',false) ORDER BY ALL").fetchall()
            actual = con.execute(f"SELECT * FROM {name}_v2("
                                 "'mixed_obs','mixed_sys') ORDER BY ALL").fetchall()
            assert actual == expected
    finally:
        con.execute(f'SET threads = {previous_threads}')


def test_precomputed_intercepts_preserve_time_gaps_and_empty_probes(con):
    """Shifted intercepts use t+1, never the next available row after a gap."""
    con.execute("""
        CREATE OR REPLACE TABLE intercept_sys AS
        SELECT id AS probe_id, 1 AS k, CASE WHEN id = 2 THEN 1 ELSE 0 END AS kdiff,
               1 AS cidx, 0 AS burn, [1e0] AS tmat, [1e0] AS tmat_t,
               [1e0] AS rqr, [0e0] AS a1f, [1e0] AS p1f
        FROM range(1, 4) r(id)
    """)
    con.execute("""
        CREATE OR REPLACE TABLE intercept_obs AS
        SELECT id AS probe_id, t, yd, ct
        FROM range(1, 3) r(id) CROSS JOIN
             (VALUES (1::BIGINT, 2e0, 10e0), (2, NULL, 20e0),
                     (4, 100e0, 99e0)) v(t, yd, ct)
    """)
    states = con.execute("""
        SELECT * FROM _sarimax_kfilter_state_v2('intercept_obs','intercept_sys')
        ORDER BY probe_id
    """).fetchall()
    assert states == [
        (1, 2, [32.0], [2.0], 1, 0.0, 4.0),
        (2, 2, [22.0], [2.0], 1, 0.0, 4.0),
        (3, 0, [0.0], [1.0], 0, 0.0, 0.0),
    ]
    trace = con.execute("""
        SELECT probe_id, t, v, f FROM
            _sarimax_kfilter_v2('intercept_obs','intercept_sys')
        ORDER BY probe_id, t
    """).fetchall()
    assert trace == [(1, 1, 2.0, 1.0), (1, 2, None, 1.0),
                     (2, 1, 2.0, 1.0), (2, 2, None, 1.0)]


@pytest.mark.parametrize('k', [4, 14, 27])
@pytest.mark.parametrize('invalid', [None, 'null', 'inf', 'nan', 'negative_zero'])
def test_shift_products_match_dense(con, k, invalid):
    rng = np.random.default_rng(k)
    tm = np.eye(k, k, 1).ravel().tolist()
    b = rng.normal(size=k*k).tolist()
    if invalid:
        b[k+1] = {'null': None, 'inf': float('inf'), 'nan': float('nan'),
                  'negative_zero': -0.0}[invalid]
    con.execute('CREATE OR REPLACE TABLE shift_products AS '
                'SELECT ?::DOUBLE[] AS tm, ?::DOUBLE[] AS b', [tm, b])
    row = con.execute(f"""
        WITH m AS MATERIALIZED (
            SELECT *, _sarimax_transition_rows(tm,{k}) AS nz FROM shift_products
        )
        SELECT _sarimax_transition_left(tm,nz,b,{k},{k},is_shift := true),
               _sarimax_mmul(tm,b,{k},{k},{k}),
               _sarimax_transition_right(tm,nz,b,{k},is_shift := true),
               _sarimax_mmul(b,_sarimax_mtrans(tm,{k},{k}),{k},{k},{k}),
               _sarimax_transition_left(tm,nz,list_slice(b,1,{k}),{k},1,is_shift := true),
               _sarimax_mmul(tm,list_slice(b,1,{k}),{k},{k},1)
        FROM m
    """).fetchone()
    for got, expected in zip(row[::2], row[1::2]):
        assert [v is None for v in got] == [v is None for v in expected]
        np.testing.assert_array_equal(np.asarray(got, dtype=float),
                                      np.asarray(expected, dtype=float))
        if invalid in [None, 'negative_zero']:
            assert np.asarray(got).tobytes() == np.asarray(expected).tobytes()
