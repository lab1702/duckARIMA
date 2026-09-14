"""Finite shift covariance agrees with general elimination, including fallbacks."""
from pathlib import Path

import duckdb
import numpy as np
import pytest


@pytest.fixture(scope='module')
def con():
    c = duckdb.connect()
    c.execute((Path(__file__).resolve().parents[1] / 'sql/00_linalg.sql').read_text())
    yield c
    c.close()


@pytest.mark.parametrize('k', [1, 2, 4, 14])
@pytest.mark.parametrize('threads', [1, 4])
def test_shift_matches_dense(con, k, threads):
    con.execute(f'SET threads = {threads}')
    rng = np.random.default_rng(k)
    t = np.eye(k, k, 1)
    # Mixed signs and magnitudes expose changes in accumulation order.
    q = rng.normal(size=(k, k)) * np.exp2(rng.integers(-20, 20, size=(k, k)))
    con.execute('CREATE OR REPLACE TABLE shift_m AS SELECT ?::DOUBLE[] AS t, ?::DOUBLE[] AS q',
                [t.ravel().tolist(), q.ravel().tolist()])
    got, expected = con.execute(f'SELECT _sarimax_lyap(t,q,{k}), '
                               f'_sarimax_lyap_dense(t,q,{k}) FROM shift_m').fetchone()
    assert np.asarray(got).tobytes() == np.asarray(expected).tobytes()
    p = np.asarray(got).reshape(k, k)
    np.testing.assert_allclose(p, t @ p @ t.T + q, rtol=1e-12, atol=1e-9)


@pytest.mark.parametrize('case', ['ar', 'near_shift', 'null_t', 'inf_t', 'nan_t',
                                  'null_q', 'inf_q', 'nan_q', 'overflow', 'negative_zero'])
def test_general_and_exceptional_inputs(con, case):
    t = [0.0, 1.0, 0.0, 0.0]
    q = [2.0, -0.2, -0.2, 1.0]
    if case == 'ar':
        t[0] = 0.8
    elif case == 'near_shift':
        t[1] = 1.0 - 1e-12
    elif case == 'overflow':
        q = [1.7e308] * 4
    elif case == 'negative_zero':
        q = [-0.0] * 4
    else:
        kind, target = case.split('_')
        (t if target == 't' else q)[0] = {'null': None, 'inf': float('inf'),
                                         'nan': float('nan')}[kind]
    con.execute('CREATE OR REPLACE TABLE shift_m AS SELECT ?::DOUBLE[] AS t, ?::DOUBLE[] AS q',
                [t, q])
    got, expected = con.execute('SELECT _sarimax_lyap(t,q,2), '
                               '_sarimax_lyap_dense(t,q,2) FROM shift_m').fetchone()
    assert [v is None for v in got] == [v is None for v in expected]
    np.testing.assert_array_equal(np.asarray(got, dtype=float), np.asarray(expected, dtype=float))
    if case == 'negative_zero':
        assert np.asarray(got).tobytes() == np.asarray(expected).tobytes()
