"""Invalid state initialization must not become missing data during burn-in."""
from pathlib import Path

import duckdb
import pytest


@pytest.fixture(scope="module")
def con():
    c = duckdb.connect()
    c.execute("SET threads=1")
    c.execute((Path(__file__).resolve().parents[1] / "sarimax_macros.sql").read_text())
    yield c
    c.close()


@pytest.mark.parametrize('d,sd,s', [(0, 0, 1), (1, 0, 1), (0, 3, 2)])
@pytest.mark.parametrize('concentrate', [False, True])
@pytest.mark.parametrize('missing_prefix', [False, True])
@pytest.mark.parametrize('sign', [-1, 1])
def test_invalid_boundary_probe_stays_invalid(
        con, d, sd, s, concentrate, missing_prefix, sign):
    # Finite unconstrained values round to an exact unit AR coefficient.
    params = [sign * 1e8] + ([] if concentrate else [1.0])
    y = [float(i) for i in range(1, 13)]
    if missing_prefix:
        y[:2] = [None, None]
    result = con.execute('SELECT _sarimax_ll_x_v2(?,?,[]::DOUBLE[][],[]::BIGINT[],0,1,0,0,0,?,?,?,0,?)',
                         [params, y, s, d, sd, concentrate]).fetchone()[0]
    assert result['ll'] is None
    assert result['scale2'] is None
    shared = con.execute('''WITH probe AS MATERIALIZED (
        SELECT _sarimax_transform_params_v2(?,0,1,0,0,0,?) AS cp),
        gains AS MATERIALIZED (
        SELECT _sarimax_kf_gains_v2(cp,?,[]::DOUBLE[][],[]::BIGINT[],0,1,0,0,0,?,?,?,0,?) AS g FROM probe)
        SELECT _sarimax_ll_mean_v2(g,?,list_transform(range(1,14),lambda i:0e0)) FROM gains''',
        [params, concentrate, y, s, d, sd, concentrate, y]).fetchone()[0]
    assert shared == result
