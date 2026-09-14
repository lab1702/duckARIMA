"""Third seasonal differences must preserve the complete observation design."""
from pathlib import Path

import duckdb
import numpy as np
import pytest
from statsmodels.tsa.statespace.sarimax import SARIMAX

pytestmark = pytest.mark.filterwarnings('ignore::DeprecationWarning:statsmodels')


@pytest.fixture(scope='module')
def con():
    c = duckdb.connect()
    c.execute('SET threads=1')
    c.execute((Path(__file__).resolve().parents[1] / 'sarimax_macros.sql').read_text())
    yield c
    c.close()


@pytest.mark.parametrize('s', [2, 7])
@pytest.mark.parametrize('trend,concentrate,missing', [('n', False, False), ('ct', True, True)])
def test_third_seasonal_filter_and_shared_gradient(con, s, trend, concentrate, missing):
    n = 50
    t = np.arange(1, n + 1)
    y = np.random.default_rng(52).normal(size=n)
    x = np.sin(.7 * t)
    if missing:
        y[[3, 29]] = np.nan
    yl = [None if np.isnan(v) else float(v) for v in y]
    kt = 2 if trend == 'ct' else 0
    degs = [0, 1] if kt else []
    params = ([.03, .001] if kt else []) + [.2, .3, -.2] + ([] if concentrate else [1.])
    ref = SARIMAX(y, exog=x, order=(1, 0, 1), seasonal_order=(0, 3, 0, s),
                  trend=trend, simple_differencing=False, concentrate_scale=concentrate)
    ref.ssm.tolerance = 0
    expected = ref.filter(params)
    args = [params, yl, [[float(v)] for v in x], degs, s, kt, concentrate]
    result = con.execute('SELECT _sarimax_ll_c_v2(?, ?, ?, ?,1,1,1,0,0,?,0,3,?,?)', args).fetchone()[0]
    np.testing.assert_allclose(result['ll'], expected.llf, atol=1e-7, rtol=1e-9)
    np.testing.assert_allclose(result['scale2'], expected.scale, atol=1e-7, rtol=1e-9)
    gains = con.execute('SELECT _sarimax_kf_gains_v2(?, ?, ?, ?,1,1,1,0,0,?,0,3,?,?)', args).fetchone()[0]
    yd = [None if v is None else v - .2 * xx for v, xx in zip(yl, x)]
    ct = [.03 + .001 * tt if kt else 0. for tt in range(1, n + 2)]
    shared = con.execute('SELECT _sarimax_ll_mean_v2(?, ?, ?)', [gains, yd, ct]).fetchone()[0]
    assert shared == result
    con.execute('CREATE OR REPLACE TABLE design_probes AS SELECT 1 AS probe_id, ? AS params', [params])
    con.execute("CREATE OR REPLACE TABLE design_sys AS SELECT * FROM _sarimax_systems_v2('design_probes',1,1,1,0,0,?,0,3,?,?)", [s, kt, concentrate])
    con.execute('CREATE OR REPLACE TABLE design_obs(probe_id INTEGER,t BIGINT,yd DOUBLE,ct DOUBLE)')
    con.executemany('INSERT INTO design_obs VALUES (1,?,?,?)', [(i + 1, v, ct[i]) for i, v in enumerate(yd)])
    trace = con.execute("SELECT v,f FROM _sarimax_kfilter_v2('design_obs','design_sys') ORDER BY t").fetchnumpy()
    np.testing.assert_allclose(np.ma.filled(trace['v'], np.nan), expected.filter_results.forecasts_error[0], atol=1e-7, rtol=1e-7, equal_nan=True)
    np.testing.assert_allclose(trace['f'] * result['scale2'], expected.filter_results.forecasts_error_cov[0, 0], atol=1e-6, rtol=1e-7)
    # Both recursive implementations must share the same likelihood statistics.
    state = con.execute("SELECT cnt,sumlogf,ssq FROM _sarimax_kfilter_state_v2('design_obs','design_sys')").fetchone()
    final = con.execute("SELECT cnt,sumlogf,ssq FROM _sarimax_kfilter_v2('design_obs','design_sys') ORDER BY t DESC LIMIT 1").fetchone()
    assert state == final
    con.execute("CREATE OR REPLACE TABLE design_state AS SELECT * FROM _sarimax_kfilter_state_v2('design_obs','design_sys')")
    con.execute('CREATE OR REPLACE TABLE design_future AS SELECT 1 AS probe_id,h,.2*sin(.7*(?+h)) AS d, CASE WHEN ? THEN .03+.001*(?+h-1) ELSE 0e0 END AS ct FROM range(1,6) q(h)', [n, bool(kt), n])
    forecast = con.execute("SELECT mean_diff,var_diff FROM _sarimax_fc_diff_v2('design_state','design_sys','design_future',5) ORDER BY h").fetchnumpy()
    expected_fc = expected.get_forecast(5, exog=np.sin(.7 * np.arange(n + 1, n + 6)))
    np.testing.assert_allclose(forecast['mean_diff'], expected_fc.predicted_mean, atol=1e-6, rtol=1e-7)
    np.testing.assert_allclose(forecast['var_diff'] * result['scale2'], expected_fc.var_pred_mean, atol=1e-6, rtol=1e-7)


@pytest.mark.parametrize('out_of_core', [False, True])
def test_third_seasonal_public_fit_forecast(con, out_of_core):
    y = np.random.default_rng(52).normal(size=60)
    con.execute('CREATE OR REPLACE TABLE design_data(t BIGINT,y DOUBLE)')
    con.executemany('INSERT INTO design_data VALUES (?,?)', list(enumerate(y, 1)))
    ref = SARIMAX(y, order=(0, 0, 0), seasonal_order=(0, 3, 0, 2),
                  simple_differencing=False, concentrate_scale=True)
    ref.ssm.tolerance = 0
    expected = ref.filter([])
    con.execute("CREATE OR REPLACE TABLE design_model AS SELECT * FROM sarimax_fit('design_data','y',0,0,0,sd:=3,s:=2,simple_differencing:=false,concentrate:=true,compute_bse:=false,t_col:='t',out_of_core:=?)", [out_of_core])
    meta = dict(con.execute("SELECT name,value FROM design_model WHERE kind='meta'").fetchall())
    np.testing.assert_allclose([meta['loglik'], meta['sigma2']], [expected.llf, expected.scale], atol=1e-7, rtol=1e-9)
    forecast = con.execute("SELECT yhat,se FROM sarimax_forecast('design_model','design_data','y',5,t_col:='t') ORDER BY h").fetchnumpy()
    expected_fc = expected.get_forecast(5)
    np.testing.assert_allclose(forecast['yhat'], expected_fc.predicted_mean, atol=1e-7, rtol=1e-7)
    np.testing.assert_allclose(forecast['se'], expected_fc.se_mean, atol=1e-7, rtol=1e-7)
