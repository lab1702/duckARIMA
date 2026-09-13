"""Public API regressions for missing data and parameter-free models."""
from pathlib import Path

import duckdb
import numpy as np
import pytest


@pytest.fixture(scope="module")
def con():
    c = duckdb.connect()
    c.execute("SET threads=1")
    c.execute((Path(__file__).resolve().parents[1] / "sarimax_macros.sql").read_text())
    c.execute("CREATE TABLE obs AS SELECT t, sin(.4*t) AS x, "
              "2*sin(.4*t)+cos(1.7*t) AS y FROM range(1,41) q(t)")
    c.execute("CREATE TABLE model AS SELECT * FROM sarimax_fit("
              "'obs','y',0,0,0,exog_cols := ['x'],compute_bse := false)")
    yield c
    c.close()


@pytest.mark.parametrize("future", [
    "SELECT 3.0 AS x, 0.0 AS y WHERE false",
    "SELECT 3.0 AS x, 0.0 AS y",
    "SELECT CASE WHEN i=2 THEN NULL ELSE 3.0 END AS x, 0.0 AS y FROM range(3) q(i)",
])
def test_forecast_rejects_missing_future_exog(con, future):
    con.execute("CREATE OR REPLACE TABLE future AS " + future)
    with pytest.raises(duckdb.Error, match="future exog"):
        con.execute("SELECT * FROM sarimax_forecast('model','obs','y',3,"
                    "newdata := 'future',exog_cols := ['x'])").fetchall()


def test_future_table_needs_no_target(con):
    con.execute("CREATE OR REPLACE TABLE future AS SELECT i, 3.0+i AS x FROM range(3) q(i)")
    got = con.execute("SELECT yhat FROM sarimax_forecast('model','obs','y',3,"
                      "newdata := 'future',exog_cols := ['x']) ORDER BY h").fetchnumpy()["yhat"]
    beta = con.execute("SELECT value FROM model WHERE kind='param' AND idx=1").fetchone()[0]
    np.testing.assert_allclose(got, beta * np.array([3.,4.,5.]), atol=1e-10)


@pytest.mark.parametrize("d", [0, 1])
def test_concentrated_parameter_free_forecast(con, d):
    con.execute("CREATE OR REPLACE TABLE white AS SELECT t, sin(t) AS y FROM range(1,31) q(t)")
    con.execute(f"CREATE OR REPLACE TABLE wm AS SELECT * FROM sarimax_fit('white','y',0,{d},0,concentrate := true,t_col := 't')")
    got = con.execute("SELECT yhat,se FROM sarimax_forecast('wm','white','y',3,t_col := 't') ORDER BY h").fetchall()
    sigma2 = con.execute("SELECT value FROM wm WHERE kind='meta' AND name='sigma2'").fetchone()[0]
    np.testing.assert_allclose([r[0] for r in got], np.sin(30) if d else 0, atol=1e-10)
    np.testing.assert_allclose([r[1]**2 for r in got], sigma2 * (np.arange(1,4) if d else np.ones(3)), atol=1e-10)
    assert con.execute("SELECT count(*) FROM sarimax_residuals('wm','white','y',t_col := 't')").fetchone()[0] == 30-d


def test_seasonal_missing_residuals(con):
    con.execute("CREATE OR REPLACE TABLE seasonal AS SELECT t, CASE WHEN t=10 THEN NULL ELSE sin(t) END AS y FROM range(1,31) q(t)")
    con.execute("CREATE OR REPLACE TABLE sm AS SELECT * FROM sarimax_fit('seasonal','y',0,0,0,sd := 1,s := 4,compute_bse := false,t_col := 't')")
    rows = con.execute("SELECT t,v FROM sarimax_residuals('sm','seasonal','y',t_col := 't') ORDER BY t").fetchall()
    assert [t for t,v in rows if v is None] == [6,10]
    for t,v in rows:
        if v is not None:
            assert v == pytest.approx(np.sin(t+4)-np.sin(t), abs=1e-12)


@pytest.mark.parametrize("d,sd,s", [(0,1,4),(1,1,4),(2,1,2),(4,3,2)])
def test_dynamic_differencing_matches_staged_missingness(con, d, sd, s):
    con.execute("CREATE OR REPLACE TABLE gaps AS SELECT t, CASE WHEN t=15 THEN NULL ELSE sin(t) END AS y FROM range(1,41) q(t)")
    outputs = []
    for macro in ["_sarimax_diff_nt", "_sarimax_diff_dyn"]:
        outputs.append(con.execute(f"SELECT t,w FROM {macro}('gaps','t','y',{d},{sd},{s}) ORDER BY t").fetchall())
    assert [t for t,w in outputs[0] if w is None] == [t for t,w in outputs[1] if w is None]
    np.testing.assert_allclose([w for t,w in outputs[0] if w is not None],
                               [w for t,w in outputs[1] if w is not None], atol=1e-12)
