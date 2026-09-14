"""Required future regressors must remain finite after model-scale preparation."""
from pathlib import Path

import duckdb
import numpy as np
import pytest


@pytest.fixture(scope="module", params=[(0, True), (1, True), (1, False)])
def con(request):
    d, simple = request.param
    c = duckdb.connect()
    c.execute("SET threads=1")
    c.execute((Path(__file__).resolve().parents[1] / "sarimax_macros.sql").read_text())
    c.execute("CREATE TABLE obs AS SELECT t,sin(t) AS x,2*sin(t)+cos(1.7*t) AS y "
              "FROM range(1,31) q(t)")
    c.execute(f"CREATE TABLE model AS SELECT * FROM sarimax_fit('obs','y',0,{d},0, "
              f"simple_differencing := {simple},exog_cols := ['x'],t_col := 't',compute_bse := false)")
    yield c
    c.close()


@pytest.mark.parametrize("bad", [float('inf'), float('-inf'), float('nan')])
@pytest.mark.parametrize("position", [0, 2])
def test_forecast_rejects_nonfinite_required_regressor(con, bad, position):
    values = [0.2, 0.4, 0.6]
    values[position] = bad
    con.execute("CREATE OR REPLACE TABLE future(t BIGINT,x DOUBLE)")
    con.executemany("INSERT INTO future VALUES (?,?)", list(enumerate(values, 1)))
    with pytest.raises(duckdb.Error, match="non-finite future exog"):
        con.execute("SELECT * FROM sarimax_forecast('model','obs','y',3, "
                    "newdata := 'future',exog_cols := ['x'],t_col := 't')").fetchall()


def test_unused_future_rows_do_not_invalidate_forecast(con):
    con.execute("CREATE OR REPLACE TABLE future AS SELECT t, "
                "CASE WHEN t=4 THEN 'Infinity'::DOUBLE ELSE t*.2 END AS x FROM range(1,5) q(t)")
    rows = con.execute("SELECT yhat,se,lo,hi FROM sarimax_forecast('model','obs','y',3, "
                       "newdata := 'future',exog_cols := ['x'],t_col := 't')").fetchall()
    assert len(rows) == 3
    assert np.isfinite(rows).all()
