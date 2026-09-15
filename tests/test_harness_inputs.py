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


@pytest.mark.parametrize("out_of_core", [False, True])
@pytest.mark.parametrize("simple_differencing", [False, True])
def test_fit_rejects_null_training_exog(con, out_of_core, simple_differencing):
    con.execute("CREATE OR REPLACE TABLE bad_exog AS SELECT t, "
                "CASE WHEN t=15 THEN NULL ELSE sin(.4*t) END AS x, "
                "2*sin(.4*t)+cos(1.7*t) AS y FROM range(1,41) q(t)")
    with pytest.raises(duckdb.Error, match="exog.*NULL"):
        con.execute("SELECT * FROM sarimax_fit('bad_exog','y',0,0,0,"
                    "exog_cols := ['x'],t_col := 't',compute_bse := false,"
                    f"out_of_core := {out_of_core},"
                    f"simple_differencing := {simple_differencing})").fetchall()


@pytest.mark.parametrize("out_of_core", [False, True])
@pytest.mark.parametrize("simple_differencing", [False, True])
@pytest.mark.parametrize("concentrate", [False, True])
def test_fit_rejects_no_usable_observations(con, out_of_core,
                                           simple_differencing, concentrate):
    # Differencing poisons every row; without simple differencing the only
    # observed row is inside the diffuse burn-in period.
    y = "CASE WHEN t%2=0 THEN NULL ELSE sin(t) END" if simple_differencing else "CASE WHEN t=1 THEN 1.0 ELSE NULL END"
    con.execute(f"CREATE OR REPLACE TABLE no_obs AS SELECT t,{y} AS y "
                "FROM range(1,21) q(t)")
    with pytest.raises(duckdb.Error, match="no usable observations"):
        con.execute("SELECT * FROM sarimax_fit('no_obs','y',0,1,0,"
                    "t_col := 't',compute_bse := false,"
                    f"out_of_core := {out_of_core},concentrate := {concentrate},"
                    f"simple_differencing := {simple_differencing})").fetchall()


@pytest.mark.parametrize("out_of_core", [False, True])
@pytest.mark.parametrize("simple_differencing", [False, True])
@pytest.mark.parametrize("d,exog", [(1, "['constant']"), (0, "['x','duplicate']")])
def test_fit_rejects_unidentified_exog(con, out_of_core,
                                      simple_differencing, d, exog):
    con.execute("CREATE OR REPLACE TABLE singular_exog AS SELECT t,sin(t) AS y,"
                "1.0 AS constant,sin(.4*t) AS x,2*sin(.4*t) AS duplicate "
                "FROM range(1,41) q(t)")
    with pytest.raises(duckdb.Error, match="identically zero|rank-deficient"):
        con.execute(f"SELECT * FROM sarimax_fit('singular_exog','y',0,{d},0,"
                    f"exog_cols := {exog},t_col := 't',compute_bse := false,"
                    f"out_of_core := {out_of_core},"
                    f"simple_differencing := {simple_differencing})").fetchall()


@pytest.mark.parametrize("out_of_core", [False, True])
@pytest.mark.parametrize("projection", ["*", "count(*)", "spec"])
@pytest.mark.parametrize("value", ["0e0", "'Infinity'::DOUBLE"])
def test_fit_rejects_nonfinite_optimum(con, out_of_core, projection, value):
    con.execute(f"CREATE OR REPLACE TABLE poisoned AS SELECT t,{value} AS y "
                "FROM range(1,31) q(t)")
    columns = "value" if projection == "spec" else projection
    suffix = " WHERE kind='spec'" if projection == "spec" else ""
    with pytest.raises(duckdb.Error, match="non-finite loglikelihood"):
        con.execute(f"SELECT {columns} FROM sarimax_fit('poisoned','y',0,0,0,"
                    "concentrate := true,compute_bse := false,t_col := 't',"
                    f"out_of_core := {out_of_core})" + suffix).fetchall()


@pytest.mark.parametrize("d,sd,s", [(0,0,1),(3,0,1),(4,0,1),(2,1,2),(4,3,2)])
def test_dynamic_differencing_matches_staged_arithmetic(con, d, sd, s):
    con.execute("CREATE OR REPLACE TABLE offset_data AS SELECT t,"
                "1e16+2*(t%7) AS y FROM range(1,41) q(t)")
    expected = con.execute(f"SELECT t,w FROM _sarimax_diff_nt("
                           f"'offset_data','t','y',{d},{sd},{s}) ORDER BY t").fetchall()
    actual = con.execute(f"SELECT t,w FROM _sarimax_diff_dyn("
                         f"'offset_data','t','y',{d},{sd},{s}) ORDER BY t").fetchall()
    assert actual == expected
    con.execute("CREATE OR REPLACE TABLE offset_exog AS "
                "SELECT t,1 AS j,y AS x FROM offset_data UNION ALL "
                "SELECT t,2 AS j,-y AS x FROM offset_data")
    exog = con.execute(f"SELECT t,j,x FROM _sarimax_diff_exog_dyn("
                       f"'offset_exog',{d},{sd},{s}) ORDER BY j,t").fetchall()
    assert exog == [(t,1,w) for t,w in expected] + [(t,2,-w) for t,w in expected]


def test_public_residuals_use_fitted_differencing(con):
    con.execute("CREATE OR REPLACE TABLE offset_obs AS SELECT t,"
                "1e16+2*(t%7) AS y FROM range(1,41) q(t)")
    con.execute("CREATE OR REPLACE TABLE offset_model AS SELECT * FROM "
                "sarimax_fit('offset_obs','y',0,3,0,concentrate := true,"
                "compute_bse := false,t_col := 't')")
    actual = con.execute("SELECT t,v FROM sarimax_residuals("
                         "'offset_model','offset_obs','y',t_col := 't') ORDER BY t").fetchall()
    expected = con.execute("SELECT t,w FROM _sarimax_diff_nt("
                           "'offset_obs','t','y',3,0,1) ORDER BY t").fetchall()
    assert actual == expected


@pytest.mark.parametrize("time_col", [None, "time'key"])
def test_grid_sql_escapes_literal_names(con, time_col):
    con.execute('CREATE OR REPLACE TABLE "sales\'2026" AS SELECT '
                't AS "time\'key",sin(t) AS "units\'sold" FROM range(1,31) q(t)')
    con.execute("CREATE OR REPLACE TABLE quoted_orders AS "
                "SELECT 0 AS p,0 AS d,0 AS q,0 AS sp,0 AS sd,0 AS sq,1 AS s")
    sql = con.execute("SELECT sarimax_grid_sql(?,?, 'quoted_orders',t_col := ?)",
                      ["sales'2026", "units'sold", time_col]).fetchone()[0]
    result = con.execute(sql).fetchall()
    assert len(result) == 1
    assert np.isfinite(result[0][7:10]).all()
    assert result[0][10] == 1


@pytest.mark.parametrize("out_of_core", [False, True])
@pytest.mark.parametrize("simple_differencing", [False, True])
@pytest.mark.parametrize("trend", ['c', 'ct'])
def test_fit_rejects_constant_exog_with_constant_trend(
        con, out_of_core, simple_differencing, trend):
    con.execute("CREATE OR REPLACE TABLE constant_exog AS SELECT t,"
                "1.0 AS x,3+sin(t) AS y FROM range(1,31) q(t)")
    with pytest.raises(duckdb.Error, match="rank-deficient"):
        con.execute("SELECT * FROM sarimax_fit('constant_exog','y',0,0,0,"
                    "exog_cols := ['x'],t_col := 't',compute_bse := false,"
                    f"trend := '{trend}',out_of_core := {out_of_core},"
                    f"simple_differencing := {simple_differencing})").fetchall()


@pytest.mark.parametrize("level", ['-0.5', '0', '1', '1.5', 'NULL',
                                   "'NaN'::DOUBLE", "'Infinity'::DOUBLE"])
def test_forecast_rejects_invalid_confidence_level(con, level):
    con.execute("CREATE OR REPLACE TABLE future AS "
                "SELECT 3.0 AS x FROM range(3)")
    with pytest.raises(duckdb.Error, match="level must be finite and strictly between 0 and 1"):
        con.execute("SELECT * FROM sarimax_forecast('model','obs','y',3,"
                    "newdata := 'future',exog_cols := ['x'],"
                    f"level := {level})").fetchall()
