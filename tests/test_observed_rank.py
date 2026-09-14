"""Regression identification uses the observed model-scale design."""
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


@pytest.mark.parametrize("out_of_core", [False, True])
@pytest.mark.parametrize("simple_differencing", [False, True])
@pytest.mark.parametrize("case", ["zero", "collinear", "constant_trend"])
def test_missing_targets_cannot_identify_regressors(con, out_of_core, simple_differencing, case):
    x = "CASE WHEN t%2=0 THEN 1e0 ELSE 0e0 END" if case == "zero" else "sin(t)"
    if case == "constant_trend":
        x = "CASE WHEN t%2=0 THEN sin(t) ELSE 1e0 END"
    con.execute(f"CREATE OR REPLACE TABLE obs AS SELECT t, "
                f"CASE WHEN t%2=0 THEN NULL ELSE sin(t) END AS y, {x} AS x, "
                "CASE WHEN t%2=0 THEN cos(t) ELSE sin(t) END AS x2 FROM range(1,31) q(t)")
    cols = "['x','x2']" if case == "collinear" else "['x']"
    trend = "c" if case == "constant_trend" else "n"
    with pytest.raises(duckdb.Error, match="identically zero|rank-deficient"):
        con.execute(f"SELECT * FROM sarimax_fit('obs','y',0,0,0,exog_cols := {cols}, "
                    f"trend := '{trend}',t_col := 't',compute_bse := false, "
                    f"simple_differencing := {simple_differencing},out_of_core := {out_of_core})").fetchall()


@pytest.mark.parametrize("out_of_core", [False, True])
def test_differencing_missingness_masks_rank_rows(con, out_of_core):
    con.execute("CREATE OR REPLACE TABLE obs AS SELECT t, "
                "CASE WHEN t%4=0 THEN NULL ELSE sin(t) END AS y, "
                "sum(CASE WHEN t%4 IN (0,1) THEN 1e0 ELSE 0e0 END) OVER (ORDER BY t) AS x "
                "FROM range(1,41) q(t)")
    with pytest.raises(duckdb.Error, match="identically zero|rank-deficient"):
        con.execute("SELECT * FROM sarimax_fit('obs','y',0,1,0,exog_cols := ['x'], "
                    "t_col := 't',compute_bse := false,simple_differencing := true, "
                    f"out_of_core := {out_of_core})").fetchall()


@pytest.mark.parametrize("out_of_core", [False, True])
def test_raw_integrated_missing_data_still_identifies_exog(con, out_of_core):
    # Adjacent differencing would leave no observations, but the raw filter
    # observes every other row and can identify this changing regressor.
    con.execute("CREATE OR REPLACE TABLE obs AS SELECT t, cos(t) AS x, "
                "CASE WHEN t%2=0 THEN NULL ELSE 2*cos(t)+sin(t) END AS y FROM range(1,31) q(t)")
    rows = con.execute("SELECT * FROM sarimax_fit('obs','y',0,1,0,exog_cols := ['x'], "
                       "t_col := 't',compute_bse := false,simple_differencing := false, "
                       f"out_of_core := {out_of_core})").fetchall()
    assert any(row[0] == 'param' and row[1] == 'x' for row in rows)
