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


@pytest.mark.parametrize("out_of_core", [False, True])
@pytest.mark.parametrize("concentrate", [False, True])
@pytest.mark.parametrize("d,sd,s,basis", [(1,0,1,"1e0"), (2,0,1,"t::DOUBLE"),
                                        (0,1,4,"(t%4)::DOUBLE"), (1,1,4,"t::DOUBLE"),
                                        (0,2,4,"t*(t%4)::DOUBLE")])
def test_missing_targets_cannot_identify_diffuse_regressor(con, out_of_core,
                                                         concentrate,d,sd,s,basis):
    con.execute(f"""
        CREATE OR REPLACE TABLE obs AS
        SELECT t,CASE WHEN t%5=0 THEN NULL ELSE sin(t) END AS y,
               CASE WHEN t%5=0 THEN cos(t) ELSE {basis} END AS x
        FROM range(1,61) q(t)
    """)
    with pytest.raises(duckdb.Error,match="rank-deficient.*integration states"):
        con.execute("""
            SELECT * FROM sarimax_fit('obs','y',0,?,0,sd:=?,s:=?,exog_cols:=['x'],
                t_col:='t',simple_differencing:=false,concentrate:=?,out_of_core:=?)
        """, [d,sd,s,concentrate,out_of_core]).fetchall()


def test_unobserved_seasonal_phases_do_not_reject_identified_design(con):
    # Only one seasonal phase is observed; redundant nuisance columns must
    # be skipped without rejecting an independent observed regressor.
    con.execute("""
        CREATE OR REPLACE TABLE observed_design AS
        SELECT t,1 AS j,cos(t) AS x FROM range(1,61) a(t) WHERE t%4=1
    """)
    assert con.execute("SELECT ok FROM _sarimax_observed_diffuse_rank('observed_design',0,1,4,true)").fetchone()[0]
