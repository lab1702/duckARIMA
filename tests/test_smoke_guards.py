"""Fault injection into the assertions used by the standalone SQL smoke path."""
from pathlib import Path
import re

import duckdb
import pytest


SCRIPT = (Path(__file__).with_name("smoke.sql")).read_text()
GUARDS = {
    name: statement
    for statement, name in re.findall(
        r"(SELECT CASE\b.*?END AS (check_\w+);)", SCRIPT, re.DOTALL
    )
}


@pytest.fixture
def con():
    c = duckdb.connect()
    c.execute("""
        CREATE TABLE smoke_model(kind VARCHAR, name VARCHAR, idx INTEGER, value DOUBLE);
        INSERT INTO smoke_model VALUES
            ('meta', 'converged', NULL, 1), ('meta', 'loglik', NULL, -10),
            ('meta', 'sigma2', NULL, 1), ('param', 'intercept', 1, .1),
            ('param', 'drift', 2, .01);
        CREATE TABLE smoke_model_v2 AS SELECT * FROM smoke_model;
        INSERT INTO smoke_model VALUES ('param', 'sigma2', 3, 1);
        CREATE TABLE smoke_fc AS
            SELECT h, 0e0 AS yhat, h::DOUBLE AS se,
                   -h::DOUBLE AS lo, h::DOUBLE AS hi FROM range(1,13) r(h);
        CREATE TABLE smoke_fc_v2 AS SELECT * FROM smoke_fc;
    """)
    yield c
    c.close()


@pytest.mark.parametrize("guard,table,predicate", [
    ("check_1", "smoke_model", "name='converged'"),
    ("check_2", "smoke_model", "name='loglik'"),
    ("check_3", "smoke_model", "kind='param' AND name='sigma2'"),
    ("check_v2_1", "smoke_model_v2", "name='converged'"),
    ("check_v2_2", "smoke_model_v2", "name='loglik'"),
    ("check_v2_3", "smoke_model_v2", "name='sigma2'"),
])
@pytest.mark.parametrize("missing", [False, True])
def test_smoke_rejects_null_or_missing_metadata(con, guard, table, predicate, missing):
    con.execute(GUARDS[guard]).fetchall()
    mutation = f"DELETE FROM {table}" if missing else f"UPDATE {table} SET value=NULL"
    con.execute(f"{mutation} WHERE {predicate}")
    with pytest.raises(duckdb.Error, match="smoke"):
        con.execute(GUARDS[guard]).fetchall()


@pytest.mark.parametrize("table,guard", [
    ("smoke_fc", "check_5"), ("smoke_fc_v2", "check_v2_5"),
])
@pytest.mark.parametrize("column", ["yhat", "se", "lo", "hi"])
@pytest.mark.parametrize("bad_value", [None, float("nan"), float("inf")])
def test_smoke_rejects_nonfinite_forecasts(con, table, guard, column, bad_value):
    con.execute(GUARDS[guard]).fetchall()
    con.execute(f"UPDATE {table} SET {column}=? WHERE h=5", [bad_value])
    with pytest.raises(duckdb.Error, match="smoke"):
        con.execute(GUARDS[guard]).fetchall()


def test_smoke_rejects_missing_endpoint(con):
    con.execute("DELETE FROM smoke_fc WHERE h=12")
    with pytest.raises(duckdb.Error, match="smoke"):
        con.execute(GUARDS["check_6"]).fetchall()


def test_smoke_rejects_missing_trend_parameter(con):
    con.execute("DELETE FROM smoke_model_v2 WHERE kind='param' AND idx=2")
    with pytest.raises(duckdb.Error, match="smoke"):
        con.execute(GUARDS["check_v2_3"]).fetchall()
