"""Harness-level unit tests: the constant-key exog column dispatch (32-column
cap) and its named failure beyond the cap."""
import os

import duckdb
import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

SQL_FILES = ["sql/00_linalg.sql", "sql/01_prep.sql", "sql/02_ssm.sql",
             "sql/03_filter.sql", "sql/04_estimate.sql", "sql/05_forecast.sql",
             "sql/06_harness.sql"]


@pytest.fixture(scope="module")
def con():
    c = duckdb.connect()
    c.execute("SET lambda_syntax = 'DISABLE_SINGLE_ARROW'")
    for f in SQL_FILES:
        with open(os.path.join(ROOT, f), encoding="utf-8") as fh:
            c.execute(fh.read())
    return c


def make_wide_table(c, ncols, n=20, name="_wide"):
    rng = np.random.default_rng(ncols)
    cols = ", ".join(f"x{j} DOUBLE" for j in range(1, ncols + 1))
    c.execute(f"CREATE OR REPLACE TABLE {name} (t BIGINT, y DOUBLE, {cols})")
    vals = rng.standard_normal((n, ncols + 1))
    ph = ", ".join(["?"] * (ncols + 2))
    c.executemany(f"INSERT INTO {name} VALUES ({ph})",
                  [(i + 1, *map(float, vals[i])) for i in range(n)])
    return [f"x{j}" for j in range(1, ncols + 1)], vals[:, 1:]


@pytest.mark.parametrize("r", [1, 13, 32])
def test_exog_extraction_up_to_32(con, r):
    names, vals = make_wide_table(con, r)
    lst = "[" + ", ".join(f"'{n}'" for n in names) + "]"
    got = con.execute(
        f"SELECT t, j, x FROM _sarimax_exog_of('_wide', {lst}, 'y', 't') "
        f"ORDER BY j, t").df()
    assert len(got) == 20 * r
    for j in range(1, r + 1):
        np.testing.assert_array_equal(
            got[got.j == j]["x"].to_numpy(), vals[:, j - 1],
            err_msg=f"column {j} extracted wrong values")


def test_exog_extraction_errors_beyond_32(con):
    names, _ = make_wide_table(con, 33)
    lst = "[" + ", ".join(f"'{n}'" for n in names) + "]"
    with pytest.raises(duckdb.Error, match="at most 32 exogenous columns"):
        # aggregate over x so the projection is actually evaluated
        # (count(*) would never touch the CASE and the lazy error can't fire)
        con.execute(
            f"SELECT sum(x) FROM _sarimax_exog_of('_wide', {lst}, 'y', 't')")


def test_exog_extraction_empty_list(con):
    make_wide_table(con, 1)
    got = con.execute(
        "SELECT count(*) FROM _sarimax_exog_of('_wide', []::VARCHAR[], 'y', 't')"
    ).fetchone()[0]
    assert got == 0
