"""Strongly significant coefficients retain representable tail probabilities."""
from pathlib import Path

import duckdb
import pytest
from scipy.stats import norm


@pytest.mark.parametrize("z", [-12, -9, -5, 0, 5, 9, 12])
def test_summary_small_pvalue(z):
    with duckdb.connect() as con:
        con.execute("SET threads=1")
        con.execute((Path(__file__).resolve().parents[1] / "sarimax_macros.sql").read_text())
        con.execute("CREATE TABLE model(kind VARCHAR, name VARCHAR, idx INT, value DOUBLE)")
        con.execute("INSERT INTO model VALUES ('param','beta',1,?),('bse','beta',1,1)", [z])
        got = con.execute("SELECT p_value FROM sarimax_summary('model','unused','y')").fetchone()[0]
        assert got == pytest.approx(2 * norm.sf(abs(z)), rel=1e-12, abs=0)
