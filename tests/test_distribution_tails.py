"""Distribution tails remain accurate at public API boundary values."""
from pathlib import Path

import duckdb
import numpy as np
import pytest
from scipy.stats import chi2, norm


@pytest.fixture(scope="module")
def con():
    c = duckdb.connect()
    c.execute("SET threads=1")
    c.execute((Path(__file__).resolve().parents[1] / "sarimax_macros.sql").read_text())
    yield c
    c.close()


@pytest.mark.parametrize("df", [1, 2, 10, 100, 1000, 10000, 100000, 1000000])
@pytest.mark.parametrize("z", [-4, -1, 0, 1, 4])
def test_chi_square_transition(con, df, z):
    x = max(0.001, df + z * np.sqrt(2 * df))
    got = con.execute("SELECT _sarimax_chi2_sf(?, ?)", [x, df]).fetchone()[0]
    assert got == pytest.approx(chi2.sf(x, df), rel=2e-9, abs=2e-12)


@pytest.mark.parametrize("level", [0.5, 0.95, 0.999999, np.nextafter(1.0, 0.0)])
def test_forecast_confidence_upper_boundary(con, level):
    con.execute("CREATE OR REPLACE TABLE obs AS SELECT t, sin(t) AS y FROM range(1,31) q(t)")
    con.execute("CREATE OR REPLACE TABLE model AS SELECT * FROM sarimax_fit("
                "'obs','y',0,0,0,concentrate := true,t_col := 't')")
    rows = con.execute("SELECT yhat,se,lo,hi FROM sarimax_forecast("
                       "'model','obs','y',3,t_col := 't',level := ?)", [float(level)]).fetchall()
    quantile = -norm.ppf((1 - level) / 2)
    for mean, se, lo, hi in rows:
        assert np.isfinite([mean, se, lo, hi]).all()
        assert lo == pytest.approx(mean - quantile * se, rel=1e-10, abs=1e-12)
        assert hi == pytest.approx(mean + quantile * se, rel=1e-10, abs=1e-12)
