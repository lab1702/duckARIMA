"""Known drift means cannot also be supplied as free regression columns."""
from pathlib import Path

import duckdb
import numpy as np
import pytest
from statsmodels.tsa.statespace.sarimax import SARIMAX


@pytest.fixture(scope="module")
def con():
    c = duckdb.connect()
    c.execute("SET threads=1")
    c.execute((Path(__file__).resolve().parents[1] / "sarimax_macros.sql").read_text())
    yield c
    c.close()


@pytest.mark.parametrize("out_of_core", [False, True])
@pytest.mark.parametrize("trend", ["t", "ct"])
@pytest.mark.parametrize("q", [0, 1])
@pytest.mark.parametrize("case", ["stationary", "simple", "integrated", "seasonal3"])
def test_drift_regressor_overlap(con, out_of_core, trend, q, case):
    n = 30
    u = [float(max(t - 1, 1)) for t in range(1, n + 1)]
    d = 1 if case in ("simple", "integrated") else 0
    sd, s = (3, 2) if case == "seasonal3" else (0, 1)
    simple = case == "simple"
    x = []
    for i, v in enumerate(u):
        if case in ("simple", "integrated"):
            v += x[-1] if x else 0
        x.append(v)
    if case == "seasonal3":
        # Seasonal state initialization is not equivalent to zero prehistory.
        # Obtain the deterministic mean from an independent implementation.
        reference = SARIMAX(np.full(n, np.nan), order=(0, 0, 0),
                            seasonal_order=(0, 3, 0, 2), trend="t")
        a1 = np.zeros(reference.k_states)
        a1[-1] = 1.0
        reference.ssm.initialize_known(a1, np.eye(reference.k_states))
        x = reference.filter([1.0, 1.0]).forecasts[0].tolist()
    if simple:
        # Prefix an anchor so differencing produces u at model t=1..n.
        x = [0.0] + x
    con.execute("CREATE OR REPLACE TABLE obs(t BIGINT,y DOUBLE,x DOUBLE)")
    con.executemany("INSERT INTO obs VALUES (?,?,?)",
                    [(i, 3*v + np.sin(1.7*i), v) for i, v in enumerate(x, 1)])
    with pytest.raises(duckdb.Error, match="rank-deficient"):
        con.execute(f"SELECT * FROM sarimax_fit('obs','y',0,{d},{q},sd := {sd},s := {s}, "
                    f"trend := '{trend}',exog_cols := ['x'],t_col := 't',compute_bse := false, "
                    f"simple_differencing := {simple},out_of_core := {out_of_core})").fetchall()
