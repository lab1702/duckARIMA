"""Layer 2 acceptance (spec 5.2 / milestone M1): the SQL state-space builders must
reproduce statsmodels' ssm matrices coefficient-for-coefficient (1e-14) at the
fixture probe parameter vectors, including the obs_intercept vector for exog
fixtures."""
import os

import duckdb
import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIXDIR = os.path.join(HERE, "fixtures")

FIXTURES = sorted(
    d for d in os.listdir(FIXDIR) if os.path.isdir(os.path.join(FIXDIR, d)))

TOL = 1e-14


@pytest.fixture(scope="module")
def con():
    c = duckdb.connect()
    c.execute("SET lambda_syntax = 'DISABLE_SINGLE_ARROW'")
    with open(os.path.join(ROOT, "sql", "02_ssm.sql")) as f:
        c.execute(f.read())
    return c


def load(fx, name):
    return pd.read_parquet(os.path.join(FIXDIR, fx, name + ".parquet"))


def diff_series(z, d, D, s):
    z = np.asarray(z, dtype=float)
    for _ in range(d):
        z = np.diff(z)
    for _ in range(D):
        z = z[s:] - z[:-s]
    return z


def sql_double_list(vals):
    return "[" + ", ".join(repr(float(v)) + "::DOUBLE" for v in vals) + "]::DOUBLE[]"


@pytest.mark.parametrize("fx", FIXTURES)
def test_ssm_matches_fixture(con, fx):
    spec = load(fx, "spec").iloc[0]
    probes = load(fx, "probes")
    ssm = load(fx, "ssm")
    r, p, q, P, Q, s = (int(spec[c]) for c in ["r", "p", "q", "bigp", "bigq", "s"])
    seff = max(s, 1)

    for pid in sorted(ssm["probe_id"].unique()):
        params = (probes[probes.probe_id == pid]
                  .sort_values("k")["constrained"].to_numpy())
        got = con.execute(
            f"SELECT name, i, j, v FROM _sarimax_ssm_rel({sql_double_list(params)},"
            f" {r}, {p}, {q}, {P}, {Q}, {seff})").df()
        want = ssm[(ssm.probe_id == pid) & (ssm.name != "obs_intercept")]

        g = got.set_index(["name", "i", "j"])["v"].sort_index()
        w = want.set_index(["name", "i", "j"])["v"].sort_index()
        assert list(g.index) == list(w.index), f"{fx} probe {pid}: shape mismatch"
        np.testing.assert_allclose(g.to_numpy(), w.to_numpy(), rtol=0, atol=TOL,
                                   err_msg=f"{fx} probe {pid}")


@pytest.mark.parametrize("fx", [f for f in FIXTURES])
def test_obs_intercept_matches_fixture(con, fx):
    spec = load(fx, "spec").iloc[0]
    r = int(spec["r"])
    ssm = load(fx, "ssm")
    want_all = ssm[ssm.name == "obs_intercept"]
    if r == 0:
        assert len(want_all) == 0
        return

    probes = load(fx, "probes")
    exog = load(fx, "exog")
    d, D, s, n = (int(spec[c]) for c in ["d", "bigd", "s", "n"])

    # difference the in-sample exog per column (pinned order: d ordinary then D seasonal)
    xd = {j: diff_series(
            exog[(exog.j == j) & (exog.t <= n)].sort_values("t")["x"].to_numpy(),
            d, D, s)
          for j in range(1, r + 1)}
    rows = [(t + 1, j, xd[j][t]) for j in xd for t in range(len(xd[j]))]
    con.execute("CREATE OR REPLACE TABLE _test_exog_diff (t BIGINT, j INT, x DOUBLE)")
    con.executemany("INSERT INTO _test_exog_diff VALUES (?, ?, ?)", rows)

    for pid in sorted(want_all["probe_id"].unique()):
        params = (probes[probes.probe_id == pid]
                  .sort_values("k")["constrained"].to_numpy())
        beta = params[:r]
        got = con.execute(
            f"SELECT t, d FROM _sarimax_obs_intercept('_test_exog_diff',"
            f" {sql_double_list(beta)}) ORDER BY t").df()
        want = want_all[want_all.probe_id == pid].sort_values("i")
        assert len(got) == len(want)
        np.testing.assert_allclose(got["d"].to_numpy(), want["v"].to_numpy(),
                                   rtol=0, atol=TOL, err_msg=f"{fx} probe {pid}")
