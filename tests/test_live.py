"""Live cross-check (spec section 11): in addition to the frozen golden
fixtures, recompute references AGAINST STATSMODELS IN-PROCESS on fixed-seed
data and compare the SQL implementation directly. This guards against silent
environment drift (a statsmodels/NumPy upgrade changing reference numbers
would fail here even though the committed fixtures still pass).

All comparisons happen at FIXED parameter vectors -- transform, system
matrices, filter trace, loglikelihood, and forecasts -- never at a fitted
optimum, so no optimizer-tolerance slack is involved (statsmodels' L-BFGS
endpoint is only defined to ~1e-3 gradient slack; see tests/test_estimate.py
for how fitted-parameter comparison is handled).
"""
import os
import sys

import duckdb
import numpy as np
import pytest
from statsmodels.tsa.statespace.sarimax import SARIMAX

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

# statsmodels' own NumPy-2.5 deprecation (endog.shape assignment in mlemodel);
# third-party noise, not a duckARIMA warning
pytestmark = pytest.mark.filterwarnings(
    "ignore::DeprecationWarning:statsmodels")

from generate_fixtures import (  # noqa: E402
    PINNED_OPTIONS, diff_series, integrate_forecast, integration_weights,
    simulate_sarima)

SQL_FILES = ["sql/00_linalg.sql", "sql/01_prep.sql", "sql/02_ssm.sql",
             "sql/03_filter.sql", "sql/04_estimate.sql", "sql/05_forecast.sql",
             "sql/06_harness.sql"]

H = 12
N_PROBES = 5


class LiveCase:
    def __init__(self, name, order, seasonal_order, n, seed, r=0, beta=()):
        self.name, self.order, self.seasonal_order = name, order, seasonal_order
        self.n, self.seed, self.r, self.beta = n, seed, r, np.asarray(beta, float)


CASES = [
    LiveCase("live_arma_2_0_1", (2, 0, 1), (0, 0, 0, 0), n=150, seed=71),
    LiveCase("live_sarimax_111_011_4", (1, 1, 1), (0, 1, 1, 4), n=120, seed=72,
             r=1, beta=(1.2,)),
]


def build_case(case):
    rng = np.random.default_rng(case.seed)
    p, d, q = case.order
    P, D, Q, s = case.seasonal_order
    phi = [0.5][:p] + [-0.2] * max(0, p - 1)
    theta = [0.3][:q]
    Theta = [-0.4][:Q]
    u = simulate_sarima(rng, case.n, case.order, (P, D, Q, max(s, 1)),
                        (phi, theta, [], Theta), 1.0)
    if case.r:
        t_all = np.arange(1, case.n + H + 1, dtype=float)
        X = np.column_stack([np.sin(2 * np.pi * t_all / 30.0) + t_all / 100.0])
        xd = np.column_stack([diff_series(X[:case.n, j], d, D, s)
                              for j in range(case.r)])
        w = diff_series(u, d, D, s) + xd @ case.beta
        # rebuild the original-scale series by integrating w onto u's stage heads
        z = w
        stages = []
        zz = u
        for _ in range(d):
            stages.append(zz)
            zz = np.diff(zz)
        for _ in range(D):
            stages.append(zz)
            zz = zz[s:] - zz[:-s]
        for k in range(len(stages) - 1, -1, -1):
            prev = stages[k]
            lag = s if k >= d else 1
            head = prev[:lag]
            out = np.concatenate([head, np.zeros(len(z))])
            for i in range(len(z)):
                out[lag + i] = z[i] + out[i]
            z = out
        y = z
        assert len(y) == case.n
        return y, X
    return u, None


@pytest.fixture(scope="module")
def con():
    c = duckdb.connect()
    c.execute("SET lambda_syntax = 'DISABLE_SINGLE_ARROW'")
    for f in SQL_FILES:
        with open(os.path.join(ROOT, f), encoding="utf-8") as fh:
            c.execute(fh.read())
    return c


def dl(vals):
    return "[" + ", ".join(repr(float(v)) + "::DOUBLE" for v in vals) + "]::DOUBLE[]"


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_live_against_statsmodels(con, case):
    y, X = build_case(case)
    p, d, q = case.order
    P, D, Q, s = case.seasonal_order
    seff = max(s, 1)
    r = case.r
    n = case.n

    model = SARIMAX(y, exog=X[:n] if r else None, order=case.order,
                    seasonal_order=case.seasonal_order, **PINNED_OPTIONS)
    model.ssm.tolerance = 0.0
    assert model.ssm.loglikelihood_burn == 0
    kp = len(model.param_names)
    n_eff = model.nobs

    # fixed-seed probe vectors through statsmodels' own transform
    rng = np.random.default_rng(case.seed + 500)
    probes_unc = [rng.normal(scale=0.5, size=kp) for _ in range(N_PROBES)]
    for u in probes_unc:
        u[-1] = 1.0 + 0.2 * rng.standard_normal()
        if r:
            u[:r] = case.beta + rng.normal(scale=0.5, size=r)
    probes_con = [model.transform_params(u) for u in probes_unc]

    # ---- transform pair, live --------------------------------------------
    blocks = f"{r}, {p}, {q}, {P}, {Q}"
    for u, c_ref in zip(probes_unc, probes_con):
        got = con.execute(
            f"SELECT _sarimax_transform_params({dl(u)}, {blocks})").fetchone()[0]
        np.testing.assert_allclose(got, c_ref, rtol=0, atol=1e-10)
        rt = con.execute(
            f"SELECT _sarimax_untransform_params({dl(c_ref)}, {blocks})").fetchone()[0]
        np.testing.assert_allclose(rt, u, rtol=1e-10, atol=1e-10)

    # ---- system matrices, live -------------------------------------------
    cpar = probes_con[0]
    model.update(np.asarray(cpar), transformed=True)
    T_ref = model["transition"];  T_ref = T_ref[:, :, 0] if T_ref.ndim == 3 else T_ref
    R_ref = model["selection"].reshape(-1)
    got = con.execute(
        f"SELECT name, i, j, v FROM _sarimax_ssm_rel({dl(cpar)}, {blocks}, {seff})"
    ).df()
    k = model.k_states
    T_got = np.zeros((k, k))
    for _, row in got[got.name == "transition"].iterrows():
        T_got[int(row.i) - 1, int(row.j) - 1] = row.v
    np.testing.assert_allclose(T_got, T_ref, rtol=0, atol=1e-14)
    R_got = (got[got.name == "selection"].sort_values("i")["v"].to_numpy())
    np.testing.assert_allclose(R_got, R_ref, rtol=0, atol=1e-14)

    # ---- filter trace + loglikelihood, live -------------------------------
    w = diff_series(y, d, D, s)
    con.execute("CREATE OR REPLACE TABLE _lv_w (t BIGINT, w DOUBLE)")
    con.executemany("INSERT INTO _lv_w VALUES (?, ?)",
                    [(t + 1, float(w[t])) for t in range(len(w))])
    con.execute("CREATE OR REPLACE TABLE _lv_exd (t BIGINT, j INT, x DOUBLE)")
    if r:
        for j in range(r):
            xall = diff_series(np.asarray(X[:, j]), d, D, s)   # full n+H coverage
            con.executemany("INSERT INTO _lv_exd VALUES (?, ?, ?)",
                            [(t + 1, j + 1, float(xall[t])) for t in range(len(xall))])
    con.execute("CREATE OR REPLACE TABLE _lv_probes (probe_id BIGINT, params DOUBLE[])")
    for pid, c_ref in enumerate(probes_con, start=1):
        con.execute("INSERT INTO _lv_probes VALUES (?, ?)", [pid, list(map(float, c_ref))])
    con.execute("CREATE OR REPLACE TABLE _lv_exd_in AS SELECT * FROM _lv_exd WHERE t <= " + str(n_eff))
    con.execute(f"""CREATE OR REPLACE TABLE _lv_sys AS
        SELECT * FROM _sarimax_systems('_lv_probes', {blocks}, {seff})""")
    con.execute("""CREATE OR REPLACE TABLE _lv_obs AS
        SELECT * FROM _sarimax_obs_adj('_lv_w', '_lv_exd_in', '_lv_probes')""")
    trace = con.execute("""SELECT probe_id, t, v, f, ll_acc
                           FROM _sarimax_kfilter('_lv_obs', '_lv_sys')
                           ORDER BY probe_id, t""").df()
    for pid, c_ref in enumerate(probes_con, start=1):
        fr = model.filter(np.asarray(c_ref), transformed=True)
        v_ref = fr.forecasts_error.ravel()
        f_ref = fr.forecasts_error_cov.ravel()
        g = trace[trace.probe_id == pid]
        np.testing.assert_allclose(g["v"].to_numpy(), v_ref, rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(g["f"].to_numpy(), f_ref, rtol=1e-9, atol=0)
        ll_ref = model.loglike(np.asarray(c_ref), transformed=True)
        ll_got = g["ll_acc"].iloc[-1]
        assert ll_got == pytest.approx(ll_ref, rel=1e-10, abs=1e-8)

    # ---- forecasts at fixed parameters, live ------------------------------
    cpar = probes_con[0]
    fr = model.filter(np.asarray(cpar), transformed=True)
    fc = fr.get_forecast(H, exog=X[n:] if r else None)
    mean_ref = np.asarray(fc.predicted_mean, float)
    se_ref = np.asarray(fc.se_mean, float)

    con.execute("CREATE OR REPLACE TABLE _lv_p1 (probe_id BIGINT, params DOUBLE[])")
    con.execute("INSERT INTO _lv_p1 VALUES (1, ?)", [list(map(float, cpar))])
    con.execute("CREATE OR REPLACE TABLE _lv_series (t BIGINT, y DOUBLE)")
    con.executemany("INSERT INTO _lv_series VALUES (?, ?)",
                    [(t + 1, float(y[t])) for t in range(n)])
    con.execute(f"""CREATE OR REPLACE TABLE _lv_anch AS
        SELECT * FROM _sarimax_diff_anchors('_lv_series', 't', 'y', {d}, {D}, {seff})""")
    run = con.execute(f"""
        SELECT h, mean_diff, se_diff, mean_orig, se_orig
        FROM _sarimax_forecast_run('_lv_w', '_lv_exd', '_lv_p1', '_lv_anch',
                                   {blocks}, {seff}, {d}, {D}, {H})
        ORDER BY h""").df()
    np.testing.assert_allclose(run["mean_diff"].to_numpy(), mean_ref,
                               rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(run["se_diff"].to_numpy(), se_ref, rtol=1e-9)

    # original-scale reference: exact integration (same math as the fixture
    # generator, computed live)
    Tm = model["transition"];  Tm = Tm[:, :, 0] if Tm.ndim == 3 else Tm
    Rm = model["selection"].reshape(k)
    Zm = model["design"].reshape(k)
    RQR = np.outer(Rm, Rm) * float(model["state_cov"].ravel()[0])
    a = fr.filter_results.predicted_state[:, -1].copy()
    Pm = fr.filter_results.predicted_state_cov[:, :, -1].copy()
    Ps = []
    for _ in range(H):
        Ps.append(Pm.copy())
        Pm = Tm @ Pm @ Tm.T + RQR
    Om = np.zeros((H, H))
    for i in range(H):
        g = Zm.copy()
        for j in range(i, H):
            Om[j, i] = Om[i, j] = g @ Ps[i] @ Zm
            g = g @ Tm
    mean_orig_ref = integrate_forecast(mean_ref, y, d, D, s)
    C = integration_weights(n, d, D, s, H)
    se_orig_ref = np.sqrt(np.einsum("hl,lm,hm->h", C, Om, C))
    np.testing.assert_allclose(run["mean_orig"].to_numpy(), mean_orig_ref,
                               rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(run["se_orig"].to_numpy(), se_orig_ref, rtol=1e-9)
