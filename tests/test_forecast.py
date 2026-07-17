"""Layer 5 acceptance: Tier-3 forecasts of spec section 3 (sql/05_forecast.sql).

Per fixture (all eleven), against the golden forecast.parquet (h = 1..36):
  * mean_diff  rel <= 1e-6
  * se_diff    rel <= 1e-5
  * mean_orig  rel <= 1e-6
  * se_orig    rel <= 1e-5
Relative-with-floor comparison: abs(a-b) <= tol * max(abs(b), 1e-9); reference
values below 1e-9 in magnitude (differenced means crossing zero) are compared
absolutely at 1e-9.

Plus: Omega self-consistency (sqrt of the Omega diagonal == se_diff at 1e-12
rel), exact agreement of the inline multi-probe mean inversion with Layer 1's
_sarimax_undiff_forecast on the single-probe case, the future-exog coverage
failure (error naming the missing t range), bitwise threads=1 vs default
determinism, and a clean CLI load (zero warnings, zero deprecations).

Table setup mirrors the pinned conventions: w = np.diff applied d times FIRST
then seasonal lag-s D times; exog differenced identically per j over the FULL
t = 1..n+36 coverage (in-sample ++ future concatenated, spec 5.5); the probe is
theta-hat (fitted.parquet 'constrained', ordered by k); anchors come from
Layer 1's _sarimax_diff_anchors on the raw series.
"""
import subprocess
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FIXDIR = HERE / "fixtures"

FIXTURES = sorted(d.name for d in FIXDIR.iterdir() if d.is_dir())

SQL_FILES = [
    "sql/00_linalg.sql",
    "sql/01_prep.sql",
    "sql/02_ssm.sql",
    "sql/03_filter.sql",
    "sql/05_forecast.sql",
]

H = 36

pytestmark = pytest.mark.filterwarnings("error::DeprecationWarning")


# ---------------------------------------------------------------------------
# harness
# ---------------------------------------------------------------------------

def make_con(threads=None):
    c = duckdb.connect()
    c.execute("SET lambda_syntax = 'DISABLE_SINGLE_ARROW'")
    if threads:
        c.execute(f"SET threads = {threads}")
    for f in SQL_FILES:
        c.execute((ROOT / f).read_text(encoding="utf-8"))
    return c


@pytest.fixture(scope="module")
def con():
    c = make_con()
    yield c
    c.close()


def load(fx, name):
    return pd.read_parquet(FIXDIR / fx / f"{name}.parquet")


def diff_series(z, d, D, s):
    """Pinned differencing order: d ordinary diffs FIRST, then D seasonal lag-s."""
    z = np.asarray(z, dtype=float)
    for _ in range(d):
        z = np.diff(z)
    for _ in range(D):
        z = z[s:] - z[:-s]
    return z


def spec_ints(spec):
    return {k: int(spec[k]) for k in
            ["p", "d", "q", "bigp", "bigd", "bigq", "s", "r", "n", "n_eff"]}


def setup_fixture_tables(c, fx, prefix="_fc"):
    """Create <prefix>_{w, series, exd, probes, anchors}; returns the spec ints.

    _exd is the lockstep-differenced FULL exog (t = 1..n_eff+36): the
    concatenated in-sample ++ future original-scale exog differenced per j.
    """
    sp = spec_ints(load(fx, "spec").iloc[0])
    d, D, s, n, n_eff, r = sp["d"], sp["bigd"], sp["s"], sp["n"], sp["n_eff"], sp["r"]
    seff = max(s, 1)

    ser = load(fx, "series").sort_values("t")
    y = ser["y"].to_numpy()
    w = diff_series(y, d, D, seff)
    assert len(w) == n_eff

    c.register("_reg_ser", ser)
    c.execute(f"CREATE OR REPLACE TABLE {prefix}_series AS "
              "SELECT t::BIGINT AS t, y::DOUBLE AS y FROM _reg_ser")
    c.register("_reg_w", pd.DataFrame(
        {"t": np.arange(1, n_eff + 1, dtype=np.int64), "w": w}))
    c.execute(f"CREATE OR REPLACE TABLE {prefix}_w AS "
              "SELECT t::BIGINT AS t, w::DOUBLE AS w FROM _reg_w")

    c.execute(f"CREATE OR REPLACE TABLE {prefix}_exd (t BIGINT, j INT, x DOUBLE)")
    if r:
        exog = load(fx, "exog")
        frames = []
        for j in range(1, r + 1):
            xj = exog[exog.j == j].sort_values("t")["x"].to_numpy()
            assert len(xj) == n + H, f"{fx}: exog j={j} must cover t = 1..n+36"
            xd = diff_series(xj, d, D, seff)
            assert len(xd) == n_eff + H
            frames.append(pd.DataFrame({
                "t": np.arange(1, n_eff + H + 1, dtype=np.int64),
                "j": np.full(n_eff + H, j, dtype=np.int32),
                "x": xd}))
        c.register("_reg_exd", pd.concat(frames, ignore_index=True))
        c.execute(f"INSERT INTO {prefix}_exd SELECT t, j, x FROM _reg_exd")

    params = [float(v) for v in load(fx, "fitted").sort_values("k")["constrained"]]
    c.execute(f"CREATE OR REPLACE TABLE {prefix}_probes (probe_id BIGINT, params DOUBLE[])")
    c.execute(f"INSERT INTO {prefix}_probes VALUES (1, ?)", [params])

    c.execute(f"CREATE OR REPLACE TABLE {prefix}_anchors AS SELECT * FROM "
              f"_sarimax_diff_anchors('{prefix}_series', 't', 'y', {d}, {D}, {seff})")
    return sp


def run_forecast(c, sp, prefix="_fc", hmax=H):
    seff = max(sp["s"], 1)
    return c.execute(
        f"SELECT * FROM _sarimax_forecast_run("
        f"'{prefix}_w', '{prefix}_exd', '{prefix}_probes', '{prefix}_anchors', "
        f"{sp['r']}, {sp['p']}, {sp['q']}, {sp['bigp']}, {sp['bigq']}, {seff}, "
        f"{sp['d']}, {sp['bigd']}, {hmax}) ORDER BY probe_id, h").df()


def build_chain(c, sp, prefix, hmax=H):
    """Materialize the pipeline stages individually (for the component tests)."""
    seff = max(sp["s"], 1)
    c.execute(f"CREATE OR REPLACE TABLE {prefix}_sys AS SELECT * FROM _sarimax_systems("
              f"'{prefix}_probes', {sp['r']}, {sp['p']}, {sp['q']}, "
              f"{sp['bigp']}, {sp['bigq']}, {seff})")
    c.execute(f"CREATE OR REPLACE TABLE {prefix}_obs AS SELECT * FROM "
              f"_sarimax_obs_adj('{prefix}_w', '{prefix}_exd', '{prefix}_probes')")
    c.execute(f"CREATE OR REPLACE TABLE {prefix}_state AS SELECT * FROM "
              f"_sarimax_kfilter_state('{prefix}_obs', '{prefix}_sys')")
    if sp["r"]:
        c.execute(f"CREATE OR REPLACE TABLE {prefix}_dfut AS SELECT * FROM "
                  f"_sarimax_fc_dfut('{prefix}_exd', '{prefix}_probes', {sp['n_eff']}, {hmax})")
    else:
        c.execute(f"CREATE OR REPLACE TABLE {prefix}_dfut (probe_id BIGINT, h BIGINT, d DOUBLE)")
    c.execute(f"CREATE OR REPLACE TABLE {prefix}_fcd AS SELECT * FROM "
              f"_sarimax_fc_diff('{prefix}_state', '{prefix}_sys', '{prefix}_dfut', {hmax})")


def assert_close(got, want, rtol, label):
    """abs(a-b) <= rtol * max(abs(b), 1e-9); |b| < 1e-9 compared absolutely at
    1e-9. Returns the max relative deviation for reporting."""
    got = np.asarray(got, dtype=float)
    want = np.asarray(want, dtype=float)
    assert got.shape == want.shape, f"{label}: shape {got.shape} vs {want.shape}"
    err = np.abs(got - want)
    tiny = np.abs(want) < 1e-9
    ok = np.where(tiny, err <= 1e-9, err <= rtol * np.maximum(np.abs(want), 1e-9))
    rel = np.where(tiny, 0.0, err / np.maximum(np.abs(want), 1e-9))
    assert bool(np.all(ok)), (
        f"{label}: max rel dev {rel.max():.3e} (tol {rtol:.0e}) "
        f"at h = {int(np.argmax(rel)) + 1}: got {got[np.argmax(rel)]!r} "
        f"vs want {want[np.argmax(rel)]!r}")
    return float(rel.max())


# ---------------------------------------------------------------------------
# T3: forecasts on both scales against the golden fixtures
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fx", FIXTURES)
def test_t3_forecast(con, fx):
    sp = setup_fixture_tables(con, fx)
    got = run_forecast(con, sp)
    want = load(fx, "forecast").sort_values("h")

    assert list(got["h"]) == list(range(1, H + 1)), f"{fx}: h must be dense 1..{H}"
    assert list(got["probe_id"].unique()) == [1]

    devs = {
        "mean_diff": assert_close(got["mean_diff"], want["mean_diff"], 1e-6,
                                  f"{fx}: mean_diff"),
        "se_diff": assert_close(got["se_diff"], want["se_diff"], 1e-5,
                                f"{fx}: se_diff"),
        "mean_orig": assert_close(got["mean_orig"], want["mean_orig"], 1e-6,
                                  f"{fx}: mean_orig"),
        "se_orig": assert_close(got["se_orig"], want["se_orig"], 1e-5,
                                f"{fx}: se_orig"),
    }
    print(f"{fx}: max rel devs " +
          " ".join(f"{k}={v:.2e}" for k, v in devs.items()))


# ---------------------------------------------------------------------------
# Omega self-consistency: sqrt(diag Omega) == se_diff (same numbers, two paths)
# ---------------------------------------------------------------------------

def test_omega_diag_matches_se_diff(con):
    fx = "airline"
    sp = setup_fixture_tables(con, fx, prefix="_om")
    build_chain(con, sp, "_om")
    fcd = con.execute(
        "SELECT h, var_diff, omega FROM _om_fcd ORDER BY h").df()
    se_run = run_forecast(con, sp, prefix="_om")["se_diff"].to_numpy()

    diag = []
    for _, row in fcd.iterrows():
        om = np.asarray(row["omega"], dtype=float)
        assert len(om) == int(row["h"]), "omega row h must have length h"
        diag.append(om[-1])
    diag = np.sqrt(np.asarray(diag))
    np.testing.assert_allclose(diag, se_run, rtol=1e-12, atol=0,
                               err_msg="sqrt(diag Omega) vs se_diff")


# ---------------------------------------------------------------------------
# Inline multi-probe mean inversion == Layer 1 _sarimax_undiff_forecast
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fx", ["airline", "arima_2_1_2", "sarima_111_111_12"])
def test_mean_orig_matches_undiff_forecast_exactly(con, fx):
    """_sarimax_fc_orig reimplements Layer 1's inversion with a probe
    partition; on a single probe the two must agree BITWISE."""
    sp = setup_fixture_tables(con, fx, prefix="_ag")
    build_chain(con, sp, "_ag")
    seff = max(sp["s"], 1)
    con.execute("CREATE OR REPLACE TABLE _ag_fcw AS "
                "SELECT h, mean_diff AS w FROM _ag_fcd")
    ref = con.execute(
        f"SELECT yhat FROM _sarimax_undiff_forecast('_ag_fcw', '_ag_anchors', "
        f"{sp['d']}, {sp['bigd']}, {seff}) ORDER BY h").df()["yhat"].to_numpy()
    ours = con.execute(
        f"SELECT mean_orig FROM _sarimax_fc_orig('_ag_fcd', '_ag_anchors', "
        f"{sp['d']}, {sp['bigd']}, {seff}, {H}) ORDER BY h").df()["mean_orig"].to_numpy()
    assert np.array_equal(ref, ours), (
        f"{fx}: inline mean inversion differs from _sarimax_undiff_forecast "
        f"(max abs dev {np.max(np.abs(ref - ours)):.3e})")


# ---------------------------------------------------------------------------
# future-exog coverage failure: error() naming the missing t range
# ---------------------------------------------------------------------------

def test_future_exog_coverage_error(con):
    fx = "arimax_1_1_1"
    sp = setup_fixture_tables(con, fx, prefix="_cv")
    n_eff = sp["n_eff"]
    # drop the last 3 future rows for every regressor
    con.execute(f"DELETE FROM _cv_exd WHERE t > {n_eff + H - 3}")

    with pytest.raises(duckdb.Error) as ei:
        con.execute(f"SELECT * FROM _sarimax_fc_dfut('_cv_exd', '_cv_probes', "
                    f"{n_eff}, {H})").df()
    msg = str(ei.value)
    assert "_sarimax_fc_dfut" in msg
    assert f"{n_eff + H - 2}..{n_eff + H}" in msg, f"missing t range not named: {msg}"
    assert "j = 1" in msg

    # the convenience chain must fail the same way
    with pytest.raises(duckdb.Error, match="future exog coverage is incomplete"):
        run_forecast(con, sp, prefix="_cv")


# ---------------------------------------------------------------------------
# determinism: bitwise-identical output at threads=1 vs default
# ---------------------------------------------------------------------------

def test_determinism_across_threads():
    fx = "sarima_111_111_12"
    outs = []
    for th in (1, None):
        c = make_con(threads=th)
        sp = setup_fixture_tables(c, fx)
        got = run_forecast(c, sp)
        outs.append(got)
        c.close()
    a, b = outs
    for col in ["probe_id", "h", "mean_diff", "se_diff", "mean_orig", "se_orig"]:
        assert np.array_equal(a[col].to_numpy(), b[col].to_numpy()), \
            f"thread-count nondeterminism in {col}"


# ---------------------------------------------------------------------------
# CLI load: zero warnings, zero deprecations
# ---------------------------------------------------------------------------

def test_cli_load_clean():
    cmds = []
    for f in SQL_FILES:
        cmds += ["-c", f".read {f}"]
    proc = subprocess.run(
        ["duckdb"] + cmds,
        cwd=str(ROOT), capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, f"duckdb CLI load failed: {proc.stderr}"
    combined = (proc.stdout + proc.stderr).lower()
    assert "deprecat" not in combined
    assert "warning" not in combined
    assert proc.stderr.strip() == ""
