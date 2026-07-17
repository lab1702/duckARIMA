"""Assembly checks (spec section 11): the shipped sarimax_macros.sql is exactly
what tools/build_macros.py produces from sql/00..06, it loads clean in a fresh
session with zero warnings, and the public macros exist."""
import os
import subprocess
import sys
import warnings

import duckdb
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

sys.path.insert(0, os.path.join(ROOT, "tools"))
import build_macros  # noqa: E402

PUBLIC = ["sarimax_fit", "sarimax_forecast", "sarimax_summary",
          "sarimax_evaluate", "sarimax_residuals", "sarimax_ljungbox"]


def test_shipped_file_matches_build():
    shipped = os.path.join(ROOT, "sarimax_macros.sql")
    assert os.path.exists(shipped), "sarimax_macros.sql not built/committed"
    with open(shipped, encoding="utf-8") as f:
        on_disk = f.read()
    assert on_disk == build_macros.build(), \
        "sarimax_macros.sql is stale: rerun python tools/build_macros.py"


def test_loads_clean_and_public_macros_exist():
    con = duckdb.connect()
    con.execute("SET lambda_syntax = 'DISABLE_SINGLE_ARROW'")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with open(os.path.join(ROOT, "sarimax_macros.sql"), encoding="utf-8") as f:
            con.execute(f.read())
    have = {r[0] for r in con.execute(
        "SELECT function_name FROM duckdb_functions() WHERE database_name = 'memory'"
    ).fetchall()}
    for name in PUBLIC:
        assert name in have, f"public macro missing: {name}"


def test_cli_smoke_sql_passes():
    """The pure-SQL path: duckdb < tests/smoke.sql from the project root."""
    with open(os.path.join(HERE, "smoke.sql"), encoding="utf-8") as f:
        script = f.read()
    r = subprocess.run(["duckdb"], input=script.encode("utf-8"),
                       capture_output=True, cwd=ROOT, timeout=1200)
    out = (r.stdout or b"").decode("utf-8", errors="replace") \
        + (r.stderr or b"").decode("utf-8", errors="replace")
    assert r.returncode == 0, f"smoke.sql failed:\n{out[-3000:]}"
    assert "SMOKE TEST PASSED" in out, f"smoke.sql did not pass:\n{out[-3000:]}"
    assert "Deprecated" not in out and "deprecated" not in out, \
        f"deprecation warning in smoke run:\n{out[-3000:]}"
