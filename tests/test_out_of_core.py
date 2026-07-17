"""Larger-than-memory / relational-likelihood regression coverage.

The default estimator intentionally uses input-sized LIST values.  These
tests pin the separate out_of_core contract: native time ordering, a
relational likelihood equal to the scalar kernel on a simple model, and a
spill-safe initializer that succeeds where an input-sized LIST cannot.
"""

import json
from pathlib import Path

import duckdb
import pytest


ROOT = Path(__file__).resolve().parents[1]
MACROS = (ROOT / "sarimax_macros.sql").read_text()


def make_con():
    con = duckdb.connect()
    con.execute("SET lambda_syntax = 'DISABLE_SINGLE_ARROW'")
    con.execute(MACROS)
    return con


@pytest.fixture(scope="module")
def con():
    c = make_con()
    yield c
    c.close()


def test_explicit_time_order_preserves_native_bigint(con):
    """Ordering must not round BIGINT keys through DOUBLE (> 2**53)."""
    con.execute("CREATE TABLE _ooc_order(ts BIGINT, y DOUBLE)")
    con.execute(
        "INSERT INTO _ooc_order VALUES (9007199254740993, 2), "
        "(9007199254740992, 1)"
    )
    rows = con.execute(
        "SELECT t, y FROM _sarimax_series_of('_ooc_order', 'y', 'ts') "
        "ORDER BY t"
    ).fetchall()
    assert rows == [(1, 1.0), (2, 2.0)]


def test_relational_likelihood_matches_scalar_kernel(con):
    con.execute(
        "CREATE TABLE _ooc_y AS "
        "SELECT i::BIGINT AS t, sin(i::DOUBLE) AS y FROM range(1, 101) r(i)"
    )
    con.execute("CREATE TABLE _ooc_x(t BIGINT, j BIGINT, x DOUBLE)")
    con.execute("CREATE TABLE _ooc_degs(idx BIGINT, degree BIGINT)")
    scalar, relational = con.execute("""
        WITH packed AS (
            SELECT (SELECT list(y ORDER BY t) FROM _ooc_y) AS ylist,
                   (SELECT list([]::DOUBLE[]) FROM _ooc_y) AS xmat
        )
        SELECT (_sarimax_ll_c_v2(
                    [1.0]::DOUBLE[], ylist, xmat, []::BIGINT[],
                    0, 0, 0, 0, 0, 1, 0, 0, 0, false)).ll,
               (_sarimax_ll_c_ooc_v2(
                    [1.0]::DOUBLE[], '_ooc_y', '_ooc_x', '_ooc_degs',
                    0, 0, 0, 0, 0, 1, 0, 0, 0, false)).ll
        FROM packed
    """).fetchone()
    assert relational == scalar


def test_relational_likelihood_matches_with_exog_trend_diffuse_and_missing(con):
    con.execute("""
        CREATE TABLE _ooc_v2_y AS
        SELECT i::BIGINT AS t,
               CASE WHEN i = 10 THEN NULL ELSE sin(i::DOUBLE) END AS y
        FROM range(1, 31) r(i)
    """)
    con.execute("""
        CREATE TABLE _ooc_v2_x AS
        SELECT i::BIGINT AS t, 1::BIGINT AS j, cos(i::DOUBLE) AS x
        FROM range(1, 31) r(i)
    """)
    con.execute("CREATE TABLE _ooc_v2_degs(idx BIGINT, degree BIGINT)")
    con.execute("INSERT INTO _ooc_v2_degs VALUES (1, 0)")
    scalar, relational = con.execute("""
        WITH packed AS (
            SELECT (SELECT list(y ORDER BY t) FROM _ooc_v2_y) AS ylist,
                   (SELECT list(xrow ORDER BY t)
                    FROM (SELECT t, list(x ORDER BY j) AS xrow
                          FROM _ooc_v2_x GROUP BY t)) AS xmat
        )
        SELECT (_sarimax_ll_c_v2(
                    [0.1, 0.2, 0.4, 0.8]::DOUBLE[],
                    ylist, xmat, [0]::BIGINT[],
                    1, 1, 0, 0, 0, 1, 1, 0, 1, false)).ll,
               (_sarimax_ll_c_ooc_v2(
                    [0.1, 0.2, 0.4, 0.8]::DOUBLE[],
                    '_ooc_v2_y', '_ooc_v2_x', '_ooc_v2_degs',
                    1, 1, 0, 0, 0, 1, 1, 0, 1, false)).ll
        FROM packed
    """).fetchone()
    assert relational == scalar


def test_relational_likelihood_runs_under_constrained_memory(tmp_path):
    """Exercise the keyed filter itself under the low-memory contract."""
    c = make_con()
    spill = tmp_path / "likelihood-spill"
    spill.mkdir()
    c.execute("SET memory_limit = '20MB'")
    c.execute("SET threads = 1")
    c.execute("SET temp_directory = ?", [spill.as_posix()])
    c.execute("""
        CREATE VIEW _ooc_ll_y AS
        SELECT i::BIGINT AS t, sin(i::DOUBLE) AS y
        FROM range(1, 2001) r(i)
    """)
    c.execute("CREATE TABLE _ooc_ll_x(t BIGINT, j BIGINT, x DOUBLE)")
    c.execute("CREATE TABLE _ooc_ll_degs(idx BIGINT, degree BIGINT)")
    ll = c.execute("""
        SELECT (_sarimax_ll_c_ooc_v2(
            [1.0]::DOUBLE[], '_ooc_ll_y', '_ooc_ll_x', '_ooc_ll_degs',
            0, 0, 0, 0, 0, 1, 0, 0, 0, false)).ll
    """).fetchone()[0]
    assert ll is not None
    c.close()


def test_out_of_core_bse_uses_relational_likelihood(con):
    con.execute("""
        CREATE TABLE _ooc_bse_y AS
        SELECT i::BIGINT AS t, sin(i::DOUBLE) AS y
        FROM range(1, 13) r(i)
    """)
    con.execute("CREATE TABLE _ooc_bse_x(t BIGINT, j BIGINT, x DOUBLE)")
    con.execute("CREATE TABLE _ooc_bse_degs(idx BIGINT, degree BIGINT)")
    bse = con.execute("""
        SELECT bse FROM _sarimax_bse_ooc_v2(
            [1.0]::DOUBLE[], '_ooc_bse_y', '_ooc_bse_x', '_ooc_bse_degs',
            0, 0, 0, 0, 0, 1, 0, 0, 0, false)
    """).fetchone()[0]
    assert len(bse) == 1
    assert bse[0] is not None and bse[0] > 0


def test_ooc_initializer_avoids_nonspillable_input_list(tmp_path):
    """A data-cardinality scalar aggregate works below the LIST footprint."""
    c = make_con()
    spill = tmp_path / "duckdb-spill"
    spill.mkdir()
    c.execute("SET memory_limit = '20MB'")
    c.execute("SET threads = 1")
    c.execute("SET temp_directory = ?", [spill.as_posix()])
    c.execute("""
        CREATE VIEW _ooc_big_y AS
        SELECT i::BIGINT + 1 AS t, sin(i::DOUBLE) AS y
        FROM range(5000000) r(i)
    """)

    x0, params0 = c.execute("""
        SELECT * FROM _sarimax_start_params_ooc_v2(
            '_ooc_big_y', 0, 0, 0, 0, 0, 1, 0, 0, 0, false)
    """).fetchone()
    assert len(x0) == len(params0) == 1
    assert x0[0] > 0 and params0[0] > 0

    with pytest.raises(duckdb.OutOfMemoryException):
        c.execute("SELECT len(list(y ORDER BY t)) FROM _ooc_big_y").fetchone()
    c.close()


def test_ordering_stage_uses_duckdb_external_spill(tmp_path):
    db_path = tmp_path / "spill-profile.duckdb"
    spill = tmp_path / "spill-profile.tmp"
    spill.mkdir()
    c = duckdb.connect(str(db_path))
    c.execute("SET lambda_syntax = 'DISABLE_SINGLE_ARROW'")
    c.execute(MACROS)
    # Persist more sortable state than the later query memory budget.  The
    # coprime permutation prevents the optimizer from treating ts as ordered.
    c.execute("""
        CREATE TABLE _ooc_unsorted AS
        SELECT ((i * 1000003) % 2000000)::BIGINT AS ts,
               sin(i::DOUBLE) AS y
        FROM range(2000000) r(i)
    """)
    c.execute("SET memory_limit = '20MB'")
    c.execute("SET threads = 1")
    c.execute("SET temp_directory = ?", [spill.as_posix()])
    c.execute("SET max_temp_directory_size = '1GB'")
    profile_json = c.execute("""
        EXPLAIN (ANALYZE, FORMAT JSON)
        SELECT sum(t::DOUBLE * y), count(*) FILTER (WHERE invalid_time)
        FROM _sarimax_series_checked_of('_ooc_unsorted', 'y', 'ts')
    """).fetchone()[1]
    profile = json.loads(profile_json)
    assert profile["system_peak_temp_dir_size"] > 0
    c.close()


def test_out_of_core_requires_unique_explicit_time(con):
    con.execute("CREATE TABLE _ooc_bad_t(ts BIGINT, y DOUBLE)")
    con.execute("INSERT INTO _ooc_bad_t VALUES (1, 1), (1, 2)")

    with pytest.raises(duckdb.Error, match="explicit unique t_col"):
        con.execute("""
            SELECT * FROM sarimax_fit(
                '_ooc_bad_t', 'y', 0, 0, 0,
                out_of_core := true, compute_bse := false)
        """).fetchall()

    with pytest.raises(duckdb.Error, match="t_col to be unique"):
        con.execute("""
            SELECT * FROM sarimax_fit(
                '_ooc_bad_t', 'y', 0, 0, 0, t_col := 'ts',
                out_of_core := true, compute_bse := false)
        """).fetchall()


def test_out_of_core_rejects_null_exog_before_optimization(con):
    con.execute("CREATE TABLE _ooc_bad_x(ts BIGINT, y DOUBLE, x DOUBLE)")
    con.execute("INSERT INTO _ooc_bad_x VALUES (1, 1, 2), (2, 2, NULL)")

    with pytest.raises(duckdb.Error, match="exog contains NULL"):
        con.execute("""
            SELECT * FROM sarimax_fit(
                '_ooc_bad_x', 'y', 0, 0, 0,
                exog_cols := ['x'], t_col := 'ts',
                out_of_core := true, compute_bse := false)
        """).fetchall()

def test_public_out_of_core_fit_can_skip_bse(con):
    con.execute("""
        CREATE TABLE _ooc_public AS
        SELECT i::BIGINT AS ts, sin(i::DOUBLE) AS y
        FROM range(1, 31) r(i)
    """)
    con.execute("""
        CREATE TABLE _ooc_model AS
        SELECT * FROM sarimax_fit(
            '_ooc_public', 'y', 0, 0, 0, t_col := 'ts',
            out_of_core := true, compute_bse := false)
    """)
    sigma2 = con.execute("""
        SELECT value FROM _ooc_model
        WHERE kind = 'param' AND name = 'sigma2'
    """).fetchone()[0]
    bse = con.execute("""
        SELECT value FROM _ooc_model
        WHERE kind = 'bse' AND name = 'sigma2'
    """).fetchone()[0]
    converged = con.execute("""
        SELECT value FROM _ooc_model
        WHERE kind = 'meta' AND name = 'converged'
    """).fetchone()[0]
    assert sigma2 > 0
    assert bse is None
    assert converged == 1.0
