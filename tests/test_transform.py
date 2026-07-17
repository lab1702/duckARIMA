"""Layer 4 transform acceptance (spec 5.4): the SQL transform pair must match
statsmodels' transform_params/untransform_params on the fixture probe points to
1e-10 (T1-class) and round-trip to 1e-12."""
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


@pytest.fixture(scope="module")
def con():
    c = duckdb.connect()
    c.execute("SET lambda_syntax = 'DISABLE_SINGLE_ARROW'")
    for name in ["00_linalg.sql", "02_ssm.sql", "03_filter.sql", "04_estimate.sql"]:
        with open(os.path.join(ROOT, "sql", name)) as f:
            c.execute(f.read())
    return c


def load(fx, name):
    return pd.read_parquet(os.path.join(FIXDIR, fx, name + ".parquet"))


def dl(vals):
    return "[" + ", ".join(repr(float(v)) + "::DOUBLE" for v in vals) + "]::DOUBLE[]"


@pytest.mark.parametrize("fx", FIXTURES)
def test_transform_matches_statsmodels(con, fx):
    spec = load(fx, "spec").iloc[0]
    probes = load(fx, "probes")
    r, p, q, P, Q = (int(spec[c]) for c in ["r", "p", "q", "bigp", "bigq"])
    blocks = f"{r}, {p}, {q}, {P}, {Q}"

    for pid, g in probes.groupby("probe_id"):
        g = g.sort_values("k")
        unc = g["unconstrained"].to_numpy()
        conp = g["constrained"].to_numpy()

        got_c = con.execute(
            f"SELECT _sarimax_transform_params({dl(unc)}, {blocks})").fetchone()[0]
        np.testing.assert_allclose(got_c, conp, rtol=0, atol=1e-10,
                                   err_msg=f"{fx} probe {pid} transform")

        got_u = con.execute(
            f"SELECT _sarimax_untransform_params({dl(conp)}, {blocks})").fetchone()[0]
        np.testing.assert_allclose(got_u, unc, rtol=1e-10, atol=1e-10,
                                   err_msg=f"{fx} probe {pid} untransform")

        # round-trip identity through our own pair, 1e-12
        rt = con.execute(
            f"SELECT _sarimax_transform_params("
            f"_sarimax_untransform_params({dl(conp)}, {blocks}), {blocks})").fetchone()[0]
        np.testing.assert_allclose(rt, conp, rtol=0, atol=1e-12,
                                   err_msg=f"{fx} probe {pid} roundtrip")
