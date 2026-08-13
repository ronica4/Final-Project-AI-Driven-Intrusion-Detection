"""
Integration test for Step 1A -- runs against the REAL downloaded dataset, not
synthetic data (unlike tests/test_schema.py). Skipped automatically on any
machine that doesn't have data/exf2021/ populated yet (e.g. a fresh clone
before Step 0B's manual download step), so this never blocks `pytest` for
someone who hasn't pulled the data down.
"""

from pathlib import Path

import pytest
import yaml

from schema.unified import B_ONLY_COLUMNS, UNIFIED_COLUMNS, validate_schema

CONFIG_PATH = Path("config/config.yaml")
DATA_DIR = Path("data/exf2021")

pytestmark = pytest.mark.skipif(
    not DATA_DIR.exists() or not any(DATA_DIR.rglob("*.csv")),
    reason="data/exf2021/ not populated -- run Step 0B first",
)


@pytest.fixture(scope="module")
def config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def loaded(config):
    from ingestion.exf2021 import Exf2021Loader

    loader = Exf2021Loader(config)
    return loader.load()


def test_schema_passes(loaded):
    X, y, meta = loaded
    validate_schema(X, mode="full")  # must not raise
    assert list(X.columns) == UNIFIED_COLUMNS


def test_b_only_columns_are_fully_nan(loaded):
    X, _, _ = loaded
    assert X[B_ONLY_COLUMNS].isna().all().all()


def test_light_attack_never_sampled(loaded):
    _, _, meta = loaded
    assert (
        meta["class_counts_sampled"]["light_attack"]
        == meta["class_counts_raw"]["light_attack"]
    )


def test_y_is_binary_int(loaded):
    _, y, _ = loaded
    assert set(y.unique()).issubset({0, 1})


def test_attack_subclass_never_used_as_a_feature(loaded):
    X, _, meta = loaded
    assert "attack_subclass" not in X.columns
    assert meta["attack_subclass"] is not None
    assert set(meta["attack_subclass"].unique()) == {
        "heavy_attack",
        "light_attack",
        "benign",
    }


def test_leakage_mode_is_deliberately_nonconformant(config):
    from ingestion.exf2021 import Exf2021Loader

    loader = Exf2021Loader(config, include_leakage_columns=True)
    X, _, meta = loader.load()
    assert "_leakage_sld" in X.columns
    assert meta["schema_validated"] is False
    assert "sld" not in meta["dropped_columns"]
