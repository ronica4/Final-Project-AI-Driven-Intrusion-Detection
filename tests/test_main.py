"""
Step 1C smoke tests -- main.py end to end against real dohbrw2020 data.
exf2021 is exercised only via config/registry wiring (no branch on args.dataset
in main.py itself); the loader's own real-data behaviour is covered by
tests/test_exf2021_loader.py.
"""

from __future__ import annotations

import subprocess
import sys

import pytest
import yaml

CONFIG_PATH = "config/config.yaml"


def _config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def has_dohbrw2020_data() -> bool:
    from pathlib import Path

    data_dir = Path(_config()["paths"]["dohbrw2020"])
    return (data_dir / "l2-benign.csv").exists()


def _run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "main.py", *args], capture_output=True, text=True
    )


def test_load_dataset_dispatches_without_hardcoding_name(has_dohbrw2020_data):
    if not has_dohbrw2020_data:
        pytest.skip("data/dohbrw2020/ not populated")
    from main import load_dataset

    X, y, meta = load_dataset("dohbrw2020", "hard", _config())
    assert meta["dataset_name"] == "dohbrw2020"
    assert meta["framing"] == "hard"
    assert len(X) == len(y)


@pytest.mark.parametrize(
    "args",
    [
        ["--dataset", "dohbrw2020", "--framing", "hard", "--mode", "eda"],
        ["--dataset", "dohbrw2020", "--framing", "easy", "--mode", "eda"],
        ["--dataset", "dohbrw2020", "--families", "intersection", "--mode", "eda"],
    ],
)
def test_cli_smoke(args, has_dohbrw2020_data):
    if not has_dohbrw2020_data:
        pytest.skip("data/dohbrw2020/ not populated")
    result = _run(args)
    assert result.returncode == 0, result.stderr
    assert "dataset          : dohbrw2020" in result.stdout


def test_invalid_families_rejected_by_argparse(has_dohbrw2020_data):
    if not has_dohbrw2020_data:
        pytest.skip("data/dohbrw2020/ not populated")
    result = _run(["--dataset", "dohbrw2020", "--families", "bogus"])
    assert result.returncode != 0
