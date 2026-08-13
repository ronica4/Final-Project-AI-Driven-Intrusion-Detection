"""
CLI entry point. DRAFT -- argument parsing only per Step 0D; --mode handlers
are wired in Step 1C once both loaders exist.
"""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DNS exfiltration detection pipeline")
    parser.add_argument("--dataset", choices=["exf2021", "dohbrw2020"], required=True)
    parser.add_argument(
        "--mode", choices=["eda", "train", "eval", "xdataset", "cascade"], default="eda"
    )
    parser.add_argument("--framing", choices=["hard", "easy"], default="hard")
    parser.add_argument(
        "--families", choices=["full", "intersection", "F1_only"], default="full"
    )
    parser.add_argument("--config", default="config/config.yaml")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    raise NotImplementedError(
        "main.py wiring is Step 1C (owner: B) -- loaders don't exist yet."
    )
