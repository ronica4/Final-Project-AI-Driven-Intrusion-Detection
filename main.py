"""
CLI entry point. Step 1C (owner: B) -- wires the registry + schema contract
together. This is also where Chapter 7's "proof of abstraction" comes from:
past the load()/project()/validate_schema() calls below, nothing in this
file -- or in any module it calls into for --mode eda -- ever names a
dataset. Swap --dataset exf2021 for --dataset dohbrw2020 and the rest of the
pipeline runs unchanged.
"""

from __future__ import annotations

import argparse
import inspect

import yaml

from ingestion import registry
from schema.unified import FAMILY_MAP, project, validate_schema


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


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_dataset(dataset: str, framing: str, config: dict):
    """Construct the right loader and hand back (X, y, meta), already
    validated. This is the one function anywhere in the codebase that is
    allowed to know a dataset name -- everything downstream takes X/y/meta.

    Loaders differ in whether they accept a `framing` kwarg (only
    dohbrw2020 has more than one framing); introspecting the constructor
    keeps this function from hardcoding which dataset name that applies to.
    """
    loader_cls = registry.get(dataset)
    kwargs = {}
    if "framing" in inspect.signature(loader_cls.__init__).parameters:
        kwargs["framing"] = framing
    loader = loader_cls(config, **kwargs)
    return loader.load()


def print_banner(dataset: str, families: str, X, y, meta: dict) -> None:
    print("=" * 70)
    print(f"dataset          : {meta['dataset_name']}")
    print(f"framing          : {meta['framing']}")
    print(f"families         : {families}")
    print(f"n_rows_raw       : {meta['n_rows_raw']}")
    print(f"n_rows_sampled   : {meta['n_rows_after_sampling']}")
    print(f"positive_rate    : {y.mean():.4f}")
    print(f"class_counts_raw : {meta['class_counts_raw']}")
    print(f"dropped_columns  : {list(meta['dropped_columns'])}")
    for family, cols in FAMILY_MAP.items():
        present = [c for c in cols if c in X.columns]
        if present:
            nan_count = int(X[present].isna().sum().sum())
            print(f"  {family:<28} cols={present} nan={nan_count}")
    print("=" * 70)


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)

    X, y, meta = load_dataset(args.dataset, args.framing, config)

    X_selected = project(X, mode=args.families)
    validate_schema(X_selected, mode=args.families)

    print_banner(args.dataset, args.families, X_selected, y, meta)

    if args.mode == "eda":
        return

    raise NotImplementedError(
        f"--mode {args.mode} is not wired yet -- see PROJECT_PLAN.md Phase 2/3."
    )


if __name__ == "__main__":
    main()
