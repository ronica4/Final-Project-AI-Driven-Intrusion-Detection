"""
AbstractLoader -- the enforcing contract of the Dataset Dependency Rule.

Only code under ingestion/ is allowed to know which dataset it's reading.
Every module downstream (preprocessing, feature engineering, training,
evaluation) must be entirely agnostic of the dataset that produced its input.
A grader must be able to swap Dataset A for Dataset B and have the rest of
the pipeline run unchanged.

STATUS: DRAFT pending joint review (Step 0D is a joint step; drafted solo
while Teammate B was mid-Step-0A, to be reviewed together once both are back
online).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class AbstractLoader(ABC):
    """Base class every dataset-specific loader must implement.

    Concrete loaders live in ingestion/exf2021.py and ingestion/dohbrw2020.py.
    Nothing outside ingestion/ should ever import a concrete loader directly --
    always go through ingestion/registry.py.
    """

    @abstractmethod
    def load(self) -> tuple[pd.DataFrame, pd.Series, dict]:
        """Returns (X, y, meta).

        X    MUST have exactly schema.unified.UNIFIED_COLUMNS, in order.
             Validate with schema.unified.validate_schema(X, mode="full")
             before returning.
        y    MUST be a binary int Series (1 = exfiltration), same index as X.
        meta MUST be a dict carrying provenance ONLY -- never consumed by any
             downstream module except evaluation/error_analysis.py, which is
             explicitly allowed to read meta["attack_subclass"] for
             post-prediction forensics (never for training).

             Required keys:
               dataset_name          : str   -- e.g. "exf2021", "dohbrw2020"
               framing                : str   -- "n/a" for Dataset A;
                                                  "hard" | "easy" for Dataset B
               n_rows_raw             : int
               n_rows_after_sampling  : int
               class_counts_raw       : dict[str, int]
               class_counts_sampled   : dict[str, int]
               dropped_columns        : dict[str, str]  -- column -> reason
               attack_subclass        : pd.Series | None
                                         -- aligned to X.index. Carries the
                                            finer-grained label (e.g.
                                            "heavy_attack" / "light_attack" /
                                            "benign", or the DoH tunnel tool
                                            name). Lives in meta specifically
                                            so it CANNOT leak into training --
                                            only error_analysis.py may read it,
                                            and only after predictions exist.
        """
        raise NotImplementedError
