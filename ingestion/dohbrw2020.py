"""
Dataset B loader -- CIRA-CIC-DoHBrw-2020 (encrypted DNS-over-HTTPS).

STATUS: Step 1B, Teammate B. Reads the real downloaded layout locked in
Step 0C (docs/header_reconciliation_dohbrw2020.md): four pre-aggregated,
pre-labeled CSVs, not a single combined file:

    data/dohbrw2020/l1-doh.csv       -- DoH traffic, Benign+Malicious combined
    data/dohbrw2020/l1-nondoh.csv    -- ordinary HTTPS, not DoH at all
    data/dohbrw2020/l2-benign.csv    -- Benign-DoH only (19,807 rows)
    data/dohbrw2020/l2-malicious.csv -- Malicious-DoH only (249,836 rows)

l1-doh.csv is never read: its row count (269,643) is an exact union of
l2-benign + l2-malicious (19,807 + 249,836), so reading it would silently
double-count every DoH row.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ingestion.base import AbstractLoader
from schema.unified import DROPPED_COLUMNS, UNIFIED_COLUMNS, validate_schema

# Raw column -> unified column, per the Step 0C-locked arithmetic
# (docs/header_reconciliation_dohbrw2020.md section 6 / schema/unified.py
# COLUMN_SOURCE). Zero renames were needed against the original spec --
# every raw column name matched verbatim.
_RAW_TO_UNIFIED: dict[str, str] = {
    "PacketLengthMean": "vol_primary",
    "PacketLengthMedian": "vol_secondary",
    "FlowBytesSent": "vol_total",
    "PacketLengthCoefficientofVariation": "rand_entropy",
    "PacketLengthMode": "struct_segments",
    "PacketLengthStandardDeviation": "struct_max_segment",
    "PacketTimeMean": "time_central",
    "PacketTimeStandardDeviation": "time_dispersion",
    "PacketTimeSkewFromMedian": "time_skew",
    "FlowSentRate": "disp_uniqueness",
}

# Identifiers dropped by default -- testbed artifacts (fixed IPs/ports in the
# lab). Retained only when include_leakage_columns=True, for the Step 2C
# deliberate leakage demonstration.
_LEAKAGE_COLUMNS: list[str] = [
    "SourceIP",
    "DestinationIP",
    "SourcePort",
    "DestinationPort",
    "TimeStamp",
]


class DohBrw2020Loader(AbstractLoader):
    """Loads CIRA-CIC-DoHBrw-2020 under one of two framings (D4/D8).

    framing="hard" (PRIMARY): Benign-DoH vs Malicious-DoH, balanced ~1:1.
        non-DoH rows are dropped entirely. Raw data is 93% positive
        (19,807 benign vs 249,836 malicious) -- left unbalanced, an
        always-malicious classifier scores F1=0.962 on it, so this is not a
        usable "hard" framing at all. We keep every Benign-DoH row (the
        scarce class -- same logic as D5's light-attack rule) and subsample
        Malicious-DoH down to match, random_state=42. sample_frac is
        IGNORED entirely under this framing -- applying it (e.g. 0.25) still
        leaves ~76% positive, which is still broken.

    framing="easy" (SECONDARY, reported as inflated): positives =
        Malicious-DoH, negatives = Benign-DoH + non-DoH. ~98% of negatives
        are ordinary HTTPS and trivially separable on packet-length stats
        alone. sample_frac applies UNIFORMLY across all three raw groups
        here (no scarce class to protect on this side) -- this preserves
        the natural ~0.21 positive rate at any sample_frac, which is the
        point of using this framing only as a contrast case, never the
        headline number.
    """

    def __init__(
        self,
        config: dict,
        framing: str = "hard",
        include_leakage_columns: bool = False,
    ) -> None:
        if framing not in ("hard", "easy"):
            raise ValueError(f"framing must be 'hard' or 'easy', got {framing!r}")
        self.config = config
        self.framing = framing
        self.include_leakage_columns = include_leakage_columns
        self.data_dir = Path(config["paths"]["dohbrw2020"])
        self.sample_frac = config["sampling"]["sample_frac"]
        self.random_state = config["sampling"]["random_state"]

    def _read(self, filename: str) -> pd.DataFrame:
        return pd.read_csv(self.data_dir / filename)

    def load(self) -> tuple[pd.DataFrame, pd.Series, dict]:
        benign = self._read("l2-benign.csv")
        malicious = self._read("l2-malicious.csv")

        class_counts_raw: dict[str, int] = {
            "Benign-DoH": len(benign),
            "Malicious-DoH": len(malicious),
        }

        if self.framing == "hard":
            raw, y, attack_subclass, class_counts_sampled = self._frame_hard(
                benign, malicious
            )
            sample_frac_applied = None  # explicitly overridden -- see class docstring
        else:
            nondoh = self._read("l1-nondoh.csv")
            class_counts_raw["NonDoH"] = len(nondoh)
            raw, y, attack_subclass, class_counts_sampled = self._frame_easy(
                benign, malicious, nondoh
            )
            sample_frac_applied = self.sample_frac

        n_rows_raw = sum(class_counts_raw.values())

        X, dropped_columns = self._project(raw)
        validate_schema(X[UNIFIED_COLUMNS], mode="full")

        meta = {
            "dataset_name": "dohbrw2020",
            "framing": self.framing,
            "n_rows_raw": n_rows_raw,
            "n_rows_after_sampling": len(X),
            "class_counts_raw": class_counts_raw,
            "class_counts_sampled": class_counts_sampled,
            "dropped_columns": dropped_columns,
            "attack_subclass": attack_subclass,
            "positive_rate": float(y.mean()),
            "sample_frac_applied": sample_frac_applied,
        }

        return X, y, meta

    def _frame_hard(
        self, benign: pd.DataFrame, malicious: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series, dict[str, int]]:
        # D8: keep all of the scarce class, subsample the abundant one to match.
        n = min(len(benign), len(malicious))
        malicious_s = malicious.sample(n=n, random_state=self.random_state)

        raw = pd.concat([benign, malicious_s], ignore_index=True)
        y = pd.Series(
            [0] * len(benign) + [1] * len(malicious_s), index=raw.index, name="y"
        )
        attack_subclass = pd.Series(
            ["benign_doh"] * len(benign) + ["malicious_doh"] * len(malicious_s),
            index=raw.index,
            name="attack_subclass",
        )
        class_counts_sampled = {
            "Benign-DoH": len(benign),
            "Malicious-DoH": len(malicious_s),
        }
        return raw, y, attack_subclass, class_counts_sampled

    def _frame_easy(
        self, benign: pd.DataFrame, malicious: pd.DataFrame, nondoh: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series, dict[str, int]]:
        benign_s = benign.sample(frac=self.sample_frac, random_state=self.random_state)
        malicious_s = malicious.sample(
            frac=self.sample_frac, random_state=self.random_state
        )
        nondoh_s = nondoh.sample(frac=self.sample_frac, random_state=self.random_state)

        raw = pd.concat([benign_s, malicious_s, nondoh_s], ignore_index=True)
        y = pd.Series(
            [0] * len(benign_s) + [1] * len(malicious_s) + [0] * len(nondoh_s),
            index=raw.index,
            name="y",
        )
        attack_subclass = pd.Series(
            ["benign_doh"] * len(benign_s)
            + ["malicious_doh"] * len(malicious_s)
            + ["non_doh"] * len(nondoh_s),
            index=raw.index,
            name="attack_subclass",
        )
        class_counts_sampled = {
            "Benign-DoH": len(benign_s),
            "Malicious-DoH": len(malicious_s),
            "NonDoH": len(nondoh_s),
        }
        return raw, y, attack_subclass, class_counts_sampled

    def _project(self, raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
        out = pd.DataFrame(index=raw.index)
        for raw_col, unified_col in _RAW_TO_UNIFIED.items():
            out[unified_col] = raw[raw_col]
        out["rand_dispersion"] = np.log1p(raw["PacketLengthVariance"])
        out = out[UNIFIED_COLUMNS]

        dropped_columns: dict[str, str] = {}
        if self.include_leakage_columns:
            for col in _LEAKAGE_COLUMNS:
                out[col] = raw[col]
        else:
            for col in _LEAKAGE_COLUMNS:
                dropped_columns[col] = DROPPED_COLUMNS["dataset_b"][col]

        return out, dropped_columns
