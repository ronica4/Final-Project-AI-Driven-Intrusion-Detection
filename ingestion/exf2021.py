"""
Dataset A loader -- CIC-Bell-DNS-EXF-2021 (plaintext DNS, stateless features only, per D1).

Reads the REAL file layout locked in Step 0C (docs/header_reconciliation_exf2021.md),
which diverges from the project brief in several ways -- see that doc for the full
audit. Summary relevant to this loader:

    - Attacks and benign are already-separate files (no 60/40 unmixing needed).
    - Attack files are further split by exfiltrated payload type (audio/compressed/
      exe/image/text/video), 6 files each for heavy and light.
    - Benign spans THREE separate sources: heavy_benign/ (3 files), light_benign/
      (1 file), and an easy-to-miss top_level_benign/ (2 files, ~221K rows -- over a
      third of total benign volume). All three must be read or benign is silently
      halved.
    - `sld` is raw text AND class-skewed (22 unique values in attack traffic vs.
      11K-22K in benign, D11) -- dropped unconditionally unless a caller explicitly
      asks for it for the Step 2C leakage demonstration.
    - `subdomain` is a boolean has-subdomain flag, not raw text -- not leakage, but
      also not part of the unified schema, so it is simply never selected into X.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from ingestion.base import AbstractLoader
from schema.unified import B_ONLY_COLUMNS, DROPPED_COLUMNS, UNIFIED_COLUMNS, validate_schema

_PAYLOAD_TYPE_RE = re.compile(r"stateless_features-(?:heavy|light)_(\w+)\.pcap\.csv$")


class Exf2021Loader(AbstractLoader):
    """Loads CIC-Bell-DNS-EXF-2021 stateless features into the unified schema.

    Parameters
    ----------
    config : dict
        Parsed config/config.yaml. Uses config["paths"]["exf2021"],
        config["sampling"]["sample_frac"], config["sampling"]["random_state"].
    include_leakage_columns : bool
        If True, returns the SPECIAL, NON-CONFORMANT output used ONLY by the
        Step 2C leakage demonstration: X gains an extra "_leakage_sld" column
        (raw text, unencoded) beyond the 11 unified columns, and
        validate_schema() is deliberately NOT called on the result, because
        that result does not and must not satisfy the Dataset Dependency Rule
        contract. No module other than the Step 2C leakage-demo script may
        ever construct a loader with this flag set.
    """

    def __init__(self, config: dict, include_leakage_columns: bool = False):
        self.config = config
        self.include_leakage_columns = include_leakage_columns
        self.data_dir = Path(config["paths"]["exf2021"])
        self.sample_frac = config["sampling"]["sample_frac"]
        self.random_state = config["sampling"]["random_state"]

    # -- internal helpers ----------------------------------------------------

    @staticmethod
    def _read_glob(directory: Path, pattern: str, tag_payload_type: bool) -> pd.DataFrame:
        paths = sorted(directory.glob(pattern))
        if not paths:
            raise FileNotFoundError(
                f"No files matched {directory / pattern} -- check Step 0B download "
                f"landed in the expected layout (see docs/header_reconciliation_exf2021.md)."
            )
        frames = []
        for p in paths:
            df = pd.read_csv(p)
            if tag_payload_type:
                m = _PAYLOAD_TYPE_RE.search(p.name)
                df["payload_type"] = m.group(1) if m else None
            else:
                df["payload_type"] = None
            frames.append(df)
        return pd.concat(frames, ignore_index=True)

    def _read_all_raw(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        heavy_attacks = self._read_glob(
            self.data_dir / "heavy_attacks" / "Attacks",
            "stateless_features-heavy_*.pcap.csv",
            tag_payload_type=True,
        )
        light_attacks = self._read_glob(
            self.data_dir / "light_attacks" / "Attacks",
            "stateless_features-light_*.pcap.csv",
            tag_payload_type=True,
        )
        heavy_attacks["attack_subclass"] = "heavy_attack"
        light_attacks["attack_subclass"] = "light_attack"
        attacks_raw = pd.concat([heavy_attacks, light_attacks], ignore_index=True)

        # Benign spans three sources -- all three required (see docs/header_reconciliation_exf2021.md).
        heavy_benign = self._read_glob(
            self.data_dir / "heavy_benign" / "Benign",
            "stateless_features-benign_heavy_*.pcap.csv",
            tag_payload_type=False,
        )
        light_benign = self._read_glob(
            self.data_dir / "light_benign" / "Benign",
            "stateless_features-light_benign.pcap.csv",
            tag_payload_type=False,
        )
        top_level_benign = self._read_glob(
            self.data_dir / "top_level_benign" / "Benign",
            "stateless_features-benign_*.pcap.csv",
            tag_payload_type=False,
        )
        benign_raw = pd.concat(
            [heavy_benign, light_benign, top_level_benign], ignore_index=True
        )
        benign_raw["attack_subclass"] = "benign"

        return attacks_raw, benign_raw

    def _subsample(
        self, attacks_raw: pd.DataFrame, benign_raw: pd.DataFrame
    ) -> tuple[pd.DataFrame, dict, dict]:
        light_rows = attacks_raw[attacks_raw["attack_subclass"] == "light_attack"]
        heavy_rows = attacks_raw[attacks_raw["attack_subclass"] == "heavy_attack"]

        class_counts_raw = {
            "heavy_attack": int(len(heavy_rows)),
            "light_attack": int(len(light_rows)),
            "benign": int(len(benign_raw)),
        }

        # D5: light_attack is NEVER sampled -- smallest class AND the analytical
        # centre of the project. Heavy and benign are sampled by config.sample_frac.
        if self.sample_frac < 1.0:
            heavy_rows = heavy_rows.sample(
                frac=self.sample_frac, random_state=self.random_state
            )
            benign_raw = benign_raw.sample(
                frac=self.sample_frac, random_state=self.random_state
            )

        combined = pd.concat([light_rows, heavy_rows, benign_raw], ignore_index=True)

        class_counts_sampled = {
            "heavy_attack": int((combined["attack_subclass"] == "heavy_attack").sum()),
            "light_attack": int((combined["attack_subclass"] == "light_attack").sum()),
            "benign": int((combined["attack_subclass"] == "benign").sum()),
        }

        return combined, class_counts_raw, class_counts_sampled

    def _build_unified_columns(self, combined: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(index=combined.index)
        X["vol_primary"] = combined["len"]
        X["vol_secondary"] = combined["subdomain_length"]
        X["vol_total"] = combined["FQDN_count"]
        X["rand_entropy"] = combined["entropy"]

        # Guard len == 0 to avoid divide-by-zero (locked arithmetic, Step 0C).
        denom = combined["len"].replace(0, np.nan)
        X["rand_dispersion"] = (
            combined["numeric"] + combined["special"] + combined["upper"]
        ) / denom

        X["struct_segments"] = combined["labels"]
        X["struct_max_segment"] = combined["labels_max"]

        # F4/F5 are unobservable in Dataset A (stateless-only, per D1/D2). Emitting
        # NaN here -- rather than inventing a value -- is itself the finding.
        for col in B_ONLY_COLUMNS:
            X[col] = np.nan

        return X[UNIFIED_COLUMNS]

    # -- public contract ------------------------------------------------------

    def load(self) -> tuple[pd.DataFrame, pd.Series, dict]:
        attacks_raw, benign_raw = self._read_all_raw()
        combined, class_counts_raw, class_counts_sampled = self._subsample(
            attacks_raw, benign_raw
        )

        dropped_columns: dict[str, str] = {}

        # Leakage guard (locked at 0C / D11): sld is raw text + class-skewed
        # cardinality. timestamp is never part of the unified schema.
        leakage_sld = None
        if self.include_leakage_columns:
            leakage_sld = combined["sld"].copy()
        else:
            dropped_columns["sld"] = DROPPED_COLUMNS["dataset_a"]["sld"]
        dropped_columns["timestamp"] = DROPPED_COLUMNS["dataset_a"]["timestamp"]

        X = self._build_unified_columns(combined)

        y = (combined["attack_subclass"] != "benign").astype(int)
        y.name = "y"

        attack_subclass = combined["attack_subclass"].copy()
        attack_subclass.index = X.index
        payload_type = combined["payload_type"].copy()
        payload_type.index = X.index

        if self.include_leakage_columns:
            # DELIBERATELY NON-CONFORMANT -- see class docstring. Used only by
            # the Step 2C leakage demonstration; validate_schema() is skipped
            # on purpose because this output does not satisfy the Dataset
            # Dependency Rule contract by design.
            X = X.copy()
            X["_leakage_sld"] = leakage_sld.values
        else:
            validate_schema(X, mode="full")

        meta = {
            "dataset_name": "exf2021",
            "framing": "n/a",
            "n_rows_raw": int(len(attacks_raw) + len(benign_raw)),
            "n_rows_after_sampling": int(len(X)),
            "class_counts_raw": class_counts_raw,
            "class_counts_sampled": class_counts_sampled,
            "dropped_columns": dropped_columns,
            "attack_subclass": attack_subclass,
            "payload_type": payload_type,  # bonus dimension, not a training feature
            "schema_validated": not self.include_leakage_columns,
        }

        return X, y, meta
