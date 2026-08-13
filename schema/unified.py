"""
Unified Feature Schema for DNS exfiltration detection across two vantage points:
plaintext DNS (CIC-Bell-DNS-EXF-2021) and encrypted DoH (CIRA-CIC-DoHBrw-2020).

STATUS: LOCKED — both datasets verified against real downloaded files, 13 Aug 2026.
Dataset A: docs/header_reconciliation_exf2021.md (one drop: sld, plus timestamp).
Dataset B: docs/header_reconciliation_dohbrw2020.md (zero renames needed -- every
proposed column name matched the real CSV header verbatim). Step 0C is complete
for both sides; Step 0D (this file + ingestion/base.py + registry + config) can
proceed on the locked contract below.

THE CORE CLAIM THIS FILE ENCODES (see PROJECT_PLAN.md "Key Design Decisions", D2):

Five behavioural families were originally proposed as symmetric across both
datasets. They are not. Three of them (F1-F3) have a genuine, if imperfect,
counterpart on both sides of the encryption boundary -- we call this the
INTERSECTION. Two of them (F4-F5) exist only in Dataset B, because Dataset A
(restricted to stateless features per D1) has no temporal or dispersion
telemetry at all -- we call this the B_ONLY extension.

This split is a deliberate scientific claim, not a limitation to apologise for:
the encryption boundary destroys different amounts of information for
different feature families, and measuring exactly how much is the project's
central result.

    F1 Payload volume        -- INTERSECTION -- semantically equivalent both sides
    F2 Encoding randomness   -- INTERSECTION -- B-side is a statistical PROXY, not an equivalent
    F3 Structural complexity -- INTERSECTION -- B-side is a WEAK proxy
    F4 Temporal rhythm       -- B_ONLY       -- unobservable in Dataset A (stateless)
    F5 Endpoint dispersion   -- B_ONLY       -- unobservable in Dataset A (stateless)

Evaluation rules that follow directly from this split (see PROJECT_PLAN.md):
    - In-domain, Dataset B          -> mode="full"          (all 11 columns)
    - In-domain, Dataset A          -> mode="intersection"   (F1-F3 only; F4/F5 are all-NaN)
    - Cross-dataset transfer        -> mode="intersection"   (the only columns both sides share)
    - Ablation (D3)                 -> mode="F1_only" vs. mode="intersection"

F1's three columns are named vol_primary / vol_secondary / vol_total rather
than vol_mean / vol_max / vol_dispersion (D9): neither dataset actually
supplies a mean, a max, and a dispersion measure for volume, so the original
names were aspirational. FQDN_count and FlowBytesSent are both cumulative
measures, making vol_total a genuine semantic pairing rather than a fudge.
"""

from __future__ import annotations

import pandas as pd


# ---------------------------------------------------------------------------
# The 11 canonical output columns, in family order. Every loader must emit
# exactly these, in exactly this order. This order also matters for the
# 1D-CNN (Step 2F): the model's receptive field is only meaningful if related
# features sit physically adjacent in the vector.
# ---------------------------------------------------------------------------
UNIFIED_COLUMNS: list[str] = [
    # F1 -- Payload volume (INTERSECTION)
    "vol_primary",
    "vol_secondary",
    "vol_total",
    # F2 -- Encoding randomness (INTERSECTION)
    "rand_entropy",
    "rand_dispersion",
    # F3 -- Structural complexity (INTERSECTION)
    "struct_segments",
    "struct_max_segment",
    # F4 -- Temporal rhythm (B_ONLY)
    "time_central",
    "time_dispersion",
    "time_skew",
    # F5 -- Endpoint dispersion (B_ONLY)
    "disp_uniqueness",
]

FAMILY_MAP: dict[str, list[str]] = {
    "F1_payload_volume": ["vol_primary", "vol_secondary", "vol_total"],
    "F2_encoding_randomness": ["rand_entropy", "rand_dispersion"],
    "F3_structural_complexity": ["struct_segments", "struct_max_segment"],
    "F4_temporal_rhythm": ["time_central", "time_dispersion", "time_skew"],
    "F5_endpoint_dispersion": ["disp_uniqueness"],
}

FAMILY_ROLE: dict[str, str] = {
    "F1_payload_volume": "INTERSECTION",
    "F2_encoding_randomness": "INTERSECTION",
    "F3_structural_complexity": "INTERSECTION",
    "F4_temporal_rhythm": "B_ONLY",
    "F5_endpoint_dispersion": "B_ONLY",
}

# Security meaning of each family -- quoted directly in Chapter 1.3 (theoretical
# feature rationale) and Chapter 5.1 (unified schema spec).
FAMILY_SECURITY_MEANING: dict[str, str] = {
    "F1_payload_volume": (
        "Data volume moving through a channel that should only carry small "
        "lookups. Exfiltration must push far more bytes per unit than normal "
        "traffic, because the payload itself is the smuggled data."
    ),
    "F2_encoding_randomness": (
        "Entropy signature of base64URL-encoded payload. Encoded binary data "
        "looks statistically random; legitimate hostnames do not."
    ),
    "F3_structural_complexity": (
        "Carrier segmentation forced by DNS label-length limits (63 bytes/label, "
        "253 bytes/FQDN). Chunking a file into many labels produces structurally "
        "unusual names."
    ),
    "F4_temporal_rhythm": (
        "Machine-paced vs. human-paced timing. Automated exfiltration has "
        "regular inter-arrival timing; human browsing does not."
    ),
    "F5_endpoint_dispersion": (
        "Fan-out and destination rarity. Exfiltration channels concentrate "
        "traffic toward a small number of attacker-controlled endpoints."
    ),
}

# ---------------------------------------------------------------------------
# Per-column source mapping -- the actual arithmetic each loader implements.
# Both datasets LOCKED (Step 0C, 13 Aug 2026) against real downloaded files.
# ---------------------------------------------------------------------------
COLUMN_SOURCE: dict[str, dict[str, str]] = {
    "vol_primary": {
        "dataset_a": "len",  # LOCKED
        "dataset_b": "PacketLengthMean",  # LOCKED
    },
    "vol_secondary": {
        "dataset_a": "subdomain_length",  # LOCKED
        "dataset_b": "PacketLengthMedian",  # LOCKED
    },
    "vol_total": {
        "dataset_a": "FQDN_count",  # LOCKED
        "dataset_b": "FlowBytesSent",  # LOCKED
    },
    "rand_entropy": {
        "dataset_a": "entropy",  # LOCKED
        "dataset_b": "PacketLengthCoefficientofVariation",  # LOCKED
    },
    "rand_dispersion": {
        "dataset_a": "(numeric + special + upper) / len",  # LOCKED -- guard len == 0
        "dataset_b": "log1p(PacketLengthVariance)",  # LOCKED
    },
    "struct_segments": {
        "dataset_a": "labels",  # LOCKED
        "dataset_b": "PacketLengthMode",  # LOCKED
    },
    "struct_max_segment": {
        "dataset_a": "labels_max",  # LOCKED
        "dataset_b": "PacketLengthStandardDeviation",  # LOCKED
    },
    "time_central": {
        "dataset_a": None,  # unobservable -- stateless Dataset A has no temporal telemetry
        "dataset_b": "PacketTimeMean",  # LOCKED
    },
    "time_dispersion": {
        "dataset_a": None,
        "dataset_b": "PacketTimeStandardDeviation",  # LOCKED
    },
    "time_skew": {
        "dataset_a": None,
        "dataset_b": "PacketTimeSkewFromMedian",  # LOCKED
    },
    "disp_uniqueness": {
        "dataset_a": None,
        "dataset_b": "FlowSentRate",  # LOCKED
    },
}

# Columns dropped from each dataset's raw files, and why. Populated here so a
# single source of truth exists for what the leakage audit removed; loaders
# import this rather than hardcoding drop lists independently.
DROPPED_COLUMNS: dict[str, dict[str, str]] = {
    "dataset_a": {
        "sld": (
            "Raw text (real hostnames, NetBIOS-encoded strings) AND class-skewed "
            "cardinality -- 22 unique values in attack traffic vs. 11,134-22,153 "
            "in benign (D11). Structurally identical to the SourceIP leakage in "
            "Dataset B. Retained only behind include_leakage_columns=True for the "
            "Step 2C leakage demonstration."
        ),
        "timestamp": (
            "Wall-clock capture time, not a security-relevant per-query feature. "
            "No equivalent in Dataset B to intersect against."
        ),
    },
    "dataset_b": {
        "SourceIP": "Testbed artifact -- attacker used a fixed IP; see Step 2C leakage demo.",
        "DestinationIP": "Same testbed-artifact class as SourceIP.",
        "SourcePort": "Same testbed-artifact class as SourceIP.",
        "DestinationPort": "Same testbed-artifact class as SourceIP.",
        "TimeStamp": "Capture-session artifact, not a generalisable behavioural signal.",
    },
}

# ---------------------------------------------------------------------------
# Derived column groups -- the ONLY sanctioned way downstream code selects a
# family subset. No module outside this file should ever hardcode a column
# list; that is exactly the kind of dataset-shaped assumption the Dataset
# Dependency Rule (PROJECT_PLAN.md Step 0D) exists to prevent.
# ---------------------------------------------------------------------------
INTERSECTION_COLUMNS: list[str] = [
    c for c in UNIFIED_COLUMNS
    if FAMILY_ROLE[[fam for fam, cols in FAMILY_MAP.items() if c in cols][0]] == "INTERSECTION"
]

B_ONLY_COLUMNS: list[str] = [
    c for c in UNIFIED_COLUMNS
    if FAMILY_ROLE[[fam for fam, cols in FAMILY_MAP.items() if c in cols][0]] == "B_ONLY"
]

# The ablation arm (D3): F1 is the one family predicted to survive encryption
# unchanged. F2/F3 have B-side realisations that are proxies, not equivalents,
# and the ablation tests whether including them helps or hurts transfer.
F1_ONLY_COLUMNS: list[str] = list(FAMILY_MAP["F1_payload_volume"])

VALID_MODES: tuple[str, ...] = ("full", "intersection", "F1_only")

_MODE_COLUMNS: dict[str, list[str]] = {
    "full": UNIFIED_COLUMNS,
    "intersection": INTERSECTION_COLUMNS,
    "F1_only": F1_ONLY_COLUMNS,
}


class SchemaViolation(Exception):
    """Raised when a DataFrame does not conform to the unified schema contract."""


def validate_schema(X: pd.DataFrame, mode: str = "full") -> None:
    """Hard assert on the loader contract. Raises SchemaViolation, never warns.

    mode="full"         -> X.columns must equal UNIFIED_COLUMNS exactly, in order
    mode="intersection" -> X.columns must equal INTERSECTION_COLUMNS exactly, in order
    mode="F1_only"      -> X.columns must equal F1_ONLY_COLUMNS exactly, in order

    Also asserts: mode is valid, no duplicate columns, all dtypes numeric,
    X is not empty. Does NOT assert absence of NaN -- NaN is a legitimate,
    reportable signal in this schema (e.g. every B_ONLY column on Dataset A).
    """
    if mode not in VALID_MODES:
        raise SchemaViolation(
            f"Invalid mode {mode!r}. Must be one of {VALID_MODES}. "
            f"A typo'd mode string must raise, never silently pass."
        )

    expected = _MODE_COLUMNS[mode]

    if list(X.columns) != expected:
        raise SchemaViolation(
            f"Schema violation under mode={mode!r}.\n"
            f"Expected (in order): {expected}\n"
            f"Got (in order):      {list(X.columns)}"
        )

    if X.columns.duplicated().any():
        dupes = X.columns[X.columns.duplicated()].tolist()
        raise SchemaViolation(f"Duplicate columns: {dupes}")

    non_numeric = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    if non_numeric:
        raise SchemaViolation(f"Non-numeric columns: {non_numeric}")

    if X.empty:
        raise SchemaViolation("X is empty.")


def project(X: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Return the column subset for a given evaluation mode.

    The ONLY sanctioned way downstream code selects families -- no ad-hoc
    column lists anywhere else in the codebase.
    """
    if mode not in VALID_MODES:
        raise SchemaViolation(f"Invalid mode {mode!r}. Must be one of {VALID_MODES}.")
    return X[_MODE_COLUMNS[mode]]
