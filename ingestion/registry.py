"""
Plain name -> loader class registry. No dynamic imports, no plugin magic --
a grader must be able to read this file in five seconds and know exactly
what --dataset exf2021 resolves to.

STATUS: DRAFT. References Exf2021Loader and DohBrw2020Loader, which do not
exist yet -- they are Step 1A (Teammate A) and Step 1B (Teammate B). This
file's import lines are commented out until those modules exist, so that
`import ingestion.registry` doesn't fail at import time for anyone testing
schema/unified.py or ingestion/base.py in isolation before Phase 1 starts.
"""

from __future__ import annotations

from typing import Type

from ingestion.base import AbstractLoader

# --- Uncomment as each loader lands (Step 1A / Step 1B) -------------------
# from ingestion.exf2021 import Exf2021Loader
# from ingestion.dohbrw2020 import DohBrw2020Loader

_REGISTRY: dict[str, Type[AbstractLoader]] = {
    # "exf2021": Exf2021Loader,
    # "dohbrw2020": DohBrw2020Loader,
}


def get(name: str) -> Type[AbstractLoader]:
    """Look up a loader class by its CLI --dataset name."""
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown dataset {name!r}. Registered datasets: {list(_REGISTRY)}"
        )
    return _REGISTRY[name]
