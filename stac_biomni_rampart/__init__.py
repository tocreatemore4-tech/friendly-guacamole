# Copyright (c) 2026.
# STAC-on-Biomni replication for Microsoft RAMPART -- research red-teaming use.
"""Replication of STAC (Sequential Tool Attack Chaining, arXiv:2509.25624) as a
Microsoft RAMPART attack strategy, targeting the Biomni A1 biomedical agent
(snap-stanford/biomni) in no-datalake mode.

Public surface:
  * :class:`~stac_biomni_rampart.adapter.BiomniAdapter` /
    :class:`~stac_biomni_rampart.adapter.MockBiomniAdapter`
  * :class:`~stac_biomni_rampart.stac_execution.STACExecution`
  * :func:`~stac_biomni_rampart.cases.load_cases`,
    :class:`~stac_biomni_rampart.cases.STACCase`
  * :mod:`~stac_biomni_rampart.defenses`
"""

from __future__ import annotations

__all__ = [
    "BiomniAdapter",
    "MockBiomniAdapter",
    "STACExecution",
    "STACCase",
    "load_cases",
    "get_case",
]

from stac_biomni_rampart.adapter import BiomniAdapter, MockBiomniAdapter
from stac_biomni_rampart.cases import STACCase, get_case, load_cases
from stac_biomni_rampart.stac_execution import STACExecution
