"""Placeholder for a measured-connectome MB extract ([A], not yet implemented).

Path to real wiring: extend `etl.py` CORE_TYPES at the repository root with the
FlyWire v783 primary types for Kenyon cells (KCab, KCa'b', KCg-m ...), MBONs
(MBON01..MBON35), PAM/PPL1 dopaminergic neurons and APL, write a
`data/mb_circuit.json` with signed KC→MBON and DAN→compartment edges, and load it
here into an `MBTopology`. Until then `build_mb_topology` (statistics only) is used
and every run records `topology.category = "B"`.
"""
from __future__ import annotations

from .mb_topology import MBTopology


def load_measured_mb(path: str) -> MBTopology:  # pragma: no cover - future work
    raise NotImplementedError("measured MB extract not available yet; see module docstring")
