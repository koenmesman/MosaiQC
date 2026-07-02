"""opt_core package.

Wraps the compiled C++ extension module `_core`.

Important: do NOT create a Python file named `_core.py` in this package.
It would shadow the compiled extension and cause circular-import errors.
"""

from . import _core  # compiled extension

sum_weights = _core.sum_weights
TopologyGraph = _core.TopologyGraph
NodeErrorRates = _core.NodeErrorRates
EdgeErrorRates = _core.EdgeErrorRates
CircuitGraph = _core.CircuitGraph
TemporalGateEvents = _core.TemporalGateEvents
TemporalGraph = _core.TemporalGraph
ConnectivityGraph = _core.ConnectivityGraph
CircuitInfo = _core.CircuitInfo
circuit_from_qiskit = _core.circuit_from_qiskit
global_temporal_METIS_solver = _core.global_temporal_METIS_solver
from .device_manager import assign_hardware
from .cutting_solver import find_cutting_placement, global_METIS_solver, refine_partition

__all__ = [
    "sum_weights",
    "TopologyGraph",
    "NodeErrorRates",
    "EdgeErrorRates",
    "CircuitGraph",
    "TemporalGateEvents",
    "TemporalGraph",
    "ConnectivityGraph",
    "CircuitInfo",
    "circuit_from_qiskit",
    "global_temporal_METIS_solver",
    "_core",
    "assign_hardware",
    "global_METIS_solver",
    "find_cutting_placement",
    "refine_partition",
]
