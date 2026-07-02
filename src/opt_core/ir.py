# qopt_core/ir.py  (pure Python)

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

QubitId = int

@dataclass
class GateIR:
    name: str
    qubits: Tuple[QubitId, ...]   # e.g. (0, 1)
    params: List[float]          # gate parameters, if any

@dataclass
class DeviceIR:
    n_qubits: int
    couplings: List[Tuple[QubitId, QubitId]]  # coupling graph
    single_qubit_error: Dict[QubitId, float] = None
    two_qubit_error: Dict[Tuple[QubitId, QubitId], float] = None

@dataclass
class CircuitIR:
    gates: List[GateIR]
    device: Optional[DeviceIR] = None
