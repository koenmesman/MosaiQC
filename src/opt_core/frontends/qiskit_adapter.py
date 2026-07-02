"""Qiskit adapters for the pure-Python IR dataclasses."""

from qiskit import QuantumCircuit
from ..ir import GateIR, DeviceIR, CircuitIR


def circuit_from_qiskit(circ: QuantumCircuit) -> CircuitIR:
    gates = []

    for instr, qargs, _cargs in circ.data:
        qubits = tuple(circ.find_bit(q).index for q in qargs)
        params = []
        for p in instr.params:
            try:
                params.append(float(p))
            except TypeError:
                pass

        gates.append(GateIR(name=instr.name, qubits=qubits, params=params))

    return CircuitIR(gates=gates, device=None)


def device_from_qiskit(backend) -> DeviceIR:
    from qiskit_aer.noise import NoiseModel

    model = NoiseModel.from_backend(backend)
    coupling_map = getattr(backend, "coupling_map", None)
    couplings = list(coupling_map.get_edges()) if coupling_map is not None else []
    return DeviceIR(
        n_qubits=int(backend.num_qubits),
        couplings=couplings,
        single_qubit_error=getattr(model, "single_qubit_error", None),
        two_qubit_error=getattr(model, "two_qubit_error", None),
    )
