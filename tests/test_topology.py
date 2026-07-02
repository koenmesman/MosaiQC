"""
Sanity test: TopologyGraph(backend=...) matches TopologyGraph(noise_model=NoiseModel.from_backend(...)).

This specifically checks the C++/pybind conversion path that builds a NoiseModel from a backend.

Run as a script:
    python tests/test_topology_graph_backend_conversion.py

Or with pytest:
    pytest -q
"""

from __future__ import annotations

from qiskit_ibm_runtime.fake_provider import FakeBelemV2
from qiskit_aer.noise import NoiseModel

import opt_core


def _assert_close(a: float, b: float, tol: float = 1e-12) -> None:
    if abs(a - b) > tol:
        raise AssertionError(f"{a} != {b} (tol={tol})")


def test_backend_conversion_matches_noise_model() -> None:
    backend = FakeBelemV2()

    # Path A: C++ TopologyGraph converts backend -> NoiseModel internally.
    topo_from_backend = opt_core.TopologyGraph(backend=backend)
    print("okay")
    # Path B: Python converts backend -> NoiseModel, then C++ TopologyGraph consumes that model.
    model = NoiseModel.from_backend(backend)
    topo_from_model = opt_core.TopologyGraph(noise_model=model)

    # Compare qubit sets
    qs_a = set(topo_from_backend.qubits())
    qs_b = set(topo_from_model.qubits())
    assert qs_a == qs_b

    # Compare directed edges
    edges_a = set(tuple(e) for e in topo_from_backend.directed_edges())
    edges_b = set(tuple(e) for e in topo_from_model.directed_edges())
    assert edges_a == edges_b

    # Compare per-node rates
    for q in qs_a:
        na = topo_from_backend.node(q)
        nb = topo_from_model.node(q)
        _assert_close(na.readout, nb.readout)
        _assert_close(na.x, nb.x)
        _assert_close(na.reset, nb.reset)
        _assert_close(na.sum_error, nb.sum_error)

    # Compare per-edge CX rates
    for (q0, q1) in edges_a:
        ea = topo_from_backend.edge(q0, q1)
        eb = topo_from_model.edge(q0, q1)
        _assert_close(ea.cx, eb.cx)


if __name__ == "__main__":
    # Lightweight script output similar to your example.
    backend = FakeBelemV2()
    print("okay")
    topo = opt_core.TopologyGraph(backend=backend)
    print("Directed edges:")
    print(topo.directed_edges())
