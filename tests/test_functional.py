if __name__ == "__main__":
    import MosaiQC as mq
    import time
    from qiskit.circuit.library import CDKMRippleCarryAdder
    from qiskit.circuit.random import random_circuit
    from qiskit import transpile

    QUBITS = 200

    try:
        from qiskit_ibm_runtime.fake_provider import (
            FakeAlgiers,
            FakeBrisbane,
            FakeWashingtonV2
        )
    except Exception as e:
        raise SystemExit(
            "qiskit_ibm_runtime fake providers are not available in this environment.\n"
            f"Import error: {e}"
        )

    backends = [FakeAlgiers(), FakeBrisbane(), FakeWashingtonV2()]
    s = time.perf_counter()
    qc = random_circuit(QUBITS, QUBITS)
    qc = transpile(qc, basis_gates=['u', 'cx', 'swap', 'crx', 'cry', 'crz'])

    sb = time.perf_counter()
    circuit = mq.CircuitGraph(qc)
    placement, hw_counts, q_budget, circuit_cpp, topo_graphs = mq.find_cutting_placement(qc, backends, include_topologies=True)
    result = mq.refine_partition(circuit_cpp, q_budget, warm_start=placement, topology_graphs=topo_graphs,
     ged_weight=1, max_passes=QUBITS**2, parallel_ged=True,
     cut_weight=1, overlap_weight=27, ged_balance=1 - 1/QUBITS,
     use_tabu=True, tabu_tenure=7, tabu_max_iters=QUBITS*2, parallel_tabu=True,
    )
    sc = time.perf_counter()
    print("time elapsed:",sb-s, sc-sb)
    print(result)
    