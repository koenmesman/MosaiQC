if __name__ == "__main__":
    import MosaiQC as mq
    import time
    from qiskit.circuit.library import CDKMRippleCarryAdder
    from qiskit.circuit.random import random_circuit
    from qiskit import transpile

    QUBITS = 75

    try:
        from qiskit_ibm_runtime.fake_provider import (
            FakeAlgiers,
            FakeBrisbane,
            FakeWashingtonV2,
            FakeMelbourneV2,
            FakePrague
        )
    except Exception as e:
        raise SystemExit(
            "qiskit_ibm_runtime fake providers are not available in this environment.\n"
            f"Import error: {e}"
        )
    # 27, 127, 127
    #backends = [FakeAlgiers(), FakeBrisbane(), FakeWashingtonV2()]
    # 27, 15, 33
    backends = [FakeAlgiers(), FakeMelbourneV2(), FakePrague()]
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
    primary = result["primary"]
    extra_block = result["extra_block"]  # include this so overlap is handled correctly

    # Cut gates from refine_partition now include both graph cuts and overlap-derived cuts.
    cut_gates = result["cut_gates"]
    print("cut_gates:", cut_gates)

    # Just the original qc.data indices (graph + overlap)
    cut_idx = result["cut_gate_indices"]
    print("cut_idx:", cut_idx)

    # Exact Qiskit instructions from the original circuit
    cut_instructions = [qc.data[i] for i in cut_idx]
    print("cut_instructions:", cut_instructions)



