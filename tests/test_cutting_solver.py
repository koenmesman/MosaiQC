import math
#import pytest
from collections import Counter
from qiskit import QuantumCircuit
from qiskit.circuit import CircuitInstruction, Qubit
from qiskit.quantum_info import PauliList
from qiskit_addon_cutting import cut_gates, expand_observables, partition_problem
from qiskit_addon_cutting.instructions import CutWire, Move
from qiskit_addon_cutting.qpd import TwoQubitQPDGate

import MosaiQC as mq


class DummyBackend:
    def __init__(self, num_qubits: int) -> None:
        self.num_qubits = num_qubits


class DummyTopologyGraph:
    def __init__(self, num_qubits: int, undirected_edges: list[tuple[int, int]]) -> None:
        self._num_qubits = num_qubits
        self._directed_edges: list[tuple[int, int]] = []
        for a, b in undirected_edges:
            self._directed_edges.append((a, b))
            self._directed_edges.append((b, a))

    def qubits(self) -> list[int]:
        return list(range(self._num_qubits))

    def directed_edges(self) -> list[tuple[int, int]]:
        return self._directed_edges


def _flatten(nested):
    out = []
    for item in nested:
        if isinstance(item, list):
            out.extend(item)
        else:
            out.append(item)
    return out


def _cut_wires_stable(circuit: QuantumCircuit) -> QuantumCircuit:
    new_circuit = QuantumCircuit()
    mapping = list(range(len(circuit.qubits)))
    cut_wire_counts = Counter(
        circuit.find_bit(instruction.qubits[0]).index
        for instruction in circuit.data
        if instruction.operation.name == "cut_wire"
    )

    for qubit in circuit.qubits:
        index = circuit.find_bit(qubit).index
        for _ in range(cut_wire_counts.get(index, 0)):
            mapping[index + 1 :] = [item + 1 for item in mapping[index + 1 :]]
            new_circuit.add_bits([Qubit()])
        new_circuit.add_bits([qubit])

    for qreg in circuit.qregs:
        new_circuit.add_register(qreg)

    new_circuit.add_bits(circuit.clbits)
    for creg in circuit.cregs:
        new_circuit.add_register(creg)

    for instruction in circuit.data:
        gate_indices = [circuit.find_bit(qubit).index for qubit in instruction.qubits]
        if instruction.operation.name == "cut_wire":
            new_circuit.compose(
                TwoQubitQPDGate.from_instruction(Move()),
                qubits=[mapping[gate_indices[0]], mapping[gate_indices[0]] + 1],
                inplace=True,
            )
            mapping[gate_indices[0]] += 1
        else:
            new_circuit.compose(
                instruction.operation,
                qubits=[mapping[index] for index in gate_indices],
                clbits=[circuit.find_bit(clbit).index for clbit in instruction.clbits],
                inplace=True,
            )

    return new_circuit


def _generate_cut_circuit(qc: QuantumCircuit, gate_cuts, wire_cuts) -> QuantumCircuit:
    cut_qc, _ = cut_gates(qc, sorted({int(cut) for cut in gate_cuts})) if gate_cuts else (qc.copy(), None)
    for insert_before_idx, qubit in sorted({(int(idx), int(q)) for idx, q in wire_cuts}, reverse=True):
        cut_qc.data.insert(
            insert_before_idx,
            CircuitInstruction(CutWire(), [cut_qc.qubits[qubit]], []),
        )
    if wire_cuts:
        cut_qc = _cut_wires_stable(cut_qc)
    return cut_qc


def _realized_subcircuit_sizes(qc: QuantumCircuit, result: dict) -> list[int]:
    cut_qc = _generate_cut_circuit(qc, result["cut_gate_indices"], result["wire_cut_indices"])
    partition_labels = _partition_labels_for_cut_circuit(qc, result)
    observables = PauliList(["Z" + "I" * (qc.num_qubits - 1)])
    if result["wire_cut_indices"]:
        observables = expand_observables(observables, qc, cut_qc)
    problem = partition_problem(
        circuit=cut_qc,
        partition_labels=partition_labels,
        observables=observables,
    )
    subcircuits = problem.subcircuits
    items = subcircuits.items() if isinstance(subcircuits, dict) else enumerate(subcircuits)
    return sorted(circuit.num_qubits for _label, circuit in items)


def _partition_labels_for_cut_circuit(qc: QuantumCircuit, result: dict) -> list[int]:
    primary = [int(label) for label in result["primary"]]
    transitions_by_qubit = {}
    for insert_before_idx, qubit, left_label, right_label in result.get("wire_cut_partition_labels", []):
        transitions_by_qubit.setdefault(int(qubit), []).append(
            (int(insert_before_idx), int(left_label), int(right_label))
        )
    wire_cut_counts = Counter(int(qubit) for _insert_before_idx, qubit in result.get("wire_cut_indices", []))

    labels = []
    for qubit in range(qc.num_qubits):
        transitions = sorted(transitions_by_qubit.get(qubit, []))
        if wire_cut_counts[qubit] == 0:
            labels.append(primary[qubit])
            continue
        assert len(transitions) == wire_cut_counts[qubit]
        segment_labels = [transitions[0][1], transitions[0][2]]
        for _insert_before_idx, left_label, right_label in transitions[1:]:
            assert left_label == segment_labels[-1]
            segment_labels.append(right_label)
        labels.extend(segment_labels)
    return labels


def test_find_cutting_placement_shape_and_block_ids():
    qc = QuantumCircuit(10)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(2, 3)
    qc.cx(3, 4)
    qc.cx(0, 4)
    qc.cx(5, 6)
    qc.cx(6, 7)
    qc.cx(7, 8)
    qc.cx(8, 9)
    qc.cx(0, 9)
    qc.cx(3, 7)


    backends = [DummyBackend(3), DummyBackend(4), DummyBackend(4)]

    try:
        placement, hw_counts, q_budget, circuit_cpp = mq.find_cutting_placement(qc, backends)
        result = mq.refine_partition(circuit_cpp, q_budget, warm_start=placement)
    except RuntimeError as exc:
        if "METIS is not available" in str(exc):
            #pytest.skip("METIS is not available in this build")
            print("METIS is not available in this build")

        raise

    capacities = _flatten(q_budget)

    assert len(placement) == qc.num_qubits
    assert len(hw_counts) == len(backends)
    assert len(capacities) > 0
    assert sum(capacities) >= qc.num_qubits
    assert all(0 <= part < len(capacities) for part in placement)
    print("okay")
    print(result)

test_find_cutting_placement_shape_and_block_ids()


def test_refine_partition_overlap_is_beneficial():
    qc = QuantumCircuit(4)
    # Event order on qubit 0: [1, 3, 2]
    # With warm_start below, this is A, B, A for overlap(0 -> partition 0).
    # Adding that overlap removes two graph cuts and pays one weighted gate cut.
    qc.cx(0, 1)
    qc.cx(0, 3)
    qc.cx(0, 2)

    graph = mq.CircuitGraph(qc)
    capacities = [4, 4]
    warm_start = [1, 0, 0, 1]

    no_overlap = mq.refine_partition(
        graph,
        capacities,
        warm_start=warm_start,
        enable_overlap=False,
        max_passes=0,
        overlap_penalty=0.0,
        cut_weight=27,
        overlap_weight=16,
        ged_weight=0.0,
    )

    with_overlap = mq.refine_partition(
        graph,
        capacities,
        warm_start=warm_start,
        warm_start_extra_blocks=[[0], [], [], []],
        enable_overlap=True,
        max_passes=0,
        overlap_penalty=0.0,
        cut_weight=27,
        overlap_weight=16,
        ged_weight=0.0,
    )

    print(with_overlap)

    assert all(not x for x in no_overlap["extra_block"])
    assert any(x for x in with_overlap["extra_block"])
    assert with_overlap["fast_cost"] < no_overlap["fast_cost"]
    assert with_overlap["extra_block"][0] == [0]
    assert with_overlap["fast_cost"] == 27 * with_overlap["cut_cost"] + 16 * with_overlap["overlap_cost"]
    assert no_overlap["ged_eval_count"] == 0
    assert with_overlap["ged_eval_count"] == 0

test_refine_partition_overlap_is_beneficial()


def test_refine_partition_final_overlap_gate_cut_cost_is_not_underreported():
    qc = QuantumCircuit(4)
    # For qubit 0 overlapped into partition 0, the partner labels are A, B, A.
    # Gate-cutting the middle segment is enough to remove both wire cuts.
    qc.cx(0, 1)
    qc.cx(0, 3)
    qc.cx(0, 2)

    graph = mq.CircuitGraph(qc)
    result = mq.refine_partition(
        graph,
        [4, 4],
        warm_start=[1, 0, 0, 1],
        warm_start_extra_blocks=[[0], [], [], []],
        enable_overlap=True,
        max_passes=0,
        overlap_penalty=0.0,
        cut_weight=1,
        overlap_weight=1,
        ged_weight=0.0,
    )

    assert math.isclose(result["cut_cost"], 0.0, abs_tol=1e-12)
    assert math.isclose(result["overlap_cost"], math.log(9.0), rel_tol=1e-12)
    assert math.isclose(result["overlap_cost"], result["overlap_cost_history"][0], rel_tol=1e-12)
    assert result["overlap_cut_gate_indices"] == [1]
    assert result["overlap_wire_cut_indices"] == []


test_refine_partition_final_overlap_gate_cut_cost_is_not_underreported()


def test_mutual_overlap_counts_both_replicas():
    qc = QuantumCircuit(2)
    qc.cx(0, 1)

    graph = mq.CircuitGraph(qc)
    result = mq.refine_partition(
        graph,
        [2, 2],
        warm_start=[1, 0],
        warm_start_extra_blocks=[[0], [1]],
        enable_overlap=True,
        max_passes=0,
        overlap_penalty=0.0,
        cut_weight=1,
        overlap_weight=1,
        ged_weight=0.0,
    )

    assert result["partition_used_qubits"] == [2, 2]
    assert result["cut_gate_indices"] == []
    assert result["overlap_cut_gate_indices"] == []
    assert result["wire_cut_indices"] == []
    assert math.isclose(result["fast_cost"], 0.0, abs_tol=1e-12)


test_mutual_overlap_counts_both_replicas()


def test_refine_partition_final_overlap_wire_cut_cost_is_not_underreported():
    qc = QuantumCircuit(3)
    # For qubit 0 overlapped into partition 0, the partner labels are A, B.
    # That boundary is represented as a wire cut with cost log(16).
    qc.swap(0, 1)
    qc.swap(0, 2)

    graph = mq.CircuitGraph(qc)
    result = mq.refine_partition(
        graph,
        [3, 3],
        warm_start=[1, 0, 1],
        warm_start_extra_blocks=[[0], [], []],
        enable_overlap=True,
        max_passes=0,
        overlap_penalty=0.0,
        cut_weight=1,
        overlap_weight=1,
        ged_weight=0.0,
    )

    assert math.isclose(result["cut_cost"], 0.0, abs_tol=1e-12)
    assert math.isclose(result["overlap_cost"], math.log(16.0), rel_tol=1e-12)
    assert math.isclose(result["overlap_cost"], result["overlap_cost_history"][0], rel_tol=1e-12)
    assert result["overlap_cut_gate_indices"] == []
    assert result["overlap_wire_cut_indices"] == [(1, 0)]


test_refine_partition_final_overlap_wire_cut_cost_is_not_underreported()


def test_multi_overlap_gate_cuts_are_realized_as_valid_subcircuits():
    qc = QuantumCircuit(4)
    # q0 is shared by partitions 0, 1, and 2. CX segments are cheap enough
    # that the multi-label min-first evaluator gate-cuts the separating events.
    qc.cx(0, 1)
    qc.cx(0, 2)
    qc.cx(0, 3)
    qc.cx(0, 1)
    qc.cx(0, 2)

    capacities = [2, 3, 3]
    result = mq.refine_partition(
        mq.CircuitGraph(qc),
        capacities,
        warm_start=[0, 1, 2, 0],
        warm_start_extra_blocks=[[1, 2], [], [], []],
        enable_overlap=True,
        max_passes=0,
        overlap_penalty=0.0,
        cut_weight=1,
        overlap_weight=1,
        ged_weight=0.0,
    )

    assert result["extra_blocks"][0] == [1, 2]
    assert all(used <= capacity for used, capacity in zip(result["partition_used_qubits"], capacities))
    assert result["wire_cut_indices"] == []
    assert result["overlap_cut_gate_indices"]
    assert _realized_subcircuit_sizes(qc, result) == [1, 1, 2]


test_multi_overlap_gate_cuts_are_realized_as_valid_subcircuits()


def test_multi_overlap_wire_cuts_are_realized_as_valid_subcircuits():
    qc = QuantumCircuit(4)
    # SWAP segments are expensive enough that wire cuts are preferred. The
    # multi-label evaluator must emit two wire cuts for the three labels.
    qc.swap(0, 1)
    qc.swap(0, 2)
    qc.swap(0, 3)

    capacities = [2, 2, 2]
    result = mq.refine_partition(
        mq.CircuitGraph(qc),
        capacities,
        warm_start=[0, 1, 2, 0],
        warm_start_extra_blocks=[[1, 2], [], [], []],
        enable_overlap=True,
        max_passes=0,
        overlap_penalty=0.0,
        cut_weight=1,
        overlap_weight=1,
        ged_weight=0.0,
    )

    assert result["extra_blocks"][0] == [1, 2]
    assert all(used <= capacity for used, capacity in zip(result["partition_used_qubits"], capacities))
    assert result["overlap_cut_gate_indices"] == []
    assert result["wire_cut_indices"] == [(1, 0), (2, 0)]
    assert _realized_subcircuit_sizes(qc, result) == [2, 2, 2]


test_multi_overlap_wire_cuts_are_realized_as_valid_subcircuits()


def test_refine_partition_lazy_ged_callback():
    qc = QuantumCircuit(4)
    qc.cx(0, 1)
    qc.cx(1, 2)
    graph = mq.CircuitGraph(qc)

    capacities = [2, 2]
    warm_start = [0, 0, 1, 1]

    result = mq.refine_partition(
        graph,
        capacities,
        warm_start=warm_start,
        max_passes=0,
        ged_weight=1.0,
        ged_callback=lambda primary, extra: 1.0,
    )

    print(result)

    assert result["ged_evaluated"] is True
    assert result["ged_cost"] == 0.0
    assert result["ged_score"] == 1.0
    assert result["ged_cost_history"] == [1.0]
    assert result["total_cost"] == result["normalized cut cost"] + result["ged_cost_history"][-1]
    assert result["ged_eval_count"] == 1

test_refine_partition_lazy_ged_callback()


def test_refine_partition_tabu_mode_with_parallel_scoring():
    qc = QuantumCircuit(6)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(2, 3)
    qc.cx(3, 4)
    qc.cx(4, 5)
    qc.cx(0, 5)
    graph = mq.CircuitGraph(qc)

    capacities = [3, 3]
    warm_start = [0, 1, 0, 1, 0, 1]

    baseline = mq.refine_partition(
        graph,
        capacities,
        warm_start=warm_start,
        max_passes=0,
        ged_weight=0.0,
    )

    tabu = mq.refine_partition(
        graph,
        capacities,
        warm_start=warm_start,
        use_tabu=True,
        tabu_tenure=5,
        tabu_max_iters=24,
        parallel_tabu=True,
        ged_weight=0.0,
    )

    assert len(tabu["primary"]) == qc.num_qubits
    assert tabu["fast_cost"] <= baseline["fast_cost"]
    assert tabu["ged_eval_count"] == 0


test_refine_partition_tabu_mode_with_parallel_scoring()
