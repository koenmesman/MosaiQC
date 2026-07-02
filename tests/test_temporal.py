from qiskit import QuantumCircuit
import opt_core


class DummyBackend:
    def __init__(self, num_qubits: int) -> None:
        self.num_qubits = num_qubits


def _flatten(nested):
    out = []
    for item in nested:
        if isinstance(item, list):
            out.extend(item)
        else:
            out.append(item)
    return out


def test_temporal_connectivity_basic():
    qc = QuantumCircuit(4)
    qc.cx(0, 1)  # g0
    qc.cx(1, 3)  # g1
    qc.cx(0, 1)  # g2
    qc.x(2)      # ignored
    qc.cx(1, 2)  # g3

    cg = opt_core.CircuitGraph(qc)
    te = cg.temporal_events

    assert te.num_qubits == 4
    assert te.num_events == 4

    # partner timelines per qubit (merged 2Q events) in chronological order
    assert te.partner_timeline(1) == [0, 3, 0, 2]
    assert te.partner_timeline(0) == [1, 1]
    assert te.partner_timeline_with_ids(1) == [(0, 0), (3, 1), (0, 2), (2, 3)]
    assert te.event_original_indices(0) == [0]
    assert te.event_original_indices(1) == [1]
    assert te.event_original_indices(2) == [2]
    assert te.event_original_indices(3) == [4]

    tg = cg.temporal_graph
    assert tg.num_nodes == 4
    assert tg.num_edges == 4
    assert tg.node_weights == [4, 4, 4, 4]
    assert [tg.node(i) for i in range(tg.num_nodes)] == [
        (0, 0, 1),
        (1, 1, 3),
        (2, 0, 1),
        (4, 1, 2),
    ]
    assert {i: tg.neighbors(i) for i in range(tg.num_nodes)} == {
        0: ([1, 2], [1, 1]),
        1: ([0, 2], [1, 1]),
        2: ([0, 1, 3], [1, 1, 1]),
        3: ([2], [1]),
    }

    # fast lookup example: common qubit 1, set_a={0}, set_b={2,3}
    assert cg.count_temporal_cross_edges(1, [0], [2, 3]) == 3

    # 1Q counts
    assert cg.oneq_counts() == [0, 0, 1, 0]
    assert cg.node_weights() == [0, 0, 1, 0]

    # connectivity weights (2Q gate op_weight sums)
    conn = cg.connectivity
    assert conn.num_qubits == 4
    assert conn.node_weights == [0, 0, 1, 0]
    assert conn.count(0, 1) == 18.0
    assert conn.count(1, 0) == 18.0
    assert conn.count(1, 3) == 9.0
    assert conn.count(3, 1) == 9.0
    assert conn.count(1, 2) == 9.0
    assert conn.count(2, 1) == 9.0

    # neighbor listing
    n1, w1 = conn.neighbors(1)
    assert set(n1) == {0, 2, 3}
    # weights aligned with neighbors; convert to dict for comparison
    assert dict(zip(n1, w1)) == {0: 18.0, 2: 9.0, 3: 9.0}
    n1_i, w1_i, e1_i = conn.neighbors_with_ids(1)
    assert n1_i == n1
    assert w1_i == w1
    assert len(e1_i) == 3

    # edge identifiers + provenance back to original qc.data indices
    assert conn.num_edge_ids == 3
    e01 = conn.edge_id(0, 1)
    e10 = conn.edge_id(1, 0)
    e12 = conn.edge_id(1, 2)
    e13 = conn.edge_id(1, 3)
    assert e01 == e10 and e01 >= 0
    assert e12 >= 0
    assert e13 >= 0
    assert conn.edge_id(0, 3) == -1
    assert set(conn.edge_endpoints(e01)) == {0, 1}
    assert set(conn.edge_endpoints(e12)) == {1, 2}
    assert set(conn.edge_endpoints(e13)) == {1, 3}
    assert conn.edge_original_indices(e01) == [0, 2]
    assert conn.edge_original_indices(e12) == [4]
    assert conn.edge_original_indices(e13) == [1]
    assert conn.edge_original_indices_by_qubits(0, 1) == [0, 2]
    assert conn.edge_original_indices_by_qubits(1, 2) == [4]
    assert conn.edge_original_indices_by_qubits(1, 3) == [1]
    assert conn.edge_original_indices_by_qubits(0, 3) == []


def test_temporal_connectivity_merges_consecutive_same_pair():
    qc = QuantumCircuit(3)
    # Three consecutive 2Q gates acting on the same unordered pair {0,1}
    qc.cx(0, 1)  # original g0
    qc.cx(1, 0)  # original g1 (reverse direction)
    qc.cx(0, 1)  # original g2

    cg = opt_core.CircuitGraph(qc)
    te = cg.temporal_events

    # All three are merged into a single temporal event.
    assert te.num_events == 1
    assert te.event_multiplicity(0) == 3

    # Partner timelines have a single entry
    assert te.partner_timeline(0) == [1]
    assert te.partner_timeline(1) == [0]

    # Endpoints are still {0,1}
    e0 = set(te.event_endpoints(0))
    assert e0 == {0, 1}
    assert te.event_original_indices(0) == [0, 1, 2]
    assert te.partner_timeline_with_ids(0) == [(1, 0)]
    assert te.partner_timeline_with_ids(1) == [(0, 0)]

    tg = cg.temporal_graph
    assert tg.num_nodes == 3
    assert tg.num_edges == 2
    assert tg.node_weights == [2, 2, 2]
    assert {i: tg.neighbors(i) for i in range(tg.num_nodes)} == {
        0: ([1], [1]),
        1: ([0, 2], [1, 1]),
        2: ([1], [1]),
    }

    conn = cg.connectivity
    eid = conn.edge_id(0, 1)
    assert eid >= 0
    assert conn.edge_original_indices(eid) == [0, 1, 2]


def test_temporal_graph_can_be_partitioned_with_metis():
    qc = QuantumCircuit(4)
    qc.cx(0, 1)
    qc.cx(1, 3)
    qc.cx(0, 1)
    qc.cx(1, 2)

    cg = opt_core.CircuitGraph(qc)
    temporal_graph = cg.temporal_graph

    assert temporal_graph.num_nodes == 4
    assert sum(temporal_graph.node_weights) == 16

    try:
        placement = opt_core.global_temporal_METIS_solver(
            temporal_graph,
            [8, 8],
            imbalance=1.05,
            seed=7,
        )
    except RuntimeError as exc:
        if "METIS is not available" in str(exc):
            return
        raise

    assert len(placement) == temporal_graph.num_nodes
    assert all(part in (0, 1) for part in placement)
    assert len(set(placement)) == 2


def test_find_cutting_placement_wire_method_returns_temporal_graph():
    qc = QuantumCircuit(4)
    qc.cx(0, 1)
    qc.cx(1, 3)
    qc.cx(0, 1)
    qc.cx(1, 2)

    try:
        placement, hw_counts, q_budget, circuit_cpp, extras = opt_core.find_cutting_placement(
            qc,
            [DummyBackend(2), DummyBackend(2)],
            method="wire",
            imbalance=1.05,
            seed=7,
            include_initial_extras=True,
        )
    except RuntimeError as exc:
        if "METIS is not available" in str(exc):
            return
        raise

    assert isinstance(circuit_cpp, opt_core.CircuitGraph)
    assert len(placement) == qc.num_qubits
    assert len(extras) == qc.num_qubits
    assert all(0 <= part < len(_flatten(q_budget)) for part in placement)
    assert all(part != placement[q] for q, blocks in enumerate(extras) for part in blocks)
    assert sum(_flatten(q_budget)) >= sum(circuit_cpp.temporal_graph.node_weights)
    assert hw_counts == [1, 1]

    result = opt_core.refine_partition(
        circuit_cpp,
        q_budget,
        warm_start=placement,
        warm_start_extra_blocks=extras,
        max_passes=0,
        ged_weight=0.0,
    )
    assert result["primary"] == placement
    assert result["extra_blocks"] == extras

    duplicated_primary_extras = [blocks + [placement[q]] for q, blocks in enumerate(extras)]
    normalized = opt_core.refine_partition(
        circuit_cpp,
        q_budget,
        warm_start=placement,
        warm_start_extra_blocks=duplicated_primary_extras,
        max_passes=0,
        ged_weight=0.0,
    )
    assert normalized["extra_blocks"] == extras


def test_cut_edges_report_original_instruction_indices():
    qc = QuantumCircuit(3)
    qc.cx(0, 1)  # qc.data index 0
    qc.x(0)      # qc.data index 1 (ignored by CircuitGraph ops)
    qc.cx(1, 2)  # qc.data index 2
    qc.cx(0, 2)  # qc.data index 3

    cg = opt_core.CircuitGraph(qc)

    assert cg.ops_with_indices() == [
        (0, 0, 1, "cx"),
        (2, 1, 2, "cx"),
        (3, 0, 2, "cx"),
    ]

    primary = [0, 0, 1]
    assert cg.cut_edge_indices(primary) == [2, 3]
    assert cg.cut_edges(primary) == [
        (2, 1, 2, "cx"),
        (3, 0, 2, "cx"),
    ]

    extra_block = [-1, 1, -1]
    assert cg.cut_edge_indices(primary, extra_block) == [3]
