"""Cutting placement helpers."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Optional

from . import _core
from .device_manager import assign_hardware


def _flatten_capacities(q_budget: Iterable[object]) -> list[int]:
    capacities: list[int] = []
    for item in q_budget:
        if isinstance(item, Iterable) and not isinstance(item, (str, bytes)):
            for value in item:
                capacities.append(int(value))
        else:
            capacities.append(int(item))
    return capacities


def _copy_q_budget(q_budget: Iterable[object]) -> list[list[int]]:
    copied: list[list[int]] = []
    for item in q_budget:
        if isinstance(item, Iterable) and not isinstance(item, (str, bytes)):
            copied.append([int(value) for value in item])
        else:
            copied.append([int(item)])
    return copied


def _capacity_shape(q_budget: Iterable[object]) -> list[Optional[int]]:
    shape: list[Optional[int]] = []
    for item in q_budget:
        if isinstance(item, Iterable) and not isinstance(item, (str, bytes)):
            shape.append(len(list(item)))
        else:
            shape.append(None)
    return shape


def _unflatten_capacities(capacities: list[int], shape: list[Optional[int]]) -> list[object]:
    out: list[object] = []
    pos = 0
    for size in shape:
        if size is None:
            out.append(capacities[pos])
            pos += 1
        else:
            out.append(capacities[pos:pos + size])
            pos += size
    return out


def _scale_capacities_to_total(capacities: list[int], total: int) -> list[int]:
    if not capacities:
        return capacities
    current_total = sum(capacities)
    if total <= current_total:
        return capacities

    scaled = [max(1, (cap * total) // current_total) for cap in capacities]
    remainder = total - sum(scaled)
    order = sorted(
        range(len(capacities)),
        key=lambda i: ((capacities[i] * total) % current_total, capacities[i]),
        reverse=True,
    )
    for i in range(remainder):
        scaled[order[i % len(order)]] += 1
    return scaled


def _scale_q_budget_to_total(q_budget: Iterable[object], total: int) -> list[object]:
    shape = _capacity_shape(q_budget)
    capacities = _flatten_capacities(q_budget)
    return _unflatten_capacities(_scale_capacities_to_total(capacities, total), shape)


def _capacity_prefix_for_total(q_budget: Iterable[object], total: int) -> list[int]:
    capacities = _flatten_capacities(q_budget)
    if total <= 0:
        return []

    out: list[int] = []
    current_total = 0
    for capacity in capacities:
        out.append(capacity)
        current_total += capacity
        if current_total >= total:
            break
    return out


def _enforce_qubit_capacities(
    primary: list[int],
    extra_blocks: list[list[int]],
    capacities: list[int],
    *,
    add_capacity: int | None = None,
) -> int:
    k = len(capacities)
    used = [0] * k
    primary_counts = [0] * k
    primary_by_block = [set() for _ in range(k)]
    extras_by_block = [set() for _ in range(k)]
    unplaced: list[int] = []

    for qubit, block in enumerate(primary):
        if block < 0:
            unplaced.append(qubit)
        else:
            used[block] += 1
            primary_counts[block] += 1
            primary_by_block[block].add(qubit)
        for extra in extra_blocks[qubit]:
            used[extra] += 1
            extras_by_block[extra].add(qubit)

    added_parts = 0

    def add_partition() -> int:
        nonlocal k, added_parts
        if add_capacity is None:
            raise ValueError("unplaced qubits require available capacity")
        capacities.append(int(add_capacity))
        used.append(0)
        primary_counts.append(0)
        primary_by_block.append(set())
        extras_by_block.append(set())
        block = k
        k += 1
        added_parts += 1
        return block

    for qubit in unplaced:
        target = -1
        for block in range(k):
            if used[block] < capacities[block]:
                target = block
                break
        if target < 0:
            target = add_partition()
        primary[qubit] = target
        used[target] += 1
        primary_counts[target] += 1
        primary_by_block[target].add(qubit)

    def remove_extra(block: int) -> bool:
        qubits = extras_by_block[block]
        if not qubits:
            return False
        qubit = qubits.pop()
        extra_blocks[qubit].remove(block)
        used[block] -= 1
        return True

    under_primary = [block for block in range(k) if primary_counts[block] < capacities[block]]
    for source in range(k):
        while primary_counts[source] > capacities[source]:
            while under_primary and primary_counts[under_primary[-1]] >= capacities[under_primary[-1]]:
                under_primary.pop()
            if not under_primary:
                break
            target = under_primary[-1]
            qubit = next(iter(primary_by_block[source]))
            if target in extra_blocks[qubit]:
                extra_blocks[qubit].remove(target)
                extras_by_block[target].remove(qubit)
                used[target] -= 1
            while used[target] >= capacities[target] and remove_extra(target):
                pass
            if used[target] >= capacities[target]:
                under_primary.pop()
                continue
            primary[qubit] = target
            primary_by_block[source].remove(qubit)
            primary_by_block[target].add(qubit)
            primary_counts[source] -= 1
            primary_counts[target] += 1
            used[source] -= 1
            used[target] += 1

    free_blocks = [block for block in range(k) if used[block] < capacities[block]]
    for block in range(k):
        while used[block] > capacities[block]:
            qubits = extras_by_block[block]
            if not qubits:
                break
            qubit = qubits.pop()
            extra_blocks[qubit].remove(block)
            used[block] -= 1
            while free_blocks and used[free_blocks[-1]] >= capacities[free_blocks[-1]]:
                free_blocks.pop()
            target = -1
            for candidate in reversed(free_blocks):
                if candidate != primary[qubit] and candidate not in extra_blocks[qubit]:
                    target = candidate
                    break
            if target >= 0:
                extra_blocks[qubit].append(target)
                extras_by_block[target].add(qubit)
                used[target] += 1

    return added_parts


def _topologies_for_counts(hardwares, hw_counts: Iterable[int]):
    per_hw_topologies = [_core.TopologyGraph(backend=hw) for hw in hardwares]
    topology_graphs = []
    for hw_index, count in enumerate(hw_counts):
        topology_graphs.extend([per_hw_topologies[hw_index]] * int(count))
    return topology_graphs


def _expand_hardware_capacity(hardwares, hw_counts: list[int], q_budget, expand_hw: int):
    if expand_hw < 0:
        raise ValueError("expand_hw must be non-negative")

    q_budget = _copy_q_budget(q_budget)
    if expand_hw == 0:
        return hw_counts, q_budget

    if not hardwares:
        raise ValueError("expand_hw requires at least one hardware backend")

    # Add requested extra partitions to the last backend type.
    hw_counts = list(hw_counts)
    hw_index = len(hardwares) - 1
    hw_qubits = int(hardwares[hw_index].num_qubits)
    hw_counts[hw_index] += expand_hw
    q_budget[hw_index].extend([hw_qubits] * expand_hw)
    return hw_counts, q_budget


def global_METIS_solver(circuit_graph, q_budget, *, imbalance: float = 1.03, seed: int = 42) -> list[int]:
    capacities = _flatten_capacities(q_budget)
    if hasattr(circuit_graph, "num_nodes"):
        if len(capacities) == 1:
            return [0] * int(circuit_graph.num_nodes)
        capacities = _scale_capacities_to_total(capacities, sum(circuit_graph.node_weights))
        return _core.global_temporal_METIS_solver(circuit_graph, capacities, imbalance=imbalance, seed=seed)
    if len(capacities) == 1:
        return [0] * int(circuit_graph.num_qubits())
    return _core.global_METIS_solver(circuit_graph, capacities, imbalance=imbalance, seed=seed)


def find_cutting_placement(circuit, hardwares, *, imbalance: float = 1.03, seed: int = 42, include_topologies: bool = False, expand_hw=0, method: str = "gate", include_initial_extras: bool = False):
    """Return an initial qubit->partition placement from METIS."""
    if method not in {"gate", "wire"}:
        raise ValueError("method must be 'gate' or 'wire'")

    if include_topologies:
        hw_counts, q_budget, topology_graphs = assign_hardware(circuit.num_qubits, hardwares, return_topologies=True)
    else:
        hw_counts, q_budget = assign_hardware(circuit.num_qubits, hardwares)

    if expand_hw:
        current_parts = len(_flatten_capacities(q_budget))
        max_extra_parts = max(0, int(circuit.num_qubits) - current_parts)
        hw_counts, q_budget = _expand_hardware_capacity(
            hardwares,
            hw_counts,
            q_budget,
            min(int(expand_hw), max_extra_parts),
        )
        if include_topologies:
            topology_graphs = _topologies_for_counts(hardwares, hw_counts)

    circuit_graph = _core.CircuitGraph(circuit)
    circuit_cpp = circuit_graph
    initial_extra_blocks: list[list[int]] = [[] for _ in range(circuit.num_qubits)]

    if method == "wire":
        temporal_graph = circuit_graph.temporal_graph
        votes: list[dict[int, int]] = [dict() for _ in range(circuit.num_qubits)]
        temporal_nodes = [temporal_graph.node(node_id) for node_id in range(temporal_graph.num_nodes)]
        active_qubits = {q for _, q0, q1 in temporal_nodes for q in (q0, q1)}

        if active_qubits:
            temporal_q_budget = _capacity_prefix_for_total(q_budget, len(active_qubits))
            temporal_placement = global_METIS_solver(temporal_graph, temporal_q_budget, imbalance=imbalance, seed=seed)

            # Convert temporal-node partitions back to a qubit-level warm start.
            for (_, q0, q1), part in zip(temporal_nodes, temporal_placement):
                votes[q0][part] = votes[q0].get(part, 0) + 1
                votes[q1][part] = votes[q1].get(part, 0) + 1

        initial_placement = [
            max(qubit_votes.items(), key=lambda item: (item[1], -item[0]))[0] if qubit_votes else -1
            for qubit_votes in votes
        ]
        for qubit, qubit_votes in enumerate(votes):
            primary = initial_placement[qubit]
            initial_extra_blocks[qubit] = sorted(part for part in qubit_votes if part != primary)
        capacities = _flatten_capacities(q_budget)
        added_parts = _enforce_qubit_capacities(
            initial_placement,
            initial_extra_blocks,
            capacities,
            add_capacity=int(hardwares[-1].num_qubits),
        )
        if added_parts:
            hw_counts, q_budget = _expand_hardware_capacity(hardwares, hw_counts, q_budget, added_parts)
            if include_topologies:
                topology_graphs = _topologies_for_counts(hardwares, hw_counts)
    else:
        initial_placement = global_METIS_solver(circuit_cpp, q_budget, imbalance=imbalance, seed=seed)

    if include_topologies:
        if include_initial_extras:
            return initial_placement, hw_counts, q_budget, circuit_cpp, topology_graphs, initial_extra_blocks
        return initial_placement, hw_counts, q_budget, circuit_cpp, topology_graphs
    if include_initial_extras:
        return initial_placement, hw_counts, q_budget, circuit_cpp, initial_extra_blocks
    return initial_placement, hw_counts, q_budget, circuit_cpp


def refine_partition(
    circuit_graph,
    q_budget,
    *,
    warm_start: list[int] | None = None,
    max_passes: int = 5,
    enable_overlap: bool = True,
    overlap_penalty: float = 0.0,
    cut_weight: int = 1,
    overlap_weight: int = 1,
    ged_weight: float = 1.0,
    parallel_ged: bool = False,
    use_tabu: bool = False,
    tabu_tenure: int = 7,
    tabu_max_iters: int = 0,
    parallel_tabu: bool = False,
    ged_balance: float = 1.0,
    ged_eval_interval: int = 2,
    ged_candidate_count: int = 10,
    ged_callback=None,
    topology_graphs=None,
    hardwares=None,
    overlap_local_callback=None,
    warm_start_extra_blocks=None,
    optimize_hardware_placement: bool = True,
    store_history: bool = True,
    qap_use_routing: bool = True,
):
    """Refine a (warm-start) placement with optional overlap and lazy GED evaluation.

    Args:
        circuit_graph: _core.CircuitGraph
        q_budget: capacities, can be nested iterables
        warm_start: optional list[int] primary assignment (e.g. output of global_METIS_solver)
        warm_start_extra_blocks: optional list[list[int]] initial extra block memberships per qubit
        max_passes: number of greedy refinement passes
        enable_overlap: allow extra block memberships per node
        overlap_penalty: base penalty per overlapped node
        cut_weight: multiplier applied to cut_cost in the fast objective
        overlap_weight: multiplier applied to overlap_cost in the fast objective
        ged_weight: multiplier applied to mapping mismatch in the combined objective. When GED
            is evaluated, total_cost is (fast_cost / upper_bound_cuts) + (1 - ged_score) * ged_weight.
        parallel_ged: enable parallel per-partition GED evaluation in C++ (OpenMP build required).
        use_tabu: enable tabu-search mode (allows non-improving moves with tabu memory).
        tabu_tenure: tabu tenure in iterations for reverse moves in tabu mode.
        tabu_max_iters: max tabu iterations (0 derives from max_passes and problem size).
        parallel_tabu: enable parallel move scoring in tabu mode (OpenMP build required).
            Falls back to sequential scoring when overlap_local_callback is provided.
        ged_balance: float > 0. After the forced initial GED evaluation,
            GED is evaluated only when fast_cost < best_fast * ged_balance,
            where best_fast is the fast cost of the current best-total incumbent.
        ged_eval_interval: in tabu mode, run GED-aware candidate selection every N iterations
            when > 0. This lets ged_weight steer accepted moves.
        ged_candidate_count: number of top fast-objective candidate moves to GED-score on
            GED-aware tabu iterations.
        topology_graphs: list of hardware topology graphs aligned to partitions. If provided,
            GED is computed natively in C++ by comparing partition-induced circuit subgraphs
            against these hardware topologies.
        ged_callback: optional Python GED callback (used only when topology_graphs is not provided)
        hardwares: optional hardware backends; used only when topology_graphs is not provided.
            Topology graphs are derived via assign_hardware(..., return_topologies=True).
        overlap_local_callback: callable(v, primary, extra) -> float; local per-overlap penalty
        optimize_hardware_placement: when True, same-capacity hardware topologies are permuted
            during mapping-score evaluation and the best score is used. When False, partition
            label p is scored against topology_graphs[p].
        store_history: when False, skip retaining per-move and per-GED history arrays.
        qap_use_routing: when True, native QAP scoring uses a fast error-weighted
            all-pairs routing proxy for non-adjacent hardware qubit pairs.
    """
    capacities = _flatten_capacities(q_budget)
    if warm_start is None:
        warm_start = []
    if warm_start_extra_blocks is None:
        warm_start_extra_blocks = []

    if topology_graphs is None and hardwares is not None:
        _, _, topology_graphs = assign_hardware(circuit_graph.num_qubits(), hardwares, return_topologies=True)

    return _core.refine_partition(
        circuit_graph,
        capacities,
        warm_start=warm_start,
        warm_start_extra_blocks=warm_start_extra_blocks,
        max_passes=max_passes,
        enable_overlap=enable_overlap,
        overlap_penalty=overlap_penalty,
        cut_weight=cut_weight,
        overlap_weight=overlap_weight,
        ged_weight=ged_weight,
        parallel_ged=parallel_ged,
        use_tabu=use_tabu,
        tabu_tenure=tabu_tenure,
        tabu_max_iters=tabu_max_iters,
        parallel_tabu=parallel_tabu,
        ged_balance=ged_balance,
        ged_eval_interval=ged_eval_interval,
        ged_candidate_count=ged_candidate_count,
        topology_graphs=topology_graphs,
        ged_callback=ged_callback,
        overlap_local_callback=overlap_local_callback,
        optimize_hardware_placement=optimize_hardware_placement,
        store_history=store_history,
        qap_use_routing=qap_use_routing,
    )
