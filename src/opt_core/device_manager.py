from itertools import accumulate

def pick(A, b):
    # faster, but less optimal
    single = min((x for x in A if x >= b), default=None)
    if single is not None:
        return [single]  # or return single if you prefer

    B = sorted(A, reverse=True)
    k = next(i for i, s in enumerate(accumulate(B), 1) if s >= b)
    return B[:k]

from bisect import bisect_left, insort

def min_k_then_min_sum_local(A, b):
    # slower, but more accurate meaning faster in overall optimization
    A = list(A)

    # 1) smallest single a >= b
    single = min((x for x in A if x >= b), default=None)
    if single is not None:
        return [single]

    # 2) find minimal k using k largest
    B = sorted(A, reverse=True)
    s = 0
    k = 0
    for k, x in enumerate(B, 1):
        s += x
        if s >= b:
            break

    chosen = B[:k]
    chosen_sum = sum(chosen)

    # multiset of remaining values (sorted ascending for bisect)
    remaining = sorted(B[k:])

    # 3) improvement loop: try to replace chosen items with smaller ones
    #    while maintaining sum >= b and reducing total sum.
    improved = True
    while improved:
        improved = False

        # try replacing larger chosen elements first (more room)
        chosen.sort(reverse=True)

        for i, x in enumerate(chosen):
            # Need y such that chosen_sum - x + y >= b  =>  y >= b - (chosen_sum - x)
            need = b - (chosen_sum - x)

            # We want the smallest y that still works, but also y < x to improve
            idx = bisect_left(remaining, need)
            if idx < len(remaining) and remaining[idx] < x:
                y = remaining.pop(idx)          # take that smallest feasible replacement
                insort(remaining, x)            # put x back into remaining

                chosen_sum = chosen_sum - x + y
                chosen[i] = y
                improved = True
                # restart scanning (greedy improvement)
                break

    return chosen


def assign_hardware(qubits : int, hardwares, *, return_topologies: bool = False):
    q_index = []
    for h in hardwares:
        q_index.append(h.num_qubits)
    total_hw_qubits = sum(q_index)
    duplicity = qubits // total_hw_qubits
    hw_counts = [duplicity]*len(hardwares)

    remainder = qubits % total_hw_qubits
    if remainder:
        if remainder in q_index:
            picked_i = [q_index.index(remainder)]
        else:
            picked_q = min_k_then_min_sum_local(q_index, remainder)
            picked_i = [q_index.index(i) for i in picked_q]

        for i in picked_i:
            hw_counts[i] += 1
    qubit_budget = [[q]*m for q,m in zip(q_index, hw_counts)]

    if not return_topologies:
        return hw_counts, qubit_budget

    # Build one topology per hardware backend, then reuse references per partition copy.
    from . import _core

    per_hw_topologies = [_core.TopologyGraph(backend=hw) for hw in hardwares]
    partition_topologies = []
    for i, count in enumerate(hw_counts):
        partition_topologies.extend([per_hw_topologies[i]] * count)

    return hw_counts, qubit_budget, partition_topologies
