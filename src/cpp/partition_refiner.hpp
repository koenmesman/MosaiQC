#pragma once

#include <vector>

class CircuitGraph;

namespace pybind11 {
class module_;
}

// Refine a warm-start partition with a fast objective (uncovered edges + overlap penalties)
// and optional lazy evaluation of an expensive GED-like cost via callback.
//
// Binds a Python function:
//   refine_partition(graph, capacities, warm_start=[], max_passes=5, enable_overlap=True,
//                    overlap_penalty=0.0, cut_weight=1, overlap_weight=1,
//                    ged_weight=1.0, parallel_ged=False, topology_graphs=None, ged_callback=None,
//                    overlap_local_callback=None, use_tabu=False, tabu_tenure=7,
//                    tabu_max_iters=0, parallel_tabu=False, ged_balance=1.0,
//                    ged_eval_interval=0, ged_candidate_count=0,
//                    store_history=True, qap_use_routing=True) -> dict
void bind_partition_refiner(pybind11::module_& m);
