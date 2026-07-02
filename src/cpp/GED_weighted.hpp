#pragma once

#include <utility>
#include <vector>

class CircuitGraph;
class TopologyGraph;

namespace GEDWeighted {

// Full GED variant with:
// - node insertion forbidden
// - node deletion free
// - edge deletion free
// - edge insertion cost = edge_ins_cost
//
// edges1 and edges2 are undirected edge lists (u, v). Self-loops are ignored.
double ged_no_node_insertion_free_deletions(const std::vector<std::pair<int, int>>& edges1,
                                            const std::vector<std::pair<int, int>>& edges2,
                                            double edge_ins_cost = 1.0);

// Partition-level GED cost:
// Compares each partition-induced circuit subgraph against the mapped hardware topology graph
// with weighted objective:
//   maximize sum(node_oneq_count * hw_node_x_logprob + edge_2q_count * hw_edge_cx_logprob)
// over injective circuit->hardware node mappings that preserve required circuit edges.
//
// Let part_cost = -best_score for each partition, and mean_cost = average(part_cost).
// Returns GED score g = exp(-mean_cost), so g is in [0, 1].
// Larger g means better mapping quality.
//
// If hardware_placement is empty, partition p uses hardwares[p].
// Otherwise, partition p uses hardwares[hardware_placement[p]].
// If parallel_partitions is true and OpenMP is available in the build,
// per-partition solves may run in parallel.
double GED_cost(const std::vector<int>& partition_mapping_primary,
                const std::vector<int>& partition_mapping_extra,
                const CircuitGraph& circuit_graph,
                const std::vector<const TopologyGraph*>& hardwares,
                const std::vector<int>& hardware_placement = {},
                bool parallel_partitions = false);

}  // namespace GEDWeighted
