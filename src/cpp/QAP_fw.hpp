#pragma once

#include <vector>

class CircuitGraph;
class TopologyGraph;

namespace QAP_fw {

struct QAPEvaluation {
    double score = 0.0;
    double cost = 0.0;
};

// Frank-Wolfe relaxation for weighted graph matching per partition.
//
// For each partition, we construct:
// - hardware graph H: node/edge weights from TopologyGraph (log_one_minus_error values)
// - circuit graph C: integer node/edge weights
//
// We maximize overlap score:
//   sum(node products) + sum(edge products)
// under injective node assignment between the two partition graphs
// (mapping always from smaller graph into larger graph).
//
// Missing hardware edges are allowed with a finite penalty that is worse than
// present couplings in the edge product.
//
// Returns 1 - mean squared placement infidelity, where each partition placement
// fidelity is exp(maximized log-success score). Higher is better, and
// 1 - returned_score equals:
//   sum((1 - placement_fidelity)^2) / num_partitions
// Optional normalization flags divide the maximized partition log-success score.
// If one of the legacy normalization flags is true, it overrides
// sqrt_weight_log_cost_score. With sqrt_weight_log_cost_score, cost is the mean
// normalized log-domain cost; otherwise cost is 1 - score.
QAPEvaluation QAP_eval_fw(const std::vector<int>& partition_mapping_primary,
                          const std::vector<int>& partition_mapping_extra,
                          const CircuitGraph& circuit_graph,
                          const std::vector<const TopologyGraph*>& hardwares,
                          const std::vector<int>& hardware_placement = {},
                          int max_iters = 80,
                          double tol = 1e-5,
                          bool parallel_partitions = false,
                          bool use_routed_hardware_edges = false,
                          bool mean_product_score = false,
                          bool circuit_weighted_mean_score = false,
                          bool gate_count_normalized_score = false,
                          bool sqrt_weight_log_cost_score = true);

double QAP_cost_fw(const std::vector<int>& partition_mapping_primary,
                   const std::vector<int>& partition_mapping_extra,
                   const CircuitGraph& circuit_graph,
                   const std::vector<const TopologyGraph*>& hardwares,
                   const std::vector<int>& hardware_placement = {},
                   int max_iters = 80,
                   double tol = 1e-5,
                   bool parallel_partitions = false,
                   bool use_routed_hardware_edges = false,
                   bool mean_product_score = false,
                   bool circuit_weighted_mean_score = false,
                   bool gate_count_normalized_score = false,
                   bool sqrt_weight_log_cost_score = true);

}  // namespace QAP_fw
