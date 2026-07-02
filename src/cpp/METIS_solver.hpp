#pragma once

#include "circuit_graph.hpp"

#include <vector>

namespace pybind11 {
class module_;
}

// Partition CircuitGraph qubits into uneven parts based on capacities.
// Returns a vector of length num_qubits with block id per qubit.
std::vector<int> global_metis_solver(const CircuitGraph& graph,
                                     const std::vector<int>& capacities,
                                     double imbalance = 1.03,
                                     int seed = 42);

// Partition temporal 2Q-gate nodes into uneven parts based on capacities.
// Returns a vector of length temporal_graph.num_nodes with block id per gate node.
std::vector<int> global_temporal_metis_solver(const CircuitGraph::TemporalGraph& graph,
                                              const std::vector<int>& capacities,
                                              double imbalance = 1.03,
                                              int seed = 42);

void bind_metis_solver(pybind11::module_& m);
