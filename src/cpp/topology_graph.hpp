#pragma once

#include <pybind11/pybind11.h>

#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

namespace py = pybind11;

// Mirror the Python TopologyGraph node/edge attributes.
struct NodeErrorRates {
    double readout = 0.0;
    double x = 0.0;
    double reset = 0.0;
    double sum_error = 0.0;
};

struct EdgeErrorRates {
    double cx = 0.0;
};

struct PairHash {
    std::size_t operator()(const std::pair<int, int>& p) const noexcept {
        // Good enough for small integer keys.
        return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(p.first)) << 32) ^
               static_cast<std::uint32_t>(p.second);
    }
};

class TopologyGraph {
public:
    struct WeightedGedCache {
        int n = 0;
        std::vector<double> node_x;     // log(1-error) per hardware node
        std::vector<int> degree;        // undirected degree per hardware node
        std::vector<double> edge_score; // dense matrix n*n; -inf means absent
        std::vector<int> order;         // candidate iteration order
        double best_edge_score = 0.0;
    };

    // Exactly one of backend/noise_model must be provided.
    TopologyGraph(py::object backend = py::none(), py::object noise_model = py::none());

    // Recompute nodes/edges from the stored noise model.
    void rebuild();

    // Basic accessors (C++ friendly).
    std::vector<int> qubits() const;
    bool has_qubit(int q) const;
    NodeErrorRates node(int q) const;

    std::vector<std::pair<int, int>> directed_edges() const;
    bool has_cx(int q0, int q1) const;
    EdgeErrorRates edge(int q0, int q1) const;
    const WeightedGedCache& weighted_ged_cache() const noexcept { return weighted_ged_cache_; }

    // Convenience for Python interoperability.
    // Builds a networkx.DiGraph with the same node/edge attributes.
    py::object to_networkx() const;

private:
    py::object noise_model_;  // keep the Python noise model alive
    std::unordered_map<int, NodeErrorRates> nodes_;
    std::unordered_map<std::pair<int, int>, EdgeErrorRates, PairHash> edges_;
    WeightedGedCache weighted_ged_cache_;

    void build_from_noise_model_(const py::object& noise_model);
    void rebuild_weighted_ged_cache_();
};

// Binder implemented in topology_graph.cpp
void bind_topology_graph(py::module_& m);
