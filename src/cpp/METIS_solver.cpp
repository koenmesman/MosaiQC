#include "METIS_solver.hpp"

#include "circuit_graph.hpp"

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#if OPTCORE_HAS_METIS
#include <metis.h>
#else
using idx_t = int;
#endif

namespace py = pybind11;

namespace {

std::vector<int> validate_and_normalize_capacities(const std::vector<int>& capacities,
                                                   std::uint64_t required_capacity,
                                                   const char* context) {
    if (capacities.empty()) {
        throw std::invalid_argument(std::string(context) + ": capacities must be non-empty");
    }

    for (std::size_t i = 0; i < capacities.size(); ++i) {
        if (capacities[i] <= 0) {
            throw std::invalid_argument(std::string(context) + ": capacities must be > 0 (bad entry at index " +
                                        std::to_string(i) + ")");
        }
    }

    const int sum_caps = std::accumulate(capacities.begin(), capacities.end(), 0);
    if (static_cast<std::uint64_t>(sum_caps) < required_capacity) {
        throw std::invalid_argument(std::string(context) + ": sum(capacities) is below total graph node weight");
    }

    return capacities;
}

#if !OPTCORE_HAS_METIS
std::vector<int> partition_csr_with_metis(std::uint32_t num_nodes,
                                          const std::vector<std::uint32_t>&,
                                          const std::vector<std::uint32_t>&,
                                          const std::vector<idx_t>&,
                                          const std::vector<idx_t>&,
                                          const std::vector<int>&,
                                          std::uint64_t,
                                          double,
                                          int,
                                          const char* context) {
    (void)num_nodes;
    throw std::runtime_error(std::string(context) + ": METIS is not available in this build");
}
#else
std::vector<int> partition_csr_with_metis(std::uint32_t num_nodes,
                                          const std::vector<std::uint32_t>& row_ptr,
                                          const std::vector<std::uint32_t>& col_idx,
                                          const std::vector<idx_t>& vertex_weights,
                                          const std::vector<idx_t>& adj_weights,
                                          const std::vector<int>& normalized_capacities,
                                          std::uint64_t total_node_weight,
                                          double imbalance,
                                          int seed,
                                          const char* context) {
    if (imbalance < 1.0) {
        throw std::invalid_argument(std::string(context) + ": imbalance must be >= 1.0");
    }

    const idx_t n = static_cast<idx_t>(num_nodes);
    if (static_cast<std::uint32_t>(n) != num_nodes) {
        throw std::runtime_error(std::string(context) + ": idx_t overflow for graph nodes");
    }

    const std::size_t nparts_size = normalized_capacities.size();
    if (nparts_size == 1) {
        return std::vector<int>(static_cast<std::size_t>(n), 0);
    }
    if (num_nodes == 0u) {
        throw std::invalid_argument(std::string(context) + ": cannot partition an empty graph into multiple parts");
    }
    if (nparts_size > static_cast<std::size_t>(n)) {
        throw std::invalid_argument(std::string(context) + ": number of parts must be <= number of graph nodes");
    }

    std::vector<idx_t> xadj(row_ptr.size());
    for (std::size_t i = 0; i < row_ptr.size(); ++i) {
        xadj[i] = static_cast<idx_t>(row_ptr[i]);
    }

    std::vector<idx_t> adjncy(col_idx.size());
    for (std::size_t i = 0; i < col_idx.size(); ++i) {
        adjncy[i] = static_cast<idx_t>(col_idx[i]);
    }

    const idx_t ncon = 1;
    const idx_t nparts = static_cast<idx_t>(nparts_size);
    const int sum_caps = std::accumulate(normalized_capacities.begin(), normalized_capacities.end(), 0);

    std::vector<real_t> tpwgts(static_cast<std::size_t>(nparts) * static_cast<std::size_t>(ncon));
    for (std::size_t i = 0; i < normalized_capacities.size(); ++i) {
        tpwgts[i] = static_cast<real_t>(normalized_capacities[i]) / static_cast<real_t>(sum_caps);
    }

    const double capacity_slack = static_cast<double>(sum_caps) / static_cast<double>(total_node_weight);
    const double effective_imbalance = std::max(imbalance, capacity_slack);
    std::vector<real_t> ubvec(static_cast<std::size_t>(ncon), static_cast<real_t>(effective_imbalance));

    idx_t options[METIS_NOPTIONS];
    METIS_SetDefaultOptions(options);
    options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_CUT;
    options[METIS_OPTION_NUMBERING] = 0;
    options[METIS_OPTION_SEED] = static_cast<idx_t>(seed);

    std::vector<idx_t> part(static_cast<std::size_t>(n), 0);
    idx_t objval = 0;

    int status = METIS_PartGraphKway(
        const_cast<idx_t*>(&n),
        const_cast<idx_t*>(&ncon),
        xadj.data(),
        adjncy.data(),
        const_cast<idx_t*>(vertex_weights.empty() ? nullptr : vertex_weights.data()),
        nullptr, // vsize
        const_cast<idx_t*>(adj_weights.empty() ? nullptr : adj_weights.data()),
        const_cast<idx_t*>(&nparts),
        tpwgts.data(),
        ubvec.data(),
        options,
        &objval,
        part.data()
    );

    if (status != METIS_OK) {
        throw std::runtime_error(std::string(context) + ": METIS_PartGraphKway failed with status=" +
                                 std::to_string(status));
    }

    std::vector<int> out(part.size());
    for (std::size_t i = 0; i < part.size(); ++i) {
        out[i] = static_cast<int>(part[i]);
    }
    return out;
}
#endif

}  // namespace

std::vector<int> global_metis_solver(const CircuitGraph& graph,
                                     const std::vector<int>& capacities,
                                     double imbalance,
                                     int seed) {
    const auto& C = graph.connectivity();
    const auto normalized_capacities =
        validate_and_normalize_capacities(capacities, C.num_qubits, "global_metis_solver()");

#if !OPTCORE_HAS_METIS
    (void)graph;
    (void)normalized_capacities;
    (void)imbalance;
    (void)seed;
    throw std::runtime_error("global_metis_solver(): METIS is not available in this build");
#else
    std::vector<idx_t> adjwgt(C.weight.size());
    for (std::size_t i = 0; i < C.weight.size(); ++i) {
        adjwgt[i] = static_cast<idx_t>(C.weight[i]);
    }
    std::vector<std::uint32_t> col_idx(C.col_idx.size());
    for (std::size_t i = 0; i < C.col_idx.size(); ++i) {
        col_idx[i] = static_cast<std::uint32_t>(C.col_idx[i]);
    }

    return partition_csr_with_metis(
        C.num_qubits,
        C.row_ptr,
        col_idx,
        {},
        adjwgt,
        normalized_capacities,
        C.num_qubits,
        imbalance,
        seed,
        "global_metis_solver()");
#endif
}

std::vector<int> global_temporal_metis_solver(const CircuitGraph::TemporalGraph& graph,
                                              const std::vector<int>& capacities,
                                              double imbalance,
                                              int seed) {
    const std::uint64_t total_node_weight =
        std::accumulate(graph.node_weight.begin(), graph.node_weight.end(), std::uint64_t{0});
    const auto normalized_capacities =
        validate_and_normalize_capacities(capacities, total_node_weight, "global_temporal_metis_solver()");

#if !OPTCORE_HAS_METIS
    (void)graph;
    (void)normalized_capacities;
    (void)imbalance;
    (void)seed;
    throw std::runtime_error("global_temporal_metis_solver(): METIS is not available in this build");
#else
    std::vector<idx_t> vwgt(graph.node_weight.size());
    for (std::size_t i = 0; i < graph.node_weight.size(); ++i) {
        vwgt[i] = static_cast<idx_t>(graph.node_weight[i]);
    }
    std::vector<idx_t> adjwgt(graph.edge_weight.size(), static_cast<idx_t>(1));
    return partition_csr_with_metis(graph.num_nodes,
                                    graph.row_ptr,
                                    graph.col_idx,
                                    vwgt,
                                    adjwgt,
                                    normalized_capacities,
                                    total_node_weight,
                                    imbalance,
                                    seed,
                                    "global_temporal_metis_solver()");
#endif
}

void bind_metis_solver(py::module_& m) {
    m.def(
        "global_METIS_solver",
        &global_metis_solver,
        py::arg("graph"),
        py::arg("capacities"),
        py::arg("imbalance") = 1.03,
        py::arg("seed") = 42,
        "Partition qubits of a CircuitGraph into blocks using METIS and target capacities."
    );
    m.def(
        "global_temporal_METIS_solver",
        &global_temporal_metis_solver,
        py::arg("temporal_graph"),
        py::arg("capacities"),
        py::arg("imbalance") = 1.03,
        py::arg("seed") = 42,
        "Partition temporal 2Q-gate nodes using METIS and target capacities."
    );
}
