#include "circuit_graph.hpp"

#include <stdexcept>
#include <utility>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

// pybind is only needed in the .cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace {

inline void bump_stamp_or_reset(std::uint32_t& stamp, std::vector<std::uint32_t>& stamps) {
    // stamp==0 is reserved; if we wrap, clear stamps to 0 and restart at 1.
    ++stamp;
    if (stamp == 0u) {
        std::fill(stamps.begin(), stamps.end(), 0u);
        stamp = 1u;
    }
}

inline bool shares_any_block(int up, int ux, int vp, int vx) {
    return (up == vp) || (ux >= 0 && ux == vp) || (vx >= 0 && up == vx) || (ux >= 0 && vx >= 0 && ux == vx);
}

inline bool is_parametrized_controlled_rotation(std::string_view gate_name) {
    return gate_name == "crx" || gate_name == "cry" || gate_name == "crz";
}

double op_weight(std::string_view gate_name, double angle = 0.0) {
    if (gate_name == "cx") {
        return std::log(9.0);
    }
    if (gate_name == "swap") {
        return std::log(49.0);
    }
    if (is_parametrized_controlled_rotation(gate_name)) {
        const double x = 1.0 + std::abs(2.0 * std::sin(angle));
        return 2 * std::log(x);
    }
    return 0.0;
}

double extract_rotation_angle(const py::object& instruction, const std::string& gate_name, std::uint32_t instruction_index) {
    if (!py::hasattr(instruction, "params")) {
        throw std::runtime_error(
            "Gate '" + gate_name + "' at instruction index " + std::to_string(instruction_index) +
            " is missing params for rotation angle extraction");
    }
    py::object params = instruction.attr("params");
    if (py::len(params) < 1) {
        throw std::runtime_error(
            "Gate '" + gate_name + "' at instruction index " + std::to_string(instruction_index) +
            " has empty params; expected rotation angle at params[0]");
    }

    py::object angle_obj = params.attr("__getitem__")(0);
    try {
        const double angle = py::float_(angle_obj).cast<double>();
        if (!std::isfinite(angle)) {
            throw std::runtime_error(
                "Gate '" + gate_name + "' at instruction index " + std::to_string(instruction_index) +
                " has non-finite rotation angle");
        }
        return angle;
    } catch (const py::cast_error&) {
        throw std::runtime_error(
            "Gate '" + gate_name + "' at instruction index " + std::to_string(instruction_index) +
            " has non-numeric rotation angle; bind parameters before building CircuitGraph");
    }
}

} // namespace

double CircuitGraph::ConnectivityGraph::count(std::uint32_t q, std::uint32_t r) const {
    if (q >= num_qubits || r >= num_qubits) {
        throw std::out_of_range("ConnectivityGraph.count(): qubit out of range");
    }
    const auto b = row_ptr[static_cast<std::size_t>(q)];
    const auto e = row_ptr[static_cast<std::size_t>(q) + 1];
    // Binary search in the sorted neighbor slice.
    auto it = std::lower_bound(
        col_idx.begin() + static_cast<std::ptrdiff_t>(b),
        col_idx.begin() + static_cast<std::ptrdiff_t>(e),
        static_cast<std::uint16_t>(r));
    if (it == col_idx.begin() + static_cast<std::ptrdiff_t>(e) || *it != static_cast<std::uint16_t>(r)) {
        return 0.0;
    }
    const auto k = static_cast<std::size_t>(std::distance(col_idx.begin(), it));
    return weight[k];
}

std::int64_t CircuitGraph::ConnectivityGraph::edge_identifier(std::uint32_t q, std::uint32_t r) const {
    if (q >= num_qubits || r >= num_qubits) {
        throw std::out_of_range("ConnectivityGraph.edge_identifier(): qubit out of range");
    }
    const auto b = row_ptr[static_cast<std::size_t>(q)];
    const auto e = row_ptr[static_cast<std::size_t>(q) + 1];
    auto it = std::lower_bound(
        col_idx.begin() + static_cast<std::ptrdiff_t>(b),
        col_idx.begin() + static_cast<std::ptrdiff_t>(e),
        static_cast<std::uint16_t>(r));
    if (it == col_idx.begin() + static_cast<std::ptrdiff_t>(e) || *it != static_cast<std::uint16_t>(r)) {
        return -1;
    }
    const auto k = static_cast<std::size_t>(std::distance(col_idx.begin(), it));
    return static_cast<std::int64_t>(edge_id[k]);
}

std::vector<std::uint32_t> CircuitGraph::ConnectivityGraph::edge_gate_indices_by_id(std::uint32_t id) const {
    if (id >= num_edge_ids()) {
        throw std::out_of_range("ConnectivityGraph.edge_gate_indices_by_id(): invalid edge id");
    }
    const auto b = edge_original_row_ptr[static_cast<std::size_t>(id)];
    const auto e = edge_original_row_ptr[static_cast<std::size_t>(id) + 1];
    return std::vector<std::uint32_t>(
        edge_original_indices.begin() + static_cast<std::ptrdiff_t>(b),
        edge_original_indices.begin() + static_cast<std::ptrdiff_t>(e));
}

std::vector<std::uint32_t> CircuitGraph::ConnectivityGraph::edge_gate_indices_by_qubits(std::uint32_t q,
                                                                                        std::uint32_t r) const {
    const std::int64_t id = edge_identifier(q, r);
    if (id < 0) return {};
    return edge_gate_indices_by_id(static_cast<std::uint32_t>(id));
}

void CircuitGraph::TemporalGateEvents::build(std::uint32_t Q,
                                             const std::vector<Operation>& merged_ops,
                                             const std::vector<std::uint32_t>& mult,
                                             const std::vector<double>& event_weight,
                                             const std::vector<std::vector<std::uint32_t>>& event_original_gate_indices) {
    if (Q > 65535u) {
        throw std::invalid_argument("TemporalGateEvents supports up to 65535 qubits (uint16 endpoints)");
    }
    if (merged_ops.size() != mult.size()) {
        throw std::invalid_argument("TemporalGateEvents.build(): ops and multiplicity must have same length");
    }
    if (merged_ops.size() != event_weight.size()) {
        throw std::invalid_argument("TemporalGateEvents.build(): ops and event_weight must have same length");
    }
    if (merged_ops.size() != event_original_gate_indices.size()) {
        throw std::invalid_argument("TemporalGateEvents.build(): ops and event_original_gate_indices must have same length");
    }
    num_qubits = Q;
    num_events = static_cast<std::uint32_t>(merged_ops.size());

    q0.resize(num_events);
    q1.resize(num_events);
    multiplicity.assign(mult.begin(), mult.end());
    weight.assign(event_weight.begin(), event_weight.end());

    // Count incidences per qubit.
    std::vector<std::uint32_t> deg(num_qubits, 0);
    for (const auto& op : merged_ops) {
        if (op.q0 >= num_qubits || op.q1 >= num_qubits) {
            throw std::out_of_range("TemporalGateEvents.build(): qubit index out of range");
        }
        if (op.q0 == op.q1) {
            throw std::invalid_argument("TemporalGateEvents.build(): 2Q gate cannot have identical qubits");
        }
        deg[op.q0] += 1;
        deg[op.q1] += 1;
    }

    row_ptr.assign(static_cast<std::size_t>(num_qubits) + 1, 0);
    for (std::uint32_t q = 0; q < num_qubits; ++q) {
        row_ptr[static_cast<std::size_t>(q) + 1] = row_ptr[static_cast<std::size_t>(q)] + deg[q];
    }
    const std::uint32_t total = row_ptr[static_cast<std::size_t>(num_qubits)];
    partners.assign(total, 0);
    partner_weight.assign(total, 0.0);
    partner_event_id.assign(total, 0);

    event_original_row_ptr.assign(static_cast<std::size_t>(num_events) + 1u, 0u);
    std::size_t total_original = 0;
    for (std::size_t eidx = 0; eidx < event_original_gate_indices.size(); ++eidx) {
        event_original_row_ptr[eidx] = static_cast<std::uint32_t>(total_original);
        total_original += event_original_gate_indices[eidx].size();
    }
    event_original_row_ptr[static_cast<std::size_t>(num_events)] = static_cast<std::uint32_t>(total_original);
    event_original_indices.clear();
    event_original_indices.reserve(total_original);
    for (const auto& idxs : event_original_gate_indices) {
        event_original_indices.insert(event_original_indices.end(), idxs.begin(), idxs.end());
    }

    // Fill partners in time order using per-qubit cursors.
    std::vector<std::uint32_t> cursor = row_ptr;
    for (std::uint32_t e = 0; e < num_events; ++e) {
        const auto& op = merged_ops[static_cast<std::size_t>(e)];
        const auto a = op.q0;
        const auto b = op.q1;
        const double w = weight[static_cast<std::size_t>(e)];
        q0[e] = static_cast<std::uint16_t>(a);
        q1[e] = static_cast<std::uint16_t>(b);

        const auto ia = cursor[a]++;
        const auto ib = cursor[b]++;
        partners[static_cast<std::size_t>(ia)] = static_cast<std::uint16_t>(b);
        partners[static_cast<std::size_t>(ib)] = static_cast<std::uint16_t>(a);
        partner_weight[static_cast<std::size_t>(ia)] = w;
        partner_weight[static_cast<std::size_t>(ib)] = w;
        partner_event_id[static_cast<std::size_t>(ia)] = e;
        partner_event_id[static_cast<std::size_t>(ib)] = e;
    }
}

std::uint32_t CircuitGraph::TemporalGateEvents::timeline_len(std::uint32_t q) const {
    if (q >= num_qubits) throw std::out_of_range("TemporalGateEvents.timeline_len(): qubit out of range");
    return row_ptr[static_cast<std::size_t>(q) + 1] - row_ptr[static_cast<std::size_t>(q)];
}

void CircuitGraph::build_temporal_() {
    // Merge consecutive 2Q gates that act on the same *unordered* qubit pair.
    // This treats (a,b) and (b,a) as identical for the purpose of temporal nodes.
    //
    // Since ops_ contains only 2Q gates (we filtered out 1Q in from_qiskit),
    // "consecutive" refers to adjacent entries in ops_.
    std::vector<Operation> merged;
    std::vector<std::uint32_t> mult;
    std::vector<double> event_weight;
    std::vector<std::vector<std::uint32_t>> event_original_gate_indices;
    merged.reserve(ops_.size());
    mult.reserve(ops_.size());
    event_weight.reserve(ops_.size());
    event_original_gate_indices.reserve(ops_.size());

    auto unordered_key = [](std::uint32_t a, std::uint32_t b) -> std::uint64_t {
        const std::uint32_t lo = (a < b) ? a : b;
        const std::uint32_t hi = (a < b) ? b : a;
        return (static_cast<std::uint64_t>(lo) << 32) | static_cast<std::uint64_t>(hi);
    };

    std::uint64_t last_key = 0;
    CircuitGraph::GateId last_gate_kind = 0;
    bool has_last = false;
    for (const auto& op : ops_) {
        const double w = op_weight(gate_name(op.gate), op.angle);
        const std::uint64_t k = unordered_key(op.q0, op.q1);
        if (!has_last || k != last_key || op.gate != last_gate_kind) {
            merged.push_back(op);
            mult.push_back(1);
            event_weight.push_back(w);
            event_original_gate_indices.push_back({op.original_index});
            last_key = k;
            last_gate_kind = op.gate;
            has_last = true;
        } else {
            // same unordered pair as previous: extend multiplicity
            mult.back() += 1;
            event_weight.back() += w;
            event_original_gate_indices.back().push_back(op.original_index);
        }
    }

    temporal_.build(num_qubits_, merged, mult, event_weight, event_original_gate_indices);
}

void CircuitGraph::build_temporal_graph_() {
    temporal_graph_.num_nodes = static_cast<std::uint32_t>(ops_.size());
    const std::uint32_t n = temporal_graph_.num_nodes;

    std::vector<std::uint32_t> connectivity_degree(num_qubits_, 0u);
    std::unordered_set<std::uint64_t> connectivity_edges;
    connectivity_edges.reserve(ops_.size() * 2u + 1u);
    auto unordered_key = [](std::uint32_t a, std::uint32_t b) -> std::uint64_t {
        const std::uint32_t lo = (a < b) ? a : b;
        const std::uint32_t hi = (a < b) ? b : a;
        return (static_cast<std::uint64_t>(lo) << 32) | static_cast<std::uint64_t>(hi);
    };
    for (const auto& op : ops_) {
        if (op.q0 >= num_qubits_ || op.q1 >= num_qubits_) {
            throw std::out_of_range("build_temporal_graph_(): qubit index out of range");
        }
        if (op.q0 == op.q1) {
            throw std::invalid_argument("build_temporal_graph_(): 2Q gate cannot have identical qubits");
        }
        if (connectivity_edges.insert(unordered_key(op.q0, op.q1)).second) {
            connectivity_degree[static_cast<std::size_t>(op.q0)] += 1u;
            connectivity_degree[static_cast<std::size_t>(op.q1)] += 1u;
        }
    }

    temporal_graph_.node_weight.resize(static_cast<std::size_t>(n));
    temporal_graph_.original_index.resize(static_cast<std::size_t>(n));
    temporal_graph_.q0.resize(static_cast<std::size_t>(n));
    temporal_graph_.q1.resize(static_cast<std::size_t>(n));

    struct Edge {
        std::uint32_t u;
        std::uint32_t v;
    };

    std::vector<Edge> edges;
    edges.reserve(ops_.size() * 2u);
    std::vector<std::uint32_t> last_on_qubit(num_qubits_, kNoOperationIndex);

    for (std::uint32_t node = 0; node < n; ++node) {
        const auto& op = ops_[static_cast<std::size_t>(node)];
        if (op.q0 >= num_qubits_ || op.q1 >= num_qubits_) {
            throw std::out_of_range("build_temporal_graph_(): qubit index out of range");
        }
        if (op.q0 == op.q1) {
            throw std::invalid_argument("build_temporal_graph_(): 2Q gate cannot have identical qubits");
        }

        temporal_graph_.original_index[static_cast<std::size_t>(node)] = op.original_index;
        temporal_graph_.q0[static_cast<std::size_t>(node)] = op.q0;
        temporal_graph_.q1[static_cast<std::size_t>(node)] = op.q1;
        temporal_graph_.node_weight[static_cast<std::size_t>(node)] =
            connectivity_degree[static_cast<std::size_t>(op.q0)] +
            connectivity_degree[static_cast<std::size_t>(op.q1)];

        auto add_temporal_edge = [&](std::uint32_t previous) {
            if (previous == kNoOperationIndex || previous == node) return;
            const std::uint32_t u = (previous < node) ? previous : node;
            const std::uint32_t v = (previous < node) ? node : previous;
            edges.push_back(Edge{u, v});
        };

        add_temporal_edge(last_on_qubit[static_cast<std::size_t>(op.q0)]);
        add_temporal_edge(last_on_qubit[static_cast<std::size_t>(op.q1)]);
        last_on_qubit[static_cast<std::size_t>(op.q0)] = node;
        last_on_qubit[static_cast<std::size_t>(op.q1)] = node;
    }

    std::sort(edges.begin(), edges.end(), [](const Edge& lhs, const Edge& rhs) {
        if (lhs.u != rhs.u) return lhs.u < rhs.u;
        return lhs.v < rhs.v;
    });
    edges.erase(std::unique(edges.begin(), edges.end(), [](const Edge& lhs, const Edge& rhs) {
                    return lhs.u == rhs.u && lhs.v == rhs.v;
                }),
                edges.end());

    std::vector<std::uint32_t> deg(static_cast<std::size_t>(n), 0u);
    for (const auto& edge : edges) {
        deg[static_cast<std::size_t>(edge.u)] += 1u;
        deg[static_cast<std::size_t>(edge.v)] += 1u;
    }

    temporal_graph_.row_ptr.assign(static_cast<std::size_t>(n) + 1u, 0u);
    for (std::uint32_t node = 0; node < n; ++node) {
        temporal_graph_.row_ptr[static_cast<std::size_t>(node) + 1u] =
            temporal_graph_.row_ptr[static_cast<std::size_t>(node)] + deg[static_cast<std::size_t>(node)];
    }

    const std::uint32_t nnz = temporal_graph_.row_ptr[static_cast<std::size_t>(n)];
    temporal_graph_.col_idx.assign(static_cast<std::size_t>(nnz), 0u);
    temporal_graph_.edge_weight.assign(static_cast<std::size_t>(nnz), 1u);

    std::vector<std::uint32_t> cursor = temporal_graph_.row_ptr;
    for (const auto& edge : edges) {
        std::uint32_t k = cursor[static_cast<std::size_t>(edge.u)]++;
        temporal_graph_.col_idx[static_cast<std::size_t>(k)] = edge.v;
        k = cursor[static_cast<std::size_t>(edge.v)]++;
        temporal_graph_.col_idx[static_cast<std::size_t>(k)] = edge.u;
    }

    for (std::uint32_t node = 0; node < n; ++node) {
        const auto b = temporal_graph_.row_ptr[static_cast<std::size_t>(node)];
        const auto e = temporal_graph_.row_ptr[static_cast<std::size_t>(node) + 1u];
        std::sort(temporal_graph_.col_idx.begin() + static_cast<std::ptrdiff_t>(b),
                  temporal_graph_.col_idx.begin() + static_cast<std::ptrdiff_t>(e));
    }
}

std::uint32_t CircuitGraph::count_temporal_cross_edges(Qubit common_qubit,
                                                      const std::vector<Qubit>& set_a,
                                                      const std::vector<Qubit>& set_b) const {
    if (common_qubit >= num_qubits_) {
        throw std::out_of_range("count_temporal_cross_edges(): common qubit out of range");
    }
    if (mark_stamp_.size() != num_qubits_) {
        // Should not happen, but keep the method robust if called on a moved-from object.
        mark_stamp_.assign(num_qubits_, 0);
        mark_label_.assign(num_qubits_, 0);
        mark_cur_stamp_ = 1;
    }

    bump_stamp_or_reset(mark_cur_stamp_, mark_stamp_);
    const std::uint32_t stamp = mark_cur_stamp_;

    // Mark A as 1, B as 2. If overlap exists, B wins (arbitrary but deterministic).
    for (Qubit q : set_a) {
        if (q >= num_qubits_) continue;
        mark_stamp_[q] = stamp;
        mark_label_[q] = 1;
    }
    for (Qubit q : set_b) {
        if (q >= num_qubits_) continue;
        mark_stamp_[q] = stamp;
        mark_label_[q] = 2;
    }
    // Ensure the common qubit never matches as a partner by accident.
    mark_stamp_[common_qubit] = stamp;
    mark_label_[common_qubit] = 0;

    const auto b = temporal_.row_ptr[static_cast<std::size_t>(common_qubit)];
    const auto e = temporal_.row_ptr[static_cast<std::size_t>(common_qubit) + 1];
    if (e <= b + 1) return 0u;

    auto label_of = [&](std::uint32_t q) -> std::uint8_t {
        return (q < num_qubits_ && mark_stamp_[q] == stamp) ? mark_label_[q] : 0;
    };

    std::uint32_t count = 0;
    std::uint8_t prev_label = label_of(temporal_.partners[static_cast<std::size_t>(b)]);
    for (std::uint32_t idx = b + 1; idx < e; ++idx) {
        const std::uint32_t cur_partner = temporal_.partners[static_cast<std::size_t>(idx)];
        const std::uint8_t cur_label = label_of(cur_partner);
        if ((prev_label == 1 && cur_label == 2) || (prev_label == 2 && cur_label == 1)) {
            ++count;
        }
        prev_label = cur_label;
    }
    return count;
}

std::vector<CircuitGraph::Operation> CircuitGraph::cut_edges(const std::vector<int>& primary,
                                                            const std::vector<int>& extra_block) const {
    if (primary.size() != static_cast<std::size_t>(num_qubits_)) {
        throw std::invalid_argument("cut_edges(): primary placement length must match num_qubits");
    }
    if (!extra_block.empty() && extra_block.size() != static_cast<std::size_t>(num_qubits_)) {
        throw std::invalid_argument("cut_edges(): extra_block length must match num_qubits when provided");
    }

    std::vector<Operation> cuts;
    cuts.reserve(ops_.size());
    for (const auto& op : ops_) {
        const int up = primary[static_cast<std::size_t>(op.q0)];
        const int vp = primary[static_cast<std::size_t>(op.q1)];
        const int ux = extra_block.empty() ? -1 : extra_block[static_cast<std::size_t>(op.q0)];
        const int vx = extra_block.empty() ? -1 : extra_block[static_cast<std::size_t>(op.q1)];
        if (!shares_any_block(up, ux, vp, vx)) {
            cuts.push_back(op);
        }
    }
    return cuts;
}

void CircuitGraph::build_connectivity_() {
    // Build undirected weighted connectivity from 2Q ops.
    // We store it symmetrically (both directions) in CSR.
    if (num_qubits_ > 65535u) {
        throw std::invalid_argument("ConnectivityGraph supports up to 65535 qubits (uint16 neighbors)");
    }

    struct WeightedArc {
        std::uint64_t key;
        double weight;
        std::uint32_t original_index;
    };

    // Pack (src,dst) into a 64-bit key for sorting: high 32 bits=src, low 32 bits=dst.
    std::vector<WeightedArc> arcs;
    arcs.reserve(ops_.size() * 2);
    for (const auto& op : ops_) {
        const std::uint32_t a = op.q0;
        const std::uint32_t b = op.q1;
        if (a >= num_qubits_ || b >= num_qubits_) {
            throw std::out_of_range("build_connectivity_(): qubit index out of range");
        }
        if (a == b) continue;
        const double w = op_weight(gate_name(op.gate), op.angle);
        if (w <= 0.0) continue;
        arcs.push_back(WeightedArc{
            (static_cast<std::uint64_t>(a) << 32) | static_cast<std::uint64_t>(b), w, op.original_index});
        arcs.push_back(WeightedArc{
            (static_cast<std::uint64_t>(b) << 32) | static_cast<std::uint64_t>(a), w, op.original_index});
    }
    std::sort(arcs.begin(), arcs.end(), [](const WeightedArc& lhs, const WeightedArc& rhs) {
        if (lhs.key != rhs.key) return lhs.key < rhs.key;
        return lhs.original_index < rhs.original_index;
    });

    // Compress into unique (src,dst,total_weight).
    struct Entry {
        std::uint32_t src;
        std::uint32_t dst;
        double w;
        std::uint32_t id;
    };
    std::vector<Entry> entries;
    entries.reserve(arcs.size());

    std::unordered_map<std::uint64_t, std::uint32_t> undirected_edge_id;
    undirected_edge_id.reserve(arcs.size() / 2u + 1u);
    std::vector<std::uint16_t> edge_q0;
    std::vector<std::uint16_t> edge_q1;
    edge_q0.reserve(arcs.size() / 2u);
    edge_q1.reserve(arcs.size() / 2u);
    std::vector<std::vector<std::uint32_t>> edge_sources;
    edge_sources.reserve(arcs.size() / 2u);
    auto undirected_key = [](std::uint32_t a, std::uint32_t b) -> std::uint64_t {
        const std::uint32_t lo = (a < b) ? a : b;
        const std::uint32_t hi = (a < b) ? b : a;
        return (static_cast<std::uint64_t>(lo) << 32) | static_cast<std::uint64_t>(hi);
    };

    for (std::size_t i = 0; i < arcs.size();) {
        const std::uint64_t key = arcs[i].key;
        std::size_t j = i + 1;
        double total_weight = arcs[i].weight;
        std::vector<std::uint32_t> source_indices;
        source_indices.push_back(arcs[i].original_index);
        while (j < arcs.size() && arcs[j].key == key) {
            total_weight += arcs[j].weight;
            source_indices.push_back(arcs[j].original_index);
            ++j;
        }
        const std::uint32_t src = static_cast<std::uint32_t>(key >> 32);
        const std::uint32_t dst = static_cast<std::uint32_t>(key & 0xFFFFFFFFu);
        const std::uint64_t ukey = undirected_key(src, dst);
        auto id_it = undirected_edge_id.find(ukey);
        std::uint32_t eid = 0;
        if (id_it == undirected_edge_id.end()) {
            eid = static_cast<std::uint32_t>(edge_q0.size());
            undirected_edge_id.emplace(ukey, eid);
            edge_q0.push_back(static_cast<std::uint16_t>((src < dst) ? src : dst));
            edge_q1.push_back(static_cast<std::uint16_t>((src < dst) ? dst : src));
            auto unique_end = std::unique(source_indices.begin(), source_indices.end());
            source_indices.erase(unique_end, source_indices.end());
            edge_sources.push_back(std::move(source_indices));
        } else {
            eid = id_it->second;
        }
        entries.push_back(Entry{src, dst, total_weight, eid});
        i = j;
    }

    // Row counts (#neighbors) per src.
    std::vector<std::uint32_t> deg(num_qubits_, 0);
    for (const auto& e : entries) {
        deg[e.src] += 1;
    }

    connectivity_.num_qubits = num_qubits_;
    connectivity_.node_weight = oneq_counts_;
    connectivity_.row_ptr.assign(static_cast<std::size_t>(num_qubits_) + 1, 0);
    for (std::uint32_t q = 0; q < num_qubits_; ++q) {
        connectivity_.row_ptr[static_cast<std::size_t>(q) + 1] = connectivity_.row_ptr[static_cast<std::size_t>(q)] + deg[q];
    }
    const std::uint32_t nnz = connectivity_.row_ptr[static_cast<std::size_t>(num_qubits_)];
    connectivity_.col_idx.assign(nnz, 0);
    connectivity_.weight.assign(nnz, 0.0);
    connectivity_.edge_id.assign(nnz, 0u);
    connectivity_.edge_q0 = std::move(edge_q0);
    connectivity_.edge_q1 = std::move(edge_q1);
    connectivity_.edge_original_row_ptr.assign(connectivity_.edge_q0.size() + 1u, 0u);
    std::size_t total_sources = 0;
    for (std::size_t eidx = 0; eidx < edge_sources.size(); ++eidx) {
        connectivity_.edge_original_row_ptr[eidx] = static_cast<std::uint32_t>(total_sources);
        total_sources += edge_sources[eidx].size();
    }
    connectivity_.edge_original_row_ptr[edge_sources.size()] = static_cast<std::uint32_t>(total_sources);
    connectivity_.edge_original_indices.clear();
    connectivity_.edge_original_indices.reserve(total_sources);
    for (const auto& srcs : edge_sources) {
        connectivity_.edge_original_indices.insert(connectivity_.edge_original_indices.end(), srcs.begin(), srcs.end());
    }

    // Fill CSR. Since entries are sorted by (src,dst), each row is naturally sorted by dst.
    std::vector<std::uint32_t> cursor = connectivity_.row_ptr; // copy
    for (const auto& e : entries) {
        const std::uint32_t k = cursor[e.src]++;
        connectivity_.col_idx[static_cast<std::size_t>(k)] = static_cast<std::uint16_t>(e.dst);
        connectivity_.weight[static_cast<std::size_t>(k)] = e.w;
        connectivity_.edge_id[static_cast<std::size_t>(k)] = e.id;
    }
}

CircuitGraph::CircuitGraph(std::uint32_t num_qubits,
                           std::vector<Operation> ops,
                           std::vector<std::string> gate_names)
    : num_qubits_(num_qubits),
      ops_(std::move(ops)),
      oneq_counts_(num_qubits, 0),
      gate_names_(std::move(gate_names)) {
    // Rebuild reverse map for consistency
    gate_to_id_.reserve(gate_names_.size());
    for (GateId id = 0; id < static_cast<GateId>(gate_names_.size()); ++id) {
        gate_to_id_.emplace(gate_names_[id], id);
    }
    for (std::size_t i = 0; i < ops_.size(); ++i) {
        if (ops_[i].original_index == kNoOperationIndex) {
            ops_[i].original_index = static_cast<std::uint32_t>(i);
        }
    }

    // Build temporal connectivity from provided ops_
    build_temporal_();
    build_temporal_graph_();

    // Build connectivity graph from provided ops_
    build_connectivity_();

    // Init membership marking buffers
    mark_stamp_.assign(num_qubits_, 0);
    mark_label_.assign(num_qubits_, 0);
    mark_cur_stamp_ = 1;
}

CircuitGraph::GateId CircuitGraph::intern_gate_(const std::string& name) {
    auto it = gate_to_id_.find(name);
    if (it != gate_to_id_.end()) return it->second;

    GateId id = static_cast<GateId>(gate_names_.size());
    gate_names_.push_back(name);
    gate_to_id_.emplace(gate_names_.back(), id);
    return id;
}

const std::string& CircuitGraph::gate_name(GateId id) const {
    if (id >= gate_names_.size()) {
        throw std::out_of_range("gate_name(): invalid gate id");
    }
    return gate_names_[id];
}

CircuitGraph CircuitGraph::from_qiskit(py::handle qc_handle) {
    py::object qc = py::reinterpret_borrow<py::object>(qc_handle);

    // Read number of qubits
    std::uint32_t num_qubits = qc.attr("num_qubits").cast<std::uint32_t>();

    CircuitGraph out(num_qubits, /*ops=*/{}, /*gate_names=*/{});
    out.ops_.reserve(128); // small guess; will grow if needed
    out.oneq_counts_.assign(num_qubits, 0);

    // Qiskit: qc.data is iterable of (instruction, qargs, cargs)
    py::iterable data = qc.attr("data");

    std::uint32_t instruction_index = 0;
    for (py::handle item : data) {
        py::object inst;
        py::object qargs;

        py::object item_obj = py::reinterpret_borrow<py::object>(item);
        if (py::isinstance<py::tuple>(item_obj)) {
            // Legacy Qiskit: (inst, qargs, cargs)
            py::tuple t = item_obj.cast<py::tuple>();
            if (t.size() < 2) {
                ++instruction_index;
                continue;
            }
            inst = py::reinterpret_borrow<py::object>(t[0]);
            qargs = py::reinterpret_borrow<py::object>(t[1]);
        } else if (py::hasattr(item_obj, "operation") && py::hasattr(item_obj, "qubits")) {
            // Qiskit CircuitInstruction
            inst = item_obj.attr("operation");
            qargs = item_obj.attr("qubits");
        } else if (py::hasattr(item_obj, "instruction") && py::hasattr(item_obj, "qargs")) {
            inst = item_obj.attr("instruction");
            qargs = item_obj.attr("qargs");
        } else {
            ++instruction_index;
            continue;
        }

        std::size_t nq = py::len(qargs);
        if (nq == 1) {
            py::object q0_obj = qargs.attr("__getitem__")(0);
            auto q0 = q0_obj.attr("_index").cast<std::int64_t>();
            if (q0 >= 0 && static_cast<std::uint32_t>(q0) < num_qubits) {
                out.oneq_counts_[static_cast<std::uint32_t>(q0)] += 1;
            }
            ++instruction_index;
            continue;
        }
        if (nq != 2) {
            ++instruction_index;
            continue;
        }

        // qargs[i]._index (Qiskit Qubit index)
        py::object q0_obj = qargs.attr("__getitem__")(0);
        py::object q1_obj = qargs.attr("__getitem__")(1);
        auto q0 = q0_obj.attr("_index").cast<std::int64_t>();
        auto q1 = q1_obj.attr("_index").cast<std::int64_t>();

        if (q0 < 0 || q1 < 0) {
            throw std::runtime_error("Encountered negative qubit index from Qiskit");
        }

        // inst.name (gate name)
        std::string gname = inst.attr("name").cast<std::string>();
        GateId gid = out.intern_gate_(gname);
        double angle = 0.0;
        if (is_parametrized_controlled_rotation(gname)) {
            angle = extract_rotation_angle(inst, gname, instruction_index);
        }

        Operation op;
        op.q0 = static_cast<Qubit>(q0);
        op.q1 = static_cast<Qubit>(q1);
        op.gate = gid;
        op.angle = angle;
        op.original_index = instruction_index;

        out.ops_.push_back(op);
        ++instruction_index;
    }

    // Build temporal connectivity once ops_ is finalized
    out.build_temporal_();
    out.build_temporal_graph_();

    // Build connectivity graph once ops_ is finalized
    out.build_connectivity_();

    // Mark buffers already sized in constructor, but ensure correct length.
    out.mark_stamp_.assign(num_qubits, 0);
    out.mark_label_.assign(num_qubits, 0);
    out.mark_cur_stamp_ = 1;

    return out;
}
