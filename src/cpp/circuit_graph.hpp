#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

// Forward declarations to avoid forcing pybind includes in every file that includes this header.
namespace pybind11 {
class handle;
}

class CircuitGraph {
public:
    using Qubit  = std::uint32_t;
    using GateId = std::uint32_t;
    static constexpr std::uint32_t kNoOperationIndex = 0xFFFFFFFFu;

    struct Operation {
        Qubit  q0 = 0;
        Qubit  q1 = 0;
        GateId gate = 0;   // interned ID for gate name (e.g., "cx")
        double angle = 0.0; // parameter angle for rotation-style gates (e.g., crx/cry/crz)
        std::uint32_t original_index = kNoOperationIndex; // index in original qc.data
    };

    // Temporal 2Q "event" storage after merging consecutive gates on the same *unordered* qubit pair.
    //
    // We treat each merged 2Q run as a node (event-id e in [0, E)).
    // For fast queries that reason about adjacency *via a given qubit c*, we additionally store,
    // for every qubit, the sequence of partner-qubits it interacts with in chronological order.
    //
    // This is exactly the data needed to count edges between event-nodes that are consecutive on c.
    struct TemporalGateEvents {
        // endpoints per merged event (stored as uint16_t; supports up to 65535 qubits)
        std::vector<std::uint16_t> q0;  // q0[e]
        std::vector<std::uint16_t> q1;  // q1[e]

        // multiplicity[e] = how many original consecutive 2Q gates were merged into this event.
        std::vector<std::uint32_t> multiplicity;
        // weight[e] = sum of op_weight() contributions of the merged gates in this event.
        std::vector<double> weight;

        // Per-qubit chronological partner timeline in CSR-like form:
        // partners[row_ptr[c] .. row_ptr[c+1]) gives the partner qubit for each 2Q event touching c,
        // in time order.
        std::vector<std::uint32_t> row_ptr;  // size Q+1
        std::vector<std::uint16_t> partners; // size = total incidences (2*num_events)
        // Per-incidence merged-event weight, aligned with `partners`.
        std::vector<double> partner_weight;  // size = total incidences (2*num_events)
        // Per-incidence merged-event identifier, aligned with `partners`.
        std::vector<std::uint32_t> partner_event_id; // size = total incidences (2*num_events)

        // Flattened mapping from merged-event id -> original qc.data instruction indices.
        // Original indices for event e are:
        //   event_original_indices[event_original_row_ptr[e] .. event_original_row_ptr[e+1])
        std::vector<std::uint32_t> event_original_row_ptr; // size = num_events + 1
        std::vector<std::uint32_t> event_original_indices;

        std::uint32_t num_qubits = 0;
        std::uint32_t num_events = 0;

        void build(std::uint32_t Q,
                   const std::vector<Operation>& merged_ops,
                   const std::vector<std::uint32_t>& mult,
                   const std::vector<double>& event_weight,
                   const std::vector<std::vector<std::uint32_t>>& event_original_gate_indices);

        std::uint32_t timeline_len(std::uint32_t q) const;
    };

    // Temporal graph over raw 2Q gates.
    //
    // Each 2Q operation in operations() is a node. An undirected edge connects two
    // nodes when they are consecutive 2Q operations on at least one common qubit.
    // The graph is stored symmetrically in CSR so it can be passed directly to
    // METIS-style graph partitioning. Node weights are the sum of the static
    // connectivity degrees of the two target qubits; edge weights are 1.
    struct TemporalGraph {
        std::uint32_t num_nodes = 0;
        std::vector<std::uint32_t> row_ptr;     // size num_nodes + 1
        std::vector<std::uint32_t> col_idx;     // size nnz
        std::vector<std::uint32_t> node_weight; // size num_nodes
        std::vector<std::uint32_t> edge_weight; // size nnz, all 1

        // Node metadata aligned with operations(): node i is the i-th 2Q op.
        std::vector<std::uint32_t> original_index;
        std::vector<std::uint32_t> q0;
        std::vector<std::uint32_t> q1;

        std::uint32_t nnz() const noexcept { return static_cast<std::uint32_t>(col_idx.size()); }
        std::uint32_t num_edges() const noexcept { return nnz() / 2u; }
    };

    // Weighted qubit connectivity graph induced by 2Q gates.
    //
    // Stored in CSR (Compressed Sparse Row) format:
    //   row_ptr[q]..row_ptr[q+1] indexes neighbors of qubit q
    //   col_idx[k] is the neighbor qubit id
    //   weight[k] is the sum of op_weight() contributions between q and col_idx[k]
    //   node_weight[q] is the number of 1Q gates on qubit q
    //
    // Notes:
    // - This is undirected connectivity stored symmetrically (both q->r and r->q).
    // - Construction is done once (O(G log G) via sort+compress, where G=#2Q gates).
    struct ConnectivityGraph {
        std::uint32_t num_qubits = 0;
        std::vector<std::uint32_t> row_ptr;   // size Q+1
        std::vector<std::uint16_t> col_idx;   // size nnz
        std::vector<double> weight;           // size nnz
        // Stable edge identifier per CSR entry (aligned with col_idx/weight).
        // Both directions of the same undirected edge share the same id.
        std::vector<std::uint32_t> edge_id;   // size nnz

        // Endpoints for each undirected edge id.
        std::vector<std::uint16_t> edge_q0;   // size num_edge_ids
        std::vector<std::uint16_t> edge_q1;   // size num_edge_ids

        // Flattened mapping from edge id -> original qc.data instruction indices.
        // Original indices for edge-id e are:
        //   edge_original_indices[edge_original_row_ptr[e] .. edge_original_row_ptr[e+1])
        std::vector<std::uint32_t> edge_original_row_ptr; // size = num_edge_ids + 1
        std::vector<std::uint32_t> edge_original_indices;
        std::vector<std::uint32_t> node_weight; // size Q

        std::uint32_t nnz() const noexcept { return static_cast<std::uint32_t>(col_idx.size()); }
        std::uint32_t num_edge_ids() const noexcept {
            return edge_original_row_ptr.empty()
                       ? 0u
                       : static_cast<std::uint32_t>(edge_original_row_ptr.size() - 1u);
        }

        // Return weighted interaction sum between q and r (0.0 if not connected).
        // Rows are kept sorted by neighbor id, so we can binary search.
        double count(std::uint32_t q, std::uint32_t r) const;
        std::int64_t edge_identifier(std::uint32_t q, std::uint32_t r) const;
        std::vector<std::uint32_t> edge_gate_indices_by_id(std::uint32_t id) const;
        std::vector<std::uint32_t> edge_gate_indices_by_qubits(std::uint32_t q, std::uint32_t r) const;
    };

    // Construct directly from C++ data (useful for tests / non-py entrypoints)
    CircuitGraph(std::uint32_t num_qubits,
                 std::vector<Operation> ops,
                 std::vector<std::string> gate_names);

    // Build from a Qiskit QuantumCircuit (Python object) via pybind11.
    // Expects qc.num_qubits and qc.data to exist, and qargs[i]._index for qubit indices.
    static CircuitGraph from_qiskit(pybind11::handle qc);

    // Accessors
    std::uint32_t num_qubits() const noexcept { return num_qubits_; }
    const std::vector<Operation>& operations() const noexcept { return ops_; }
    const std::vector<std::uint32_t>& oneq_counts() const noexcept { return oneq_counts_; }
    const std::vector<std::uint32_t>& node_weights() const noexcept { return oneq_counts_; }
    const TemporalGateEvents& temporal_events() const noexcept { return temporal_; }
    const TemporalGraph& temporal_graph() const noexcept { return temporal_graph_; }
    const ConnectivityGraph& connectivity() const noexcept { return connectivity_; }
    std::vector<Operation> cut_edges(const std::vector<int>& primary,
                                     const std::vector<int>& extra_block = {}) const;

    // Lookup: given a common qubit c and two (changing) sets A and B of partner-qubits,
    // count how many temporal edges (between consecutive merged 2Q events on c)
    // connect a partner in A to a partner in B.
    //
    // Formally, for partner timeline p[0..k-1] on c, returns
    //   |{ i in [0,k-2] : (p[i] in A and p[i+1] in B) or (p[i] in B and p[i+1] in A) }|
    std::uint32_t count_temporal_cross_edges(Qubit common_qubit,
                                            const std::vector<Qubit>& set_a,
                                            const std::vector<Qubit>& set_b) const;

    // Gate-name lookup (intern table)
    const std::string& gate_name(GateId id) const;
    std::size_t num_gate_kinds() const noexcept { return gate_names_.size(); }

private:
    // Intern a gate name -> GateId (creates new ID if not present)
    GateId intern_gate_(const std::string& name);

    void build_temporal_();
    void build_temporal_graph_();
    void build_connectivity_();

private:
    std::uint32_t num_qubits_ = 0;
    std::vector<Operation> ops_;
    std::vector<std::uint32_t> oneq_counts_;

    // Interner: name -> id, and id -> name
    std::unordered_map<std::string, GateId> gate_to_id_;
    std::vector<std::string> gate_names_;

    TemporalGateEvents temporal_;
    TemporalGraph temporal_graph_;
    ConnectivityGraph connectivity_;

    // Fast per-query membership marking for changing sets.
    // label: 0=none, 1=in A, 2=in B (current query only).
    mutable std::vector<std::uint32_t> mark_stamp_;
    mutable std::vector<std::uint8_t>  mark_label_;
    mutable std::uint32_t mark_cur_stamp_ = 1;
};
