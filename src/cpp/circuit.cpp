#include "circuit.hpp"

#include "circuit_graph.hpp"

#include <stdexcept>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace {

CircuitInfo circuit_from_qiskit_impl(py::handle qc_handle) {
    py::object qc = py::reinterpret_borrow<py::object>(qc_handle);

    CircuitInfo out;
    out.num_qubits = qc.attr("num_qubits").cast<std::uint32_t>();

    py::object data = qc.attr("data");
    for (py::handle item : data) {
        py::object inst;
        py::object qargs;

        py::object item_obj = py::reinterpret_borrow<py::object>(item);
        if (py::isinstance<py::tuple>(item_obj)) {
            py::tuple t = item_obj.cast<py::tuple>();
            if (t.size() < 2) {
                continue;
            }
            inst = py::reinterpret_borrow<py::object>(t[0]);
            qargs = py::reinterpret_borrow<py::object>(t[1]);
        } else if (py::hasattr(item_obj, "operation") && py::hasattr(item_obj, "qubits")) {
            inst = item_obj.attr("operation");
            qargs = item_obj.attr("qubits");
        } else if (py::hasattr(item_obj, "instruction") && py::hasattr(item_obj, "qargs")) {
            inst = item_obj.attr("instruction");
            qargs = item_obj.attr("qargs");
        } else {
            continue;
        }

        std::size_t nq = py::len(qargs);
        if (nq != 2) {
            continue;
        }

        py::object q0_obj = qargs.attr("__getitem__")(0);
        py::object q1_obj = qargs.attr("__getitem__")(1);
        auto q0 = q0_obj.attr("_index").cast<std::int64_t>();
        auto q1 = q1_obj.attr("_index").cast<std::int64_t>();
        if (q0 < 0 || q1 < 0) {
            throw std::runtime_error("Encountered negative qubit index from Qiskit");
        }

        CircuitOp op;
        op.q0 = static_cast<std::uint32_t>(q0);
        op.q1 = static_cast<std::uint32_t>(q1);
        op.gate = inst.attr("name").cast<std::string>();
        out.ops.push_back(std::move(op));
    }

    return out;
}

}  // namespace

void bind_circuit(py::module_& m) {
    py::class_<CircuitOp>(m, "CircuitOp")
        .def(py::init<>())
        .def_readwrite("q0", &CircuitOp::q0)
        .def_readwrite("q1", &CircuitOp::q1)
        .def_readwrite("gate", &CircuitOp::gate);

    py::class_<CircuitInfo>(m, "CircuitInfo")
        .def(py::init<>())
        .def_readwrite("num_qubits", &CircuitInfo::num_qubits)
        .def_readwrite("ops", &CircuitInfo::ops);

    m.def("circuit_from_qiskit", &circuit_from_qiskit_impl, py::arg("qc"),
          "Convert a Qiskit QuantumCircuit into a CircuitInfo.");

    py::class_<CircuitGraph::TemporalGateEvents>(m, "TemporalGateEvents")
        .def(py::init<>())
        .def_property_readonly("num_qubits", [](const CircuitGraph::TemporalGateEvents& te) { return te.num_qubits; })
        .def_property_readonly("num_events", [](const CircuitGraph::TemporalGateEvents& te) { return te.num_events; })
        .def("timeline_len", &CircuitGraph::TemporalGateEvents::timeline_len, py::arg("qubit"),
             "Return the number of merged 2Q events touching the given qubit.")
        .def("event_endpoints", [](const CircuitGraph::TemporalGateEvents& te, std::uint32_t event) {
             if (event >= te.num_events) throw std::out_of_range("event_endpoints(): invalid event id");
             const auto e = static_cast<std::size_t>(event);
             return py::make_tuple(static_cast<std::uint32_t>(te.q0[e]),
                                   static_cast<std::uint32_t>(te.q1[e]));
         }, py::arg("event"), "Return (q0, q1) endpoints of a merged 2Q event node.")
        .def("event_multiplicity", [](const CircuitGraph::TemporalGateEvents& te, std::uint32_t event) {
             if (event >= te.num_events) throw std::out_of_range("event_multiplicity(): invalid event id");
             return te.multiplicity[static_cast<std::size_t>(event)];
         }, py::arg("event"), "Return how many original consecutive 2Q gates were merged into this event.")
        .def("event_weight", [](const CircuitGraph::TemporalGateEvents& te, std::uint32_t event) {
             if (event >= te.num_events) throw std::out_of_range("event_weight(): invalid event id");
             return te.weight[static_cast<std::size_t>(event)];
         }, py::arg("event"), "Return summed op_weight contribution for this merged 2Q event.")
        .def("event_original_indices", [](const CircuitGraph::TemporalGateEvents& te, std::uint32_t event) {
             if (event >= te.num_events) throw std::out_of_range("event_original_indices(): invalid event id");
             const auto b = te.event_original_row_ptr[static_cast<std::size_t>(event)];
             const auto e = te.event_original_row_ptr[static_cast<std::size_t>(event) + 1];
             py::list out;
             for (std::uint32_t i = b; i < e; ++i) {
                 out.append(te.event_original_indices[static_cast<std::size_t>(i)]);
             }
             return out;
         }, py::arg("event"), "Return original qc.data instruction indices merged into this temporal event.")
        .def("partner_timeline", [](const CircuitGraph::TemporalGateEvents& te, std::uint32_t q) {
             if (q >= te.num_qubits) throw std::out_of_range("partner_timeline(): qubit out of range");
             const auto b = te.row_ptr[static_cast<std::size_t>(q)];
             const auto e = te.row_ptr[static_cast<std::size_t>(q) + 1];
             py::list out;
             for (std::uint32_t i = b; i < e; ++i) {
                 out.append(static_cast<std::uint32_t>(te.partners[static_cast<std::size_t>(i)]));
             }
             return out;
         }, py::arg("qubit"), "Return the chronological partner-qubit timeline for this qubit (merged 2Q events).")
        .def("partner_timeline_with_ids", [](const CircuitGraph::TemporalGateEvents& te, std::uint32_t q) {
             if (q >= te.num_qubits) throw std::out_of_range("partner_timeline_with_ids(): qubit out of range");
             const auto b = te.row_ptr[static_cast<std::size_t>(q)];
             const auto e = te.row_ptr[static_cast<std::size_t>(q) + 1];
             py::list out;
             for (std::uint32_t i = b; i < e; ++i) {
                 const auto k = static_cast<std::size_t>(i);
                 out.append(py::make_tuple(
                     static_cast<std::uint32_t>(te.partners[k]),
                     te.partner_event_id[k]));
             }
             return out;
         }, py::arg("qubit"),
         "Return [(partner_qubit, temporal_event_id)] timeline for this qubit.");

    py::class_<CircuitGraph::TemporalGraph>(m, "TemporalGraph")
        .def(py::init<>())
        .def_property_readonly("num_nodes", [](const CircuitGraph::TemporalGraph& tg) { return tg.num_nodes; })
        .def_property_readonly("nnz", [](const CircuitGraph::TemporalGraph& tg) { return tg.nnz(); })
        .def_property_readonly("num_edges", [](const CircuitGraph::TemporalGraph& tg) { return tg.num_edges(); })
        .def_property_readonly("row_ptr", [](const CircuitGraph::TemporalGraph& tg) {
             py::list out;
             for (std::uint32_t v : tg.row_ptr) out.append(v);
             return out;
         }, "Return CSR row pointer array.")
        .def_property_readonly("col_idx", [](const CircuitGraph::TemporalGraph& tg) {
             py::list out;
             for (std::uint32_t v : tg.col_idx) out.append(v);
             return out;
         }, "Return CSR column index array.")
        .def_property_readonly("node_weights", [](const CircuitGraph::TemporalGraph& tg) {
             py::list out;
             for (std::uint32_t w : tg.node_weight) out.append(w);
             return out;
         }, "Return temporal node weights as degree(q0) + degree(q1) for each 2Q gate node.")
        .def_property_readonly("edge_weights", [](const CircuitGraph::TemporalGraph& tg) {
             py::list out;
             for (std::uint32_t w : tg.edge_weight) out.append(w);
             return out;
         }, "Return temporal edge weights aligned with col_idx; all entries are 1.")
        .def("node", [](const CircuitGraph::TemporalGraph& tg, std::uint32_t node) {
             if (node >= tg.num_nodes) throw std::out_of_range("node(): invalid temporal node id");
             const auto i = static_cast<std::size_t>(node);
             return py::make_tuple(tg.original_index[i], tg.q0[i], tg.q1[i]);
         }, py::arg("node"), "Return (original_qc_data_index, q0, q1) for a temporal graph node.")
        .def("neighbors", [](const CircuitGraph::TemporalGraph& tg, std::uint32_t node) {
             if (node >= tg.num_nodes) throw std::out_of_range("neighbors(): invalid temporal node id");
             const auto b = tg.row_ptr[static_cast<std::size_t>(node)];
             const auto e = tg.row_ptr[static_cast<std::size_t>(node) + 1u];
             py::list ns;
             py::list ws;
             for (std::uint32_t k = b; k < e; ++k) {
                 ns.append(tg.col_idx[static_cast<std::size_t>(k)]);
                 ws.append(tg.edge_weight[static_cast<std::size_t>(k)]);
             }
             return py::make_tuple(ns, ws);
         }, py::arg("node"), "Return (neighbor_nodes, edge_weights) for a temporal graph node.");

    py::class_<CircuitGraph::ConnectivityGraph>(m, "ConnectivityGraph")
        .def(py::init<>())
        .def_property_readonly("num_qubits", [](const CircuitGraph::ConnectivityGraph& cg) { return cg.num_qubits; })
        .def_property_readonly("nnz", [](const CircuitGraph::ConnectivityGraph& cg) { return cg.nnz(); })
        .def_property_readonly("num_edge_ids", [](const CircuitGraph::ConnectivityGraph& cg) { return cg.num_edge_ids(); })
        .def_property_readonly("node_weights", [](const CircuitGraph::ConnectivityGraph& cg) {
             py::list out;
             for (std::uint32_t w : cg.node_weight) out.append(w);
             return out;
         }, "Return node weights (1Q gate counts) for each qubit.")
        .def("count", &CircuitGraph::ConnectivityGraph::count, py::arg("q"), py::arg("r"),
             "Return weighted interaction sum between qubits q and r (0.0 if none).")
        .def("edge_id", &CircuitGraph::ConnectivityGraph::edge_identifier, py::arg("q"), py::arg("r"),
             "Return connectivity edge id for (q, r), or -1 if no edge exists.")
        .def("edge_endpoints", [](const CircuitGraph::ConnectivityGraph& cg, std::uint32_t edge_id) {
             if (edge_id >= cg.num_edge_ids()) throw std::out_of_range("edge_endpoints(): invalid edge id");
             const auto e = static_cast<std::size_t>(edge_id);
             return py::make_tuple(static_cast<std::uint32_t>(cg.edge_q0[e]),
                                   static_cast<std::uint32_t>(cg.edge_q1[e]));
         }, py::arg("edge_id"),
         "Return (q0, q1) endpoints for an undirected connectivity edge id.")
        .def("edge_original_indices", &CircuitGraph::ConnectivityGraph::edge_gate_indices_by_id, py::arg("edge_id"),
             "Return original qc.data instruction indices contributing to this connectivity edge id.")
        .def("edge_original_indices_by_qubits", &CircuitGraph::ConnectivityGraph::edge_gate_indices_by_qubits,
             py::arg("q"), py::arg("r"),
             "Return original qc.data instruction indices contributing to connectivity edge (q, r).")
        .def("neighbors", [](const CircuitGraph::ConnectivityGraph& cg, std::uint32_t q) {
             if (q >= cg.num_qubits) throw std::out_of_range("neighbors(): qubit out of range");
             const auto b = cg.row_ptr[static_cast<std::size_t>(q)];
             const auto e = cg.row_ptr[static_cast<std::size_t>(q) + 1];
             py::list ns;
             py::list ws;
             for (std::uint32_t k = b; k < e; ++k) {
                 ns.append(static_cast<std::uint32_t>(cg.col_idx[static_cast<std::size_t>(k)]));
                 ws.append(cg.weight[static_cast<std::size_t>(k)]);
             }
             return py::make_tuple(ns, ws);
         }, py::arg("qubit"), "Return (neighbors, weights) for a qubit.")
        .def("neighbors_with_ids", [](const CircuitGraph::ConnectivityGraph& cg, std::uint32_t q) {
             if (q >= cg.num_qubits) throw std::out_of_range("neighbors_with_ids(): qubit out of range");
             const auto b = cg.row_ptr[static_cast<std::size_t>(q)];
             const auto e = cg.row_ptr[static_cast<std::size_t>(q) + 1];
             py::list ns;
             py::list ws;
             py::list ids;
             for (std::uint32_t k = b; k < e; ++k) {
                 const auto idx = static_cast<std::size_t>(k);
                 ns.append(static_cast<std::uint32_t>(cg.col_idx[idx]));
                 ws.append(cg.weight[idx]);
                 ids.append(cg.edge_id[idx]);
             }
             return py::make_tuple(ns, ws, ids);
         }, py::arg("qubit"), "Return (neighbors, weights, edge_ids) for a qubit.");

    py::class_<CircuitGraph>(m, "CircuitGraph")
        .def(py::init([](py::object qc) {
                return CircuitGraph::from_qiskit(qc);
             }),
             py::arg("qc"))
        .def("num_qubits", &CircuitGraph::num_qubits)
        .def("num_gate_kinds", &CircuitGraph::num_gate_kinds)
        .def("gate_name", &CircuitGraph::gate_name, py::arg("id"))
        .def("oneq_counts", [](const CircuitGraph& graph) {
                py::list out;
                for (std::uint32_t c : graph.oneq_counts()) {
                    out.append(c);
                }
                return out;
             })
        .def("node_weights", [](const CircuitGraph& graph) {
                py::list out;
                for (std::uint32_t w : graph.node_weights()) {
                    out.append(w);
                }
                return out;
             }, "Return node weights (1Q gate counts) for each qubit.")
        .def("ops", [](const CircuitGraph& graph) {
                py::list out;
                for (const auto& op : graph.operations()) {
                    out.append(py::make_tuple(op.q0, op.q1, graph.gate_name(op.gate)));
                }
                return out;
             })
        .def("ops_with_indices", [](const CircuitGraph& graph) {
                py::list out;
                for (const auto& op : graph.operations()) {
                    out.append(py::make_tuple(op.original_index, op.q0, op.q1, graph.gate_name(op.gate)));
                }
                return out;
             }, "Return 2Q operations as tuples: (original_qc_data_index, q0, q1, gate_name).")
        .def("cut_edges", [](const CircuitGraph& graph,
                              const std::vector<int>& primary,
                              const std::vector<int>& extra_block) {
                py::list out;
                for (const auto& op : graph.cut_edges(primary, extra_block)) {
                    out.append(py::make_tuple(op.original_index, op.q0, op.q1, graph.gate_name(op.gate)));
                }
                return out;
             }, py::arg("primary"), py::arg("extra_block") = std::vector<int>{},
             "Return cut 2Q edges as tuples: (original_qc_data_index, q0, q1, gate_name).")
        .def("cut_edge_indices", [](const CircuitGraph& graph,
                                     const std::vector<int>& primary,
                                     const std::vector<int>& extra_block) {
                py::list out;
                for (const auto& op : graph.cut_edges(primary, extra_block)) {
                    out.append(op.original_index);
                }
                return out;
             }, py::arg("primary"), py::arg("extra_block") = std::vector<int>{},
             "Return original qc.data indices for cut 2Q edges under the provided placement.")
        .def_property_readonly("temporal_events",
             [](const CircuitGraph& graph) -> const CircuitGraph::TemporalGateEvents& {
                 return graph.temporal_events();
             },
             py::return_value_policy::reference_internal,
             "Temporal merged 2Q events + per-qubit partner timelines (for fast temporal-edge queries).")
        .def_property_readonly("temporal_graph",
             [](const CircuitGraph& graph) -> const CircuitGraph::TemporalGraph& {
                 return graph.temporal_graph();
             },
             py::return_value_policy::reference_internal,
             "Temporal graph with raw 2Q gates as nodes and unit-weight temporal edges (CSR).")
        .def_property_readonly("connectivity",
             [](const CircuitGraph& graph) -> const CircuitGraph::ConnectivityGraph& {
                 return graph.connectivity();
             },
             py::return_value_policy::reference_internal,
             "Weighted qubit connectivity graph induced by 2Q gates (CSR).")
        .def("count_temporal_cross_edges", &CircuitGraph::count_temporal_cross_edges,
             py::arg("common_qubit"), py::arg("set_a"), py::arg("set_b"),
             "Count how many temporal edges (between consecutive merged 2Q events on the common qubit) connect partners from set_a to set_b.")
        .def("qubits", [](const CircuitGraph& graph) {
                py::list out;
                const auto n = graph.num_qubits();
                for (std::uint32_t q = 0; q < n; ++q) {
                    out.append(q);
                }
                return out;
             });
}
