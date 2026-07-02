#include "topology_graph.hpp"

#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_set>

using namespace std;

namespace {

double safe_float(const py::handle& obj, double fallback = 0.0) {
    try {
        return py::cast<double>(obj);
    } catch (...) {
        return fallback;
    }
}

py::object dict_get(py::handle dict_like, py::handle key, py::handle fallback) {
        try {
        return dict_like.attr("get")(key, fallback);
    } catch (...) {
        py::object fb = py::reinterpret_borrow<py::object>(fallback);
        fb.inc_ref();
        return fb;
    }
}

py::object get_item0(const py::object& seq, std::size_t i) {
    return seq.attr("__getitem__")(py::int_(static_cast<long long>(i)));
}

double log_one_minus_error(double error_rate) {
    if (!(error_rate >= 0.0)) {
        return 0.0;
    }
    if (error_rate >= 1.0) {
        return -std::numeric_limits<double>::infinity();
    }
    // Stable for small error_rate values.
    return std::log1p(-error_rate);
}

std::uint64_t undirected_key(int a, int b) {
    const int lo = (a < b) ? a : b;
    const int hi = (a < b) ? b : a;
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(lo)) << 32) |
           static_cast<std::uint32_t>(hi);
}

}  // namespace

TopologyGraph::TopologyGraph(py::object backend, py::object noise_model) {

    const bool has_backend = !backend.is_none();
    const bool has_noise = !noise_model.is_none();

    if (has_backend == has_noise) {
        throw std::invalid_argument(
            "TopologyGraph: provide exactly one of backend or noise_model.");
    }

    if (has_noise) {
        noise_model_ = std::move(noise_model);
    } else {
        // Call NoiseModel.from_backend(backend) from C++.
        py::module_ aer_noise = py::module_::import("qiskit_aer.noise");
        py::object NoiseModel = aer_noise.attr("NoiseModel");
        noise_model_ = NoiseModel.attr("from_backend")(backend);
    }

    build_from_noise_model_(noise_model_);
}

void TopologyGraph::rebuild() {
    if (noise_model_.is_none()) {
        throw std::runtime_error("TopologyGraph: no stored noise_model to rebuild from.");
    }
    build_from_noise_model_(noise_model_);
}

std::vector<int> TopologyGraph::qubits() const {
    std::vector<int> qs;
    qs.reserve(nodes_.size());
    for (const auto& kv : nodes_) {
        qs.push_back(kv.first);
    }
    std::sort(qs.begin(), qs.end());
    return qs;
}

bool TopologyGraph::has_qubit(int q) const {
    return nodes_.find(q) != nodes_.end();
}

NodeErrorRates TopologyGraph::node(int q) const {
    const auto it = nodes_.find(q);
    if (it == nodes_.end()) {
        throw std::out_of_range("TopologyGraph: unknown qubit id");
    }
    return it->second;
}

std::vector<std::pair<int, int>> TopologyGraph::directed_edges() const {
    std::vector<std::pair<int, int>> es;
    es.reserve(edges_.size());
    for (const auto& kv : edges_) {
        es.push_back(kv.first);
    }
    std::sort(es.begin(), es.end());
    return es;
}

bool TopologyGraph::has_cx(int q0, int q1) const {
    return edges_.find({q0, q1}) != edges_.end();
}

EdgeErrorRates TopologyGraph::edge(int q0, int q1) const {
    const auto it = edges_.find({q0, q1});
    if (it == edges_.end()) {
        throw std::out_of_range("TopologyGraph: unknown directed edge");
    }
    return it->second;
}

py::object TopologyGraph::to_networkx() const {
    py::module_ nx = py::module_::import("networkx");
    py::object G = nx.attr("DiGraph")();

    for (const auto& kv : nodes_) {
        const int q = kv.first;
        const NodeErrorRates& r = kv.second;
        py::dict attrs;
        attrs["readout"] = r.readout;
        attrs["x"] = r.x;
        attrs["reset"] = r.reset;
        attrs["sum_error"] = r.sum_error;
        G.attr("add_node")(q, **attrs);
    }

    for (const auto& kv : edges_) {
        const int q0 = kv.first.first;
        const int q1 = kv.first.second;
        const EdgeErrorRates& r = kv.second;
        py::dict attrs;
        attrs["cx"] = r.cx;
        G.attr("add_edge")(q0, q1, **attrs);
    }

    return G;
}

void TopologyGraph::rebuild_weighted_ged_cache_() {
    auto& out = weighted_ged_cache_;
    const auto qids = qubits();  // sorted for stable local indexing
    out.n = static_cast<int>(qids.size());
    out.node_x.assign(static_cast<std::size_t>(out.n), 0.0);
    out.degree.assign(static_cast<std::size_t>(out.n), 0);
    out.edge_score.assign(static_cast<std::size_t>(out.n) * static_cast<std::size_t>(out.n),
                          -std::numeric_limits<double>::infinity());
    out.best_edge_score = -std::numeric_limits<double>::infinity();

    std::unordered_map<int, int> q_to_local;
    q_to_local.reserve(qids.size() * 2u + 1u);
    for (int i = 0; i < out.n; ++i) {
        q_to_local.emplace(qids[static_cast<std::size_t>(i)], i);
        out.node_x[static_cast<std::size_t>(i)] = nodes_.at(qids[static_cast<std::size_t>(i)]).x;
        out.edge_score[static_cast<std::size_t>(i) * static_cast<std::size_t>(out.n) +
                       static_cast<std::size_t>(i)] = 0.0;
    }

    std::unordered_set<std::uint64_t> counted_undirected;
    counted_undirected.reserve(2u * static_cast<std::size_t>(out.n) + 1u);

    for (const auto& kv : edges_) {
        const int q0 = kv.first.first;
        const int q1 = kv.first.second;
        auto it_u = q_to_local.find(q0);
        auto it_v = q_to_local.find(q1);
        if (it_u == q_to_local.end() || it_v == q_to_local.end()) continue;
        const int u = it_u->second;
        const int v = it_v->second;
        if (u == v) continue;

        const double cx = kv.second.cx;
        const std::size_t uv = static_cast<std::size_t>(u) * static_cast<std::size_t>(out.n) +
                               static_cast<std::size_t>(v);
        const std::size_t vu = static_cast<std::size_t>(v) * static_cast<std::size_t>(out.n) +
                               static_cast<std::size_t>(u);
        out.edge_score[uv] = std::max(out.edge_score[uv], cx);
        out.edge_score[vu] = std::max(out.edge_score[vu], cx);
        out.best_edge_score = std::max(out.best_edge_score, cx);

        const std::uint64_t key = undirected_key(u, v);
        if (counted_undirected.insert(key).second) {
            out.degree[static_cast<std::size_t>(u)] += 1;
            out.degree[static_cast<std::size_t>(v)] += 1;
        }
    }

    out.order.resize(static_cast<std::size_t>(out.n));
    for (int i = 0; i < out.n; ++i) out.order[static_cast<std::size_t>(i)] = i;
    std::sort(out.order.begin(), out.order.end(), [&](int a, int b) {
        const double xa = out.node_x[static_cast<std::size_t>(a)];
        const double xb = out.node_x[static_cast<std::size_t>(b)];
        if (xa != xb) return xa > xb;
        const int da = out.degree[static_cast<std::size_t>(a)];
        const int db = out.degree[static_cast<std::size_t>(b)];
        if (da != db) return da > db;
        return a < b;
    });
}

void TopologyGraph::build_from_noise_model_(const py::object& noise_model) {
    nodes_.clear();
    edges_.clear();
    // --- Nodes: single-qubit errors ---
    py::object qubits_obj = noise_model.attr("noise_qubits");

    // Internals we rely on (same as your Python implementation).
    py::object local_readout = noise_model.attr("_local_readout_errors");
    py::object local_qerrors = noise_model.attr("_local_quantum_errors");

    // Extract sub-dicts for x/reset/cx; fall back to empty dict if absent.
    py::dict x_map = dict_get(local_qerrors, py::str("x"), py::dict()).cast<py::dict>();
    py::dict reset_map = dict_get(local_qerrors, py::str("reset"), py::dict()).cast<py::dict>();

    for (py::handle qh : qubits_obj) {
        const int q = qh.cast<int>();

        // --- Readout error ---
        py::object readout_err = dict_get(local_readout, py::make_tuple(q), py::none());
        double readout_rate = 0.0;
        if (!readout_err.is_none()) {
            try {
                py::dict d = readout_err.attr("to_dict")().cast<py::dict>();
                py::list probs = d["probabilities"].cast<py::list>();
                // probabilities[0][1]
                readout_rate = safe_float(probs[0].cast<py::sequence>()[1], 0.0);
            } catch (...) {
                readout_rate = 0.0;
            }
        }

        // --- X error ---
        py::object x_err = dict_get(x_map, py::make_tuple(q), py::none());
        double x_rate = 0.0;
        if (!x_err.is_none()) {
            try {
                py::dict d = x_err.attr("to_dict")().cast<py::dict>();
                py::list probs = dict_get(d, py::str("probabilities"), py::list()).cast<py::list>();
                // sum(probabilities[1:])
                for (std::size_t i = 1; i < static_cast<std::size_t>(py::len(probs)); ++i) {
                    x_rate += safe_float(get_item0(probs, i), 0.0);
                }
            } catch (...) {
                x_rate = 0.0;
            }
        }

        // --- Reset error ---
        py::object reset_err = dict_get(reset_map, py::make_tuple(q), py::none());
        double reset_rate = 0.0;
        if (!reset_err.is_none()) {
            try {
                py::object probs = reset_err.attr("probabilities");
                if (py::len(probs) > 0) {
                    reset_rate = 1.0 - safe_float(get_item0(probs, 0), 1.0);
                }
            } catch (...) {
                reset_rate = 0.0;
            }
        }

        NodeErrorRates rates;
        rates.readout = readout_rate;
        rates.x = log_one_minus_error(x_rate);
        rates.reset = reset_rate;
        rates.sum_error = readout_rate + rates.x + reset_rate;
        nodes_.emplace(q, rates);
    }

    // --- Edges: CX errors ---
    py::dict cx_map = dict_get(local_qerrors, py::str("cx"), py::dict()).cast<py::dict>();
    for (auto item : cx_map) {
        const py::tuple key = item.first.cast<py::tuple>();
        if (py::len(key) != 2) {
            continue;
        }
        const int q0 = key[0].cast<int>();
        const int q1 = key[1].cast<int>();
        const py::object err_obj = item.second.cast<py::object>();

        double cx_error = 0.0;
        try {
            py::dict d = err_obj.attr("to_dict")().cast<py::dict>();
            py::list probs = dict_get(d, py::str("probabilities"), py::list()).cast<py::list>();
            if (py::len(probs) > 0) {
                cx_error   = 1.0 - safe_float(get_item0(probs, 0), 1.0);
            }
        } catch (...) {
            cx_error = 0.0;
        }

        edges_.emplace(std::make_pair(q0, q1), EdgeErrorRates{log_one_minus_error(cx_error)});
    }

    // Build the weighted GED cache once per topology build/rebuild.
    rebuild_weighted_ged_cache_();
}

void bind_topology_graph(py::module_& m) {
    py::class_<NodeErrorRates>(m, "NodeErrorRates")
        .def(py::init<>())
        .def_readwrite("readout", &NodeErrorRates::readout)
        .def_readwrite("x", &NodeErrorRates::x)
        .def_readwrite("reset", &NodeErrorRates::reset)
        .def_readwrite("sum_error", &NodeErrorRates::sum_error);

    py::class_<EdgeErrorRates>(m, "EdgeErrorRates")
        .def(py::init<>())
        .def_readwrite("cx", &EdgeErrorRates::cx);

    py::class_<TopologyGraph>(m, "TopologyGraph")
        .def(py::init<py::object, py::object>(),
             py::arg("backend") = py::none(),
             py::arg("noise_model") = py::none())
        .def("rebuild", &TopologyGraph::rebuild)
        .def("qubits", &TopologyGraph::qubits)
        .def("has_qubit", &TopologyGraph::has_qubit, py::arg("q"))
        .def("node", &TopologyGraph::node, py::arg("q"))
        .def("directed_edges", &TopologyGraph::directed_edges)
        .def("has_cx", &TopologyGraph::has_cx, py::arg("q0"), py::arg("q1"))
        .def("edge", &TopologyGraph::edge, py::arg("q0"), py::arg("q1"))
        .def("to_networkx", &TopologyGraph::to_networkx);
}
