#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <numeric>
#include <vector>

namespace py = pybind11;

// forward declare the binder implemented in device.cpp
void bind_device(py::module_&);
// forward declare the binder implemented in topology_graph.cpp
void bind_topology_graph(py::module_&);
// forward declare the binder implemented in circuit.cpp
void bind_circuit(py::module_&);
// forward declare the binder implemented in METIS_solver.cpp
void bind_metis_solver(py::module_&);
// forward declare the binder implemented in partition_refiner.cpp
void bind_partition_refiner(py::module_&);

double sum_weights(const std::vector<double>& weights) {
    return std::accumulate(weights.begin(), weights.end(), 0.0);
}

PYBIND11_MODULE(_core, m) {
    m.doc() = "C++ extension for opt_core";

    // bind C++ functions
    m.def("sum_weights", &sum_weights, py::arg("weights"), "Sum a list of weights.");

    // bind DeviceInfo + helpers
    bind_device(m);

    // bind TopologyGraph
    bind_topology_graph(m);

    // bind CircuitGraph + CircuitInfo
    bind_circuit(m);
    // bind METIS-based partitioning solver
    bind_metis_solver(m);

    // bind partition refiner (warm-start capable)
    bind_partition_refiner(m);
}
