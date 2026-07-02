#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "device.hpp"

namespace py = pybind11;

void bind_device(py::module_& m) {
    py::class_<DeviceInfo>(m, "DeviceInfo")
        .def(py::init<>())
        .def_readwrite("num_qubits", &DeviceInfo::num_qubits)
        .def_readwrite("couplings",  &DeviceInfo::couplings);

    m.def("count_edges", [](const DeviceInfo& d) {
        return static_cast<int>(d.couplings.size());
    });
}
