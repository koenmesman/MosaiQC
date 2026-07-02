#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct CircuitOp {
    std::uint32_t q0 = 0;
    std::uint32_t q1 = 0;
    std::string gate;
};

struct CircuitInfo {
    std::uint32_t num_qubits = 0;
    std::vector<CircuitOp> ops;
};
