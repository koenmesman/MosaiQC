#pragma once
#include <vector>
#include <utility>

struct DeviceInfo {
    int num_qubits = 0;
    std::vector<std::pair<int,int>> couplings;
};
