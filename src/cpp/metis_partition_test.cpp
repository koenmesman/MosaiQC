// src/cpp/metis_partition_test.cpp
//
// Standalone test: use CircuitGraph as G, then partition qubits with METIS using uneven part sizes.
//
// Build (example):
//   g++ -O3 -std=c++17 -Isrc/cpp src/cpp/metis_partition_test.cpp src/cpp/circuit_graph.cpp -lmetis -o metis_test
//
// Or via CMake target (see below).

#include <metis.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#include "circuit_graph.hpp"

// Generate random 2Q ops to induce a connectivity graph.
static CircuitGraph make_random_circuit_graph(
    std::uint32_t num_qubits,
    std::uint32_t num_twoq_ops,
    std::uint32_t seed = 1234
) {
    if (num_qubits < 2) throw std::invalid_argument("num_qubits must be >= 2");

    std::mt19937 rng(seed);
    std::uniform_int_distribution<std::uint32_t> qdist(0, num_qubits - 1);

    // We'll intern only one gate kind: "cx" at id=0
    std::vector<std::string> gate_names = {"cx"};
    std::vector<CircuitGraph::Operation> ops;
    ops.reserve(num_twoq_ops);

    for (std::uint32_t i = 0; i < num_twoq_ops; ++i) {
        std::uint32_t a = qdist(rng);
        std::uint32_t b = qdist(rng);
        while (b == a) b = qdist(rng);

        CircuitGraph::Operation op;
        op.q0 = a;
        op.q1 = b;
        op.gate = 0; // "cx"
        ops.push_back(op);
    }

    return CircuitGraph(num_qubits, std::move(ops), std::move(gate_names));
}

// Make random capacities whose sum >= n.
// Example: n=10, k=3 might produce [4,4,3] or other vectors.
static std::vector<int> random_capacities_sum_geq_n(int n, int k, std::uint32_t seed = 7) {
    if (n <= 0 || k <= 0) throw std::invalid_argument("n,k must be > 0");

    std::mt19937 rng(seed);

    // Start with 1 each to avoid zero blocks, then distribute the rest.
    std::vector<int> caps(k, 1);
    int remaining = n - k;
    if (remaining < 0) remaining = 0;

    std::uniform_int_distribution<int> pick(0, k - 1);
    for (int i = 0; i < remaining; ++i) caps[pick(rng)]++;

    // Add a small slack (extra capacity) sometimes:
    std::uniform_int_distribution<int> slack_dist(0, std::max(0, n / 10));
    int slack = slack_dist(rng);
    for (int i = 0; i < slack; ++i) caps[pick(rng)]++;

    // Ensure sum >= n
    int sum = std::accumulate(caps.begin(), caps.end(), 0);
    if (sum < n) caps[pick(rng)] += (n - sum);

    return caps;
}

static void print_vec(const std::vector<int>& v, const char* name) {
    std::cout << name << " = [";
    for (size_t i = 0; i < v.size(); ++i) {
        std::cout << v[i] << (i + 1 < v.size() ? ", " : "");
    }
    std::cout << "]\n";
}

int main() {
    try {
        // -----------------------------
        // 1) Build a CircuitGraph G in-memory
        // -----------------------------
        const std::uint32_t n_qubits = 10;
        const std::uint32_t n_twoq_ops = 40; // increase for denser connectivity

        CircuitGraph G = make_random_circuit_graph(n_qubits, n_twoq_ops, /*seed=*/123);

        // METIS partitions "vertices" -> here: qubits (nodes in connectivity graph)
        const auto& C = G.connectivity();

        // -----------------------------
        // 2) Define partition capacities (your subsets sizes)
        //    For the concrete example: [4,4,3]
        //    You can also uncomment the random generator below.
        // -----------------------------
        std::vector<int> capacities = {4, 4, 3};
        // std::vector<int> capacities = random_capacities_sum_geq_n((int)n_qubits, /*k=*/3);

        const int k = (int)capacities.size();
        const int sum_caps = std::accumulate(capacities.begin(), capacities.end(), 0);

        if (sum_caps < (int)n_qubits) {
            throw std::runtime_error("sum(capacities) must be >= number of graph nodes");
        }

        std::cout << "n_qubits = " << n_qubits << "\n";
        print_vec(capacities, "capacities");
        std::cout << "sum(capacities) = " << sum_caps
                  << "  (overlap budget = " << (sum_caps - (int)n_qubits) << ")\n\n";

        // -----------------------------
        // 3) Convert CircuitGraph connectivity CSR -> METIS CSR (idx_t/real_t)
        // -----------------------------
        // CircuitGraph:
        //   row_ptr: uint32_t size Q+1
        //   col_idx: uint16_t size nnz
        //   weight : uint32_t size nnz
        //
        // METIS expects:
        //   xadj   : idx_t size n+1
        //   adjncy : idx_t size nnz
        //   adjwgt : idx_t size nnz (optional)
        //
        // Note: CircuitGraph connectivity is already symmetric (undirected) and sorted.

        const idx_t n = (idx_t)C.num_qubits;
        if ((std::uint32_t)n != C.num_qubits) throw std::runtime_error("idx_t overflow for n");

        std::vector<idx_t> xadj(C.row_ptr.size());
        for (size_t i = 0; i < C.row_ptr.size(); ++i) xadj[i] = (idx_t)C.row_ptr[i];

        std::vector<idx_t> adjncy(C.col_idx.size());
        for (size_t i = 0; i < C.col_idx.size(); ++i) adjncy[i] = (idx_t)C.col_idx[i];

        std::vector<idx_t> adjwgt(C.weight.size());
        for (size_t i = 0; i < C.weight.size(); ++i) adjwgt[i] = (idx_t)C.weight[i];

        // -----------------------------
        // 4) METIS parameters for uneven part sizes
        // -----------------------------
        const idx_t ncon = 1;
        const idx_t nparts = (idx_t)k;

        // target part weights: fractions that sum to 1
        std::vector<real_t> tpwgts((size_t)nparts * (size_t)ncon);
        for (int i = 0; i < k; ++i) {
            tpwgts[(size_t)i] = (real_t)capacities[i] / (real_t)sum_caps;
        }

        // imbalance tolerance (1.03 = allow ~3% over target)
        std::vector<real_t> ubvec((size_t)ncon, (real_t)1.03);

        idx_t options[METIS_NOPTIONS];
        METIS_SetDefaultOptions(options);
        options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_CUT; // minimize (weighted) edge cut
        options[METIS_OPTION_NUMBERING] = 0;               // 0-based
        options[METIS_OPTION_SEED] = 42;

        std::vector<idx_t> part((size_t)n, 0);
        idx_t objval = 0;

        // No vertex weights (each vertex weight 1); if you want weighted balancing, pass vwgt instead.
        idx_t* vwgt = nullptr;

        int status = METIS_PartGraphKway(
            const_cast<idx_t*>(&n),
            const_cast<idx_t*>(&ncon),
            xadj.data(),
            adjncy.data(),
            vwgt,
            nullptr,           // vsize
            adjwgt.data(),     // edge weights -> weighted cut
            const_cast<idx_t*>(&nparts),
            tpwgts.data(),
            ubvec.data(),
            options,
            &objval,
            part.data()
        );

        if (status != METIS_OK) {
            std::cerr << "METIS_PartGraphKway failed, status=" << status << "\n";
            return 1;
        }

        // -----------------------------
        // 5) Print results
        // -----------------------------
        std::vector<int> counts(k, 0);
        for (idx_t v = 0; v < n; ++v) counts[(int)part[(size_t)v]]++;

        std::cout << "METIS OK\n";
        std::cout << "Weighted cut objval = " << objval << "\n\n";
        std::cout << "Assignment (qubit -> block):\n";
        for (idx_t v = 0; v < n; ++v) {
            std::cout << "  " << (int)v << " -> " << (int)part[(size_t)v] << "\n";
        }

        std::cout << "\nBlock sizes (actual vs target approx):\n";
        for (int b = 0; b < k; ++b) {
            double target = (double)tpwgts[(size_t)b] * (double)n;
            std::cout << "  block " << b << ": " << counts[b]
                      << "  (target ~ " << target << ")\n";
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 2;
    }
}
