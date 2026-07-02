#include "QAP_fw.hpp"

#include "circuit_graph.hpp"
#include "topology_graph.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(OPTCORE_HAS_OPENMP) && OPTCORE_HAS_OPENMP
#include <omp.h>
#endif

namespace QAP_fw {

namespace {

struct WeightedGraph {
    int n = 0;
    std::vector<double> node;  // size n
    std::vector<double> edge;  // dense n*n
};

struct PartitionProblem {
    struct WeightedEdge {
        int u = 0;
        int v = 0;
        double w = 0.0;
    };

    std::vector<double> node_weight;
    std::vector<WeightedEdge> edges;
};

double infinity_cost() {
    return std::numeric_limits<double>::infinity();
}

inline std::size_t dense_idx(int r, int c, int ncols) {
    return static_cast<std::size_t>(r) * static_cast<std::size_t>(ncols) +
           static_cast<std::size_t>(c);
}

WeightedGraph build_hardware_graph_prob(const TopologyGraph& hw, bool use_routed_hardware_edges) {
    WeightedGraph g;
    const auto& cache = hw.weighted_ged_cache();
    g.n = cache.n;
    g.node.assign(static_cast<std::size_t>(g.n), 0.0);
    g.edge.assign(static_cast<std::size_t>(g.n) * static_cast<std::size_t>(g.n), 0.0);

    std::vector<double> direct_cost(static_cast<std::size_t>(g.n) * static_cast<std::size_t>(g.n),
                                    infinity_cost());
    double max_present_cost = 0.0;
    double sum_present_cost = 0.0;
    int present_edge_count = 0;
    bool has_present_edge = false;
    for (int u = 0; u < g.n; ++u) {
        direct_cost[dense_idx(u, u, g.n)] = 0.0;
        for (int v = 0; v < g.n; ++v) {
            if (u == v) continue;
            const double lv = cache.edge_score[dense_idx(u, v, g.n)];
            if (!std::isfinite(lv)) continue;
            const double cost = std::max(0.0, -lv);
            direct_cost[dense_idx(u, v, g.n)] = cost;
            max_present_cost = std::max(max_present_cost, cost);
            sum_present_cost += cost;
            present_edge_count += 1;
            has_present_edge = true;
        }
    }
    const double missing_edge_score = has_present_edge ? -(2.0 * max_present_cost + 1.0) : -1.0;
    const double missing_edge_cost = -missing_edge_score;
    const double mean_present_cost =
        present_edge_count > 0 ? sum_present_cost / static_cast<double>(present_edge_count) : 0.0;

    std::vector<double> routed_cost;
    std::vector<int> routed_hops;
    if (use_routed_hardware_edges) {
        routed_cost = direct_cost;
        routed_hops.assign(static_cast<std::size_t>(g.n) * static_cast<std::size_t>(g.n),
                           std::numeric_limits<int>::max());
        for (int u = 0; u < g.n; ++u) {
            routed_hops[dense_idx(u, u, g.n)] = 0;
            for (int v = 0; v < g.n; ++v) {
                if (u != v && std::isfinite(direct_cost[dense_idx(u, v, g.n)])) {
                    routed_hops[dense_idx(u, v, g.n)] = 1;
                }
            }
        }
        for (int m = 0; m < g.n; ++m) {
            for (int u = 0; u < g.n; ++u) {
                const double um = routed_cost[dense_idx(u, m, g.n)];
                if (!std::isfinite(um)) continue;
                for (int v = 0; v < g.n; ++v) {
                    const double mv = routed_cost[dense_idx(m, v, g.n)];
                    if (!std::isfinite(mv)) continue;
                    const double candidate = um + mv;
                    double& current = routed_cost[dense_idx(u, v, g.n)];
                    const int candidate_hops =
                        routed_hops[dense_idx(u, m, g.n)] + routed_hops[dense_idx(m, v, g.n)];
                    int& current_hops = routed_hops[dense_idx(u, v, g.n)];
                    if (candidate < current ||
                        (candidate == current && candidate_hops < current_hops)) {
                        current = candidate;
                        current_hops = candidate_hops;
                    }
                }
            }
        }
    }

    for (int i = 0; i < g.n; ++i) {
        // Keep TopologyGraph's raw log_one_minus_error node weights.
        g.node[static_cast<std::size_t>(i)] = cache.node_x[static_cast<std::size_t>(i)];
    }

    for (int u = 0; u < g.n; ++u) {
        for (int v = 0; v < g.n; ++v) {
            if (u == v) continue;
            const double lv =
                cache.edge_score[dense_idx(u, v, g.n)];  // cx is log(1-error), -inf if missing
            if (std::isfinite(lv)) {
                // Keep direct log_one_minus_error for present couplings.
                g.edge[dense_idx(u, v, g.n)] = lv;
                continue;
            }
            double missing_cost = missing_edge_cost;
            if (use_routed_hardware_edges && !routed_cost.empty()) {
                const double path_cost = routed_cost[dense_idx(u, v, g.n)];
                if (std::isfinite(path_cost)) {
                    const int path_hops = routed_hops[dense_idx(u, v, g.n)];
                    const int extra_hops = std::max(0, path_hops - 1);
                    // Fast routing proxy: a nonlocal interaction is approximated by
                    // SWAP-style movement along the best error-weighted path, plus
                    // a backend-noise-scaled route-length penalty. This discourages
                    // long low-error detours without requiring full transpilation.
                    missing_cost = 3.0 * path_cost +
                                   0.25 * static_cast<double>(extra_hops) * mean_present_cost;
                }
            }
            // Missing/routed pairs are finite but worse than direct couplings.
            g.edge[dense_idx(u, v, g.n)] = -missing_cost;
        }
    }
    return g;
}

WeightedGraph build_circuit_partition_graph(const PartitionProblem& prob) {
    WeightedGraph g;
    g.n = static_cast<int>(prob.node_weight.size());
    g.node = prob.node_weight;
    g.edge.assign(static_cast<std::size_t>(g.n) * static_cast<std::size_t>(g.n), 0.0);
    for (const auto& e : prob.edges) {
        if (e.u == e.v) continue;
        g.edge[dense_idx(e.u, e.v, g.n)] += e.w;
        g.edge[dense_idx(e.v, e.u, g.n)] += e.w;
    }
    return g;
}

// Hungarian (rectangular, max weight):
// Assign each "task" i in [0, cols) to a distinct "worker" r in [0, rows),
// maximizing W[r, i]. Requires rows >= cols.
std::vector<int> hungarian_max_rect(const std::vector<double>& W, int rows, int cols) {
    if (rows < cols) {
        throw std::invalid_argument("hungarian_max_rect(): rows must be >= cols");
    }

    // Transform to min-cost assignment for n<=m:
    // tasks -> rows in Hungarian notation (n = cols), workers -> cols (m = rows).
    const int n = cols;
    const int m = rows;
    std::vector<double> u(static_cast<std::size_t>(n) + 1u, 0.0);
    std::vector<double> v(static_cast<std::size_t>(m) + 1u, 0.0);
    std::vector<int> p(static_cast<std::size_t>(m) + 1u, 0);
    std::vector<int> way(static_cast<std::size_t>(m) + 1u, 0);

    for (int i = 1; i <= n; ++i) {
        p[0] = i;
        int j0 = 0;
        std::vector<double> minv(static_cast<std::size_t>(m) + 1u, std::numeric_limits<double>::infinity());
        std::vector<unsigned char> used(static_cast<std::size_t>(m) + 1u, 0);
        do {
            used[static_cast<std::size_t>(j0)] = 1;
            const int i0 = p[static_cast<std::size_t>(j0)];
            double delta = std::numeric_limits<double>::infinity();
            int j1 = 0;
            for (int j = 1; j <= m; ++j) {
                if (used[static_cast<std::size_t>(j)] != 0) continue;
                // cost = -weight
                const int worker = j - 1;
                const int task = i0 - 1;
                const double cur = -W[dense_idx(worker, task, cols)] - u[static_cast<std::size_t>(i0)] -
                                   v[static_cast<std::size_t>(j)];
                if (cur < minv[static_cast<std::size_t>(j)]) {
                    minv[static_cast<std::size_t>(j)] = cur;
                    way[static_cast<std::size_t>(j)] = j0;
                }
                if (minv[static_cast<std::size_t>(j)] < delta) {
                    delta = minv[static_cast<std::size_t>(j)];
                    j1 = j;
                }
            }
            for (int j = 0; j <= m; ++j) {
                if (used[static_cast<std::size_t>(j)] != 0) {
                    u[static_cast<std::size_t>(p[static_cast<std::size_t>(j)])] += delta;
                    v[static_cast<std::size_t>(j)] -= delta;
                } else {
                    minv[static_cast<std::size_t>(j)] -= delta;
                }
            }
            j0 = j1;
        } while (p[static_cast<std::size_t>(j0)] != 0);

        do {
            const int j1 = way[static_cast<std::size_t>(j0)];
            p[static_cast<std::size_t>(j0)] = p[static_cast<std::size_t>(j1)];
            j0 = j1;
        } while (j0 != 0);
    }

    // task -> worker
    std::vector<int> assign(static_cast<std::size_t>(cols), -1);
    for (int j = 1; j <= m; ++j) {
        const int task_1b = p[static_cast<std::size_t>(j)];
        if (task_1b > 0 && task_1b <= n) {
            assign[static_cast<std::size_t>(task_1b - 1)] = j - 1;
        }
    }
    return assign;
}

void assignment_to_matrix(const std::vector<int>& assign, int rows, int cols, std::vector<double>& X) {
    X.assign(static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols), 0.0);
    for (int c = 0; c < cols; ++c) {
        const int r = assign[static_cast<std::size_t>(c)];
        if (r >= 0) X[dense_idx(r, c, cols)] = 1.0;
    }
}

double dot(const std::vector<double>& a, const std::vector<double>& b) {
    double out = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) out += a[i] * b[i];
    return out;
}

// Return A*X where:
// A is nL x nL, X is nL x nS, result is nL x nS.
void mul_A_X(const std::vector<double>& A, const std::vector<double>& X, int nL, int nS, std::vector<double>& AX) {
    AX.assign(static_cast<std::size_t>(nL) * static_cast<std::size_t>(nS), 0.0);
    for (int a = 0; a < nL; ++a) {
        for (int b = 0; b < nL; ++b) {
            const double av = A[dense_idx(a, b, nL)];
            if (av == 0.0) continue;
            const std::size_t ax_base = dense_idx(a, 0, nS);
            const std::size_t bx_base = dense_idx(b, 0, nS);
            for (int i = 0; i < nS; ++i) {
                AX[ax_base + static_cast<std::size_t>(i)] += av * X[bx_base + static_cast<std::size_t>(i)];
            }
        }
    }
}

// Return M*B where:
// M is nL x nS, B is nS x nS, result is nL x nS.
void mul_M_B(const std::vector<double>& M, const std::vector<double>& B, int nL, int nS, std::vector<double>& MB) {
    MB.assign(static_cast<std::size_t>(nL) * static_cast<std::size_t>(nS), 0.0);
    for (int a = 0; a < nL; ++a) {
        for (int i = 0; i < nS; ++i) {
            const double mv = M[dense_idx(a, i, nS)];
            if (mv == 0.0) continue;
            for (int j = 0; j < nS; ++j) {
                MB[dense_idx(a, j, nS)] += mv * B[dense_idx(i, j, nS)];
            }
        }
    }
}

double objective_score(const WeightedGraph& large,
                       const WeightedGraph& small,
                       const std::vector<double>& X) {
    const int nL = large.n;
    const int nS = small.n;

    double node_term = 0.0;
    for (int a = 0; a < nL; ++a) {
        const double lw = large.node[static_cast<std::size_t>(a)];
        for (int i = 0; i < nS; ++i) {
            node_term += X[dense_idx(a, i, nS)] * lw * small.node[static_cast<std::size_t>(i)];
        }
    }

    std::vector<double> AX;
    std::vector<double> AXB;
    mul_A_X(large.edge, X, nL, nS, AX);
    mul_M_B(AX, small.edge, nL, nS, AXB);
    const double edge_term = dot(AXB, X);

    return node_term + edge_term;
}

// Quadratic term Q(U, V) = <(A*U*B), V>.
double quad_bilinear(const WeightedGraph& large,
                     const WeightedGraph& small,
                     const std::vector<double>& U,
                     const std::vector<double>& V) {
    const int nL = large.n;
    const int nS = small.n;
    std::vector<double> AU;
    std::vector<double> AUB;
    mul_A_X(large.edge, U, nL, nS, AU);
    mul_M_B(AU, small.edge, nL, nS, AUB);
    return dot(AUB, V);
}

void gradient_score(const WeightedGraph& large,
                    const WeightedGraph& small,
                    const std::vector<double>& X,
                    std::vector<double>& G) {
    const int nL = large.n;
    const int nS = small.n;
    G.assign(static_cast<std::size_t>(nL) * static_cast<std::size_t>(nS), 0.0);

    // Node term gradient.
    for (int a = 0; a < nL; ++a) {
        const double lw = large.node[static_cast<std::size_t>(a)];
        for (int i = 0; i < nS; ++i) {
            G[dense_idx(a, i, nS)] = lw * small.node[static_cast<std::size_t>(i)];
        }
    }

    // Edge term gradient:
    // grad = 2 * (A*X*B) for symmetric A and B.
    std::vector<double> AX;
    std::vector<double> AXB;
    mul_A_X(large.edge, X, nL, nS, AX);
    mul_M_B(AX, small.edge, nL, nS, AXB);
    for (std::size_t k = 0; k < G.size(); ++k) G[k] += 2.0 * AXB[k];
}

// Solve one partition with Frank-Wolfe and return a discrete rounded score.
double solve_partition_fw_score(const WeightedGraph& large,
                                const WeightedGraph& small,
                                int max_iters,
                                double tol) {
    const int rows = large.n;
    const int cols = small.n;
    if (cols == 0) return 0.0;
    if (rows < cols) return -infinity_cost();
    if (max_iters <= 0) max_iters = 1;
    if (!(tol > 0.0)) tol = 1e-6;

    // Initialize from node affinity only.
    std::vector<double> node_aff(static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols), 0.0);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            node_aff[dense_idx(r, c, cols)] =
                large.node[static_cast<std::size_t>(r)] * small.node[static_cast<std::size_t>(c)];
        }
    }
    std::vector<int> init_assign = hungarian_max_rect(node_aff, rows, cols);
    std::vector<double> X;
    assignment_to_matrix(init_assign, rows, cols, X);

    std::vector<double> G;
    std::vector<double> S;
    std::vector<double> D(static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols), 0.0);

    for (int it = 0; it < max_iters; ++it) {
        gradient_score(large, small, X, G);

        // Linear oracle: maximize <G, S> over rectangular assignment polytope.
        const std::vector<int> assign = hungarian_max_rect(G, rows, cols);
        assignment_to_matrix(assign, rows, cols, S);

        for (std::size_t k = 0; k < D.size(); ++k) D[k] = S[k] - X[k];
        const double fw_gap = dot(G, D);  // ascent gap for maximization
        if (fw_gap <= tol) break;

        const double a = fw_gap;
        const double b = quad_bilinear(large, small, D, D);

        // Maximize f(X + gamma D) on gamma in [0,1]:
        // f = f0 + a*gamma + b*gamma^2
        double gamma = 0.0;
        if (b < 0.0) {
            gamma = std::clamp(-a / (2.0 * b), 0.0, 1.0);
        } else {
            gamma = (a + b > 0.0) ? 1.0 : 0.0;
        }
        if (gamma <= 1e-12) break;

        for (std::size_t k = 0; k < X.size(); ++k) X[k] += gamma * D[k];
    }

    // Round relaxed X to an injective mapping using a final assignment on X entries.
    const std::vector<int> final_assign = hungarian_max_rect(X, rows, cols);
    std::vector<double> X_round;
    assignment_to_matrix(final_assign, rows, cols, X_round);
    return objective_score(large, small, X_round);
}

}  // namespace

double QAP_cost_fw(const std::vector<int>& partition_mapping_primary,
                   const std::vector<int>& partition_mapping_extra,
                   const CircuitGraph& circuit_graph,
                   const std::vector<const TopologyGraph*>& hardwares,
                   const std::vector<int>& hardware_placement,
                   int max_iters,
                   double tol,
                   bool parallel_partitions,
                   bool use_routed_hardware_edges) {
    const int n = static_cast<int>(circuit_graph.num_qubits());
    if (static_cast<int>(partition_mapping_primary.size()) != n ||
        static_cast<int>(partition_mapping_extra.size()) != n) {
        throw std::invalid_argument(
            "QAP_cost_fw(): partition_mapping_primary and partition_mapping_extra must have length num_qubits");
    }
    if (hardwares.empty()) {
        throw std::invalid_argument("QAP_cost_fw(): hardwares must be non-empty");
    }

    const int num_parts = hardware_placement.empty()
                              ? static_cast<int>(hardwares.size())
                              : static_cast<int>(hardware_placement.size());
    if (num_parts <= 0) {
        throw std::invalid_argument("QAP_cost_fw(): number of partitions must be positive");
    }

    // Partition -> hardware index map.
    std::vector<int> part_to_hw(static_cast<std::size_t>(num_parts), 0);
    if (hardware_placement.empty()) {
        for (int p = 0; p < num_parts; ++p) part_to_hw[static_cast<std::size_t>(p)] = p;
    } else {
        part_to_hw = hardware_placement;
    }
    for (int p = 0; p < num_parts; ++p) {
        const int hw = part_to_hw[static_cast<std::size_t>(p)];
        if (hw < 0 || hw >= static_cast<int>(hardwares.size())) {
            throw std::invalid_argument("QAP_cost_fw(): hardware_placement contains out-of-range hardware index");
        }
        if (hardwares[static_cast<std::size_t>(hw)] == nullptr) {
            throw std::invalid_argument("QAP_cost_fw(): hardwares contains null topology pointer");
        }
    }

    // Validate partition ids.
    for (int q = 0; q < n; ++q) {
        const int p = partition_mapping_primary[static_cast<std::size_t>(q)];
        if (p < 0 || p >= num_parts) {
            throw std::invalid_argument("QAP_cost_fw(): primary mapping contains invalid partition id");
        }
        const int x = partition_mapping_extra[static_cast<std::size_t>(q)];
        if (x >= num_parts) {
            throw std::invalid_argument("QAP_cost_fw(): extra mapping contains invalid partition id");
        }
    }

    // Prebuild hardware graphs in [0,1].
    std::vector<WeightedGraph> hw_graphs(hardwares.size());
    for (std::size_t i = 0; i < hardwares.size(); ++i) {
        hw_graphs[i] = build_hardware_graph_prob(*hardwares[i], use_routed_hardware_edges);
    }

    // Build partition circuit subgraphs.
    std::vector<PartitionProblem> probs(static_cast<std::size_t>(num_parts));
    std::vector<std::unordered_map<int, int>> part_local(static_cast<std::size_t>(num_parts));
    const auto& oneq = circuit_graph.node_weights();
    for (int q = 0; q < n; ++q) {
        const int p = partition_mapping_primary[static_cast<std::size_t>(q)];
        auto& map_p = part_local[static_cast<std::size_t>(p)];
        if (map_p.find(q) == map_p.end()) {
            const int lid = static_cast<int>(probs[static_cast<std::size_t>(p)].node_weight.size());
            map_p.emplace(q, lid);
            probs[static_cast<std::size_t>(p)].node_weight.push_back(
                static_cast<double>(oneq[static_cast<std::size_t>(q)]));
        }

        const int x = partition_mapping_extra[static_cast<std::size_t>(q)];
        if (x >= 0 && x != p) {
            auto& map_x = part_local[static_cast<std::size_t>(x)];
            if (map_x.find(q) == map_x.end()) {
                const int lid = static_cast<int>(probs[static_cast<std::size_t>(x)].node_weight.size());
                map_x.emplace(q, lid);
                probs[static_cast<std::size_t>(x)].node_weight.push_back(
                    static_cast<double>(oneq[static_cast<std::size_t>(q)]));
            }
        }
    }

    const auto& C = circuit_graph.connectivity();
    for (std::uint32_t u = 0; u < C.num_qubits; ++u) {
        for (std::uint32_t ei = C.row_ptr[u]; ei < C.row_ptr[u + 1]; ++ei) {
            const int v = static_cast<int>(C.col_idx[ei]);
            if (static_cast<int>(u) >= v) continue;  // unique undirected edge

            const int up = partition_mapping_primary[static_cast<std::size_t>(u)];
            const int ux = partition_mapping_extra[static_cast<std::size_t>(u)];
            const int vp = partition_mapping_primary[static_cast<std::size_t>(v)];
            const int vx = partition_mapping_extra[static_cast<std::size_t>(v)];

            int shared0 = -1;
            int shared1 = -1;
            if (up == vp) shared0 = up;
            if (ux >= 0 && ux == vp) {
                if (shared0 < 0) shared0 = ux;
                else if (shared0 != ux) shared1 = ux;
            }
            if (vx >= 0 && up == vx) {
                if (shared0 < 0) shared0 = vx;
                else if (shared0 != vx && shared1 < 0) shared1 = vx;
            }
            if (ux >= 0 && vx >= 0 && ux == vx) {
                if (shared0 < 0) shared0 = ux;
                else if (shared0 != ux && shared1 < 0) shared1 = ux;
            }

            const std::uint32_t edge_id = C.edge_id[static_cast<std::size_t>(ei)];
            if (edge_id >= C.num_edge_ids()) {
                throw std::runtime_error("QAP_cost_fw(): connectivity edge id is out of range");
            }
            const double w = static_cast<double>(
                C.edge_original_row_ptr[static_cast<std::size_t>(edge_id) + 1u] -
                C.edge_original_row_ptr[static_cast<std::size_t>(edge_id)]);
            auto add_edge_to_part = [&](int part_id) {
                auto& prob = probs[static_cast<std::size_t>(part_id)];
                const auto& idx = part_local[static_cast<std::size_t>(part_id)];
                const int lu = idx.at(static_cast<int>(u));
                const int lv = idx.at(v);
                prob.edges.push_back(PartitionProblem::WeightedEdge{lu, lv, w});
            };

            if (shared0 >= 0) add_edge_to_part(shared0);
            if (shared1 >= 0) add_edge_to_part(shared1);
        }
    }

    double total_squared_infidelity = 0.0;
    int infeasible_flag = 0;

#if defined(OPTCORE_HAS_OPENMP) && OPTCORE_HAS_OPENMP
    if (parallel_partitions && num_parts > 1) {
#pragma omp parallel for reduction(+:total_squared_infidelity)
        for (int p = 0; p < num_parts; ++p) {
            const WeightedGraph circ_g = build_circuit_partition_graph(probs[static_cast<std::size_t>(p)]);
            const WeightedGraph& hw_g = hw_graphs[static_cast<std::size_t>(part_to_hw[static_cast<std::size_t>(p)])];

            const WeightedGraph* large = &hw_g;
            const WeightedGraph* small = &circ_g;
            if (circ_g.n > hw_g.n) {
                large = &circ_g;
                small = &hw_g;
            }

            const double s = solve_partition_fw_score(*large, *small, max_iters, tol);
            if (!std::isfinite(s)) {
#pragma omp atomic write
                infeasible_flag = 1;
            } else {
                const double placement_fidelity = std::clamp(std::exp(s), 0.0, 1.0);
                const double placement_infidelity = 1.0 - placement_fidelity;
                total_squared_infidelity += placement_infidelity * placement_infidelity;
            }
        }
    } else
#endif
    {
#if !(defined(OPTCORE_HAS_OPENMP) && OPTCORE_HAS_OPENMP)
        (void)parallel_partitions;
#endif
        for (int p = 0; p < num_parts; ++p) {
            const WeightedGraph circ_g = build_circuit_partition_graph(probs[static_cast<std::size_t>(p)]);
            const WeightedGraph& hw_g = hw_graphs[static_cast<std::size_t>(part_to_hw[static_cast<std::size_t>(p)])];

            const WeightedGraph* large = &hw_g;
            const WeightedGraph* small = &circ_g;
            if (circ_g.n > hw_g.n) {
                large = &circ_g;
                small = &hw_g;
            }

            const double s = solve_partition_fw_score(*large, *small, max_iters, tol);
            if (!std::isfinite(s)) return 0.0;
            const double placement_fidelity = std::clamp(std::exp(s), 0.0, 1.0);
            const double placement_infidelity = 1.0 - placement_fidelity;
            total_squared_infidelity += placement_infidelity * placement_infidelity;
        }
    }

    if (infeasible_flag != 0) return 0.0;
    const double mean_squared_infidelity =
        total_squared_infidelity / static_cast<double>(num_parts);
    return std::max(0.0, 1.0 - mean_squared_infidelity);
}

}  // namespace QAP_fw
