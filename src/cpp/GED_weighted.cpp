#include "GED_weighted.hpp"

#include "circuit_graph.hpp"
#include "topology_graph.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#if defined(OPTCORE_HAS_OPENMP) && OPTCORE_HAS_OPENMP
#include <omp.h>
#endif

namespace GEDWeighted {

namespace {

inline std::uint64_t undirected_key(int a, int b) {
    const int lo = (a < b) ? a : b;
    const int hi = (a < b) ? b : a;
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(lo)) << 32) |
           static_cast<std::uint32_t>(hi);
}

struct DenseSimpleGraph {
    int n = 0;
    int m = 0;
    int words = 0;
    std::vector<int> degree;
    std::vector<std::uint64_t> adj_bits;  // row-major: n * words

    bool has_edge(int u, int v) const noexcept {
        const std::size_t base = static_cast<std::size_t>(u) * static_cast<std::size_t>(words);
        const std::size_t word = static_cast<std::size_t>(v >> 6);
        const std::uint64_t bit = 1ull << static_cast<unsigned>(v & 63);
        return (adj_bits[base + word] & bit) != 0ull;
    }

    void set_edge(int u, int v) noexcept {
        const std::size_t ubase = static_cast<std::size_t>(u) * static_cast<std::size_t>(words);
        const std::size_t vbase = static_cast<std::size_t>(v) * static_cast<std::size_t>(words);
        const std::size_t uw = static_cast<std::size_t>(v >> 6);
        const std::size_t vw = static_cast<std::size_t>(u >> 6);
        const std::uint64_t ubit = 1ull << static_cast<unsigned>(v & 63);
        const std::uint64_t vbit = 1ull << static_cast<unsigned>(u & 63);
        adj_bits[ubase + uw] |= ubit;
        adj_bits[vbase + vw] |= vbit;
    }
};

DenseSimpleGraph build_dense_graph(const std::vector<std::pair<int, int>>& edges) {
    DenseSimpleGraph out;

    std::vector<int> nodes;
    nodes.reserve(edges.size() * 2u);
    for (const auto& e : edges) {
        if (e.first == e.second) continue;
        nodes.push_back(e.first);
        nodes.push_back(e.second);
    }
    std::sort(nodes.begin(), nodes.end());
    nodes.erase(std::unique(nodes.begin(), nodes.end()), nodes.end());

    out.n = static_cast<int>(nodes.size());
    if (out.n == 0) return out;

    out.words = (out.n + 63) >> 6;
    out.degree.assign(static_cast<std::size_t>(out.n), 0);
    out.adj_bits.assign(static_cast<std::size_t>(out.n) * static_cast<std::size_t>(out.words), 0ull);

    std::unordered_map<int, int> id;
    id.reserve(nodes.size() * 2u);
    for (int i = 0; i < out.n; ++i) {
        id.emplace(nodes[static_cast<std::size_t>(i)], i);
    }

    std::unordered_set<std::uint64_t> uniq;
    uniq.reserve(edges.size() * 2u + 1u);
    for (const auto& e : edges) {
        if (e.first == e.second) continue;
        const int u = id[e.first];
        const int v = id[e.second];
        if (u == v) continue;
        const std::uint64_t key = undirected_key(u, v);
        if (uniq.insert(key).second) {
            out.set_edge(u, v);
            out.degree[static_cast<std::size_t>(u)] += 1;
            out.degree[static_cast<std::size_t>(v)] += 1;
            out.m += 1;
        }
    }
    return out;
}

double infinity_cost() {
    return std::numeric_limits<double>::infinity();
}

using HardwareWeightedCache = TopologyGraph::WeightedGedCache;

struct PartitionProblem {
    struct WeightedEdge {
        int u = 0;
        int v = 0;
        double w = 0.0;  // connectivity multiplicity for this circuit edge
    };

    std::vector<double> node_weight;  // number of 1Q gates per circuit node
    std::vector<WeightedEdge> edges;  // undirected local-node edges
    std::vector<std::vector<std::pair<int, double>>> adj; // local adjacency (neighbor, edge weight)
    double total_edge_weight = 0.0;
};

double solve_partition_max_score(const PartitionProblem& prob, const HardwareWeightedCache& hw) {
    const int n2 = static_cast<int>(prob.node_weight.size());
    const int n1 = hw.n;
    if (n2 == 0) return 0.0;
    if (n2 > n1) return -std::numeric_limits<double>::infinity();
    if (prob.total_edge_weight > 0.0 && !std::isfinite(hw.best_edge_score)) {
        return -std::numeric_limits<double>::infinity();
    }

    // Degree in circuit subgraph (undirected).
    std::vector<int> circ_deg(static_cast<std::size_t>(n2), 0);
    for (int u = 0; u < n2; ++u) {
        circ_deg[static_cast<std::size_t>(u)] = static_cast<int>(prob.adj[static_cast<std::size_t>(u)].size());
    }

    // Search order: dense/high-weight circuit nodes first.
    std::vector<int> order2(static_cast<std::size_t>(n2), 0);
    for (int i = 0; i < n2; ++i) order2[static_cast<std::size_t>(i)] = i;
    std::sort(order2.begin(), order2.end(), [&](int a, int b) {
        const double wa = prob.node_weight[static_cast<std::size_t>(a)];
        const double wb = prob.node_weight[static_cast<std::size_t>(b)];
        if (circ_deg[static_cast<std::size_t>(a)] != circ_deg[static_cast<std::size_t>(b)]) {
            return circ_deg[static_cast<std::size_t>(a)] > circ_deg[static_cast<std::size_t>(b)];
        }
        if (wa != wb) return wa > wb;
        return a < b;
    });

    std::vector<int> pos2(static_cast<std::size_t>(n2), -1);
    for (int d = 0; d < n2; ++d) pos2[static_cast<std::size_t>(order2[static_cast<std::size_t>(d)])] = d;

    std::vector<std::vector<std::pair<int, double>>> prev_edges(static_cast<std::size_t>(n2));
    std::vector<double> decided_edge_prefix(static_cast<std::size_t>(n2) + 1u, 0.0);
    for (int d = 0; d < n2; ++d) {
        const int u = order2[static_cast<std::size_t>(d)];
        auto& pe = prev_edges[static_cast<std::size_t>(d)];
        pe.reserve(prob.adj[static_cast<std::size_t>(u)].size());
        double add_w = 0.0;
        for (const auto& nw : prob.adj[static_cast<std::size_t>(u)]) {
            const int v = nw.first;
            const int pd = pos2[static_cast<std::size_t>(v)];
            if (pd >= 0 && pd < d) {
                pe.emplace_back(pd, nw.second);
                add_w += nw.second;
            }
        }
        decided_edge_prefix[static_cast<std::size_t>(d + 1)] =
            decided_edge_prefix[static_cast<std::size_t>(d)] + add_w;
    }

    std::vector<double> rem_node_weight_from_depth(static_cast<std::size_t>(n2) + 1u, 0.0);
    for (int d = n2 - 1; d >= 0; --d) {
        rem_node_weight_from_depth[static_cast<std::size_t>(d)] =
            rem_node_weight_from_depth[static_cast<std::size_t>(d + 1)] +
            prob.node_weight[static_cast<std::size_t>(order2[static_cast<std::size_t>(d)])];
    }

    std::vector<int> map_depth_to_hw(static_cast<std::size_t>(n2), -1);
    std::vector<unsigned char> used1(static_cast<std::size_t>(n1), 0);

    auto eval_increment = [&](int depth, int cand_hw, bool& feasible) -> double {
        feasible = true;
        const int u2 = order2[static_cast<std::size_t>(depth)];
        double inc = prob.node_weight[static_cast<std::size_t>(u2)] *
                     hw.node_x[static_cast<std::size_t>(cand_hw)];

        for (const auto& e : prev_edges[static_cast<std::size_t>(depth)]) {
            const int pd = e.first;
            const double w = e.second;
            const int other_hw = map_depth_to_hw[static_cast<std::size_t>(pd)];
            const double hw_edge = hw.edge_score[static_cast<std::size_t>(cand_hw) * static_cast<std::size_t>(n1) +
                                                 static_cast<std::size_t>(other_hw)];
            if (!std::isfinite(hw_edge)) {
                feasible = false;
                return 0.0;
            }
            inc += w * hw_edge;
        }
        return inc;
    };

    auto best_unused_node_x = [&]() -> double {
        double best = -std::numeric_limits<double>::infinity();
        for (int i = 0; i < n1; ++i) {
            if (used1[static_cast<std::size_t>(i)] == 0) {
                best = std::max(best, hw.node_x[static_cast<std::size_t>(i)]);
            }
        }
        return best;
    };

    // Greedy incumbent for tighter pruning.
    double best = -std::numeric_limits<double>::infinity();
    {
        std::vector<int> greedy_map(static_cast<std::size_t>(n2), -1);
        std::vector<unsigned char> greedy_used(static_cast<std::size_t>(n1), 0);
        double score = 0.0;
        bool ok = true;
        for (int d = 0; d < n2; ++d) {
            const int u2 = order2[static_cast<std::size_t>(d)];
            int best_c = -1;
            double best_inc = -std::numeric_limits<double>::infinity();
            for (int cand : hw.order) {
                if (greedy_used[static_cast<std::size_t>(cand)] != 0) continue;
                if (hw.degree[static_cast<std::size_t>(cand)] < circ_deg[static_cast<std::size_t>(u2)]) continue;

                bool feasible = true;
                double inc = prob.node_weight[static_cast<std::size_t>(u2)] *
                             hw.node_x[static_cast<std::size_t>(cand)];
                for (const auto& e : prev_edges[static_cast<std::size_t>(d)]) {
                    const int other_hw = greedy_map[static_cast<std::size_t>(e.first)];
                    const double hw_edge = hw.edge_score[static_cast<std::size_t>(cand) * static_cast<std::size_t>(n1) +
                                                         static_cast<std::size_t>(other_hw)];
                    if (!std::isfinite(hw_edge)) {
                        feasible = false;
                        break;
                    }
                    inc += e.second * hw_edge;
                }
                if (feasible && inc > best_inc) {
                    best_inc = inc;
                    best_c = cand;
                }
            }
            if (best_c < 0) {
                ok = false;
                break;
            }
            greedy_used[static_cast<std::size_t>(best_c)] = 1;
            greedy_map[static_cast<std::size_t>(d)] = best_c;
            score += best_inc;
        }
        if (ok) best = score;
    }

    std::function<void(int, double)> dfs = [&](int depth, double cur) {
        if (depth == n2) {
            if (cur > best) best = cur;
            return;
        }

        const double rem_nodes_w = rem_node_weight_from_depth[static_cast<std::size_t>(depth)];
        const double best_x = best_unused_node_x();
        if (!std::isfinite(best_x)) return;

        const double rem_edges_w = prob.total_edge_weight - decided_edge_prefix[static_cast<std::size_t>(depth)];
        double ub = cur + rem_nodes_w * best_x;
        if (rem_edges_w > 0.0) {
            if (!std::isfinite(hw.best_edge_score)) return;
            ub += rem_edges_w * hw.best_edge_score;
        }
        if (ub <= best + 1e-12) return;

        const int u2 = order2[static_cast<std::size_t>(depth)];
        for (int cand : hw.order) {
            if (used1[static_cast<std::size_t>(cand)] != 0) continue;
            if (hw.degree[static_cast<std::size_t>(cand)] < circ_deg[static_cast<std::size_t>(u2)]) continue;

            bool feasible = false;
            const double inc = eval_increment(depth, cand, feasible);
            if (!feasible) continue;

            used1[static_cast<std::size_t>(cand)] = 1;
            map_depth_to_hw[static_cast<std::size_t>(depth)] = cand;
            dfs(depth + 1, cur + inc);
            map_depth_to_hw[static_cast<std::size_t>(depth)] = -1;
            used1[static_cast<std::size_t>(cand)] = 0;
        }
    };

    if (!std::isfinite(best)) {
        // No greedy feasible mapping. DFS may still find one.
        best = -std::numeric_limits<double>::infinity();
    }
    dfs(0, 0.0);
    return best;
}

}  // namespace

double ged_no_node_insertion_free_deletions(const std::vector<std::pair<int, int>>& edges1,
                                            const std::vector<std::pair<int, int>>& edges2,
                                            double edge_ins_cost) {
    if (edge_ins_cost < 0.0) {
        throw std::invalid_argument("ged_no_node_insertion_free_deletions(): edge_ins_cost must be >= 0");
    }

    const DenseSimpleGraph g1 = build_dense_graph(edges1);
    const DenseSimpleGraph g2 = build_dense_graph(edges2);

    if (g2.n > g1.n) return infinity_cost();  // node insertion is forbidden
    if (g2.m == 0) return 0.0;

    // Trivial fast paths.
    if (g1.m == 0) return edge_ins_cost * static_cast<double>(g2.m);
    if (g2.n <= 1) return 0.0;

    // Order G2 nodes by descending degree (and id tie-break).
    std::vector<int> order2(static_cast<std::size_t>(g2.n), 0);
    for (int i = 0; i < g2.n; ++i) order2[static_cast<std::size_t>(i)] = i;
    std::sort(order2.begin(), order2.end(), [&](int a, int b) {
        const int da = g2.degree[static_cast<std::size_t>(a)];
        const int db = g2.degree[static_cast<std::size_t>(b)];
        if (da != db) return da > db;
        return a < b;
    });

    // Candidate iteration order for G1 (high-degree first).
    std::vector<int> order1(static_cast<std::size_t>(g1.n), 0);
    for (int i = 0; i < g1.n; ++i) order1[static_cast<std::size_t>(i)] = i;
    std::sort(order1.begin(), order1.end(), [&](int a, int b) {
        const int da = g1.degree[static_cast<std::size_t>(a)];
        const int db = g1.degree[static_cast<std::size_t>(b)];
        if (da != db) return da > db;
        return a < b;
    });

    // For each depth d, store previous depths p<d where edge exists in G2.
    std::vector<std::vector<int>> prev_edge_depths(static_cast<std::size_t>(g2.n));
    std::vector<int> decided_edges_prefix(static_cast<std::size_t>(g2.n) + 1u, 0);
    for (int d = 0; d < g2.n; ++d) {
        const int u2 = order2[static_cast<std::size_t>(d)];
        auto& vec = prev_edge_depths[static_cast<std::size_t>(d)];
        vec.reserve(static_cast<std::size_t>(g2.degree[static_cast<std::size_t>(u2)]));
        for (int p = 0; p < d; ++p) {
            const int v2 = order2[static_cast<std::size_t>(p)];
            if (g2.has_edge(u2, v2)) vec.push_back(p);
        }
        decided_edges_prefix[static_cast<std::size_t>(d + 1)] =
            decided_edges_prefix[static_cast<std::size_t>(d)] + static_cast<int>(vec.size());
    }

    // Greedy initialization to get a good incumbent quickly.
    int best_matched = 0;
    {
        std::vector<int> map_depth_to_g1(static_cast<std::size_t>(g2.n), -1);
        std::vector<unsigned char> used1(static_cast<std::size_t>(g1.n), 0);
        int matched = 0;
        for (int d = 0; d < g2.n; ++d) {
            int best_candidate = -1;
            int best_inc = -1;
            for (int cand : order1) {
                if (used1[static_cast<std::size_t>(cand)] != 0) continue;
                int inc = 0;
                for (int p : prev_edge_depths[static_cast<std::size_t>(d)]) {
                    const int mapped_p = map_depth_to_g1[static_cast<std::size_t>(p)];
                    if (mapped_p >= 0 && g1.has_edge(cand, mapped_p)) ++inc;
                }
                if (inc > best_inc) {
                    best_inc = inc;
                    best_candidate = cand;
                }
            }
            if (best_candidate < 0) break;
            used1[static_cast<std::size_t>(best_candidate)] = 1;
            map_depth_to_g1[static_cast<std::size_t>(d)] = best_candidate;
            matched += best_inc;
        }
        best_matched = matched;
        if (best_matched == g2.m) return 0.0;
    }

    std::vector<int> map_depth_to_g1(static_cast<std::size_t>(g2.n), -1);
    std::vector<unsigned char> used1(static_cast<std::size_t>(g1.n), 0);

    std::function<void(int, int)> dfs = [&](int depth, int matched) {
        if (depth == g2.n) {
            if (matched > best_matched) best_matched = matched;
            return;
        }

        // Upper bound: all undecided G2 edges could match.
        const int undecided = g2.m - decided_edges_prefix[static_cast<std::size_t>(depth)];
        if (matched + undecided <= best_matched) return;

        for (int cand : order1) {
            if (used1[static_cast<std::size_t>(cand)] != 0) continue;

            int inc = 0;
            for (int p : prev_edge_depths[static_cast<std::size_t>(depth)]) {
                const int mapped_p = map_depth_to_g1[static_cast<std::size_t>(p)];
                if (mapped_p >= 0 && g1.has_edge(cand, mapped_p)) ++inc;
            }

            const int new_matched = matched + inc;
            const int undecided_after = g2.m - decided_edges_prefix[static_cast<std::size_t>(depth + 1)];
            if (new_matched + undecided_after <= best_matched) continue;

            used1[static_cast<std::size_t>(cand)] = 1;
            map_depth_to_g1[static_cast<std::size_t>(depth)] = cand;
            dfs(depth + 1, new_matched);
            map_depth_to_g1[static_cast<std::size_t>(depth)] = -1;
            used1[static_cast<std::size_t>(cand)] = 0;

            if (best_matched == g2.m) return;  // global optimum reached
        }
    };

    dfs(0, 0);
    const int missing = g2.m - best_matched;
    return edge_ins_cost * static_cast<double>(missing);
}

double GED_cost(const std::vector<int>& partition_mapping_primary,
                const std::vector<int>& partition_mapping_extra,
                const CircuitGraph& circuit_graph,
                const std::vector<const TopologyGraph*>& hardwares,
                const std::vector<int>& hardware_placement,
                bool parallel_partitions) {
    const int n = static_cast<int>(circuit_graph.num_qubits());
    if (static_cast<int>(partition_mapping_primary.size()) != n ||
        static_cast<int>(partition_mapping_extra.size()) != n) {
        throw std::invalid_argument(
            "GED_cost(): partition_mapping_primary and partition_mapping_extra must have length num_qubits");
    }
    if (hardwares.empty()) {
        throw std::invalid_argument("GED_cost(): hardwares must be non-empty");
    }

    const int num_parts = hardware_placement.empty()
                              ? static_cast<int>(hardwares.size())
                              : static_cast<int>(hardware_placement.size());
    if (num_parts <= 0) {
        throw std::invalid_argument("GED_cost(): number of partitions must be positive");
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
            throw std::invalid_argument("GED_cost(): hardware_placement contains out-of-range hardware index");
        }
        if (hardwares[static_cast<std::size_t>(hw)] == nullptr) {
            throw std::invalid_argument("GED_cost(): hardwares contains null topology pointer");
        }
    }

    // Validate partition ids.
    for (int q = 0; q < n; ++q) {
        const int p = partition_mapping_primary[static_cast<std::size_t>(q)];
        if (p < 0 || p >= num_parts) {
            throw std::invalid_argument("GED_cost(): primary mapping contains invalid partition id");
        }
        const int x = partition_mapping_extra[static_cast<std::size_t>(q)];
        if (x >= num_parts) {
            throw std::invalid_argument("GED_cost(): extra mapping contains invalid partition id");
        }
    }

    // Hardware caches are precomputed and stored in each TopologyGraph.
    std::vector<const HardwareWeightedCache*> hw_cache(hardwares.size(), nullptr);
    for (std::size_t i = 0; i < hardwares.size(); ++i) {
        hw_cache[i] = &hardwares[i]->weighted_ged_cache();
    }

    // Build per-partition node sets and local-index maps.
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

    for (int p = 0; p < num_parts; ++p) {
        const int np = static_cast<int>(probs[static_cast<std::size_t>(p)].node_weight.size());
        probs[static_cast<std::size_t>(p)].adj.assign(static_cast<std::size_t>(np), {});
    }

    // Build weighted partition edge sets from circuit connectivity.
    const auto& C = circuit_graph.connectivity();
    for (std::uint32_t u = 0; u < C.num_qubits; ++u) {
        for (std::uint32_t ei = C.row_ptr[u]; ei < C.row_ptr[u + 1]; ++ei) {
            const int v = static_cast<int>(C.col_idx[ei]);
            if (static_cast<int>(u) >= v) continue;  // undirected unique edge

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

            const double w = static_cast<double>(C.weight[static_cast<std::size_t>(ei)]);
            auto add_edge_to_part = [&](int part_id) {
                auto& prob = probs[static_cast<std::size_t>(part_id)];
                const auto& idx = part_local[static_cast<std::size_t>(part_id)];
                const int lu = idx.at(static_cast<int>(u));
                const int lv = idx.at(v);
                prob.edges.push_back(PartitionProblem::WeightedEdge{lu, lv, w});
                prob.adj[static_cast<std::size_t>(lu)].push_back({lv, w});
                prob.adj[static_cast<std::size_t>(lv)].push_back({lu, w});
                prob.total_edge_weight += w;
            };

            if (shared0 >= 0) add_edge_to_part(shared0);
            if (shared1 >= 0) add_edge_to_part(shared1);
        }
    }

    // Maximize weighted score per partition.
    // Convert mean cost to bounded score g = exp(-mean_cost) in [0, 1].
    double total_cost = 0.0;
    int infeasible_flag = 0;

#if defined(OPTCORE_HAS_OPENMP) && OPTCORE_HAS_OPENMP
    if (parallel_partitions && num_parts > 1) {
#pragma omp parallel for reduction(+:total_cost)
        for (int p = 0; p < num_parts; ++p) {
            const int hw_id = part_to_hw[static_cast<std::size_t>(p)];
            const double best_score =
                solve_partition_max_score(probs[static_cast<std::size_t>(p)], *hw_cache[static_cast<std::size_t>(hw_id)]);
            if (!std::isfinite(best_score)) {
#pragma omp atomic write
                infeasible_flag = 1;
            } else {
                total_cost += -best_score;
            }
        }
    } else
#endif
    {
#if !(defined(OPTCORE_HAS_OPENMP) && OPTCORE_HAS_OPENMP)
        (void)parallel_partitions;
#endif
        for (int p = 0; p < num_parts; ++p) {
            const int hw_id = part_to_hw[static_cast<std::size_t>(p)];
            const double best_score =
                solve_partition_max_score(probs[static_cast<std::size_t>(p)], *hw_cache[static_cast<std::size_t>(hw_id)]);
            if (!std::isfinite(best_score)) return 0.0;
            total_cost += -best_score;
        }
    }

    if (infeasible_flag != 0) return 0.0;
    const double mean_cost = total_cost / static_cast<double>(num_parts);
    return std::exp(-mean_cost);
}

}  // namespace GEDWeighted
