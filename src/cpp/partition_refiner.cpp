#include "partition_refiner.hpp"

#include "circuit_graph.hpp"
#include "QAP_fw.hpp"
#include "topology_graph.hpp"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <tuple>
#include <string>
#include <unordered_set>
#include <vector>
#include <utility>
#include <cstddef>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#if OPTCORE_HAS_OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

namespace {

struct Solution {
    std::vector<int> primary;
    std::vector<std::vector<int>> extra_blocks;
};

enum class MoveType { NONE, MOVE_PRIMARY, ADD_OVERLAP, REMOVE_OVERLAP, SWAP_PRIMARY };

struct MoveCandidate {
    MoveType type = MoveType::NONE;
    int v = -1;
    int to = -1;
    int with = -1;
    double d_cut = 0.0;
    double d_ov = 0.0;
    double d_fast = 0.0;
    bool valid = false;
};

struct OverlapCutTrace {
    std::vector<std::uint32_t> gate_cut_indices;
    // Wire cuts are represented as (insert_before_qc_data_index, qubit).
    std::vector<std::pair<std::uint32_t, std::uint32_t>> wire_cut_indices;
    // Wire cut label transitions are (insert_before_qc_data_index, qubit, left_label, right_label).
    std::vector<std::tuple<std::uint32_t, std::uint32_t, int, int>> wire_cut_partition_labels;
    double cost = 0.0;
};

struct OverlapSegmentUsage {
    int primary_segments = 0;
    int extra_segments = 0;
};

struct MultiOverlapEvaluation {
    OverlapCutTrace trace;
    std::vector<int> segment_counts;
};

static void validate_capacities(const std::vector<int>& capacities, std::uint32_t n) {
    if (capacities.empty()) {
        throw std::invalid_argument("refine_partition(): capacities must be non-empty");
    }
    for (std::size_t i = 0; i < capacities.size(); ++i) {
        if (capacities[i] <= 0) {
            throw std::invalid_argument(
                "refine_partition(): capacities must be > 0 (bad entry at index " + std::to_string(i) + ")");
        }
    }
    const long long sum_caps = std::accumulate(capacities.begin(), capacities.end(), 0LL);
    if (sum_caps < static_cast<long long>(n)) {
        throw std::invalid_argument("refine_partition(): sum(capacities) must be >= num_qubits");
    }
}

static bool contains_block(const std::vector<int>& blocks, int block) {
    return std::find(blocks.begin(), blocks.end(), block) != blocks.end();
}

static void add_extra_block(std::vector<int>& blocks, int block) {
    if (block >= 0 && !contains_block(blocks, block)) blocks.push_back(block);
}

static bool remove_extra_block(std::vector<int>& blocks, int block) {
    auto it = std::find(blocks.begin(), blocks.end(), block);
    if (it == blocks.end()) return false;
    blocks.erase(it);
    return true;
}

static bool shares_any_block(int ap, int ax, int bp, int bx) {
    if (ap == bp) return true;
    if (ax >= 0 && ax == bp) return true;
    if (bx >= 0 && bx == ap) return true;
    if (ax >= 0 && bx >= 0 && ax == bx) return true;
    return false;
}

static bool shares_any_block(int ap,
                             const std::vector<int>& a_extra,
                             int bp,
                             const std::vector<int>& b_extra) {
    if (ap == bp) return true;
    if (contains_block(a_extra, bp)) return true;
    if (contains_block(b_extra, ap)) return true;
    for (const int block : a_extra) {
        if (contains_block(b_extra, block)) return true;
    }
    return false;
}

static std::vector<int> first_extra_blocks(const std::vector<std::vector<int>>& extra_blocks) {
    std::vector<int> out(extra_blocks.size(), -1);
    for (std::size_t i = 0; i < extra_blocks.size(); ++i) {
        if (!extra_blocks[i].empty()) out[i] = extra_blocks[i].front();
    }
    return out;
}

static Solution relabel_solution(const Solution& sol, const std::vector<int>& label_map) {
    Solution out = sol;
    for (int& label : out.primary) {
        if (label >= 0 && static_cast<std::size_t>(label) < label_map.size()) {
            label = label_map[static_cast<std::size_t>(label)];
        }
    }
    for (auto& blocks : out.extra_blocks) {
        for (int& label : blocks) {
            if (label >= 0 && static_cast<std::size_t>(label) < label_map.size()) {
                label = label_map[static_cast<std::size_t>(label)];
            }
        }
        std::sort(blocks.begin(), blocks.end());
        blocks.erase(std::unique(blocks.begin(), blocks.end()), blocks.end());
    }
    return out;
}

template <typename T>
static std::string format_vector(const std::vector<T>& values) {
    std::string out = "[";
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i != 0) out += ",";
        out += std::to_string(values[i]);
    }
    out += "]";
    return out;
}

static std::string format_blocks(const std::vector<std::vector<int>>& blocks) {
    std::string out = "[";
    for (std::size_t i = 0; i < blocks.size(); ++i) {
        if (i != 0) out += ",";
        out += format_vector(blocks[i]);
    }
    out += "]";
    return out;
}

static double get_overlap_cost(const std::vector<int>& placement,
                               const std::vector<std::vector<int>>& extra_blocks,
                               int overlap_pos,
                               int overlap_partition,
                               const CircuitGraph::TemporalGateEvents& events) {

    if (overlap_pos < 0 || static_cast<std::size_t>(overlap_pos) >= placement.size()) {
        throw std::out_of_range("get_overlap_cost(): overlap_pos out of range");
    }
    if (placement.size() != static_cast<std::size_t>(events.num_qubits)) {
        throw std::invalid_argument("get_overlap_cost(): placement length must match temporal_events.num_qubits");
    }
    if (!extra_blocks.empty() && extra_blocks.size() != placement.size()) {
        throw std::invalid_argument("get_overlap_cost(): extra_blocks length must match placement length");
    }

    const int partition_b = placement[static_cast<std::size_t>(overlap_pos)];
    if (overlap_partition == partition_b) return 0.0;

    const auto b = events.row_ptr[static_cast<std::size_t>(overlap_pos)];
    const auto e = events.row_ptr[static_cast<std::size_t>(overlap_pos) + 1];
    if (e <= b) return 0.0;

    auto partner_has_extra_block = [&](std::uint32_t partner, int block) -> bool {
        const auto idx = static_cast<std::size_t>(partner);
        return !extra_blocks.empty() && contains_block(extra_blocks[idx], block);
    };

    auto partner_label = [&](std::uint32_t partner) -> int {
        const auto idx = static_cast<std::size_t>(partner);
        const int primary = placement[idx];
        if (primary == overlap_partition) return 1; // A
        if (primary == partition_b) return 2;       // B
        if (partner_has_extra_block(partner, overlap_partition)) return 1;
        if (partner_has_extra_block(partner, partition_b)) return 2;
        return 0;
    };

    const auto& partners = events.partners;
    const bool has_partner_weight = events.partner_weight.size() == partners.size();
    const double* partner_weight = has_partner_weight ? events.partner_weight.data() : nullptr;

    std::vector<int> labels;
    std::vector<double> weights;
    labels.reserve(static_cast<std::size_t>(e - b));
    weights.reserve(static_cast<std::size_t>(e - b));
    for (std::uint32_t i = b; i < e; ++i) {
        const auto idx = static_cast<std::size_t>(i);
        const int lbl = partner_label(partners[idx]);
        if (lbl != 0) {
            const double w = has_partner_weight ? partner_weight[idx] : 1.0;
            if (!labels.empty() && labels.back() == lbl) {
                weights.back() += w;
            } else {
                labels.push_back(lbl);
                weights.push_back(w);
            }
        }
    }
    const int n = static_cast<int>(labels.size());
    constexpr double kBaseWireCost = 2.772588722239781; // log(16)
    if (n == 1) {
        if (labels.front() != 1) return 0.0;
        return std::min(weights.front(), 2.0 * kBaseWireCost);
    }
    if (n <= 1) return 0.0;

    struct Node {
        int prev = -1;
        int next = -1;
        int label = 0;
        double w = 0.0;
        int ref_idx = -1; // representative index in the original merged label list
        std::vector<int> segment_ids;
        std::uint32_t version = 0;
        bool alive = true;
    };
    struct HeapEntry {
        double w = 0.0;
        int idx = -1;
        std::uint32_t version = 0;
    };
    struct HeapGreater {
        bool operator()(const HeapEntry& a, const HeapEntry& b) const noexcept { return a.w > b.w; }
    };

    std::vector<Node> nodes(static_cast<std::size_t>(n));
    int head = 0;
    std::priority_queue<HeapEntry, std::vector<HeapEntry>, HeapGreater> heap;
    for (int i = 0; i < n; ++i) {
        auto& node = nodes[static_cast<std::size_t>(i)];
        node.prev = i - 1;
        node.next = (i + 1 < n) ? (i + 1) : -1;
        node.label = labels[static_cast<std::size_t>(i)];
        node.w = weights[static_cast<std::size_t>(i)];
        node.ref_idx = i;
        node.segment_ids = {i};
        heap.push(HeapEntry{node.w, i, 0});
    }

    auto erase_node = [&](int idx) {
        auto& node = nodes[static_cast<std::size_t>(idx)];
        if (!node.alive) return;
        const int lp = node.prev;
        const int rn = node.next;
        if (idx == head) head = rn;
        if (lp >= 0) nodes[static_cast<std::size_t>(lp)].next = rn;
        if (rn >= 0) nodes[static_cast<std::size_t>(rn)].prev = lp;
        node.alive = false;
    };

    constexpr double kEndpointThreshold = kBaseWireCost;
    constexpr double kInteriorThreshold = 2.0 * kBaseWireCost;

    std::vector<int> gate_cut_segment_ids;
    gate_cut_segment_ids.reserve(static_cast<std::size_t>(n));
    double cut_cost = 0.0;
    while (!heap.empty()) {
        const HeapEntry top = heap.top();
        heap.pop();

        auto& node = nodes[static_cast<std::size_t>(top.idx)];
        if (!node.alive || node.version != top.version) continue;
        if (top.w > kInteriorThreshold) break;

        const int lp = node.prev;
        const int rn = node.next;
        if (lp < 0 || rn < 0) {
            if (top.w <= kEndpointThreshold) {
                cut_cost += top.w;
                gate_cut_segment_ids.insert(
                    gate_cut_segment_ids.end(),
                    node.segment_ids.begin(),
                    node.segment_ids.end());
            }
            erase_node(top.idx);
            continue;
        }

        if (nodes[static_cast<std::size_t>(lp)].label == nodes[static_cast<std::size_t>(rn)].label) {
            cut_cost += top.w;
            gate_cut_segment_ids.insert(
                gate_cut_segment_ids.end(),
                node.segment_ids.begin(),
                node.segment_ids.end());
            auto& left = nodes[static_cast<std::size_t>(lp)];
            auto& right = nodes[static_cast<std::size_t>(rn)];
            left.w += right.w;
            left.segment_ids.insert(left.segment_ids.end(), right.segment_ids.begin(), right.segment_ids.end());
            ++left.version;
            heap.push(HeapEntry{left.w, lp, left.version});
            erase_node(top.idx);
            erase_node(rn);
            continue;
        }

        node.w = kInteriorThreshold + 1.0;
        ++node.version;
        heap.push(HeapEntry{node.w, top.idx, node.version});
    }

    std::vector<std::uint8_t> is_gate_cut(static_cast<std::size_t>(n), 0u);
    for (const int seg_id : gate_cut_segment_ids) {
        if (seg_id >= 0 && seg_id < n) {
            is_gate_cut[static_cast<std::size_t>(seg_id)] = 1u;
        }
    }

    int wire_cuts = 0;
    int prev = -1;
    for (int i = 0; i < n; ++i) {
        if (is_gate_cut[static_cast<std::size_t>(i)] != 0u) continue;
        const int cur = labels[static_cast<std::size_t>(i)];
        if (prev >= 0 && cur != prev) ++wire_cuts;
        prev = cur;
    }

    return cut_cost + static_cast<double>(wire_cuts) * kBaseWireCost;
}

static OverlapSegmentUsage get_overlap_segment_usage(const std::vector<int>& placement,
                                                     const std::vector<std::vector<int>>& extra_blocks,
                                                     int overlap_pos,
                                                     int overlap_partition,
                                                     const CircuitGraph::TemporalGateEvents& events) {
    OverlapSegmentUsage usage;
    if (overlap_pos < 0 || static_cast<std::size_t>(overlap_pos) >= placement.size()) {
        throw std::out_of_range("get_overlap_segment_usage(): overlap_pos out of range");
    }
    if (placement.size() != static_cast<std::size_t>(events.num_qubits)) {
        throw std::invalid_argument("get_overlap_segment_usage(): placement length must match temporal_events.num_qubits");
    }
    if (!extra_blocks.empty() && extra_blocks.size() != placement.size()) {
        throw std::invalid_argument("get_overlap_segment_usage(): extra_blocks length must match placement length");
    }

    const int partition_b = placement[static_cast<std::size_t>(overlap_pos)];
    if (overlap_partition == partition_b) return usage;

    const auto b = events.row_ptr[static_cast<std::size_t>(overlap_pos)];
    const auto e = events.row_ptr[static_cast<std::size_t>(overlap_pos) + 1];
    if (e <= b) return usage;

    auto partner_has_extra_block = [&](std::uint32_t partner, int block) -> bool {
        const auto idx = static_cast<std::size_t>(partner);
        return !extra_blocks.empty() && contains_block(extra_blocks[idx], block);
    };

    auto partner_label = [&](std::uint32_t partner) -> int {
        const auto idx = static_cast<std::size_t>(partner);
        const int primary = placement[idx];
        if (primary == overlap_partition) return 1; // A
        if (primary == partition_b) return 2;       // B
        if (partner_has_extra_block(partner, overlap_partition)) return 1;
        if (partner_has_extra_block(partner, partition_b)) return 2;
        return 0;
    };

    const auto& partners = events.partners;
    const bool has_partner_weight = events.partner_weight.size() == partners.size();
    const double* partner_weight = has_partner_weight ? events.partner_weight.data() : nullptr;

    std::vector<int> labels;
    std::vector<double> weights;
    labels.reserve(static_cast<std::size_t>(e - b));
    weights.reserve(static_cast<std::size_t>(e - b));
    for (std::uint32_t i = b; i < e; ++i) {
        const auto idx = static_cast<std::size_t>(i);
        const int lbl = partner_label(partners[idx]);
        if (lbl != 0) {
            const double w = has_partner_weight ? partner_weight[idx] : 1.0;
            if (!labels.empty() && labels.back() == lbl) {
                weights.back() += w;
            } else {
                labels.push_back(lbl);
                weights.push_back(w);
            }
        }
    }

    const int n = static_cast<int>(labels.size());
    constexpr double kBaseWireCost = 2.772588722239781; // log(16)
    if (n == 1) {
        if (labels.front() == 1) {
            usage.extra_segments = (weights.front() <= 2.0 * kBaseWireCost) ? 0 : 1;
        } else if (labels.front() == 2) {
            usage.primary_segments = 1;
        }
        return usage;
    }
    if (n <= 1) return usage;

    struct Node {
        int prev = -1;
        int next = -1;
        int label = 0;
        double w = 0.0;
        std::vector<int> segment_ids;
        std::uint32_t version = 0;
        bool alive = true;
    };
    struct HeapEntry {
        double w = 0.0;
        int idx = -1;
        std::uint32_t version = 0;
    };
    struct HeapGreater {
        bool operator()(const HeapEntry& a, const HeapEntry& b) const noexcept { return a.w > b.w; }
    };

    std::vector<Node> nodes(static_cast<std::size_t>(n));
    int head = 0;
    std::priority_queue<HeapEntry, std::vector<HeapEntry>, HeapGreater> heap;
    for (int i = 0; i < n; ++i) {
        auto& node = nodes[static_cast<std::size_t>(i)];
        node.prev = i - 1;
        node.next = (i + 1 < n) ? (i + 1) : -1;
        node.label = labels[static_cast<std::size_t>(i)];
        node.w = weights[static_cast<std::size_t>(i)];
        node.segment_ids = {i};
        heap.push(HeapEntry{node.w, i, 0});
    }

    auto erase_node = [&](int idx) {
        auto& node = nodes[static_cast<std::size_t>(idx)];
        if (!node.alive) return;
        const int lp = node.prev;
        const int rn = node.next;
        if (idx == head) head = rn;
        if (lp >= 0) nodes[static_cast<std::size_t>(lp)].next = rn;
        if (rn >= 0) nodes[static_cast<std::size_t>(rn)].prev = lp;
        node.alive = false;
    };

    constexpr double kEndpointThreshold = kBaseWireCost;
    constexpr double kInteriorThreshold = 2.0 * kBaseWireCost;

    std::vector<int> gate_cut_segment_ids;
    gate_cut_segment_ids.reserve(static_cast<std::size_t>(n));
    while (!heap.empty()) {
        const HeapEntry top = heap.top();
        heap.pop();

        auto& node = nodes[static_cast<std::size_t>(top.idx)];
        if (!node.alive || node.version != top.version) continue;
        if (top.w > kInteriorThreshold) break;

        const int lp = node.prev;
        const int rn = node.next;
        if (lp < 0 || rn < 0) {
            if (top.w <= kEndpointThreshold) {
                gate_cut_segment_ids.insert(
                    gate_cut_segment_ids.end(),
                    node.segment_ids.begin(),
                    node.segment_ids.end());
            }
            erase_node(top.idx);
            continue;
        }

        if (nodes[static_cast<std::size_t>(lp)].label == nodes[static_cast<std::size_t>(rn)].label) {
            gate_cut_segment_ids.insert(
                gate_cut_segment_ids.end(),
                node.segment_ids.begin(),
                node.segment_ids.end());
            auto& left = nodes[static_cast<std::size_t>(lp)];
            auto& right = nodes[static_cast<std::size_t>(rn)];
            left.w += right.w;
            left.segment_ids.insert(left.segment_ids.end(), right.segment_ids.begin(), right.segment_ids.end());
            ++left.version;
            heap.push(HeapEntry{left.w, lp, left.version});
            erase_node(top.idx);
            erase_node(rn);
            continue;
        }

        node.w = kInteriorThreshold + 1.0;
        ++node.version;
        heap.push(HeapEntry{node.w, top.idx, node.version});
    }

    std::vector<std::uint8_t> is_gate_cut(static_cast<std::size_t>(n), 0u);
    for (const int seg_id : gate_cut_segment_ids) {
        if (seg_id >= 0 && seg_id < n) {
            is_gate_cut[static_cast<std::size_t>(seg_id)] = 1u;
        }
    }

    int active_label = 0;
    for (int i = 0; i < n; ++i) {
        if (is_gate_cut[static_cast<std::size_t>(i)] != 0u) continue;
        const int lbl = labels[static_cast<std::size_t>(i)];
        if (lbl != active_label) {
            if (lbl == 1) ++usage.extra_segments;
            if (lbl == 2) ++usage.primary_segments;
            active_label = lbl;
        }
    }
    return usage;
}

static void append_event_original_indices(const CircuitGraph::TemporalGateEvents& events,
                                          std::uint32_t event_id,
                                          std::vector<std::uint32_t>& out) {
    if (event_id >= events.num_events) return;
    if (events.event_original_row_ptr.size() != static_cast<std::size_t>(events.num_events) + 1u) return;
    const auto b = events.event_original_row_ptr[static_cast<std::size_t>(event_id)];
    const auto e = events.event_original_row_ptr[static_cast<std::size_t>(event_id) + 1u];
    if (e > events.event_original_indices.size() || b > e) return;
    out.insert(out.end(),
               events.event_original_indices.begin() + static_cast<std::ptrdiff_t>(b),
               events.event_original_indices.begin() + static_cast<std::ptrdiff_t>(e));
}

static bool event_original_first_last(const CircuitGraph::TemporalGateEvents& events,
                                      std::uint32_t event_id,
                                      std::uint32_t& first,
                                      std::uint32_t& last) {
    if (event_id >= events.num_events) return false;
    if (events.event_original_row_ptr.size() != static_cast<std::size_t>(events.num_events) + 1u) return false;
    const auto b = events.event_original_row_ptr[static_cast<std::size_t>(event_id)];
    const auto e = events.event_original_row_ptr[static_cast<std::size_t>(event_id) + 1u];
    if (e > events.event_original_indices.size() || b >= e) return false;
    first = events.event_original_indices[static_cast<std::size_t>(b)];
    last = events.event_original_indices[static_cast<std::size_t>(e - 1u)];
    return true;
}

static MultiOverlapEvaluation evaluate_multi_overlap_timeline(
    const std::vector<int>& placement,
    const std::vector<std::vector<int>>& extra_blocks,
    int overlap_pos,
    int k,
    const CircuitGraph::TemporalGateEvents& events) {
    MultiOverlapEvaluation evaluation;
    evaluation.segment_counts.assign(static_cast<std::size_t>(k), 0);

    if (overlap_pos < 0 || static_cast<std::size_t>(overlap_pos) >= placement.size()) {
        throw std::out_of_range("evaluate_multi_overlap_timeline(): overlap_pos out of range");
    }
    if (placement.size() != static_cast<std::size_t>(events.num_qubits)) {
        throw std::invalid_argument(
            "evaluate_multi_overlap_timeline(): placement length must match temporal_events.num_qubits");
    }
    if (!extra_blocks.empty() && extra_blocks.size() != placement.size()) {
        throw std::invalid_argument(
            "evaluate_multi_overlap_timeline(): extra_blocks length must match placement length");
    }

    const int primary = placement[static_cast<std::size_t>(overlap_pos)];
    if (primary < 0 || primary >= k) {
        throw std::invalid_argument("evaluate_multi_overlap_timeline(): primary label out of range");
    }

    std::vector<int> involved;
    involved.push_back(primary);
    for (const int block : extra_blocks[static_cast<std::size_t>(overlap_pos)]) {
        if (block >= 0 && block < k && !contains_block(involved, block)) {
            involved.push_back(block);
        }
    }

    if (involved.size() <= 1) {
        evaluation.segment_counts[static_cast<std::size_t>(primary)] = 1;
        return evaluation;
    }

    const auto b = events.row_ptr[static_cast<std::size_t>(overlap_pos)];
    const auto e = events.row_ptr[static_cast<std::size_t>(overlap_pos) + 1];
    if (e <= b) {
        evaluation.segment_counts[static_cast<std::size_t>(primary)] = 1;
        return evaluation;
    }

    auto partner_label = [&](std::uint32_t partner) -> int {
        const auto idx = static_cast<std::size_t>(partner);
        const int partner_primary = placement[idx];
        if (contains_block(involved, partner_primary)) return partner_primary;
        if (!extra_blocks.empty()) {
            for (const int block : extra_blocks[idx]) {
                if (contains_block(involved, block)) return block;
            }
        }
        return -1;
    };

    const auto& partners = events.partners;
    const bool has_partner_weight = events.partner_weight.size() == partners.size();
    const double* partner_weight = has_partner_weight ? events.partner_weight.data() : nullptr;
    const bool has_partner_event_id = events.partner_event_id.size() == partners.size();
    const std::uint32_t* partner_event_id = has_partner_event_id ? events.partner_event_id.data() : nullptr;
    constexpr std::uint32_t kInvalidRaw = std::numeric_limits<std::uint32_t>::max();

    struct TimelineSegment {
        int label = -1;
        double weight = 0.0;
        std::vector<std::uint32_t> events;
        std::uint32_t first_raw = kInvalidRaw;
    };

    std::vector<TimelineSegment> segments;
    segments.reserve(static_cast<std::size_t>(e - b));
    for (std::uint32_t i = b; i < e; ++i) {
        const auto idx = static_cast<std::size_t>(i);
        const int lbl = partner_label(partners[idx]);
        if (lbl < 0) continue;

        const double w = has_partner_weight ? partner_weight[idx] : 1.0;
        std::uint32_t ev = kInvalidRaw;
        std::uint32_t first_raw = kInvalidRaw;
        if (has_partner_event_id) {
            ev = partner_event_id[idx];
            std::uint32_t first = 0;
            std::uint32_t last = 0;
            if (event_original_first_last(events, ev, first, last)) {
                first_raw = first;
            }
        }

        if (!segments.empty() && segments.back().label == lbl) {
            segments.back().weight += w;
            if (ev != kInvalidRaw) segments.back().events.push_back(ev);
        } else {
            TimelineSegment segment;
            segment.label = lbl;
            segment.weight = w;
            segment.first_raw = first_raw;
            if (ev != kInvalidRaw) segment.events.push_back(ev);
            segments.push_back(std::move(segment));
        }
    }

    if (segments.empty()) {
        evaluation.segment_counts[static_cast<std::size_t>(primary)] = 1;
        return evaluation;
    }

    struct ActiveNode {
        int label = -1;
        double weight = 0.0;
        std::vector<int> segment_ids;
        std::uint32_t first_raw = kInvalidRaw;
    };

    auto build_active_nodes = [&](const std::vector<std::uint8_t>& removed) {
        std::vector<ActiveNode> nodes;
        for (int i = 0; i < static_cast<int>(segments.size()); ++i) {
            if (removed[static_cast<std::size_t>(i)] != 0u) continue;
            const auto& segment = segments[static_cast<std::size_t>(i)];
            if (!nodes.empty() && nodes.back().label == segment.label) {
                nodes.back().weight += segment.weight;
                nodes.back().segment_ids.push_back(i);
            } else {
                ActiveNode node;
                node.label = segment.label;
                node.weight = segment.weight;
                node.segment_ids = {i};
                node.first_raw = segment.first_raw;
                nodes.push_back(std::move(node));
            }
        }
        return nodes;
    };

    auto segment_counts_for_nodes = [&](const std::vector<ActiveNode>& nodes) {
        std::vector<int> counts(static_cast<std::size_t>(k), 0);
        for (const auto& node : nodes) {
            if (node.label >= 0 && node.label < k) {
                counts[static_cast<std::size_t>(node.label)]++;
            }
        }
        counts[static_cast<std::size_t>(primary)] =
            std::max(1, counts[static_cast<std::size_t>(primary)]);
        return counts;
    };

    constexpr double kBaseWireCost = 2.772588722239781; // log(16)
    std::vector<std::uint8_t> removed(segments.size(), 0u);
    double gate_cut_cost = 0.0;

    while (true) {
        const std::vector<ActiveNode> nodes = build_active_nodes(removed);
        if (nodes.empty()) break;
        const int current_wire_cuts = std::max(0, static_cast<int>(nodes.size()) - 1);

        int best_node = -1;
        double best_weight = std::numeric_limits<double>::infinity();
        int best_saved_wire_cuts = 0;
        for (int i = 0; i < static_cast<int>(nodes.size()); ++i) {
            std::vector<std::uint8_t> candidate_removed = removed;
            for (const int segment_id : nodes[static_cast<std::size_t>(i)].segment_ids) {
                candidate_removed[static_cast<std::size_t>(segment_id)] = 1u;
            }
            const std::vector<ActiveNode> candidate_nodes = build_active_nodes(candidate_removed);
            const int candidate_wire_cuts = std::max(0, static_cast<int>(candidate_nodes.size()) - 1);
            const int saved_wire_cuts = current_wire_cuts - candidate_wire_cuts;
            if (saved_wire_cuts <= 0) continue;
            const double saved_wire_cost = static_cast<double>(saved_wire_cuts) * kBaseWireCost;
            const double weight = nodes[static_cast<std::size_t>(i)].weight;
            if (weight <= saved_wire_cost &&
                (weight < best_weight ||
                 (weight == best_weight && saved_wire_cuts > best_saved_wire_cuts))) {
                best_node = i;
                best_weight = weight;
                best_saved_wire_cuts = saved_wire_cuts;
            }
        }

        if (best_node < 0) break;
        const auto& node = nodes[static_cast<std::size_t>(best_node)];
        gate_cut_cost += node.weight;
        for (const int segment_id : node.segment_ids) {
            removed[static_cast<std::size_t>(segment_id)] = 1u;
        }
    }

    const std::vector<ActiveNode> final_nodes = build_active_nodes(removed);
    evaluation.segment_counts = segment_counts_for_nodes(final_nodes);
    const int wire_cut_count = std::max(0, static_cast<int>(final_nodes.size()) - 1);
    evaluation.trace.cost = gate_cut_cost + static_cast<double>(wire_cut_count) * kBaseWireCost;

    for (int i = 0; i < static_cast<int>(segments.size()); ++i) {
        if (removed[static_cast<std::size_t>(i)] == 0u) continue;
        for (const std::uint32_t ev : segments[static_cast<std::size_t>(i)].events) {
            append_event_original_indices(events, ev, evaluation.trace.gate_cut_indices);
        }
    }

    for (int i = 1; i < static_cast<int>(final_nodes.size()); ++i) {
        const auto& left = final_nodes[static_cast<std::size_t>(i - 1)];
        const auto& right = final_nodes[static_cast<std::size_t>(i)];
        const std::uint32_t insert_before = final_nodes[static_cast<std::size_t>(i)].first_raw;
        if (insert_before != kInvalidRaw) {
            evaluation.trace.wire_cut_indices.push_back(
                std::make_pair(insert_before, static_cast<std::uint32_t>(overlap_pos)));
            evaluation.trace.wire_cut_partition_labels.push_back(std::make_tuple(
                insert_before,
                static_cast<std::uint32_t>(overlap_pos),
                left.label,
                right.label));
        }
    }

    std::sort(evaluation.trace.gate_cut_indices.begin(), evaluation.trace.gate_cut_indices.end());
    evaluation.trace.gate_cut_indices.erase(
        std::unique(evaluation.trace.gate_cut_indices.begin(), evaluation.trace.gate_cut_indices.end()),
        evaluation.trace.gate_cut_indices.end());
    std::sort(evaluation.trace.wire_cut_indices.begin(), evaluation.trace.wire_cut_indices.end());
    evaluation.trace.wire_cut_indices.erase(
        std::unique(evaluation.trace.wire_cut_indices.begin(), evaluation.trace.wire_cut_indices.end()),
        evaluation.trace.wire_cut_indices.end());
    std::sort(
        evaluation.trace.wire_cut_partition_labels.begin(),
        evaluation.trace.wire_cut_partition_labels.end());
    evaluation.trace.wire_cut_partition_labels.erase(
        std::unique(
            evaluation.trace.wire_cut_partition_labels.begin(),
            evaluation.trace.wire_cut_partition_labels.end()),
        evaluation.trace.wire_cut_partition_labels.end());
    return evaluation;
}

static OverlapCutTrace get_overlap_cut_trace(const std::vector<int>& placement,
                                             const std::vector<std::vector<int>>& extra_blocks,
                                             int overlap_pos,
                                             int overlap_partition,
                                             const CircuitGraph::TemporalGateEvents& events) {
    OverlapCutTrace trace;
    if (overlap_pos < 0 || static_cast<std::size_t>(overlap_pos) >= placement.size()) {
        throw std::out_of_range("get_overlap_cut_trace(): overlap_pos out of range");
    }
    if (placement.size() != static_cast<std::size_t>(events.num_qubits)) {
        throw std::invalid_argument("get_overlap_cut_trace(): placement length must match temporal_events.num_qubits");
    }
    if (!extra_blocks.empty() && extra_blocks.size() != placement.size()) {
        throw std::invalid_argument("get_overlap_cut_trace(): extra_blocks length must match placement length");
    }

    const int partition_b = placement[static_cast<std::size_t>(overlap_pos)];
    if (overlap_partition == partition_b) return trace;

    const auto b = events.row_ptr[static_cast<std::size_t>(overlap_pos)];
    const auto e = events.row_ptr[static_cast<std::size_t>(overlap_pos) + 1];
    if (e <= b) return trace;

    auto partner_has_extra_block = [&](std::uint32_t partner, int block) -> bool {
        const auto idx = static_cast<std::size_t>(partner);
        return !extra_blocks.empty() && contains_block(extra_blocks[idx], block);
    };

    auto partner_label = [&](std::uint32_t partner) -> int {
        const auto idx = static_cast<std::size_t>(partner);
        const int primary = placement[idx];
        if (primary == overlap_partition) return 1; // A
        if (primary == partition_b) return 2;       // B
        if (partner_has_extra_block(partner, overlap_partition)) return 1;
        if (partner_has_extra_block(partner, partition_b)) return 2;
        return 0;
    };

    const auto& partners = events.partners;
    const bool has_partner_weight = events.partner_weight.size() == partners.size();
    const double* partner_weight = has_partner_weight ? events.partner_weight.data() : nullptr;
    const bool has_partner_event_id = events.partner_event_id.size() == partners.size();
    const std::uint32_t* partner_event_id = has_partner_event_id ? events.partner_event_id.data() : nullptr;
    if (!has_partner_event_id ||
        events.event_original_row_ptr.size() != static_cast<std::size_t>(events.num_events) + 1u) {
        return trace;
    }

    std::vector<int> labels;
    std::vector<double> weights;
    std::vector<std::vector<std::uint32_t>> segment_events;
    std::vector<std::uint32_t> segment_first_raw;
    labels.reserve(static_cast<std::size_t>(e - b));
    weights.reserve(static_cast<std::size_t>(e - b));
    segment_events.reserve(static_cast<std::size_t>(e - b));
    segment_first_raw.reserve(static_cast<std::size_t>(e - b));
    constexpr std::uint32_t kInvalidRaw = std::numeric_limits<std::uint32_t>::max();
    for (std::uint32_t i = b; i < e; ++i) {
        const auto idx = static_cast<std::size_t>(i);
        const int lbl = partner_label(partners[idx]);
        if (lbl != 0) {
            const double w = has_partner_weight ? partner_weight[idx] : 1.0;
            const std::uint32_t ev = partner_event_id[idx];
            if (!labels.empty() && labels.back() == lbl) {
                weights.back() += w;
                segment_events.back().push_back(ev);
            } else {
                std::uint32_t first_raw = kInvalidRaw;
                std::uint32_t first = 0;
                std::uint32_t last = 0;
                if (event_original_first_last(events, ev, first, last)) {
                    first_raw = first;
                }
                labels.push_back(lbl);
                weights.push_back(w);
                segment_events.push_back({ev});
                segment_first_raw.push_back(first_raw);
            }
        }
    }
    const int n = static_cast<int>(labels.size());
    constexpr double kBaseWireCost = 2.772588722239781; // log(16)
    if (n == 1) {
        if (labels.front() == 1 && weights.front() <= 2.0 * kBaseWireCost) {
            trace.cost += weights.front();
            for (const std::uint32_t ev : segment_events.front()) {
                append_event_original_indices(events, ev, trace.gate_cut_indices);
            }
            std::sort(trace.gate_cut_indices.begin(), trace.gate_cut_indices.end());
            trace.gate_cut_indices.erase(
                std::unique(trace.gate_cut_indices.begin(), trace.gate_cut_indices.end()),
                trace.gate_cut_indices.end());
        } else if (labels.front() == 1 && segment_first_raw.front() != kInvalidRaw) {
            trace.cost += 2.0 * kBaseWireCost;
            trace.wire_cut_indices.push_back(
                std::make_pair(segment_first_raw.front(), static_cast<std::uint32_t>(overlap_pos)));
        }
        return trace;
    }
    if (n <= 1) return trace;

    struct Node {
        int prev = -1;
        int next = -1;
        int label = 0;
        double w = 0.0;
        int ref_idx = -1; // representative index in the original merged label list
        std::vector<int> segment_ids;
        std::uint32_t version = 0;
        bool alive = true;
        std::vector<std::uint32_t> events;
    };
    struct HeapEntry {
        double w = 0.0;
        int idx = -1;
        std::uint32_t version = 0;
    };
    struct HeapGreater {
        bool operator()(const HeapEntry& a, const HeapEntry& b) const noexcept { return a.w > b.w; }
    };

    std::vector<Node> nodes(static_cast<std::size_t>(n));
    int head = 0;
    std::priority_queue<HeapEntry, std::vector<HeapEntry>, HeapGreater> heap;
    for (int i = 0; i < n; ++i) {
        auto& node = nodes[static_cast<std::size_t>(i)];
        node.prev = i - 1;
        node.next = (i + 1 < n) ? (i + 1) : -1;
        node.label = labels[static_cast<std::size_t>(i)];
        node.w = weights[static_cast<std::size_t>(i)];
        node.ref_idx = i;
        node.segment_ids = {i};
        node.events = std::move(segment_events[static_cast<std::size_t>(i)]);
        heap.push(HeapEntry{node.w, i, 0});
    }

    auto erase_node = [&](int idx) {
        auto& node = nodes[static_cast<std::size_t>(idx)];
        if (!node.alive) return;
        const int lp = node.prev;
        const int rn = node.next;
        if (idx == head) head = rn;
        if (lp >= 0) nodes[static_cast<std::size_t>(lp)].next = rn;
        if (rn >= 0) nodes[static_cast<std::size_t>(rn)].prev = lp;
        node.alive = false;
    };

    auto append_node_indices = [&](const Node& node, std::vector<std::uint32_t>& out) {
        for (const std::uint32_t ev : node.events) {
            append_event_original_indices(events, ev, out);
        }
    };

    constexpr double kEndpointThreshold = kBaseWireCost;
    constexpr double kInteriorThreshold = 2.0 * kBaseWireCost;

    std::vector<int> gate_cut_segment_ids;
    gate_cut_segment_ids.reserve(static_cast<std::size_t>(n));
    while (!heap.empty()) {
        const HeapEntry top = heap.top();
        heap.pop();

        auto& node = nodes[static_cast<std::size_t>(top.idx)];
        if (!node.alive || node.version != top.version) continue;
        if (top.w > kInteriorThreshold) break;

        const int lp = node.prev;
        const int rn = node.next;
        if (lp < 0 || rn < 0) {
            if (top.w <= kEndpointThreshold) {
                trace.cost += top.w;
                append_node_indices(node, trace.gate_cut_indices);
                gate_cut_segment_ids.insert(
                    gate_cut_segment_ids.end(),
                    node.segment_ids.begin(),
                    node.segment_ids.end());
            }
            erase_node(top.idx);
            continue;
        }

        if (nodes[static_cast<std::size_t>(lp)].label == nodes[static_cast<std::size_t>(rn)].label) {
            trace.cost += top.w;
            append_node_indices(node, trace.gate_cut_indices);
            gate_cut_segment_ids.insert(
                gate_cut_segment_ids.end(),
                node.segment_ids.begin(),
                node.segment_ids.end());
            auto& left = nodes[static_cast<std::size_t>(lp)];
            auto& right = nodes[static_cast<std::size_t>(rn)];
            left.w += right.w;
            left.events.insert(left.events.end(), right.events.begin(), right.events.end());
            left.segment_ids.insert(left.segment_ids.end(), right.segment_ids.begin(), right.segment_ids.end());
            ++left.version;
            heap.push(HeapEntry{left.w, lp, left.version});
            erase_node(top.idx);
            erase_node(rn);
            continue;
        }

        node.w = kInteriorThreshold + 1.0;
        ++node.version;
        heap.push(HeapEntry{node.w, top.idx, node.version});
    }

    std::vector<std::uint8_t> is_gate_cut(static_cast<std::size_t>(n), 0u);
    for (const int seg_id : gate_cut_segment_ids) {
        if (seg_id >= 0 && seg_id < n) {
            is_gate_cut[static_cast<std::size_t>(seg_id)] = 1u;
        }
    }
    int prev_seg = -1;
    for (int seg = 0; seg < n; ++seg) {
        if (is_gate_cut[static_cast<std::size_t>(seg)] != 0u) continue;
        if (prev_seg >= 0 && labels[static_cast<std::size_t>(prev_seg)] != labels[static_cast<std::size_t>(seg)]) {
            const std::uint32_t insert_before = segment_first_raw[static_cast<std::size_t>(seg)];
            if (insert_before != kInvalidRaw) {
                trace.cost += kBaseWireCost;
                trace.wire_cut_indices.push_back(
                    std::make_pair(insert_before, static_cast<std::uint32_t>(overlap_pos)));
            }
        }
        prev_seg = seg;
    }

    std::sort(trace.gate_cut_indices.begin(), trace.gate_cut_indices.end());
    trace.gate_cut_indices.erase(
        std::unique(trace.gate_cut_indices.begin(), trace.gate_cut_indices.end()),
        trace.gate_cut_indices.end());
    std::sort(trace.wire_cut_indices.begin(), trace.wire_cut_indices.end());
    trace.wire_cut_indices.erase(
        std::unique(trace.wire_cut_indices.begin(), trace.wire_cut_indices.end()),
        trace.wire_cut_indices.end());
    return trace;
}

static OverlapCutTrace compute_overlap_cut_trace(const CircuitGraph& graph,
                                                 const Solution& sol,
                                                 double overlap_penalty,
                                                 const py::object& overlap_local_callback) {
    OverlapCutTrace out;
    const auto& temporal_events = graph.temporal_events();
    int k = 0;
    for (const int label : sol.primary) k = std::max(k, label + 1);
    for (const auto& blocks : sol.extra_blocks) {
        for (const int label : blocks) k = std::max(k, label + 1);
    }
    for (std::size_t v = 0; v < sol.primary.size(); ++v) {
        if (sol.extra_blocks[v].empty()) continue;
        out.cost += overlap_penalty * static_cast<double>(sol.extra_blocks[v].size());
        const auto local = evaluate_multi_overlap_timeline(
            sol.primary,
            sol.extra_blocks,
            static_cast<int>(v),
            k,
            temporal_events);
        out.cost += local.trace.cost;
        out.gate_cut_indices.insert(
            out.gate_cut_indices.end(),
            local.trace.gate_cut_indices.begin(),
            local.trace.gate_cut_indices.end());
        out.wire_cut_indices.insert(
            out.wire_cut_indices.end(),
            local.trace.wire_cut_indices.begin(),
            local.trace.wire_cut_indices.end());
        out.wire_cut_partition_labels.insert(
            out.wire_cut_partition_labels.end(),
            local.trace.wire_cut_partition_labels.begin(),
            local.trace.wire_cut_partition_labels.end());
        for (const int extra : sol.extra_blocks[v]) {
            if (!overlap_local_callback.is_none()) {
                out.cost += overlap_local_callback(static_cast<int>(v), sol.primary[v], extra).cast<double>();
            }
        }
    }
    std::sort(out.gate_cut_indices.begin(), out.gate_cut_indices.end());
    out.gate_cut_indices.erase(
        std::unique(out.gate_cut_indices.begin(), out.gate_cut_indices.end()),
        out.gate_cut_indices.end());
    std::sort(out.wire_cut_indices.begin(), out.wire_cut_indices.end());
    out.wire_cut_indices.erase(
        std::unique(out.wire_cut_indices.begin(), out.wire_cut_indices.end()),
        out.wire_cut_indices.end());
    std::sort(out.wire_cut_partition_labels.begin(), out.wire_cut_partition_labels.end());
    out.wire_cut_partition_labels.erase(
        std::unique(out.wire_cut_partition_labels.begin(), out.wire_cut_partition_labels.end()),
        out.wire_cut_partition_labels.end());
    return out;
}

static double overlap_cost_for_vertex(const std::vector<int>& placement,
                                      const std::vector<std::vector<int>>& extra_blocks,
                                      int v,
                                      int primary,
                                      double overlap_penalty,
                                      const py::object& overlap_local_callback,
                                      const CircuitGraph::TemporalGateEvents& temporal_events) {
    if (extra_blocks[static_cast<std::size_t>(v)].empty()) return 0.0;
    int k = 0;
    for (const int label : placement) k = std::max(k, label + 1);
    for (const auto& blocks : extra_blocks) {
        for (const int label : blocks) k = std::max(k, label + 1);
    }
    const auto evaluation = evaluate_multi_overlap_timeline(
        placement,
        extra_blocks,
        v,
        k,
        temporal_events);
    double c = evaluation.trace.cost;
    for (const int extra : extra_blocks[static_cast<std::size_t>(v)]) {
        c += overlap_penalty;
        if (!overlap_local_callback.is_none()) {
            c += overlap_local_callback(v, primary, extra).cast<double>();
        }
    }
    return c;
}

static std::vector<int> init_round_robin(int n, const std::vector<int>& capacities) {
    const int k = static_cast<int>(capacities.size());
    std::vector<int> primary(n, 0);
    std::vector<int> used(k, 0);
    int b = 0;
    for (int v = 0; v < n; ++v) {
        int tries = 0;
        while (tries < k && used[b] >= capacities[b]) {
            b = (b + 1) % k;
            ++tries;
        }
        if (tries == k) {
            throw std::runtime_error("refine_partition(): failed to initialize within capacities");
        }
        primary[v] = b;
        used[b]++;
        b = (b + 1) % k;
    }
    return primary;
}

static std::vector<int> compute_used_contrib_for_vertex(const CircuitGraph& graph,
                                                        const std::vector<int>& primary,
                                                        const std::vector<std::vector<int>>& extra_blocks,
                                                        int v,
                                                        int k) {
    std::vector<int> contrib(static_cast<std::size_t>(k), 0);
    const int block = primary[static_cast<std::size_t>(v)];
    if (extra_blocks[static_cast<std::size_t>(v)].empty()) {
        contrib[static_cast<std::size_t>(block)] = 1;
        return contrib;
    }
    const auto& temporal_events = graph.temporal_events();
    const auto evaluation = evaluate_multi_overlap_timeline(
        primary,
        extra_blocks,
        v,
        k,
        temporal_events);
    for (int label = 0; label < k; ++label) {
        contrib[static_cast<std::size_t>(label)] =
            evaluation.segment_counts[static_cast<std::size_t>(label)];
    }
    return contrib;
}

static void compute_used_counts(const CircuitGraph& graph,
                                const Solution& sol,
                                const std::vector<int>& capacities,
                                std::vector<int>& used,
                                std::vector<std::vector<int>>* used_contrib = nullptr) {
    const int k = static_cast<int>(capacities.size());
    used.assign(static_cast<std::size_t>(k), 0);
    if (used_contrib != nullptr) {
        used_contrib->assign(
            sol.primary.size(),
            std::vector<int>(static_cast<std::size_t>(k), 0));
    }
    for (std::size_t v = 0; v < sol.primary.size(); ++v) {
        std::vector<int> contrib = compute_used_contrib_for_vertex(
            graph,
            sol.primary,
            sol.extra_blocks,
            static_cast<int>(v),
            k);
        if (used_contrib != nullptr) {
            (*used_contrib)[v] = contrib;
        }
        for (int b = 0; b < k; ++b) {
            used[static_cast<std::size_t>(b)] += contrib[static_cast<std::size_t>(b)];
        }
    }
}

static bool solution_capacity_feasible(const CircuitGraph& graph,
                                       const Solution& sol,
                                       const std::vector<int>& capacities,
                                       std::vector<int>* used_out = nullptr) {
    std::vector<int> used;
    compute_used_counts(graph, sol, capacities, used);
    bool feasible = true;
    for (std::size_t b = 0; b < capacities.size(); ++b) {
        if (used[b] > capacities[b]) {
            feasible = false;
            break;
        }
    }
    if (used_out != nullptr) {
        *used_out = std::move(used);
    }
    return feasible;
}

static bool overlap_realizes_wire_cut_for_block(const CircuitGraph& graph,
                                                const std::vector<int>& primary,
                                                const std::vector<std::vector<int>>& extra_blocks,
                                                int v,
                                                int block,
                                                int k) {
    const auto evaluation = evaluate_multi_overlap_timeline(
        primary,
        extra_blocks,
        v,
        k,
        graph.temporal_events());
    if (block < 0 || block >= k) return false;
    if (evaluation.segment_counts[static_cast<std::size_t>(block)] <= 0) return false;
    for (const auto& item : evaluation.trace.wire_cut_partition_labels) {
        const int left = std::get<2>(item);
        const int right = std::get<3>(item);
        if (left == block || right == block) return true;
    }
    return false;
}

static double compute_cut_cost(const CircuitGraph::ConnectivityGraph& C, const Solution& sol) {
    double cost = 0.0;
    for (std::uint32_t u = 0; u < C.num_qubits; ++u) {
        const int up = sol.primary[u];
        const auto& ux = sol.extra_blocks[u];
        for (std::uint32_t ei = C.row_ptr[u]; ei < C.row_ptr[u + 1]; ++ei) {
            const std::uint32_t v = static_cast<std::uint32_t>(C.col_idx[ei]);
            if (u < v) {
                const int vp = sol.primary[v];
                const auto& vx = sol.extra_blocks[v];
                if (!shares_any_block(up, ux, vp, vx)) {
                    cost += static_cast<double>(C.weight[ei]);
                }
            }
        }
    }
    return cost;
}

static inline double weighted_fast_cost(double cut_cost,
                                        double overlap_cost,
                                        int cut_weight,
                                        int overlap_weight) {
    return static_cast<double>(cut_weight) * cut_cost + static_cast<double>(overlap_weight) * overlap_cost;
}

static std::vector<std::vector<int>>
build_overlap_dependents(const CircuitGraph::TemporalGateEvents& events) {
    std::vector<std::vector<int>> dependents(static_cast<std::size_t>(events.num_qubits));
    for (std::uint32_t q = 0; q < events.num_qubits; ++q) {
        const auto b = events.row_ptr[static_cast<std::size_t>(q)];
        const auto e = events.row_ptr[static_cast<std::size_t>(q) + 1u];
        auto& deps = dependents[static_cast<std::size_t>(q)];
        deps.reserve(static_cast<std::size_t>(e - b));
        for (std::uint32_t i = b; i < e; ++i) {
            deps.push_back(static_cast<int>(events.partners[static_cast<std::size_t>(i)]));
        }
        std::sort(deps.begin(), deps.end());
        deps.erase(std::unique(deps.begin(), deps.end()), deps.end());
    }
    return dependents;
}

static std::vector<double> compute_overlap_contribs(const CircuitGraph& graph,
                                                    const Solution& sol,
                                                    double overlap_penalty,
                                                    const py::object& overlap_local_callback) {
    const auto& temporal_events = graph.temporal_events();
    std::vector<double> contrib(sol.primary.size(), 0.0);
    for (std::size_t v = 0; v < sol.primary.size(); ++v) {
        contrib[v] = overlap_cost_for_vertex(
            sol.primary,
            sol.extra_blocks,
            static_cast<int>(v),
            sol.primary[v],
            overlap_penalty,
            overlap_local_callback,
            temporal_events);
    }
    return contrib;
}

static double sum_values(const std::vector<double>& values) {
    return std::accumulate(values.begin(), values.end(), 0.0);
}

static double delta_cut_move_primary(const CircuitGraph::ConnectivityGraph& C,
                                    const Solution& sol,
                                    int v,
                                    int new_primary) {
    const int old_primary = sol.primary[v];
    std::vector<int> new_extra = sol.extra_blocks[v];
    remove_extra_block(new_extra, new_primary);

    double delta = 0.0;
    for (std::uint32_t ei = C.row_ptr[v]; ei < C.row_ptr[v + 1]; ++ei) {
        const int u = static_cast<int>(C.col_idx[ei]);
        const int up = sol.primary[u];
        const auto& ux = sol.extra_blocks[u];
        const bool before = shares_any_block(old_primary, sol.extra_blocks[v], up, ux);
        const bool after = shares_any_block(new_primary, new_extra, up, ux);
        if (before && !after) delta += static_cast<double>(C.weight[ei]);
        if (!before && after) delta -= static_cast<double>(C.weight[ei]);
    }
    return delta;
}

static double delta_cut_add_overlap(const CircuitGraph::ConnectivityGraph& C,
                                   const Solution& sol,
                                   int v,
                                   int add_block) {
    const int vp = sol.primary[v];
    if (contains_block(sol.extra_blocks[v], add_block)) return 0.0;
    std::vector<int> after_extra = sol.extra_blocks[v];
    add_extra_block(after_extra, add_block);

    double delta = 0.0;
    for (std::uint32_t ei = C.row_ptr[v]; ei < C.row_ptr[v + 1]; ++ei) {
        const int u = static_cast<int>(C.col_idx[ei]);
        const int up = sol.primary[u];
        const auto& ux = sol.extra_blocks[u];
        const bool before = shares_any_block(vp, sol.extra_blocks[v], up, ux);
        const bool after = shares_any_block(vp, after_extra, up, ux);
        if (before && !after) delta += static_cast<double>(C.weight[ei]);
        if (!before && after) delta -= static_cast<double>(C.weight[ei]);
    }
    return delta;
}

static double delta_cut_remove_overlap(const CircuitGraph::ConnectivityGraph& C,
                                      const Solution& sol,
                                      int v,
                                      int remove_block) {
    const int vp = sol.primary[v];
    if (!contains_block(sol.extra_blocks[v], remove_block)) return 0.0;
    std::vector<int> after_extra = sol.extra_blocks[v];
    remove_extra_block(after_extra, remove_block);

    double delta = 0.0;
    for (std::uint32_t ei = C.row_ptr[v]; ei < C.row_ptr[v + 1]; ++ei) {
        const int u = static_cast<int>(C.col_idx[ei]);
        const int up = sol.primary[u];
        const auto& ux = sol.extra_blocks[u];
        const bool before = shares_any_block(vp, sol.extra_blocks[v], up, ux);
        const bool after = shares_any_block(vp, after_extra, up, ux);
        if (before && !after) delta += static_cast<double>(C.weight[ei]);
        if (!before && after) delta -= static_cast<double>(C.weight[ei]);
    }
    return delta;
}

static bool is_primary_boundary_vertex(const CircuitGraph::ConnectivityGraph& C,
                                       const Solution& sol,
                                       int v) {
    const int vp = sol.primary[static_cast<std::size_t>(v)];
    for (std::uint32_t ei = C.row_ptr[static_cast<std::uint32_t>(v)];
         ei < C.row_ptr[static_cast<std::uint32_t>(v) + 1];
         ++ei) {
        const int u = C.col_idx[ei];
        if (sol.primary[static_cast<std::size_t>(u)] != vp) return true;
    }
    return false;
}

static void repair_primary_capacity(const CircuitGraph::ConnectivityGraph& C,
                                   Solution& sol,
                                   const std::vector<int>& capacities) {
    const int n = static_cast<int>(sol.primary.size());
    const int k = static_cast<int>(capacities.size());

    std::vector<int> counts(k, 0);
    for (int v = 0; v < n; ++v) counts[sol.primary[v]]++;

    auto find_over = [&]() -> int {
        for (int b = 0; b < k; ++b) if (counts[b] > capacities[b]) return b;
        return -1;
    };
    auto find_under = [&]() -> int {
        for (int b = 0; b < k; ++b) if (counts[b] < capacities[b]) return b;
        return -1;
    };

    while (true) {
        int from = find_over();
        int to = find_under();
        if (from < 0 || to < 0) break;

        int best_v = -1;
        double best_delta = std::numeric_limits<double>::infinity();
        for (int v = 0; v < n; ++v) {
            if (sol.primary[v] != from) continue;
            const double d = delta_cut_move_primary(C, sol, v, to);
            if (d < best_delta) {
                best_delta = d;
                best_v = v;
            }
        }
        if (best_v < 0) break;
        sol.primary[best_v] = to;
        remove_extra_block(sol.extra_blocks[best_v], to);
        counts[from]--;
        counts[to]++;
    }
}

static py::dict refine_partition_impl(const CircuitGraph& graph,
                                     const std::vector<int>& capacities,
                                     const std::vector<int>& warm_start,
                                     const std::vector<std::vector<int>>& warm_start_extra_blocks,
                                     int max_passes,
                                     bool enable_overlap,
                                     double overlap_penalty,
                                     int cut_weight,
                                     int overlap_weight,
                                     double ged_weight,
                                     bool parallel_ged,
                                     py::object topology_graphs,
                                     py::object ged_callback,
                                     py::object overlap_local_callback,
                                     bool use_tabu,
                                     int tabu_tenure,
                                     int tabu_max_iters,
                                     bool parallel_tabu,
                                     double ged_balance,
                                     int ged_eval_interval,
                                     int ged_candidate_count,
                                     bool optimize_hardware_placement,
                                     bool store_history,
                                     bool qap_use_routing) {
    
    validate_capacities(capacities, graph.num_qubits());
    const auto& C = graph.connectivity();
    const int upper_bound_cuts = static_cast<int>(graph.operations().size())*cut_weight;
    const int n = static_cast<int>(C.num_qubits);
    const int k = static_cast<int>(capacities.size());

    if (max_passes < 0) throw std::invalid_argument("refine_partition(): max_passes must be >= 0");
    if (overlap_penalty < 0.0) throw std::invalid_argument("refine_partition(): overlap_penalty must be >= 0");
    if (cut_weight < 0) throw std::invalid_argument("refine_partition(): cut_weight must be >= 0");
    if (overlap_weight < 0) throw std::invalid_argument("refine_partition(): overlap_weight must be >= 0");
    if (ged_weight < 0.0) throw std::invalid_argument("refine_partition(): ged_weight must be >= 0");
    if (ged_balance <= 0.0) {
        throw std::invalid_argument("refine_partition(): ged_balance must be > 0");
    }
    if (ged_eval_interval < 0) {
        throw std::invalid_argument("refine_partition(): ged_eval_interval must be >= 0");
    }
    if (ged_candidate_count < 0) {
        throw std::invalid_argument("refine_partition(): ged_candidate_count must be >= 0");
    }
    if (tabu_tenure <= 0) throw std::invalid_argument("refine_partition(): tabu_tenure must be > 0");
    if (tabu_max_iters < 0) throw std::invalid_argument("refine_partition(): tabu_max_iters must be >= 0");
    if (!topology_graphs.is_none() && !ged_callback.is_none()) {
        throw std::invalid_argument("refine_partition(): provide either topology_graphs or ged_callback, not both");
    }

    Solution sol;
    if (!warm_start.empty()) {
        if ((int)warm_start.size() != n) {
            throw std::invalid_argument("refine_partition(): warm_start must have length num_qubits");
        }
        sol.primary = warm_start;
    } else {
        sol.primary = init_round_robin(n, capacities);
    }
    sol.extra_blocks.assign(n, {});

    for (int v = 0; v < n; ++v) {
        if (sol.primary[v] < 0 || sol.primary[v] >= k) {
            throw std::invalid_argument("refine_partition(): warm_start contains invalid block id");
        }
    }

    if (!warm_start_extra_blocks.empty()) {
        if ((int)warm_start_extra_blocks.size() != n) {
            throw std::invalid_argument("refine_partition(): warm_start_extra_blocks must have length num_qubits");
        }
        for (int v = 0; v < n; ++v) {
            for (const int block : warm_start_extra_blocks[static_cast<std::size_t>(v)]) {
                if (block < 0 || block >= k) {
                    throw std::invalid_argument(
                        "refine_partition(): warm_start_extra_blocks contains invalid block id");
                }
                if (block != sol.primary[static_cast<std::size_t>(v)]) {
                    add_extra_block(sol.extra_blocks[static_cast<std::size_t>(v)], block);
                }
            }
        }
    }

    // Ensure primary assignment respects capacities
    repair_primary_capacity(C, sol, capacities);

    std::vector<int> used;
    std::vector<std::vector<int>> used_contrib;
    compute_used_counts(graph, sol, capacities, used, &used_contrib);
    for (int b = 0; b < k; ++b) {
        if (used[static_cast<std::size_t>(b)] > capacities[static_cast<std::size_t>(b)]) {
            throw std::invalid_argument(
                "refine_partition(): warm_start_extra_blocks exceed partition capacities");
        }
    }

    double cut_cost = compute_cut_cost(C, sol);
    std::vector<double> overlap_contrib = compute_overlap_contribs(graph, sol, overlap_penalty, overlap_local_callback);
    double ov_cost = sum_values(overlap_contrib);
    double fast_cost = weighted_fast_cost(cut_cost, ov_cost, cut_weight, overlap_weight);
    const auto overlap_dependents = build_overlap_dependents(graph.temporal_events());

    auto subtract_used_contrib = [&](std::vector<int>& target, const std::vector<int>& contrib) {
        for (int b = 0; b < k; ++b) {
            target[static_cast<std::size_t>(b)] -= contrib[static_cast<std::size_t>(b)];
        }
    };

    auto add_used_contrib = [&](std::vector<int>& target, const std::vector<int>& contrib) {
        for (int b = 0; b < k; ++b) {
            target[static_cast<std::size_t>(b)] += contrib[static_cast<std::size_t>(b)];
        }
    };

    auto capacity_feasible_after = [&](const std::vector<int>& primary_after,
                                       const std::vector<std::vector<int>>& extra_after,
                                       const std::vector<int>& affected) -> bool {
        std::vector<int> candidate_used = used;
        for (const int q : affected) {
            subtract_used_contrib(candidate_used, used_contrib[static_cast<std::size_t>(q)]);
        }
        for (const int q : affected) {
            const std::vector<int> contrib = compute_used_contrib_for_vertex(
                graph,
                primary_after,
                extra_after,
                q,
                k);
            add_used_contrib(candidate_used, contrib);
        }
        for (int b = 0; b < k; ++b) {
            if (candidate_used[static_cast<std::size_t>(b)] > capacities[static_cast<std::size_t>(b)]) {
                return false;
            }
        }
        Solution candidate;
        candidate.primary = primary_after;
        candidate.extra_blocks = extra_after;
        return solution_capacity_feasible(graph, candidate, capacities);
    };
    std::vector<double> cut_cost_history;
    std::vector<double> overlap_cost_history;
    std::vector<double> fast_cost_history;
    std::vector<int> ged_iteration_history;
    std::vector<double> ged_cost_history;
    std::vector<double> ged_score_history;
    std::vector<double> normalized_fast_cost_history;
    std::vector<double> combined_cost_history;
    std::vector<double> best_combined_cost_history;
    if (store_history) {
        cut_cost_history.push_back(cut_cost);
        overlap_cost_history.push_back(ov_cost);
        fast_cost_history.push_back(fast_cost);
    }
    int accepted_move_count = 0;

    Solution best_sol = sol;
    std::vector<int> best_hardware_placement(static_cast<std::size_t>(k));
    std::iota(best_hardware_placement.begin(), best_hardware_placement.end(), 0);
    double best_fast = fast_cost;

    double best_ged = 0.0;
    double best_total = std::numeric_limits<double>::infinity();
    bool ged_evaluated = false;
    int ged_eval_count = 0;
    const bool use_cpp_ged = !topology_graphs.is_none();
    std::vector<const TopologyGraph*> topology_graph_ptrs;
    if (use_cpp_ged) {
        std::vector<py::object> topology_graph_items;
        try {
            topology_graph_items = topology_graphs.cast<std::vector<py::object>>();
        } catch (const py::cast_error&) {
            throw std::invalid_argument(
                "refine_partition(): topology_graphs must be a sequence of TopologyGraph objects");
        }
        if (static_cast<int>(topology_graph_items.size()) != k) {
            throw std::invalid_argument(
                "refine_partition(): topology_graphs length must equal number of partitions/capacities");
        }
        topology_graph_ptrs.reserve(topology_graph_items.size());
        try {
            for (const py::object& topo_obj : topology_graph_items) {
                const TopologyGraph& topo = py::cast<const TopologyGraph&>(topo_obj);
                topology_graph_ptrs.push_back(&topo);
            }
        } catch (const py::cast_error&) {
            throw std::invalid_argument(
                "refine_partition(): topology_graphs must contain TopologyGraph objects");
        }
    }

    std::vector<std::vector<int>> interchangeable_hardware_groups;
    if (use_cpp_ged && optimize_hardware_placement) {
        std::map<int, std::vector<int>> labels_by_capacity;
        for (int p = 0; p < k; ++p) {
            labels_by_capacity[capacities[static_cast<std::size_t>(p)]].push_back(p);
        }
        for (const auto& item : labels_by_capacity) {
            if (item.second.size() > 1) {
                interchangeable_hardware_groups.push_back(item.second);
            }
        }
    }

    auto update_best = [&]() {
        if (!solution_capacity_feasible(graph, sol, capacities)) {
            return;
        }
        if (fast_cost < best_fast) {
            best_fast = fast_cost;
            best_sol = sol;
        }
    };

    auto ged_available = [&]() {
        return ged_weight > 0.0 && (use_cpp_ged || !ged_callback.is_none());
    };

    auto eval_best_cpp_mapping_for_solution =
        [&](const Solution& candidate, const std::vector<int>& ged_extra_block)
            -> std::pair<double, std::vector<int>> {
        std::vector<int> placement(static_cast<std::size_t>(k));
        std::iota(placement.begin(), placement.end(), 0);

        double best_score = -std::numeric_limits<double>::infinity();
        std::vector<int> best_placement = placement;

        std::function<void(std::size_t)> visit_group = [&](std::size_t group_index) {
            if (group_index >= interchangeable_hardware_groups.size()) {
                const double score = QAP_fw::QAP_cost_fw(
                    candidate.primary,
                    ged_extra_block,
                    graph,
                    topology_graph_ptrs,
                    placement,
                    80,
                    1e-5,
                    parallel_ged,
                    qap_use_routing);
                if (score > best_score) {
                    best_score = score;
                    best_placement = placement;
                }
                return;
            }

            const auto& labels = interchangeable_hardware_groups[group_index];
            std::vector<int> hardware_labels = labels;
            std::sort(hardware_labels.begin(), hardware_labels.end());
            do {
                for (std::size_t i = 0; i < labels.size(); ++i) {
                    placement[static_cast<std::size_t>(labels[i])] = hardware_labels[i];
                }
                visit_group(group_index + 1);
            } while (std::next_permutation(hardware_labels.begin(), hardware_labels.end()));

            for (int label : labels) {
                placement[static_cast<std::size_t>(label)] = label;
            }
        };

        visit_group(0);
        return {best_score, best_placement};
    };

    auto eval_ged_score_for_solution = [&](const Solution& candidate)
        -> std::pair<double, std::vector<int>> {
        double g = 0.0;
        const std::vector<int> ged_extra_block = first_extra_blocks(candidate.extra_blocks);
        if (use_cpp_ged) {
            ++ged_eval_count;
            return eval_best_cpp_mapping_for_solution(candidate, ged_extra_block);
        } else {
            py::gil_scoped_acquire gil;
            ++ged_eval_count;
            g = ged_callback(candidate.primary, candidate.extra_blocks).cast<double>();
        }
        std::vector<int> identity(static_cast<std::size_t>(k));
        std::iota(identity.begin(), identity.end(), 0);
        return {g, identity};
    };

    auto record_ged_candidate = [&](const Solution& candidate,
                                    double candidate_fast_cost,
                                    int iteration_index,
                                    bool force_init = false) -> double {
        if (!ged_available()) return std::numeric_limits<double>::infinity();
        if (!solution_capacity_feasible(graph, candidate, capacities)) {
            return std::numeric_limits<double>::infinity();
        }

        const auto mapping_eval = eval_ged_score_for_solution(candidate);
        const double g = mapping_eval.first;
        // g is a mapping quality score in (0, 1]; higher is better.
        if (!(g > 0.0)) return std::numeric_limits<double>::infinity();
        const double ged_mismatch_cost = std::max(0.0, 1.0 - g);
        //const double ged_mismatch_cost = std::max(0.0, g);
        const double total = (candidate_fast_cost / upper_bound_cuts) + ged_mismatch_cost * ged_weight;
        if (store_history) {
            ged_iteration_history.push_back(iteration_index);
            ged_score_history.push_back(g);
            ged_cost_history.push_back(ged_mismatch_cost);
            normalized_fast_cost_history.push_back(candidate_fast_cost / upper_bound_cuts);
            combined_cost_history.push_back(total);
        }
        if (force_init || total < best_total) {
            best_total = total;
            best_ged = g;
            ged_evaluated = true;
            best_sol = candidate;
            best_hardware_placement = mapping_eval.second;
            best_fast = candidate_fast_cost;
        }
        if (store_history) {
            best_combined_cost_history.push_back(best_total);
        }
        return total;
    };

    auto maybe_eval_ged = [&](bool force_init = false) {
        if (ged_weight == 0.0){
            update_best();
            return;
        }
        if (!use_cpp_ged && ged_callback.is_none()){
            update_best();
            return;
        }
        if (!force_init && fast_cost >= (best_fast * ged_balance)) return;
        record_ged_candidate(
            sol,
            fast_cost,
            accepted_move_count,
            force_init);
    };

    // Always evaluate GED once to initialize best_total baseline.
    maybe_eval_ged(/*force_init=*/true);

    auto better_move = [](const MoveCandidate& lhs, const MoveCandidate& rhs) {
        if (!lhs.valid) return false;
        if (!rhs.valid) return true;
        if (lhs.d_fast < rhs.d_fast - 1e-12) return true;
        if (lhs.d_fast > rhs.d_fast + 1e-12) return false;
        if (lhs.type != rhs.type) return static_cast<int>(lhs.type) < static_cast<int>(rhs.type);
        if (lhs.v != rhs.v) return lhs.v < rhs.v;
        if (lhs.to != rhs.to) return lhs.to < rhs.to;
        return lhs.with < rhs.with;
    };

    auto solution_after_move = [&](const MoveCandidate& move) {
        Solution candidate = sol;
        const int v = move.v;
        const auto vx = candidate.extra_blocks[static_cast<std::size_t>(v)];
        if (move.type == MoveType::MOVE_PRIMARY) {
            std::vector<int> newx = vx;
            remove_extra_block(newx, move.to);
            candidate.primary[static_cast<std::size_t>(v)] = move.to;
            candidate.extra_blocks[static_cast<std::size_t>(v)] = std::move(newx);
        } else if (move.type == MoveType::ADD_OVERLAP) {
            add_extra_block(candidate.extra_blocks[static_cast<std::size_t>(v)], move.to);
        } else if (move.type == MoveType::REMOVE_OVERLAP) {
            remove_extra_block(candidate.extra_blocks[static_cast<std::size_t>(v)], move.to);
        } else if (move.type == MoveType::SWAP_PRIMARY) {
            const int u = move.with;
            const int old_vp = candidate.primary[static_cast<std::size_t>(v)];
            const int old_up = candidate.primary[static_cast<std::size_t>(u)];

            auto new_vx = candidate.extra_blocks[static_cast<std::size_t>(v)];
            remove_extra_block(new_vx, old_up);
            auto new_ux = candidate.extra_blocks[static_cast<std::size_t>(u)];
            remove_extra_block(new_ux, old_vp);

            candidate.primary[static_cast<std::size_t>(v)] = old_up;
            candidate.extra_blocks[static_cast<std::size_t>(v)] = std::move(new_vx);
            candidate.primary[static_cast<std::size_t>(u)] = old_vp;
            candidate.extra_blocks[static_cast<std::size_t>(u)] = std::move(new_ux);
        }
        return candidate;
    };

    auto scan_vertex_moves = [&](int v, auto&& consider) {
        const int vp = sol.primary[static_cast<std::size_t>(v)];
        const auto& vx = sol.extra_blocks[static_cast<std::size_t>(v)];

        auto affected_for_changed = [&](const std::vector<int>& changed) {
            std::vector<int> affected;
            for (const int q : changed) {
                affected.push_back(q);
                const auto& deps = overlap_dependents[static_cast<std::size_t>(q)];
                affected.insert(affected.end(), deps.begin(), deps.end());
            }
            std::sort(affected.begin(), affected.end());
            affected.erase(std::unique(affected.begin(), affected.end()), affected.end());
            return affected;
        };

        auto overlap_delta_for_state = [&](const std::vector<int>& placement_after,
                                           const std::vector<std::vector<int>>& extra_after,
                                           const std::vector<int>& affected) -> double {
            const auto& temporal_events = graph.temporal_events();
            double before = 0.0;
            double after = 0.0;
            for (const int q : affected) {
                const auto idx = static_cast<std::size_t>(q);
                before += overlap_contrib[idx];
                after += overlap_cost_for_vertex(
                    placement_after,
                    extra_after,
                    q,
                    placement_after[idx],
                    overlap_penalty,
                    overlap_local_callback,
                    temporal_events);
            }
            return after - before;
        };

        auto exact_overlap_delta_for_one_vertex = [&](int new_primary,
                                                      const std::vector<int>& new_extra) -> double {
            std::vector<int> placement_after = sol.primary;
            std::vector<std::vector<int>> extra_after = sol.extra_blocks;
            placement_after[static_cast<std::size_t>(v)] = new_primary;
            extra_after[static_cast<std::size_t>(v)] = new_extra;
            return overlap_delta_for_state(placement_after, extra_after, affected_for_changed({v}));
        };

        // Move primary
        for (int b = 0; b < k; ++b) {
            if (b == vp) continue;

            const double d_cut = delta_cut_move_primary(C, sol, v, b);
            std::vector<int> newx = vx;
            remove_extra_block(newx, b);
            std::vector<int> placement_after = sol.primary;
            std::vector<std::vector<int>> extra_after = sol.extra_blocks;
            placement_after[static_cast<std::size_t>(v)] = b;
            extra_after[static_cast<std::size_t>(v)] = newx;
            const std::vector<int> affected = affected_for_changed({v});
            if (!capacity_feasible_after(placement_after, extra_after, affected)) continue;
            const double d_ov = overlap_delta_for_state(placement_after, extra_after, affected);
            consider(MoveCandidate{
                MoveType::MOVE_PRIMARY, v, b, -1, d_cut, d_ov,
                weighted_fast_cost(d_cut, d_ov, cut_weight, overlap_weight), true});
        }

        // Add overlap
        if (enable_overlap) {
            for (int b = 0; b < k; ++b) {
                if (b == vp) continue;
                if (contains_block(vx, b)) continue;
                if (used[static_cast<std::size_t>(b)] >= capacities[static_cast<std::size_t>(b)]) continue;
                const double d_cut = delta_cut_add_overlap(C, sol, v, b);
                std::vector<int> newx = vx;
                add_extra_block(newx, b);
                std::vector<int> placement_after = sol.primary;
                std::vector<std::vector<int>> extra_after = sol.extra_blocks;
                extra_after[static_cast<std::size_t>(v)] = newx;
                if (!overlap_realizes_wire_cut_for_block(graph, placement_after, extra_after, v, b, k)) continue;
                const std::vector<int> affected = affected_for_changed({v});
                if (!capacity_feasible_after(placement_after, extra_after, affected)) continue;
                const double d_ov = overlap_delta_for_state(placement_after, extra_after, affected);
                consider(MoveCandidate{
                    MoveType::ADD_OVERLAP, v, b, -1, d_cut, d_ov,
                    weighted_fast_cost(d_cut, d_ov, cut_weight, overlap_weight), true});
            }
        }

        // Remove overlap
        for (const int block : vx) {
            const double d_cut = delta_cut_remove_overlap(C, sol, v, block);
            std::vector<int> newx = vx;
            remove_extra_block(newx, block);
            std::vector<int> placement_after = sol.primary;
            std::vector<std::vector<int>> extra_after = sol.extra_blocks;
            extra_after[static_cast<std::size_t>(v)] = newx;
            const std::vector<int> affected = affected_for_changed({v});
            if (!capacity_feasible_after(placement_after, extra_after, affected)) continue;
            const double d_ov = overlap_delta_for_state(placement_after, extra_after, affected);
            consider(MoveCandidate{
                MoveType::REMOVE_OVERLAP, v, block, -1, d_cut, d_ov,
                weighted_fast_cost(d_cut, d_ov, cut_weight, overlap_weight), true});
        }

        // Boundary swap of primary assignments with a neighboring cut-edge endpoint.
        if (is_primary_boundary_vertex(C, sol, v)) {
            for (std::uint32_t ei = C.row_ptr[static_cast<std::uint32_t>(v)];
                 ei < C.row_ptr[static_cast<std::uint32_t>(v) + 1];
                 ++ei) {
                const int u = C.col_idx[ei];
                if (u == v) continue;
                if (v > u) continue;
                if (sol.primary[static_cast<std::size_t>(u)] == vp) continue;

                const int up = sol.primary[static_cast<std::size_t>(u)];
                std::vector<int> vx_after = vx;
                remove_extra_block(vx_after, up);
                double d_cut = delta_cut_move_primary(C, sol, v, up);

                Solution tmp = sol;
                tmp.primary[static_cast<std::size_t>(v)] = up;
                tmp.extra_blocks[static_cast<std::size_t>(v)] = vx_after;

                std::vector<int> ux_after = tmp.extra_blocks[static_cast<std::size_t>(u)];
                remove_extra_block(ux_after, vp);
                d_cut += delta_cut_move_primary(C, tmp, u, vp);

                std::vector<int> placement_after = sol.primary;
                std::vector<std::vector<int>> extra_after = sol.extra_blocks;
                placement_after[static_cast<std::size_t>(v)] = up;
                extra_after[static_cast<std::size_t>(v)] = std::move(vx_after);
                placement_after[static_cast<std::size_t>(u)] = vp;
                extra_after[static_cast<std::size_t>(u)] = std::move(ux_after);
                const std::vector<int> affected = affected_for_changed({v, u});
                if (!capacity_feasible_after(placement_after, extra_after, affected)) continue;
                const double d_ov = overlap_delta_for_state(placement_after, extra_after, affected);

                consider(MoveCandidate{
                    MoveType::SWAP_PRIMARY, v, -1, u, d_cut, d_ov,
                    weighted_fast_cost(d_cut, d_ov, cut_weight, overlap_weight), true});
            }
        }
    };

    auto apply_move = [&](const MoveCandidate& move) {
        const int v = move.v;
        const auto vx = sol.extra_blocks[static_cast<std::size_t>(v)];
        std::vector<int> changed_vertices;
        changed_vertices.push_back(v);
        if (move.type == MoveType::SWAP_PRIMARY) {
            changed_vertices.push_back(move.with);
        }
        std::vector<int> affected;
        for (const int q : changed_vertices) {
            affected.push_back(q);
            const auto& deps = overlap_dependents[static_cast<std::size_t>(q)];
            affected.insert(affected.end(), deps.begin(), deps.end());
        }
        std::sort(affected.begin(), affected.end());
        affected.erase(std::unique(affected.begin(), affected.end()), affected.end());
        double old_affected_overlap = 0.0;
        for (const int q : affected) {
            old_affected_overlap += overlap_contrib[static_cast<std::size_t>(q)];
        }

        cut_cost += move.d_cut;

        if (move.type == MoveType::MOVE_PRIMARY) {
            const int newp = move.to;
            std::vector<int> newx = vx;
            remove_extra_block(newx, newp);

            sol.primary[static_cast<std::size_t>(v)] = newp;
            sol.extra_blocks[static_cast<std::size_t>(v)] = std::move(newx);
        } else if (move.type == MoveType::ADD_OVERLAP) {
            add_extra_block(sol.extra_blocks[static_cast<std::size_t>(v)], move.to);
        } else if (move.type == MoveType::REMOVE_OVERLAP) {
            remove_extra_block(sol.extra_blocks[static_cast<std::size_t>(v)], move.to);
        } else if (move.type == MoveType::SWAP_PRIMARY) {
            const int u = move.with;
            const int old_vp = sol.primary[static_cast<std::size_t>(v)];
            const int old_up = sol.primary[static_cast<std::size_t>(u)];

            const auto old_vx = sol.extra_blocks[static_cast<std::size_t>(v)];
            auto new_vx = old_vx;
            remove_extra_block(new_vx, old_up);

            const auto old_ux = sol.extra_blocks[static_cast<std::size_t>(u)];
            auto new_ux = old_ux;
            remove_extra_block(new_ux, old_vp);

            sol.primary[static_cast<std::size_t>(v)] = old_up;
            sol.extra_blocks[static_cast<std::size_t>(v)] = std::move(new_vx);
            sol.primary[static_cast<std::size_t>(u)] = old_vp;
            sol.extra_blocks[static_cast<std::size_t>(u)] = std::move(new_ux);
        }
        for (const int q : affected) {
            subtract_used_contrib(used, used_contrib[static_cast<std::size_t>(q)]);
        }
        for (const int q : affected) {
            const auto idx = static_cast<std::size_t>(q);
            used_contrib[idx] = compute_used_contrib_for_vertex(
                graph,
                sol.primary,
                sol.extra_blocks,
                q,
                k);
            add_used_contrib(used, used_contrib[idx]);
        }

        const auto& temporal_events = graph.temporal_events();
        double new_affected_overlap = 0.0;
        for (const int q : affected) {
            const auto idx = static_cast<std::size_t>(q);
            overlap_contrib[idx] = overlap_cost_for_vertex(
                sol.primary,
                sol.extra_blocks,
                q,
                sol.primary[idx],
                overlap_penalty,
                overlap_local_callback,
                temporal_events);
            new_affected_overlap += overlap_contrib[idx];
        }
        ov_cost += new_affected_overlap - old_affected_overlap;
        fast_cost = weighted_fast_cost(cut_cost, ov_cost, cut_weight, overlap_weight);
        ++accepted_move_count;
        if (store_history) {
            cut_cost_history.push_back(cut_cost);
            overlap_cost_history.push_back(ov_cost);
            fast_cost_history.push_back(fast_cost);
        }
    };

    if (!use_tabu) {
        for (int pass = 0; pass < max_passes; ++pass) {
            bool improved_any = false;

            for (int v = 0; v < n; ++v) {
                MoveCandidate best_local;
                scan_vertex_moves(v, [&](const MoveCandidate& cand) {
                    if (better_move(cand, best_local)) best_local = cand;
                });

                if (!best_local.valid || best_local.d_fast >= -1e-12) continue;

                improved_any = true;
                apply_move(best_local);

                maybe_eval_ged();
            }

            if (!improved_any) break;
        }
    } else {
        const int tabu_iter_limit = (tabu_max_iters > 0) ? tabu_max_iters : std::max(1, max_passes * std::max(1, n));
        std::vector<int> tabu_primary(static_cast<std::size_t>(n) * static_cast<std::size_t>(k), 0);
        std::vector<int> tabu_extra(static_cast<std::size_t>(n) * static_cast<std::size_t>(k), 0);
        auto key = [k](int v, int block) -> std::size_t {
            return static_cast<std::size_t>(v) * static_cast<std::size_t>(k) + static_cast<std::size_t>(block);
        };

        const bool can_parallel_tabu_eval =
#if OPTCORE_HAS_OPENMP
            parallel_tabu && overlap_local_callback.is_none();
#else
            false;
#endif

        for (int iter = 1; iter <= tabu_iter_limit; ++iter) {
            MoveCandidate best_move;
            std::vector<MoveCandidate> ged_candidates;
            const bool ged_aware_iter =
                ged_available() &&
                ged_eval_interval > 0 &&
                ged_candidate_count > 0 &&
                (iter % ged_eval_interval == 0);

            auto insert_ged_candidate = [&](const MoveCandidate& cand) {
                if (!cand.valid) return;
                auto pos = ged_candidates.begin();
                while (pos != ged_candidates.end() && !better_move(cand, *pos)) {
                    ++pos;
                }
                ged_candidates.insert(pos, cand);
                if (static_cast<int>(ged_candidates.size()) > ged_candidate_count) {
                    ged_candidates.pop_back();
                }
            };

            auto consider_tabu = [&](MoveCandidate best_so_far, const MoveCandidate& cand) -> MoveCandidate {
                if (!cand.valid) return best_so_far;

                bool is_tabu = false;
                if (cand.type == MoveType::MOVE_PRIMARY) {
                    is_tabu = tabu_primary[key(cand.v, cand.to)] > iter;
                } else if (cand.type == MoveType::ADD_OVERLAP || cand.type == MoveType::REMOVE_OVERLAP) {
                    is_tabu = tabu_extra[key(cand.v, cand.to)] > iter;
                } else if (cand.type == MoveType::SWAP_PRIMARY) {
                    const int up = sol.primary[static_cast<std::size_t>(cand.with)];
                    const int vp = sol.primary[static_cast<std::size_t>(cand.v)];
                    is_tabu = (tabu_primary[key(cand.v, up)] > iter) ||
                              (tabu_primary[key(cand.with, vp)] > iter);
                }

                const bool aspiration = (fast_cost + cand.d_fast) < (best_fast - 1e-12);
                if (is_tabu && !aspiration) return best_so_far;

                if (ged_aware_iter) {
                    insert_ged_candidate(cand);
                }
                return better_move(cand, best_so_far) ? cand : best_so_far;
            };

            if (can_parallel_tabu_eval && !ged_aware_iter) {
#if OPTCORE_HAS_OPENMP
#pragma omp parallel
                {
                    MoveCandidate local_best;
#pragma omp for schedule(static)
                    for (int v = 0; v < n; ++v) {
                        scan_vertex_moves(v, [&](const MoveCandidate& cand) {
                            local_best = consider_tabu(local_best, cand);
                        });
                    }
#pragma omp critical
                    {
                        if (better_move(local_best, best_move)) best_move = local_best;
                    }
                }
#endif
            } else {
                for (int v = 0; v < n; ++v) {
                    scan_vertex_moves(v, [&](const MoveCandidate& cand) {
                        best_move = consider_tabu(best_move, cand);
                    });
                }
            }

            if (!best_move.valid) break;

            bool selected_by_ged = false;
            if (ged_aware_iter && !ged_candidates.empty()) {
                MoveCandidate best_ged_move;
                double best_ged_total = std::numeric_limits<double>::infinity();
                const int next_iteration_index = accepted_move_count + 1;
                for (const MoveCandidate& cand : ged_candidates) {
                    const Solution candidate_solution = solution_after_move(cand);
                    const double candidate_fast_cost = fast_cost + cand.d_fast;
                    const double total = record_ged_candidate(
                        candidate_solution,
                        candidate_fast_cost,
                        next_iteration_index);
                    if (total < best_ged_total) {
                        best_ged_total = total;
                        best_ged_move = cand;
                    }
                }
                if (best_ged_move.valid) {
                    best_move = best_ged_move;
                    selected_by_ged = true;
                }
            }

            const int v = best_move.v;
            const int old_vp = sol.primary[static_cast<std::size_t>(v)];
            int old_u = -1;
            int old_up = -1;
            if (best_move.type == MoveType::SWAP_PRIMARY) {
                old_u = best_move.with;
                old_up = sol.primary[static_cast<std::size_t>(old_u)];
            }

            apply_move(best_move);

            const int tabu_until = iter + tabu_tenure;
            if (best_move.type == MoveType::MOVE_PRIMARY) {
                tabu_primary[key(v, old_vp)] = tabu_until;
            } else if (best_move.type == MoveType::ADD_OVERLAP) {
                tabu_extra[key(v, best_move.to)] = tabu_until;
            } else if (best_move.type == MoveType::REMOVE_OVERLAP) {
                tabu_extra[key(v, best_move.to)] = tabu_until;
            } else if (best_move.type == MoveType::SWAP_PRIMARY) {
                tabu_primary[key(v, old_vp)] = tabu_until;
                tabu_primary[key(old_u, old_up)] = tabu_until;
            }

            if (!selected_by_ged) {
                maybe_eval_ged();
            }
        }
    }

    // If GED evaluated, best_sol is incumbent; else best_sol is best_fast.
    // Keep partition labels as produced by the optimizer. Hardware placement
    // optimization is returned separately via optimized_hardware_placement.
    // Relabeling primary/extra block labels here can change temporal overlap
    // interpretation because extra block order is meaningful for wire cuts.
    const std::vector<int> selected_hardware_placement = best_hardware_placement;
    const bool apply_output_hardware_labels =
        false;
    Solution out_sol = best_sol;
    std::vector<int> output_hardware_placement(static_cast<std::size_t>(k));
    std::iota(output_hardware_placement.begin(), output_hardware_placement.end(), 0);

    const double out_cut = compute_cut_cost(C, out_sol);
    const OverlapCutTrace overlap_trace =
        compute_overlap_cut_trace(graph, out_sol, overlap_penalty, overlap_local_callback);
    const double out_ov = overlap_trace.cost;
    const double out_fast = weighted_fast_cost(out_cut, out_ov, cut_weight, overlap_weight);
    const double out_ged_score = ged_evaluated ? best_ged : 0.0;
    const double out_ged_cost = ged_evaluated ? std::max(0.0, 1.0 - best_ged) : 0.0;
    const double out_total = ged_evaluated ? best_total : out_fast;
    const double norm_cut = out_fast/upper_bound_cuts;
    std::vector<int> best_used;
    compute_used_counts(graph, best_sol, capacities, best_used);
    std::vector<int> out_used;
    compute_used_counts(graph, out_sol, capacities, out_used);
    for (int b = 0; b < k; ++b) {
        if (out_used[static_cast<std::size_t>(b)] > capacities[static_cast<std::size_t>(b)]) {
            throw std::runtime_error(
                "refine_partition(): internal error: selected final solution exceeds partition capacity; "
                "block=" + std::to_string(b) +
                " capacity=" + std::to_string(capacities[static_cast<std::size_t>(b)]) +
                " out_used=" + format_vector(out_used) +
                " best_used_before_relabel=" + format_vector(best_used) +
                " capacities=" + format_vector(capacities) +
                " selected_hardware_placement=" + format_vector(selected_hardware_placement) +
                " apply_output_hardware_labels=" + std::to_string(apply_output_hardware_labels ? 1 : 0) +
                " ged_evaluated=" + std::to_string(ged_evaluated ? 1 : 0) +
                " best_primary=" + format_vector(best_sol.primary) +
                " best_extra_blocks=" + format_blocks(best_sol.extra_blocks) +
                " out_primary=" + format_vector(out_sol.primary) +
                " out_extra_blocks=" + format_blocks(out_sol.extra_blocks));
        }
    }

    std::vector<std::uint32_t> graph_cut_gate_indices;
    {
        graph_cut_gate_indices.reserve(graph.operations().size());
        for (const auto& op : graph.operations()) {
            const int up = out_sol.primary[static_cast<std::size_t>(op.q0)];
            const int vp = out_sol.primary[static_cast<std::size_t>(op.q1)];
            const auto& ux = out_sol.extra_blocks[static_cast<std::size_t>(op.q0)];
            const auto& vx = out_sol.extra_blocks[static_cast<std::size_t>(op.q1)];
            if (!shares_any_block(up, ux, vp, vx)) {
                graph_cut_gate_indices.push_back(op.original_index);
            }
        }
        std::sort(graph_cut_gate_indices.begin(), graph_cut_gate_indices.end());
        graph_cut_gate_indices.erase(
            std::unique(graph_cut_gate_indices.begin(), graph_cut_gate_indices.end()),
            graph_cut_gate_indices.end());
    }
    std::vector<std::uint32_t> overlap_cut_gate_indices = overlap_trace.gate_cut_indices;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> overlap_wire_cut_indices =
        overlap_trace.wire_cut_indices;
    std::vector<std::uint32_t> cut_gate_indices = graph_cut_gate_indices;
    cut_gate_indices.insert(cut_gate_indices.end(), overlap_cut_gate_indices.begin(), overlap_cut_gate_indices.end());
    std::sort(cut_gate_indices.begin(), cut_gate_indices.end());
    cut_gate_indices.erase(std::unique(cut_gate_indices.begin(), cut_gate_indices.end()), cut_gate_indices.end());

    py::list cut_gates;
    {
        std::unordered_set<std::uint32_t> keep(cut_gate_indices.begin(), cut_gate_indices.end());
        for (const auto& op : graph.operations()) {
            if (keep.find(op.original_index) == keep.end()) continue;
            cut_gates.append(py::make_tuple(
                op.original_index,
                op.q0,
                op.q1,
                graph.gate_name(op.gate)));
        }
    }

    py::dict out;
    out["primary"] = out_sol.primary;
    out["extra_block"] = out_sol.extra_blocks;
    out["extra_blocks"] = out_sol.extra_blocks;
    out["legacy_extra_block"] = first_extra_blocks(out_sol.extra_blocks);
    out["cut_cost"] = out_cut;
    out["overlap_cost"] = out_ov;
    out["fast_cost"] = out_fast;
    out["cut_weight"] = cut_weight;
    out["overlap_weight"] = overlap_weight;
    out["hardware_placement"] = output_hardware_placement;
    out["optimized_hardware_placement"] = selected_hardware_placement;
    out["partition_used_qubits"] = out_used;
    out["normalized cut cost"] = norm_cut;
    out["graph_cut_gate_indices"] = graph_cut_gate_indices;
    out["overlap_cut_gate_indices"] = overlap_cut_gate_indices;
    out["overlap_wire_cut_indices"] = overlap_wire_cut_indices;
    out["wire_cut_indices"] = overlap_wire_cut_indices;
    out["wire_cut_partition_labels"] = overlap_trace.wire_cut_partition_labels;
    out["cut_gate_indices"] = cut_gate_indices;
    out["cut_gates"] = cut_gates;
    out["ged_cost"] = out_ged_cost;
    out["ged_score"] = out_ged_score;
    out["total_cost"] = out_total;
    out["ged_evaluated"] = ged_evaluated;
    out["ged_eval_count"] = ged_eval_count;
    out["cut_cost_history"] = cut_cost_history;
    out["overlap_cost_history"] = overlap_cost_history;
    out["fast_cost_history"] = fast_cost_history;
    out["cost_history"] = fast_cost_history;
    out["ged_iteration_history"] = ged_iteration_history;
    out["ged_cost_history"] = ged_cost_history;
    out["ged_score_history"] = ged_score_history;
    out["normalized_fast_cost_history"] = normalized_fast_cost_history;
    out["combined_cost_history"] = combined_cost_history;
    out["best_combined_cost_history"] = best_combined_cost_history;
    return out;
}

} // namespace

void bind_partition_refiner(py::module_& m) {
    m.def(
        "refine_partition",
        &refine_partition_impl,
        py::arg("graph"),
        py::arg("capacities"),
        py::arg("warm_start") = std::vector<int>{},
        py::arg("warm_start_extra_blocks") = std::vector<std::vector<int>>{},
        py::arg("max_passes") = 5,
        py::arg("enable_overlap") = true,
        py::arg("overlap_penalty") = 0.0,
        py::arg("cut_weight") = 1,
        py::arg("overlap_weight") = 1,
        py::arg("ged_weight") = 1.0,
        py::arg("parallel_ged") = false,
        py::arg("topology_graphs") = py::none(),
        py::arg("ged_callback") = py::none(),
        py::arg("overlap_local_callback") = py::none(),
        py::arg("use_tabu") = false,
        py::arg("tabu_tenure") = 7,
        py::arg("tabu_max_iters") = 0,
        py::arg("parallel_tabu") = false,
        py::arg("ged_balance") = 1.0,
        py::arg("ged_eval_interval") = 0,
        py::arg("ged_candidate_count") = 0,
        py::arg("optimize_hardware_placement") = true,
        py::arg("store_history") = true,
        py::arg("qap_use_routing") = true,
        R"pbdoc(
Refine a warm-start partition for a CircuitGraph.

Fast objective = uncovered-edge cost (weighted) + overlap penalties.
Optionally performs lazy GED evaluation either:
  1) in C++ using `topology_graphs`, or
  2) via Python `ged_callback`.

Args:
  graph: CircuitGraph
  capacities: list[int], per-block capacities, sum >= num_qubits
  warm_start: list[int], initial primary assignment (block per qubit). If empty, round-robin init.
  warm_start_extra_blocks: list[list[int]], optional initial overlap memberships per qubit.
  max_passes: int, number of greedy refinement passes
  enable_overlap: bool, allow extra block memberships per node
  overlap_penalty: float, base penalty per overlapped node
  cut_weight: int, multiplier for cut_cost in the fast objective
  overlap_weight: int, multiplier for overlap_cost in the fast objective
  ged_weight: float, multiplier applied to mapping mismatch in the combined objective. When GED is evaluated:
              total_cost = (fast_cost / upper_bound_cuts) + (1 - ged_score) * ged_weight.
  parallel_ged: bool, when true and OpenMP is available, parallelize per-partition GED solves.
  topology_graphs: list[TopologyGraph], one per partition. If provided, C++ computes weighted GED:
                   using a Frank-Wolfe QAP solver over node/edge overlap products.
  ged_callback: callable(primary:list[int], extra_block:list[list[int]]) -> float, expensive GED-like score.
                Called when the GED gate accepts an incumbent state, and on GED-aware tabu
                iterations for the top candidate moves.
  overlap_local_callback: callable(v:int, primary:int, extra:int) -> float, local extra overlap penalty.
  use_tabu: bool, when true use tabu search instead of strict greedy descent.
  tabu_tenure: int, tabu tenure (in iterations) for reverse moves in tabu mode.
  tabu_max_iters: int, max tabu iterations (0 means derive from max_passes*num_qubits).
  parallel_tabu: bool, parallelize tabu move scoring over vertices (OpenMP build required).
                 Falls back to sequential scoring when overlap_local_callback is provided.
  ged_balance: float > 0, GED-gating multiplier. After the forced initial GED evaluation,
               GED is evaluated only when fast_cost < best_fast * ged_balance, where
               best_fast is the fast cost of the current best-total incumbent.
  ged_eval_interval: int, in tabu mode, evaluate GED during move selection every N iterations
                     when > 0. This makes ged_weight steer the selected move instead of only
                     reranking already accepted states.
  ged_candidate_count: int, number of top fast-objective candidate moves to GED-score on
                       GED-aware tabu iterations. Larger values increase steering strength
                       and GED runtime.
  optimize_hardware_placement: bool, when true, mapping-score evaluation tries all
                       same-capacity hardware placements and uses the best score. When false,
                       partition label p is scored against topology_graphs[p].
  store_history: bool, when false, skip retaining per-move and per-GED history arrays.
                 Final costs, placement, cut indices, and ged_eval_count are still returned.
  qap_use_routing: bool, when true, QAP scoring uses an error-weighted all-pairs
                 routing proxy for non-adjacent hardware qubit pairs. When false,
                 QAP scores direct hardware couplings and applies a fixed finite
                 penalty to missing couplings.

		Returns:
	  dict with keys: primary, extra_block, extra_blocks, legacy_extra_block, cut_cost, overlap_cost, fast_cost, cut_weight, overlap_weight,
	                  hardware_placement,
	                  graph_cut_gate_indices, overlap_cut_gate_indices, overlap_wire_cut_indices,
	                  wire_cut_indices, cut_gate_indices, cut_gates,
	                  ged_cost, ged_score, total_cost, ged_evaluated, ged_eval_count
	  overlap_wire_cut_indices/wire_cut_indices entries are (insert_before_qc_data_index, qubit).
)pbdoc"
    );
}
