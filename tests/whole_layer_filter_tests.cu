#include "types.hpp"
#ifndef BEAM_FILTER_ORACLE_ONLY
#include "cuda_check.hpp"
#include "whole_layer.hpp"
#endif
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <iostream>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {
using namespace beam;
constexpr int rotation_size = 64;
constexpr int width = 5;
constexpr unsigned batch_size = 2;  // exercise offsets and a partial final batch
constexpr unsigned layers = 10;

int shift(int move) {
    const int inverse = static_cast<int>(MOVE_COUNT) - 1 - move;
    if (move == inverse) return rotation_size / 2;
    const int distance = 1 + (std::min(move, inverse) * 7) % 31;
    return move < inverse ? distance : -distance;
}
int rotate(int value, int move) {
    return (value + shift(move) + rotation_size) % rotation_size;
}
Hash128 hash_for(int value) {
    // Many states share each individual word. Both words together are unique.
    return {static_cast<std::uint64_t>(value % 4), static_cast<std::uint64_t>(value / 4)};
}
unsigned score_for(unsigned layer, unsigned parent, int move, int value) {
    return 3 * value + (parent + move + layer) % 3;
}

struct Choice {
    int value;
    unsigned score;
    unsigned parent;
    int move;
};
using Row = std::vector<Choice>;
using Trace = std::vector<Row>;

// Enumerate transitions from the integer rotation, independently of CUDA's
// child hashes, materialization tables, and sorting/deduplication order.
Trace reference(unsigned window, bool no_backtrack) {
    Row parents{{0, 0, 0, -1}};
    Trace trace;
    std::deque<std::set<int>> history;  // reference TPU starts with empty rows
    for (unsigned layer = 1; layer <= layers; ++layer) {
        Row candidates;
        for (unsigned p = 0; p < parents.size(); ++p) {
            for (int move = 0; move < static_cast<int>(MOVE_COUNT); ++move) {
                if (no_backtrack && parents[p].move >= 0 &&
                    move == static_cast<int>(MOVE_COUNT) - 1 - parents[p].move) continue;
                const int value = rotate(parents[p].value, move);
                if (std::any_of(history.begin(), history.end(),
                    [value](const auto& row) { return row.count(value) != 0; })) continue;
                candidates.push_back({value, score_for(layer, p, move, value), p, move});
            }
        }
        std::sort(candidates.begin(), candidates.end(), [](const Choice& a, const Choice& b) {
            return std::tie(a.score, a.parent, a.move) < std::tie(b.score, b.parent, b.move);
        });
        std::set<int> accepted;
        parents.clear();
        for (const auto& choice : candidates) {
            if (!accepted.insert(choice.value).second) continue;
            parents.push_back(choice);
            if (parents.size() == width) break;
        }
        history.push_back(std::move(accepted));
        if (history.size() > window) history.pop_front();
        trace.push_back(parents);
    }
    return trace;
}

std::vector<int> states(const Trace& trace) {
    std::vector<int> values;
    for (const auto& row : trace) {
        if (row.size() != width) throw std::runtime_error("fixture produced a partial beam");
        for (const auto& choice : row) values.push_back(choice.value);
    }
    return values;
}
void check_coverage(const Trace& expected) {
    const auto values = states(expected);
    std::set<std::uint64_t> low, high;
    std::set<int> unique(values.begin(), values.end());
    for (int value : unique) {
        const auto h = hash_for(value);
        low.insert(h.lo);
        high.insert(h.hi);
    }
    if (unique.size() <= low.size() || unique.size() <= high.size())
        throw std::runtime_error("fixture missed collisions in the individual hash words");
    for (unsigned window : {0u, 3u, 5u}) {
        if (values == states(reference(window, true)))
            throw std::runtime_error("fixture missed incorrect history window " + std::to_string(window));
    }
    if (values == states(reference(4, false)))
        throw std::runtime_error("fixture missed missing inverse-move rejection");
}

#ifndef BEAM_FILTER_ORACLE_ONLY
template<class T> struct DeviceArray {
    T* ptr = nullptr;
    explicit DeviceArray(std::size_t count) { BEAM_CUDA_CHECK(cudaMalloc(&ptr, count * sizeof(T))); }
    ~DeviceArray() { cudaFree(ptr); }
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;
};
struct Event {
    cudaEvent_t value{};
    Event() { BEAM_CUDA_CHECK(cudaEventCreateWithFlags(&value, cudaEventDisableTiming)); }
    ~Event() { cudaEventDestroy(value); }
};
struct CheckpointFile {
    std::filesystem::path path = std::filesystem::temp_directory_path() /
        ("cube555-filter-test-" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".bin");
    ~CheckpointFile() {
        std::error_code ignored;
        std::filesystem::remove(path, ignored);
    }
};
struct Fixture {
    StaticDeviceMemory& memory;
    const std::vector<std::uint8_t>& generators;
    unsigned layer;
};

// Replace only Stream1/2. Read the actual device frontier produced by the
// last select(), then supply this batch's scores and hashes to expand().
void launch(const DispatcherRingSlotLaunchContext& job, void* user) {
    auto& f = *static_cast<Fixture*>(user);
    std::vector<State128> parents(job.count);
    BEAM_CUDA_CHECK(cudaMemcpyAsync(parents.data(), f.memory.current_frontier_states + job.parent_base,
        parents.size() * sizeof(State128), cudaMemcpyDeviceToHost, job.stream1_lane));
    BEAM_CUDA_CHECK(cudaStreamSynchronize(job.stream1_lane));
    std::vector<unsigned> scores(job.count * MOVE_COUNT);
    std::vector<Hash128> hashes(scores.size());
    for (unsigned p = 0; p < job.count; ++p) {
        for (unsigned move = 0; move < MOVE_COUNT; ++move) {
            auto idx = p * MOVE_COUNT + move;
            int value = parents[p].v[f.generators[move * STATE_STORAGE_LEN]];
            hashes[idx] = hash_for(value);
            scores[idx] = score_for(f.layer, static_cast<unsigned>(job.parent_base) + p, move, value);
        }
    }
    BEAM_CUDA_CHECK(cudaMemcpyAsync(f.memory.streams.score_ring, scores.data(),
        scores.size() * sizeof(unsigned), cudaMemcpyHostToDevice, job.stream1_lane));
    BEAM_CUDA_CHECK(cudaMemcpyAsync(f.memory.streams.hash_ring, hashes.data(),
        hashes.size() * sizeof(Hash128), cudaMemcpyHostToDevice, job.stream2_lane));
    BEAM_CUDA_CHECK(cudaStreamSynchronize(job.stream2_lane));
}

void check_device(const Trace& expected) {
    std::vector<std::uint8_t> gen(MOVE_COUNT * STATE_STORAGE_LEN);
    std::array<std::uint8_t, MOVE_COUNT> inverses{};
    for (unsigned move = 0; move < MOVE_COUNT; ++move) {
        inverses[move] = static_cast<std::uint8_t>(MOVE_COUNT - 1 - move);
        for (unsigned p = 0; p < STATE_LEN; ++p)
            gen[move * STATE_STORAGE_LEN + p] = p < rotation_size ? rotate(p, move) : p;
    }
    State128 root{};
    for (int p = 0; p < rotation_size; ++p) root.v[p] = p;
    DeviceArray<State128> frontier(width), next(width);
    DeviceArray<std::uint8_t> generators(gen.size());
    DeviceArray<CandidateMeta> selected(width);
    DeviceArray<unsigned> scores(batch_size * MOVE_COUNT), count(1);
    DeviceArray<Hash128> hashes(batch_size * MOVE_COUNT);
    DeviceArray<std::uint64_t> base(1);
    BEAM_CUDA_CHECK(cudaMemcpy(frontier.ptr, &root, sizeof(root), cudaMemcpyHostToDevice));
    BEAM_CUDA_CHECK(cudaMemcpy(generators.ptr, gen.data(), gen.size(), cudaMemcpyHostToDevice));
    StaticDeviceMemory memory{};
    memory.current_frontier_states = frontier.ptr;
    memory.final.next_frontier_states_tmp = next.ptr;
    memory.final.final_candidate_buffer = selected.ptr;
    memory.streams.score_ring = scores.ptr;
    memory.streams.hash_ring = hashes.ptr;
    memory.streams.parent_base = base.ptr;
    memory.streams.count = count.ptr;
    StaticMemoryPlan plan{};
    plan.config.world_size = 1;
    plan.config.b_micro = batch_size;
    plan.frontier_states = width;
    DispatcherDeviceTables tables{};
    tables.generators = generators.ptr;
    Fixture fixture{memory, gen, 0};
    DispatcherRingSlotLauncher launcher{launch, &fixture, "synthetic cyclic group"};
    if (whole_layer_inverse_moves(gen) != inverses)
        throw std::runtime_error("incorrect inverse permutation table");
    auto search = std::make_unique<WholeLayer>(plan, inverses);
    Event ready;
    CheckpointFile checkpoint;
    std::array<CandidateMeta, width> actual{};
    std::array<State128, width> materialized{};
    std::uint64_t parents = 1;
    for (const auto& row : expected) {
        ++fixture.layer;
        const auto jobs = search->expand(memory, nullptr, launcher, parents);
        if (jobs.ring_slot_jobs_launched != (parents + batch_size - 1) / batch_size)
            throw std::runtime_error("wrong number of expansion batches");
        auto result = search->select(memory, tables, nullptr, actual.data(), width, ready.value);
        if (result.next_frontier_size != row.size()) throw std::runtime_error("wrong survivor count");
        BEAM_CUDA_CHECK(cudaMemcpy(materialized.data(), frontier.ptr, sizeof(materialized), cudaMemcpyDeviceToHost));
        for (unsigned i = 0; i < row.size(); ++i) {
            const auto& ref = row[i];
            const auto h = hash_for(ref.value);
            const auto& a = actual[i];
            if (a.hash.lo != h.lo || a.hash.hi != h.hi || a.score_key != ref.score ||
                a.parent_idx != ref.parent || a.route_packed != static_cast<unsigned>(ref.move))
                throw std::runtime_error("wrong survivor at layer " + std::to_string(fixture.layer) +
                    " rank " + std::to_string(i));
            for (int p = 0; p < rotation_size; ++p)
                if (materialized[i].v[p] != (ref.value + p) % rotation_size)
                    throw std::runtime_error("incorrect device frontier at layer " + std::to_string(fixture.layer));
        }
        parents = result.next_frontier_size;
        if (fixture.layer == 5) {
            search->save_history(checkpoint.path, static_cast<unsigned>(parents));
            search.reset();
            search = std::make_unique<WholeLayer>(plan, inverses);
            search->restore_history(checkpoint.path, static_cast<unsigned>(parents));
        }
    }
}
#endif
}

int main() {
    try {
        if (beam::STATE_LEN < rotation_size || beam::MOVE_COUNT < 10 || beam::MOVE_COUNT > 62) {
            std::cout << "SKIP: cyclic fixture requires at least 64 positions and 10-62 moves\n";
            return 77;
        }
        const auto expected = reference(4, true);
        check_coverage(expected);
#ifndef BEAM_FILTER_ORACLE_ONLY
        int devices = 0;
        auto status = cudaGetDeviceCount(&devices);
        if (status == cudaErrorNoDevice || (status == cudaSuccess && !devices)) {
            std::cout << "SKIP: CUDA device unavailable\n";
            return 77;
        }
        BEAM_CUDA_CHECK(status);
        check_device(expected);
        std::cout << "WholeLayer CUDA matches all ten reference layers\n";
#else
        std::cout << "CPU oracle distinguishes inverse, missing history, and wrong expiry\n";
#endif
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
