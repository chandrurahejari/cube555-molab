#pragma once

#include "dispatcher.hpp"
#include <array>
#include <filesystem>
#include <memory>
#include <vector>

namespace beam {

using InverseMoves = std::array<std::uint8_t, MOVE_COUNT>;
InverseMoves whole_layer_inverse_moves(const std::vector<std::uint8_t>& generators);

// Single-GPU, exhaustive expansion of one beam layer. The same Stream1
// launcher supplies model scores and radius-table hits as the sharded path.
class WholeLayer {
public:
    WholeLayer(const StaticMemoryPlan& plan, const InverseMoves& inverse_moves);
    ~WholeLayer();
    void save_history(const std::filesystem::path& path, std::uint32_t frontier_count) const;
    void restore_history(const std::filesystem::path& path, std::uint32_t frontier_count);
    DepthDispatchState expand(StaticDeviceMemory& memory, cudaStream_t stream,
                              const DispatcherRingSlotLauncher& launcher, std::uint64_t parents);
    FinalizeDepthState select(StaticDeviceMemory& memory, const DispatcherDeviceTables& tables,
                              cudaStream_t stream, CandidateMeta* history, std::uint32_t history_capacity,
                              cudaEvent_t history_ready);
private:
    struct Storage;
    std::unique_ptr<Storage> storage_;
};
}
