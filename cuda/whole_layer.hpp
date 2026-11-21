#pragma once

#include "dispatcher.hpp"
#include <memory>

namespace beam {

// Single-GPU, exhaustive expansion of one beam layer. The same Stream1
// launcher supplies model scores and radius-table hits as the sharded path.
class WholeLayer {
public:
    explicit WholeLayer(const StaticMemoryPlan& plan);
    ~WholeLayer();
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
