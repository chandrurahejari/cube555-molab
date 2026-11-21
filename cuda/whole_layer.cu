#include "whole_layer.hpp"
#include "cuda_check.hpp"
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace beam {
namespace {
struct Child {
    Hash128 hash;
    std::uint32_t score;
    std::uint32_t ordinal;  // parent * MOVE_COUNT + move
};
struct ByQuality {
    __host__ __device__ bool operator()(const Child& a, const Child& b) const {
        return a.score < b.score || (a.score == b.score && a.ordinal < b.ordinal);
    }
};
struct ByStateThenQuality {
    __host__ __device__ bool operator()(const Child& a, const Child& b) const {
        if (a.hash.hi != b.hash.hi) return a.hash.hi < b.hash.hi;
        if (a.hash.lo != b.hash.lo) return a.hash.lo < b.hash.lo;
        return ByQuality{}(a,b);
    }
};
struct SameState {
    __host__ __device__ bool operator()(const Child& a, const Child& b) const {
        return a.hash.hi == b.hash.hi && a.hash.lo == b.hash.lo;
    }
};
__global__ void set_batch(std::uint64_t* base, std::uint32_t* count,
                          std::uint64_t parent, std::uint32_t size) {
    *base = parent;
    *count = size;
}
__global__ void save_children(Child* children, std::uint32_t offset, std::uint32_t size,
                              const Hash128* hashes, const std::uint32_t* scores) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) children[offset+i] = Child{hashes[i], scores[i], offset+i};
}
__global__ void build_frontier(const Child* chosen, std::uint32_t size,
                               const State128* current, const std::uint8_t* generators,
                               State128* next, CandidateMeta* history) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    const Child child = chosen[i];
    const std::uint32_t parent = child.ordinal / MOVE_COUNT;
    const std::uint32_t move = child.ordinal % MOVE_COUNT;
    State128 result{};
    const State128 state = current[parent];
    for (std::uint32_t p=0; p<STATE_LEN; ++p)
        result.v[p] = state.v[generators[move * STATE_STORAGE_LEN + p]];
    next[i] = result;
    history[i] = CandidateMeta{child.hash, parent, child.score, move};
}
}
struct WholeLayer::Storage {
    Child* children = nullptr;
    std::uint64_t width;
    std::uint32_t batch;
    std::uint32_t size = 0;
    ~Storage() { if (children) cudaFree(children); }
};
WholeLayer::WholeLayer(const StaticMemoryPlan& plan) : storage_(std::make_unique<Storage>()) {
    auto& s = *storage_;
    s.width = plan.frontier_states;
    s.batch = plan.config.b_micro;
    if (plan.config.world_size != 1 || s.batch == 0 || s.width == 0 ||
        s.width > std::numeric_limits<std::uint32_t>::max() / MOVE_COUNT)
        throw std::runtime_error("Whole-layer beam exceeds the single-GPU candidate index range");
    BEAM_CUDA_CHECK(cudaMalloc(&s.children, s.width * MOVE_COUNT * sizeof(Child)));
}
WholeLayer::~WholeLayer() = default;
DepthDispatchState WholeLayer::expand(StaticDeviceMemory& m, cudaStream_t stream,
                                     const DispatcherRingSlotLauncher& launcher, std::uint64_t parents) {
    auto& s = *storage_;
    if (!parents || parents > s.width) throw std::runtime_error("Invalid whole-layer frontier");
    DepthDispatchState state{};
    for (std::uint64_t base = 0; base < parents; base += s.batch) {
        const auto count = static_cast<std::uint32_t>(std::min<std::uint64_t>(s.batch, parents-base));
        set_batch<<<1,1,0,stream>>>(m.streams.parent_base, m.streams.count, base, count);
        BEAM_CUDA_CHECK(cudaGetLastError());
        DispatcherRingSlotLaunchContext job{};
        job.job=job.ring=job.ring_slot=job.lane=0;
        job.b_micro=s.batch;
        job.parent_base=base;
        job.count=count;
        job.stream1_lane=job.stream2_lane=stream;
        launcher.launch(job, launcher.user);
        const auto candidates = count * static_cast<std::uint32_t>(MOVE_COUNT);
        save_children<<<(candidates+255)/256,256,0,stream>>>(
            s.children, static_cast<std::uint32_t>(base * MOVE_COUNT), candidates,
            m.streams.hash_ring, m.streams.score_ring);
        BEAM_CUDA_CHECK(cudaGetLastError());
        ++state.ring_slot_jobs_launched;
    }
    BEAM_CUDA_CHECK(cudaStreamSynchronize(stream)); // publish hits before the host reads the goal counters
    s.size = static_cast<std::uint32_t>(parents*MOVE_COUNT);
    state.frontier_cursor=state.frontier_size=parents;
    state.depth_drained=true;
    return state;
}
FinalizeDepthState WholeLayer::select(StaticDeviceMemory& m, const DispatcherDeviceTables& tables,
                                     cudaStream_t stream, CandidateMeta* history, std::uint32_t history_capacity,
                                     cudaEvent_t history_ready) {
    auto& s = *storage_;
    auto policy = thrust::cuda::par.on(stream);
    auto* end = s.children+s.size;
    thrust::sort(policy, s.children,end,ByStateThenQuality{});
    end = thrust::unique(policy,s.children,end,SameState{});
    thrust::sort(policy,s.children,end,ByQuality{});
    const auto keep = static_cast<std::uint32_t>(std::min<std::uint64_t>(s.width,end-s.children));
    if (!keep || keep > history_capacity) throw std::runtime_error("Invalid whole-layer selection size");
    build_frontier<<<(keep+255)/256,256,0,stream>>>(s.children,keep,m.current_frontier_states,tables.generators,
        m.final.next_frontier_states_tmp,m.final.final_candidate_buffer);
    BEAM_CUDA_CHECK(cudaGetLastError());
    BEAM_CUDA_CHECK(cudaMemcpyAsync(m.current_frontier_states,m.final.next_frontier_states_tmp,
        static_cast<std::size_t>(keep)*sizeof(State128),cudaMemcpyDeviceToDevice,stream));
    BEAM_CUDA_CHECK(cudaMemcpyAsync(history,m.final.final_candidate_buffer,static_cast<std::size_t>(keep)*sizeof(CandidateMeta),
        cudaMemcpyDeviceToHost,stream));
    BEAM_CUDA_CHECK(cudaEventRecord(history_ready,stream));
    BEAM_CUDA_CHECK(cudaEventSynchronize(history_ready)); // scratch may be reused by the next layer
    FinalizeDepthState result{};
    result.next_frontier_size=result.final_candidate_count=result.final_request_count=keep;
    result.final_threshold=history[keep-1].score_key;
    return result;
}
}
