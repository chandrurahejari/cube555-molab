#include "whole_layer.hpp"
#include "cuda_check.hpp"
#include "whole_layer_checkpoint.hpp"
#include "tpu_position_weights.cuh"
#include "tpu_routing.cuh"
#include "../src/hash.hpp"
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/remove.h>
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
    std::uint64_t tpu_hash;
    std::uint8_t route;
    std::uint32_t tpu_score;
};
struct BySeedQ {
    __host__ __device__ bool operator()(const Child& a, const Child& b) const {
        return a.tpu_score < b.tpu_score || (a.tpu_score == b.tpu_score && a.ordinal < b.ordinal);
    }
};
struct InvalidMove {
    __host__ __device__ bool operator()(const Child& c) const { return c.score==UINT32_MAX; }
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
constexpr unsigned recent_layers = 4;
constexpr std::uint32_t empty_slot = UINT32_MAX;
struct Frontiers {
    Hash128* keys = nullptr;
    std::uint32_t* slots = nullptr;
    std::uint32_t width = 0;
    std::uint32_t capacity = 1;
};
__device__ void insert_key(Frontiers f, unsigned row, std::uint32_t pos, Hash128 h) {
    auto* slots = f.slots + static_cast<std::size_t>(row) * f.capacity;
    const auto mask = f.capacity - 1;
    auto slot = static_cast<std::uint32_t>(hash128_owner_distribution_key(h)) & mask;
    while (atomicCAS(slots+slot, empty_slot, pos) != empty_slot) slot = (slot+1) & mask;
}
struct Reject {
    Frontiers frontiers;
    unsigned active;
    __device__ bool operator()(const Child& c) const {
        if (c.score == UINT32_MAX) return true;
        const auto mask = frontiers.capacity-1;
        const auto start = static_cast<std::uint32_t>(hash128_owner_distribution_key(c.hash)) & mask;
        for (unsigned row=0; row<active; ++row) {
            const auto* slots = frontiers.slots + static_cast<std::size_t>(row)*frontiers.capacity;
            const auto* keys = frontiers.keys + static_cast<std::size_t>(row)*frontiers.width;
            auto slot = start;
            for (auto index=slots[slot]; index!=empty_slot; index=slots[slot]) {
                const auto h=keys[index];
                if (h.lo==c.hash.lo && h.hi==c.hash.hi) return true;
                slot=(slot+1)&mask;
            }
        }
        return false;
    }
};
__global__ void reindex(Frontiers f, unsigned row, std::uint32_t count) {
    const auto i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i<count) insert_key(f,row,i,f.keys[static_cast<std::size_t>(row)*f.width+i]);
}
__global__ void set_batch(std::uint64_t* base, std::uint32_t* count,
                          std::uint64_t parent, std::uint32_t size) {
    *base = parent;
    *count = size;
}
__global__ void parent_owners(const State128* parents, std::uint8_t* owner, std::uint32_t n) {
    const auto i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= n) return;
    unsigned value=0;
    for (unsigned p=0; p<150; ++p) value+=parents[i].v[p]*tpu_owner_weights[p];
    owner[i]=value & 7U;
}
__global__ void save_children(Child* children, std::uint32_t offset, std::uint32_t size,
                              const Hash128* hashes, const std::uint32_t* scores,
                              const std::uint8_t* incoming, const std::uint8_t* inverse,
                              const State128* parents, const std::uint8_t* generators,
                              const std::uint8_t* owners, bool route, bool seed_sender) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    const auto ordinal = offset+i;
    const auto last = incoming[ordinal/MOVE_COUNT];
    const bool backtrack = last != UINT8_MAX && inverse[last] == ordinal%MOVE_COUNT;
    Child child{hashes[i], backtrack ? UINT32_MAX : scores[i], ordinal, 0, 0, 0};
    if (route && !backtrack) {
        auto bits=scores[i];
        const float q=__uint_as_float(bits);
        if (!isfinite(q)) {
            child.score=UINT32_MAX;
            children[ordinal]=child;
            return;
        }
        // Keep the exact BF16 Q for ranking, including negatives and ties.
        // Fixed-point keys remain available to the original diagnostics.
        if ((bits & 0x7fffffffU)==0) bits=0;
        const auto bf16=(bits + 0x7fffU + ((bits>>16)&1U)) >> 16;
        child.tpu_score=(bf16 & 0x8000U) ? (0xffffU-bf16) : (bf16+0x8000U);
        child.score=__float2uint_rn(fminf(fmaxf(q,0.f),SCORE_MAX_Q)*SCORE_SCALE);
        const auto parent=i/MOVE_COUNT;
        const auto* permutation=generators + (i%MOVE_COUNT)*STATE_STORAGE_LEN;
        unsigned dest=0;
        for (unsigned p=0; p<150; ++p) {
            const auto value=parents[parent].v[permutation[p]];
            dest+=value*tpu_owner_weights[p];
            child.tpu_hash+=static_cast<std::uint64_t>(value)*tpu_hash_weights[p];
        }
        child.route=(seed_sender ? 0U : owners[parent]*8U) + (dest & 7U);
    }
    children[ordinal] = child;
}
__global__ void build_frontier(const Child* chosen, std::uint32_t size,
                               const State128* current, const std::uint8_t* generators,
                               State128* next, CandidateMeta* history, Frontiers f,
                               unsigned row, std::uint8_t* incoming, bool seed) {
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
    incoming[i] = seed ? UINT8_MAX : static_cast<std::uint8_t>(move);
    if (seed) return;
    f.keys[static_cast<std::size_t>(row)*f.width+i] = child.hash;
    insert_key(f,row,i,child.hash);
}
}
struct WholeLayer::Storage {
    Child* children = nullptr;
    std::uint64_t width;
    std::uint32_t batch;
    std::uint32_t size = 0;
    Frontiers recent;
    std::uint8_t* incoming = nullptr;
    std::uint8_t* inverse = nullptr;
    std::array<std::uint32_t,recent_layers> counts{};
    unsigned next_row = 0;
    bool tpu = false;
    bool seed_pending = true;
    std::uint8_t* owners = nullptr;
    std::vector<std::uint32_t> kept_scores;
    ~Storage() {
        cudaFree(children); cudaFree(recent.keys); cudaFree(recent.slots);
        cudaFree(incoming); cudaFree(inverse); cudaFree(owners);
    }
};
InverseMoves whole_layer_inverse_moves(const std::vector<std::uint8_t>& generators) {
    static_assert(MOVE_COUNT <= UINT8_MAX);
    if (generators.size() != MOVE_COUNT*STATE_STORAGE_LEN)
        throw std::runtime_error("Unexpected generator matrix size");
    InverseMoves result;
    result.fill(UINT8_MAX);
    for (unsigned m=0; m<MOVE_COUNT; ++m) {
        for (unsigned n=0; n<MOVE_COUNT; ++n) {
            bool matches = true;
            for (unsigned i=0; i<STATE_LEN; ++i) {
                const auto j=generators[m*STATE_STORAGE_LEN+i];
                if (j>=STATE_LEN || generators[n*STATE_STORAGE_LEN+j]!=i) { matches=false; break; }
            }
            if (!matches) continue;
            if (result[m]!=UINT8_MAX) throw std::runtime_error("Ambiguous inverse generator");
            result[m]=static_cast<std::uint8_t>(n);
        }
        if (result[m]==UINT8_MAX) throw std::runtime_error("Missing inverse generator");
    }
    return result;
}
WholeLayer::WholeLayer(const StaticMemoryPlan& plan,const InverseMoves& inverse_moves,bool tpu_routing) : storage_(std::make_unique<Storage>()) {
    auto& s = *storage_;
    s.width = plan.frontier_states;
    s.batch = plan.config.b_micro;
    s.tpu = tpu_routing;
    if (s.tpu && (STATE_LEN != 150 || MOVE_COUNT != 30 || s.width < 64 || s.width % 64))
        throw std::runtime_error("TPU selection needs Cube555 and a beam divisible by 64");
    if (plan.config.world_size != 1 || s.batch == 0 || s.width == 0 || s.width > (1ULL << 30) ||
        s.width > std::numeric_limits<std::uint32_t>::max() / MOVE_COUNT)
        throw std::runtime_error("Whole-layer beam exceeds the single-GPU candidate index range");
    BEAM_CUDA_CHECK(cudaMalloc(&s.children, s.width * MOVE_COUNT * sizeof(Child)));
    s.recent.width=static_cast<std::uint32_t>(s.width);
    while (s.recent.capacity < 2*s.width) s.recent.capacity*=2;
    BEAM_CUDA_CHECK(cudaMalloc(&s.recent.keys, recent_layers*s.width*sizeof(Hash128)));
    BEAM_CUDA_CHECK(cudaMalloc(&s.recent.slots, static_cast<std::size_t>(recent_layers)*s.recent.capacity*sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMalloc(&s.incoming,s.width));
    BEAM_CUDA_CHECK(cudaMemset(s.incoming, UINT8_MAX, s.width));
    BEAM_CUDA_CHECK(cudaMalloc(&s.inverse,MOVE_COUNT));
    BEAM_CUDA_CHECK(cudaMemcpy(s.inverse,inverse_moves.data(),MOVE_COUNT,cudaMemcpyHostToDevice));
    if (s.tpu) BEAM_CUDA_CHECK(cudaMalloc(&s.owners,s.batch));
}
WholeLayer::~WholeLayer() = default;
void WholeLayer::save_history(const std::filesystem::path& path, std::uint32_t frontier_count) const {
    const auto& s=*storage_;
    save_whole_layer_history(path,frontier_count,s.incoming,s.recent.keys,s.width,s.counts,s.next_row);
}
void WholeLayer::restore_history(const std::filesystem::path& path, std::uint32_t frontier_count) {
    auto& s=*storage_;
    load_whole_layer_history(path,frontier_count,s.incoming,s.recent.keys,s.width,s.counts,s.next_row);
    s.seed_pending=false;
    for (unsigned row=0; row<recent_layers; ++row) {
        if (!s.counts[row]) continue;
        BEAM_CUDA_CHECK(cudaMemset(s.recent.slots+static_cast<std::size_t>(row)*s.recent.capacity,0xff,
                                  s.recent.capacity*sizeof(std::uint32_t)));
        reindex<<<(s.counts[row]+255)/256,256>>>(s.recent,row,s.counts[row]);
        BEAM_CUDA_CHECK(cudaGetLastError());
    }
    BEAM_CUDA_CHECK(cudaDeviceSynchronize());
}
DepthDispatchState WholeLayer::expand(StaticDeviceMemory& m, cudaStream_t stream,
                                     const DispatcherRingSlotLauncher& launcher, std::uint64_t parents,
                                     const std::uint8_t* generators) {
    auto& s = *storage_;
    if (!parents || parents > s.width) throw std::runtime_error("Invalid whole-layer frontier");
    if (s.tpu && generators==nullptr) throw std::runtime_error("Missing GPU generators for TPU selection");
    const bool seed_sender=s.counts[0]==0;
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
        if (s.tpu && !seed_sender) {
            parent_owners<<<(count+255)/256,256,0,stream>>>(m.current_frontier_states+base,s.owners,count);
            BEAM_CUDA_CHECK(cudaGetLastError());
        }
        save_children<<<(candidates+255)/256,256,0,stream>>>(
            s.children, static_cast<std::uint32_t>(base * MOVE_COUNT), candidates,
            m.streams.hash_ring, m.streams.score_ring, s.incoming, s.inverse,
            m.current_frontier_states+base,generators,s.owners,s.tpu,seed_sender);
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
    const bool seed=s.tpu && s.seed_pending;
    auto sent=s.size;
    if (s.tpu && !seed) {
        end=thrust::remove_if(policy,s.children,end,InvalidMove{});
        end=tpu_routing::sender_filter(s.children,end,s.recent.width,stream);
        sent=static_cast<std::uint32_t>(end-s.children);
        end=tpu_routing::receiver_dedup(s.children,end,stream);
    } else {
        thrust::sort(policy,s.children,end,ByStateThenQuality{});
        end=thrust::unique(policy,s.children,end,SameState{});
    }
    const auto unique_count = end-s.children;
    const auto active=static_cast<unsigned>(std::count_if(s.counts.begin(),s.counts.end(),
                                                          [](auto n){return n!=0;}));
    end = thrust::remove_if(policy,s.children,end,Reject{s.recent,active});
    const auto rejected=unique_count-(end-s.children);
    if (s.tpu && !seed) end=tpu_routing::receiver_select(s.children,end,s.recent.width,stream);
    else if (seed) thrust::sort(policy,s.children,end,BySeedQ{});
    else thrust::sort(policy,s.children,end,ByQuality{});
    const auto keep = static_cast<std::uint32_t>(std::min<std::uint64_t>(s.width,end-s.children));
    if (keep > history_capacity) throw std::runtime_error("Invalid whole-layer selection size");
    if (keep) {
        BEAM_CUDA_CHECK(cudaMemsetAsync(s.recent.slots+static_cast<std::size_t>(s.next_row)*s.recent.capacity,0xff,
                                       s.recent.capacity*sizeof(std::uint32_t),stream));
        build_frontier<<<(keep+255)/256,256,0,stream>>>(s.children,keep,m.current_frontier_states,tables.generators,
            m.final.next_frontier_states_tmp,m.final.final_candidate_buffer,s.recent,s.next_row,s.incoming,seed);
        BEAM_CUDA_CHECK(cudaGetLastError());
        BEAM_CUDA_CHECK(cudaMemcpyAsync(m.current_frontier_states,m.final.next_frontier_states_tmp,
            static_cast<std::size_t>(keep)*sizeof(State128),cudaMemcpyDeviceToDevice,stream));
        BEAM_CUDA_CHECK(cudaMemcpyAsync(history,m.final.final_candidate_buffer,static_cast<std::size_t>(keep)*sizeof(CandidateMeta),
            cudaMemcpyDeviceToHost,stream));
        if (!seed) {
            s.counts[s.next_row]=keep;
            s.next_row=(s.next_row+1)%recent_layers;
        }
        s.seed_pending=false;
    }
    BEAM_CUDA_CHECK(cudaEventRecord(history_ready,stream));
    BEAM_CUDA_CHECK(cudaEventSynchronize(history_ready)); // scratch may be reused by the next layer
    FinalizeDepthState result{};
    result.next_frontier_size=result.final_candidate_count=result.final_request_count=keep;
    result.final_threshold=keep ? history[keep-1].score_key : 0;
    result.whole_layer_generated=s.size;
    result.whole_layer_sent=sent;
    result.whole_layer_unique=static_cast<std::uint32_t>(unique_count);
    result.whole_layer_rejected=static_cast<std::uint32_t>(rejected);
    result.whole_layer_min_key=keep ? history[0].score_key : 0;
    result.whole_layer_median_key=keep ? history[keep/2].score_key : 0;
    if (s.tpu && !seed && keep) {
        // The output consists of eight ranked frontiers, not one global ranking.
        s.kept_scores.resize(keep);
        std::transform(history,history+keep,s.kept_scores.begin(),[](const CandidateMeta& c){return c.score_key;});
        const auto limits=std::minmax_element(s.kept_scores.begin(),s.kept_scores.end());
        result.whole_layer_min_key=*limits.first;
        result.final_threshold=*limits.second;
        auto median=s.kept_scores.begin()+keep/2;
        std::nth_element(s.kept_scores.begin(),median,s.kept_scores.end());
        result.whole_layer_median_key=*median;
    }
    return result;
}
}
