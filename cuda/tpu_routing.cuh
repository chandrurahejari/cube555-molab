#pragma once

#include "cuda_check.hpp"
#include "config.hpp"
#include <thrust/iterator/counting_iterator.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/unique.h>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace beam::tpu_routing {

constexpr unsigned ranks = 8;
constexpr unsigned oversample = 2;
// Route is src * ranks + dst. In each input record ordinal must order flat
// parent/move indices within a sender, even while its beam is partially full.
struct Peer {
    template<class Child> __host__ __device__ unsigned operator()(const Child& x) const {
        return x.route;
    }
};
struct Owner {
    template<class Child> __host__ __device__ unsigned operator()(const Child& x) const {
        return x.route % ranks;
    }
};
__host__ __device__ inline std::uint64_t signed_hash_order(std::uint64_t x) {
    return x ^ (std::uint64_t{1} << 63);
}
// tpu_score is the exact ordered BF16 key produced by transformer inference.
// Lower keys are better. It preserves negative and near-zero Q values which
// the native nonnegative, 1/1024 telemetry score cannot represent.
struct SenderOrder {
    template<class Child> __host__ __device__ bool operator()(const Child& a, const Child& b) const {
        if (a.route != b.route) return a.route < b.route;
        if (a.tpu_score != b.tpu_score) return a.tpu_score < b.tpu_score;
        return a.ordinal < b.ordinal;
    }
};
struct ReceiveHashOrder {
    template<class Child> __host__ __device__ bool operator()(const Child& a, const Child& b) const {
        const auto da = Owner{}(a), db = Owner{}(b);
        if (da != db) return da < db;
        if (a.tpu_hash != b.tpu_hash) return signed_hash_order(a.tpu_hash) < signed_hash_order(b.tpu_hash);
        // Stable argsort(hash) on the all_to_all receive buffer means first
        // sender, then that sender's bucket order. A later sender may have
        // given this same child a better Q and must still lose this tie.
        return SenderOrder{}(a, b);
    }
};
struct ReceiveDuplicate {
    template<class Child> __host__ __device__ bool operator()(const Child& a, const Child& b) const {
        return Owner{}(a) == Owner{}(b) && a.tpu_hash == b.tpu_hash;
    }
};
struct ReceiverOrder {
    template<class Child> __host__ __device__ bool operator()(const Child& a, const Child& b) const {
        const auto da = Owner{}(a), db = Owner{}(b);
        if (da != db) return da < db;
        if (a.tpu_score != b.tpu_score) return a.tpu_score < b.tpu_score;
        return signed_hash_order(a.tpu_hash) < signed_hash_order(b.tpu_hash);
    }
};

template<unsigned Groups, class Child, class GroupOf>
__global__ void find_boundaries(const Child* values, std::uint32_t count,
                                std::uint32_t* bounds, GroupOf group) {
    const unsigned g = threadIdx.x;
    if (g > Groups) return;
    std::uint32_t low = 0, high = count;
    while (low < high) {
        const auto mid = low + (high - low) / 2;
        if (group(values[mid]) < g) low = mid + 1;
        else high = mid;
    }
    bounds[g] = low;
}
template<unsigned Groups> struct BeyondQuota {
    const std::uint32_t* bounds;
    std::uint32_t quota;
    __device__ bool operator()(std::uint32_t index) const {
        // Only immutable offsets are read while thrust compacts in place.
        unsigned low = 0, high = Groups;
        while (low + 1 < high) {
            const auto mid = low + (high - low) / 2;
            if (bounds[mid] <= index) low = mid;
            else high = mid;
        }
        return index - bounds[low] >= quota;
    }
};
struct BoundaryBuffer {
    std::uint32_t* device = nullptr;
    explicit BoundaryBuffer(unsigned count) { BEAM_CUDA_CHECK(cudaMalloc(&device, count * sizeof(*device))); }
    ~BoundaryBuffer() { cudaFree(device); }
    BoundaryBuffer(const BoundaryBuffer&) = delete;
    BoundaryBuffer& operator=(const BoundaryBuffer&) = delete;
};
template<unsigned Groups, class Child, class GroupOf>
Child* cap_groups(Child* first, Child* last, unsigned quota, GroupOf group, cudaStream_t stream) {
    if (first == last) return first;
    if (last - first > std::numeric_limits<std::uint32_t>::max())
        throw std::length_error("TPU candidate indices exceed uint32");
    BoundaryBuffer bounds(Groups + 1);
    find_boundaries<Groups><<<1, Groups + 1, 0, stream>>>(
        first, static_cast<std::uint32_t>(last - first), bounds.device, group);
    BEAM_CUDA_CHECK(cudaGetLastError());
    return thrust::remove_if(thrust::cuda::par.on(stream), first, last,
        thrust::make_counting_iterator(std::uint32_t{0}), BeyondQuota<Groups>{bounds.device, quota});
}
inline unsigned local_width(unsigned width) {
    if (width % ranks != 0 || width < ranks * ranks / oversample)
        throw std::invalid_argument("TPU routing requires beam divisible by 8 and at least 32");
    return width / ranks;
}

// Input excludes forbidden inverse moves. History is applied AFTER send
// quotas: histories cannot free a bucket slot for a lower-scoring sibling.
template<class Child>
Child* sender_filter(Child* first, Child* last, unsigned width, cudaStream_t stream) {
    const auto local = local_width(width);
    thrust::sort(thrust::cuda::par.on(stream), first, last, SenderOrder{});
    return cap_groups<ranks * ranks>(first, last, oversample * local / ranks, Peer{}, stream);
}
template<class Child>
Child* receiver_dedup(Child* first, Child* last, cudaStream_t stream) {
    auto policy = thrust::cuda::par.on(stream);
    thrust::sort(policy, first, last, ReceiveHashOrder{});
    return thrust::unique(policy, first, last, ReceiveDuplicate{});
}
// Input is deduplicated and free of any hits in the four previous beams.
// Output is in ascending destination rank, then its chosen slot order.
template<class Child>
Child* receiver_select(Child* first, Child* last, unsigned width, cudaStream_t stream) {
    const auto local = local_width(width);
    thrust::sort(thrust::cuda::par.on(stream), first, last, ReceiverOrder{});
    return cap_groups<ranks>(first, last, local, Owner{}, stream);
}
}  // namespace beam::tpu_routing
