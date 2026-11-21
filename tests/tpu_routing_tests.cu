#ifndef BEAM_ROUTING_ORACLE_ONLY
#include "tpu_routing.cuh"
#endif
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
struct Child {
    std::uint64_t tpu_hash;
    std::uint32_t score, ordinal;
    std::uint8_t route;
    std::uint32_t tpu_score;
};
using Children = std::vector<Child>;
struct Stages {
    Children sent, received, kept;
};
constexpr unsigned width = 1024;

bool in_history(const Child& c) { return c.tpu_hash % 13 == 1; }

Stages reference(const Children& inputs) {
    constexpr unsigned n = 8;
    std::array<std::array<Children,n>,n> buckets;
    for (const auto& c : inputs) buckets[c.route / n][c.route % n].push_back(c);
    Stages out;
    for (auto& source : buckets) for (auto& bucket : source) {
        // Input order is flat parent/move order, as for lax.top_k's stable ties.
        std::stable_sort(bucket.begin(),bucket.end(),[](const Child& a, const Child& b) {
            return a.tpu_score < b.tpu_score;
        });
        if (bucket.size() > 2 * (width / n) / n) bucket.resize(2 * (width / n) / n);
        out.sent.insert(out.sent.end(),bucket.begin(),bucket.end());
    }
    for (unsigned dst = 0; dst < n; ++dst) {
        Children received;
        for (auto& src : buckets) received.insert(received.end(), src[dst].begin(), src[dst].end());
        // Stable JAX argsort preserves the received sequence for equal hashes.
        std::stable_sort(received.begin(),received.end(),[](const Child& a, const Child& b) {
            return static_cast<std::int64_t>(a.tpu_hash) < static_cast<std::int64_t>(b.tpu_hash);
        });
        received.erase(std::unique(received.begin(),received.end(),[](const Child& a, const Child& b) {
            return a.tpu_hash == b.tpu_hash;
        }),received.end());
        out.received.insert(out.received.end(),received.begin(),received.end());
        received.erase(std::remove_if(received.begin(),received.end(),in_history),received.end());
        std::stable_sort(received.begin(),received.end(),[](const Child& a, const Child& b) {
            return a.tpu_score < b.tpu_score;
        });
        if (received.size() > width/n) received.resize(width/n);
        out.kept.insert(out.kept.end(),received.begin(),received.end());
    }
    return out;
}
Children fixture(bool sparse) {
    std::mt19937 rng(21);
    Children input;
    for (unsigned i = 0; i < (sparse ? 340u : 40000u); ++i) {
        const unsigned key = rng() % (sparse ? 55 : 2000);
        const unsigned dst = rng() % (sparse ? 3 : 8);
        const unsigned src = rng() % (sparse ? 6 : 8);
        const auto h = key % 3 == 0 ? UINT64_MAX - key : key;
        const unsigned score = 100000 + rng() % 3000;
        input.push_back({h, score, i, static_cast<std::uint8_t>(src*8 + dst),
                         static_cast<std::uint32_t>(rng()%17)});
    }
    return input;
}
void check_precedence() {
    // Sender 0's duplicate loses on score but wins on the original receive
    // order. Both real scores subsequently occupy the same BF16 bucket.
    Children inputs{{10,100100,0,0,42000}, {10,100000,1,8,41999}, {UINT64_MAX,100220,2,8,42000}};
    const auto expected = reference(inputs);
    if (expected.received.size()!=2 || expected.received[1].ordinal!=0 ||
        expected.kept.size()!=2 || expected.kept[0].ordinal!=2 || expected.kept[1].ordinal!=0)
        throw std::runtime_error("reference lost sender or stable signed-hash precedence");
}

#ifndef BEAM_ROUTING_ORACLE_ONLY
void compare(const char* stage, const Children& expected, Child* first, Child* end) {
    if (static_cast<std::size_t>(end-first) != expected.size())
        throw std::runtime_error(std::string(stage) + ": different size");
    Children actual(expected.size());
    if (!actual.empty())
        BEAM_CUDA_CHECK(cudaMemcpy(actual.data(),first,actual.size()*sizeof(Child),cudaMemcpyDeviceToHost));
    for (std::size_t i=0; i<actual.size(); ++i) {
        if (actual[i].ordinal != expected[i].ordinal || actual[i].tpu_hash != expected[i].tpu_hash ||
            actual[i].score != expected[i].score || actual[i].route != expected[i].route ||
            actual[i].tpu_score != expected[i].tpu_score)
            throw std::runtime_error(std::string(stage) + ": different record at " + std::to_string(i));
    }
}
struct Previous {
    __device__ bool operator()(const Child& c) const { return c.tpu_hash % 13 == 1; }
};
void check_device(const Children& input, const Stages& expected) {
    Child* first{};
    BEAM_CUDA_CHECK(cudaMalloc(&first,input.size()*sizeof(Child)));
    try {
        BEAM_CUDA_CHECK(cudaMemcpy(first,input.data(),input.size()*sizeof(Child),cudaMemcpyHostToDevice));
        auto end = beam::tpu_routing::sender_filter(first,first+input.size(),width,nullptr);
        compare("sender",expected.sent,first,end);
        end = beam::tpu_routing::receiver_dedup(first,end,nullptr);
        compare("receive hash",expected.received,first,end);
        end = thrust::remove_if(thrust::cuda::par.on(nullptr),first,end,Previous{});
        end = beam::tpu_routing::receiver_select(first,end,width,nullptr);
        compare("receiver top-k",expected.kept,first,end);
    } catch (...) {
        cudaFree(first);
        throw;
    }
    BEAM_CUDA_CHECK(cudaFree(first));
}
#endif
}

int main() {
    try {
        check_precedence();
#ifndef BEAM_ROUTING_ORACLE_ONLY
        int devices = 0;
        auto status = cudaGetDeviceCount(&devices);
        if (status == cudaErrorNoDevice || (status == cudaSuccess && !devices)) return 77;
        BEAM_CUDA_CHECK(status);
#endif
        for (bool sparse : {false,true}) {
            const auto input = fixture(sparse);
            const auto expected = reference(input);
            if (expected.kept.empty() || (sparse && expected.kept.size() >= width) ||
                (!sparse && expected.kept.size()!=width) || expected.sent.size() <= expected.received.size())
                throw std::runtime_error("fixture did not exercise receiver dedup and beam quota");
#ifndef BEAM_ROUTING_ORACLE_ONLY
            check_device(input,expected);
            std::cout << "CUDA matched ";
#else
            std::cout << "CPU reference: ";
#endif
            std::cout << input.size() << " input; " << expected.sent.size() << " sent; "
                      << expected.received.size() << " unique; " << expected.kept.size() << " kept\n";
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
