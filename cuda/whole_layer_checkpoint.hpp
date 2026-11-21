#pragma once

#include "cuda_check.hpp"
#include "types.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace beam {
namespace whole_layer_checkpoint {

// Serialized fields are explicit 64-bit words, independent of C++ struct padding.
// magic/version, Hash128 size, width, frontier length, next row, four active lengths.
using Header = std::array<std::uint64_t, 9>;
constexpr std::uint64_t format = 0x574c484953540001ULL;
using Counts = std::array<std::uint32_t, 4>;

inline std::uint64_t file_bytes(const Header& header, std::uint64_t width,
                                std::uint32_t frontier_count) {
    if (header[0] != format || header[1] != sizeof(Hash128) || header[2] != width ||
        header[3] != frontier_count || frontier_count == 0 || frontier_count > width ||
        width > std::numeric_limits<std::uint32_t>::max() || header[4] >= Counts{}.size())
        throw std::runtime_error("Whole-layer checkpoint format or geometry mismatch");
    std::uint64_t bytes = sizeof(Header) + header[3];
    for (std::size_t i = 5; i < header.size(); ++i) {
        if (header[i] > width)
            throw std::runtime_error("Whole-layer checkpoint history exceeds beam width");
        bytes += header[i] * sizeof(Hash128);
    }
    const auto latest = (header[4]+3) % 4;
    if (header[5+latest] != frontier_count)
        throw std::runtime_error("Checkpoint frontier disagrees with the latest history row");
    const auto active = std::count_if(header.begin()+5, header.end(), [](auto n) { return n != 0; });
    if (active < 4 && (header[4] != static_cast<std::uint64_t>(active) ||
        std::any_of(header.begin()+5,header.begin()+5+active,[](auto n){return n==0;})))
        throw std::runtime_error("Invalid checkpoint history ring");
    return bytes;
}

constexpr std::size_t block_bytes = 1024 * 1024;

inline void write_device(std::ostream& file, std::vector<char>& buffer, const void* source,
                         std::uint64_t bytes) {
    auto device = static_cast<const char*>(source);
    while (bytes != 0) {
        auto n = static_cast<std::size_t>(std::min<std::uint64_t>(buffer.size(), bytes));
        BEAM_CUDA_CHECK(cudaMemcpy(buffer.data(), device, n, cudaMemcpyDeviceToHost));
        file.write(buffer.data(), n);
        device += n;
        bytes -= n;
    }
}

inline void read_device(std::istream& file, std::vector<char>& buffer, void* destination,
                        std::uint64_t bytes) {
    auto device = static_cast<char*>(destination);
    while (bytes != 0) {
        auto n = static_cast<std::size_t>(std::min<std::uint64_t>(buffer.size(), bytes));
        file.read(buffer.data(), n);
        BEAM_CUDA_CHECK(cudaMemcpy(device, buffer.data(), n, cudaMemcpyHostToDevice));
        device += n;
        bytes -= n;
    }
}

}  // namespace whole_layer_checkpoint

inline void save_whole_layer_history(const std::filesystem::path& path, std::uint32_t frontier_count,
                                     const std::uint8_t* incoming_device, const Hash128* rows_device,
                                     std::uint64_t width, const std::array<std::uint32_t, 4>& counts,
                                     std::uint32_t next_row) {
    using namespace whole_layer_checkpoint;
    Header header{format, sizeof(Hash128), width, frontier_count, next_row,
                  counts[0], counts[1], counts[2], counts[3]};
    file_bytes(header, width, frontier_count);
    std::ofstream file;
    file.exceptions(std::ios::failbit | std::ios::badbit);
    file.open(path, std::ios::binary | std::ios::trunc);
    file.write(reinterpret_cast<const char*>(header.data()), sizeof(header));
    std::vector<char> buffer(block_bytes);
    write_device(file, buffer, incoming_device, frontier_count);
    for (std::size_t row = 0; row < counts.size(); ++row)
        write_device(file, buffer, rows_device + row * width, std::uint64_t{counts[row]} * sizeof(Hash128));
    file.close();
}

inline void load_whole_layer_history(const std::filesystem::path& path, std::uint32_t frontier_count,
                                     std::uint8_t* incoming_device, Hash128* rows_device,
                                     std::uint64_t width, std::array<std::uint32_t, 4>& counts,
                                     std::uint32_t& next_row) {
    using namespace whole_layer_checkpoint;
    if (!std::filesystem::is_regular_file(path))
        throw std::runtime_error("Checkpoint lacks whole-layer search history; start a fresh search");
    std::ifstream file;
    file.exceptions(std::ios::failbit | std::ios::badbit);
    file.open(path, std::ios::binary);
    Header header{};
    file.read(reinterpret_cast<char*>(header.data()), sizeof(header));
    if (std::filesystem::file_size(path) != file_bytes(header, width, frontier_count))
        throw std::runtime_error("Wrong whole-layer checkpoint length");
    std::vector<char> buffer(block_bytes);
    for (std::uint32_t offset=0; offset<frontier_count;) {
        const auto n=std::min<std::size_t>(buffer.size(), frontier_count-offset);
        file.read(buffer.data(), n);
        if (std::any_of(buffer.begin(),buffer.begin()+n,[](char m) {
                    return static_cast<unsigned char>(m) >= MOVE_COUNT;
                })) throw std::runtime_error("Checkpoint contains an invalid incoming move");
        BEAM_CUDA_CHECK(cudaMemcpy(incoming_device+offset,buffer.data(),n,cudaMemcpyHostToDevice));
        offset += static_cast<std::uint32_t>(n);
    }
    for (std::size_t row = 0; row < counts.size(); ++row)
        read_device(file, buffer, rows_device + row * width, header[5 + row] * sizeof(Hash128));
    for (std::size_t row = 0; row < counts.size(); ++row)
        counts[row] = static_cast<std::uint32_t>(header[5 + row]);
    next_row = static_cast<std::uint32_t>(header[4]);
}

}  // namespace beam
