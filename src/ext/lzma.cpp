// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 CERN
// Author: Andreas Joachim Peters <andreas.joachim.peters@cern.ch>
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may
// not use this file except in compliance with the License. You may obtain
// a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "ext/lzma.hpp"

#include <cstring>
#include <vector>

#ifdef SPECTRA8_HAVE_LZMA
#include <lzma.h>
#endif

namespace spectra8::detail {

// xz stream magic (XZ Utils).
inline constexpr std::byte kXzMagic[6] = {
    std::byte{0xFD}, std::byte{0x37}, std::byte{0x7A},
    std::byte{0x58}, std::byte{0x5A}, std::byte{0x00}};

bool looks_like_xz(std::span<const std::byte> src) noexcept {
  return src.size() >= 6 && std::memcmp(src.data(), kXzMagic, 6) == 0;
}

#ifdef SPECTRA8_HAVE_LZMA

namespace {

[[nodiscard]] std::uint32_t preset_from_level(int level) noexcept {
  if (level < 0) level = 0;
  if (level > 9) level = 9;
  return static_cast<std::uint32_t>(level);
}

}  // namespace

bool lzma_available() noexcept { return true; }

std::size_t lzma_compress_bound(std::size_t src_size) noexcept {
  return lzma_stream_buffer_bound(src_size);
}

Result<std::size_t> lzma_compress(std::span<const std::byte> src,
                                  std::span<std::byte> dst,
                                  int level) noexcept {
  const std::size_t bound = lzma_stream_buffer_bound(src.size());
  if (dst.size() < bound && dst.size() < src.size() + 64)
    return {Status::BufferTooSmall, 0};
  std::size_t out_pos = 0;
  const lzma_ret rc = lzma_easy_buffer_encode(
      preset_from_level(level), LZMA_CHECK_CRC64, nullptr,
      reinterpret_cast<const uint8_t*>(src.data()), src.size(),
      reinterpret_cast<uint8_t*>(dst.data()), &out_pos, dst.size());
  if (rc == LZMA_BUF_ERROR) return {Status::BufferTooSmall, 0};
  if (rc != LZMA_OK) return {Status::InternalError, 0};
  return {Status::Ok, out_pos};
}

Result<std::size_t> lzma_decompress(std::span<const std::byte> src,
                                    std::span<std::byte> dst) noexcept {
  uint64_t memlimit = UINT64_MAX;
  std::size_t in_pos = 0;
  std::size_t out_pos = 0;
  const lzma_ret rc = lzma_stream_buffer_decode(
      &memlimit, 0, nullptr, reinterpret_cast<const uint8_t*>(src.data()),
      &in_pos, src.size(), reinterpret_cast<uint8_t*>(dst.data()), &out_pos,
      dst.size());
  if (rc == LZMA_BUF_ERROR) return {Status::BufferTooSmall, 0};
  if (rc != LZMA_OK) return {Status::CorruptInput, 0};
  return {Status::Ok, out_pos};
}

std::optional<std::uint64_t> lzma_content_size(
    std::span<const std::byte> src) noexcept {
  if (!looks_like_xz(src)) return std::nullopt;
  // Decode once into a growing scratch to recover the uncompressed size from
  // the Index (xz does not put it in the stream header).
  std::size_t cap = src.size() * 4 + 4096;
  if (cap < 65536) cap = 65536;
  for (int attempt = 0; attempt < 8; ++attempt) {
    std::vector<std::byte> tmp(cap);
    auto r = lzma_decompress(src, tmp);
    if (r.status == Status::Ok) return static_cast<std::uint64_t>(r.value);
    if (r.status != Status::BufferTooSmall) return std::nullopt;
    cap *= 2;
  }
  return std::nullopt;
}

#else  // !SPECTRA8_HAVE_LZMA

bool lzma_available() noexcept { return false; }

std::size_t lzma_compress_bound(std::size_t) noexcept { return 0; }

Result<std::size_t> lzma_compress(std::span<const std::byte>,
                                  std::span<std::byte>, int) noexcept {
  return {Status::InvalidArgument, 0};
}

Result<std::size_t> lzma_decompress(std::span<const std::byte>,
                                    std::span<std::byte>) noexcept {
  return {Status::UnsupportedVersion, 0};
}

std::optional<std::uint64_t> lzma_content_size(
    std::span<const std::byte>) noexcept {
  return std::nullopt;
}

#endif

}  // namespace spectra8::detail
