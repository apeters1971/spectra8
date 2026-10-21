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

#include "ext/zstd.hpp"

#include <cstring>

#ifdef SPECTRA8_HAVE_ZSTD
#include <zstd.h>
// Error enum / ZSTD_getErrorCode live here; zstd.h alone is enough on some
// newer installs but not on EL9's libzstd-devel.
#include <zstd_errors.h>
#endif

namespace spectra8::detail {

namespace {
// zstd magic number, little-endian 0xFD2FB528.
inline constexpr std::byte kZstdMagic[4] = {
    std::byte{0x28}, std::byte{0xB5}, std::byte{0x2F}, std::byte{0xFD}};
}  // namespace

bool looks_like_zstd(std::span<const std::byte> src) noexcept {
  return src.size() >= 4 &&
         std::memcmp(src.data(), kZstdMagic, 4) == 0;
}

#ifdef SPECTRA8_HAVE_ZSTD

bool zstd_available() noexcept { return true; }

int zstd_max_level() noexcept { return ZSTD_maxCLevel(); }

std::size_t zstd_compress_bound(std::size_t src_size) noexcept {
  return ZSTD_compressBound(src_size);
}

Result<std::size_t> zstd_compress(std::span<const std::byte> src,
                                  std::span<std::byte> dst,
                                  int level) noexcept {
  if (level < 1) level = 1;
  const int max_lv = ZSTD_maxCLevel();
  if (level > max_lv) level = max_lv;
  const size_t n =
      ZSTD_compress(dst.data(), dst.size(), src.data(), src.size(), level);
  if (ZSTD_isError(n)) {
    if (ZSTD_getErrorCode(n) == ZSTD_error_dstSize_tooSmall)
      return {Status::BufferTooSmall, 0};
    return {Status::InternalError, 0};
  }
  return {Status::Ok, n};
}

Result<std::size_t> zstd_decompress(std::span<const std::byte> src,
                                    std::span<std::byte> dst) noexcept {
  const size_t n =
      ZSTD_decompress(dst.data(), dst.size(), src.data(), src.size());
  if (ZSTD_isError(n)) {
    if (ZSTD_getErrorCode(n) == ZSTD_error_dstSize_tooSmall)
      return {Status::BufferTooSmall, 0};
    return {Status::CorruptInput, 0};
  }
  return {Status::Ok, n};
}

std::optional<std::uint64_t> zstd_content_size(
    std::span<const std::byte> src) noexcept {
  const unsigned long long cs =
      ZSTD_getFrameContentSize(src.data(), src.size());
  if (cs == ZSTD_CONTENTSIZE_ERROR || cs == ZSTD_CONTENTSIZE_UNKNOWN)
    return std::nullopt;
  return static_cast<std::uint64_t>(cs);
}

#else  // !SPECTRA8_HAVE_ZSTD

bool zstd_available() noexcept { return false; }

int zstd_max_level() noexcept { return 22; }

std::size_t zstd_compress_bound(std::size_t) noexcept { return 0; }

Result<std::size_t> zstd_compress(std::span<const std::byte>,
                                  std::span<std::byte>, int) noexcept {
  return {Status::InvalidArgument, 0};
}

Result<std::size_t> zstd_decompress(std::span<const std::byte>,
                                    std::span<std::byte>) noexcept {
  return {Status::UnsupportedVersion, 0};
}

std::optional<std::uint64_t> zstd_content_size(
    std::span<const std::byte>) noexcept {
  return std::nullopt;
}

#endif

}  // namespace spectra8::detail
