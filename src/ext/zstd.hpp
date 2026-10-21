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

// Optional libzstd backend (native zstd frames). Implementation is a no-op
// stub when SPECTRA8_HAVE_ZSTD is not defined.
#ifndef SPECTRA8_SRC_EXT_ZSTD_HPP
#define SPECTRA8_SRC_EXT_ZSTD_HPP

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>

#include "spectra8/spectra8.hpp"

namespace spectra8::detail {

[[nodiscard]] bool zstd_available() noexcept;

// ZSTD_maxCLevel() when linked; 22 otherwise (unused when zstd is unavailable).
[[nodiscard]] int zstd_max_level() noexcept;

[[nodiscard]] std::size_t zstd_compress_bound(std::size_t src_size) noexcept;

[[nodiscard]] Result<std::size_t> zstd_compress(std::span<const std::byte> src,
                                                std::span<std::byte> dst,
                                                int level) noexcept;

[[nodiscard]] Result<std::size_t> zstd_decompress(
    std::span<const std::byte> src, std::span<std::byte> dst) noexcept;

[[nodiscard]] std::optional<std::uint64_t> zstd_content_size(
    std::span<const std::byte> src) noexcept;

[[nodiscard]] bool looks_like_zstd(std::span<const std::byte> src) noexcept;

}  // namespace spectra8::detail

#endif  // SPECTRA8_SRC_EXT_ZSTD_HPP
