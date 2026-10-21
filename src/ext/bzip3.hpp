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

// Optional libbzip3 backend (native bzip3 frames). Implementation is a no-op
// stub when SPECTRA8_HAVE_BZIP3 is not defined.
#ifndef SPECTRA8_SRC_EXT_BZIP3_HPP
#define SPECTRA8_SRC_EXT_BZIP3_HPP

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>

#include "spectra8/spectra8.hpp"

namespace spectra8::detail {

[[nodiscard]] bool bzip3_available() noexcept;

[[nodiscard]] std::size_t bzip3_compress_bound(std::size_t src_size) noexcept;

// `block_log` 0 => default (~1 MiB). Otherwise clamped to bzip3's legal range.
// Native bzip3 frame, or an identity copy when libbzip3 cannot roundtrip (for
// use as an L4EF inner payload).
[[nodiscard]] Result<std::size_t> bzip3_compress_payload(std::span<const std::byte> src,
                                                         std::span<std::byte> dst,
                                                         int block_log) noexcept;

[[nodiscard]] Result<std::size_t> bzip3_compress(std::span<const std::byte> src,
                                                 std::span<std::byte> dst,
                                                 int block_log) noexcept;

[[nodiscard]] Result<std::size_t> bzip3_decompress(
    std::span<const std::byte> src, std::span<std::byte> dst) noexcept;

[[nodiscard]] std::optional<std::uint64_t> bzip3_content_size(
    std::span<const std::byte> src) noexcept;

[[nodiscard]] bool looks_like_bzip3(std::span<const std::byte> src) noexcept;

}  // namespace spectra8::detail

#endif  // SPECTRA8_SRC_EXT_BZIP3_HPP
