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

// One-shot "L4EF" frames: optional byte filters + native zstd/bzip3 payload.
#ifndef SPECTRA8_SRC_EXT_FILTER_FRAME_HPP
#define SPECTRA8_SRC_EXT_FILTER_FRAME_HPP

#include "spectra8/spectra8.hpp"

#include "util/common.hpp"
#include "util/stats.hpp"

namespace spectra8::detail {

inline constexpr u8 kL4efMagic[4] = {'L', '4', 'E', 'F'};
inline constexpr u8 kL4efVersion = 1;
// Standalone stored fallback for external codecs (not a byte-transform filter id).
inline constexpr u8 kL4efFilterStored = 0xFF;

bool looks_like_l4ef(std::span<const std::byte> src) noexcept;

std::size_t l4ef_compress_bound(std::size_t src_size, Codec codec) noexcept;

Result<std::size_t> compress_filtered_frame(std::span<const std::byte> src,
                                            std::span<std::byte> dst, Codec codec,
                                            const CompressParams& params) noexcept;

// Wrap raw `src` in a standalone L4EF frame with an identity payload (used when
// native bzip3 cannot produce a self-consistent frame, e.g. tiny pages).
[[nodiscard]] Result<std::size_t> compress_stored_l4ef(std::span<const std::byte> src,
                                                       std::span<std::byte> dst,
                                                       Codec codec) noexcept;

Result<std::size_t> decompress_l4ef_frame(std::span<const std::byte> src,
                                          std::span<std::byte> dst) noexcept;

std::optional<std::uint64_t> l4ef_content_size(std::span<const std::byte> src) noexcept;

// Statistics view of a finished L4EF frame (filter id / RLE / stored fallback
// read back from the header). Used when the frame was produced indirectly,
// e.g. as the winner of an Auto race, and no observe info was left behind.
[[nodiscard]] CompressObserveInfo observe_l4ef_frame(std::span<const std::byte> frame) noexcept;

}  // namespace spectra8::detail

#endif  // SPECTRA8_SRC_EXT_FILTER_FRAME_HPP
