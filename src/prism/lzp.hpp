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

#ifndef SPECTRA8_SRC_LZP_HPP
#define SPECTRA8_SRC_LZP_HPP

#include <cstddef>
#include <cstdint>
#include <vector>

#include "util/common.hpp"

namespace spectra8::detail {

// LZP (Lempel-Ziv Prediction) pre-filter stage.
// Collapses long redundant matches without emitting offset/distance tokens,
// significantly improving BWT suffix-sorting and compression ratio on text/code.
inline constexpr std::size_t kLzpMinBlock = 128;
inline constexpr std::size_t kLzpMinMatch = 36;
inline constexpr std::uint8_t kLzpMatchMarker = 0xF2;

// Compress `src[0, n)` into `dst`. Returns true if compressed size < n and fits in dst_cap.
// On success, out_len holds the compressed size.
bool lzp_compress(const std::uint8_t* src, std::size_t n, std::uint8_t* dst,
                  std::size_t dst_cap, std::size_t& out_len) noexcept;

// Decompress `src[0, src_len)` into `dst[0, orig_len)`. Returns Status::Ok on success.
Status lzp_decompress(const std::uint8_t* src, std::size_t src_len,
                      std::uint8_t* dst, std::size_t orig_len) noexcept;

}  // namespace spectra8::detail

#endif  // SPECTRA8_SRC_LZP_HPP
