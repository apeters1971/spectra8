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

// 64-bit content checksum.
//
// This is a self-contained, well-mixed 64-bit hash in the spirit of xxHash. It
// is only required to be consistent between the encoder and decoder of this
// library (the Spectra8 format is not interchangeable with any other), so an
// internal mixer is used rather than depending on an external xxhash.h.
#ifndef SPECTRA8_SRC_CHECKSUM_HPP
#define SPECTRA8_SRC_CHECKSUM_HPP

#include "util/common.hpp"

namespace spectra8::detail {

inline u64 content_hash(const u8* data, usize len, u64 seed = 0) noexcept {
  constexpr u64 P1 = 0x9E3779B185EBCA87ull;
  constexpr u64 P2 = 0xC2B2AE3D27D4EB4Full;
  constexpr u64 P3 = 0x165667B19E3779F9ull;
  constexpr u64 P5 = 0x27D4EB2F165667C5ull;

  auto rotl = [](u64 x, int r) noexcept { return (x << r) | (x >> (64 - r)); };
  auto round = [&](u64 acc, u64 in) noexcept {
    acc += in * P2;
    acc = rotl(acc, 31);
    acc *= P1;
    return acc;
  };

  const u8* p = data;
  const u8* end = data + len;
  u64 h;

  if (len >= 32) {
    u64 v1 = seed + P1 + P2;
    u64 v2 = seed + P2;
    u64 v3 = seed;
    u64 v4 = seed - P1;
    const u8* limit = end - 32;
    do {
      v1 = round(v1, load_u64(p)); p += 8;
      v2 = round(v2, load_u64(p)); p += 8;
      v3 = round(v3, load_u64(p)); p += 8;
      v4 = round(v4, load_u64(p)); p += 8;
    } while (p <= limit);

    h = rotl(v1, 1) + rotl(v2, 7) + rotl(v3, 12) + rotl(v4, 18);
    auto merge = [&](u64 acc, u64 v) noexcept {
      acc ^= round(0, v);
      return acc * P1 + P5;
    };
    h = merge(h, v1);
    h = merge(h, v2);
    h = merge(h, v3);
    h = merge(h, v4);
  } else {
    h = seed + P5;
  }

  h += static_cast<u64>(len);

  while (p + 8 <= end) {
    h ^= round(0, load_u64(p));
    h = rotl(h, 27) * P1 + P5;
    p += 8;
  }
  if (p + 4 <= end) {
    h ^= static_cast<u64>(load_u32(p)) * P1;
    h = rotl(h, 23) * P2 + P3;
    p += 4;
  }
  while (p < end) {
    h ^= static_cast<u64>(*p++) * P5;
    h = rotl(h, 11) * P1;
  }

  h ^= h >> 33;
  h *= P2;
  h ^= h >> 29;
  h *= P3;
  h ^= h >> 32;
  return h;
}

}  // namespace spectra8::detail

#endif  // SPECTRA8_SRC_CHECKSUM_HPP
