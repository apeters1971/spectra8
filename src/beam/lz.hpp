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

// LZ model: recent-offset cache, sequence representation, offset coding and the
// fast match-length comparison.
#ifndef SPECTRA8_SRC_LZ_HPP
#define SPECTRA8_SRC_LZ_HPP

#include "util/common.hpp"
#include "util/format.hpp"

namespace spectra8::detail {

// A parsed LZ sequence: `lit_len` literals followed by a `match_len` copy from
// `offset` bytes back.
struct Sequence {
  u32 lit_len;
  u32 match_len;
  u32 offset;
};

// Move-to-front cache of recent match offsets for repeat-offset coding.
struct RecentOffsets {
  i32 r[kNumRecentOffsets];

  void init() noexcept {
    for (int i = 0; i < kNumRecentOffsets; ++i) r[i] = i + 1;
  }

  int find(i32 off) const noexcept {
    for (int i = 0; i < kNumRecentOffsets; ++i)
      if (r[i] == off) return i;
    return -1;
  }

  static_assert(kNumRecentOffsets == 7, "use_rep / use_new are unrolled for 7");

  // Move entry `idx` to the front. idx == 0 is a no-op (already at the
  // front) and is the common repeat; the rest unroll into a straight
  // shift of the prefix.
  void use_rep(int idx) noexcept {
    if (idx <= 0) return;
    const i32 v = r[idx];
    switch (idx) {
      case 6: r[6] = r[5]; [[fallthrough]];
      case 5: r[5] = r[4]; [[fallthrough]];
      case 4: r[4] = r[3]; [[fallthrough]];
      case 3: r[3] = r[2]; [[fallthrough]];
      case 2: r[2] = r[1]; [[fallthrough]];
      case 1: r[1] = r[0];
    }
    r[0] = v;
  }

  void use_new(i32 off) noexcept {
    r[6] = r[5];
    r[5] = r[4];
    r[4] = r[3];
    r[3] = r[2];
    r[2] = r[1];
    r[1] = r[0];
    r[0] = off;
  }
};

// Number of significant bits in x (0 for x == 0; __builtin_clz(0) is UB and
// the lazy parser prices a "no match" probe with offset 0).
inline int bit_width(u32 x) noexcept {
  return x ? 32 - __builtin_clz(x) : 0;
}

// Length of the common prefix of [a, a_end) and b.
// 16-byte steps first (two XOR+ctz, one trip for matches >= 8 and < 16),
// then an 8-byte tail. Same result as a pure 8-byte loop.
inline usize match_len(const u8* a, const u8* b, const u8* a_end) noexcept {
  const u8* a0 = a;
  while (a + 16 <= a_end) {
    const u64 x0 = load_u64(a) ^ load_u64(b);
    if (x0)
      return static_cast<usize>(a - a0) +
             (static_cast<usize>(__builtin_ctzll(x0)) >> 3);
    const u64 x1 = load_u64(a + 8) ^ load_u64(b + 8);
    if (x1)
      return static_cast<usize>(a - a0) + 8 +
             (static_cast<usize>(__builtin_ctzll(x1)) >> 3);
    a += 16;
    b += 16;
  }
  if (a + 8 <= a_end) {
    const u64 x = load_u64(a) ^ load_u64(b);
    if (x)
      return static_cast<usize>(a - a0) +
             (static_cast<usize>(__builtin_ctzll(x)) >> 3);
    a += 8;
    b += 8;
  }
  while (a < a_end && *a == *b) {
    ++a;
    ++b;
  }
  return static_cast<usize>(a - a0);
}

// Length of the common prefix when the first 4 bytes are already known to match.
inline usize match_len_from4(const u8* a, const u8* b, const u8* a_end) noexcept {
  const usize total = static_cast<usize>(a_end - a);
  if (total <= 4) return total;
  return 4 + match_len(a + 4, b + 4, a_end);
}

}  // namespace spectra8::detail

#endif  // SPECTRA8_SRC_LZ_HPP
