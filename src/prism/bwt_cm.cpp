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

#include "prism/bwt_cm.hpp"

#include <cstring>
#include <vector>

namespace spectra8::detail {
namespace {

struct CmState {
  std::uint16_t C0[256];
  std::uint16_t C1[256][256];
  std::uint16_t C2[512][17];

  void reset() noexcept {
    for (int i = 0; i < 256; ++i) C0[i] = 1 << 15;
    for (int i = 0; i < 256; ++i)
      for (int j = 0; j < 256; ++j) C1[i][j] = 1 << 15;
    for (int i = 0; i < 2; ++i)
      for (int j = 0; j < 256; ++j)
        for (int k = 0; k < 17; ++k)
          C2[2 * j + i][k] = static_cast<std::uint16_t>((k << 12) - (k == 16));
  }
};

inline void update0(std::uint16_t& p, int rate) noexcept {
  p = static_cast<std::uint16_t>(p - (p >> rate));
}

inline void update1(std::uint16_t& p, int rate) noexcept {
  p = static_cast<std::uint16_t>(p + ((p ^ 0xFFFF) >> rate));
}

// The two SSE interpolation counters C2[row][j], C2[row][j+1] are adjacent
// u16s, read and written as one little-endian u32 (unaligned access; every
// supported target does it). Both halves update with rate 6 in the same
// direction, and neither half can carry into the other: for update1,
// h + ((0xFFFF ^ h) >> 6) <= 0xFFFF, for update0, h - (h >> 6) >= 0. The
// masks keep the shifted-out bits of the high half out of the low half.
inline std::uint32_t load_pair(const std::uint16_t* p) noexcept {
  std::uint32_t v;
  std::memcpy(&v, p, 4);
  return v;
}
inline void store_pair(std::uint16_t* p, std::uint32_t v) noexcept {
  std::memcpy(p, &v, 4);
}
inline std::uint32_t pair_update1(std::uint32_t x) noexcept {
  return x + (((x ^ 0xFFFFFFFFu) >> 6) & 0x03FF03FFu);
}
inline std::uint32_t pair_update0(std::uint32_t x) noexcept {
  return x - ((x >> 6) & 0x03FF03FFu);
}

// Opaque statement that keeps the two arms of the bit decision from being
// if-converted into selects (see the decoder's hot-loop notes). Emits no code.
#if defined(__GNUC__) || defined(__clang__)
#define SPECTRA8_CM_KEEP_BRANCH(tag) asm volatile("# cm bit " tag ::: "memory")
#else
#define SPECTRA8_CM_KEEP_BRANCH(tag) ((void)0)
#endif

}  // namespace

bool bwt_cm_encode(const std::uint8_t* src, std::size_t n, std::vector<std::uint8_t>& out) {
  if (n == 0) return true;

  thread_local CmState st;
  st.reset();

  // Conservative upper bound for output buffer: raw size + 64 KiB
  const std::size_t bound = n + (n >> 4) + 64;
  const std::size_t initial_pos = out.size();
  out.resize(initial_pos + bound);

  std::uint8_t* out_p = out.data() + initial_pos;
  const std::uint8_t* const out_limit = out_p + bound - 8;

  std::uint32_t high = 0xFFFFFFFFu;
  std::uint32_t low = 0;
  std::uint32_t c1 = 0;
  std::uint32_t c2 = 0;
  std::uint32_t run = 0;

  for (std::size_t i = 0; i < n; ++i) {
#if defined(__GNUC__) || defined(__clang__)
    if (i + 1 < n) {
      __builtin_prefetch(st.C1[src[i + 1]], 0, 0);
    }
#endif
    std::uint8_t c = src[i];
    if (c1 == c2)
      ++run;
    else
      run = 0;

    const int f = (run > 2) ? 1 : 0;
    int ctx = 1;
    std::uint16_t* const row1 = st.C1[c1];
    const std::uint16_t* const row2 = st.C1[c2];

    while (ctx < 256) {
      const int p0 = st.C0[ctx];
      const int p1 = row1[ctx];
      const int p2 = row2[ctx];
      const int p = ((p0 + p1) * 7 + p2 + p2) >> 4;

      const int j = p >> 12;
      std::uint16_t* const c2_row = st.C2[2 * ctx + f];
      const int x1 = c2_row[j];
      const int x2 = c2_row[j + 1];
      const int ssep = x1 + (((x2 - x1) * (p & 4095)) >> 12);

      const std::uint32_t mid =
          low + static_cast<std::uint32_t>(
                    (static_cast<std::uint64_t>(high - low) *
                     static_cast<std::uint64_t>(ssep * 3 + p)) >>
                    18);

      const bool bit = (c & 0x80) != 0;
      if (bit) {
        high = mid;
      } else {
        low = mid + 1;
      }

      while ((low ^ high) < (1u << 24)) {
        if (out_p >= out_limit) return false;
        *out_p++ = static_cast<std::uint8_t>(low >> 24);
        low <<= 8;
        high = (high << 8) + 0xFFu;
      }

      if (bit) {
        update1(st.C0[ctx], 2);
        update1(row1[ctx], 4);
        update1(c2_row[j], 6);
        update1(c2_row[j + 1], 6);
        ctx = (ctx << 1) + 1;
      } else {
        update0(st.C0[ctx], 2);
        update0(row1[ctx], 4);
        update0(c2_row[j], 6);
        update0(c2_row[j + 1], 6);
        ctx = ctx << 1;
      }

      c <<= 1;
    }

    c2 = c1;
    c1 = static_cast<std::uint32_t>(ctx & 255);
  }

  // Flush remaining range coder bits
  for (int k = 0; k < 4; ++k) {
    if (out_p >= out_limit) return false;
    *out_p++ = static_cast<std::uint8_t>(low >> 24);
    low <<= 8;
  }

  out.resize(static_cast<std::size_t>(out_p - out.data()));
  return true;
}

Status bwt_cm_decode(const std::uint8_t* src, std::size_t src_len,
                     std::uint8_t* dst, std::size_t orig_len) noexcept {
  if (orig_len == 0) return Status::Ok;
  if (src_len < 4) return Status::CorruptInput;

  thread_local CmState st;
  st.reset();

  const std::uint8_t* in_p = src;
  const std::uint8_t* const in_end = src + src_len;

  std::uint32_t high = 0xFFFFFFFFu;
  std::uint32_t low = 0;
  std::uint32_t c1 = 0;
  std::uint32_t c2 = 0;
  std::uint32_t run = 0;
  std::uint32_t code = 0;

  for (int k = 0; k < 4; ++k) {
    code = (code << 8) | (in_p < in_end ? *in_p++ : 0);
  }

  // Hot loop notes (bit-exact with the encoder above):
  //  - Every probability the update needs is already in a register, and the
  //    two C2 counters travel as one u32, so a bit costs 4 loads + 3 stores
  //    (the straightforward read-modify-write form reloaded the C2 pair after
  //    the C0/C1 stores - the compiler cannot prove the u16 tables apart).
  //  - The bit decision must stay a real branch. The decoder is one serial
  //    chain (ctx -> three counter loads -> p -> SSE load -> mid -> compare),
  //    and the branch lets the core speculate the next bit's loads while
  //    this one resolves. A branchless select makes ctx data-dependent on
  //    `mid` and was measured 3x slower (14 vs 40 MB/s on text); the empty
  //    asm statements stop the compiler from if-converting the symmetric
  //    arms into selects.
  for (std::size_t i = 0; i < orig_len; ++i) {
    run = (c1 == c2) ? run + 1 : 0;
    const std::uint32_t f = (run > 2) ? 1u : 0u;
    std::uint32_t ctx = 1;
    std::uint16_t* const row1 = st.C1[c1];
    const std::uint16_t* const row2 = st.C1[c2];
    // Row 2*ctx+f of C2 is C2[f] + ctx * (2 * 17) u16s.
    std::uint16_t* const c2_base = st.C2[f];

    do {
      const std::uint32_t p0 = st.C0[ctx];
      const std::uint32_t p1 = row1[ctx];
      const std::uint32_t p2 = row2[ctx];
      const std::uint32_t p = ((p0 + p1) * 7 + p2 + p2) >> 4;

      const std::uint32_t j = p >> 12;
      std::uint16_t* const c2p = c2_base + ctx * 34 + j;
      const std::uint32_t xx = load_pair(c2p);
      const int x1 = static_cast<int>(xx & 0xFFFFu);
      const int x2 = static_cast<int>(xx >> 16);
      const std::uint32_t ssep = static_cast<std::uint32_t>(
          x1 + (((x2 - x1) * static_cast<int>(p & 4095)) >> 12));

      const std::uint32_t mid =
          low + static_cast<std::uint32_t>(
                    (static_cast<std::uint64_t>(high - low) *
                     static_cast<std::uint64_t>(ssep * 3 + p)) >>
                    18);

      if (code <= mid) {
        SPECTRA8_CM_KEEP_BRANCH("1");
        high = mid;
        st.C0[ctx] = static_cast<std::uint16_t>(p0 + ((p0 ^ 0xFFFFu) >> 2));
        row1[ctx] = static_cast<std::uint16_t>(p1 + ((p1 ^ 0xFFFFu) >> 4));
        store_pair(c2p, pair_update1(xx));
        ctx = (ctx << 1) + 1;
      } else {
        SPECTRA8_CM_KEEP_BRANCH("0");
        low = mid + 1;
        st.C0[ctx] = static_cast<std::uint16_t>(p0 - (p0 >> 2));
        row1[ctx] = static_cast<std::uint16_t>(p1 - (p1 >> 4));
        store_pair(c2p, pair_update0(xx));
        ctx = ctx << 1;
      }

      while ((low ^ high) < (1u << 24)) {
        low <<= 8;
        high = (high << 8) + 0xFFu;
        code = (code << 8) | (in_p < in_end ? *in_p++ : 0);
      }
    } while (ctx < 256);

    c2 = c1;
    c1 = ctx & 255;
    dst[i] = static_cast<std::uint8_t>(c1);
  }

  return Status::Ok;
}

}  // namespace spectra8::detail
