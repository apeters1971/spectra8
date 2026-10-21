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

#include "prism/bwt_ultra.hpp"

#include <cstring>
#include <vector>

namespace spectra8::detail {

namespace {

struct UltraState {
  std::uint16_t C0[256];
  std::uint16_t C1[256][256];
  std::uint16_t CF[256][256];
  std::uint16_t SSE[512][17];

  void reset() noexcept {
    for (int i = 0; i < 256; ++i) C0[i] = 1 << 15;
    for (int i = 0; i < 256; ++i) {
      for (int j = 0; j < 256; ++j) {
        C1[i][j] = 1 << 15;
        CF[i][j] = 1 << 15;
      }
    }
    for (int s = 0; s < 2; ++s) {
      for (int j = 0; j < 256; ++j) {
        for (int k = 0; k < 17; ++k) {
          SSE[2 * j + s][k] = static_cast<std::uint16_t>((k << 12) - (k == 16));
        }
      }
    }
  }
};

inline void update0(std::uint16_t& p, int rate) noexcept {
  p = static_cast<std::uint16_t>(p - (p >> rate));
}

inline void update1(std::uint16_t& p, int rate) noexcept {
  p = static_cast<std::uint16_t>(p + ((p ^ 0xFFFF) >> rate));
}

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

#if defined(__GNUC__) || defined(__clang__)
#define SPECTRA8_ULTRA_KEEP_BRANCH(tag) asm volatile("# ultra bit " tag ::: "memory")
#else
#define SPECTRA8_ULTRA_KEEP_BRANCH(tag) ((void)0)
#endif

}  // namespace

bool bwt_ultra_encode(const std::uint8_t* src, std::size_t n,
                      std::vector<std::uint8_t>& out) {
  if (n == 0) return true;

  thread_local UltraState st;
  st.reset();

  // Conservative upper bound for output buffer: raw size + 64 KiB
  const std::size_t bound = n + (n >> 4) + 65536;
  const std::size_t initial_pos = out.size();
  out.resize(initial_pos + bound);

  std::uint8_t* out_p = out.data() + initial_pos;
  const std::uint8_t* const out_limit = out_p + bound - 16;

  // 1. Calculate symbol frequencies
  std::uint32_t counts[256] = {0};
  for (std::size_t i = 0; i < n; ++i) {
    ++counts[src[i]];
  }

  // 2. Write 32-byte bitmask of present symbols
  std::uint8_t present[32] = {0};
  for (int c = 0; c < 256; ++c) {
    if (counts[c] > 0) {
      present[c >> 3] |= static_cast<std::uint8_t>(1 << (c & 7));
    }
  }
  for (int k = 0; k < 32; ++k) *out_p++ = present[k];

  // 3. Write varint counts
  for (int c = 0; c < 256; ++c) {
    if (counts[c] > 0) {
      std::uint64_t v = counts[c];
      while (v >= 0x80) {
        *out_p++ = static_cast<std::uint8_t>((v & 0x7F) | 0x80);
        v >>= 7;
      }
      *out_p++ = static_cast<std::uint8_t>(v);
    }
  }

  // 4. Calculate cumulative intervals
  std::size_t base[257];
  std::size_t sum = 0;
  for (int c = 0; c < 256; ++c) {
    base[c] = sum;
    sum += counts[c];
  }
  base[256] = sum;

  std::uint32_t high = 0xFFFFFFFFu;
  std::uint32_t low = 0;
  std::uint32_t c1 = 0;
  std::uint32_t c2 = 0;
  std::uint32_t run = 0;
  std::uint32_t fc = 0;

  for (std::size_t i = 0; i < n; ++i) {
#if defined(__GNUC__) || defined(__clang__)
    if (i + 1 < n) {
      __builtin_prefetch(st.C1[src[i + 1]], 0, 0);
    }
#endif
    while (i >= base[fc + 1]) ++fc;

    if (c1 == c2)
      ++run;
    else
      run = 0;

    const int f = (run > 2) ? 1 : 0;
    int ctx = 1;
    std::uint16_t* const row1 = st.C1[c1];
    const std::uint16_t* const row2 = st.C1[c2];
    std::uint16_t* const rowF = st.CF[fc];
    std::uint8_t c = src[i];

    while (ctx < 256) {
      const int p0 = st.C0[ctx];
      const int p1 = row1[ctx];
      const int p2 = row2[ctx];
      const int pf = rowF[ctx];
      const int p = ((p0 + p1) * 7 + p2 + pf) >> 4;

      const int j = p >> 12;
      std::uint16_t* const sse_row = st.SSE[2 * ctx + f];
      const int x1 = sse_row[j];
      const int x2 = sse_row[j + 1];
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
        update1(rowF[ctx], 5);
        const std::uint32_t pair = pair_update1(load_pair(sse_row + j));
        store_pair(sse_row + j, pair);
        ctx = (ctx << 1) + 1;
      } else {
        update0(st.C0[ctx], 2);
        update0(row1[ctx], 4);
        update0(rowF[ctx], 5);
        const std::uint32_t pair = pair_update0(load_pair(sse_row + j));
        store_pair(sse_row + j, pair);
        ctx = ctx << 1;
      }

      c <<= 1;
    }

    c2 = c1;
    c1 = static_cast<std::uint32_t>(ctx & 255);
  }

  for (int k = 0; k < 4; ++k) {
    if (out_p >= out_limit) return false;
    *out_p++ = static_cast<std::uint8_t>(low >> 24);
    low <<= 8;
  }

  out.resize(static_cast<std::size_t>(out_p - out.data()));
  return true;
}

Status bwt_ultra_decode(const std::uint8_t* src, std::size_t src_len,
                        std::uint8_t* dst, std::size_t n) {
  if (n == 0) return Status::Ok;
  if (!src || src_len < 36 || !dst) return Status::CorruptInput;

  thread_local UltraState st;
  st.reset();

  const std::uint8_t* in_p = src;
  const std::uint8_t* const in_limit = src + src_len;

  // 1. Read 32-byte bitmask
  const std::uint8_t* const present = in_p;
  in_p += 32;

  // 2. Read symbol counts
  std::uint32_t counts[256] = {0};
  std::uint64_t total_count = 0;
  for (int c = 0; c < 256; ++c) {
    if (present[c >> 3] & (1 << (c & 7))) {
      std::uint64_t v = 0;
      int shift = 0;
      bool done = false;
      while (in_p < in_limit && shift <= 56) {
        std::uint8_t b = *in_p++;
        v |= static_cast<std::uint64_t>(b & 0x7F) << shift;
        shift += 7;
        if (!(b & 0x80)) {
          done = true;
          break;
        }
      }
      if (!done || v == 0) return Status::CorruptInput;
      counts[c] = static_cast<std::uint32_t>(v);
      total_count += counts[c];
    }
  }

  if (total_count != n) return Status::CorruptInput;

  // 3. Compute cumulative intervals
  std::size_t base[257];
  std::size_t sum = 0;
  for (int c = 0; c < 256; ++c) {
    base[c] = sum;
    sum += counts[c];
  }
  base[256] = sum;

  if (in_p + 4 > in_limit) return Status::CorruptInput;

  std::uint32_t code = 0;
  for (int i = 0; i < 4; ++i) {
    code = (code << 8) | *in_p++;
  }
  std::uint32_t low = 0;
  std::uint32_t high = 0xFFFFFFFFu;
  std::uint32_t c1 = 0;
  std::uint32_t c2 = 0;
  std::uint32_t run = 0;
  std::uint32_t fc = 0;

  for (std::size_t i = 0; i < n; ++i) {
    while (i >= base[fc + 1]) ++fc;

    if (c1 == c2)
      ++run;
    else
      run = 0;

    const int f = (run > 2) ? 1 : 0;
    int ctx = 1;
    std::uint16_t* const row1 = st.C1[c1];
    const std::uint16_t* const row2 = st.C1[c2];
    std::uint16_t* const rowF = st.CF[fc];

    do {
      const int p0 = st.C0[ctx];
      const int p1 = row1[ctx];
      const int p2 = row2[ctx];
      const int pf = rowF[ctx];
      const int p = ((p0 + p1) * 7 + p2 + pf) >> 4;

      const int j = p >> 12;
      std::uint16_t* const sse_row = st.SSE[2 * ctx + f];
      const int x1 = sse_row[j];
      const int x2 = sse_row[j + 1];
      const int ssep = x1 + (((x2 - x1) * (p & 4095)) >> 12);

      const std::uint32_t mid =
          low + static_cast<std::uint32_t>(
                    (static_cast<std::uint64_t>(high - low) *
                     static_cast<std::uint64_t>(ssep * 3 + p)) >>
                    18);

      const bool bit = (code <= mid);
      SPECTRA8_ULTRA_KEEP_BRANCH("dec");
      if (bit) {
        high = mid;
        update1(st.C0[ctx], 2);
        update1(row1[ctx], 4);
        update1(rowF[ctx], 5);
        const std::uint32_t pair = pair_update1(load_pair(sse_row + j));
        store_pair(sse_row + j, pair);
        ctx = (ctx << 1) + 1;
      } else {
        low = mid + 1;
        update0(st.C0[ctx], 2);
        update0(row1[ctx], 4);
        update0(rowF[ctx], 5);
        const std::uint32_t pair = pair_update0(load_pair(sse_row + j));
        store_pair(sse_row + j, pair);
        ctx = ctx << 1;
      }

      while ((low ^ high) < (1u << 24)) {
        code = (code << 8) | (in_p < in_limit ? *in_p++ : 0);
        low <<= 8;
        high = (high << 8) + 0xFFu;
      }
    } while (ctx < 256);

    const std::uint8_t sym = static_cast<std::uint8_t>(ctx & 255);
    dst[i] = sym;
    if (counts[sym] == 0) return Status::CorruptInput;
    --counts[sym];
    c2 = c1;
    c1 = static_cast<std::uint32_t>(sym);
  }

  for (int c = 0; c < 256; ++c) {
    if (counts[c] != 0) return Status::CorruptInput;
  }

  return Status::Ok;
}

}  // namespace spectra8::detail
