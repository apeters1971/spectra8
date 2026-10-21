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

#include "prism/bwt.hpp"

#include <cstdlib>
#include <cstring>
#include <vector>

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
#endif

#include "prism/bwt_decode.hpp"
#include "libsais.h"

namespace spectra8::detail {

static inline void prefetch_read(const void* ptr) noexcept {
#if defined(__GNUC__) || defined(__clang__)
  __builtin_prefetch(ptr, /*rw=*/0, /*locality=*/0);
#elif defined(_MSC_VER) && (defined(__x86_64__) || defined(_M_X64))
  _mm_prefetch(reinterpret_cast<const char*>(ptr), _MM_HINT_NTA);
#else
  (void)ptr;
#endif
}

int bwt_encode_chains() noexcept {
  static int cached = 0;
  if (cached > 0) return cached;
  if (const char* e = std::getenv("SPECTRA8_BWT_ENCODE_CHAINS")) {
    const int n = std::atoi(e);
    if (n == 4 || n == 8 || n == 12 || n == 16) {
      cached = n;
      return cached;
    }
  }
  cached = kBwtEncodeChains;
  return cached;
}

usize bwt_forward(const u8* src, usize n, u8* dst, usize* anchors,
                  int num_chains) {
  // n >= 1 guaranteed by the caller. Conceptually this is the sentinel BWT:
  // append a unique smallest sentinel to src, sort all n+1 rotations, store
  // the last column with the sentinel row dropped and its index reported as
  // `primary`. The suffix array of src alone is enough to derive it, because
  // appending a smallest-unique sentinel does not change the relative order
  // of src's suffixes:
  //   - matrix row 0 is the sentinel suffix itself; its last column is the
  //     final byte src[n-1],
  //   - matrix row i >= 1 corresponds to suffix sa[i-1]; its last column is
  //     src[sa[i-1] - 1], or the (dropped) sentinel when sa[i-1] == 0, which
  //     makes that row `primary`.
  // The suffix array comes from libsais (vendored, third_party/libsais),
  // which is substantially faster than a textbook SA-IS. A thread-local
  // libsais context reuses its scratch across blocks (encode frames many
  // 1–2 MiB blocks per thread).
  thread_local std::vector<i32> sa;
  thread_local void* sais_ctx = nullptr;
  if (!sais_ctx) sais_ctx = libsais_create_ctx();
  // libsais can use a few KiB of free space at the end of SA for scratch;
  // 6 KiB matches the library's own "optimal" recommendation on the int path
  // and is harmless (still correct) for the byte path when left unused.
  constexpr i32 kSaisFs = 6 * 1024;
  sa.resize(n + static_cast<usize>(kSaisFs));
  // Block sizes are far below INT32_MAX and pointers are non-null, so libsais
  // cannot fail on this input.
  if (sais_ctx)
    (void)libsais_ctx(sais_ctx, src, sa.data(), static_cast<i32>(n), kSaisFs,
                      nullptr);
  else
    (void)libsais(src, sa.data(), static_cast<i32>(n), kSaisFs, nullptr);

  // Anchors for parallel inverse: the walk that emits output position p is at
  // row rank[(p+1) mod m] (rank = inverse of sa over the sentinel matrix).
  // For num_chains equal segments we need the row emitting each segment's top
  // position, i.e. rank at seg, 2*seg, ..., (K-1)*seg and at n. Rank n is the
  // sentinel suffix = row 0; the others are captured in the pass below.
  const usize seg = num_chains > 1
                        ? (n + static_cast<usize>(num_chains) - 1) /
                              static_cast<usize>(num_chains)
                        : 0;
  if (anchors && num_chains > 1) anchors[num_chains - 1] = 0;

  dst[0] = src[n - 1];  // row 0: sentinel suffix
  usize primary = 0;

  // Pass 1: Sequential scan of SA to find primary row and record chain anchors.
  // Sequential access is cache-friendly and very fast (~60 ms on 100M entries).
  if (anchors && num_chains > 1) {
    int anchors_remaining = num_chains - 1;
    const u64 inv_seg = ((u64{1} << 36) + seg - 1) / seg;
    for (usize i = 0; i < n; ++i) {
      const usize j = static_cast<usize>(sa[i]);
      if (j == 0) {
        primary = i + 1;
      } else if (anchors_remaining > 0) {
        const usize q = static_cast<usize>((static_cast<u64>(j) * inv_seg) >> 36);
        if (q * seg == j) {
          const usize c = q - 1;
          if (c + 1 < static_cast<usize>(num_chains)) {
            if (anchors[c] == 0) --anchors_remaining;
            anchors[c] = i + 1;
          }
        }
      }
    }
  } else {
    for (usize i = 0; i < n; ++i) {
      if (sa[i] == 0) {
        primary = i + 1;
        break;
      }
    }
  }

  // Pass 2: Pure gather loop split around the primary row (sa[primary - 1] == 0).
  // No division/multiplication, anchor tracking, or zero-checks inside the hot loop.
  // Pipelined with distance-64 software prefetching across cache lines.
  constexpr usize kPrefetchDist = 64;
  const usize p_idx = primary - 1;

  auto gather_range = [&](usize start, usize end, usize& k) {
    usize i = start;
    for (; i + 8 <= end; i += 8) {
      if (i + kPrefetchDist + 8 <= n) {
        for (usize m = 0; m < 8; ++m) {
          const i32 peek = sa[i + kPrefetchDist + m];
          if (peek > 0) prefetch_read(src + (peek - 1));
        }
      }
      for (usize m = 0; m < 8; ++m) {
        dst[k++] = src[sa[i + m] - 1];
      }
    }
    for (; i < end; ++i) {
      dst[k++] = src[sa[i] - 1];
    }
  };

  usize k = 1;
  gather_range(0, p_idx, k);
  gather_range(p_idx + 1, n, k);
  return primary;
}

usize bwt_forward_rows(const u8* src, usize n, u8* dst, const usize* positions,
                       int count, usize* rows) {
  thread_local std::vector<i32> sa;
  thread_local void* sais_ctx = nullptr;
  if (!sais_ctx) sais_ctx = libsais_create_ctx();
  constexpr i32 kSaisFs = 6 * 1024;
  sa.resize(n + static_cast<usize>(kSaisFs));
  if (sais_ctx)
    (void)libsais_ctx(sais_ctx, src, sa.data(), static_cast<i32>(n), kSaisFs,
                      nullptr);
  else
    (void)libsais(src, sa.data(), static_cast<i32>(n), kSaisFs, nullptr);

  // Mark the wanted suffix starts so the gather loop can test one byte per
  // row; the (few) hits then resolve which request matched.
  thread_local std::vector<u8> mark;
  mark.assign(n, 0);
  for (int c = 0; c < count; ++c) {
    if (positions[c] == n) {
      rows[c] = 0;  // the sentinel suffix is row 0
    } else {
      mark[positions[c]] = 1;
    }
  }

  dst[0] = src[n - 1];
  usize primary = 0;
  usize k = 1;
  for (usize i = 0; i < n; ++i) {
    const usize j = static_cast<usize>(sa[i]);
    if (mark[j]) {
      for (int c = 0; c < count; ++c)
        if (positions[c] == j) rows[c] = i + 1;
    }
    if (j == 0) {
      primary = i + 1;
      continue;
    }
    dst[k++] = src[j - 1];
  }
  return primary;
}

Status bwt_inverse(const u8* bwt, usize n, usize primary, u8* dst,
                   const usize* anchors, int num_chains) {
  // primary indexes a row of the length-(n+1) matrix, so it may equal n.
  if (primary > n) return Status::CorruptInput;
  // The K-chain walks write dst[(K-1)*ceil(n/K) - 1] on their first step, so
  // they need (K-1)*ceil(n/K) <= n, guaranteed by n >= (K-1)^2. The encoder
  // only emits chains on much larger blocks; a corrupt frame could claim a
  // tiny n with many chains, which must not write out of bounds. Walking
  // single-chain instead is memory-safe (the checksum rejects the output).
  if (num_chains > 1 &&
      n < static_cast<usize>(num_chains - 1) * static_cast<usize>(num_chains - 1)) {
    anchors = nullptr;
    num_chains = 1;
  }
  const usize m = n + 1;

  // Compact u16 link + u8 symbol LUT when m <= 64K (3 B/row vs 4 B fused nav):
  // smaller per-row footprint keeps the random LF walk closer to cache. Wired
  // for the single-chain and 4/8/16-chain walks (the chain counts the encoder
  // emits); other K fall back to the fused u32 nav.
  // Wide u32 link + u8 symbol LUT for blocks m > 16M (up to 256 MiB).
  constexpr usize kCompactNavMax = usize{1} << 16;
  const bool compact_m = m <= kCompactNavMax;
  const bool wide_m = m > (usize{1} << 24);
  const bool use_compactK =
      compact_m && anchors &&
      (num_chains == 16 || num_chains == 8 || num_chains == 4);
  const bool use_compact1 = compact_m && (!anchors || num_chains <= 1);

  thread_local std::vector<u32> nav;
  thread_local std::vector<u16> link16;
  thread_local std::vector<u32> link32;
  thread_local std::vector<u8> sym;
  if (use_compactK || use_compact1) {
    link16.resize(m);
    sym.resize(m);
    bwt_decode::build_nav_split(bwt, n, primary, link16.data(), sym.data());
  } else if (wide_m) {
    link32.resize(m);
    sym.resize(m);
    bwt_decode::build_nav_split(bwt, n, primary, link32.data(), sym.data());
  } else {
    nav.resize(m);
    bwt_decode::build_nav(bwt, n, primary, nav.data());
  }

  if (wide_m) {
    if (num_chains == 16 && anchors) {
      for (int c = 0; c < 16; ++c)
        if (anchors[c] >= m) return Status::CorruptInput;
      bwt_decode::walk16_split(link32.data(), sym.data(), n, anchors, dst);
      return Status::Ok;
    }
    if (num_chains == 8 && anchors) {
      for (int c = 0; c < 8; ++c)
        if (anchors[c] >= m) return Status::CorruptInput;
      bwt_decode::walk8_split(link32.data(), sym.data(), n, anchors, dst);
      return Status::Ok;
    }
    if (num_chains == 4 && anchors) {
      for (int c = 0; c < 4; ++c)
        if (anchors[c] >= m) return Status::CorruptInput;
      bwt_decode::walk4_split(link32.data(), sym.data(), n, anchors, dst);
      return Status::Ok;
    }
    if (num_chains > 1 && anchors) {
      for (int c = 0; c < num_chains; ++c)
        if (anchors[c] >= m) return Status::CorruptInput;
      bwt_decode::walk_n_split(link32.data(), sym.data(), n, anchors, num_chains, dst);
      return Status::Ok;
    }
    bwt_decode::walk1_split(link32.data(), sym.data(), primary, n, dst);
    return Status::Ok;
  }

  if (m <= (usize{1} << 24)) {
    if (num_chains == 16 && anchors) {
      for (int c = 0; c < 16; ++c)
        if (anchors[c] >= m) return Status::CorruptInput;
      if (use_compactK)
        bwt_decode::walk16_compact(link16.data(), sym.data(), n, anchors, dst);
      else
        bwt_decode::walk16(nav.data(), n, anchors, dst);
      return Status::Ok;
    }
    if (num_chains == 8 && anchors) {
      for (int c = 0; c < 8; ++c)
        if (anchors[c] >= m) return Status::CorruptInput;
      if (use_compactK)
        bwt_decode::walk8_compact(link16.data(), sym.data(), n, anchors, dst);
      else
        bwt_decode::walk8(nav.data(), n, anchors, dst);
      return Status::Ok;
    }
    if (num_chains == 4 && anchors) {
      for (int c = 0; c < 4; ++c)
        if (anchors[c] >= m) return Status::CorruptInput;
      if (use_compactK)
        bwt_decode::walk4_compact(link16.data(), sym.data(), n, anchors, dst);
      else
        bwt_decode::walk4(nav.data(), n, anchors, dst);
      return Status::Ok;
    }
    if (num_chains > 1 && anchors) {
      for (int c = 0; c < num_chains; ++c)
        if (anchors[c] >= m) return Status::CorruptInput;
      bwt_decode::walk_n(nav.data(), n, anchors, num_chains, dst);
      return Status::Ok;
    }
  }
  if (use_compact1)
    bwt_decode::walk1_compact(link16.data(), sym.data(), primary, n, dst);
  else
    bwt_decode::walk1(nav.data(), primary, n, dst);
  return Status::Ok;
}

void mtf_forward(u8* buf, usize n) {
  // Rank LUT (pos[symbol]): O(1) find + O(rank) rotate. After BWT, ranks are
  // heavily skewed to 0/1/2 on text, so those cases are specialized; the old
  // linear scan paid O(alphabet) even for rank 0.
  u8 order[256];
  u8 pos[256];
  for (int i = 0; i < 256; ++i) {
    order[i] = static_cast<u8>(i);
    pos[i] = static_cast<u8>(i);
  }
  for (usize i = 0; i < n; ++i) {
    const u8 c = buf[i];
    const u8 j = pos[c];
    buf[i] = j;
    if (j == 0) continue;
    if (j == 1) {
      const u8 f = order[0];
      order[0] = c;
      order[1] = f;
      pos[c] = 0;
      pos[f] = 1;
      continue;
    }
    if (j == 2) {
      const u8 f0 = order[0], f1 = order[1];
      order[0] = c;
      order[1] = f0;
      order[2] = f1;
      pos[c] = 0;
      pos[f0] = 1;
      pos[f1] = 2;
      continue;
    }
    if (j < 16) {
      // 16-byte shift: one wide load/store instead of a 15-iter byte loop.
      u8 tmp[16];
      std::memcpy(tmp, order, 16);
      std::memcpy(order + 1, tmp, j);
      for (int k = 1; k <= static_cast<int>(j); ++k)
        pos[order[k]] = static_cast<u8>(k);
    } else {
      // int indices: u8 k<=255 with ++k wraps and infinite-loops at j==255.
      std::memmove(order + 1, order, j);
      for (int k = 1; k <= static_cast<int>(j); ++k)
        pos[order[k]] = static_cast<u8>(k);
    }
    order[0] = c;
    pos[c] = 0;
  }
}

void mtf_inverse(u8* buf, usize n) { bwt_decode::mtf_inverse(buf, n); }

}  // namespace spectra8::detail
