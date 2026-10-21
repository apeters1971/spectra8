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

// BWT inverse decode helpers: nav-table construction and parallel LF walks.
#ifndef SPECTRA8_SRC_BWT_DECODE_HPP
#define SPECTRA8_SRC_BWT_DECODE_HPP

#include <cstring>
#include <utility>

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
#endif
#if defined(__SSSE3__)
#include <tmmintrin.h>
#endif

#include "util/common.hpp"
#include "util/simd.hpp"

namespace spectra8::detail {
namespace bwt_decode {

// Hint the next LF nav/link load. Enabled on x86_64 only: the random LF walk
// is DRAM-latency bound once the nav table leaves L2, and issuing the next
// hop while scattered dst stores retire keeps more misses in flight with the
// multi-chain MLP. On Apple Silicon / other ARM the HW prefetcher + larger
// effective MLP make software prefetch a net loss (measured). NTA / locality
// 0 avoids blowing out L1 with one-shot lines.
inline void lf_prefetch(const void* addr) noexcept {
#if defined(__x86_64__) || defined(_M_X64)
#if defined(__GNUC__) || defined(__clang__)
  __builtin_prefetch(addr, /*rw=*/0, /*locality=*/0);
#elif defined(_MSC_VER)
  _mm_prefetch(static_cast<const char*>(addr), _MM_HINT_NTA);
#endif
#else
  (void)addr;
#endif
}

// Compile-time unrolled loop over c in [0, K): f(std::integral_constant<int, c>).
// Straight-line code lets the compiler keep per-chain state in registers
// (an indexed runtime loop over a usize p[K] array spills it to the stack).
template <int K, class F, usize... I>
inline void unroll_impl(F&& f, std::index_sequence<I...>) {
  (f(std::integral_constant<int, static_cast<int>(I)>{}), ...);
}
template <int K, class F>
inline void unroll(F&& f) {
  unroll_impl<K>(f, std::make_index_sequence<static_cast<usize>(K)>{});
}

// Four partial histograms avoid the read-after-write stalls that dominate a
// single-array byte histogram on x86 when the same bucket is hit back-to-back
// (common in BWT L-column runs). Merge cost is 256 adds and is in the noise.
inline void histogram(const u8* bwt, usize n, u32 count[256]) {
  hist_u8(bwt, n, count);
}

inline void prefix_c(const u32 count[256], u32 C[256]) {
  u32 sum = 1;  // sentinel at F row 0
  for (int c = 0; c < 256; ++c) {
    C[c] = sum;
    sum += count[c];
  }
}

// LF table construction as kBuildChunks interleaved counting-sort streams.
//
// The textbook build is one pass of next[c]++ over the L column. BWT output
// is full of equal-byte runs, so that is a single store-to-load-forwarding
// chain (~5 cycles per byte regardless of how wide the core is). Splitting
// the column into K chunks, each with its own running counters seeded from
// the per-chunk histograms, gives K independent chains the core can overlap;
// the chunk histograms replace the whole-column histogram at the same cost.
// Measured on Apple M5 with the byte histogram included: 0.64 -> 1.1-1.2
// GB/s on text, 1.05 -> 1.25-1.4 GB/s on binary.
//
// Byte index j of the L column maps to matrix row j (j < primary) or j + 1
// (j >= primary); row `primary` is the dropped sentinel and links to row 0.
// `emit(row, rank, sym)` stores one row; `count` receives the whole-column
// histogram (needed by callers that also build other tables).
inline constexpr int kBuildChunks = 8;

template <class Emit>
inline void build_lf_chunked(const u8* bwt, usize n, usize primary,
                             u32 count[256], Emit&& emit) {
  constexpr int K = kBuildChunks;
  const usize seg = (n + K - 1) / K;
  // (K-1)*seg <= n needs n >= (K-1)^2; tiny blocks take the plain single
  // stream (their cost is nil anyway).
  if (n < static_cast<usize>((K - 1) * (K - 1))) {
    hist_u8(bwt, n, count);
    u32 next[256];
    prefix_c(count, next);
    for (usize j = 0; j < n; ++j) {
      const u8 c = bwt[j];
      emit(j + (j >= primary ? 1 : 0), next[c]++, c);
    }
    return;
  }
  alignas(64) u32 hist[K][256];
  for (int k = 0; k < K; ++k) {
    const usize lo = static_cast<usize>(k) * seg;
    const usize hi = lo + seg < n ? lo + seg : n;
    if (lo < hi)
      hist_u8(bwt + lo, hi - lo, hist[k]);
    else
      std::memset(hist[k], 0, sizeof(hist[k]));
  }
  alignas(64) u32 next[K][256];
  {
    u32 sum = 1;  // sentinel occupies F row 0
    for (int c = 0; c < 256; ++c) {
      u32 acc = sum;
      for (int k = 0; k < K; ++k) {
        next[k][c] = acc;
        acc += hist[k][c];
      }
      count[c] = acc - sum;
      sum = acc;
    }
  }
  const usize last_len = n - static_cast<usize>(K - 1) * seg;
  usize s = 0;
  for (; s < last_len; ++s) {
    unroll<K>([&](auto kc) {
      constexpr int k = decltype(kc)::value;
      const usize j = static_cast<usize>(k) * seg + s;
      const u8 c = bwt[j];
      const u32 r = next[k][c];
      next[k][c] = r + 1;
      emit(j + (j >= primary ? 1 : 0), r, c);
    });
  }
  for (; s < seg; ++s) {
    unroll<K - 1>([&](auto kc) {
      constexpr int k = decltype(kc)::value;
      const usize j = static_cast<usize>(k) * seg + s;
      const u8 c = bwt[j];
      const u32 r = next[k][c];
      next[k][c] = r + 1;
      emit(j + (j >= primary ? 1 : 0), r, c);
    });
  }
}

// Fused LF link (low 24 bits) + symbol (high 8 bits) for single-load walks.
inline void build_nav(const u8* bwt, usize n, usize primary, u32* nav) {
  u32 count[256];
  build_lf_chunked(bwt, n, primary, count, [&](usize row, u32 r, u8 c) {
    nav[row] = r | (static_cast<u32>(c) << 24);
  });
  nav[primary] = 0;  // sentinel row -> F row 0
}

// Compact LF tables for blocks with m = n+1 <= 65536: u16 link + u8 symbol per
// row (3 bytes vs 4 in fused nav), better cache footprint on small/medium blocks.
// Wide LF tables for blocks with m = n+1 > 16777216: u32 link + u8 symbol per
// row (5 bytes vs 4 in fused nav), supporting up to 256 MiB blocks.
template <typename LinkT>
inline void build_nav_split(const u8* bwt, usize n, usize primary, LinkT* link,
                            u8* sym) {
  u32 count[256];
  build_lf_chunked(bwt, n, primary, count, [&](usize row, u32 r, u8 c) {
    link[row] = static_cast<LinkT>(r);
    sym[row] = c;
  });
  link[primary] = 0;
  sym[primary] = 0;
}

inline void build_nav_compact(const u8* bwt, usize n, usize primary, u16* link,
                              u8* sym) {
  build_nav_split(bwt, n, primary, link, sym);
}

// Shuffle masks: pull lane j to lane 0, shift lanes 0..j-1 up by one.
struct MtfShufMasks {
  alignas(16) u8 m[16][16];
  constexpr MtfShufMasks() : m{} {
    for (int j = 0; j < 16; ++j) {
      m[j][0] = static_cast<u8>(j);
      for (int k = 1; k <= j; ++k) m[j][k] = static_cast<u8>(k - 1);
      for (int k = j + 1; k < 16; ++k) m[j][k] = static_cast<u8>(k);
    }
  }
};
inline const MtfShufMasks kMtfShuf;

// Pull ord[j] to the front of the rank->symbol list, shifting ord[0..j-1] up
// by one. Plain-memory variant (the MtfList below is the fast path).
inline u8 mtf_pull(u8* ord, u32 j) noexcept {
  if (j == 0) return ord[0];
  if (j == 1) {
    const u8 c = ord[1];
    ord[1] = ord[0];
    ord[0] = c;
    return c;
  }
  const u8 c = ord[j];
  if (j < 16) {
#if defined(__SSSE3__)
    __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ord));
    v = _mm_shuffle_epi8(v, _mm_load_si128(reinterpret_cast<const __m128i*>(kMtfShuf.m[j])));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(ord), v);
    return static_cast<u8>(_mm_cvtsi128_si32(v));
#elif SPECTRA8_HAVE_NEON
    uint8x16_t v = vld1q_u8(ord);
    v = vqtbl1q_u8(v, vld1q_u8(kMtfShuf.m[j]));
    vst1q_u8(ord, v);
    return static_cast<u8>(vgetq_lane_u8(v, 0));
#else
    for (u32 k = j; k > 0; --k) ord[k] = ord[k - 1];
    ord[0] = c;
#endif
  } else {
    std::memmove(ord + 1, ord, j);
    ord[0] = c;
  }
  return c;
}

// Inverse-MTF rank -> symbol list. The inverse is a serial chain through the
// list state (every pull depends on the previous one), so its latency is the
// whole cost. With the list in memory each small-rank pull is load -> shuffle
// -> store -> (next) load, a store-to-load round trip per symbol. Here the
// first 16 entries live in a vector register: a pull of rank < 16 (the vast
// majority on BWT output) is a single table shuffle on that register, and
// only ranks >= 16 touch memory (spill the head, shift ord[0..j) up by one in
// 16-byte chunks, reload). Measured on the fused mode-2 inverse: +40% on
// text, +30% on binary vs the in-memory list. On targets without SSSE3/NEON
// the list is kept in memory.
struct MtfList {
#if defined(__SSSE3__)
  __m128i head;
#elif SPECTRA8_HAVE_NEON
  uint8x16_t head;
#endif
  // ord[0..16) is only current after a spill; 16 bytes of slack let the shift
  // of rank 255 run its last 16-byte chunk past index 255.
  alignas(16) u8 ord[256 + 16];

  void init() noexcept {
    for (int i = 0; i < 256 + 16; ++i) ord[i] = static_cast<u8>(i);
#if defined(__SSSE3__)
    head = _mm_load_si128(reinterpret_cast<const __m128i*>(ord));
#elif SPECTRA8_HAVE_NEON
    head = vld1q_u8(ord);
#endif
  }

  // Current front symbol (rank 0).
  u8 front() const noexcept {
#if defined(__SSSE3__)
    return static_cast<u8>(_mm_cvtsi128_si32(head));
#elif SPECTRA8_HAVE_NEON
    return static_cast<u8>(vgetq_lane_u8(head, 0));
#else
    return ord[0];
#endif
  }

  u8 pull(u32 j) noexcept {
#if defined(__SSSE3__) || SPECTRA8_HAVE_NEON
    if (j < 16) {
#if defined(__SSSE3__)
      head = _mm_shuffle_epi8(
          head, _mm_load_si128(reinterpret_cast<const __m128i*>(kMtfShuf.m[j])));
      return static_cast<u8>(_mm_cvtsi128_si32(head));
#else
      head = vqtbl1q_u8(head, vld1q_u8(kMtfShuf.m[j]));
      return static_cast<u8>(vgetq_lane_u8(head, 0));
#endif
    }
    return pull_far(j);
#else
    return mtf_pull(ord, j);
#endif
  }

 private:
#if defined(__SSSE3__) || SPECTRA8_HAVE_NEON
  __attribute__((noinline)) u8 pull_far(u32 j) noexcept {
#if defined(__SSSE3__)
    _mm_store_si128(reinterpret_cast<__m128i*>(ord), head);
    const u8 c = ord[j];
    // Shift ord[0..j) up by one: 16-byte windows from the top down (each
    // window is read before any store overlaps it), then the saved first
    // window at offset 1 covers the remainder below the last full window.
    const __m128i first = _mm_load_si128(reinterpret_cast<const __m128i*>(ord));
    for (int k = static_cast<int>(j) - 16; k >= 0; k -= 16)
      _mm_storeu_si128(reinterpret_cast<__m128i*>(ord + k + 1),
                       _mm_loadu_si128(reinterpret_cast<const __m128i*>(ord + k)));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(ord + 1), first);
    ord[0] = c;
    head = _mm_load_si128(reinterpret_cast<const __m128i*>(ord));
#else
    vst1q_u8(ord, head);
    const u8 c = ord[j];
    const uint8x16_t first = vld1q_u8(ord);
    for (int k = static_cast<int>(j) - 16; k >= 0; k -= 16)
      vst1q_u8(ord + k + 1, vld1q_u8(ord + k));
    vst1q_u8(ord + 1, first);
    ord[0] = c;
    head = vld1q_u8(ord);
#endif
    return c;
  }
#endif
};

// Inverse MTF over a 256-byte symbol list (rank -> symbol). The inverse only
// needs the rank->symbol direction, so no symbol->position LUT is maintained.
// Index 0 is the common case on BWT output (front unchanged); runs of zeros are
// expanded with memset.
inline void mtf_inverse(u8* buf, usize n) {
  MtfList list;
  list.init();
  u8 front = 0;
  usize i = 0;
  while (i < n) {
    const u8 j = buf[i];
    if (j == 0) {
      const usize start = i;
      while (i < n && buf[i] == 0) ++i;
      const usize run = i - start;
      if (run >= 16)
        std::memset(buf + start, front, run);
      else
        for (usize k = start; k < i; ++k) buf[k] = front;
      continue;
    }
    front = list.pull(j);
    buf[i++] = front;
  }
}

inline usize seg_end(usize n, usize seg, int k) {
  const usize lim = n - static_cast<usize>(k) * seg;
  return lim < seg ? lim : seg;
}

// (A bigram LF^2 walk - two output bytes per random access - was prototyped
// here and measured slower at every block size: materializing LF^2 costs one
// random gather per row, so total random accesses rise from m to 1.5m, and
// the doubled entry size pushes the table out of L2 exactly where the walk
// is latency-bound. The unigram nav plus multi-chain parallelism stays.)

// LF table adapters for the walks: one hop reads the symbol of row p and
// advances p to LF(p).
struct NavFused {
  const u32* __restrict nav;
  inline void step(usize& p, u8& out) const noexcept {
    const u32 e = nav[p];
    const usize np = e & 0xFFFFFFu;
    lf_prefetch(nav + np);
    out = static_cast<u8>(e >> 24);
    p = np;
  }
};

template <typename LinkT>
struct NavSplit {
  const LinkT* __restrict link;
  const u8* __restrict sym;
  inline void step(usize& p, u8& out) const noexcept {
    const usize np = link[p];
    lf_prefetch(link + np);
    lf_prefetch(sym + np);
    out = sym[p];
    p = np;
  }
};

template <class Tab>
inline void walk1_tab(const Tab& t, usize primary, usize n, u8* __restrict dst) {
  usize p = primary;
  u8 s;
  // Row `primary` is the sentinel: its symbol is not part of the output.
  t.step(p, s);
  for (usize j = n; j-- > 0;) t.step(p, dst[j]);
}

// K-chain parallel inverse: chain c starts at anchors[c] and emits output
// segment c ([c*seg, (c+1)*seg), the last one shorter) backwards, so K
// random LF loads are in flight at once (memory-level parallelism).
//
// Each chain writes through a 64-byte staging line and flushes full lines
// to dst with one wide copy. Writing the K streams straight into dst is a
// trap: the segment stride is n/K, a multiple of the L1 set stride for the
// power-of-two block sizes the encoder uses, so all K write positions share
// one L1 set and above the associativity every store is a conflict miss
// (16 equal chains measured 5x slower than 8 at 256 KiB..4 MiB blocks, and
// 5x slower than the same walk on a non-power-of-two n). The staging lines
// sit in consecutive sets and the flushes are line-sized bursts. Blocks
// whose LF table exceeds L2 are latency-bound instead and unaffected.
inline constexpr usize kWalkStage = 64;

template <int K, class Tab>
inline void walk_k_tab(const Tab& t, usize n, const usize* anchors,
                       u8* __restrict dst) {
  const usize seg = (n + static_cast<usize>(K) - 1) / static_cast<usize>(K);
  usize p[K];
  usize b[K];
  for (int c = 0; c < K; ++c) {
    p[c] = anchors[c];
    b[c] = static_cast<usize>(c + 1) * seg - 1;
  }
  b[K - 1] = n - 1;
  alignas(64) u8 stage[K][kWalkStage];
  const usize main_end = n - static_cast<usize>(K - 1) * seg;
  usize o = 0;
  for (; o + kWalkStage <= main_end; o += kWalkStage) {
    for (usize k = 0; k < kWalkStage; ++k) {
      const usize slot = kWalkStage - 1 - k;
      unroll<K>([&](auto cc) {
        constexpr int c = decltype(cc)::value;
        t.step(p[c], stage[c][slot]);
      });
    }
    // stage[c][63 - k] holds output position b[c] - (o + k).
    for (int c = 0; c < K; ++c)
      std::memcpy(dst + b[c] - o - (kWalkStage - 1), stage[c], kWalkStage);
  }
  for (; o < main_end; ++o) {
    unroll<K>([&](auto cc) {
      constexpr int c = decltype(cc)::value;
      t.step(p[c], dst[b[c] - o]);
    });
  }
  // Peel: segments 0..K-2 are `seg` long, segment K-1 (the last, shortest)
  // has already finished; one more chain drops out at each boundary.
  for (int active = K - 1; active > 0; --active) {
    const usize end = seg_end(n, seg, active - 1);
    for (; o < end; ++o)
      for (int c = 0; c < active; ++c) t.step(p[c], dst[b[c] - o]);
  }
}

// Runtime chain count (any K in [2, 16]); per-chain state lives in memory,
// which is fine for the latency-bound walk this serves (non-default counts).
template <class Tab>
inline void walk_n_tab(const Tab& t, usize n, const usize* anchors, int chains,
                       u8* __restrict dst) {
  const usize seg =
      (n + static_cast<usize>(chains) - 1) / static_cast<usize>(chains);
  usize p[16];
  usize b[16];
  for (int c = 0; c < chains; ++c) {
    p[c] = anchors[c];
    b[c] = static_cast<usize>(c + 1) * seg - 1;
  }
  b[chains - 1] = n - 1;
  alignas(64) u8 stage[16][kWalkStage];
  const usize main_end = n - static_cast<usize>(chains - 1) * seg;
  usize o = 0;
  for (; o + kWalkStage <= main_end; o += kWalkStage) {
    for (usize k = 0; k < kWalkStage; ++k) {
      const usize slot = kWalkStage - 1 - k;
      for (int c = 0; c < chains; ++c) t.step(p[c], stage[c][slot]);
    }
    for (int c = 0; c < chains; ++c)
      std::memcpy(dst + b[c] - o - (kWalkStage - 1), stage[c], kWalkStage);
  }
  for (; o < main_end; ++o)
    for (int c = 0; c < chains; ++c) t.step(p[c], dst[b[c] - o]);
  for (int active = chains - 1; active > 0; --active) {
    const usize end = seg_end(n, seg, active - 1);
    for (; o < end; ++o)
      for (int c = 0; c < active; ++c) t.step(p[c], dst[b[c] - o]);
  }
}

// --- Split (link + symbol) table walks ---------------------------------------

template <typename LinkT>
inline void walk1_split(const LinkT* __restrict link, const u8* __restrict sym,
                        usize primary, usize n, u8* __restrict dst) {
  walk1_tab(NavSplit<LinkT>{link, sym}, primary, n, dst);
}
template <typename LinkT>
inline void walk16_split(const LinkT* __restrict link, const u8* __restrict sym,
                         usize n, const usize* anchors, u8* __restrict dst) {
  walk_k_tab<16>(NavSplit<LinkT>{link, sym}, n, anchors, dst);
}
template <typename LinkT>
inline void walk8_split(const LinkT* __restrict link, const u8* __restrict sym,
                        usize n, const usize* anchors, u8* __restrict dst) {
  walk_k_tab<8>(NavSplit<LinkT>{link, sym}, n, anchors, dst);
}
template <typename LinkT>
inline void walk4_split(const LinkT* __restrict link, const u8* __restrict sym,
                        usize n, const usize* anchors, u8* __restrict dst) {
  walk_k_tab<4>(NavSplit<LinkT>{link, sym}, n, anchors, dst);
}
template <typename LinkT>
inline void walk_n_split(const LinkT* __restrict link, const u8* __restrict sym,
                         usize n, const usize* anchors, int chains,
                         u8* __restrict dst) {
  walk_n_tab(NavSplit<LinkT>{link, sym}, n, anchors, chains, dst);
}

inline void walk1_compact(const u16* __restrict link, const u8* __restrict sym,
                          usize primary, usize n, u8* __restrict dst) {
  walk1_split(link, sym, primary, n, dst);
}
inline void walk16_compact(const u16* __restrict link, const u8* __restrict sym,
                           usize n, const usize* anchors, u8* __restrict dst) {
  walk16_split(link, sym, n, anchors, dst);
}
inline void walk8_compact(const u16* __restrict link, const u8* __restrict sym,
                          usize n, const usize* anchors, u8* __restrict dst) {
  walk8_split(link, sym, n, anchors, dst);
}
inline void walk4_compact(const u16* __restrict link, const u8* __restrict sym,
                          usize n, const usize* anchors, u8* __restrict dst) {
  walk4_split(link, sym, n, anchors, dst);
}

// --- Fused nav walks ---------------------------------------------------------

inline void walk1(const u32* __restrict nav, usize primary, usize n,
                  u8* __restrict dst) {
  walk1_tab(NavFused{nav}, primary, n, dst);
}
inline void walk4(const u32* __restrict nav, usize n, const usize* anchors,
                  u8* __restrict dst) {
  walk_k_tab<4>(NavFused{nav}, n, anchors, dst);
}
inline void walk8(const u32* __restrict nav, usize n, const usize* anchors,
                  u8* __restrict dst) {
  walk_k_tab<8>(NavFused{nav}, n, anchors, dst);
}
inline void walk16(const u32* __restrict nav, usize n, const usize* anchors,
                   u8* __restrict dst) {
  walk_k_tab<16>(NavFused{nav}, n, anchors, dst);
}
// Generic K-chain walk (used for 12 and any non-specialized K in [2, 15]).
inline void walk_n(const u32* __restrict nav, usize n, const usize* anchors,
                   int chains, u8* __restrict dst) {
  walk_n_tab(NavFused{nav}, n, anchors, chains, dst);
}

}  // namespace bwt_decode
}  // namespace spectra8::detail

#endif  // SPECTRA8_SRC_BWT_DECODE_HPP
