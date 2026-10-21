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

// Crystal text mode: BWT front end + the static lattice coder as back end.
//
// The static lattice coder (crystal_static.cpp) models a byte from its record
// neighbours with tables that are fixed for the whole block. That is the right
// tool for numeric records, and the wrong one for text, where the useful
// context is the preceding *words*, not the preceding 1-2 bytes. Rather than
// bolt an LZ77 layer on (zstd's answer, which tops out ~15-20% behind the
// BWT family on text) this mode borrows bzip's insight: a Burrows-Wheeler
// transform sorts every symbol by its (unbounded) right context, so all
// occurrences of "the same context" land next to each other and the local
// symbol distribution becomes extremely peaked. Move-to-front then turns
// "which of the few locally active bytes" into a small rank, and *ranks are
// stationary*: the distribution of MTF ranks barely changes across the block,
// which is exactly the property a static coder needs. bzip2 exploits the
// same thing with block-static Huffman tables; here the static lattice coder
// does the job with 12-bit rANS tables, its order-1/2 contexts over the
// previous ranks (an in-run vs. after-a-big-rank distinction that bzip2's
// table selectors only approximate) and its MDL group merging.
//
// Pipeline
//   encode:  BWT (libsais) -> MTF -> zero-run symbols -> static coder, stride 1
//   decode:  static coder -> fused run expansion + inverse MTF (+ histogram)
//            -> LF table -> K-chain inverse BWT walk
//
// Symbol alphabet (mode kModeRle): 0 = "run of rank 0" marker, followed by one
// length byte L (run = L + 1; L = 255 continues with another length byte);
// 1..255 = MTF rank. One byte per symbol, no escape, so the fused inverse is
// a fixed-shape step: read symbol, conditionally read the length, one 16-lane
// table shuffle to pull the rank to the front of the MTF list, one 16-byte
// splat store. Ranks >= 16 and runs > 16 take a rare slow path. The static
// coder's order-1/2 contexts see the marker as "previous symbol 0", so the
// length byte and the rank after a run get their own distributions for free.
//
// Two refinements (mode kModeSplit, both decided by the encoder from an
// order-1 estimate, both free for the decoder):
//   * MTF variant. The "sticky" list (rank >= 2 goes to slot 1, the slot-1
//     symbol takes the front only after a nonzero rank) keeps a long run's
//     symbol at the front across a single intruder; 0.2-2% smaller on text.
//   * F-class split. The decoder knows the first column of the sorted matrix
//     (the block's bytes sorted) before it decodes anything, so the class of
//     F[row] -- whitespace / digit / upper / lower / other -- is a free
//     context: rows whose right context starts a word see a different rank
//     distribution from rows inside a word. Symbols are routed into one
//     static stream per class. Classes are intervals of rows, so the decoder
//     switches streams a handful of times per block. Worth 0.5-1.5% on prose
//     at 1-8 MiB; the estimate turns it off where the extra tables lose.
//   Together ~1% at 1 MiB and ~2% at 8 MiB on enwik8, at unchanged decode
//   speed. The M03 family goes further (order-k context intervals, which the
//   decoder can also derive) but needs a recursive adaptive coder.
//
// Inverse BWT: the LF table is built as K_b = 8 interleaved counting-sort
// streams (the plain next[c]++ over BWT output, which is full of equal-byte
// runs, is one long store-to-load-forwarding chain). The walk runs K = 16
// chains into a scratch whose per-chain stride is skewed by a few cache
// lines: equal spacing puts all K write streams in the same L1 set, and
// above the associativity that costs a conflict miss per store (16 equal
// chains were 5x slower than 8).
//
// Payload body (after the (ver, kind) prefix written by crystal.cpp):
//   u8 mode        kModeRle (plain MTF, one stream) or kModeSplit
//   u8 chains      1, 8 or 16 (fixed by n; checked on decode)
//   varint primary
//   varint row[chains - 1]  LF walk start rows of output segments 0..K-2
//  kModeRle:
//   varint main_n, varint side_n, varint main_len
//   static coder body over the main_n symbols (main_len bytes)
//   side bytes: static coder body, or raw when side_n < 32
//  kModeSplit:
//   u8 variant     kMtfPlain / kMtfSticky
//   u8 nb          class intervals (1 = no split)
//   nb x (varint pos [omitted for the first], u8 class)
//   per class 0..max class:
//     varint main_n, varint side_n; if main_n > 0:
//     varint main_len, static coder body; side: varint len + static body
//     when side_n >= 32, else side_n raw bytes
#include <cmath>
#include <cstring>
#include <vector>
#ifdef CRYB_PROFILE
#include <chrono>
#include <cstdio>
#define CRYB_T(name) const auto name = std::chrono::steady_clock::now()
#else
#define CRYB_T(name)
#endif

#include "prism/bwt.hpp"
#include "prism/bwt_decode.hpp"
#include "util/checksum.hpp"
#include "util/common.hpp"
#include "crystal/crystal.hpp"
#include "util/simd.hpp"

// CRYB_FORCE_SSE_TEST (test builds only): exercise the SSE code path on arm64
// through sse2neon, so its lane logic is verified where x86 hardware is not
// at hand.
#if defined(CRYB_FORCE_SSE_TEST)
#include "sse2neon.h"
#define CRYB_SHUF 1
#define CRYB_USE_NEON 0
#elif SPECTRA8_HAVE_NEON
#define CRYB_SHUF 1
#define CRYB_USE_NEON 1
#elif defined(__SSSE3__) && defined(__SSE4_1__)
#include <smmintrin.h>
#define CRYB_SHUF 1
#define CRYB_USE_NEON 0
#else
#define CRYB_SHUF 0
#define CRYB_USE_NEON 0
#endif

namespace spectra8::detail {

namespace {

// Mode byte. kModeRle is the original layout (plain MTF, one symbol stream);
// kModeSplit adds an MTF-variant byte and splits the symbol stream by the
// class of the F-column byte (see file comment). The decoder reads both,
// the encoder emits kModeSplit.
constexpr u8 kModeRle = 2;
constexpr u8 kModeSplit = 3;

// MTF variants (kModeSplit). kMtfPlain is the textbook move-to-front.
// kMtfSticky: a symbol found at rank >= 2 moves to slot 1, not 0, and the
// slot-1 symbol moves to the front only if the previous symbol was not the
// front one (Balkenhol/Shtarkov's MTF-2 flavour). On BWT output the front
// symbol is usually a long run; a single intruder should not displace it.
// Chosen per block by the order-1 estimate; 0.2-2% smaller on text, never
// worse by more than 0.05% on the test sets.
constexpr u8 kMtfPlain = 0;
constexpr u8 kMtfSticky = 1;

// F-class split: the decoder knows the first column of the sorted matrix
// (the block's bytes in sorted order) before it decodes the L column, so
// the class of F[row] is a free context for the symbol at that row. Symbols
// are routed into one static stream per class; classes are intervals of
// rows, so the decoder switches streams at most kMaxBounds times per block.
constexpr int kMaxClasses = 8;
constexpr int kMaxBounds = 256;

// Symbol alphabet of kModeRle (one byte per symbol, see file comment):
//   0..15      run of 1..16 rank-0 symbols
//   16         long run: length byte(s) follow, run = 17 + L (255 continues)
//   17..254    MTF rank 1..238
//   255        rank escape: next byte is the rank (239..255)
constexpr usize kShortRunMax = 16;
constexpr u8 kSymLongRun = 16;
constexpr u8 kSymRankBase = 17;
constexpr u8 kSymRankEsc = 255;
constexpr u32 kDirectRankMax = kSymRankEsc - kSymRankBase;  // 238
// Side streams shorter than this are stored raw (a static body costs ~20 B).
constexpr usize kSideRawMax = 32;

// Chain count by block size.
constexpr usize kChains16Min = 256 * 1024;
constexpr usize kChains8Min = 64 * 1024;
// The fused nav packs (byte, row) into a u32, so the row must fit in 24
// bits; larger blocks (up to kBwtMaxBlock) take bwt_inverse's split
// link32 + symbol tables with the same chain anchors.
constexpr usize kPackedMax = usize{1} << 24;
// LF build interleave.
constexpr int kBuildChunks = 8;
// Below this the transform overhead (rows, two model headers) outweighs
// anything BWT can gain.
constexpr usize kMinBlock = 4096;
// Slack after the BWT buffer: the fused inverse writes 16 bytes per symbol
// unconditionally and up to one full run (<= 256) past the current offset.
constexpr usize kBwtSlack = 16 + 256;

int chains_for(usize n) {
  if (n >= kChains16Min) return 16;
  if (n >= kChains8Min) return 8;
  return 1;
}

// Output segment c is [c*L, (c+1)*L) with L = ceil(n/K); the last is shorter.
usize seg_end(usize n, int K, int c) {
  const usize L = (n + static_cast<usize>(K) - 1) / static_cast<usize>(K);
  const usize e = static_cast<usize>(c + 1) * L;
  return e < n ? e : n;
}

// ---------------------------------------------------------------------------
// Encoder side: MTF + zero-run symbolisation.
// ---------------------------------------------------------------------------
// Symbol stream = main symbols (exactly one byte each) followed by the side
// bytes (long-run lengths, escaped ranks) in order of use. Keeping the main
// stream fixed-width matters for the decoder: with inline length bytes the
// address of the next symbol depends on the value of the current one, and
// that load-compare-add chain (~7 cycles) bounds the whole inverse.
// Class intervals over L-column positions: [pos[k], pos[k+1]) has class
// cls[k], pos[nb] = n. One interval of class 0 when the block is not split.
struct ClassBounds {
  int nb = 1;
  usize pos[kMaxBounds + 1] = {0};
  u8 cls[kMaxBounds] = {0};
  u8 ncls = 1;
};

// Symbolise mtf[0, n) interval by interval into main[g] / side[g]; a run
// never crosses an interval boundary.
void rle_encode_split(const u8* mtf, usize n, const ClassBounds& B,
                      std::vector<u8>* main, std::vector<u8>* side) {
  for (int g = 0; g < B.ncls; ++g) {
    main[g].clear();
    side[g].clear();
  }
  main[B.cls[0]].reserve(n + 16);
  for (int k = 0; k < B.nb; ++k) {
    std::vector<u8>& out = main[B.cls[k]];
    std::vector<u8>& sd = side[B.cls[k]];
    const usize end = k + 1 < B.nb ? B.pos[k + 1] : n;
    usize i = B.pos[k];
    while (i < end) {
      if (mtf[i] == 0) {
        usize run = count_eq_run(mtf + i, end - i, 0);
        i += run;
        if (run <= kShortRunMax) {
          out.push_back(static_cast<u8>(run - 1));
          continue;
        }
        out.push_back(kSymLongRun);
        run -= kShortRunMax + 1;
        while (run >= 255) {
          sd.push_back(255);
          run -= 255;
        }
        sd.push_back(static_cast<u8>(run));
        continue;
      }
      const u32 rank = mtf[i++];
      if (rank <= kDirectRankMax) {
        out.push_back(static_cast<u8>(rank + kSymRankBase - 1));
      } else {
        out.push_back(kSymRankEsc);
        sd.push_back(static_cast<u8>(rank));
      }
    }
  }
}

// In-place sticky MTF (kMtfSticky), the counterpart of the decoder's
// kShufSticky masks: rank >= 2 -> slot 1, rank 1 -> front iff the previous
// rank was nonzero.
void mtf_sticky_forward(u8* buf, usize n) {
  u8 order[256], pos[256];
  for (int i = 0; i < 256; ++i) {
    order[i] = static_cast<u8>(i);
    pos[i] = static_cast<u8>(i);
  }
  u32 prev = 0;
  for (usize i = 0; i < n; ++i) {
    const u8 c = buf[i];
    const u32 j = pos[c];
    buf[i] = static_cast<u8>(j);
    if (j == 1) {
      if (prev != 0) {
        const u8 f = order[0];
        order[0] = c;
        order[1] = f;
        pos[c] = 0;
        pos[f] = 1;
      }
    } else if (j >= 2) {
      std::memmove(order + 2, order + 1, j - 1);
      order[1] = c;
      for (u32 k = 1; k <= j; ++k) pos[order[k]] = static_cast<u8>(k);
    }
    prev = j;
  }
}

// F byte -> class. Whitespace, digits, upper case, lower case, the rest:
// what distinguishes "which bytes precede a word start" from "which bytes
// precede a lower-case letter", the split that pays on text. Finer maps
// (per frequent byte) were measured and lose to their table overhead.
u8 f_class(u8 c) {
  if (c == ' ' || c == '\n' || c == '\t' || c == '\r') return 0;
  if (c >= '0' && c <= '9') return 1;
  if (c >= 'A' && c <= 'Z') return 2;
  if (c >= 'a' && c <= 'z') return 3;
  return 4;
}

// Class intervals from the byte histogram: F row 0 is the sentinel row
// (joined to the first interval), rows 1..n hold the bytes in sorted order;
// row `primary` has no L byte, so positions above it shift down by one.
// Intervals that end up empty are dropped and equal neighbours merged.
void class_bounds(const u32* hist, usize n, usize primary, ClassBounds& B) {
  usize rpos[kMaxBounds + 1];
  u8 rcls[kMaxBounds];
  int nb = 0;
  usize row = 1;
  for (int c = 0; c < 256; ++c) {
    if (!hist[c]) continue;
    const u8 k = f_class(static_cast<u8>(c));
    if (nb == 0) {
      rpos[0] = 0;
      rcls[0] = k;
      nb = 1;
    } else if (k != rcls[nb - 1]) {
      rpos[nb] = row;
      rcls[nb] = k;
      ++nb;
    }
    row += hist[c];
  }
  B.nb = 0;
  u8 maxc = 0;
  for (int k = 0; k < nb; ++k) {
    const usize p = rpos[k] - (rpos[k] > primary ? 1 : 0);
    if (B.nb > 0 && p == B.pos[B.nb - 1]) {
      B.cls[B.nb - 1] = rcls[k];  // previous interval vanished
    } else if (B.nb > 0 && B.cls[B.nb - 1] == rcls[k]) {
      continue;
    } else {
      B.pos[B.nb] = p;
      B.cls[B.nb] = rcls[k];
      ++B.nb;
    }
    if (B.nb >= 2 && B.cls[B.nb - 1] == B.cls[B.nb - 2]) --B.nb;
  }
  if (B.nb == 0) {
    B.nb = 1;
    B.pos[0] = 0;
    B.cls[0] = 0;
  }
  B.pos[B.nb] = n;
  for (int k = 0; k < B.nb; ++k) maxc = maxc > B.cls[k] ? maxc : B.cls[k];
  B.ncls = static_cast<u8>(maxc + 1);
}

// ---------------------------------------------------------------------------
// Decoder side.
// ---------------------------------------------------------------------------

// Shuffle masks: pull lane j to lane 0, shift lanes 0..j-1 up by one.
struct ShufMasks {
  alignas(16) u8 m[16][16];
  ShufMasks() {
    for (int j = 0; j < 16; ++j) {
      m[j][0] = static_cast<u8>(j);
      for (int k = 1; k <= j; ++k) m[j][k] = static_cast<u8>(k - 1);
      for (int k = j + 1; k < 16; ++k) m[j][k] = static_cast<u8>(k);
    }
  }
};
const ShufMasks kShuf;

// Lane masks: 0xFF for lanes <= r.
struct LaneMasks {
  alignas(16) u8 m[16][16];
  LaneMasks() {
    for (int r = 0; r < 16; ++r)
      for (int k = 0; k < 16; ++k) m[r][k] = k <= r ? 0xFF : 0x00;
  }
};
const LaneMasks kLaneLE;

// Sticky-MTF shuffle masks, indexed by [previous rank nonzero][rank]:
// rank 0 identity; rank 1 swaps lanes 0/1 only after a nonzero rank;
// rank j >= 2 pulls lane j to lane 1 and shifts lanes 1..j-1 up by one.
struct ShufStickyMasks {
  alignas(16) u8 m[2][16][16];
  alignas(16) u8 ins1[16];  // lanes [0, x, 1, 2, ..., 14]: slot for a rank >= 16
  ShufStickyMasks() {
    for (int pz = 0; pz < 2; ++pz)
      for (int j = 0; j < 16; ++j) {
        for (int k = 0; k < 16; ++k) m[pz][j][k] = static_cast<u8>(k);
        if (j == 1) {
          if (pz) {
            m[pz][j][0] = 1;
            m[pz][j][1] = 0;
          }
        } else if (j >= 2) {
          m[pz][j][1] = static_cast<u8>(j);
          for (int k = 2; k <= j; ++k) m[pz][j][k] = static_cast<u8>(k - 1);
        }
      }
    ins1[0] = 0;
    ins1[1] = 0;
    for (int k = 2; k < 16; ++k) ins1[k] = static_cast<u8>(k - 1);
  }
};
const ShufStickyMasks kShufSticky;

// Per-class symbol streams of a kModeSplit block.
struct StreamSet {
  const u8* sp[kMaxClasses];
  const u8* sp_end[kMaxClasses];
  const u8* ex[kMaxClasses];
  const u8* ex_end[kMaxClasses];
};

// Fused: zero-run expansion + inverse MTF + per-chunk histogram of the
// recovered BWT bytes. Writes exactly n bytes into bwt (which has kBwtSlack
// bytes of slack) or returns false. sym has >= 16 bytes of readable slack.
bool rle_mtf_inverse(const u8* sym, usize main_n, usize sym_n, u8* bwt,
                     usize n, u32 (*hist)[256]) {
  const u8* ex = sym + main_n;
  const u8* const ex_end = sym + sym_n;
  alignas(16) u8 ord[256];
  for (int s = 0; s < 256; ++s) ord[s] = static_cast<u8>(s);
  std::memset(hist, 0, sizeof(u32) * 256 * kBuildChunks);
  const usize seg = (n + kBuildChunks - 1) / kBuildChunks;
  int chunk = 0;
  usize chunk_end = seg < n ? seg : n;
  u32* h = hist[0];

#if CRYB_SHUF
#if CRYB_USE_NEON
  uint8x16_t v = vld1q_u8(ord);
#else
  __m128i v = _mm_load_si128(reinterpret_cast<const __m128i*>(ord));
#endif
#endif
  usize i = 0, o = 0;
  for (; i < main_n && o < n; ++i) {
    const u32 t = sym[i];
    usize len = t < kSymLongRun ? t + 1 : 1;
    // Rank: 0 for the run symbols, t - 16 for direct ranks; the escape maps
    // to >= 16 here and is resolved on the slow path.
    u32 j = t >= kSymRankBase ? t - kSymRankBase + 1 : 0;
    u8 front;
    if (j == 0) {
#if CRYB_SHUF
#if CRYB_USE_NEON
      front = static_cast<u8>(vgetq_lane_u8(v, 0));
#else
      front = static_cast<u8>(_mm_cvtsi128_si32(v));
#endif
#else
      front = ord[0];
#endif
    } else if (j < 16) {
#if CRYB_SHUF
#if CRYB_USE_NEON
      v = vqtbl1q_u8(v, vld1q_u8(kShuf.m[j]));
      front = static_cast<u8>(vgetq_lane_u8(v, 0));
#else
      v = _mm_shuffle_epi8(
          v, _mm_load_si128(reinterpret_cast<const __m128i*>(kShuf.m[j])));
      front = static_cast<u8>(_mm_cvtsi128_si32(v));
#endif
#else
      front = ord[j];
      for (u32 k = j; k > 0; --k) ord[k] = ord[k - 1];
      ord[0] = front;
#endif
    } else {
      // Rank >= 16 (or escaped): shift ord[0..j) up by one across 16-byte
      // chunks. Chunk k becomes [chunk k-1 lane 15, chunk k lanes 0..14];
      // the chunk holding j only shifts its lanes <= j & 15.
      if (t == kSymRankEsc) j = ex < ex_end ? *ex++ : 0;
#if CRYB_SHUF
      const u32 c = j >> 4, rr = j & 15;
#if CRYB_USE_NEON
      vst1q_u8(ord, v);
      front = ord[j];
      if (c) {
        const uint8x16_t cur = vld1q_u8(ord + 16 * c);
        const uint8x16_t prev = vld1q_u8(ord + 16 * c - 16);
        vst1q_u8(ord + 16 * c,
                 vbslq_u8(vld1q_u8(kLaneLE.m[rr]), vextq_u8(prev, cur, 15), cur));
        for (u32 k = c - 1; k >= 1; --k)
          vst1q_u8(ord + 16 * k,
                   vextq_u8(vld1q_u8(ord + 16 * k - 16), vld1q_u8(ord + 16 * k), 15));
      }
      v = vsetq_lane_u8(front, vextq_u8(v, v, 15), 0);
#else
      _mm_store_si128(reinterpret_cast<__m128i*>(ord), v);
      front = ord[j];
      if (c) {
        __m128i* const oc = reinterpret_cast<__m128i*>(ord + 16 * c);
        const __m128i cur = _mm_load_si128(oc);
        const __m128i prev = _mm_load_si128(oc - 1);
        _mm_store_si128(
            oc, _mm_blendv_epi8(cur, _mm_alignr_epi8(cur, prev, 15),
                                _mm_load_si128(reinterpret_cast<const __m128i*>(kLaneLE.m[rr]))));
        for (u32 k = c - 1; k >= 1; --k) {
          __m128i* const ok = reinterpret_cast<__m128i*>(ord + 16 * k);
          _mm_store_si128(ok, _mm_alignr_epi8(_mm_load_si128(ok), _mm_load_si128(ok - 1), 15));
        }
      }
      v = _mm_insert_epi8(_mm_alignr_epi8(v, v, 15), front, 0);
#endif
#else
      front = ord[j];
      std::memmove(ord + 1, ord, j);
      ord[0] = front;
#endif
    }
    if (t == kSymLongRun) {
      // Long run: 17 + side bytes, continued while a byte is 255. Clamp to
      // what fits: a corrupt stream must not write past the slack.
      len = kShortRunMax + 1;
      while (ex < ex_end) {
        const u32 more = *ex++;
        len += more;
        if (more != 255) break;
      }
    }
    // Clamp every symbol, not just long runs: a corrupt short run crossing
    // n would otherwise spin the histogram split below (part == 0 forever)
    // and write past the slack.
    if (len > n - o) len = n - o;
#if CRYB_SHUF
#if CRYB_USE_NEON
    vst1q_u8(bwt + o, vdupq_n_u8(front));
#else
    _mm_storeu_si128(reinterpret_cast<__m128i*>(bwt + o), _mm_set1_epi8(front));
#endif
#else
    std::memset(bwt + o, front, 16);
#endif
    if (len > 16) std::memset(bwt + o + 16, front, len - 16);
    // Per-chunk histogram; a run straddling a chunk end is split (rare).
    if (o + len > chunk_end) {
      usize rem = len, at = o;
      while (rem) {
        const usize part = rem < chunk_end - at ? rem : chunk_end - at;
        h[front] += static_cast<u32>(part);
        at += part;
        rem -= part;
        if (at == chunk_end && at < n) {
          ++chunk;
          chunk_end = chunk_end + seg < n ? chunk_end + seg : n;
          h = hist[chunk];
        }
      }
    } else {
      h[front] += static_cast<u32>(len);
    }
    o += len;
  }
  return o == n && i == main_n && ex == ex_end;
}

// kModeSplit counterpart of rle_mtf_inverse: symbols come from the stream
// of the current class interval, and the MTF update is the plain or the
// sticky variant. Same output contract (n bytes into bwt, per-chunk
// histogram); every stream must be consumed exactly.
template <bool kSticky>
bool rle_mtf_inverse_split(StreamSet& S, const ClassBounds& B, u8* bwt, usize n,
                           u32 (*hist)[256]) {
  alignas(16) u8 ord[256];
  for (int s = 0; s < 256; ++s) ord[s] = static_cast<u8>(s);
  std::memset(hist, 0, sizeof(u32) * 256 * kBuildChunks);
  const usize seg = (n + kBuildChunks - 1) / kBuildChunks;
  int chunk = 0;
  usize chunk_end = seg < n ? seg : n;
  u32* h = hist[0];

#if CRYB_SHUF
#if CRYB_USE_NEON
  uint8x16_t v = vld1q_u8(ord);
#else
  __m128i v = _mm_load_si128(reinterpret_cast<const __m128i*>(ord));
#endif
#endif
  int bi = 0;
  u32 g = B.cls[0];
  usize next_bound = B.nb > 1 ? B.pos[1] : n + 1;
  // Current class's stream cursors live in locals; swapped on a class change.
  const u8* sp = S.sp[g];
  const u8* spe = S.sp_end[g];
  const u8* ex = S.ex[g];
  const u8* exe = S.ex_end[g];
  u32 pz = 0;  // previous rank nonzero (sticky variant)
  usize o = 0;
  while (o < n) {
    if (sp == spe) return false;
    const u32 t = *sp++;
    usize len = t < kSymLongRun ? t + 1 : 1;
    u32 j = t >= kSymRankBase ? t - kSymRankBase + 1 : 0;
    u8 c;
    if (j == 0) {
#if CRYB_SHUF
#if CRYB_USE_NEON
      c = static_cast<u8>(vgetq_lane_u8(v, 0));
#else
      c = static_cast<u8>(_mm_cvtsi128_si32(v));
#endif
#else
      c = ord[0];
#endif
    } else if (j < 16) {
#if CRYB_SHUF
#if CRYB_USE_NEON
      if constexpr (kSticky) {
        c = static_cast<u8>(vgetq_lane_u8(vqtbl1q_u8(v, vdupq_n_u8(static_cast<u8>(j))), 0));
        v = vqtbl1q_u8(v, vld1q_u8(kShufSticky.m[pz][j]));
      } else {
        v = vqtbl1q_u8(v, vld1q_u8(kShuf.m[j]));
        c = static_cast<u8>(vgetq_lane_u8(v, 0));
      }
#else
      if constexpr (kSticky) {
        c = static_cast<u8>(_mm_cvtsi128_si32(
            _mm_shuffle_epi8(v, _mm_set1_epi8(static_cast<char>(j)))));
        v = _mm_shuffle_epi8(
            v, _mm_load_si128(reinterpret_cast<const __m128i*>(kShufSticky.m[pz][j])));
      } else {
        v = _mm_shuffle_epi8(
            v, _mm_load_si128(reinterpret_cast<const __m128i*>(kShuf.m[j])));
        c = static_cast<u8>(_mm_cvtsi128_si32(v));
      }
#endif
#else
      c = ord[j];
      if constexpr (kSticky) {
        if (j == 1) {
          if (pz) {
            ord[1] = ord[0];
            ord[0] = c;
          }
        } else {
          std::memmove(ord + 2, ord + 1, j - 1);
          ord[1] = c;
        }
      } else {
        for (u32 k = j; k > 0; --k) ord[k] = ord[k - 1];
        ord[0] = c;
      }
#endif
    } else {
      // Rank >= 16 (or escaped): the list beyond lane 15 shifts by one
      // across 16-byte chunks exactly as in rle_mtf_inverse; only the new
      // head chunk differs between the variants. An escaped rank below 2
      // only occurs in corrupt input and is treated as "no move".
      if (t == kSymRankEsc) j = ex < exe ? *ex++ : 0;
#if CRYB_SHUF
      const u32 cj = j >> 4, rr = j & 15;
#if CRYB_USE_NEON
      vst1q_u8(ord, v);
      c = ord[j];
      if (cj) {
        const uint8x16_t cur = vld1q_u8(ord + 16 * cj);
        const uint8x16_t prev = vld1q_u8(ord + 16 * cj - 16);
        vst1q_u8(ord + 16 * cj,
                 vbslq_u8(vld1q_u8(kLaneLE.m[rr]), vextq_u8(prev, cur, 15), cur));
        for (u32 k = cj - 1; k >= 1; --k)
          vst1q_u8(ord + 16 * k,
                   vextq_u8(vld1q_u8(ord + 16 * k - 16), vld1q_u8(ord + 16 * k), 15));
        if constexpr (kSticky)
          v = vsetq_lane_u8(c, vqtbl1q_u8(v, vld1q_u8(kShufSticky.ins1)), 1);
        else
          v = vsetq_lane_u8(c, vextq_u8(v, v, 15), 0);
      }
#else
      _mm_store_si128(reinterpret_cast<__m128i*>(ord), v);
      c = ord[j];
      if (cj) {
        __m128i* const oc = reinterpret_cast<__m128i*>(ord + 16 * cj);
        const __m128i cur = _mm_load_si128(oc);
        const __m128i prev = _mm_load_si128(oc - 1);
        _mm_store_si128(
            oc, _mm_blendv_epi8(cur, _mm_alignr_epi8(cur, prev, 15),
                                _mm_load_si128(reinterpret_cast<const __m128i*>(kLaneLE.m[rr]))));
        for (u32 k = cj - 1; k >= 1; --k) {
          __m128i* const ok = reinterpret_cast<__m128i*>(ord + 16 * k);
          _mm_store_si128(ok, _mm_alignr_epi8(_mm_load_si128(ok), _mm_load_si128(ok - 1), 15));
        }
        if constexpr (kSticky)
          v = _mm_insert_epi8(
              _mm_shuffle_epi8(v, _mm_load_si128(reinterpret_cast<const __m128i*>(kShufSticky.ins1))),
              c, 1);
        else
          v = _mm_insert_epi8(_mm_alignr_epi8(v, v, 15), c, 0);
      }
#endif
#else
      c = ord[j];
      if (j >= 16) {
        if constexpr (kSticky) {
          std::memmove(ord + 2, ord + 1, j - 1);
          ord[1] = c;
        } else {
          std::memmove(ord + 1, ord, j);
          ord[0] = c;
        }
      }
#endif
    }
    if constexpr (kSticky) pz = j != 0;
    if (t == kSymLongRun) {
      len = kShortRunMax + 1;
      while (ex < exe) {
        const u32 more = *ex++;
        len += more;
        if (more != 255) break;
      }
    }
    if (len > n - o) len = n - o;
#if CRYB_SHUF
#if CRYB_USE_NEON
    vst1q_u8(bwt + o, vdupq_n_u8(c));
#else
    _mm_storeu_si128(reinterpret_cast<__m128i*>(bwt + o), _mm_set1_epi8(static_cast<char>(c)));
#endif
#else
    std::memset(bwt + o, c, 16);
#endif
    if (len > 16) std::memset(bwt + o + 16, c, len - 16);
    if (o + len > chunk_end) {
      usize rem = len, at = o;
      while (rem) {
        const usize part = rem < chunk_end - at ? rem : chunk_end - at;
        h[c] += static_cast<u32>(part);
        at += part;
        rem -= part;
        if (at == chunk_end && at < n) {
          ++chunk;
          chunk_end = chunk_end + seg < n ? chunk_end + seg : n;
          h = hist[chunk];
        }
      }
    } else {
      h[c] += static_cast<u32>(len);
    }
    o += len;
    if (o >= next_bound) {
      S.sp[g] = sp;
      S.ex[g] = ex;
      do {
        ++bi;
        g = B.cls[bi];
        next_bound = bi + 1 < B.nb ? B.pos[bi + 1] : n + 1;
      } while (o >= next_bound);
      sp = S.sp[g];
      spe = S.sp_end[g];
      ex = S.ex[g];
      exe = S.ex_end[g];
    }
  }
  S.sp[g] = sp;
  S.ex[g] = ex;
  for (int k = 0; k < B.ncls; ++k)
    if (S.sp[k] != S.sp_end[k] || S.ex[k] != S.ex_end[k]) return false;
  return true;
}

// Fused LF table: nav[row] = (L-column byte << 24) | LF(row), over the
// (n + 1)-row sentinel matrix; row `primary` (the dropped sentinel) links to
// row 0. Built as kBuildChunks interleaved counting-sort streams, each with
// its own running counters seeded from the per-chunk histograms.
void build_nav_chunked(const u8* bwt, usize n, usize primary,
                       u32 (*hist)[256], u32* nav) {
  constexpr int K = kBuildChunks;
  const usize seg = (n + K - 1) / K;
  alignas(64) u32 next[K][256];
  {
    u32 tot[256] = {0};
    for (int k = 0; k < K; ++k)
      for (int c = 0; c < 256; ++c) tot[c] += hist[k][c];
    u32 sum = 1;  // sentinel occupies F row 0
    for (int c = 0; c < 256; ++c) {
      u32 acc = sum;
      for (int k = 0; k < K; ++k) {
        next[k][c] = acc;
        acc += hist[k][c];
      }
      sum += tot[c];
    }
  }
  // Byte index j maps to row j (j < primary) or j + 1 (j >= primary).
  const usize last_len = n - static_cast<usize>(K - 1) * seg;
  usize s = 0;
  for (; s < last_len; ++s) {
    for (int k = 0; k < K; ++k) {
      const usize j = static_cast<usize>(k) * seg + s;
      const u8 c = bwt[j];
      const u32 r = next[k][c];
      next[k][c] = r + 1;
      nav[j + (j >= primary)] = r | (static_cast<u32>(c) << 24);
    }
  }
  for (; s < seg; ++s) {
    for (int k = 0; k < K - 1; ++k) {
      const usize j = static_cast<usize>(k) * seg + s;
      const u8 c = bwt[j];
      const u32 r = next[k][c];
      next[k][c] = r + 1;
      nav[j + (j >= primary)] = r | (static_cast<u32>(c) << 24);
    }
  }
  nav[primary] = 0;
}

// Single-chain LF walk (small blocks). Multi-chain blocks use
// bwt_decode::walk_k_tab over the same fused nav: K register-resident rows,
// 64-byte staging lines flushed straight into dst.
void walk_single(const u32* __restrict nav, usize n, usize row,
                 u8* __restrict dst) {
  u32 p = static_cast<u32>(row);
  for (usize o = n; o-- > 0;) {
    const u32 e = nav[p];
    dst[o] = static_cast<u8>(e >> 24);
    p = e & 0xFFFFFFu;
  }
}

// Order-1 static-coder cost estimate (bits) of a symbol stream: conditional
// entropy given the previous symbol plus a table-description charge per
// used (context, symbol) cell. Tracks the real static coder within a few %
// (it has richer contexts but pays for its tables the same way).
double order1_cost_bits(const u8* p, usize n) {
  if (n == 0) return 0.0;
  thread_local std::vector<u32> cnt;
  cnt.assign(256 * 256, 0);
  u32 prev = 0;
  for (usize i = 0; i < n; ++i) {
    cnt[prev * 256 + p[i]]++;
    prev = p[i];
  }
  double bits = 0.0;
  usize cells = 0;
  for (u32 c = 0; c < 256; ++c) {
    const u32* row = &cnt[c * 256];
    u32 tot = 0;
    for (int s = 0; s < 256; ++s) tot += row[s];
    if (!tot) continue;
    const double lt = std::log2(static_cast<double>(tot));
    for (int s = 0; s < 256; ++s) {
      if (!row[s]) continue;
      bits += row[s] * (lt - std::log2(static_cast<double>(row[s])));
      ++cells;
    }
  }
  return bits + static_cast<double>(cells) * 10.0;
}

// Transform state handed from the estimate to the encode of the same block.
// The estimate always recomputes; the encode consumes a valid entry for the
// same block and invalidates it. Identity is (pointer, length, content
// hash): a pointer/length match alone is not enough, since the estimate is
// also run on blocks that then take the lattice path, and a later encode of
// different data in a reused buffer (Mosaic/Oracle tiles, the next file)
// would be served the stale transform -- and emit a payload of the wrong
// block. The hash costs ~1 ms/MiB against a ~30 ms transform.
struct Prepared {
  bool valid = false;
  const u8* src = nullptr;
  usize n = 0;
  u64 hash = 0;
  int K = 1;
  usize primary = 0;
  usize rows[kBwtMaxChains] = {0};
  u8 variant = kMtfPlain;
  ClassBounds bounds;
  std::vector<u8> bwt, mtf, alt;
  std::vector<u8> main[kMaxClasses], side[kMaxClasses];
  double est_bits = 0.0;  // order-1 estimate of the chosen configuration
};
thread_local Prepared g_prep;

// Order-1 estimate of a set of symbol streams (side streams below
// kSideRawMax are stored raw).
double streams_cost_bits(const std::vector<u8>* main, const std::vector<u8>* side, int ncls) {
  double bits = 0.0;
  for (int g = 0; g < ncls; ++g) {
    if (main[g].empty()) continue;
    bits += order1_cost_bits(main[g].data(), main[g].size()) + 3 * 8.0;
    bits += side[g].size() >= kSideRawMax
                ? order1_cost_bits(side[g].data(), side[g].size()) + 8.0
                : static_cast<double>(side[g].size()) * 8.0;
  }
  return bits;
}

// The transform and the plain-MTF single-stream symbolisation, priced with
// the order-1 estimate (the detection stage's number; its 0.93 calibration
// in crystal_bwt_estimate was measured on exactly this configuration). The
// variant/split decisions are made in build_crystal_bwt by real encodes.
void prepare(const u8* src, usize n, bool reuse) {
  Prepared& P = g_prep;
  const u64 h = content_hash(src, n);
  if (reuse && P.valid && P.src == src && P.n == n && P.hash == h) return;
  P.valid = true;
  P.src = src;
  P.n = n;
  P.hash = h;
  P.K = chains_for(n);
  usize positions[kBwtMaxChains];
  for (int c = 0; c < P.K; ++c) positions[c] = seg_end(n, P.K, c);
  P.bwt.resize(n);
  P.primary = bwt_forward_rows(src, n, P.bwt.data(), positions, P.K, P.rows);

  P.bounds = ClassBounds{};
  P.bounds.pos[1] = n;
  P.variant = kMtfPlain;
  P.mtf.assign(P.bwt.begin(), P.bwt.end());
  mtf_forward(P.mtf.data(), n);
  rle_encode_split(P.mtf.data(), n, P.bounds, P.main, P.side);
  P.est_bits = streams_cost_bits(P.main, P.side, 1);
}

// Static bodies for a set of streams at `level`; returns the total size.
struct Bodies {
  std::vector<u8> main[kMaxClasses], side[kMaxClasses];
  usize total = 0;
};
usize encode_streams(const std::vector<u8>* main, const std::vector<u8>* side, int ncls,
                     int level, Bodies& out) {
  out.total = 0;
  for (int g = 0; g < ncls; ++g) {
    out.main[g].clear();
    out.side[g].clear();
    if (main[g].empty()) continue;
    build_crystal_static(main[g].data(), main[g].size(), level, 1, out.main[g]);
    out.total += out.main[g].size() + 3;
    if (side[g].size() >= kSideRawMax) {
      build_crystal_static(side[g].data(), side[g].size(), level, 1, out.side[g]);
      out.total += out.side[g].size() + 1;
    } else {
      out.side[g] = side[g];
      out.total += side[g].size();
    }
  }
  return out.total;
}

}  // namespace

// Detection stage. The BWT path does not announce itself in byte statistics
// (zero-heavy numeric records repeat n-grams as eagerly as prose), so the
// decision is made on a cost estimate instead: run the transform (libsais,
// ~1/3 of a lattice encode) and price the symbol stream with an order-1
// model. The caller compares this with the real lattice size and only pays
// for the second static encode when the estimate promises a clear win.
usize crystal_bwt_estimate(const u8* src, usize n) {
  if (n < kMinBlock) return ~usize{0};
  prepare(src, n, /*reuse=*/false);
  const Prepared& P = g_prep;
  // The static coder's two-feature contexts and MDL grouping beat plain
  // order-1 by 2-10% on these streams (more on numeric planes); 0.93 is the
  // observed median over Silesia and the verdict sets.
  return static_cast<usize>(P.est_bits / 8.0 * 0.93) + 8 + 3 * static_cast<usize>(P.K);
}

void build_crystal_bwt(const u8* src, usize n, int level,
                      std::vector<u8>& payload) {
  prepare(src, n, /*reuse=*/true);
  Prepared& P = g_prep;
  P.valid = false;
  const int K = P.K;

  // Two decisions by real level-1 encodes (the order-1 estimate charges
  // tables per used cell and misjudges the split by 2-3%, the size of the
  // effect): first the MTF variant on the single stream, then whether the
  // F-class split of the chosen variant pays. Three screen encodes at level
  // 1 add ~50% to a level-5 text encode (nothing at level 9); the winner is
  // re-encoded at `level`. Levels 1-2 are the throughput levels and skip the
  // screens (plain MTF, one stream: the original configuration). The side
  // bytes always get their own body: sharing tables with the main stream
  // costs ~1% (their spread-out values pollute the peaked rank contexts).
  // Coding the L column without MTF was tried and is 20-30% larger on every
  // input, with or without the split.
  constexpr int kScreenLevel = 1;
  constexpr int kScreenMinLevel = 3;
  thread_local Bodies bodA, bodB;
  thread_local std::vector<u8> amain[kMaxClasses], aside[kMaxClasses];
  Bodies* best = &bodA;
  Bodies* trial = &bodB;
  if (level >= kScreenMinLevel) {
    encode_streams(P.main, P.side, 1, kScreenLevel, *best);

    P.alt.assign(P.bwt.begin(), P.bwt.end());
    mtf_sticky_forward(P.alt.data(), n);
    rle_encode_split(P.alt.data(), n, P.bounds, amain, aside);
    if (encode_streams(amain, aside, 1, kScreenLevel, *trial) < best->total) {
      std::swap(best, trial);
      P.variant = kMtfSticky;
      P.mtf.swap(P.alt);
      P.main[0].swap(amain[0]);
      P.side[0].swap(aside[0]);
    }

    u32 hist[256] = {0};
    for (usize i = 0; i < n; ++i) hist[src[i]]++;
    ClassBounds split;
    class_bounds(hist, n, P.primary, split);
    if (split.nb > 1) {
      rle_encode_split(P.mtf.data(), n, split, amain, aside);
      const usize dir = 2 + 7 * static_cast<usize>(split.nb);  // interval directory
      if (encode_streams(amain, aside, split.ncls, kScreenLevel, *trial) + dir < best->total) {
        std::swap(best, trial);
        P.bounds = split;
        for (int g = 0; g < split.ncls; ++g) {
          P.main[g].swap(amain[g]);
          P.side[g].swap(aside[g]);
        }
      }
    }
  }
  const ClassBounds& B = P.bounds;
  encode_streams(P.main, P.side, B.ncls, level, *best);

  ByteWriter w(payload);
  w.byte(kModeSplit);
  w.byte(static_cast<u8>(K));
  w.varint(P.primary);
  for (int c = 0; c + 1 < K; ++c) w.varint(P.rows[c]);
  w.byte(P.variant);
  w.byte(static_cast<u8>(B.nb));
  for (int k = 0; k < B.nb; ++k) {
    if (k) w.varint(B.pos[k]);
    w.byte(B.cls[k]);
  }
  for (int g = 0; g < B.ncls; ++g) {
    const std::vector<u8>& m = P.main[g];
    const std::vector<u8>& s = P.side[g];
    w.varint(m.size());
    w.varint(s.size());
    if (m.empty()) continue;
    w.varint(best->main[g].size());
    w.bytes(best->main[g]);
    if (s.size() >= kSideRawMax) w.varint(best->side[g].size());
    w.bytes(best->side[g]);
  }
}

Status decode_crystal_bwt(const u8* data, usize len, u8* dst, usize n) {
  ByteReader r(data, len);
  const u8 mode = r.byte();
  const int K = r.byte();
  const usize primary = static_cast<usize>(r.varint());
  if (K != chains_for(n)) return Status::CorruptInput;
  usize rows[kBwtMaxChains];
  for (int c = 0; c + 1 < K; ++c) rows[c] = static_cast<usize>(r.varint());
  rows[K - 1] = 0;  // sentinel row emits src[n-1]
  if (r.error || (mode != kModeRle && mode != kModeSplit)) return Status::CorruptInput;
  // The encoder never takes this path below kMinBlock, and the chunked LF
  // build assumes 7 * ceil(n/8) < n.
  if (n < kMinBlock || primary > n) return Status::CorruptInput;
  for (int c = 0; c < K; ++c)
    if (rows[c] > n) return Status::CorruptInput;

  thread_local std::vector<u8> bwt, sym;
  thread_local std::vector<u32> nav;
  bwt.resize(n + kBwtSlack);
  alignas(64) u32 hist[kBuildChunks][256];
  CRYB_T(t0);
  if (mode == kModeRle) {
    const usize main_n = static_cast<usize>(r.varint());
    const usize side_n = static_cast<usize>(r.varint());
    const usize main_len = static_cast<usize>(r.varint());
    // Every main symbol covers >= 1 byte; side bytes are at most one per
    // main symbol (final length or escaped rank) plus one 255-continuation
    // per 255 output bytes.
    if (r.error || main_n > n || side_n > main_n + n / 255 + 1)
      return Status::CorruptInput;
    const u8* main_body = r.take(main_len);
    if (!main_body) return Status::CorruptInput;
    sym.resize(main_n + side_n + 16);
    std::memset(sym.data() + main_n + side_n, 0, 16);
    if (main_n) {
      const Status st = decode_crystal_static(main_body, main_len, sym.data(), main_n);
      if (st != Status::Ok) return st;
    }
    if (side_n >= kSideRawMax) {
      const Status st =
          decode_crystal_static(r.p, r.remaining(), sym.data() + main_n, side_n);
      if (st != Status::Ok) return st;
    } else if (side_n) {
      const u8* raw = r.take(side_n);
      if (!raw) return Status::CorruptInput;
      std::memcpy(sym.data() + main_n, raw, side_n);
    }
    if (!rle_mtf_inverse(sym.data(), main_n, main_n + side_n, bwt.data(), n, hist))
      return Status::CorruptInput;
  } else {
    const u8 variant = r.byte();
    const int nb = r.byte();
    if (r.error || variant > kMtfSticky || nb < 1 || nb > kMaxBounds)
      return Status::CorruptInput;
    ClassBounds B;
    B.nb = nb;
    u8 maxc = 0;
    for (int k = 0; k < nb; ++k) {
      B.pos[k] = k ? static_cast<usize>(r.varint()) : 0;
      B.cls[k] = r.byte();
      if (r.error || B.cls[k] >= kMaxClasses || (k && B.pos[k] <= B.pos[k - 1]) ||
          B.pos[k] >= n)
        return Status::CorruptInput;
      maxc = maxc > B.cls[k] ? maxc : B.cls[k];
    }
    B.pos[nb] = n;
    B.ncls = static_cast<u8>(maxc + 1);
    // Stream directory first (lengths), bodies decoded into one buffer with
    // 16 bytes of slack per stream.
    usize main_n[kMaxClasses], side_n[kMaxClasses], off[kMaxClasses];
    usize total = 0, main_sum = 0, side_sum = 0;
    for (int g = 0; g < B.ncls; ++g) {
      main_n[g] = static_cast<usize>(r.varint());
      side_n[g] = static_cast<usize>(r.varint());
      if (r.error || main_n[g] > n || side_n[g] > main_n[g] + n / 255 + 1 ||
          (main_n[g] == 0 && side_n[g] != 0))
        return Status::CorruptInput;
      main_sum += main_n[g];
      side_sum += side_n[g];
      off[g] = total;
      total += main_n[g] + side_n[g] + 16;
      if (main_n[g] == 0) continue;
      const usize main_len = static_cast<usize>(r.varint());
      const u8* main_body = r.take(main_len);
      if (r.error || !main_body) return Status::CorruptInput;
      sym.resize(total);
      std::memset(sym.data() + off[g] + main_n[g] + side_n[g], 0, 16);
      const Status st =
          decode_crystal_static(main_body, main_len, sym.data() + off[g], main_n[g]);
      if (st != Status::Ok) return st;
      if (side_n[g] >= kSideRawMax) {
        const usize side_len = static_cast<usize>(r.varint());
        const u8* side_body = r.take(side_len);
        if (r.error || !side_body) return Status::CorruptInput;
        const Status ss = decode_crystal_static(side_body, side_len,
                                                sym.data() + off[g] + main_n[g], side_n[g]);
        if (ss != Status::Ok) return ss;
      } else if (side_n[g]) {
        const u8* raw = r.take(side_n[g]);
        if (!raw) return Status::CorruptInput;
        std::memcpy(sym.data() + off[g] + main_n[g], raw, side_n[g]);
      }
    }
    if (main_sum > n || side_sum > main_sum + n / 255 + 1) return Status::CorruptInput;
    sym.resize(total);
    StreamSet S;
    for (int g = 0; g < B.ncls; ++g) {
      S.sp[g] = sym.data() + off[g];
      S.sp_end[g] = S.sp[g] + main_n[g];
      S.ex[g] = S.sp_end[g];
      S.ex_end[g] = S.ex[g] + side_n[g];
    }
    const bool ok = variant == kMtfSticky
                        ? rle_mtf_inverse_split<true>(S, B, bwt.data(), n, hist)
                        : rle_mtf_inverse_split<false>(S, B, bwt.data(), n, hist);
    if (!ok) return Status::CorruptInput;
  }
  CRYB_T(t1);
#ifdef CRYB_PROFILE
  std::fprintf(stderr, "  static + rle+mtf %.2f ms (%zu bytes, mode %u)\n",
               std::chrono::duration<double, std::milli>(t1 - t0).count(), n, mode);
#endif
  if (n + 1 > kPackedMax) {
    // Rows do not fit the packed nav: the generic inverse (split 32-bit
    // links + symbol table) with the same K-chain anchors.
    return bwt_inverse(bwt.data(), n, primary, dst, K > 1 ? rows : nullptr, K);
  }
  nav.resize(n + 1);
  CRYB_T(t2);
  build_nav_chunked(bwt.data(), n, primary, hist, nav.data());
  CRYB_T(t3);
  switch (K) {
    case 16: bwt_decode::walk_k_tab<16>(bwt_decode::NavFused{nav.data()}, n, rows, dst); break;
    case 8: bwt_decode::walk_k_tab<8>(bwt_decode::NavFused{nav.data()}, n, rows, dst); break;
    default: walk_single(nav.data(), n, rows[0], dst); break;
  }
#ifdef CRYB_PROFILE
  const auto t4 = std::chrono::steady_clock::now();
  std::fprintf(stderr, "  build_nav %.2f ms  walk%d %.2f ms  total %.2f ms\n",
               std::chrono::duration<double, std::milli>(t3 - t2).count(), K,
               std::chrono::duration<double, std::milli>(t4 - t3).count(),
               std::chrono::duration<double, std::milli>(t4 - t0).count());
#endif
  return Status::Ok;
}

}  // namespace spectra8::detail
