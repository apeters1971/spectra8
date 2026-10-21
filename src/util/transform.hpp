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

// Optional, reversible byte-layout transforms ("filters") applied to a block
// before the LZ + entropy stage, and inverted after decoding. They do not
// compress on their own; they re-arrange bytes so the existing coder finds far
// more structure in numeric / columnar data.
//
// On-disk filter IDs (1 byte, written only for filtered compressed blocks):
//
//   0  = raw                       (no transform)
//   1  = shuffle16                 byte transpose, 2-byte elements
//   2  = shuffle32                 byte transpose, 4-byte elements
//   3  = shuffle64                 byte transpose, 8-byte elements
//   4  = delta16                   element delta, 2-byte LE integers
//   5  = delta32                   element delta, 4-byte LE integers
//   6  = delta64                   element delta, 8-byte LE integers
//   7  = zigzag_delta16            delta + zig-zag, 2-byte LE elements
//   8  = zigzag_delta32            delta + zig-zag, 4-byte LE elements
//   9  = zigzag_delta64            delta + zig-zag, 8-byte LE elements
//   10 = delta32 + shuffle32       4-byte element delta, then byte transpose
//   11 = zigzag_delta32 + shuffle32
//   12 = bitshuffle32              bit-plane transpose, 4-byte elements
//   13 = dict32                    dictionary remap, 4-byte elements
//   14 = dict64                    dictionary remap, 8-byte elements
//   15 = delta64 + shuffle64       8-byte element delta, then byte transpose
//   16 = zigzag32 + shuffle32      zig-zag raw values (no delta), transpose
//   17 = zigzag64 + shuffle64      zig-zag raw values (no delta), transpose
//   18 = bitshuffle64              bit-plane transpose, 8-byte elements
//   19 = delta32 + bitshuffle32    4-byte element delta, then bit-plane transpose
//   20 = dict16                    dictionary remap, 2-byte elements
//   21 = for16                     frame-of-reference + bit packing, 2-byte
//   22 = for32                     frame-of-reference + bit packing, 4-byte
//   23 = for64                     frame-of-reference + bit packing, 8-byte
//   24 = alp32                     decimal integerization, 4-byte floats
//   25 = alp64                     decimal integerization, 8-byte floats
//
//   - shuffleW     : groups the same byte position of every element together,
//                    so the near-constant high bytes of integer/float arrays
//                    collapse into long runs ("0 0 1 1 2 2 3 3" layout).
//   - deltaW       : treat the block as little-endian W-byte elements and store
//                    the difference between consecutive elements; helps smooth
//                    / monotonic numeric columns.
//   - zigzag_deltaW: as delta, but map the signed difference through zig-zag so
//                    small +/- changes become small unsigned values; helps both
//                    integer and floating-point columns.
//   - delta + shuffle : delta to expose small residuals, then transpose so the
//                    (mostly zero) high residual bytes form long runs. The
//                    64-bit variant (id 15) targets monotonic offset / index
//                    columns of 8-byte elements (RNTuple SplitIndex64-style).
//   - zigzag + shuffle : zig-zag each raw element value (NO delta), then
//                    transpose. For signed columns oscillating around zero
//                    (RNTuple SplitZigzagInt-style) the sign-extension bytes
//                    (0x00/0xFF) fold into near-zero high bytes that the
//                    transpose turns into long runs; a delta would only
//                    inflate the residuals on such data.
//   - bitshuffle32/64 : transpose the bit matrix of 4- or 8-byte elements (all
//                    bit-0 of every element, then all bit-1, ...). Excellent
//                    when only a few low bits actually vary.
//   - delta32 + bitshuffle32 : element delta first, so a counter's residual
//                    occupies a few low bits, then the bit-plane transpose.
//   - dictW        : replace each W-byte element (W = 2, 4 or 8) with a dense
//                    id into the block's sorted set of distinct values
//                    (<= 64K entries) and code the id byte planes + a
//                    delta-varint dictionary. The strongest transform for
//                    low-cardinality numeric columns (a few hundred distinct
//                    values per block).
//   - forW         : frame-of-reference. Subtract the block minimum of W-byte
//                    integers and bit-pack the residuals. A few outliers are
//                    stored raw beside a tighter frame so one spike does not
//                    widen every value. For columns that are almost unique
//                    but sit in a narrow window.
//   - alpW         : decimal integerization of f32/f64. Find a power of ten
//                    that maps most values onto integers, then frame-of-
//                    reference pack those integers. Values that do not
//                    round-trip are stored raw.
//
// dict, for and alp are NOT size-preserving: the filtered stream is
// self-framed (dict_filter.hpp / pack_filter.hpp) and the block header
// records its length, mirroring the byte-RLE intermediate-length varint.
// They never go through apply_filter / invert_filter.
//
// A cheap, data-driven detector picks the best filter (or none) per block, so
// the transform is only applied when it is expected to help. The whole feature
// is opt-in (CompressParams::filters / `spectra8 --filter`).
#ifndef SPECTRA8_SRC_TRANSFORM_HPP
#define SPECTRA8_SRC_TRANSFORM_HPP

#include <bit>
#include <cmath>
#include <vector>

#include "util/common.hpp"

#if defined(SPECTRA8_USE_SSE2NEON) && (defined(__aarch64__) || defined(__arm__))
// Test/compat hook: route the SSE2 kernels through sse2neon on arm so the
// x86 code path can be exercised on an arm host.
#include "sse2neon.h"
#define SPECTRA8_TRANSFORM_SSE2 1
#elif defined(__aarch64__)
#include <arm_neon.h>
#define SPECTRA8_TRANSFORM_NEON 1
#elif defined(__SSE2__) || defined(_M_X64)
#include <emmintrin.h>
#define SPECTRA8_TRANSFORM_SSE2 1
#endif

namespace spectra8::detail {

enum class Filter : u8 {
  None = 0,
  Shuffle16 = 1,
  Shuffle32 = 2,
  Shuffle64 = 3,
  Delta16 = 4,
  Delta32 = 5,
  Delta64 = 6,
  ZigzagDelta16 = 7,
  ZigzagDelta32 = 8,
  ZigzagDelta64 = 9,
  DeltaShuffle32 = 10,
  ZigzagDeltaShuffle32 = 11,
  BitShuffle32 = 12,
  Dict32 = 13,
  Dict64 = 14,
  DeltaShuffle64 = 15,
  ZigzagShuffle32 = 16,
  ZigzagShuffle64 = 17,
  BitShuffle64 = 18,
  DeltaBitShuffle32 = 19,
  Dict16 = 20,
  For16 = 21,
  For32 = 22,
  For64 = 23,
  Alp32 = 24,
  Alp64 = 25,
  XorShuffle32 = 26,
  XorShuffle64 = 27,
  Delta2Shuffle32 = 28,
  ZigzagDelta2Shuffle32 = 29,
  Delta2Shuffle64 = 30,
  BcjX86 = 31,
  DigitNorm = 32,
  WordNorm = 33,
  Base64 = 34,
  BitRound32 = 35,
  BitRound64 = 36,
  // Lag-2 residual (x[i] - x[i-2], or XOR) then byte transpose. SPDP's first
  // stage: a lag rather than a derivative, so mantissa noise is not doubled.
  // Targets interleaved pairs (x/y, re/im, two sensors) whose lag-1 residual
  // alternates between two far-apart values.
  Lag2Shuffle32 = 37,
  Lag2Shuffle64 = 38,
  XorLag2Shuffle32 = 39,
  // AArch64 BL / ADRP target normalizer (xz --arm64 transform).
  BcjArm64 = 40,
  kMaxValue = BcjArm64
};

// Dictionary filters carry a self-framed, variable-length representation (see
// dict_filter.hpp) and bypass apply_filter / invert_filter entirely.
inline bool filter_is_dict(Filter f) noexcept {
  return f == Filter::Dict16 || f == Filter::Dict32 || f == Filter::Dict64;
}

inline int dict_filter_width(Filter f) noexcept {
  if (f == Filter::Dict16) return 2;
  if (f == Filter::Dict32) return 4;
  return 8;
}

inline Filter dict_filter_id(int w) noexcept {
  if (w == 2) return Filter::Dict16;
  if (w == 4) return Filter::Dict32;
  return Filter::Dict64;
}

// Frame-of-reference and decimal-integerization streams are self-framed too
// (see pack_filter.hpp): variable length, no stacked byte-RLE.
inline bool filter_is_for(Filter f) noexcept {
  return f == Filter::For16 || f == Filter::For32 || f == Filter::For64;
}

inline bool filter_is_alp(Filter f) noexcept {
  return f == Filter::Alp32 || f == Filter::Alp64;
}

inline bool filter_is_digit_norm(Filter f) noexcept {
  return f == Filter::DigitNorm;
}

inline bool filter_is_word_norm(Filter f) noexcept {
  return f == Filter::WordNorm;
}

inline bool filter_is_base64(Filter f) noexcept {
  return f == Filter::Base64;
}

inline bool filter_is_framed(Filter f) noexcept {
  return filter_is_dict(f) || filter_is_for(f) || filter_is_alp(f) ||
         filter_is_digit_norm(f) || filter_is_word_norm(f) || filter_is_base64(f);
}

// A mask with exactly one FilterId bit (None/bit0 ignored) means the caller
// is forcing that transform. Auto-detect must not entropy-reject it.
inline bool singleton_forced_filter(u64 mask, Filter& out) noexcept {
  const u64 bits = mask & ~u64{1};
  if (bits == 0 || (bits & (bits - 1)) != 0) return false;
  const unsigned id = static_cast<unsigned>(__builtin_ctzll(bits));
  if (id == 0 || id > static_cast<unsigned>(Filter::kMaxValue)) return false;
  out = static_cast<Filter>(id);
  return true;
}

inline int framed_filter_width(Filter f) noexcept {
  switch (f) {
    case Filter::Dict16:
    case Filter::For16: return 2;
    case Filter::Dict32:
    case Filter::For32:
    case Filter::Alp32: return 4;
    case Filter::Dict64:
    case Filter::For64:
    case Filter::Alp64: return 8;
    default: return 0;
  }
}

// Bit 7 of the on-disk filter byte means the LZ/BWT payload decodes to a
// byte-RLE stream (see byte_rle.hpp) that expands to the filtered block before
// the inverse transform.
inline constexpr u8 kFilterByteRle = 0x80;

inline bool filter_is_valid(u8 f) noexcept {
  return (f & ~kFilterByteRle) <= static_cast<u8>(Filter::kMaxValue);
}

inline Filter filter_id_from_byte(u8 fb) noexcept {
  return static_cast<Filter>(fb & ~kFilterByteRle);
}

inline bool filter_byte_has_rle(u8 fb) noexcept {
  return (fb & kFilterByteRle) != 0;
}

inline u8 pack_filter_byte(Filter f, bool rle) noexcept {
  return static_cast<u8>(f) | (rle ? kFilterByteRle : 0);
}

// --- little-endian element load/store (w = 2/4/8) ---------------------------

inline u64 load_le(const u8* p, int w) noexcept {
  switch (w) {
    case 2: return static_cast<u64>(p[0]) | (static_cast<u64>(p[1]) << 8);
    case 4: return static_cast<u64>(load_u32(p));
    case 8: return load_u64(p);
    default: return 0;
  }
}
inline void store_le(u8* p, u64 v, int w) noexcept {
  switch (w) {
    case 2:
      p[0] = static_cast<u8>(v);
      p[1] = static_cast<u8>(v >> 8);
      break;
    case 4: store_u32(p, static_cast<u32>(v)); break;
    case 8: store_u64(p, v); break;
    default: break;
  }
}

inline u64 width_mask(int w) noexcept {
  const int bits = w * 8;
  return (bits >= 64) ? ~u64{0} : ((u64{1} << bits) - 1);
}

// --- byte transpose ("shuffle"), W-byte elements ----------------------------
//
// The strided scalar loops were the top decode-side hotspot on x86 (perf on a
// Zen 3 EPYC showed invert_filter at ~11% of a whole verdict run; Apple's
// wider out-of-order core hides the striding). The bulk of the buffer is
// handled by 16-element SIMD transpose kernels (NEON de/interleaving loads on
// arm64, SSE2 pack/unpack on x86-64 - both in the baseline ISA, no -march
// needed); a shared scalar loop finishes the sub-16 remainder.

inline void shuffle_fwd(const u8* in, usize n, u8* out, int w) noexcept {
  const usize cnt = n / static_cast<usize>(w);
  usize e0 = 0;
#if defined(SPECTRA8_TRANSFORM_NEON)
  if (w == 2) {
    for (; e0 + 16 <= cnt; e0 += 16) {
      const uint8x16x2_t v = vld2q_u8(in + 2 * e0);
      vst1q_u8(out + e0, v.val[0]);
      vst1q_u8(out + cnt + e0, v.val[1]);
    }
  } else if (w == 4) {
    for (; e0 + 16 <= cnt; e0 += 16) {
      const uint8x16x4_t v = vld4q_u8(in + 4 * e0);
      for (int p = 0; p < 4; ++p)
        vst1q_u8(out + static_cast<usize>(p) * cnt + e0, v.val[p]);
    }
  } else if (w == 8) {
    // vld4q on each 64-byte half yields lanes (b_j, b_{j+4}) alternating per
    // element; one uzp per pair separates them into the two byte planes.
    for (; e0 + 16 <= cnt; e0 += 16) {
      const uint8x16x4_t a = vld4q_u8(in + 8 * e0);
      const uint8x16x4_t b = vld4q_u8(in + 8 * e0 + 64);
      for (int j = 0; j < 4; ++j) {
        const uint8x16x2_t uz = vuzpq_u8(a.val[j], b.val[j]);
        vst1q_u8(out + static_cast<usize>(j) * cnt + e0, uz.val[0]);
        vst1q_u8(out + static_cast<usize>(j + 4) * cnt + e0, uz.val[1]);
      }
    }
  }
#elif defined(SPECTRA8_TRANSFORM_SSE2)
  // Stride-2 byte deinterleave via mask+pack; applied log2(w) times.
  const __m128i m8 = _mm_set1_epi16(0x00FF);
  const auto ld = [](const u8* p) {
    return _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
  };
  const auto st = [](u8* p, __m128i v) {
    _mm_storeu_si128(reinterpret_cast<__m128i*>(p), v);
  };
  const auto pe = [m8](__m128i a, __m128i b) {  // even-index bytes of a:b
    return _mm_packus_epi16(_mm_and_si128(a, m8), _mm_and_si128(b, m8));
  };
  const auto po = [](__m128i a, __m128i b) {  // odd-index bytes of a:b
    return _mm_packus_epi16(_mm_srli_epi16(a, 8), _mm_srli_epi16(b, 8));
  };
  if (w == 2) {
    for (; e0 + 16 <= cnt; e0 += 16) {
      const __m128i r0 = ld(in + 2 * e0);
      const __m128i r1 = ld(in + 2 * e0 + 16);
      st(out + e0, pe(r0, r1));
      st(out + cnt + e0, po(r0, r1));
    }
  } else if (w == 4) {
    for (; e0 + 16 <= cnt; e0 += 16) {
      const __m128i r0 = ld(in + 4 * e0);
      const __m128i r1 = ld(in + 4 * e0 + 16);
      const __m128i r2 = ld(in + 4 * e0 + 32);
      const __m128i r3 = ld(in + 4 * e0 + 48);
      const __m128i ev0 = pe(r0, r1), ev1 = pe(r2, r3);  // b0 b2 ...
      const __m128i od0 = po(r0, r1), od1 = po(r2, r3);  // b1 b3 ...
      st(out + e0, pe(ev0, ev1));
      st(out + cnt + e0, pe(od0, od1));
      st(out + 2 * cnt + e0, po(ev0, ev1));
      st(out + 3 * cnt + e0, po(od0, od1));
    }
  } else if (w == 8) {
    for (; e0 + 16 <= cnt; e0 += 16) {
      const u8* s = in + 8 * e0;
      const __m128i e0v = pe(ld(s), ld(s + 16));         // b0 b2 b4 b6 ...
      const __m128i e1v = pe(ld(s + 32), ld(s + 48));
      const __m128i e2v = pe(ld(s + 64), ld(s + 80));
      const __m128i e3v = pe(ld(s + 96), ld(s + 112));
      const __m128i o0v = po(ld(s), ld(s + 16));         // b1 b3 b5 b7 ...
      const __m128i o1v = po(ld(s + 32), ld(s + 48));
      const __m128i o2v = po(ld(s + 64), ld(s + 80));
      const __m128i o3v = po(ld(s + 96), ld(s + 112));
      const __m128i ee0 = pe(e0v, e1v), ee1 = pe(e2v, e3v);  // b0 b4 ...
      const __m128i eo0 = po(e0v, e1v), eo1 = po(e2v, e3v);  // b2 b6 ...
      const __m128i oe0 = pe(o0v, o1v), oe1 = pe(o2v, o3v);  // b1 b5 ...
      const __m128i oo0 = po(o0v, o1v), oo1 = po(o2v, o3v);  // b3 b7 ...
      st(out + e0, pe(ee0, ee1));
      st(out + cnt + e0, pe(oe0, oe1));
      st(out + 2 * cnt + e0, pe(eo0, eo1));
      st(out + 3 * cnt + e0, pe(oo0, oo1));
      st(out + 4 * cnt + e0, po(ee0, ee1));
      st(out + 5 * cnt + e0, po(oe0, oe1));
      st(out + 6 * cnt + e0, po(eo0, eo1));
      st(out + 7 * cnt + e0, po(oo0, oo1));
    }
  }
#endif
  for (int p = 0; p < w; ++p) {
    u8* dp = out + static_cast<usize>(p) * cnt;
    for (usize e = e0; e < cnt; ++e) dp[e] = in[e * static_cast<usize>(w) + p];
  }
  const usize tail = cnt * static_cast<usize>(w);
  for (usize i = tail; i < n; ++i) out[i] = in[i];  // leftover bytes verbatim
}

inline void shuffle_inv(const u8* in, usize n, u8* out, int w) noexcept {
  const usize cnt = n / static_cast<usize>(w);
  usize e0 = 0;
#if defined(SPECTRA8_TRANSFORM_NEON)
  if (w == 2) {
    for (; e0 + 16 <= cnt; e0 += 16) {
      uint8x16x2_t v;
      v.val[0] = vld1q_u8(in + e0);
      v.val[1] = vld1q_u8(in + cnt + e0);
      vst2q_u8(out + 2 * e0, v);
    }
  } else if (w == 4) {
    for (; e0 + 16 <= cnt; e0 += 16) {
      uint8x16x4_t v;
      for (int p = 0; p < 4; ++p)
        v.val[p] = vld1q_u8(in + static_cast<usize>(p) * cnt + e0);
      vst4q_u8(out + 4 * e0, v);
    }
  } else if (w == 8) {
    for (; e0 + 16 <= cnt; e0 += 16) {
      uint8x16x4_t c, d;
      for (int j = 0; j < 4; ++j) {
        const uint8x16x2_t z =
            vzipq_u8(vld1q_u8(in + static_cast<usize>(j) * cnt + e0),
                     vld1q_u8(in + static_cast<usize>(j + 4) * cnt + e0));
        c.val[j] = z.val[0];
        d.val[j] = z.val[1];
      }
      vst4q_u8(out + 8 * e0, c);
      vst4q_u8(out + 8 * e0 + 64, d);
    }
  }
#elif defined(SPECTRA8_TRANSFORM_SSE2)
  const auto ld = [](const u8* p) {
    return _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
  };
  const auto st = [](u8* p, __m128i v) {
    _mm_storeu_si128(reinterpret_cast<__m128i*>(p), v);
  };
  if (w == 2) {
    for (; e0 + 16 <= cnt; e0 += 16) {
      const __m128i a = ld(in + e0);
      const __m128i b = ld(in + cnt + e0);
      st(out + 2 * e0, _mm_unpacklo_epi8(a, b));
      st(out + 2 * e0 + 16, _mm_unpackhi_epi8(a, b));
    }
  } else if (w == 4) {
    for (; e0 + 16 <= cnt; e0 += 16) {
      const __m128i p0 = ld(in + e0);
      const __m128i p1 = ld(in + cnt + e0);
      const __m128i p2 = ld(in + 2 * cnt + e0);
      const __m128i p3 = ld(in + 3 * cnt + e0);
      const __m128i t0 = _mm_unpacklo_epi8(p0, p1);  // b0b1 e0..7
      const __m128i t1 = _mm_unpackhi_epi8(p0, p1);  // b0b1 e8..15
      const __m128i t2 = _mm_unpacklo_epi8(p2, p3);  // b2b3 e0..7
      const __m128i t3 = _mm_unpackhi_epi8(p2, p3);
      st(out + 4 * e0, _mm_unpacklo_epi16(t0, t2));
      st(out + 4 * e0 + 16, _mm_unpackhi_epi16(t0, t2));
      st(out + 4 * e0 + 32, _mm_unpacklo_epi16(t1, t3));
      st(out + 4 * e0 + 48, _mm_unpackhi_epi16(t1, t3));
    }
  } else if (w == 8) {
    for (; e0 + 16 <= cnt; e0 += 16) {
      __m128i p[8];
      for (int j = 0; j < 8; ++j)
        p[j] = ld(in + static_cast<usize>(j) * cnt + e0);
      const __m128i a0 = _mm_unpacklo_epi8(p[0], p[1]);  // b0b1 e0..7
      const __m128i a1 = _mm_unpackhi_epi8(p[0], p[1]);  // b0b1 e8..15
      const __m128i a2 = _mm_unpacklo_epi8(p[2], p[3]);
      const __m128i a3 = _mm_unpackhi_epi8(p[2], p[3]);
      const __m128i a4 = _mm_unpacklo_epi8(p[4], p[5]);
      const __m128i a5 = _mm_unpackhi_epi8(p[4], p[5]);
      const __m128i a6 = _mm_unpacklo_epi8(p[6], p[7]);
      const __m128i a7 = _mm_unpackhi_epi8(p[6], p[7]);
      const __m128i u0 = _mm_unpacklo_epi16(a0, a2);  // b0..b3 e0..3
      const __m128i u1 = _mm_unpackhi_epi16(a0, a2);  // b0..b3 e4..7
      const __m128i u2 = _mm_unpacklo_epi16(a4, a6);  // b4..b7 e0..3
      const __m128i u3 = _mm_unpackhi_epi16(a4, a6);
      const __m128i u4 = _mm_unpacklo_epi16(a1, a3);  // b0..b3 e8..11
      const __m128i u5 = _mm_unpackhi_epi16(a1, a3);
      const __m128i u6 = _mm_unpacklo_epi16(a5, a7);
      const __m128i u7 = _mm_unpackhi_epi16(a5, a7);
      u8* d = out + 8 * e0;
      st(d, _mm_unpacklo_epi32(u0, u2));        // e0 e1
      st(d + 16, _mm_unpackhi_epi32(u0, u2));   // e2 e3
      st(d + 32, _mm_unpacklo_epi32(u1, u3));   // e4 e5
      st(d + 48, _mm_unpackhi_epi32(u1, u3));
      st(d + 64, _mm_unpacklo_epi32(u4, u6));   // e8 e9
      st(d + 80, _mm_unpackhi_epi32(u4, u6));
      st(d + 96, _mm_unpacklo_epi32(u5, u7));
      st(d + 112, _mm_unpackhi_epi32(u5, u7));
    }
  }
#endif
  for (int p = 0; p < w; ++p) {
    const u8* sp = in + static_cast<usize>(p) * cnt;
    for (usize e = e0; e < cnt; ++e) out[e * static_cast<usize>(w) + p] = sp[e];
  }
  const usize tail = cnt * static_cast<usize>(w);
  for (usize i = tail; i < n; ++i) out[i] = in[i];
}

// --- element delta (no zig-zag), W-byte LE elements -------------------------

inline void delta_fwd(const u8* in, usize n, u8* out, int w) noexcept {
  const usize cnt = n / static_cast<usize>(w);
  if (w == 4) {
    usize e = 0;
    u32 prev = 0;
#if defined(SPECTRA8_TRANSFORM_NEON)
    for (; e + 4 <= cnt; e += 4) {
      uint32x4_t cur = vld1q_u32(reinterpret_cast<const u32*>(in + 4 * e));
      uint32x4_t pvec = vextq_u32(vsetq_lane_u32(prev, vdupq_n_u32(0), 3), cur, 3);
      vst1q_u32(reinterpret_cast<u32*>(out + 4 * e), vsubq_u32(cur, pvec));
      prev = vgetq_lane_u32(cur, 3);
    }
#elif defined(SPECTRA8_TRANSFORM_SSE2)
    for (; e + 4 <= cnt; e += 4) {
      __m128i cur = _mm_loadu_si128(reinterpret_cast<const __m128i*>(in + 4 * e));
      __m128i pvec = _mm_or_si128(_mm_slli_si128(cur, 4), _mm_cvtsi32_si128(static_cast<int>(prev)));
      _mm_storeu_si128(reinterpret_cast<__m128i*>(out + 4 * e), _mm_sub_epi32(cur, pvec));
      prev = load_u32(in + 4 * (e + 3));
    }
#endif
    for (; e < cnt; ++e) {
      const u32 v = load_u32(in + 4 * e);
      store_u32(out + 4 * e, v - prev);
      prev = v;
    }
  } else if (w == 2) {
    usize e = 0;
    u16 prev = 0;
#if defined(SPECTRA8_TRANSFORM_SSE2)
    for (; e + 8 <= cnt; e += 8) {
      __m128i cur = _mm_loadu_si128(reinterpret_cast<const __m128i*>(in + 2 * e));
      __m128i pvec = _mm_or_si128(_mm_slli_si128(cur, 2), _mm_cvtsi32_si128(static_cast<int>(prev)));
      _mm_storeu_si128(reinterpret_cast<__m128i*>(out + 2 * e), _mm_sub_epi16(cur, pvec));
      prev = static_cast<u16>(in[2 * (e + 7)] | (in[2 * (e + 7) + 1] << 8));
    }
#endif
    for (; e < cnt; ++e) {
      const u16 v = static_cast<u16>(in[2 * e] | (in[2 * e + 1] << 8));
      const u16 d = static_cast<u16>(v - prev);
      out[2 * e] = static_cast<u8>(d);
      out[2 * e + 1] = static_cast<u8>(d >> 8);
      prev = v;
    }
  } else if (w == 8) {
    u64 prev = 0;
    for (usize e = 0; e < cnt; ++e) {
      const u64 v = load_u64(in + 8 * e);
      store_u64(out + 8 * e, v - prev);
      prev = v;
    }
  } else {
    const u64 mask = width_mask(w);
    u64 prev = 0;
    for (usize e = 0; e < cnt; ++e) {
      const u64 v = load_le(in + e * static_cast<usize>(w), w);
      store_le(out + e * static_cast<usize>(w), (v - prev) & mask, w);
      prev = v;
    }
  }
  const usize tail = cnt * static_cast<usize>(w);
  for (usize i = tail; i < n; ++i) out[i] = in[i];
}
inline void delta_inv(const u8* in, usize n, u8* out, int w) noexcept {
  const usize cnt = n / static_cast<usize>(w);
  if (w == 4) {
    usize e = 0;
    u32 prev = 0;
#if defined(SPECTRA8_TRANSFORM_NEON)
    // In-register prefix sum (log-step shifted adds) of each vector, then
    // the running carry. Only `carry += last lane` is loop-carried (one
    // vector add per 16 bytes); the lane broadcast is off that chain. The
    // scalar loop is one dependent add per element.
    const uint32x4_t z4 = vdupq_n_u32(0);
    uint32x4_t carry = z4;
    for (; e + 4 <= cnt; e += 4) {
      uint32x4_t d = vld1q_u32(reinterpret_cast<const u32*>(in + 4 * e));
      d = vaddq_u32(d, vextq_u32(z4, d, 3));
      d = vaddq_u32(d, vextq_u32(z4, d, 2));
      const uint32x4_t last = vdupq_laneq_u32(d, 3);
      vst1q_u32(reinterpret_cast<u32*>(out + 4 * e), vaddq_u32(d, carry));
      carry = vaddq_u32(carry, last);
    }
    prev = vgetq_lane_u32(carry, 0);
#elif defined(SPECTRA8_TRANSFORM_SSE2)
    // Same structure as the NEON kernel: the loop-carried work is one
    // vector add (carry += broadcast last lane), not a lane extract +
    // re-broadcast through a GPR.
    __m128i carry = _mm_setzero_si128();
    for (; e + 4 <= cnt; e += 4) {
      __m128i d = _mm_loadu_si128(reinterpret_cast<const __m128i*>(in + 4 * e));
      d = _mm_add_epi32(d, _mm_slli_si128(d, 4));
      d = _mm_add_epi32(d, _mm_slli_si128(d, 8));
      const __m128i last = _mm_shuffle_epi32(d, 0xFF);
      _mm_storeu_si128(reinterpret_cast<__m128i*>(out + 4 * e), _mm_add_epi32(d, carry));
      carry = _mm_add_epi32(carry, last);
    }
    prev = static_cast<u32>(_mm_cvtsi128_si32(carry));
#endif
    for (; e < cnt; ++e) {
      const u32 d = load_u32(in + 4 * e);
      const u32 v = prev + d;
      store_u32(out + 4 * e, v);
      prev = v;
    }
  } else if (w == 2) {
    usize e = 0;
    u16 prev = 0;
#if defined(SPECTRA8_TRANSFORM_NEON)
    const uint16x8_t z8 = vdupq_n_u16(0);
    uint16x8_t carry = z8;
    for (; e + 8 <= cnt; e += 8) {
      uint16x8_t d = vld1q_u16(reinterpret_cast<const u16*>(in + 2 * e));
      d = vaddq_u16(d, vextq_u16(z8, d, 7));
      d = vaddq_u16(d, vextq_u16(z8, d, 6));
      d = vaddq_u16(d, vextq_u16(z8, d, 4));
      const uint16x8_t last = vdupq_laneq_u16(d, 7);
      vst1q_u16(reinterpret_cast<u16*>(out + 2 * e), vaddq_u16(d, carry));
      carry = vaddq_u16(carry, last);
    }
    prev = vgetq_lane_u16(carry, 0);
#elif defined(SPECTRA8_TRANSFORM_SSE2)
    __m128i carry = _mm_setzero_si128();
    for (; e + 8 <= cnt; e += 8) {
      __m128i d = _mm_loadu_si128(reinterpret_cast<const __m128i*>(in + 2 * e));
      d = _mm_add_epi16(d, _mm_slli_si128(d, 2));
      d = _mm_add_epi16(d, _mm_slli_si128(d, 4));
      d = _mm_add_epi16(d, _mm_slli_si128(d, 8));
      const __m128i last = _mm_shuffle_epi32(_mm_shufflehi_epi16(d, 0xFF), 0xEE);
      _mm_storeu_si128(reinterpret_cast<__m128i*>(out + 2 * e), _mm_add_epi16(d, carry));
      carry = _mm_add_epi16(carry, last);
    }
    prev = static_cast<u16>(_mm_cvtsi128_si32(carry));
#endif
    for (; e < cnt; ++e) {
      const u16 d = static_cast<u16>(in[2 * e] | (in[2 * e + 1] << 8));
      const u16 v = static_cast<u16>(prev + d);
      out[2 * e] = static_cast<u8>(v);
      out[2 * e + 1] = static_cast<u8>(v >> 8);
      prev = v;
    }
  } else if (w == 8) {
    u64 prev = 0;
    for (usize e = 0; e < cnt; ++e) {
      const u64 d = load_u64(in + 8 * e);
      const u64 v = prev + d;
      store_u64(out + 8 * e, v);
      prev = v;
    }
  } else {
    const u64 mask = width_mask(w);
    u64 prev = 0;
    for (usize e = 0; e < cnt; ++e) {
      const u64 d = load_le(in + e * static_cast<usize>(w), w);
      const u64 v = (prev + d) & mask;
      store_le(out + e * static_cast<usize>(w), v, w);
      prev = v;
    }
  }
  const usize tail = cnt * static_cast<usize>(w);
  for (usize i = tail; i < n; ++i) out[i] = in[i];
}

// --- zig-zag delta, W-byte LE elements --------------------------------------

inline void zigzag_delta_fwd(const u8* in, usize n, u8* out, int w) noexcept {
  const int bits = w * 8;
  const u64 mask = width_mask(w);
  const usize cnt = n / static_cast<usize>(w);
  u64 prev = 0;
  for (usize e = 0; e < cnt; ++e) {
    const u64 v = load_le(in + e * static_cast<usize>(w), w);
    const u64 d = (v - prev) & mask;
    const u64 zz = ((d << 1) ^ (u64{0} - (d >> (bits - 1)))) & mask;  // zig-zag
    store_le(out + e * static_cast<usize>(w), zz, w);
    prev = v;
  }
  const usize tail = cnt * static_cast<usize>(w);
  for (usize i = tail; i < n; ++i) out[i] = in[i];
}
inline void zigzag_delta_inv(const u8* in, usize n, u8* out, int w) noexcept {
  const u64 mask = width_mask(w);
  const usize cnt = n / static_cast<usize>(w);
  u64 prev = 0;
  for (usize e = 0; e < cnt; ++e) {
    const u64 zz = load_le(in + e * static_cast<usize>(w), w);
    const u64 d = (zz >> 1) ^ (u64{0} - (zz & 1));  // un-zig-zag
    const u64 v = (prev + d) & mask;
    store_le(out + e * static_cast<usize>(w), v, w);
    prev = v;
  }
  const usize tail = cnt * static_cast<usize>(w);
  for (usize i = tail; i < n; ++i) out[i] = in[i];
}

// --- zig-zag of raw values (no delta), W-byte LE elements --------------------
//
// Maps each signed element through zig-zag ((v << 1) ^ (v >> (bits-1)),
// arithmetic shift) so values oscillating around zero become small unsigned
// values. Unlike zigzag_delta above, consecutive elements stay independent.

inline void zigzag_fwd(const u8* in, usize n, u8* out, int w) noexcept {
  const int bits = w * 8;
  const u64 mask = width_mask(w);
  const usize cnt = n / static_cast<usize>(w);
  for (usize e = 0; e < cnt; ++e) {
    const u64 v = load_le(in + e * static_cast<usize>(w), w);
    const u64 zz = ((v << 1) ^ (u64{0} - (v >> (bits - 1)))) & mask;
    store_le(out + e * static_cast<usize>(w), zz, w);
  }
  const usize tail = cnt * static_cast<usize>(w);
  for (usize i = tail; i < n; ++i) out[i] = in[i];
}
inline void zigzag_inv(const u8* in, usize n, u8* out, int w) noexcept {
  const u64 mask = width_mask(w);
  const usize cnt = n / static_cast<usize>(w);
  for (usize e = 0; e < cnt; ++e) {
    const u64 zz = load_le(in + e * static_cast<usize>(w), w);
    const u64 v = ((zz >> 1) ^ (u64{0} - (zz & 1))) & mask;
    store_le(out + e * static_cast<usize>(w), v, w);
  }
  const usize tail = cnt * static_cast<usize>(w);
  for (usize i = tail; i < n; ++i) out[i] = in[i];
}

// --- XOR delta, W-byte LE elements ------------------------------------------

inline void xor_delta_fwd(const u8* in, usize n, u8* out, int w) noexcept {
  const usize cnt = n / static_cast<usize>(w);
  if (w == 4) {
    usize e = 0;
    u32 prev = 0;
#if defined(SPECTRA8_TRANSFORM_NEON)
    for (; e + 4 <= cnt; e += 4) {
      uint32x4_t cur = vld1q_u32(reinterpret_cast<const u32*>(in + 4 * e));
      uint32x4_t pvec = vextq_u32(vsetq_lane_u32(prev, vdupq_n_u32(0), 3), cur, 3);
      vst1q_u32(reinterpret_cast<u32*>(out + 4 * e), veorq_u32(cur, pvec));
      prev = vgetq_lane_u32(cur, 3);
    }
#elif defined(SPECTRA8_TRANSFORM_SSE2)
    for (; e + 4 <= cnt; e += 4) {
      __m128i cur = _mm_loadu_si128(reinterpret_cast<const __m128i*>(in + 4 * e));
      __m128i pvec = _mm_or_si128(_mm_slli_si128(cur, 4), _mm_cvtsi32_si128(static_cast<int>(prev)));
      _mm_storeu_si128(reinterpret_cast<__m128i*>(out + 4 * e), _mm_xor_si128(cur, pvec));
      prev = load_u32(in + 4 * (e + 3));
    }
#endif
    for (; e < cnt; ++e) {
      const u32 v = load_u32(in + 4 * e);
      store_u32(out + 4 * e, v ^ prev);
      prev = v;
    }
  } else if (w == 8) {
    u64 prev = 0;
    for (usize e = 0; e < cnt; ++e) {
      const u64 v = load_u64(in + 8 * e);
      store_u64(out + 8 * e, v ^ prev);
      prev = v;
    }
  } else {
    const u64 mask = width_mask(w);
    u64 prev = 0;
    for (usize e = 0; e < cnt; ++e) {
      const u64 v = load_le(in + e * static_cast<usize>(w), w);
      store_le(out + e * static_cast<usize>(w), (v ^ prev) & mask, w);
      prev = v;
    }
  }
  const usize tail = cnt * static_cast<usize>(w);
  for (usize i = tail; i < n; ++i) out[i] = in[i];
}

inline void xor_delta_inv(const u8* in, usize n, u8* out, int w) noexcept {
  const usize cnt = n / static_cast<usize>(w);
  if (w == 4) {
    usize e = 0;
    u32 prev = 0;
#if defined(SPECTRA8_TRANSFORM_NEON)
    for (; e + 4 <= cnt; e += 4) {
      uint32x4_t d = vld1q_u32(reinterpret_cast<const u32*>(in + 4 * e));
      d = veorq_u32(d, vextq_u32(vdupq_n_u32(0), d, 3));
      d = veorq_u32(d, vextq_u32(vdupq_n_u32(0), d, 2));
      d = veorq_u32(d, vdupq_n_u32(prev));
      vst1q_u32(reinterpret_cast<u32*>(out + 4 * e), d);
      prev = vgetq_lane_u32(d, 3);
    }
#elif defined(SPECTRA8_TRANSFORM_SSE2)
    for (; e + 4 <= cnt; e += 4) {
      __m128i d = _mm_loadu_si128(reinterpret_cast<const __m128i*>(in + 4 * e));
      d = _mm_xor_si128(d, _mm_slli_si128(d, 4));
      d = _mm_xor_si128(d, _mm_slli_si128(d, 8));
      d = _mm_xor_si128(d, _mm_set1_epi32(static_cast<int>(prev)));
      _mm_storeu_si128(reinterpret_cast<__m128i*>(out + 4 * e), d);
      prev = static_cast<u32>(_mm_cvtsi128_si32(_mm_srli_si128(d, 12)));
    }
#endif
    for (; e < cnt; ++e) {
      const u32 d = load_u32(in + 4 * e);
      const u32 v = prev ^ d;
      store_u32(out + 4 * e, v);
      prev = v;
    }
  } else if (w == 8) {
    u64 prev = 0;
    for (usize e = 0; e < cnt; ++e) {
      const u64 d = load_u64(in + 8 * e);
      const u64 v = prev ^ d;
      store_u64(out + 8 * e, v);
      prev = v;
    }
  } else {
    const u64 mask = width_mask(w);
    u64 prev = 0;
    for (usize e = 0; e < cnt; ++e) {
      const u64 d = load_le(in + e * static_cast<usize>(w), w);
      const u64 v = (prev ^ d) & mask;
      store_le(out + e * static_cast<usize>(w), v, w);
      prev = v;
    }
  }
  const usize tail = cnt * static_cast<usize>(w);
  for (usize i = tail; i < n; ++i) out[i] = in[i];
}

// --- second-order delta (no zig-zag), W-byte LE elements -------------------

inline void delta2_fwd(const u8* in, usize n, u8* out, int w) noexcept {
  const usize cnt = n / static_cast<usize>(w);
  if (w == 4) {
    usize e = 0;
#if defined(SPECTRA8_TRANSFORM_NEON)
    uint32x4_t prev_v = vdupq_n_u32(0);
    for (; e + 4 <= cnt; e += 4) {
      uint32x4_t cur = vld1q_u32(reinterpret_cast<const u32*>(in + 4 * e));
      uint32x4_t prev1 = vextq_u32(prev_v, cur, 3);
      uint32x4_t prev2 = vextq_u32(prev_v, cur, 2);
      uint32x4_t pred = vsubq_u32(vshlq_n_u32(prev1, 1), prev2);
      vst1q_u32(reinterpret_cast<u32*>(out + 4 * e), vsubq_u32(cur, pred));
      prev_v = cur;
    }
#elif defined(SPECTRA8_TRANSFORM_SSE2)
    __m128i prev_v = _mm_setzero_si128();
    for (; e + 4 <= cnt; e += 4) {
      __m128i cur = _mm_loadu_si128(reinterpret_cast<const __m128i*>(in + 4 * e));
      __m128i prev1 = _mm_or_si128(_mm_slli_si128(cur, 4), _mm_srli_si128(prev_v, 12));
      __m128i prev2 = _mm_or_si128(_mm_slli_si128(cur, 8), _mm_srli_si128(prev_v, 8));
      __m128i pred = _mm_sub_epi32(_mm_slli_epi32(prev1, 1), prev2);
      _mm_storeu_si128(reinterpret_cast<__m128i*>(out + 4 * e), _mm_sub_epi32(cur, pred));
      prev_v = cur;
    }
#endif
    u32 p1 = (e > 0) ? load_u32(in + 4 * (e - 1)) : 0;
    u32 p2 = (e > 1) ? load_u32(in + 4 * (e - 2)) : 0;
    for (; e < cnt; ++e) {
      const u32 v = load_u32(in + 4 * e);
      const u32 pred = 2u * p1 - p2;
      store_u32(out + 4 * e, v - pred);
      p2 = p1;
      p1 = v;
    }
  } else if (w == 8) {
    u64 p1 = 0, p2 = 0;
    for (usize e = 0; e < cnt; ++e) {
      const u64 v = load_u64(in + 8 * e);
      const u64 pred = 2u * p1 - p2;
      store_u64(out + 8 * e, v - pred);
      p2 = p1;
      p1 = v;
    }
  }
  const usize tail = cnt * static_cast<usize>(w);
  for (usize i = tail; i < n; ++i) out[i] = in[i];
}

inline void delta2_inv(const u8* in, usize n, u8* out, int w) noexcept {
  const usize cnt = n / static_cast<usize>(w);
  if (w == 4) {
    u32 p1 = 0, p2 = 0;
    for (usize e = 0; e < cnt; ++e) {
      const u32 r = load_u32(in + 4 * e);
      const u32 pred = 2u * p1 - p2;
      const u32 v = r + pred;
      store_u32(out + 4 * e, v);
      p2 = p1;
      p1 = v;
    }
  } else if (w == 8) {
    u64 p1 = 0, p2 = 0;
    for (usize e = 0; e < cnt; ++e) {
      const u64 r = load_u64(in + 8 * e);
      const u64 pred = 2u * p1 - p2;
      const u64 v = r + pred;
      store_u64(out + 8 * e, v);
      p2 = p1;
      p1 = v;
    }
  }
  const usize tail = cnt * static_cast<usize>(w);
  for (usize i = tail; i < n; ++i) out[i] = in[i];
}

// --- second-order delta with zig-zag, 4-byte LE elements -------------------

inline void zigzag_delta2_fwd(const u8* in, usize n, u8* out, int w) noexcept {
  const usize cnt = n / static_cast<usize>(w);
  if (w == 4) {
    usize e = 0;
#if defined(SPECTRA8_TRANSFORM_NEON)
    uint32x4_t prev_v = vdupq_n_u32(0);
    for (; e + 4 <= cnt; e += 4) {
      uint32x4_t cur = vld1q_u32(reinterpret_cast<const u32*>(in + 4 * e));
      uint32x4_t prev1 = vextq_u32(prev_v, cur, 3);
      uint32x4_t prev2 = vextq_u32(prev_v, cur, 2);
      uint32x4_t pred = vsubq_u32(vshlq_n_u32(prev1, 1), prev2);
      uint32x4_t d = vsubq_u32(cur, pred);
      uint32x4_t s1 = vshlq_n_u32(d, 1);
      uint32x4_t s2 = vreinterpretq_u32_s32(vshrq_n_s32(vreinterpretq_s32_u32(d), 31));
      uint32x4_t zz = veorq_u32(s1, s2);
      vst1q_u32(reinterpret_cast<u32*>(out + 4 * e), zz);
      prev_v = cur;
    }
#elif defined(SPECTRA8_TRANSFORM_SSE2)
    __m128i prev_v = _mm_setzero_si128();
    for (; e + 4 <= cnt; e += 4) {
      __m128i cur = _mm_loadu_si128(reinterpret_cast<const __m128i*>(in + 4 * e));
      __m128i prev1 = _mm_or_si128(_mm_slli_si128(cur, 4), _mm_srli_si128(prev_v, 12));
      __m128i prev2 = _mm_or_si128(_mm_slli_si128(cur, 8), _mm_srli_si128(prev_v, 8));
      __m128i pred = _mm_sub_epi32(_mm_slli_epi32(prev1, 1), prev2);
      __m128i d = _mm_sub_epi32(cur, pred);
      __m128i s1 = _mm_slli_epi32(d, 1);
      __m128i s2 = _mm_srai_epi32(d, 31);
      __m128i zz = _mm_xor_si128(s1, s2);
      _mm_storeu_si128(reinterpret_cast<__m128i*>(out + 4 * e), zz);
      prev_v = cur;
    }
#endif
    u32 p1 = (e > 0) ? load_u32(in + 4 * (e - 1)) : 0;
    u32 p2 = (e > 1) ? load_u32(in + 4 * (e - 2)) : 0;
    for (; e < cnt; ++e) {
      const u32 v = load_u32(in + 4 * e);
      const u32 pred = 2u * p1 - p2;
      const u32 d = v - pred;
      const u32 zz = (d << 1) ^ (0u - (d >> 31));
      store_u32(out + 4 * e, zz);
      p2 = p1;
      p1 = v;
    }
  }
  const usize tail = cnt * static_cast<usize>(w);
  for (usize i = tail; i < n; ++i) out[i] = in[i];
}

inline void zigzag_delta2_inv(const u8* in, usize n, u8* out, int w) noexcept {
  const usize cnt = n / static_cast<usize>(w);
  if (w == 4) {
    u32 p1 = 0, p2 = 0;
    for (usize e = 0; e < cnt; ++e) {
      const u32 zz = load_u32(in + 4 * e);
      const u32 d = (zz >> 1) ^ (0u - (zz & 1u));
      const u32 pred = 2u * p1 - p2;
      const u32 v = d + pred;
      store_u32(out + 4 * e, v);
      p2 = p1;
      p1 = v;
    }
  }
  const usize tail = cnt * static_cast<usize>(w);
  for (usize i = tail; i < n; ++i) out[i] = in[i];
}

// --- bit-plane transpose ("bitshuffle"), W-byte elements --------------------

// Per-thread scratch used by bitshuffle and the two-stage (delta-then-shuffle)
// filters.
inline std::vector<u8>& filter_scratch(usize n) {
  static thread_local std::vector<u8> s;
  s.resize(n);
  return s;
}

// Transpose the 8x8 bit matrix held in a u64 (byte k = row k, bit j = column
// j, both little-endian) with three delta swaps: bit (8k+j) moves to (8j+k).
// Self-inverse. ~15 ALU ops for 8 bytes, vs 8 test-and-set ops per *byte* in
// the naive loop - this is what makes the bitshuffle filter cheap enough to
// not dominate decode on top of the actual entropy decoder.
inline u64 bit_transpose8(u64 x) noexcept {
  u64 t = (x ^ (x >> 7)) & 0x00AA00AA00AA00AAull;
  x ^= t ^ (t << 7);
  t = (x ^ (x >> 14)) & 0x0000CCCC0000CCCCull;
  x ^= t ^ (t << 14);
  t = (x ^ (x >> 28)) & 0x00000000F0F0F0F0ull;
  x ^= t ^ (t << 28);
  return x;
}

// bitshuffle == byte transpose (shuffle) + an 8x8 bit transpose inside each
// byte plane: output bit plane b is bit (b&7) of byte plane (b>>3). The fast
// path needs the element count to be a multiple of 8 so bit planes start on
// byte boundaries (any power-of-two block satisfies this); otherwise the
// bit-exact scalar layout loop below is used.

inline void bitshuffle_fwd(const u8* in, usize n, u8* out, int w) noexcept {
  const usize cnt = n / static_cast<usize>(w);
  const usize region = cnt * static_cast<usize>(w);
  if (cnt >= 8 && (cnt & 7) == 0) {
    u8* planes = filter_scratch(region).data();
    shuffle_fwd(in, region, planes, w);
    const usize q = cnt >> 3;  // bytes per bit plane
    for (int p = 0; p < w; ++p) {
      const u8* sp = planes + static_cast<usize>(p) * cnt;
      u8* op = out + static_cast<usize>(p) * cnt;
      for (usize i = 0; i < q; ++i) {
        const u64 x = bit_transpose8(load_u64(sp + 8 * i));
        for (int j = 0; j < 8; ++j)
          op[static_cast<usize>(j) * q + i] = static_cast<u8>(x >> (8 * j));
      }
    }
    for (usize i = region; i < n; ++i) out[i] = in[i];
    return;
  }
  const int bits = w * 8;
  std::memset(out, 0, region);
  for (int b = 0; b < bits; ++b) {
    const usize base = static_cast<usize>(b) * cnt;
    for (usize e = 0; e < cnt; ++e) {
      const u64 v = load_le(in + e * static_cast<usize>(w), w);
      if ((v >> b) & 1) {
        const usize pos = base + e;
        out[pos >> 3] |= static_cast<u8>(1u << (pos & 7));
      }
    }
  }
  for (usize i = region; i < n; ++i) out[i] = in[i];
}
inline void bitshuffle_inv(const u8* in, usize n, u8* out, int w) noexcept {
  const usize cnt = n / static_cast<usize>(w);
  const usize region = cnt * static_cast<usize>(w);
  if (cnt >= 8 && (cnt & 7) == 0) {
    u8* planes = filter_scratch(region).data();
    const usize q = cnt >> 3;
    for (int p = 0; p < w; ++p) {
      const u8* sp = in + static_cast<usize>(p) * cnt;
      u8* op = planes + static_cast<usize>(p) * cnt;
      for (usize i = 0; i < q; ++i) {
        u64 x = 0;
        for (int j = 0; j < 8; ++j)
          x |= static_cast<u64>(sp[static_cast<usize>(j) * q + i]) << (8 * j);
        store_u64(op + 8 * i, bit_transpose8(x));
      }
    }
    shuffle_inv(planes, region, out, w);
    for (usize i = region; i < n; ++i) out[i] = in[i];
    return;
  }
  const int bits = w * 8;
  for (usize e = 0; e < cnt; ++e) {
    u64 v = 0;
    for (int b = 0; b < bits; ++b) {
      const usize pos = static_cast<usize>(b) * cnt + e;
      v |= static_cast<u64>((in[pos >> 3] >> (pos & 7)) & 1) << b;
    }
    store_le(out + e * static_cast<usize>(w), v, w);
  }
  for (usize i = region; i < n; ++i) out[i] = in[i];
}

// --- BCJ x86 branch target normalizer (Filter::BcjX86) -----------------------
//
// Converts relative 32-bit call/jump displacements (0xE8 / 0xE9) into absolute
// addresses (forward) and back (inverse). Repeated calls to the same target
// across an executable or shared library map to identical byte sequences,
// creating long LZ matches and collapsing entropy.
//
// Only displacements whose top byte is 0x00 or 0xFF (|rel| < 16 MiB) are
// converted, the xz/7-Zip rule: a real call in a block of any size has a
// small displacement, while an 0xE8 byte inside data (strings, tables,
// floats) is followed by arbitrary bytes. Rewriting those would turn two
// copies of the same data at different offsets into different byte strings
// and break the LZ matches between them. The converted value is sign-folded
// to 25 bits so its top byte is again 0x00/0xFF; the inverse applies the
// same test and sees exactly the converted positions.

inline bool bcj_x86_in_range(u32 v) noexcept {
  const u32 top = v >> 24;
  return top == 0x00 || top == 0xFF;
}

inline u32 bcj_x86_fold25(u32 v) noexcept {
  v &= 0x01FFFFFFu;
  v |= 0u - (v & 0x01000000u);
  return v;
}

template <bool kFwd>
inline void bcj_x86_run(const u8* in, usize n, u8* out) noexcept {
  if (out != in) std::memcpy(out, in, n);
  if (n < 5) return;
  const usize limit = n - 5;
  for (usize i = 0; i <= limit;) {
    // Opcode bytes are 1-3 % of machine code: skip 8 bytes at a time while
    // none of them is 0xE8/0xE9 (SWAR zero-byte test on x | 0x01 ^ 0xE9).
    if (i + 8 <= limit + 1) {
      const u64 x = (load_u64(out + i) | 0x0101010101010101ull) ^ 0xE9E9E9E9E9E9E9E9ull;
      const u64 m = (x - 0x0101010101010101ull) & ~x & 0x8080808080808080ull;
      if (m == 0) {
        i += 8;
        continue;
      }
      // The lowest set bit of m marks the first 0xE8/0xE9 exactly (higher
      // bits may be borrow artefacts, which is why only the first is used).
      i += static_cast<usize>(std::countr_zero(m)) >> 3;
    }
    const u8 b = out[i];
    if (b == 0xE8 || b == 0xE9) {
      const u32 v = load_u32(out + i + 1);
      if (bcj_x86_in_range(v)) {
        const u32 pc = static_cast<u32>(i + 5);
        store_u32(out + i + 1, bcj_x86_fold25(kFwd ? v + pc : v - pc));
      }
      i += 5;
    } else {
      ++i;
    }
  }
}

inline void bcj_x86_fwd(const u8* in, usize n, u8* out) noexcept { bcj_x86_run<true>(in, n, out); }
inline void bcj_x86_inv(const u8* in, usize n, u8* out) noexcept { bcj_x86_run<false>(in, n, out); }

// --- BCJ ARM64 branch target normalizer (Filter::BcjArm64) --------------------
//
// AArch64 code is a stream of aligned 32-bit words. Two instruction forms
// carry PC-relative immediates that recur with the same target: BL (imm26,
// in words) and ADRP (imm21, in 4 KiB pages; the paired ADD/LDR carries the
// page offset and is already position-independent). Both are converted to
// absolute form, the same transform as xz's --arm64 filter:
//   BL   0x94000000 | imm26            -> imm26 + (pos >> 2)
//   ADRP 1 immlo[2] 10000 immhi[19] Rd -> imm21 + (pos >> 12), only when
//        |imm21| < 2^17 pages (512 MiB) so that the sign-folded 18-bit result
//        round-trips; larger ADRP immediates are left alone on both sides.

inline u32 bcj_arm64_adrp_imm(u32 w) noexcept {
  return ((w >> 29) & 3u) | ((w >> 3) & 0x001FFFFCu);
}

template <bool kFwd>
inline void bcj_arm64_run(const u8* in, usize n, u8* out) noexcept {
  if (out != in) std::memcpy(out, in, n);
  for (usize i = 0; i + 4 <= n; i += 4) {
    u32 w = load_u32(out + i);
    if ((w >> 26) == 0x25u) {  // BL
      const u32 pc = static_cast<u32>(i) >> 2;
      const u32 imm = kFwd ? w + pc : w - pc;
      w = 0x94000000u | (imm & 0x03FFFFFFu);
      store_u32(out + i, w);
    } else if ((w & 0x9F000000u) == 0x90000000u) {  // ADRP
      const u32 src = bcj_arm64_adrp_imm(w);
      // Signed 18-bit range test in one addition: bits 18..20 of
      // (src + 2^17) are zero iff -2^17 <= src < 2^17.
      if ((src + 0x00020000u) & 0x001C0000u) continue;
      const u32 pc = static_cast<u32>(i) >> 12;
      const u32 dst = kFwd ? src + pc : src - pc;
      w &= 0x9000001Fu;
      w |= (dst & 3u) << 29;
      w |= (dst & 0x0003FFFCu) << 3;
      w |= (0u - (dst & 0x00020000u)) & 0x00E00000u;  // sign-fold to 18 bits
      store_u32(out + i, w);
    }
  }
}

inline void bcj_arm64_fwd(const u8* in, usize n, u8* out) noexcept { bcj_arm64_run<true>(in, n, out); }
inline void bcj_arm64_inv(const u8* in, usize n, u8* out) noexcept { bcj_arm64_run<false>(in, n, out); }

// --- executable detection -----------------------------------------------------
//
// The byte-filter detector scores candidates by an order-1 adjacency proxy,
// which cannot see what a BCJ does: it leaves local statistics alone and
// creates long-range repeats. Executables are therefore recognised directly,
// by the file header when the block starts at one and otherwise by the
// instruction stream itself:
//   x86    0xE8/0xE9 followed by a displacement whose top byte is 0x00/0xFF
//          (a real call). In scalar code >= 2/3 of the 0xE8 bytes pass at a
//          density of one per KiB or more; in data the pass rate is ~1/128.
//          SIMD-heavy x86-64 is recognised by its REX prefix density instead
//          (>= 8 % of bytes in 0x40-0x4F) with a weaker call test.
//   ARM64  aligned words that decode as BL with a small target or as ADRP /
//          RET / STP-LDP(pre-index) frame instructions, at >= 1/40 of all
//          words, with >= 90 % of the BL targets within 16 MiB.
// A sample order-0 entropy floor rejects constant / sparse numeric columns
// whose bytes happen to pass the opcode tests (a u64 column of 232 is a
// stream of "E8 00 00 00 00 ..." and passes the x86 call test on every word),
// and the x86 displacements must vary from hit to hit: in a u64 timestamp
// column the byte that walks through 0xE8 sits four bytes before a zero
// lane, so every "call" there passes the range test with the same value.
// Several windows spread over the block are scored together so a block whose
// first half is code and second half is a symbol table is still an
// executable; the auto race uses the verdict to skip the record coders, the
// filter detector to select the matching BCJ.

enum class ExecArch : u8 { None = 0, X86 = 1, Arm64 = 2 };

inline ExecArch exec_arch_from_header(const u8* s, usize n) noexcept {
  if (n < 64) return ExecArch::None;
  // ELF: e_machine at 18 (little-endian for EM_X86_64 / EM_AARCH64 objects).
  if (s[0] == 0x7F && s[1] == 'E' && s[2] == 'L' && s[3] == 'F' && s[5] == 1) {
    const u32 m = static_cast<u32>(s[18]) | (static_cast<u32>(s[19]) << 8);
    if (m == 0x3E || m == 0x03) return ExecArch::X86;
    if (m == 0xB7) return ExecArch::Arm64;
    return ExecArch::None;
  }
  // Mach-O 64-bit, little-endian: cputype at 4.
  const u32 magic = load_u32(s);
  if (magic == 0xFEEDFACFu || magic == 0xFEEDFACEu) {
    const u32 cpu = load_u32(s + 4) & 0x00FFFFFFu;
    if (cpu == 7) return ExecArch::X86;
    if (cpu == 12) return ExecArch::Arm64;
    return ExecArch::None;
  }
  // PE: "MZ", e_lfanew at 0x3C, "PE\0\0" then Machine.
  if (s[0] == 'M' && s[1] == 'Z') {
    const u32 off = load_u32(s + 0x3C);
    if (off + 6 <= n && load_u32(s + off) == 0x00004550u) {
      const u32 m = static_cast<u32>(s[off + 4]) | (static_cast<u32>(s[off + 5]) << 8);
      if (m == 0x8664 || m == 0x014C) return ExecArch::X86;
      if (m == 0xAA64) return ExecArch::Arm64;
    }
    return ExecArch::None;
  }
  return ExecArch::None;
}

inline ExecArch exec_arch_from_code(const u8* s, usize n) noexcept {
  constexpr usize kWin = usize{1} << 13;  // 8 KiB per window
  constexpr int kWins = 8;                 // 64 KiB scanned per block at most
  if (n < 1024) return ExecArch::None;
  usize x86_e8 = 0, x86_ok = 0, x86_var = 0, x86_rex = 0;
  u32 x86_last = 0;
  usize a64_words = 0, a64_bl = 0, a64_bl_ok = 0, a64_frame = 0;
  u32 freq[256] = {};
  usize total = 0;
  const int wins = n <= kWin * kWins ? 1 : kWins;
  for (int wi = 0; wi < wins; ++wi) {
    usize start, len;
    if (wins == 1) {
      start = 0;
      len = n;
    } else {
      // Window centres at 1/16, 3/16, ..., 15/16 of the block, 4-byte aligned.
      start = ((n * static_cast<usize>(2 * wi + 1)) / (2 * kWins) - kWin / 2) & ~usize{3};
      len = kWin;
    }
    const u8* w = s + start;
    total += len;
    for (usize i = 0; i < len; ++i) ++freq[w[i]];
    for (usize i = 0; i + 5 <= len; ++i) {
      const u8 b = w[i];
      if (b == 0xE8 || b == 0xE9) {
        ++x86_e8;
        const u32 disp = load_u32(w + i + 1);
        if (bcj_x86_in_range(disp)) {
          ++x86_ok;
          // Calls have varied displacements (a different pc each time, many
          // targets); a record lane that happens to hold 0xE8 four bytes
          // before a zero lane repeats one value.
          if (disp != x86_last) ++x86_var;
          x86_last = disp;
        }
      } else if ((b & 0xF0) == 0x40) {
        ++x86_rex;
      }
    }
    for (usize i = 0; i + 4 <= len; i += 4) {
      const u32 v = load_u32(w + i);
      ++a64_words;
      if ((v >> 26) == 0x25u) {
        ++a64_bl;
        const u32 hi = (v >> 22) & 0xFu;  // top 4 bits of imm26: |target| < 16 MiB
        if (hi == 0x0 || hi == 0xF) ++a64_bl_ok;
      } else if ((v & 0x9F000000u) == 0x90000000u ||   // ADRP
                 v == 0xD65F03C0u ||                    // RET
                 (v & 0xFFC00000u) == 0xA9800000u ||    // STP pre-index (frame push)
                 (v & 0xFFC00000u) == 0xA8C00000u) {    // LDP post-index (frame pop)
        ++a64_frame;
      }
    }
  }
  // Entropy floor: machine code is dense (6+ bits/byte); columns of small
  // integers and padding are not.
  double bits = 0.0;
  for (u32 f : freq) {
    if (!f) continue;
    const double p = static_cast<double>(f) / static_cast<double>(total);
    bits -= p * std::log2(p);
  }
  if (bits >= 4.5) {
    // Plain x86 / scalar x86-64: most 0xE8 bytes are calls. SIMD-heavy
    // x86-64 (BLAS kernels) has 0xE8 as a ModRM byte in half the cases, but
    // then every other instruction starts with a REX prefix (0x40-0x4F,
    // 11-23 % of all bytes against 1-6 % in data, text and random bytes).
    const bool varied = x86_var * 2 >= x86_ok;
    const bool calls = x86_e8 >= total / 1024 && x86_ok * 3 >= x86_e8 * 2;
    const bool rex = x86_rex * 100 >= total * 8 && x86_e8 >= total / 512 &&
                     x86_ok * 5 >= x86_e8;
    if (varied && (calls || rex)) return ExecArch::X86;
  }
  // AArch64 words are low in byte entropy (4.0-4.5 bits: fixed fields,
  // repeated register encodings), so its floor is lower; the BL target test
  // is what keeps random and floating-point words out (1/8 of random BL
  // candidates pass it, 100 % of real ones).
  if (bits >= 3.0 && a64_bl + a64_frame >= a64_words / 40 && a64_bl >= a64_words / 256 &&
      a64_bl_ok * 10 >= a64_bl * 9)
    return ExecArch::Arm64;
  return ExecArch::None;
}

inline ExecArch detect_exec_arch(const u8* s, usize n) noexcept {
  const ExecArch h = exec_arch_from_header(s, n);
  if (h != ExecArch::None) return h;
  return exec_arch_from_code(s, n);
}

inline Filter bcj_filter_for(ExecArch a) noexcept {
  return a == ExecArch::X86 ? Filter::BcjX86 : a == ExecArch::Arm64 ? Filter::BcjArm64 : Filter::None;
}

// --- fused compound transforms (zero intermediate scratch) -----------------

template <int W>
inline void shuffle_scatter(const u8* chunk_buf, usize count, u8* out, usize cnt, usize e0) noexcept {
  if constexpr (W == 4) {
    usize j = 0;
#if defined(SPECTRA8_TRANSFORM_NEON)
    for (; j + 16 <= count; j += 16) {
      const uint8x16x4_t v = vld4q_u8(chunk_buf + 4 * j);
      for (int p = 0; p < 4; ++p)
        vst1q_u8(out + static_cast<usize>(p) * cnt + e0 + j, v.val[p]);
    }
#elif defined(SPECTRA8_TRANSFORM_SSE2)
    // Same even/odd pack as shuffle_fwd(w == 4). An extra unpack32 stage here
    // scrambles the byte planes (zzd2shuf32 then fails to round-trip on x86_64).
    const __m128i m8 = _mm_set1_epi16(0x00FF);
    const auto ld = [](const u8* p) { return _mm_loadu_si128(reinterpret_cast<const __m128i*>(p)); };
    const auto st = [](u8* p, __m128i v) { _mm_storeu_si128(reinterpret_cast<__m128i*>(p), v); };
    const auto pe = [m8](__m128i a, __m128i b) {
      return _mm_packus_epi16(_mm_and_si128(a, m8), _mm_and_si128(b, m8));
    };
    const auto po = [](__m128i a, __m128i b) {
      return _mm_packus_epi16(_mm_srli_epi16(a, 8), _mm_srli_epi16(b, 8));
    };
    for (; j + 16 <= count; j += 16) {
      const __m128i r0 = ld(chunk_buf + 4 * j);
      const __m128i r1 = ld(chunk_buf + 4 * j + 16);
      const __m128i r2 = ld(chunk_buf + 4 * j + 32);
      const __m128i r3 = ld(chunk_buf + 4 * j + 48);
      const __m128i ev0 = pe(r0, r1), ev1 = pe(r2, r3);
      const __m128i od0 = po(r0, r1), od1 = po(r2, r3);
      st(out + 0 * cnt + e0 + j, pe(ev0, ev1));
      st(out + 1 * cnt + e0 + j, pe(od0, od1));
      st(out + 2 * cnt + e0 + j, po(ev0, ev1));
      st(out + 3 * cnt + e0 + j, po(od0, od1));
    }
#endif
    for (; j < count; ++j) {
      out[0 * cnt + e0 + j] = chunk_buf[4 * j + 0];
      out[1 * cnt + e0 + j] = chunk_buf[4 * j + 1];
      out[2 * cnt + e0 + j] = chunk_buf[4 * j + 2];
      out[3 * cnt + e0 + j] = chunk_buf[4 * j + 3];
    }
  } else if constexpr (W == 8) {
    usize j = 0;
#if defined(SPECTRA8_TRANSFORM_NEON)
    for (; j + 16 <= count; j += 16) {
      const uint8x16x4_t a = vld4q_u8(chunk_buf + 8 * j);
      const uint8x16x4_t b = vld4q_u8(chunk_buf + 8 * j + 64);
      for (int k = 0; k < 4; ++k) {
        const uint8x16x2_t uz = vuzpq_u8(a.val[k], b.val[k]);
        vst1q_u8(out + static_cast<usize>(k) * cnt + e0 + j, uz.val[0]);
        vst1q_u8(out + static_cast<usize>(k + 4) * cnt + e0 + j, uz.val[1]);
      }
    }
#endif
    for (; j < count; ++j) {
      for (int p = 0; p < 8; ++p)
        out[static_cast<usize>(p) * cnt + e0 + j] = chunk_buf[8 * j + p];
    }
  }
}

template <int W>
inline void shuffle_gather(const u8* in, usize cnt, usize e0, u8* chunk_buf, usize count) noexcept {
  if constexpr (W == 4) {
    usize j = 0;
#if defined(SPECTRA8_TRANSFORM_NEON)
    for (; j + 16 <= count; j += 16) {
      uint8x16x4_t v;
      for (int p = 0; p < 4; ++p)
        v.val[p] = vld1q_u8(in + static_cast<usize>(p) * cnt + e0 + j);
      vst4q_u8(chunk_buf + 4 * j, v);
    }
#elif defined(SPECTRA8_TRANSFORM_SSE2)
    const auto ld = [](const u8* p) { return _mm_loadu_si128(reinterpret_cast<const __m128i*>(p)); };
    const auto st = [](u8* p, __m128i v) { _mm_storeu_si128(reinterpret_cast<__m128i*>(p), v); };
    for (; j + 16 <= count; j += 16) {
      const __m128i a0 = ld(in + 0 * cnt + e0 + j);
      const __m128i a1 = ld(in + 1 * cnt + e0 + j);
      const __m128i a2 = ld(in + 2 * cnt + e0 + j);
      const __m128i a3 = ld(in + 3 * cnt + e0 + j);
      const __m128i t0 = _mm_unpacklo_epi8(a0, a1);
      const __m128i t1 = _mm_unpackhi_epi8(a0, a1);
      const __m128i t2 = _mm_unpacklo_epi8(a2, a3);
      const __m128i t3 = _mm_unpackhi_epi8(a2, a3);
      // Match shuffle_inv(w == 4): the epi16 unpacks are already element order.
      st(chunk_buf + 4 * j, _mm_unpacklo_epi16(t0, t2));
      st(chunk_buf + 4 * j + 16, _mm_unpackhi_epi16(t0, t2));
      st(chunk_buf + 4 * j + 32, _mm_unpacklo_epi16(t1, t3));
      st(chunk_buf + 4 * j + 48, _mm_unpackhi_epi16(t1, t3));
    }
#endif
    for (; j < count; ++j) {
      chunk_buf[4 * j + 0] = in[0 * cnt + e0 + j];
      chunk_buf[4 * j + 1] = in[1 * cnt + e0 + j];
      chunk_buf[4 * j + 2] = in[2 * cnt + e0 + j];
      chunk_buf[4 * j + 3] = in[3 * cnt + e0 + j];
    }
  } else if constexpr (W == 8) {
    usize j = 0;
#if defined(SPECTRA8_TRANSFORM_NEON)
    for (; j + 16 <= count; j += 16) {
      uint8x16x4_t a, b;
      for (int k = 0; k < 4; ++k) {
        const uint8x16_t lo = vld1q_u8(in + static_cast<usize>(k) * cnt + e0 + j);
        const uint8x16_t hi = vld1q_u8(in + static_cast<usize>(k + 4) * cnt + e0 + j);
        const uint8x16x2_t zip = vzipq_u8(lo, hi);
        a.val[k] = zip.val[0];
        b.val[k] = zip.val[1];
      }
      vst4q_u8(chunk_buf + 8 * j, a);
      vst4q_u8(chunk_buf + 8 * j + 64, b);
    }
#endif
    for (; j < count; ++j) {
      for (int p = 0; p < 8; ++p)
        chunk_buf[8 * j + p] = in[static_cast<usize>(p) * cnt + e0 + j];
    }
  }
}

template <int W>
inline void fused_delta_shuffle_fwd(const u8* in, usize n, u8* out) noexcept {
  const usize cnt = n / static_cast<usize>(W);
  constexpr usize kChunk = 1024;
  alignas(64) u8 buf[kChunk * W];
  if constexpr (W == 4) {
    u32 prev = 0;
    for (usize e0 = 0; e0 < cnt; e0 += kChunk) {
      const usize cur = std::min(kChunk, cnt - e0);
      for (usize j = 0; j < cur; ++j) {
        const u32 v = load_u32(in + 4 * (e0 + j));
        store_u32(buf + 4 * j, v - prev);
        prev = v;
      }
      shuffle_scatter<4>(buf, cur, out, cnt, e0);
    }
  } else if constexpr (W == 8) {
    u64 prev = 0;
    for (usize e0 = 0; e0 < cnt; e0 += kChunk) {
      const usize cur = std::min(kChunk, cnt - e0);
      for (usize j = 0; j < cur; ++j) {
        const u64 v = load_u64(in + 8 * (e0 + j));
        store_u64(buf + 8 * j, v - prev);
        prev = v;
      }
      shuffle_scatter<8>(buf, cur, out, cnt, e0);
    }
  }
  const usize tail = cnt * static_cast<usize>(W);
  for (usize i = tail; i < n; ++i) out[i] = in[i];
}

template <int W>
inline void fused_delta_shuffle_inv(const u8* in, usize n, u8* out) noexcept {
  const usize cnt = n / static_cast<usize>(W);
  constexpr usize kChunk = 1024;
  alignas(64) u8 buf[kChunk * W];
  if constexpr (W == 4) {
    u32 prev = 0;
    for (usize e0 = 0; e0 < cnt; e0 += kChunk) {
      const usize cur = std::min(kChunk, cnt - e0);
      shuffle_gather<4>(in, cnt, e0, buf, cur);
      for (usize j = 0; j < cur; ++j) {
        const u32 d = load_u32(buf + 4 * j);
        const u32 v = d + prev;
        store_u32(out + 4 * (e0 + j), v);
        prev = v;
      }
    }
  } else if constexpr (W == 8) {
    u64 prev = 0;
    for (usize e0 = 0; e0 < cnt; e0 += kChunk) {
      const usize cur = std::min(kChunk, cnt - e0);
      shuffle_gather<8>(in, cnt, e0, buf, cur);
      for (usize j = 0; j < cur; ++j) {
        const u64 d = load_u64(buf + 8 * j);
        const u64 v = d + prev;
        store_u64(out + 8 * (e0 + j), v);
        prev = v;
      }
    }
  }
  const usize tail = cnt * static_cast<usize>(W);
  for (usize i = tail; i < n; ++i) out[i] = in[i];
}

template <int W>
inline void fused_zigzag_delta_shuffle_fwd(const u8* in, usize n, u8* out) noexcept {
  const usize cnt = n / static_cast<usize>(W);
  constexpr usize kChunk = 1024;
  alignas(64) u8 buf[kChunk * W];
  if constexpr (W == 4) {
    u32 prev = 0;
    for (usize e0 = 0; e0 < cnt; e0 += kChunk) {
      const usize cur = std::min(kChunk, cnt - e0);
      for (usize j = 0; j < cur; ++j) {
        const u32 v = load_u32(in + 4 * (e0 + j));
        const u32 d = v - prev;
        const u32 zz = (d << 1) ^ (0u - (d >> 31));
        store_u32(buf + 4 * j, zz);
        prev = v;
      }
      shuffle_scatter<4>(buf, cur, out, cnt, e0);
    }
  }
  const usize tail = cnt * static_cast<usize>(W);
  for (usize i = tail; i < n; ++i) out[i] = in[i];
}

template <int W>
inline void fused_zigzag_delta_shuffle_inv(const u8* in, usize n, u8* out) noexcept {
  const usize cnt = n / static_cast<usize>(W);
  constexpr usize kChunk = 1024;
  alignas(64) u8 buf[kChunk * W];
  if constexpr (W == 4) {
    u32 prev = 0;
    for (usize e0 = 0; e0 < cnt; e0 += kChunk) {
      const usize cur = std::min(kChunk, cnt - e0);
      shuffle_gather<4>(in, cnt, e0, buf, cur);
      for (usize j = 0; j < cur; ++j) {
        const u32 zz = load_u32(buf + 4 * j);
        const u32 d = (zz >> 1) ^ (0u - (zz & 1u));
        const u32 v = d + prev;
        store_u32(out + 4 * (e0 + j), v);
        prev = v;
      }
    }
  }
  const usize tail = cnt * static_cast<usize>(W);
  for (usize i = tail; i < n; ++i) out[i] = in[i];
}

template <int W>
inline void fused_zigzag_shuffle_fwd(const u8* in, usize n, u8* out) noexcept {
  const usize cnt = n / static_cast<usize>(W);
  constexpr usize kChunk = 1024;
  alignas(64) u8 buf[kChunk * W];
  if constexpr (W == 4) {
    for (usize e0 = 0; e0 < cnt; e0 += kChunk) {
      const usize cur = std::min(kChunk, cnt - e0);
      for (usize j = 0; j < cur; ++j) {
        const u32 v = load_u32(in + 4 * (e0 + j));
        const u32 zz = (v << 1) ^ (0u - (v >> 31));
        store_u32(buf + 4 * j, zz);
      }
      shuffle_scatter<4>(buf, cur, out, cnt, e0);
    }
  } else if constexpr (W == 8) {
    for (usize e0 = 0; e0 < cnt; e0 += kChunk) {
      const usize cur = std::min(kChunk, cnt - e0);
      for (usize j = 0; j < cur; ++j) {
        const u64 v = load_u64(in + 8 * (e0 + j));
        const u64 zz = (v << 1) ^ (0ull - (v >> 63));
        store_u64(buf + 8 * j, zz);
      }
      shuffle_scatter<8>(buf, cur, out, cnt, e0);
    }
  }
  const usize tail = cnt * static_cast<usize>(W);
  for (usize i = tail; i < n; ++i) out[i] = in[i];
}

template <int W>
inline void fused_zigzag_shuffle_inv(const u8* in, usize n, u8* out) noexcept {
  const usize cnt = n / static_cast<usize>(W);
  constexpr usize kChunk = 1024;
  alignas(64) u8 buf[kChunk * W];
  if constexpr (W == 4) {
    for (usize e0 = 0; e0 < cnt; e0 += kChunk) {
      const usize cur = std::min(kChunk, cnt - e0);
      shuffle_gather<4>(in, cnt, e0, buf, cur);
      for (usize j = 0; j < cur; ++j) {
        const u32 zz = load_u32(buf + 4 * j);
        const u32 v = (zz >> 1) ^ (0u - (zz & 1u));
        store_u32(out + 4 * (e0 + j), v);
      }
    }
  } else if constexpr (W == 8) {
    for (usize e0 = 0; e0 < cnt; e0 += kChunk) {
      const usize cur = std::min(kChunk, cnt - e0);
      shuffle_gather<8>(in, cnt, e0, buf, cur);
      for (usize j = 0; j < cur; ++j) {
        const u64 zz = load_u64(buf + 8 * j);
        const u64 v = (zz >> 1) ^ (0ull - (zz & 1ull));
        store_u64(out + 8 * (e0 + j), v);
      }
    }
  }
  const usize tail = cnt * static_cast<usize>(W);
  for (usize i = tail; i < n; ++i) out[i] = in[i];
}

template <int W>
inline void fused_xor_shuffle_fwd(const u8* in, usize n, u8* out) noexcept {
  const usize cnt = n / static_cast<usize>(W);
  constexpr usize kChunk = 1024;
  alignas(64) u8 buf[kChunk * W];
  if constexpr (W == 4) {
    u32 prev = 0;
    for (usize e0 = 0; e0 < cnt; e0 += kChunk) {
      const usize cur = std::min(kChunk, cnt - e0);
      for (usize j = 0; j < cur; ++j) {
        const u32 v = load_u32(in + 4 * (e0 + j));
        store_u32(buf + 4 * j, v ^ prev);
        prev = v;
      }
      shuffle_scatter<4>(buf, cur, out, cnt, e0);
    }
  } else if constexpr (W == 8) {
    u64 prev = 0;
    for (usize e0 = 0; e0 < cnt; e0 += kChunk) {
      const usize cur = std::min(kChunk, cnt - e0);
      for (usize j = 0; j < cur; ++j) {
        const u64 v = load_u64(in + 8 * (e0 + j));
        store_u64(buf + 8 * j, v ^ prev);
        prev = v;
      }
      shuffle_scatter<8>(buf, cur, out, cnt, e0);
    }
  }
  const usize tail = cnt * static_cast<usize>(W);
  for (usize i = tail; i < n; ++i) out[i] = in[i];
}

template <int W>
inline void fused_xor_shuffle_inv(const u8* in, usize n, u8* out) noexcept {
  const usize cnt = n / static_cast<usize>(W);
  constexpr usize kChunk = 1024;
  alignas(64) u8 buf[kChunk * W];
  if constexpr (W == 4) {
    u32 prev = 0;
    for (usize e0 = 0; e0 < cnt; e0 += kChunk) {
      const usize cur = std::min(kChunk, cnt - e0);
      shuffle_gather<4>(in, cnt, e0, buf, cur);
      for (usize j = 0; j < cur; ++j) {
        const u32 r = load_u32(buf + 4 * j);
        const u32 v = r ^ prev;
        store_u32(out + 4 * (e0 + j), v);
        prev = v;
      }
    }
  } else if constexpr (W == 8) {
    u64 prev = 0;
    for (usize e0 = 0; e0 < cnt; e0 += kChunk) {
      const usize cur = std::min(kChunk, cnt - e0);
      shuffle_gather<8>(in, cnt, e0, buf, cur);
      for (usize j = 0; j < cur; ++j) {
        const u64 r = load_u64(buf + 8 * j);
        const u64 v = r ^ prev;
        store_u64(out + 8 * (e0 + j), v);
        prev = v;
      }
    }
  }
  const usize tail = cnt * static_cast<usize>(W);
  for (usize i = tail; i < n; ++i) out[i] = in[i];
}

// --- lag-2 residual (subtract or XOR x[i-2]) + byte transpose --------------

template <int W, bool XOR>
inline void fused_lag2_shuffle_fwd(const u8* in, usize n, u8* out) noexcept {
  const usize cnt = n / static_cast<usize>(W);
  constexpr usize kChunk = 1024;
  alignas(64) u8 buf[kChunk * W];
  if constexpr (W == 4) {
    u32 p1 = 0, p2 = 0;  // x[i-1], x[i-2]
    for (usize e0 = 0; e0 < cnt; e0 += kChunk) {
      const usize cur = std::min(kChunk, cnt - e0);
      for (usize j = 0; j < cur; ++j) {
        const u32 v = load_u32(in + 4 * (e0 + j));
        store_u32(buf + 4 * j, XOR ? (v ^ p2) : (v - p2));
        p2 = p1;
        p1 = v;
      }
      shuffle_scatter<4>(buf, cur, out, cnt, e0);
    }
  } else if constexpr (W == 8) {
    u64 p1 = 0, p2 = 0;
    for (usize e0 = 0; e0 < cnt; e0 += kChunk) {
      const usize cur = std::min(kChunk, cnt - e0);
      for (usize j = 0; j < cur; ++j) {
        const u64 v = load_u64(in + 8 * (e0 + j));
        store_u64(buf + 8 * j, XOR ? (v ^ p2) : (v - p2));
        p2 = p1;
        p1 = v;
      }
      shuffle_scatter<8>(buf, cur, out, cnt, e0);
    }
  }
  const usize tail = cnt * static_cast<usize>(W);
  for (usize i = tail; i < n; ++i) out[i] = in[i];
}

template <int W, bool XOR>
inline void fused_lag2_shuffle_inv(const u8* in, usize n, u8* out) noexcept {
  const usize cnt = n / static_cast<usize>(W);
  constexpr usize kChunk = 1024;
  alignas(64) u8 buf[kChunk * W];
  if constexpr (W == 4) {
    u32 p1 = 0, p2 = 0;
    for (usize e0 = 0; e0 < cnt; e0 += kChunk) {
      const usize cur = std::min(kChunk, cnt - e0);
      shuffle_gather<4>(in, cnt, e0, buf, cur);
      for (usize j = 0; j < cur; ++j) {
        const u32 r = load_u32(buf + 4 * j);
        const u32 v = XOR ? (r ^ p2) : (r + p2);
        store_u32(out + 4 * (e0 + j), v);
        p2 = p1;
        p1 = v;
      }
    }
  } else if constexpr (W == 8) {
    u64 p1 = 0, p2 = 0;
    for (usize e0 = 0; e0 < cnt; e0 += kChunk) {
      const usize cur = std::min(kChunk, cnt - e0);
      shuffle_gather<8>(in, cnt, e0, buf, cur);
      for (usize j = 0; j < cur; ++j) {
        const u64 r = load_u64(buf + 8 * j);
        const u64 v = XOR ? (r ^ p2) : (r + p2);
        store_u64(out + 8 * (e0 + j), v);
        p2 = p1;
        p1 = v;
      }
    }
  }
  const usize tail = cnt * static_cast<usize>(W);
  for (usize i = tail; i < n; ++i) out[i] = in[i];
}

template <int W>
inline void fused_delta2_shuffle_fwd(const u8* in, usize n, u8* out) noexcept {
  const usize cnt = n / static_cast<usize>(W);
  constexpr usize kChunk = 1024;
  alignas(64) u8 buf[kChunk * W];
  if constexpr (W == 4) {
    u32 p1 = 0, p2 = 0;
    for (usize e0 = 0; e0 < cnt; e0 += kChunk) {
      const usize cur = std::min(kChunk, cnt - e0);
      for (usize j = 0; j < cur; ++j) {
        const u32 v = load_u32(in + 4 * (e0 + j));
        const u32 pred = 2u * p1 - p2;
        store_u32(buf + 4 * j, v - pred);
        p2 = p1;
        p1 = v;
      }
      shuffle_scatter<4>(buf, cur, out, cnt, e0);
    }
  } else if constexpr (W == 8) {
    u64 p1 = 0, p2 = 0;
    for (usize e0 = 0; e0 < cnt; e0 += kChunk) {
      const usize cur = std::min(kChunk, cnt - e0);
      for (usize j = 0; j < cur; ++j) {
        const u64 v = load_u64(in + 8 * (e0 + j));
        const u64 pred = 2u * p1 - p2;
        store_u64(buf + 8 * j, v - pred);
        p2 = p1;
        p1 = v;
      }
      shuffle_scatter<8>(buf, cur, out, cnt, e0);
    }
  }
  const usize tail = cnt * static_cast<usize>(W);
  for (usize i = tail; i < n; ++i) out[i] = in[i];
}

template <int W>
inline void fused_delta2_shuffle_inv(const u8* in, usize n, u8* out) noexcept {
  const usize cnt = n / static_cast<usize>(W);
  constexpr usize kChunk = 1024;
  alignas(64) u8 buf[kChunk * W];
  if constexpr (W == 4) {
    u32 p1 = 0, p2 = 0;
    for (usize e0 = 0; e0 < cnt; e0 += kChunk) {
      const usize cur = std::min(kChunk, cnt - e0);
      shuffle_gather<4>(in, cnt, e0, buf, cur);
      for (usize j = 0; j < cur; ++j) {
        const u32 r = load_u32(buf + 4 * j);
        const u32 pred = 2u * p1 - p2;
        const u32 v = r + pred;
        store_u32(out + 4 * (e0 + j), v);
        p2 = p1;
        p1 = v;
      }
    }
  } else if constexpr (W == 8) {
    u64 p1 = 0, p2 = 0;
    for (usize e0 = 0; e0 < cnt; e0 += kChunk) {
      const usize cur = std::min(kChunk, cnt - e0);
      shuffle_gather<8>(in, cnt, e0, buf, cur);
      for (usize j = 0; j < cur; ++j) {
        const u64 r = load_u64(buf + 8 * j);
        const u64 pred = 2u * p1 - p2;
        const u64 v = r + pred;
        store_u64(out + 8 * (e0 + j), v);
        p2 = p1;
        p1 = v;
      }
    }
  }
  const usize tail = cnt * static_cast<usize>(W);
  for (usize i = tail; i < n; ++i) out[i] = in[i];
}

template <int W>
inline void fused_zigzag_delta2_shuffle_fwd(const u8* in, usize n, u8* out) noexcept {
  const usize cnt = n / static_cast<usize>(W);
  constexpr usize kChunk = 1024;
  alignas(64) u8 buf[kChunk * W];
  if constexpr (W == 4) {
    u32 p1 = 0, p2 = 0;
    for (usize e0 = 0; e0 < cnt; e0 += kChunk) {
      const usize cur = std::min(kChunk, cnt - e0);
      for (usize j = 0; j < cur; ++j) {
        const u32 v = load_u32(in + 4 * (e0 + j));
        const u32 pred = 2u * p1 - p2;
        const u32 d = v - pred;
        const u32 zz = (d << 1) ^ (0u - (d >> 31));
        store_u32(buf + 4 * j, zz);
        p2 = p1;
        p1 = v;
      }
      shuffle_scatter<4>(buf, cur, out, cnt, e0);
    }
  }
  const usize tail = cnt * static_cast<usize>(W);
  for (usize i = tail; i < n; ++i) out[i] = in[i];
}

template <int W>
inline void fused_zigzag_delta2_shuffle_inv(const u8* in, usize n, u8* out) noexcept {
  const usize cnt = n / static_cast<usize>(W);
  constexpr usize kChunk = 1024;
  alignas(64) u8 buf[kChunk * W];
  if constexpr (W == 4) {
    u32 p1 = 0, p2 = 0;
    for (usize e0 = 0; e0 < cnt; e0 += kChunk) {
      const usize cur = std::min(kChunk, cnt - e0);
      shuffle_gather<4>(in, cnt, e0, buf, cur);
      for (usize j = 0; j < cur; ++j) {
        const u32 zz = load_u32(buf + 4 * j);
        const u32 d = (zz >> 1) ^ (0u - (zz & 1u));
        const u32 pred = 2u * p1 - p2;
        const u32 v = d + pred;
        store_u32(out + 4 * (e0 + j), v);
        p2 = p1;
        p1 = v;
      }
    }
  }
  const usize tail = cnt * static_cast<usize>(W);
  for (usize i = tail; i < n; ++i) out[i] = in[i];
}

// --- IEEE-754 mantissa bit-rounding (BitRound32 / BitRound64) --------------

inline u32 round_f32(u32 u, int keep_mantissa_bits = 10) noexcept {
  const u32 exp = (u >> 23) & 0xFF;
  if (exp == 0 || exp == 0xFF) return u;
  const int k = 23 - keep_mantissa_bits;
  if (k <= 0) return u;
  const u32 mask = (1u << k) - 1;
  const u32 half = 1u << (k - 1);
  u += (half - 1) + ((u >> k) & 1);
  return u & ~mask;
}

inline u64 round_f64(u64 u, int keep_mantissa_bits = 20) noexcept {
  const u64 exp = (u >> 52) & 0x7FF;
  if (exp == 0 || exp == 0x7FF) return u;
  const int k = 52 - keep_mantissa_bits;
  if (k <= 0) return u;
  const u64 mask = (1ull << k) - 1;
  const u64 half = 1ull << (k - 1);
  u += (half - 1) + ((u >> k) & 1);
  return u & ~mask;
}

template <int W>
inline void fused_bitround_shuffle_fwd(const u8* in, usize n, u8* out) noexcept {
  const usize cnt = n / static_cast<usize>(W);
  constexpr usize kChunk = 1024;
  alignas(64) u8 buf[kChunk * W];
  if constexpr (W == 4) {
    for (usize e0 = 0; e0 < cnt; e0 += kChunk) {
      const usize cur = std::min(kChunk, cnt - e0);
      for (usize j = 0; j < cur; ++j) {
        const u32 v = load_u32(in + 4 * (e0 + j));
        store_u32(buf + 4 * j, round_f32(v, 10));
      }
      shuffle_scatter<4>(buf, cur, out, cnt, e0);
    }
  } else if constexpr (W == 8) {
    for (usize e0 = 0; e0 < cnt; e0 += kChunk) {
      const usize cur = std::min(kChunk, cnt - e0);
      for (usize j = 0; j < cur; ++j) {
        const u64 v = load_u64(in + 8 * (e0 + j));
        store_u64(buf + 8 * j, round_f64(v, 20));
      }
      shuffle_scatter<8>(buf, cur, out, cnt, e0);
    }
  }
  const usize tail = cnt * static_cast<usize>(W);
  for (usize i = tail; i < n; ++i) out[i] = in[i];
}

// --- dispatch ---------------------------------------------------------------

inline void apply_filter(Filter f, const u8* in, usize n, u8* out) {
  switch (f) {
    case Filter::None: std::memcpy(out, in, n); break;
    case Filter::Shuffle16: shuffle_fwd(in, n, out, 2); break;
    case Filter::Shuffle32: shuffle_fwd(in, n, out, 4); break;
    case Filter::Shuffle64: shuffle_fwd(in, n, out, 8); break;
    case Filter::Delta16: delta_fwd(in, n, out, 2); break;
    case Filter::Delta32: delta_fwd(in, n, out, 4); break;
    case Filter::Delta64: delta_fwd(in, n, out, 8); break;
    case Filter::ZigzagDelta16: zigzag_delta_fwd(in, n, out, 2); break;
    case Filter::ZigzagDelta32: zigzag_delta_fwd(in, n, out, 4); break;
    case Filter::ZigzagDelta64: zigzag_delta_fwd(in, n, out, 8); break;
    case Filter::DeltaShuffle32: fused_delta_shuffle_fwd<4>(in, n, out); break;
    case Filter::ZigzagDeltaShuffle32: fused_zigzag_delta_shuffle_fwd<4>(in, n, out); break;
    case Filter::BitShuffle32: bitshuffle_fwd(in, n, out, 4); break;
    case Filter::BitShuffle64: bitshuffle_fwd(in, n, out, 8); break;
    case Filter::DeltaBitShuffle32: {
      // bitshuffle_fwd reuses filter_scratch, so the delta must live elsewhere.
      static thread_local std::vector<u8> delta;
      delta.resize(n);
      delta_fwd(in, n, delta.data(), 4);
      bitshuffle_fwd(delta.data(), n, out, 4);
      break;
    }
    case Filter::DeltaShuffle64: fused_delta_shuffle_fwd<8>(in, n, out); break;
    case Filter::ZigzagShuffle32: fused_zigzag_shuffle_fwd<4>(in, n, out); break;
    case Filter::ZigzagShuffle64: fused_zigzag_shuffle_fwd<8>(in, n, out); break;
    case Filter::XorShuffle32: fused_xor_shuffle_fwd<4>(in, n, out); break;
    case Filter::XorShuffle64: fused_xor_shuffle_fwd<8>(in, n, out); break;
    case Filter::Delta2Shuffle32: fused_delta2_shuffle_fwd<4>(in, n, out); break;
    case Filter::ZigzagDelta2Shuffle32: fused_zigzag_delta2_shuffle_fwd<4>(in, n, out); break;
    case Filter::Delta2Shuffle64: fused_delta2_shuffle_fwd<8>(in, n, out); break;
    case Filter::BcjX86: bcj_x86_fwd(in, n, out); break;
    case Filter::BcjArm64: bcj_arm64_fwd(in, n, out); break;
    case Filter::BitRound32: fused_bitround_shuffle_fwd<4>(in, n, out); break;
    case Filter::BitRound64: fused_bitround_shuffle_fwd<8>(in, n, out); break;
    case Filter::Lag2Shuffle32: fused_lag2_shuffle_fwd<4, false>(in, n, out); break;
    case Filter::Lag2Shuffle64: fused_lag2_shuffle_fwd<8, false>(in, n, out); break;
    case Filter::XorLag2Shuffle32: fused_lag2_shuffle_fwd<4, true>(in, n, out); break;
    default: std::memcpy(out, in, n); break;
  }
}

inline void invert_filter(Filter f, const u8* in, usize n, u8* out) {
  switch (f) {
    case Filter::None: std::memcpy(out, in, n); break;
    case Filter::Shuffle16: shuffle_inv(in, n, out, 2); break;
    case Filter::Shuffle32: shuffle_inv(in, n, out, 4); break;
    case Filter::Shuffle64: shuffle_inv(in, n, out, 8); break;
    case Filter::Delta16: delta_inv(in, n, out, 2); break;
    case Filter::Delta32: delta_inv(in, n, out, 4); break;
    case Filter::Delta64: delta_inv(in, n, out, 8); break;
    case Filter::ZigzagDelta16: zigzag_delta_inv(in, n, out, 2); break;
    case Filter::ZigzagDelta32: zigzag_delta_inv(in, n, out, 4); break;
    case Filter::ZigzagDelta64: zigzag_delta_inv(in, n, out, 8); break;
    case Filter::DeltaShuffle32: fused_delta_shuffle_inv<4>(in, n, out); break;
    case Filter::ZigzagDeltaShuffle32: fused_zigzag_delta_shuffle_inv<4>(in, n, out); break;
    case Filter::BitShuffle32: bitshuffle_inv(in, n, out, 4); break;
    case Filter::BitShuffle64: bitshuffle_inv(in, n, out, 8); break;
    case Filter::DeltaBitShuffle32: {
      static thread_local std::vector<u8> plain;
      plain.resize(n);
      bitshuffle_inv(in, n, plain.data(), 4);
      delta_inv(plain.data(), n, out, 4);
      break;
    }
    case Filter::DeltaShuffle64: fused_delta_shuffle_inv<8>(in, n, out); break;
    case Filter::ZigzagShuffle32: fused_zigzag_shuffle_inv<4>(in, n, out); break;
    case Filter::ZigzagShuffle64: fused_zigzag_shuffle_inv<8>(in, n, out); break;
    case Filter::XorShuffle32: fused_xor_shuffle_inv<4>(in, n, out); break;
    case Filter::XorShuffle64: fused_xor_shuffle_inv<8>(in, n, out); break;
    case Filter::Delta2Shuffle32: fused_delta2_shuffle_inv<4>(in, n, out); break;
    case Filter::ZigzagDelta2Shuffle32: fused_zigzag_delta2_shuffle_inv<4>(in, n, out); break;
    case Filter::Delta2Shuffle64: fused_delta2_shuffle_inv<8>(in, n, out); break;
    case Filter::BcjX86: bcj_x86_inv(in, n, out); break;
    case Filter::BcjArm64: bcj_arm64_inv(in, n, out); break;
    case Filter::BitRound32: shuffle_inv(in, n, out, 4); break;
    case Filter::BitRound64: shuffle_inv(in, n, out, 8); break;
    case Filter::Lag2Shuffle32: fused_lag2_shuffle_inv<4, false>(in, n, out); break;
    case Filter::Lag2Shuffle64: fused_lag2_shuffle_inv<8, false>(in, n, out); break;
    case Filter::XorLag2Shuffle32: fused_lag2_shuffle_inv<4, true>(in, n, out); break;
    default: std::memcpy(out, in, n); break;
  }
}

// --- detection --------------------------------------------------------------

// Cheap proxy for compressibility: the order-0 entropy (in bits) of the byte
// stream's first difference. This is sensitive to byte *adjacency*, so unlike
// plain order-0 entropy it can tell a byte-shuffle apart from no transform
// (a shuffle that lines up constant high bytes makes the difference stream
// mostly zero). Lower is better.
inline double adjacency_cost_bits(const u8* buf, usize n) noexcept {
  if (n < 2) return 0.0;
  u32 freq[256] = {0};
  u8 prev = buf[0];
  for (usize i = 1; i < n; ++i) {
    ++freq[static_cast<u8>(buf[i] - prev)];
    prev = buf[i];
  }
  const double total = static_cast<double>(n - 1);
  double bits = 0.0;
  for (int s = 0; s < 256; ++s) {
    if (!freq[s]) continue;
    const double p = static_cast<double>(freq[s]) / total;
    bits -= static_cast<double>(freq[s]) * std::log2(p);
  }
  return bits;
}

// True when a sample looks like printable text/source rather than a numeric
// column. Little-endian int/float columns are full of 0x00 high bytes; natural
// language and code are mostly printable with very few NULs. Used only to
// raise the detect_filter acceptance bar (not to hard-reject).
inline bool sample_looks_like_text(const u8* s, usize n) noexcept {
  if (n < 64) return false;
  usize printable = 0;
  usize nul = 0;
  for (usize i = 0; i < n; ++i) {
    const u8 c = s[i];
    if (c == 0) {
      ++nul;
    } else if (c == 9 || c == 10 || c == 13 || (c >= 32 && c < 127)) {
      ++printable;
    }
  }
  return printable * 20 >= n * 17 &&  // >= 85% printable
         nul * 20 < n;                // < 5% NUL (columnar LE ints fail this)
}

// Pick the filter that minimizes the adjacency cost on a representative sample,
// requiring a clear margin over "no transform" so we never add a filter that
// barely helps. Returns Filter::None when nothing is clearly worthwhile. The
// encoder still compresses the chosen filter and the raw block and keeps the
// smaller, so a wrong guess here can never enlarge a block.
// By default the sample is 16 KiB from the middle of the block. When
// `full_sample` is true the whole block is scored (CompressParams::filter_sample_full).
inline Filter detect_filter(const u8* src, usize n, bool full_sample = false,
                            u64 mask = ~u64{0}) {
  if (n < 64) return Filter::None;
  if ((mask & ~u64{1}) == 0) return Filter::None;  // nothing but None allowed
  // Executables take the matching BCJ outright: the adjacency proxy below
  // cannot value a branch-target rewrite (see detect_exec_arch), and no
  // shuffle/delta candidate applies to an instruction stream.
  {
    const Filter bcj = bcj_filter_for(detect_exec_arch(src, n));
    if (bcj != Filter::None) {
      if ((mask >> static_cast<unsigned>(bcj)) & 1ull) return bcj;
      return Filter::None;
    }
  }
  constexpr usize kSample = usize{1} << 14;  // 16 KiB is plenty to characterize
  const usize sn = full_sample ? n : min_of(n, kSample);
  const usize start = (!full_sample && n > sn) ? (n - sn) / 2 : 0;
  const u8* s = src + start;

  static thread_local std::vector<u8> tmp;
  tmp.resize(sn);

  const double none_cost = adjacency_cost_bits(s, sn);
  double best_cost = none_cost;
  Filter best = Filter::None;
  double cost_of[static_cast<unsigned>(Filter::kMaxValue) + 1];
  for (double& c : cost_of) c = 1e300;

  const Filter cands[] = {
      Filter::Shuffle16,     Filter::Shuffle32,
      Filter::Shuffle64,     Filter::Delta16,
      Filter::Delta32,       Filter::Delta64,
      Filter::ZigzagDelta16, Filter::ZigzagDelta32,
      Filter::ZigzagDelta64, Filter::DeltaShuffle32,
      Filter::ZigzagDeltaShuffle32, Filter::BitShuffle32,
      Filter::DeltaShuffle64, Filter::ZigzagShuffle32,
      Filter::ZigzagShuffle64, Filter::BitShuffle64,
      Filter::DeltaBitShuffle32,
      Filter::XorShuffle32,  Filter::XorShuffle64,
      Filter::Delta2Shuffle32, Filter::ZigzagDelta2Shuffle32,
      Filter::Delta2Shuffle64,
      Filter::Lag2Shuffle32, Filter::Lag2Shuffle64,
      Filter::XorLag2Shuffle32,
      Filter::BitRound32,    Filter::BitRound64};
  auto is_lag2 = [](Filter f) {
    return f == Filter::Lag2Shuffle32 || f == Filter::Lag2Shuffle64 ||
           f == Filter::XorLag2Shuffle32;
  };
  double best_lag2_cost = 1e300;
  Filter best_lag2 = Filter::None;
  for (Filter f : cands) {
    if (!((mask >> static_cast<unsigned>(f)) & 1ull)) continue;
    if ((f == Filter::BitRound32 || f == Filter::BitRound64) && mask == ~u64{0})
      continue;  // Lossy mantissa rounding must be explicitly requested in filter mask
    apply_filter(f, s, sn, tmp.data());
    const double c = adjacency_cost_bits(tmp.data(), sn);
    cost_of[static_cast<unsigned>(f)] = c;
    if (is_lag2(f)) {
      if (c < best_lag2_cost) {
        best_lag2_cost = c;
        best_lag2 = f;
      }
      continue;
    }
    if (c < best_cost) {
      best_cost = c;
      best = f;
    }
  }
  // The lag-2 filters are judged against their lag-1 counterpart, not just
  // the field: a lag-2 residual is the sum of two lag-1 residuals, so where
  // lag-1 fits (ramps, timestamps) lag-2 is wider in truth even when the
  // adjacency proxy says otherwise by a few percent. Lag-2 earns the
  // nomination only on period-2 structure (interleaved pairs), where lag-1
  // is poor and the proxy gap is large; there it also takes precedence over
  // a field winner within 5% (the proxy under-rates it against bitshuffle,
  // the real codecs are 5-7% smaller). Otherwise the pick is exactly what it
  // would have been without the lag-2 candidates.
  if (best_lag2 != Filter::None && best_lag2_cost <= best_cost * 1.05) {
    const Filter lag1 = best_lag2 == Filter::Lag2Shuffle32   ? Filter::DeltaShuffle32
                        : best_lag2 == Filter::Lag2Shuffle64 ? Filter::DeltaShuffle64
                                                             : Filter::XorShuffle32;
    const double c1 = cost_of[static_cast<unsigned>(lag1)];
    // With the lag-1 counterpart masked out (e.g. a single-bit mask) there is
    // no reference: fall back to the plain field comparison.
    const bool period2 = c1 < 1e300 ? best_lag2_cost <= c1 * 0.85 : best_lag2_cost < best_cost;
    if (period2) {
      best = best_lag2;
      best_cost = best_lag2_cost;
    }
  }
  // Numeric/columnar: require >= 3% adjacency improvement. Text/source: 10% —
  // weak shuffle/delta nominations on prose often beat LZ yet lose to plain
  // BWT; the keep-smaller path covers correctness, this cuts wasted encodes.
  const double margin = sample_looks_like_text(s, sn) ? 0.90 : 0.97;
  if (best != Filter::None && best_cost > none_cost * margin) return Filter::None;
  return best;
}

}  // namespace spectra8::detail

#endif  // SPECTRA8_SRC_TRANSFORM_HPP
