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

// Centralized SIMD / wide-copy helpers.
//
// The hot copy loops are written in portable C++ using fixed-size memcpy, which
// every modern compiler lowers to vector instructions (NEON on arm64, SSE on
// x86_64). When SPECTRA8_USE_SSE2NEON is enabled the vendored sse2neon.h header
// is pulled in so that explicit SSE intrinsics also map to NEON on arm64; on
// x86_64 the native SSE headers are used. Nothing in the codec hard-depends on
// raw intrinsics, so the library builds cleanly without sse2neon.
#ifndef SPECTRA8_SRC_SIMD_HPP
#define SPECTRA8_SRC_SIMD_HPP

#include <cstddef>
#include <cstdint>
#include <cstring>

#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#define SPECTRA8_HAVE_NEON 1
#else
#define SPECTRA8_HAVE_NEON 0
#endif

#if defined(__AVX2__)
#include <immintrin.h>
#define SPECTRA8_HAVE_AVX2 1
#else
#define SPECTRA8_HAVE_AVX2 0
#endif
#if defined(_MSC_VER)
#include <intrin.h>
#endif

#if defined(SPECTRA8_USE_SSE2NEON) && (defined(__aarch64__) || defined(__arm__))
#include "sse2neon.h"
#define SPECTRA8_HAVE_SSE 1
#elif defined(__SSE2__) || defined(_M_X64) || defined(_M_IX86_FP)
#include <emmintrin.h>
#define SPECTRA8_HAVE_SSE 1
#else
#define SPECTRA8_HAVE_SSE 0
#endif

namespace spectra8::detail {

inline void copy8(std::uint8_t* dst, const std::uint8_t* src) noexcept {
  std::memcpy(dst, src, 8);
}
inline void copy16(std::uint8_t* dst, const std::uint8_t* src) noexcept {
  std::memcpy(dst, src, 16);
}

// Copy [src, src_end) to dst, possibly writing a few extra bytes past the end
// in 16-byte chunks. Callers must guarantee the destination has slack
// (kCopySlack bytes). Used for non-overlapping or far-overlapping copies.
inline constexpr std::size_t kCopySlack = 16;

inline void wild_copy(std::uint8_t* dst, const std::uint8_t* src,
                      std::uint8_t* dst_end) noexcept {
  do {
    copy16(dst, src);
    dst += 16;
    src += 16;
  } while (dst < dst_end);
}

inline unsigned ctz64(std::uint64_t x) noexcept {
#if defined(_MSC_VER)
  unsigned long i = 0;
  _BitScanForward64(&i, x);
  return static_cast<unsigned>(i);
#else
  return static_cast<unsigned>(__builtin_ctzll(x));
#endif
}

// Leading run of bytes equal to `v`. AVX2 does 32 B/iter, NEON two Q-regs
// (same 32 B), otherwise two 64-bit words. Used for MTF zero-runs and RLE0.
inline std::size_t count_eq_run(const std::uint8_t* p, std::size_t n,
                                std::uint8_t v) noexcept {
  std::size_t i = 0;
#if SPECTRA8_HAVE_AVX2
  const __m256i vv = _mm256_set1_epi8(static_cast<char>(v));
  for (; i + 32 <= n; i += 32) {
    const __m256i x = _mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(p + i));
    const unsigned mask =
        static_cast<unsigned>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(x, vv)));
    if (mask != 0xFFFFFFFFu)
      return i + static_cast<std::size_t>(ctz64(static_cast<std::uint64_t>(~mask)));
  }
#elif SPECTRA8_HAVE_NEON
  const uint8x16_t vv = vdupq_n_u8(v);
  for (; i + 32 <= n; i += 32) {
    const uint8x16_t a = veorq_u8(vld1q_u8(p + i), vv);
    const uint8x16_t b = veorq_u8(vld1q_u8(p + i + 16), vv);
    if (vmaxvq_u8(vorrq_u8(a, b)) != 0) break;
  }
#elif SPECTRA8_HAVE_SSE
  const __m128i vv = _mm_set1_epi8(static_cast<char>(v));
  for (; i + 16 <= n; i += 16) {
    const __m128i x = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p + i));
    const unsigned mask = static_cast<unsigned>(_mm_movemask_epi8(_mm_cmpeq_epi8(x, vv)));
    if (mask != 0xFFFFu)
      return i + static_cast<std::size_t>(ctz64(static_cast<std::uint64_t>(mask ^ 0xFFFFu)));
  }
#endif
  const std::uint64_t splat = 0x0101010101010101ULL * static_cast<std::uint64_t>(v);
  for (; i + 16 <= n; i += 16) {
    std::uint64_t a, b;
    std::memcpy(&a, p + i, 8);
    std::memcpy(&b, p + i + 8, 8);
    a ^= splat;
    b ^= splat;
    if (a | b) {
      if (a) return i + (static_cast<std::size_t>(ctz64(a)) >> 3);
      return i + 8 + (static_cast<std::size_t>(ctz64(b)) >> 3);
    }
  }
  for (; i + 8 <= n; i += 8) {
    std::uint64_t a;
    std::memcpy(&a, p + i, 8);
    a ^= splat;
    if (a) return i + (static_cast<std::size_t>(ctz64(a)) >> 3);
  }
  for (; i < n && p[i] == v; ++i) {
  }
  return i;
}

// Leading run of bytes not equal to `v` (e.g. MTF non-zeros until the next 0).
inline std::size_t count_neq_run(const std::uint8_t* p, std::size_t n,
                                 std::uint8_t v) noexcept {
  std::size_t i = 0;
#if SPECTRA8_HAVE_AVX2
  const __m256i vv = _mm256_set1_epi8(static_cast<char>(v));
  for (; i + 32 <= n; i += 32) {
    const __m256i x = _mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(p + i));
    const unsigned mask =
        static_cast<unsigned>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(x, vv)));
    if (mask != 0)
      return i + static_cast<std::size_t>(ctz64(static_cast<std::uint64_t>(mask)));
  }
#elif SPECTRA8_HAVE_NEON
  const uint8x16_t vv = vdupq_n_u8(v);
  for (; i + 32 <= n; i += 32) {
    const uint8x16_t a = vceqq_u8(vld1q_u8(p + i), vv);
    const uint8x16_t b = vceqq_u8(vld1q_u8(p + i + 16), vv);
    if (vmaxvq_u8(vorrq_u8(a, b)) != 0) break;
  }
#elif SPECTRA8_HAVE_SSE
  const __m128i vv = _mm_set1_epi8(static_cast<char>(v));
  for (; i + 16 <= n; i += 16) {
    const __m128i x = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p + i));
    const unsigned mask = static_cast<unsigned>(_mm_movemask_epi8(_mm_cmpeq_epi8(x, vv)));
    if (mask != 0)
      return i + static_cast<std::size_t>(ctz64(static_cast<std::uint64_t>(mask)));
  }
#endif
  const std::uint64_t splat = 0x0101010101010101ULL * static_cast<std::uint64_t>(v);
  for (; i + 16 <= n; i += 16) {
    std::uint64_t a, b;
    std::memcpy(&a, p + i, 8);
    std::memcpy(&b, p + i + 8, 8);
    const std::uint64_t za = a ^ splat;
    const std::uint64_t zb = b ^ splat;
    // A zero byte in za means that lane equalled v.
    const std::uint64_t ha = (za - 0x0101010101010101ULL) & ~za & 0x8080808080808080ULL;
    const std::uint64_t hb = (zb - 0x0101010101010101ULL) & ~zb & 0x8080808080808080ULL;
    if (ha | hb) {
      if (ha) return i + (static_cast<std::size_t>(ctz64(ha)) >> 3);
      return i + 8 + (static_cast<std::size_t>(ctz64(hb)) >> 3);
    }
  }
  for (; i + 8 <= n; i += 8) {
    std::uint64_t a;
    std::memcpy(&a, p + i, 8);
    const std::uint64_t za = a ^ splat;
    const std::uint64_t ha =
        (za - 0x0101010101010101ULL) & ~za & 0x8080808080808080ULL;
    if (ha) return i + (static_cast<std::size_t>(ctz64(ha)) >> 3);
  }
  for (; i < n && p[i] != v; ++i) {
  }
  return i;
}

// Four-way byte histogram: each lane hits a private 256-bucket array so
// back-to-back equal bytes (BWT/MTF runs) do not RAW-stall one counter.
// Four arrays (4 KiB) stay in L1; eight-way was measured no faster here.
inline void hist_u8(const std::uint8_t* src, std::size_t n,
                    std::uint32_t freq[256]) noexcept {
  alignas(64) std::uint32_t c[4][256];
  std::memset(c, 0, sizeof(c));
  std::size_t i = 0;
  for (; i + 16 <= n; i += 16) {
    ++c[0][src[i + 0]];
    ++c[1][src[i + 1]];
    ++c[2][src[i + 2]];
    ++c[3][src[i + 3]];
    ++c[0][src[i + 4]];
    ++c[1][src[i + 5]];
    ++c[2][src[i + 6]];
    ++c[3][src[i + 7]];
    ++c[0][src[i + 8]];
    ++c[1][src[i + 9]];
    ++c[2][src[i + 10]];
    ++c[3][src[i + 11]];
    ++c[0][src[i + 12]];
    ++c[1][src[i + 13]];
    ++c[2][src[i + 14]];
    ++c[3][src[i + 15]];
  }
  for (; i + 4 <= n; i += 4) {
    ++c[0][src[i + 0]];
    ++c[1][src[i + 1]];
    ++c[2][src[i + 2]];
    ++c[3][src[i + 3]];
  }
  for (; i < n; ++i) ++c[0][src[i]];
  for (int s = 0; s < 256; ++s)
    freq[s] = c[0][s] + c[1][s] + c[2][s] + c[3][s];
}

}  // namespace spectra8::detail

#endif  // SPECTRA8_SRC_SIMD_HPP
