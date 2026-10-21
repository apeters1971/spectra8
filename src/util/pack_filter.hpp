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

// Self-framed numeric filters that are not size-preserving byte shuffles:
//
//   forW  (Filter::For16/32/64)
//     Frame-of-reference. Treat the block as signed little-endian W-byte
//     integers, subtract a frame minimum and bit-pack the residuals. A short
//     sliding window over the sorted values drops a few outliers (at most
//     1/32 of the block) so one spike does not widen every residual. Outliers
//     are stored raw, next to their element index.
//
//   alpW  (Filter::Alp32/64)
//     Decimal integerization for f32/f64, in the sense of ALP: search a power
//     of ten such that round(v * 10^e) / 10^e reproduces v for most elements,
//     then frame-of-reference pack those integers. Values that do not
//     round-trip (or are non-finite) are stored raw.
//
// Both decline (return false, write nothing) unless the framed stream is
// strictly smaller than the raw block. The encoder's keep-smaller trial is
// the other half of that guarantee.
//
// forW layout (LEB128 varints):
//   varint ne
//   u8     bw                  bits per inlier residual, 0..63
//   w bytes frame_min          signed little-endian reference
//   varint nexc
//   packed inlier residuals    LSB-first, (v - frame_min) in `bw` bits
//   nexc varints               exception indices (first absolute, then deltas >= 1)
//   nexc * w bytes             original element bytes of each exception
//   tail                       n % w raw bytes
//
// alpW layout:
//   varint ne
//   u8     exp_bias            exponent + 18, so 10^(bias-18) is the scale
//   u8     bw
//   8 bytes frame_min          signed little-endian integer reference
//   varint nexc
//   packed inlier integers
//   nexc index varints
//   nexc * w bytes             original float bytes
//   tail
#ifndef SPECTRA8_SRC_PACK_FILTER_HPP
#define SPECTRA8_SRC_PACK_FILTER_HPP

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include "util/dict_filter.hpp"
#include "util/digit_filter.hpp"
#include "util/base64_filter.hpp"
#include "util/word_filter.hpp"

namespace spectra8::detail {

namespace pack_detail {

inline constexpr int kAlpExpMin = -18;
inline constexpr int kAlpExpMax = 18;

struct BitWriter {
  std::vector<u8>& out;
  u32 acc = 0;
  int n = 0;
  explicit BitWriter(std::vector<u8>& o) noexcept : out(o) {}
  void put(u64 v, int bw) {
    int left = bw;
    while (left > 0) {
      const int room = 8 - n;
      const int take = left < room ? left : room;
      acc |= static_cast<u32>(v & ((1u << take) - 1)) << n;
      n += take;
      v >>= static_cast<unsigned>(take);
      left -= take;
      if (n == 8) {
        out.push_back(static_cast<u8>(acc));
        acc = 0;
        n = 0;
      }
    }
  }
  void flush() {
    if (n) out.push_back(static_cast<u8>(acc));
  }
};

struct BitReader {
  const u8* p;
  const u8* end;
  u32 acc = 0;
  int n = 0;
  bool ok = true;
  BitReader(const u8* data, usize len) noexcept : p(data), end(data + len) {}
  u64 get(int bw) {
    u64 v = 0;
    int filled = 0;
    while (filled < bw) {
      if (n == 0) {
        if (p >= end) {
          ok = false;
          return 0;
        }
        acc = *p++;
        n = 8;
      }
      int take = bw - filled;
      if (take > n) take = n;
      v |= static_cast<u64>(acc & ((1u << take) - 1)) << filled;
      acc >>= static_cast<unsigned>(take);
      n -= take;
      filled += take;
    }
    return v;
  }
};

inline i64 load_signed_w(const u8* p, int w) noexcept {
  const u64 u = load_le(p, w);
  if (w == 2) return static_cast<i64>(static_cast<std::int16_t>(static_cast<u16>(u)));
  if (w == 4) return static_cast<i64>(static_cast<i32>(static_cast<u32>(u)));
  return static_cast<i64>(u);
}

inline u64 read_residual_at(const u8* packed, usize packed_len, usize inlier_idx, int bw) noexcept {
  if (bw == 0) return 0;
  const usize bit_pos = inlier_idx * static_cast<usize>(bw);
  const usize byte_pos = bit_pos >> 3;
  const int bit_shift = static_cast<int>(bit_pos & 7);
  const u64 mask = bw >= 64 ? ~u64{0} : ((u64{1} << bw) - 1);
  if (byte_pos + 8 <= packed_len) {
    return (load_u64(packed + byte_pos) >> bit_shift) & mask;
  }
  u64 word = 0;
  const usize avail = packed_len > byte_pos ? packed_len - byte_pos : 0;
  for (usize i = 0; i < avail; ++i) {
    word |= static_cast<u64>(packed[byte_pos + i]) << (i * 8);
  }
  return (word >> bit_shift) & mask;
}

// Accelerated bit reader maintaining a 64/128-bit accumulator with word refill.
struct FastBitReader {
#if defined(__SIZEOF_INT128__)
  using u128 = unsigned __int128;
  u128 acc = 0;
#else
  u64 acc = 0;
#endif
  int bits_in_acc = 0;
  const u8* packed;
  usize packed_len;
  usize byte_pos;
  int bw;
  u64 mask;

  FastBitReader(const u8* p, usize len, usize ki_start, int bit_width) noexcept
      : packed(p), packed_len(len), bw(bit_width) {
    mask = (bw >= 64) ? ~u64{0} : ((u64{1} << bw) - 1);
    if (bw > 0) {
      const usize start_bit = ki_start * static_cast<usize>(bw);
      byte_pos = start_bit >> 3;
      const int bit_shift = static_cast<int>(start_bit & 7);
      if (byte_pos + 8 <= packed_len) {
        acc = load_u64(packed + byte_pos);
        acc >>= bit_shift;
        bits_in_acc = 64 - bit_shift;
        byte_pos += 8;
      }
    } else {
      byte_pos = 0;
    }
  }

  inline u64 next(usize ki) noexcept {
    if (bw <= 0) return 0;
#if defined(__SIZEOF_INT128__)
    if (bits_in_acc < 64 && byte_pos + 8 <= packed_len) {
      acc |= (static_cast<u128>(load_u64(packed + byte_pos)) << bits_in_acc);
      byte_pos += 8;
      bits_in_acc += 64;
    }
    if (bits_in_acc >= bw) {
      const u64 res = static_cast<u64>(acc) & mask;
      acc >>= bw;
      bits_in_acc -= bw;
      return res;
    }
#else
    if (bw <= 32) {
      if (bits_in_acc < 32 && byte_pos + 4 <= packed_len) {
        acc |= (static_cast<u64>(load_u32(packed + byte_pos)) << bits_in_acc);
        byte_pos += 4;
        bits_in_acc += 32;
      }
      if (bits_in_acc >= bw) {
        const u64 res = acc & mask;
        acc >>= bw;
        bits_in_acc -= bw;
        return res;
      }
    }
#endif
    return read_residual_at(packed, packed_len, ki, bw);
  }
};

inline void unpack_inliers_span(const u8* packed, usize packed_len, u8* dst,
                                usize e_start, usize e_end, usize ki_start,
                                int bw, i64 frame_min, int w) noexcept {
  if (bw == 0) {
    if (w == 2) {
      const u16 v = static_cast<u16>(frame_min);
      for (usize i = e_start; i < e_end; ++i) store_u16(dst + i * 2, v);
    } else if (w == 4) {
      const u32 v = static_cast<u32>(frame_min);
      for (usize i = e_start; i < e_end; ++i) store_u32(dst + i * 4, v);
    } else if (w == 8) {
      const u64 v = static_cast<u64>(frame_min);
      for (usize i = e_start; i < e_end; ++i) store_u64(dst + i * 8, v);
    }
    return;
  }

  FastBitReader reader(packed, packed_len, ki_start, bw);
  usize e = e_start;
  usize ki = ki_start;

  if (w == 4) {
#if defined(SPECTRA8_TRANSFORM_NEON)
    const uint32x4_t fmin_vec = vdupq_n_u32(static_cast<u32>(frame_min));
    for (; e + 4 <= e_end; e += 4, ki += 4) {
      const u32 r0 = static_cast<u32>(reader.next(ki));
      const u32 r1 = static_cast<u32>(reader.next(ki + 1));
      const u32 r2 = static_cast<u32>(reader.next(ki + 2));
      const u32 r3 = static_cast<u32>(reader.next(ki + 3));
      const uint32x4_t rvec = {r0, r1, r2, r3};
      vst1q_u32(reinterpret_cast<u32*>(dst + e * 4), vaddq_u32(rvec, fmin_vec));
    }
#elif defined(SPECTRA8_TRANSFORM_SSE2)
    const __m128i fmin_vec = _mm_set1_epi32(static_cast<int>(frame_min));
    for (; e + 4 <= e_end; e += 4, ki += 4) {
      const u32 r0 = static_cast<u32>(reader.next(ki));
      const u32 r1 = static_cast<u32>(reader.next(ki + 1));
      const u32 r2 = static_cast<u32>(reader.next(ki + 2));
      const u32 r3 = static_cast<u32>(reader.next(ki + 3));
      const __m128i rvec = _mm_set_epi32(static_cast<int>(r3), static_cast<int>(r2),
                                         static_cast<int>(r1), static_cast<int>(r0));
      _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + e * 4), _mm_add_epi32(rvec, fmin_vec));
    }
#else
    for (; e + 4 <= e_end; e += 4, ki += 4) {
      const u32 r0 = static_cast<u32>(reader.next(ki));
      const u32 r1 = static_cast<u32>(reader.next(ki + 1));
      const u32 r2 = static_cast<u32>(reader.next(ki + 2));
      const u32 r3 = static_cast<u32>(reader.next(ki + 3));
      store_u32(dst + (e + 0) * 4, static_cast<u32>(frame_min) + r0);
      store_u32(dst + (e + 1) * 4, static_cast<u32>(frame_min) + r1);
      store_u32(dst + (e + 2) * 4, static_cast<u32>(frame_min) + r2);
      store_u32(dst + (e + 3) * 4, static_cast<u32>(frame_min) + r3);
    }
#endif
  } else if (w == 2) {
    for (; e + 4 <= e_end; e += 4, ki += 4) {
      const u16 r0 = static_cast<u16>(reader.next(ki));
      const u16 r1 = static_cast<u16>(reader.next(ki + 1));
      const u16 r2 = static_cast<u16>(reader.next(ki + 2));
      const u16 r3 = static_cast<u16>(reader.next(ki + 3));
      store_u16(dst + (e + 0) * 2, static_cast<u16>(frame_min + r0));
      store_u16(dst + (e + 1) * 2, static_cast<u16>(frame_min + r1));
      store_u16(dst + (e + 2) * 2, static_cast<u16>(frame_min + r2));
      store_u16(dst + (e + 3) * 2, static_cast<u16>(frame_min + r3));
    }
  } else if (w == 8) {
    for (; e + 2 <= e_end; e += 2, ki += 2) {
      const u64 r0 = reader.next(ki);
      const u64 r1 = reader.next(ki + 1);
      store_u64(dst + (e + 0) * 8, static_cast<u64>(frame_min) + r0);
      store_u64(dst + (e + 1) * 8, static_cast<u64>(frame_min) + r1);
    }
  }

  for (; e < e_end; ++e, ++ki) {
    const u64 residual = reader.next(ki);
    const u64 v = static_cast<u64>(frame_min) + residual;
    store_le(dst + e * static_cast<usize>(w), v, w);
  }
}

inline void unpack_alp_inliers_span(const u8* packed, usize packed_len, u8* dst,
                                    usize e_start, usize e_end, usize ki_start,
                                    int bw, i64 frame_min, double inv, int w) noexcept {
  if (bw == 0) {
    if (w == 4) {
      const float back = static_cast<float>(static_cast<double>(frame_min) * inv);
      u32 bits;
      std::memcpy(&bits, &back, 4);
      for (usize i = e_start; i < e_end; ++i) store_u32(dst + i * 4, bits);
    } else if (w == 8) {
      const double back = static_cast<double>(frame_min) * inv;
      u64 bits;
      std::memcpy(&bits, &back, 8);
      for (usize i = e_start; i < e_end; ++i) store_u64(dst + i * 8, bits);
    }
    return;
  }

  FastBitReader reader(packed, packed_len, ki_start, bw);
  usize e = e_start;
  usize ki = ki_start;

  if (w == 4) {
#if defined(SPECTRA8_TRANSFORM_NEON)
    const float32x4_t inv_vec = vdupq_n_f32(static_cast<float>(inv));
    const int32x4_t fmin_vec = vdupq_n_s32(static_cast<i32>(frame_min));
    for (; e + 4 <= e_end; e += 4, ki += 4) {
      const u32 r0 = static_cast<u32>(reader.next(ki));
      const u32 r1 = static_cast<u32>(reader.next(ki + 1));
      const u32 r2 = static_cast<u32>(reader.next(ki + 2));
      const u32 r3 = static_cast<u32>(reader.next(ki + 3));
      const int32x4_t enc_vec = vaddq_s32(int32x4_t{static_cast<i32>(r0), static_cast<i32>(r1),
                                                    static_cast<i32>(r2), static_cast<i32>(r3)},
                                          fmin_vec);
      const float32x4_t f_vec = vmulq_f32(vcvtq_f32_s32(enc_vec), inv_vec);
      vst1q_f32(reinterpret_cast<float*>(dst + e * 4), f_vec);
    }
#elif defined(SPECTRA8_TRANSFORM_SSE2)
    const __m128 inv_vec = _mm_set1_ps(static_cast<float>(inv));
    const __m128i fmin_vec = _mm_set1_epi32(static_cast<int>(frame_min));
    for (; e + 4 <= e_end; e += 4, ki += 4) {
      const u32 r0 = static_cast<u32>(reader.next(ki));
      const u32 r1 = static_cast<u32>(reader.next(ki + 1));
      const u32 r2 = static_cast<u32>(reader.next(ki + 2));
      const u32 r3 = static_cast<u32>(reader.next(ki + 3));
      const __m128i enc_vec = _mm_add_epi32(_mm_set_epi32(static_cast<int>(r3), static_cast<int>(r2),
                                                          static_cast<int>(r1), static_cast<int>(r0)),
                                            fmin_vec);
      const __m128 f_vec = _mm_mul_ps(_mm_cvtepi32_ps(enc_vec), inv_vec);
      _mm_storeu_ps(reinterpret_cast<float*>(dst + e * 4), f_vec);
    }
#else
    for (; e + 4 <= e_end; e += 4, ki += 4) {
      const u32 r0 = static_cast<u32>(reader.next(ki));
      const u32 r1 = static_cast<u32>(reader.next(ki + 1));
      const u32 r2 = static_cast<u32>(reader.next(ki + 2));
      const u32 r3 = static_cast<u32>(reader.next(ki + 3));
      const float f0 = static_cast<float>(static_cast<double>(static_cast<i64>(frame_min + r0)) * inv);
      const float f1 = static_cast<float>(static_cast<double>(static_cast<i64>(frame_min + r1)) * inv);
      const float f2 = static_cast<float>(static_cast<double>(static_cast<i64>(frame_min + r2)) * inv);
      const float f3 = static_cast<float>(static_cast<double>(static_cast<i64>(frame_min + r3)) * inv);
      std::memcpy(dst + (e + 0) * 4, &f0, 4);
      std::memcpy(dst + (e + 1) * 4, &f1, 4);
      std::memcpy(dst + (e + 2) * 4, &f2, 4);
      std::memcpy(dst + (e + 3) * 4, &f3, 4);
    }
#endif
  } else if (w == 8) {
    for (; e + 2 <= e_end; e += 2, ki += 2) {
      const u64 r0 = reader.next(ki);
      const u64 r1 = reader.next(ki + 1);
      const i64 enc0 = static_cast<i64>(static_cast<u64>(frame_min) + r0);
      const i64 enc1 = static_cast<i64>(static_cast<u64>(frame_min) + r1);
      const double back0 = static_cast<double>(enc0) * inv;
      const double back1 = static_cast<double>(enc1) * inv;
      std::memcpy(dst + (e + 0) * 8, &back0, 8);
      std::memcpy(dst + (e + 1) * 8, &back1, 8);
    }
  }

  for (; e < e_end; ++e, ++ki) {
    const u64 residual = reader.next(ki);
    const i64 enc = static_cast<i64>(static_cast<u64>(frame_min) + residual);
    if (w == 4) {
      const float back = static_cast<float>(static_cast<double>(enc) * inv);
      std::memcpy(dst + e * 4, &back, 4);
    } else {
      const double back = static_cast<double>(enc) * inv;
      std::memcpy(dst + e * 8, &back, 8);
    }
  }
}

inline int span_bits(u64 span) noexcept {
  if (span == 0) return 0;
  return 64 - __builtin_clzll(span);
}

struct ForPlan {
  bool ok = false;
  i64 lo = 0;
  i64 hi = 0;
  int bw = 0;
  usize est = 0;
};

inline ForPlan consider_frame(i64 lo, i64 hi, usize ne, usize nexc, int w) noexcept {
  ForPlan p;
  if (hi < lo || nexc >= ne) return p;
  const u64 span = static_cast<u64>(hi) - static_cast<u64>(lo);
  const int bw = span_bits(span);
  if (bw >= 64 || bw >= w * 8) return p;
  p.ok = true;
  p.lo = lo;
  p.hi = hi;
  p.bw = bw;
  p.est = 8 + static_cast<usize>(w) +
          ((ne - nexc) * static_cast<usize>(bw) + 7) / 8 +
          nexc * (static_cast<usize>(w) + 2);
  return p;
}

inline ForPlan plan_for(const i64* v, usize ne, int w) {
  i64 lo = v[0], hi = v[0];
  for (usize i = 1; i < ne; ++i) {
    if (v[i] < lo) lo = v[i];
    if (v[i] > hi) hi = v[i];
  }
  ForPlan best = consider_frame(lo, hi, ne, 0, w);
  if (ne >= 64) {
    static thread_local std::vector<i64> sorted;
    sorted.assign(v, v + ne);
    std::sort(sorted.begin(), sorted.end());
    for (usize budget : {ne / 256, ne / 32}) {
      if (budget == 0 || budget >= ne) continue;
      const usize cover = ne - budget;
      u64 best_span = ~u64{0};
      usize bl = 0;
      for (usize l = 0; l + cover <= ne; ++l) {
        const u64 sp = static_cast<u64>(sorted[l + cover - 1]) -
                       static_cast<u64>(sorted[l]);
        if (sp < best_span) {
          best_span = sp;
          bl = l;
        }
      }
      const ForPlan cand =
          consider_frame(sorted[bl], sorted[bl + cover - 1], ne, budget, w);
      if (cand.ok && (!best.ok || cand.est < best.est)) best = cand;
    }
  }
  return best;
}

inline const double* pow10_table() noexcept {
  static const double table[] = {
      1e-18, 1e-17, 1e-16, 1e-15, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10, 1e-9,
      1e-8,  1e-7,  1e-6,  1e-5,  1e-4,  1e-3,  1e-2,  1e-1,  1e0,   1e1,
      1e2,   1e3,   1e4,   1e5,   1e6,   1e7,   1e8,   1e9,   1e10,  1e11,
      1e12,  1e13,  1e14,  1e15,  1e16,  1e17,  1e18};
  return table;
}

inline bool alp_encode_one(const u8* p, int w, double scale, double inv, i64& enc) noexcept {
  double v;
  if (w == 4) {
    float f;
    std::memcpy(&f, p, 4);
    if (!std::isfinite(f)) return false;
    v = static_cast<double>(f);
  } else {
    std::memcpy(&v, p, 8);
    if (!std::isfinite(v)) return false;
  }
  const double scaled = v * scale;
  if (!std::isfinite(scaled)) return false;
  const double r = std::round(scaled);
  if (r > static_cast<double>(std::numeric_limits<i64>::max()) ||
      r < static_cast<double>(std::numeric_limits<i64>::min()))
    return false;
  enc = static_cast<i64>(r);
  if (static_cast<double>(enc) != r) return false;
  if (w == 4) {
    const float back = static_cast<float>(static_cast<double>(enc) * inv);
    float orig;
    std::memcpy(&orig, p, 4);
    return std::memcmp(&back, &orig, 4) == 0;
  }
  const double back = static_cast<double>(enc) * inv;
  double orig;
  std::memcpy(&orig, p, 8);
  return std::memcmp(&back, &orig, 8) == 0;
}

struct AlpPlan {
  bool ok = false;
  int exp = 0;
  int bw = 0;
  i64 lo = 0;
  usize nexc = 0;
  usize est = 0;
};

inline AlpPlan plan_alp(const u8* src, usize n, int w) {
  AlpPlan best;
  const usize ne = n / static_cast<usize>(w);
  if (ne < 16) return best;
  const usize max_exc = ne / 8;
  const double* pow10 = pow10_table();
  for (int e = kAlpExpMin; e <= kAlpExpMax; ++e) {
    const double scale = pow10[e - kAlpExpMin];
    const double inv = pow10[(kAlpExpMax - e)];  // 10^(-e), index of -e
    // index of -e: -e - kAlpExpMin = -e + 18. And kAlpExpMax - e = 18 - e.
    // -e + 18 == 18 - e. Yes.
    if (!std::isfinite(scale) || !std::isfinite(inv) || scale == 0.0) continue;

    // Fast pre-filter on a small prefix: an exponent that fails > 25% of the
    // first 32 elements cannot satisfy nexc <= ne / 8 (12.5%) across the block.
    if (ne >= 48) {
      usize pre_exc = 0;
      for (usize i = 0; i < 32; ++i) {
        i64 enc = 0;
        if (!alp_encode_one(src + i * static_cast<usize>(w), w, scale, inv, enc)) {
          if (++pre_exc > 8) break;
        }
      }
      if (pre_exc > 8) continue;
    }

    i64 lo = 0, hi = 0;
    usize nexc = 0;
    bool any = false;
    for (usize i = 0; i < ne; ++i) {
      i64 enc = 0;
      if (!alp_encode_one(src + i * static_cast<usize>(w), w, scale, inv, enc)) {
        if (++nexc > max_exc) break;
        continue;
      }
      if (!any) {
        lo = hi = enc;
        any = true;
      } else {
        if (enc < lo) lo = enc;
        if (enc > hi) hi = enc;
      }
    }
    if (nexc > max_exc || !any) continue;
    const u64 span = static_cast<u64>(hi) - static_cast<u64>(lo);
    const int bw = span_bits(span);
    if (bw >= 64 || bw >= w * 8) continue;
    const usize est = 16 + ((ne - nexc) * static_cast<usize>(bw) + 7) / 8 +
                      nexc * (static_cast<usize>(w) + 2);
    if (!best.ok || est < best.est) {
      best.ok = true;
      best.exp = e;
      best.bw = bw;
      best.lo = lo;
      best.nexc = nexc;
      best.est = est;
    }
  }
  return best;
}

inline void sample_span(const u8* src, usize n, bool full_sample, const u8*& s, usize& sn) noexcept {
  constexpr usize kSample = usize{1} << 14;
  sn = full_sample ? n : min_of(n, kSample);
  const usize start = ((!full_sample && n > sn) ? (n - sn) / 2 : 0) & ~usize{7};
  sn = n - start < sn ? n - start : sn;
  s = src + start;
}

inline bool add_fits(i64 frame_min, u64 bits, i64& out) noexcept {
  const i64 v = static_cast<i64>(static_cast<u64>(frame_min) + bits);
  if (v < frame_min) return false;
  if (static_cast<u64>(v) - static_cast<u64>(frame_min) != bits) return false;
  out = v;
  return true;
}

}  // namespace pack_detail

inline bool for_filter_fwd(const u8* src, usize n, int w, std::vector<u8>& out) {
  out.clear();
  if (w != 2 && w != 4 && w != 8) return false;
  const usize ne = n / static_cast<usize>(w);
  const usize tail = n - ne * static_cast<usize>(w);
  if (ne == 0) return false;

  static thread_local std::vector<i64> vals;
  vals.resize(ne);
  for (usize e = 0; e < ne; ++e)
    vals[e] = pack_detail::load_signed_w(src + e * static_cast<usize>(w), w);
  const pack_detail::ForPlan plan = pack_detail::plan_for(vals.data(), ne, w);
  if (!plan.ok || plan.est >= n) return false;

  static thread_local std::vector<u8> packed;
  static thread_local std::vector<u32> exc;
  packed.clear();
  exc.clear();
  pack_detail::BitWriter bits(packed);
  for (usize e = 0; e < ne; ++e) {
    const i64 v = vals[e];
    if (v < plan.lo || v > plan.hi) {
      if (e > static_cast<usize>(std::numeric_limits<u32>::max())) return false;
      exc.push_back(static_cast<u32>(e));
      continue;
    }
    bits.put(static_cast<u64>(v) - static_cast<u64>(plan.lo), plan.bw);
  }
  bits.flush();
  if (exc.size() >= ne) return false;

  ByteWriter wr(out);
  wr.varint(ne);
  wr.byte(static_cast<u8>(plan.bw));
  u8 ref[8];
  store_le(ref, static_cast<u64>(plan.lo), w);
  wr.bytes(ref, static_cast<usize>(w));
  wr.varint(exc.size());
  wr.bytes(packed);
  u64 prev = 0;
  for (usize i = 0; i < exc.size(); ++i) {
    const u64 idx = exc[i];
    wr.varint(i == 0 ? idx : idx - prev);
    prev = idx;
  }
  for (u32 e : exc) wr.bytes(src + static_cast<usize>(e) * static_cast<usize>(w),
                            static_cast<usize>(w));
  if (tail) wr.bytes(src + ne * static_cast<usize>(w), tail);
  if (out.size() >= n) {
    out.clear();
    return false;
  }
  return true;
}

inline Status for_filter_inv(const u8* data, usize len, u8* dst, usize n, int w) {
  if (w != 2 && w != 4 && w != 8) return Status::CorruptInput;
  ByteReader r(data, len);
  const u64 ne64 = r.varint();
  const u8 bw_b = r.byte();
  const u8* ref = r.take(static_cast<usize>(w));
  const u64 nexc64 = r.varint();
  if (r.error || !ref) return Status::CorruptInput;
  const usize ne = n / static_cast<usize>(w);
  const usize tail = n - ne * static_cast<usize>(w);
  if (ne64 != ne || ne == 0) return Status::CorruptInput;
  if (bw_b >= 64 || bw_b >= w * 8) return Status::CorruptInput;
  if (nexc64 >= ne) return Status::CorruptInput;
  const int bw = bw_b;
  const usize nexc = static_cast<usize>(nexc64);
  const i64 frame_min = pack_detail::load_signed_w(ref, w);
  const usize packed_len =
      ((ne - nexc) * static_cast<usize>(bw) + 7) / 8;
  const u8* packed = packed_len ? r.take(packed_len) : r.p;
  if (r.error || (packed_len && !packed)) return Status::CorruptInput;

  static thread_local std::vector<u32> exc;
  exc.resize(nexc);
  u64 prev = 0;
  for (usize i = 0; i < nexc; ++i) {
    const u64 d = r.varint();
    if (r.error) return Status::CorruptInput;
    const u64 idx = i == 0 ? d : prev + d;
    if ((i > 0 && d == 0) || idx >= ne || (i > 0 && idx <= prev))
      return Status::CorruptInput;
    exc[i] = static_cast<u32>(idx);
    prev = idx;
  }
  const u8* raw = nexc ? r.take(nexc * static_cast<usize>(w)) : nullptr;
  const u8* tp = tail ? r.take(tail) : nullptr;
  if (r.error || (nexc && !raw) || (tail && !tp) || r.remaining() != 0)
    return Status::CorruptInput;

  const i64 lim = w >= 8 ? 0 : (i64{1} << (w * 8 - 1));
  i64 max_v = 0;
  const bool fast_ok = (bw < w * 8) &&
                       pack_detail::add_fits(frame_min, bw == 0 ? 0 : ((u64{1} << bw) - 1), max_v) &&
                       (w >= 8 || (frame_min >= -lim && max_v < lim));

  usize ei = 0;
  usize ki = 0;
  usize e = 0;
  if (fast_ok) {
    while (ei < nexc) {
      const usize e_next = static_cast<usize>(exc[ei]);
      if (e < e_next) {
        pack_detail::unpack_inliers_span(packed, packed_len, dst, e, e_next, ki, bw, frame_min, w);
        ki += (e_next - e);
        e = e_next;
      }
      std::memcpy(dst + e * static_cast<usize>(w), raw + ei * static_cast<usize>(w), static_cast<usize>(w));
      ++ei;
      ++e;
    }
    if (e < ne) {
      pack_detail::unpack_inliers_span(packed, packed_len, dst, e, ne, ki, bw, frame_min, w);
    }
  } else {
    for (; e < ne; ++e) {
      if (ei < nexc && exc[ei] == e) {
        std::memcpy(dst + e * static_cast<usize>(w), raw + ei * static_cast<usize>(w), static_cast<usize>(w));
        ++ei;
        continue;
      }
      const u64 residual = pack_detail::read_residual_at(packed, packed_len, ki++, bw);
      i64 v = 0;
      if (!pack_detail::add_fits(frame_min, residual, v)) return Status::CorruptInput;
      if (w < 8 && (v < -lim || v >= lim)) return Status::CorruptInput;
      store_le(dst + e * static_cast<usize>(w), static_cast<u64>(v), w);
    }
  }
  if (ei != nexc) return Status::CorruptInput;
  if (tail) std::memcpy(dst + ne * static_cast<usize>(w), tp, tail);
  return Status::Ok;
}

inline bool alp_filter_fwd(const u8* src, usize n, int w, std::vector<u8>& out) {
  out.clear();
  if (w != 4 && w != 8) return false;
  const usize ne = n / static_cast<usize>(w);
  const usize tail = n - ne * static_cast<usize>(w);
  if (ne == 0) return false;
  const pack_detail::AlpPlan plan = pack_detail::plan_alp(src, n, w);
  if (!plan.ok || plan.est >= n) return false;

  const double* pow10 = pack_detail::pow10_table();
  const double scale = pow10[plan.exp - pack_detail::kAlpExpMin];
  const double inv = pow10[pack_detail::kAlpExpMax - plan.exp];

  static thread_local std::vector<u8> packed;
  static thread_local std::vector<u32> exc;
  packed.clear();
  exc.clear();
  pack_detail::BitWriter bits(packed);
  i64 lo = 0, hi = 0;
  bool any = false;
  for (usize e = 0; e < ne; ++e) {
    i64 enc = 0;
    if (!pack_detail::alp_encode_one(src + e * static_cast<usize>(w), w, scale, inv,
                                    enc)) {
      exc.push_back(static_cast<u32>(e));
      continue;
    }
    if (!any) {
      lo = hi = enc;
      any = true;
    } else {
      if (enc < lo) lo = enc;
      if (enc > hi) hi = enc;
    }
  }
  if (!any || exc.size() >= ne) return false;
  // The plan's frame came from the same scan rules; recompute bw from this
  // pass so a mismatch cannot emit a residual that does not fit.
  const int bw = pack_detail::span_bits(static_cast<u64>(hi) - static_cast<u64>(lo));
  if (bw >= 64 || bw >= w * 8) return false;
  for (usize e = 0; e < ne; ++e) {
    i64 enc = 0;
    if (!pack_detail::alp_encode_one(src + e * static_cast<usize>(w), w, scale, inv,
                                    enc))
      continue;
    bits.put(static_cast<u64>(enc) - static_cast<u64>(lo), bw);
  }
  bits.flush();

  ByteWriter wr(out);
  wr.varint(ne);
  wr.byte(static_cast<u8>(plan.exp - pack_detail::kAlpExpMin));
  wr.byte(static_cast<u8>(bw));
  u8 ref[8];
  store_le(ref, static_cast<u64>(lo), 8);
  wr.bytes(ref, 8);
  wr.varint(exc.size());
  wr.bytes(packed);
  u64 prev = 0;
  for (usize i = 0; i < exc.size(); ++i) {
    const u64 idx = exc[i];
    wr.varint(i == 0 ? idx : idx - prev);
    prev = idx;
  }
  for (u32 e : exc)
    wr.bytes(src + static_cast<usize>(e) * static_cast<usize>(w), static_cast<usize>(w));
  if (tail) wr.bytes(src + ne * static_cast<usize>(w), tail);
  if (out.size() >= n) {
    out.clear();
    return false;
  }
  return true;
}

inline Status alp_filter_inv(const u8* data, usize len, u8* dst, usize n, int w) {
  if (w != 4 && w != 8) return Status::CorruptInput;
  ByteReader r(data, len);
  const u64 ne64 = r.varint();
  const u8 bias = r.byte();
  const u8 bw_b = r.byte();
  const u8* ref = r.take(8);
  const u64 nexc64 = r.varint();
  if (r.error || !ref) return Status::CorruptInput;
  const usize ne = n / static_cast<usize>(w);
  const usize tail = n - ne * static_cast<usize>(w);
  if (ne64 != ne || ne == 0) return Status::CorruptInput;
  if (bias > static_cast<u8>(pack_detail::kAlpExpMax - pack_detail::kAlpExpMin))
    return Status::CorruptInput;
  if (bw_b >= 64 || bw_b >= w * 8) return Status::CorruptInput;
  if (nexc64 >= ne) return Status::CorruptInput;
  const int bw = bw_b;
  const usize nexc = static_cast<usize>(nexc64);
  const i64 frame_min = static_cast<i64>(load_le(ref, 8));
  const int exp = static_cast<int>(bias) + pack_detail::kAlpExpMin;
  const double inv = pack_detail::pow10_table()[pack_detail::kAlpExpMax - exp];
  if (!std::isfinite(inv)) return Status::CorruptInput;

  const usize packed_len = ((ne - nexc) * static_cast<usize>(bw) + 7) / 8;
  const u8* packed = packed_len ? r.take(packed_len) : r.p;
  if (r.error || (packed_len && !packed)) return Status::CorruptInput;

  static thread_local std::vector<u32> exc;
  exc.resize(nexc);
  u64 prev = 0;
  for (usize i = 0; i < nexc; ++i) {
    const u64 d = r.varint();
    if (r.error) return Status::CorruptInput;
    const u64 idx = i == 0 ? d : prev + d;
    if ((i > 0 && d == 0) || idx >= ne || (i > 0 && idx <= prev))
      return Status::CorruptInput;
    exc[i] = static_cast<u32>(idx);
    prev = idx;
  }
  const u8* raw = nexc ? r.take(nexc * static_cast<usize>(w)) : nullptr;
  const u8* tp = tail ? r.take(tail) : nullptr;
  if (r.error || (nexc && !raw) || (tail && !tp) || r.remaining() != 0)
    return Status::CorruptInput;

  i64 max_enc = 0;
  const bool fast_ok = (bw < w * 8) &&
                       pack_detail::add_fits(frame_min, bw == 0 ? 0 : ((u64{1} << bw) - 1), max_enc);

  usize ei = 0;
  usize ki = 0;
  usize e = 0;
  if (fast_ok) {
    while (ei < nexc) {
      const usize e_next = static_cast<usize>(exc[ei]);
      if (e < e_next) {
        pack_detail::unpack_alp_inliers_span(packed, packed_len, dst, e, e_next, ki, bw, frame_min, inv, w);
        ki += (e_next - e);
        e = e_next;
      }
      std::memcpy(dst + e * static_cast<usize>(w), raw + ei * static_cast<usize>(w), static_cast<usize>(w));
      ++ei;
      ++e;
    }
    if (e < ne) {
      pack_detail::unpack_alp_inliers_span(packed, packed_len, dst, e, ne, ki, bw, frame_min, inv, w);
    }
  } else {
    for (; e < ne; ++e) {
      u8* dp = dst + e * static_cast<usize>(w);
      if (ei < nexc && exc[ei] == e) {
        std::memcpy(dp, raw + ei * static_cast<usize>(w), static_cast<usize>(w));
        ++ei;
        continue;
      }
      const u64 residual = pack_detail::read_residual_at(packed, packed_len, ki++, bw);
      i64 enc = 0;
      if (!pack_detail::add_fits(frame_min, residual, enc)) return Status::CorruptInput;
      if (w == 4) {
        const float back = static_cast<float>(static_cast<double>(enc) * inv);
        std::memcpy(dp, &back, 4);
      } else {
        const double back = static_cast<double>(enc) * inv;
        std::memcpy(dp, &back, 8);
      }
    }
  }
  if (ei != nexc) return Status::CorruptInput;
  if (tail) std::memcpy(dst + ne * static_cast<usize>(w), tp, tail);
  return Status::Ok;
}

// Dictionary when the sample repeats, otherwise the tighter of a narrow
// integer frame and a decimal float frame. Returns Filter::None when neither
// looks smaller than the raw block.
inline Filter detect_framed_filter(const u8* src, usize n, bool full_sample = false,
                                   u64 mask = ~u64{0}, bool allow_dict = true) {
  if (n < 64) return Filter::None;
  // No framed representation fits an instruction stream; an executable's
  // filter is the BCJ that detect_filter nominates. A dictionary nominated
  // here would take precedence over it (and then lose to the raw block).
  if (detect_exec_arch(src, n) != ExecArch::None) return Filter::None;
  if (allow_dict) {
    const int dw = detect_dict_width(src, n, full_sample, mask);
    if (dw != 0) return dict_filter_id(dw);
  }
  const u8* s = src;
  usize sn = n;
  pack_detail::sample_span(src, n, full_sample, s, sn);
  if (((mask >> static_cast<unsigned>(Filter::Base64)) & 1ULL) != 0) {
    if (base64_detail::detect_base64(s, sn)) {
      return Filter::Base64;
    }
  }
  if (((mask >> static_cast<unsigned>(Filter::DigitNorm)) & 1ULL) != 0) {
    if (digit_norm_detail::detect_digit_norm(s, sn)) {
      return Filter::DigitNorm;
    }
  }
  if (((mask >> static_cast<unsigned>(Filter::WordNorm)) & 1ULL) != 0) {
    if (word_norm_detail::detect_word_norm(s, sn)) {
      return Filter::WordNorm;
    }
  }
  Filter best = Filter::None;
  double best_ratio = 0.85;
  auto consider = [&](Filter f, usize est) {
    if (est == 0 || sn == 0) return;
    const double ratio = static_cast<double>(est) / static_cast<double>(sn);
    if (ratio < best_ratio) {
      best_ratio = ratio;
      best = f;
    }
  };
  for (int w : {2, 4, 8}) {
    const Filter id = w == 2 ? Filter::For16 : w == 4 ? Filter::For32 : Filter::For64;
    if (((mask >> static_cast<unsigned>(id)) & 1u) == 0) continue;
    const usize ne = sn / static_cast<usize>(w);
    if (ne < 16) continue;
    static thread_local std::vector<i64> vals;
    vals.resize(ne);
    for (usize e = 0; e < ne; ++e)
      vals[e] = pack_detail::load_signed_w(s + e * static_cast<usize>(w), w);
    const pack_detail::ForPlan plan = pack_detail::plan_for(vals.data(), ne, w);
    if (plan.ok) consider(id, plan.est);
  }
  for (int w : {4, 8}) {
    const Filter id = w == 4 ? Filter::Alp32 : Filter::Alp64;
    if (((mask >> static_cast<unsigned>(id)) & 1u) == 0) continue;
    const pack_detail::AlpPlan plan = pack_detail::plan_alp(s, sn, w);
    if (plan.ok) consider(id, plan.est);
  }
  return best;
}

inline bool framed_filter_fwd(Filter f, const u8* src, usize n, std::vector<u8>& out) {
  const int w = framed_filter_width(f);
  if (filter_is_dict(f)) return dict_filter_fwd(src, n, w, out);
  if (filter_is_for(f)) return for_filter_fwd(src, n, w, out);
  if (filter_is_alp(f)) return alp_filter_fwd(src, n, w, out);
  if (filter_is_digit_norm(f)) return digit_norm_detail::digit_norm_fwd(src, n, out);
  if (filter_is_word_norm(f)) return word_norm_detail::word_norm_fwd(src, n, out);
  if (filter_is_base64(f)) return base64_detail::base64_fwd(src, n, out);
  return false;
}

inline Status framed_filter_inv(Filter f, const u8* data, usize len, u8* dst, usize n) {
  const int w = framed_filter_width(f);
  if (filter_is_dict(f)) return dict_filter_inv(data, len, dst, n, w);
  if (filter_is_for(f)) return for_filter_inv(data, len, dst, n, w);
  if (filter_is_alp(f)) return alp_filter_inv(data, len, dst, n, w);
  if (filter_is_digit_norm(f)) return digit_norm_detail::digit_norm_inv(data, len, dst, n);
  if (filter_is_word_norm(f)) return word_norm_detail::word_norm_inv(data, len, dst, n);
  if (filter_is_base64(f)) return base64_detail::base64_inv(data, len, dst, n);
  return Status::CorruptInput;
}

// Single-candidate staging (verdict, external codecs): a dictionary always
// wins when it builds. A for/alp stream wins when its own adjacency cost is
// at least 3% under the byte-filtered block, so a dense packed buffer does
// not displace a shuffle the entropy coder would have preferred.
inline bool framed_beats_byte(const u8* byte_buf, usize n, Filter byte_filter,
                              const std::vector<u8>& framed) noexcept {
  if (byte_filter == Filter::None || byte_buf == nullptr || n < 2) return true;
  if (framed.size() < 2) return false;
  const double pack_bits = adjacency_cost_bits(framed.data(), framed.size());
  const double byte_bits = adjacency_cost_bits(byte_buf, n);
  return pack_bits < byte_bits * 0.97;
}

}  // namespace spectra8::detail

#endif  // SPECTRA8_SRC_PACK_FILTER_HPP
