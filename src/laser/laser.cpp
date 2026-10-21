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

#include "laser/laser.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <memory>
#include <vector>

#include "util/bits.hpp"
#include "util/simd.hpp"
#include "util/transform.hpp"

namespace spectra8::detail {
namespace {

#ifndef SPECTRA8_ALWAYS_INLINE
#if defined(__GNUC__) || defined(__clang__)
#define SPECTRA8_ALWAYS_INLINE inline __attribute__((always_inline))
#elif defined(_MSC_VER)
#define SPECTRA8_ALWAYS_INLINE __forceinline
#else
#define SPECTRA8_ALWAYS_INLINE inline
#endif
#endif

#ifndef SPECTRA8_NOINLINE
#if defined(__GNUC__) || defined(__clang__)
#define SPECTRA8_NOINLINE __attribute__((noinline))
#elif defined(_MSC_VER)
#define SPECTRA8_NOINLINE __declspec(noinline)
#else
#define SPECTRA8_NOINLINE
#endif
#endif

#ifndef SPECTRA8_LIKELY
#if defined(__GNUC__) || defined(__clang__)
#define SPECTRA8_LIKELY(x) (__builtin_expect(!!(x), 1))
#define SPECTRA8_UNLIKELY(x) (__builtin_expect(!!(x), 0))
#else
#define SPECTRA8_LIKELY(x) (x)
#define SPECTRA8_UNLIKELY(x) (x)
#endif
#endif

constexpr u8 kLaserMagic = 0xA5u;
constexpr usize kMaxMatchLen = 273;
[[maybe_unused]] constexpr usize kMinMatchLen = 2;
constexpr u32 kNumStates = 12;
constexpr u32 kNumPosBitsMax = 2;
constexpr u32 kNumPosStatesMax = 1u << kNumPosBitsMax;  // 4
constexpr u32 kNumLenToPosStates = 4;
constexpr u32 kNumPosSlotBits = 6;
constexpr u32 kNumAlignBits = 4;
constexpr u32 kAlignTableSize = 1u << kNumAlignBits;  // 16
constexpr u32 kStartPosModelIndex = 4;
constexpr u32 kEndPosModelIndex = 14;
constexpr u32 kNumFullDistancesSlots = kEndPosModelIndex - kStartPosModelIndex;  // 10
constexpr u16 kProbInit = 1024;
constexpr u32 kInfinityPrice = 0x0FFFFFFFu;

template <bool AlignedLp>
SPECTRA8_ALWAYS_INLINE u32 laser_lit_ctx(usize pos, u32 prev_hi) noexcept {
  if constexpr (AlignedLp) {
    return ((static_cast<u32>(pos) & 3u) << 2) | (prev_hi >> 2);
  } else {
    (void)pos;
    return prev_hi >> 1;
  }
}

// ─── State transition tables (12-state LZMA automaton) ──────────────────────

constexpr u8 kLiteralNextState[kNumStates] = {0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 4, 5};
constexpr u8 kMatchNextState[kNumStates] = {7, 7, 7, 7, 7, 7, 7, 10, 10, 10, 10, 10};
constexpr u8 kRepNextState[kNumStates] = {8, 8, 8, 8, 8, 8, 8, 11, 11, 11, 11, 11};
constexpr u8 kShortRepNextState[kNumStates] = {9, 9, 9, 9, 9, 9, 9, 11, 11, 11, 11, 11};

// ─── Precomputed bit-price table (1/64th bit resolution) ────────────────────
// kProbPrices[p >> 2] = round(-log2((p) / 2048.0) * 64)

constexpr std::array<u16, 512> make_prob_prices() {
  std::array<u16, 512> t{};
  constexpr int kBits = 11;
  for (u32 i = 0; i < 512; ++i) {
    u32 w = (i << 2) + 2;
    u32 bit_count = 0;
    for (int j = 0; j < 6; ++j) {
      w = w * w;
      bit_count <<= 1;
      while (w >= (1u << 16)) {
        w >>= 1;
        ++bit_count;
      }
    }
    t[i] = static_cast<u16>((kBits << 6) - 15 - bit_count);
  }
  return t;
}

constexpr auto kProbPrices = make_prob_prices();

SPECTRA8_ALWAYS_INLINE u32 price_bit(u16 prob, u32 bit) noexcept {
  const u32 p = prob ^ ((0u - bit) & 2047u);
  return kProbPrices[p >> 2];
}

SPECTRA8_ALWAYS_INLINE u32 price_bit0(u16 prob) noexcept {
  return kProbPrices[prob >> 2];
}

SPECTRA8_ALWAYS_INLINE u32 price_bit1(u16 prob) noexcept {
  return kProbPrices[(prob ^ 2047u) >> 2];
}

// ─── Distance slot table ────────────────────────────────────────────────────

struct DistSlotTable {
  std::array<u8, 2048> fast{};
  constexpr DistSlotTable() {
    fast[0] = 0;
    fast[1] = 1;
    fast[2] = 2;
    fast[3] = 3;
    u32 c = 4;
    for (u32 slot = 4; slot < 22; ++slot) {
      const u32 k = 1u << ((slot >> 1) - 1);
      for (u32 j = 0; j < k; ++j, ++c) {
        if (c < fast.size()) fast[c] = static_cast<u8>(slot);
      }
    }
  }
};

constexpr DistSlotTable kDistSlotTable{};

SPECTRA8_ALWAYS_INLINE u32 get_dist_slot(u32 dist_val) noexcept {
  if (dist_val < 2048u) return kDistSlotTable.fast[dist_val];
  const u32 b = 31u - static_cast<u32>(__builtin_clz(dist_val));
  return (b << 1) + ((dist_val >> (b - 1)) & 1u);
}

// ─── Binary Range Encoder & Branchless Range Decoder ────────────────────────

struct RangeEnc {
  u64 low = 0;
  u32 range = 0xFFFFFFFFu;
  u32 cache_size = 1;
  u8* ptr = nullptr;
  u8* begin = nullptr;
  u8* end = nullptr;
  u8 cache = 0;

  void init(u8* buf, usize cap) noexcept {
    low = 0;
    range = 0xFFFFFFFFu;
    cache_size = 1;
    cache = 0;
    ptr = buf;
    begin = buf;
    end = buf + cap;
  }

  usize size() const noexcept { return static_cast<usize>(ptr - begin); }

  SPECTRA8_NOINLINE void shift_low() noexcept {
    const u32 low32 = static_cast<u32>(low);
    const u32 hi = static_cast<u32>(low >> 32);
    if (SPECTRA8_LIKELY(low32 < 0xFF000000u || hi != 0)) {
      u8 temp = cache;
      do {
        if (SPECTRA8_LIKELY(ptr < end)) {
          *ptr++ = static_cast<u8>(temp + hi);
        }
        temp = 0xFFu;
      } while (--cache_size != 0);
      cache = static_cast<u8>(low32 >> 24);
    }
    ++cache_size;
    low = low32 << 8;
  }

  SPECTRA8_ALWAYS_INLINE void encode_bit(u16& prob, u32 bit) noexcept {
    const u32 p = prob;
    const u32 bound = (range >> 11) * p;
    const u32 mask = 0u - bit;
    low += bound & mask;
    range = (bit != 0) ? (range - bound) : bound;
    const i32 target = (bit != 0) ? 31 : 2048;
    prob = static_cast<u16>(
        static_cast<i32>(p) + ((target - static_cast<i32>(p)) >> 5));
    if (SPECTRA8_UNLIKELY(range < 0x01000000u)) {
      range <<= 8;
      shift_low();
    }
  }

  SPECTRA8_ALWAYS_INLINE void encode_bit0(u16& prob) noexcept {
    const u32 p = prob;
    range = (range >> 11) * p;
    prob = static_cast<u16>(p + ((2048u - p) >> 5));
    if (SPECTRA8_UNLIKELY(range < 0x01000000u)) {
      range <<= 8;
      shift_low();
    }
  }

  SPECTRA8_ALWAYS_INLINE void encode_bit1(u16& prob) noexcept {
    const u32 p = prob;
    const u32 bound = (range >> 11) * p;
    low += bound;
    range -= bound;
    prob = static_cast<u16>(p - (p >> 5));
    if (SPECTRA8_UNLIKELY(range < 0x01000000u)) {
      range <<= 8;
      shift_low();
    }
  }

  void finish() noexcept {
    for (int i = 0; i < 5; ++i) shift_low();
  }
};

struct RangeDec {
  u32 range = 0xFFFFFFFFu;
  u32 code = 0;
  const u8* ptr = nullptr;
  const u8* end = nullptr;

  bool init(const u8* data, usize len) noexcept {
    if (len < 5) return false;
    range = 0xFFFFFFFFu;
    code = (static_cast<u32>(data[1]) << 24) |
           (static_cast<u32>(data[2]) << 16) |
           (static_cast<u32>(data[3]) << 8) |
           static_cast<u32>(data[4]);
    ptr = data + 5;
    end = data + len;
    return true;
  }

  SPECTRA8_ALWAYS_INLINE void renorm() noexcept {
    if (SPECTRA8_UNLIKELY(range < 0x01000000u)) {
      range <<= 8;
      const u8 byte = SPECTRA8_LIKELY(ptr < end) ? *ptr++ : 0;
      code = (code << 8) | byte;
    }
  }

  SPECTRA8_ALWAYS_INLINE u32 decode_bit(u16& prob) noexcept {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    u32 p;
    u32 bound;
    u32 bit;
    __asm__(
        "movzwl %[prob], %k[p]\n\t"
        "movl %[range], %k[bound]\n\t"
        "shrl $11, %k[bound]\n\t"
        "imull %k[p], %k[bound]\n\t"
        "subl %k[bound], %[range]\n\t"
        "movl %[code], %k[bit]\n\t"
        "subl %k[bound], %[code]\n\t"
        "cmovbl %k[bit], %[code]\n\t"
        "cmovbl %k[bound], %[range]\n\t"
        "sbbl %k[bit], %k[bit]\n\t"
        "movl %k[bit], %k[bound]\n\t"
        "andl $2017, %k[bound]\n\t"
        "subl %k[p], %k[bound]\n\t"
        "addl $31, %k[bound]\n\t"
        "sarl $5, %k[bound]\n\t"
        "addl %k[bound], %k[p]\n\t"
        "movw %w[p], %[prob]\n\t"
        "incl %k[bit]\n\t"
        : [prob] "+m"(prob), [range] "+r"(range), [code] "+r"(code),
          [p] "=&r"(p), [bound] "=&r"(bound), [bit] "=&r"(bit)
        :
        : "cc");
    renorm();
    return bit;
#else
    const u32 p = prob;
    const u32 bound = (range >> 11) * p;
    const u32 bit = static_cast<u32>(code >= bound);
    range = bit ? (range - bound) : bound;
    code -= bit ? bound : 0;
    const i32 target = bit ? 31 : 2048;
    prob = static_cast<u16>(
        static_cast<i32>(p) + ((target - static_cast<i32>(p)) >> 5));
    renorm();
    return bit;
#endif
  }

  template <typename T>
  SPECTRA8_ALWAYS_INLINE void decode_tree_bit(u16* __restrict probs,
                                          T& sym) noexcept {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    u16& prob = probs[sym];
    u32 p;
    u32 bound;
    usize mask;
    __asm__(
        "movzwl %[prob], %k[p]\n\t"
        "movl %[range], %k[bound]\n\t"
        "shrl $11, %k[bound]\n\t"
        "imull %k[p], %k[bound]\n\t"
        "subl %k[bound], %[range]\n\t"
        "movl %[code], %k[mask]\n\t"
        "subl %k[bound], %[code]\n\t"
        "cmovbl %k[mask], %[code]\n\t"
        "cmovbl %k[bound], %[range]\n\t"
        "sbbq %q[mask], %q[mask]\n\t"
        "movl %k[mask], %k[bound]\n\t"
        "andl $2017, %k[bound]\n\t"
        "subl %k[p], %k[bound]\n\t"
        "addl $31, %k[bound]\n\t"
        "sarl $5, %k[bound]\n\t"
        "addl %k[bound], %k[p]\n\t"
        "movw %w[p], %[prob]\n\t"
        "leaq 1(%q[mask],%[sym],2), %[sym]\n\t"
        : [prob] "+m"(prob), [range] "+r"(range), [code] "+r"(code),
          [sym] "+r"(sym), [p] "=&r"(p), [bound] "=&r"(bound),
          [mask] "=&r"(mask)
        :
        : "cc");
    renorm();
#else
    u16& prob = probs[sym];
    const u32 p = prob;
    const u32 bound = (range >> 11) * p;
    const u32 bit = static_cast<u32>(code >= bound);
    range = bit ? (range - bound) : bound;
    code -= bit ? bound : 0;
    const i32 target = bit ? 31 : 2048;
    prob = static_cast<u16>(
        static_cast<i32>(p) + ((target - static_cast<i32>(p)) >> 5));
    sym = (sym << 1) | bit;
    renorm();
#endif
  }

  template <typename T>
  SPECTRA8_ALWAYS_INLINE void decode_matched_bit(u16* __restrict probs, T& sym,
                                             u32& m, u32& offset) noexcept {
    m <<= 1;
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    u16& prob = probs[offset + (m & offset) + sym];
    u32 p;
    u32 bound;
    usize mask;
    __asm__(
        "movzwl %[prob], %k[p]\n\t"
        "movl %[range], %k[bound]\n\t"
        "shrl $11, %k[bound]\n\t"
        "imull %k[p], %k[bound]\n\t"
        "subl %k[bound], %[range]\n\t"
        "movl %[code], %k[mask]\n\t"
        "subl %k[bound], %[code]\n\t"
        "cmovbl %k[mask], %[code]\n\t"
        "cmovbl %k[bound], %[range]\n\t"
        "sbbq %q[mask], %q[mask]\n\t"
        "movl %k[mask], %k[bound]\n\t"
        "andl $2017, %k[bound]\n\t"
        "subl %k[p], %k[bound]\n\t"
        "addl $31, %k[bound]\n\t"
        "sarl $5, %k[bound]\n\t"
        "addl %k[bound], %k[p]\n\t"
        "movw %w[p], %[prob]\n\t"
        "leaq 1(%q[mask],%[sym],2), %[sym]\n\t"
        : [prob] "+m"(prob), [range] "+r"(range), [code] "+r"(code),
          [sym] "+r"(sym), [p] "=&r"(p), [bound] "=&r"(bound),
          [mask] "=&r"(mask)
        :
        : "cc");
    offset &= m ^ static_cast<u32>(mask);
    renorm();
#else
    u16& prob = probs[offset + (m & offset) + sym];
    const u32 p = prob;
    const u32 bound = (range >> 11) * p;
    const u32 bit = static_cast<u32>(code >= bound);
    range = bit ? (range - bound) : bound;
    code -= bit ? bound : 0;
    const i32 target = bit ? 31 : 2048;
    prob = static_cast<u16>(
        static_cast<i32>(p) + ((target - static_cast<i32>(p)) >> 5));
    sym = (sym << 1) | bit;
    offset &= m ^ (bit - 1u);
    renorm();
#endif
  }
};

// ─── Fast Raw Bits Stream (Writer & Reader) ─────────────────────────────────

struct LaserRawWriter {
  u8* ptr = nullptr;
  u8* begin = nullptr;
  u8* end = nullptr;
  u64 accum = 0;
  int bit_count = 0;

  void init(u8* buf, usize cap) noexcept {
    ptr = buf;
    begin = buf;
    end = buf + cap;
    accum = 0;
    bit_count = 0;
  }

  usize size() const noexcept { return static_cast<usize>(ptr - begin); }

  SPECTRA8_ALWAYS_INLINE void put(u32 val, int bits) noexcept {
    if (bits <= 0) return;
    accum = (accum << bits) | (val & ((1u << bits) - 1u));
    bit_count += bits;
    while (bit_count >= 8) {
      bit_count -= 8;
      if (SPECTRA8_LIKELY(ptr < end)) {
        *ptr++ = static_cast<u8>(accum >> bit_count);
      }
    }
  }

  void finish() noexcept {
    if (bit_count > 0) {
      if (SPECTRA8_LIKELY(ptr < end)) {
        *ptr++ = static_cast<u8>(accum << (8 - bit_count));
      }
      accum = 0;
      bit_count = 0;
    }
  }
};

struct LaserRawReader {
  const u8* ptr = nullptr;
  const u8* end = nullptr;
  u64 accum = 0;
  int bit_count = 0;

  LaserRawReader(const u8* p, usize len) noexcept : ptr(p), end(p + len) {}

  SPECTRA8_ALWAYS_INLINE bool can_read(int bits) const noexcept {
    return static_cast<usize>(bits) <=
           static_cast<usize>(end - ptr) * 8u + static_cast<usize>(bit_count);
  }

  SPECTRA8_ALWAYS_INLINE u32 get(int bits) noexcept {
    if (bits <= 0) return 0;
    while (bit_count < bits && ptr < end) {
      accum = (accum << 8) | *ptr++;
      bit_count += 8;
    }
    if (bit_count < bits) {
      accum <<= (bits - bit_count);
      bit_count = bits;
    }
    bit_count -= bits;
    return static_cast<u32>(accum >> bit_count) & ((1u << bits) - 1u);
  }
};

// ─── Bit-tree helpers ───────────────────────────────────────────────────────

template <int Bits>
SPECTRA8_ALWAYS_INLINE void encode_bittree(RangeEnc& rc, u16* probs,
                                       u32 sym) noexcept {
  u32 m = 1;
#if defined(__clang__)
#pragma clang loop unroll(full)
#elif defined(__GNUC__)
#pragma GCC unroll 8
#endif
  for (int i = Bits - 1; i >= 0; --i) {
    const u32 bit = (sym >> i) & 1u;
    rc.encode_bit(probs[m], bit);
    m = (m << 1) | bit;
  }
}

template <int Bits>
SPECTRA8_ALWAYS_INLINE u32 decode_bittree(RangeDec& rc,
                                      u16* __restrict probs) noexcept {
  usize m = 1;
#if defined(__clang__)
#pragma clang loop unroll(full)
#elif defined(__GNUC__)
#pragma GCC unroll 8
#endif
  for (int i = 0; i < Bits; ++i) {
    rc.decode_tree_bit(probs, m);
  }
  return static_cast<u32>(m) - (1u << Bits);
}

SPECTRA8_ALWAYS_INLINE u32 decode_bittree_loop8(RangeDec& rc,
                                            u16* __restrict probs) noexcept {
  usize m = 1;
#if defined(__clang__)
#pragma clang loop unroll(disable)
#elif defined(__GNUC__)
#pragma GCC unroll 1
#endif
  for (int i = 0; i < 8; ++i) {
    rc.decode_tree_bit(probs, m);
  }
  return static_cast<u32>(m) - 256u;
}

template <int Bits>
SPECTRA8_ALWAYS_INLINE u32 price_bittree(const u16* probs, u32 sym) noexcept {
  u32 price = 0;
  u32 m = 1;
#if defined(__clang__)
#pragma clang loop unroll(full)
#elif defined(__GNUC__)
#pragma GCC unroll 8
#endif
  for (int i = Bits - 1; i >= 0; --i) {
    const u32 bit = (sym >> i) & 1u;
    price += price_bit(probs[m], bit);
    m = (m << 1) | bit;
  }
  return price;
}

SPECTRA8_ALWAYS_INLINE void encode_reverse_bittree(RangeEnc& rc, u16* probs,
                                               int num_bits,
                                               u32 sym) noexcept {
  u32 m = 1;
  for (int i = 0; i < num_bits; ++i) {
    const u32 bit = sym & 1u;
    sym >>= 1;
    rc.encode_bit(probs[m], bit);
    m = (m << 1) | bit;
  }
}

template <int NumBits>
SPECTRA8_ALWAYS_INLINE void encode_reverse_bittree_fixed(RangeEnc& rc, u16* probs,
                                                     u32 sym) noexcept {
  u32 m = 1;
#if defined(__clang__)
#pragma clang loop unroll(full)
#elif defined(__GNUC__)
#pragma GCC unroll 8
#endif
  for (int i = 0; i < NumBits; ++i) {
    const u32 bit = sym & 1u;
    sym >>= 1;
    rc.encode_bit(probs[m], bit);
    m = (m << 1) | bit;
  }
}

template <int NumBits>
SPECTRA8_ALWAYS_INLINE u32 decode_reverse_bittree(RangeDec& rc,
                                              u16* __restrict probs) noexcept {
  if constexpr (NumBits == 4) {
    usize m = 1;
    rc.decode_tree_bit(probs, m);
    rc.decode_tree_bit(probs, m);
    rc.decode_tree_bit(probs, m);
    rc.decode_tree_bit(probs, m);
    return static_cast<u32>(
        (0xF7B3D591E6A2C480ull >> ((m & 15u) << 2)) & 15u);
  } else {
    usize m = 1;
    u32 sym = 0;
#if defined(__clang__)
#pragma clang loop unroll(full)
#elif defined(__GNUC__)
#pragma GCC unroll 8
#endif
    for (int i = 0; i < NumBits; ++i) {
      rc.decode_tree_bit(probs, m);
      sym |= static_cast<u32>(m & 1u) << i;
    }
    return sym;
  }
}

SPECTRA8_ALWAYS_INLINE u32 decode_reverse_bittree(RangeDec& rc,
                                              u16* __restrict probs,
                                              int num_bits) noexcept {
  usize m = 1;
  u32 sym = 0;
#if defined(__clang__)
#pragma clang loop unroll(disable)
#elif defined(__GNUC__)
#pragma GCC unroll 1
#endif
  for (int i = 0; i < num_bits; ++i) {
    rc.decode_tree_bit(probs, m);
    sym |= static_cast<u32>(m & 1u) << i;
  }
  return sym;
}

SPECTRA8_ALWAYS_INLINE u32 price_reverse_bittree(const u16* probs, int num_bits,
                                             u32 sym) noexcept {
  u32 price = 0;
  u32 m = 1;
  for (int i = 0; i < num_bits; ++i) {
    const u32 bit = sym & 1u;
    sym >>= 1;
    price += price_bit(probs[m], bit);
    m = (m << 1) | bit;
  }
  return price;
}

template <int NumBits>
SPECTRA8_ALWAYS_INLINE u32 price_reverse_bittree_fixed(const u16* probs,
                                                   u32 sym) noexcept {
  u32 price = 0;
  u32 m = 1;
#if defined(__clang__)
#pragma clang loop unroll(full)
#elif defined(__GNUC__)
#pragma GCC unroll 8
#endif
  for (int i = 0; i < NumBits; ++i) {
    const u32 bit = sym & 1u;
    sym >>= 1;
    price += price_bit(probs[m], bit);
    m = (m << 1) | bit;
  }
  return price;
}

// ─── Nibble literal coders (hi_probs / lo_probs) ────────────────────────────

SPECTRA8_ALWAYS_INLINE void encode_nibble(RangeEnc& rc, u16* probs,
                                      u32 nib) noexcept {
  u32 sym = 1;
  u32 bit = (nib >> 3) & 1u;
  rc.encode_bit(probs[sym], bit);
  sym = (sym << 1) | bit;

  bit = (nib >> 2) & 1u;
  rc.encode_bit(probs[sym], bit);
  sym = (sym << 1) | bit;

  bit = (nib >> 1) & 1u;
  rc.encode_bit(probs[sym], bit);
  sym = (sym << 1) | bit;

  bit = nib & 1u;
  rc.encode_bit(probs[sym], bit);
}

SPECTRA8_ALWAYS_INLINE u32 decode_nibble(RangeDec& rc,
                                     u16* __restrict probs) noexcept {
  usize sym = 1;
  rc.decode_tree_bit(probs, sym);
  rc.decode_tree_bit(probs, sym);
  rc.decode_tree_bit(probs, sym);
  rc.decode_tree_bit(probs, sym);
  return static_cast<u32>(sym) & 15u;
}

[[maybe_unused]] SPECTRA8_ALWAYS_INLINE void decode_nibble_pair(
    RangeDec& __restrict rc_a, u16* __restrict probs_a, u32& out_a,
    RangeDec& __restrict rc_b, u16* __restrict probs_b, u32& out_b) noexcept {
  usize sym_a = 1;
  usize sym_b = 1;
  rc_a.decode_tree_bit(probs_a, sym_a);
  rc_b.decode_tree_bit(probs_b, sym_b);
  rc_a.decode_tree_bit(probs_a, sym_a);
  rc_b.decode_tree_bit(probs_b, sym_b);
  rc_a.decode_tree_bit(probs_a, sym_a);
  rc_b.decode_tree_bit(probs_b, sym_b);
  rc_a.decode_tree_bit(probs_a, sym_a);
  rc_b.decode_tree_bit(probs_b, sym_b);
  out_a = static_cast<u32>(sym_a) & 15u;
  out_b = static_cast<u32>(sym_b) & 15u;
}

SPECTRA8_ALWAYS_INLINE u32 price_nibble(const u16* probs, u32 nib) noexcept {
  u32 sym = 1;
  u32 bit = (nib >> 3) & 1u;
  u32 price = price_bit(probs[sym], bit);
  sym = (sym << 1) | bit;

  bit = (nib >> 2) & 1u;
  price += price_bit(probs[sym], bit);
  sym = (sym << 1) | bit;

  bit = (nib >> 1) & 1u;
  price += price_bit(probs[sym], bit);
  sym = (sym << 1) | bit;

  bit = nib & 1u;
  price += price_bit(probs[sym], bit);
  return price;
}

SPECTRA8_ALWAYS_INLINE void encode_matched_nibble(RangeEnc& rc, u16* probs,
                                              u32 nib, u32 match_nib) noexcept {
  u32 sym = 1;
  u32 m = match_nib;
  u32 offset = 0x10u;

  m <<= 1;
  u32 match_bit = m & offset;
  u32 bit = (nib >> 3) & 1u;
  rc.encode_bit(probs[offset + match_bit + sym], bit);
  sym = (sym << 1) | bit;
  offset &= m ^ (bit - 1u);

  m <<= 1;
  match_bit = m & offset;
  bit = (nib >> 2) & 1u;
  rc.encode_bit(probs[offset + match_bit + sym], bit);
  sym = (sym << 1) | bit;
  offset &= m ^ (bit - 1u);

  m <<= 1;
  match_bit = m & offset;
  bit = (nib >> 1) & 1u;
  rc.encode_bit(probs[offset + match_bit + sym], bit);
  sym = (sym << 1) | bit;
  offset &= m ^ (bit - 1u);

  m <<= 1;
  match_bit = m & offset;
  bit = nib & 1u;
  rc.encode_bit(probs[offset + match_bit + sym], bit);
}

SPECTRA8_ALWAYS_INLINE u32 decode_matched_nibble(RangeDec& rc,
                                             u16* __restrict probs,
                                             u32 match_nib) noexcept {
  usize sym = 1;
  u32 m = match_nib;
  u32 offset = 0x10u;

  rc.decode_matched_bit(probs, sym, m, offset);
  rc.decode_matched_bit(probs, sym, m, offset);
  rc.decode_matched_bit(probs, sym, m, offset);

  m <<= 1;
  rc.decode_tree_bit(probs + offset + (m & offset), sym);

  return static_cast<u32>(sym) & 15u;
}

SPECTRA8_ALWAYS_INLINE u32 price_matched_nibble(const u16* probs, u32 nib,
                                            u32 match_nib) noexcept {
  u32 price = 0;
  u32 sym = 1;
  u32 m = match_nib;
  u32 offset = 0x10u;

  m <<= 1;
  u32 match_bit = m & offset;
  u32 bit = (nib >> 3) & 1u;
  price += price_bit(probs[offset + match_bit + sym], bit);
  sym = (sym << 1) | bit;
  offset &= m ^ (bit - 1u);

  m <<= 1;
  match_bit = m & offset;
  bit = (nib >> 2) & 1u;
  price += price_bit(probs[offset + match_bit + sym], bit);
  sym = (sym << 1) | bit;
  offset &= m ^ (bit - 1u);

  m <<= 1;
  match_bit = m & offset;
  bit = (nib >> 1) & 1u;
  price += price_bit(probs[offset + match_bit + sym], bit);
  sym = (sym << 1) | bit;
  offset &= m ^ (bit - 1u);

  m <<= 1;
  match_bit = m & offset;
  bit = nib & 1u;
  price += price_bit(probs[offset + match_bit + sym], bit);

  return price;
}

// ─── Length model (2..9, 10..17, 18..273) ───────────────────────────────────

struct LenModel {
  u16 choice;
  u16 choice2;
  u16 low[kNumPosStatesMax][8];
  u16 mid[kNumPosStatesMax][8];
  u16 high[256];

  void init() noexcept {
    choice = kProbInit;
    choice2 = kProbInit;
    for (u32 ps = 0; ps < kNumPosStatesMax; ++ps) {
      std::fill(std::begin(low[ps]), std::end(low[ps]), kProbInit);
      std::fill(std::begin(mid[ps]), std::end(mid[ps]), kProbInit);
    }
    std::fill(std::begin(high), std::end(high), kProbInit);
  }

  SPECTRA8_ALWAYS_INLINE void encode(RangeEnc& rc, u32 ps, u32 len) noexcept {
    const u32 sym = len - 2;
    if (SPECTRA8_LIKELY(sym < 8)) {
      rc.encode_bit0(choice);
      encode_bittree<3>(rc, low[ps], sym);
    } else if (SPECTRA8_LIKELY(sym < 16)) {
      rc.encode_bit1(choice);
      rc.encode_bit0(choice2);
      encode_bittree<3>(rc, mid[ps], sym - 8);
    } else {
      rc.encode_bit1(choice);
      rc.encode_bit1(choice2);
      encode_bittree<8>(rc, high, sym - 16);
    }
  }

  SPECTRA8_ALWAYS_INLINE u32 decode(RangeDec& rc, u32 ps) noexcept {
    if (SPECTRA8_LIKELY(rc.decode_bit(choice) == 0)) {
      return 2u + decode_bittree<3>(rc, low[ps]);
    }
    if (SPECTRA8_LIKELY(rc.decode_bit(choice2) == 0)) {
      return 10u + decode_bittree<3>(rc, mid[ps]);
    }
    return 18u + decode_bittree_loop8(rc, high);
  }

  SPECTRA8_ALWAYS_INLINE u32 price(u32 ps, u32 len) const noexcept {
    const u32 sym = len - 2;
    const u32 c0 = price_bit0(choice);
    if (SPECTRA8_LIKELY(sym < 8)) {
      return c0 + price_bittree<3>(low[ps], sym);
    }
    const u32 c1 = price_bit1(choice);
    if (SPECTRA8_LIKELY(sym < 16)) {
      return c1 + price_bit0(choice2) + price_bittree<3>(mid[ps], sym - 8);
    }
    return c1 + price_bit1(choice2) + price_bittree<8>(high, sym - 16);
  }
};

// ─── Full Laser probability state ───────────────────────────────────────────

struct LaserModel {
  u16 is_match[kNumStates][kNumPosStatesMax];
  u16 is_rep[kNumStates];
  u16 is_rep0[kNumStates];
  u16 is_rep0_long[kNumStates][kNumPosStatesMax];
  u16 is_rep1[kNumStates];
  u16 is_rep2[kNumStates];

  LenModel len_coder;
  LenModel rep_len_coder;

  u16 dist_slot[kNumLenToPosStates][1u << kNumPosSlotBits];
  u16 spec_dist[kNumFullDistancesSlots][32];
  u16 align_probs[kAlignTableSize];

  // Up to 16 literal contexts ((lc=3, lp=0) -> 8, (lc=2, lp=2) -> 16)
  u16 hi_probs[16][48];
  u16 lo_probs[16][16][48];

  void init(u32 num_lit_ctx) noexcept {
    for (u32 s = 0; s < kNumStates; ++s) {
      std::fill(std::begin(is_match[s]), std::end(is_match[s]), kProbInit);
      is_rep[s] = kProbInit;
      is_rep0[s] = kProbInit;
      std::fill(std::begin(is_rep0_long[s]), std::end(is_rep0_long[s]), kProbInit);
      is_rep1[s] = kProbInit;
      is_rep2[s] = kProbInit;
    }
    len_coder.init();
    rep_len_coder.init();
    for (u32 ls = 0; ls < kNumLenToPosStates; ++ls) {
      std::fill(std::begin(dist_slot[ls]), std::end(dist_slot[ls]), kProbInit);
    }
    for (u32 sd = 0; sd < kNumFullDistancesSlots; ++sd) {
      std::fill(std::begin(spec_dist[sd]), std::end(spec_dist[sd]), kProbInit);
    }
    std::fill(std::begin(align_probs), std::end(align_probs), kProbInit);
    for (u32 c = 0; c < num_lit_ctx; ++c) {
      std::fill(std::begin(hi_probs[c]), std::end(hi_probs[c]), kProbInit);
      for (u32 h = 0; h < 16; ++h) {
        std::fill(std::begin(lo_probs[c][h]), std::end(lo_probs[c][h]), kProbInit);
      }
    }
  }

  SPECTRA8_ALWAYS_INLINE u32 price_literal(u32 lit_ctx, u32 state, u8 byte,
                                       u8 match_byte) const noexcept {
    const u32 hi = byte >> 4;
    const u32 lo = byte & 15u;
    const u16* hp = hi_probs[lit_ctx];
    const u16* lp = lo_probs[lit_ctx][hi];
    if (state < 7) {
      return price_nibble(hp, hi) + price_nibble(lp, lo);
    }
    const u32 m_hi = match_byte >> 4;
    const u32 m_lo = match_byte & 15u;
    const u32 p_hi = price_matched_nibble(hp, hi, m_hi);
    const u32 p_lo = (hi == m_hi) ? price_matched_nibble(lp, lo, m_lo)
                                  : price_nibble(lp, lo);
    return p_hi + p_lo;
  }

  SPECTRA8_ALWAYS_INLINE void price_dist_split(u32 dist, u32& out_slot,
                                           u32& out_extra) const noexcept {
    const u32 dist_val = dist - 1;
    const u32 slot = get_dist_slot(dist_val);
    out_slot = slot;
    u32 extra = 0;
    if (slot >= 4) {
      const int footer_bits = static_cast<int>((slot >> 1) - 1);
      const u32 base = (2u | (slot & 1u)) << footer_bits;
      const u32 rem = dist_val - base;
      if (slot < 14) {
        extra = price_reverse_bittree(spec_dist[slot - 4], footer_bits, rem);
      } else {
        const u32 direct_bits = static_cast<u32>(footer_bits - 4);
        extra = (direct_bits << 6) +
                price_reverse_bittree_fixed<4>(align_probs, rem & 15u);
      }
    }
    out_extra = extra;
  }

  SPECTRA8_ALWAYS_INLINE u32 price_dist_slot(u32 len, u32 slot) const noexcept {
    const u32 len_state = (len - 2 < 3) ? (len - 2) : 3u;
    return price_bittree<6>(dist_slot[len_state], slot);
  }

  SPECTRA8_ALWAYS_INLINE u32 price_dist(u32 len, u32 dist) const noexcept {
    u32 slot, extra;
    price_dist_split(dist, slot, extra);
    return price_dist_slot(len, slot) + extra;
  }
};

// ─── Binary Tree Match Finder + h2/h3 Close Hash Tables ─────────────────────

struct MatchPair {
  u16 len;
  u32 dist;
};

struct LaserMatchFinder {
  const u8* src = nullptr;
  usize n = 0;
  int max_depth = 32;
  int skip_depth = 12;
  u32 epoch_base = 1;

  std::vector<u32> h2;
  std::vector<u32> h3;
  std::vector<u32> head4;
  std::vector<u32> son;

  static constexpr u32 kEmpty = 0xFFFFFFFFu;
  static constexpr u32 kHash4Bits = 19;
  static constexpr u32 kHash4Size = 1u << kHash4Bits;

  void init(const u8* s, usize len, int level) {
    src = s;
    n = len;
    if (level <= 2) {
      max_depth = 12;
      skip_depth = 4;
    } else if (level <= 4) {
      max_depth = 20;
      skip_depth = 8;
    } else if (level <= 6) {
      max_depth = 36;
      skip_depth = 12;
    } else if (level <= 8) {
      max_depth = 64;
      skip_depth = 20;
    } else {
      max_depth = 112;
      skip_depth = 32;
    }

    const u32 span = static_cast<u32>(len) + 1u;
    if (h2.empty() || 0xFFFFFFFEu - epoch_base < span) {
      h2.assign(1u << 16, 0u);
      h3.assign(1u << 16, 0u);
      head4.assign(kHash4Size, 0u);
      epoch_base = 1u;
    }
    if (son.size() < 2 * len) {
      son.resize(2 * len);
    }
  }

  void advance_epoch() noexcept {
    epoch_base += static_cast<u32>(n) + 1u;
  }

  SPECTRA8_ALWAYS_INLINE static u32 hash3_fn(const u8* p) noexcept {
    const u32 v = static_cast<u32>(p[0]) | (static_cast<u32>(p[1]) << 8) |
                  (static_cast<u32>(p[2]) << 16);
    return (v * 0x9E3779B1u) >> 16;
  }

  SPECTRA8_ALWAYS_INLINE static u32 hash4_fn(const u8* p) noexcept {
    return (load_u32(p) * 0x9E3779B1u) >> (32 - kHash4Bits);
  }

  // Returns number of strictly increasing-length matches at `pos`.
  u32 get_matches(usize pos, MatchPair* pairs) noexcept {
    if (pos >= n) return 0;
    u32 count = 0;
    const usize limit = std::min<usize>(kMaxMatchLen, n - pos);
    if (limit < 2) return 0;

    const u8* const cur = src + pos;
    const u32 eb = epoch_base;
    const u32 cur_tag = eb + static_cast<u32>(pos);
    usize max_len = 1;

    // 2-byte close hash
    const u32 hv2 = load_u16(cur);
    const u32 raw_m2 = h2[hv2];
    h2[hv2] = cur_tag;
    const u32 m2 = (raw_m2 >= eb) ? (raw_m2 - eb) : kEmpty;
    if (m2 < pos && pos - m2 < 256 && cur[0] == src[m2] &&
        cur[1] == src[m2 + 1]) {
      usize l = 2;
      while (l < limit && cur[l] == src[m2 + l]) ++l;
      pairs[count++] = {static_cast<u16>(l), static_cast<u32>(pos - m2)};
      max_len = l;
    }

    if (limit >= 3) {
      const u32 hv3 = hash3_fn(cur);
      const u32 raw_m3 = h3[hv3];
      h3[hv3] = cur_tag;
      const u32 m3 = (raw_m3 >= eb) ? (raw_m3 - eb) : kEmpty;
      if (m3 < pos && m3 != m2 && pos - m3 < 32768 && cur[0] == src[m3] &&
          cur[1] == src[m3 + 1] && cur[2] == src[m3 + 2]) {
        usize l = 3;
        while (l < limit && cur[l] == src[m3 + l]) ++l;
        if (l > max_len) {
          if (count > 0 && pairs[count - 1].dist > pos - m3) --count;
          pairs[count++] = {static_cast<u16>(l), static_cast<u32>(pos - m3)};
          max_len = l;
        }
      }
    }

    if (limit < 4) return count;

    const u32 hv4 = hash4_fn(cur);
    const u32 raw_m4 = head4[hv4];
    head4[hv4] = cur_tag;
    u32 cur_match = (raw_m4 >= eb) ? (raw_m4 - eb) : kEmpty;
    if (cur_match >= pos) cur_match = kEmpty;

    u32* ptr0 = &son[2 * pos + 1];
    u32* ptr1 = &son[2 * pos];
    usize len0 = 0;
    usize len1 = 0;
    int depth = max_depth;

    while (cur_match != kEmpty && cur_match < pos && depth-- > 0) {
      const u32 dist = static_cast<u32>(pos - cur_match);
      u32* const pair = &son[2 * cur_match];
      const u8* const pb_src = src + cur_match;
      usize pb = std::min(len0, len1);
      if (pb >= limit) break;

      if (pb_src[pb] == cur[pb]) {
        ++pb;
        while (pb + 8 <= limit) {
          const u64 d = load_u64(cur + pb) ^ load_u64(pb_src + pb);
          if (d != 0) {
            pb += static_cast<usize>(ctz64(d) >> 3);
            break;
          }
          pb += 8;
        }
        while (pb < limit && pb_src[pb] == cur[pb]) ++pb;

        if (pb > max_len && pb >= 4 && count + 1 < kMaxMatchLen + 4) {
          max_len = pb;
          pairs[count++] = {static_cast<u16>(pb), dist};
        }
        if (pb >= limit) {
          *ptr1 = pair[0];
          *ptr0 = pair[1];
          return count;
        }
      }

      if (pb >= limit) break;
      if (pb_src[pb] < cur[pb]) {
        *ptr1 = cur_match;
        ptr1 = pair + 1;
        cur_match = *ptr1;
        len1 = pb;
      } else {
        *ptr0 = cur_match;
        ptr0 = pair;
        cur_match = *ptr0;
        len0 = pb;
      }
      if (cur_match >= pos) {
        cur_match = kEmpty;
        break;
      }
    }

    *ptr0 = kEmpty;
    *ptr1 = kEmpty;
    return count;
  }

  void skip(usize pos) noexcept {
    if (pos >= n) return;
    const usize limit = std::min<usize>(kMaxMatchLen, n - pos);
    if (limit < 2) return;
    const u8* const cur = src + pos;
    const u32 eb = epoch_base;
    const u32 cur_tag = eb + static_cast<u32>(pos);
    h2[load_u16(cur)] = cur_tag;
    if (limit >= 3) h3[hash3_fn(cur)] = cur_tag;
    if (limit < 4) return;

    const u32 hv4 = hash4_fn(cur);
    const u32 raw_m4 = head4[hv4];
    head4[hv4] = cur_tag;
    u32 cur_match = (raw_m4 >= eb) ? (raw_m4 - eb) : kEmpty;
    if (cur_match >= pos) cur_match = kEmpty;

    u32* ptr0 = &son[2 * pos + 1];
    u32* ptr1 = &son[2 * pos];
    usize len0 = 0;
    usize len1 = 0;
    int depth = skip_depth;

    while (cur_match != kEmpty && cur_match < pos && depth-- > 0) {
      u32* const pair = &son[2 * cur_match];
      const u8* const pb_src = src + cur_match;
      usize pb = std::min(len0, len1);
      if (pb < limit && pb_src[pb] == cur[pb]) {
        ++pb;
        while (pb + 8 <= limit) {
          const u64 d = load_u64(cur + pb) ^ load_u64(pb_src + pb);
          if (d != 0) {
            pb += static_cast<usize>(ctz64(d) >> 3);
            break;
          }
          pb += 8;
        }
        while (pb < limit && pb_src[pb] == cur[pb]) ++pb;
        if (pb >= limit) {
          *ptr1 = pair[0];
          *ptr0 = pair[1];
          return;
        }
      }
      if (pb >= limit) break;
      if (pb_src[pb] < cur[pb]) {
        *ptr1 = cur_match;
        ptr1 = pair + 1;
        cur_match = *ptr1;
        len1 = pb;
      } else {
        *ptr0 = cur_match;
        ptr0 = pair;
        cur_match = *ptr0;
        len0 = pb;
      }
    }
    *ptr0 = kEmpty;
    *ptr1 = kEmpty;
  }
};

// ─── Token emitter (writes into the 5 decoupled streams) ────────────────────

struct LaserStreams {
  RangeEnc rc_flag;
  RangeEnc rc_ld;
  RangeEnc rc_hi;
  RangeEnc rc_lo;
  LaserRawWriter raw_bits;
  std::vector<u8> arena_storage;
  bool lp_aligned_ = false;

  u32 state = 0;
  u32 reps[4] = {1, 1, 1, 1};
  static constexpr u32 pb_mask = 3;

  void init(u8* arena, usize stream_cap) noexcept {
    state = 0;
    reps[0] = reps[1] = reps[2] = reps[3] = 1;
    rc_flag.init(arena + 0 * stream_cap, stream_cap);
    rc_ld.init(arena + 1 * stream_cap, stream_cap);
    rc_hi.init(arena + 2 * stream_cap, stream_cap);
    rc_lo.init(arena + 3 * stream_cap, stream_cap);
    raw_bits.init(arena + 4 * stream_cap, stream_cap);
  }

  void init(usize n, u32 lc, u32 lp, u32 pb) {
    (void)lc;
    (void)pb;
    lp_aligned_ = (lp == 2);
    const usize cap = n + (n / 4) + 256;
    arena_storage.resize(5 * cap);
    init(arena_storage.data(), cap);
  }

  SPECTRA8_ALWAYS_INLINE void emit_literal(LaserModel& m, const u8* src,
                                       usize pos) noexcept {
    if (lp_aligned_) emit_literal<true>(m, src, pos);
    else emit_literal<false>(m, src, pos);
  }

  SPECTRA8_ALWAYS_INLINE u32 get_lit_ctx(usize pos, u8 prev_byte) const noexcept {
    return lp_aligned_ ? laser_lit_ctx<true>(pos, prev_byte >> 4)
                       : laser_lit_ctx<false>(pos, prev_byte >> 4);
  }

  template <bool AlignedLp>
  SPECTRA8_ALWAYS_INLINE void emit_literal(LaserModel& m, const u8* src,
                                       usize pos) noexcept {
    const u32 ps = static_cast<u32>(pos) & pb_mask;
    rc_flag.encode_bit0(m.is_match[state][ps]);
    const u8 prev_byte = (pos > 0) ? src[pos - 1] : 0;
    const u32 lit_ctx = laser_lit_ctx<AlignedLp>(pos, prev_byte >> 4);
    const u8 byte = src[pos];
    const u32 hi = byte >> 4;
    const u32 lo = byte & 15u;
    u16* hp = m.hi_probs[lit_ctx];
    u16* lp = m.lo_probs[lit_ctx][hi];

    if (state >= 7) {
      const u8 match_byte = (pos >= reps[0]) ? src[pos - reps[0]] : 0;
      const u32 m_hi = match_byte >> 4;
      const u32 m_lo = match_byte & 15u;
      encode_matched_nibble(rc_hi, hp, hi, m_hi);
      if (hi == m_hi) {
        encode_matched_nibble(rc_lo, lp, lo, m_lo);
      } else {
        encode_nibble(rc_lo, lp, lo);
      }
    } else {
      encode_nibble(rc_hi, hp, hi);
      encode_nibble(rc_lo, lp, lo);
    }
    state = kLiteralNextState[state];
  }

  SPECTRA8_ALWAYS_INLINE void emit_short_rep(LaserModel& m, usize pos) noexcept {
    const u32 ps = static_cast<u32>(pos) & pb_mask;
    rc_flag.encode_bit1(m.is_match[state][ps]);
    rc_flag.encode_bit1(m.is_rep[state]);
    rc_flag.encode_bit0(m.is_rep0[state]);
    rc_flag.encode_bit0(m.is_rep0_long[state][ps]);
    state = kShortRepNextState[state];
  }

  SPECTRA8_ALWAYS_INLINE void emit_long_rep(LaserModel& m, usize pos, u32 rep_idx,
                                        u32 len) noexcept {
    const u32 ps = static_cast<u32>(pos) & pb_mask;
    rc_flag.encode_bit1(m.is_match[state][ps]);
    rc_flag.encode_bit1(m.is_rep[state]);
    if (rep_idx == 0) {
      rc_flag.encode_bit0(m.is_rep0[state]);
      rc_flag.encode_bit1(m.is_rep0_long[state][ps]);
    } else {
      rc_flag.encode_bit1(m.is_rep0[state]);
      const u32 d = reps[rep_idx];
      if (rep_idx == 1) {
        rc_flag.encode_bit0(m.is_rep1[state]);
      } else {
        rc_flag.encode_bit1(m.is_rep1[state]);
        rc_flag.encode_bit(m.is_rep2[state], rep_idx - 2);
        if (rep_idx == 3) reps[3] = reps[2];
        reps[2] = reps[1];
      }
      reps[1] = reps[0];
      reps[0] = d;
    }
    m.rep_len_coder.encode(rc_ld, ps, len);
    state = kRepNextState[state];
  }

  SPECTRA8_ALWAYS_INLINE void emit_normal_match(LaserModel& m, usize pos, u32 dist,
                                            u32 len) noexcept {
    const u32 ps = static_cast<u32>(pos) & pb_mask;
    rc_flag.encode_bit1(m.is_match[state][ps]);
    rc_flag.encode_bit0(m.is_rep[state]);
    m.len_coder.encode(rc_ld, ps, len);

    const u32 len_state = (len - 2 < 3) ? (len - 2) : 3u;
    const u32 dist_val = dist - 1;
    const u32 slot = get_dist_slot(dist_val);
    encode_bittree<6>(rc_ld, m.dist_slot[len_state], slot);

    if (slot >= 4) {
      const int footer_bits = static_cast<int>((slot >> 1) - 1);
      const u32 base = (2u | (slot & 1u)) << footer_bits;
      const u32 rem = dist_val - base;
      if (slot < 14) {
        encode_reverse_bittree(rc_ld, m.spec_dist[slot - 4],
                               footer_bits, rem);
      } else {
        const int direct_bits = footer_bits - 4;
        raw_bits.put(rem >> 4, direct_bits);
        encode_reverse_bittree_fixed<4>(rc_ld, m.align_probs, rem & 15u);
      }
    }

    reps[3] = reps[2];
    reps[2] = reps[1];
    reps[1] = reps[0];
    reps[0] = dist;
    state = kMatchNextState[state];
  }

  void finish() noexcept {
    rc_flag.finish();
    rc_ld.finish();
    rc_hi.finish();
    rc_lo.finish();
    raw_bits.finish();
  }
};

// ─── Helper: match length at given offset ───────────────────────────────────

SPECTRA8_ALWAYS_INLINE u32 rep_match_len(const u8* src, usize pos, usize n,
                                     u32 dist) noexcept {
  if (pos < dist) return 0;
  const usize limit = std::min<usize>(kMaxMatchLen, n - pos);
  if (limit == 0) return 0;
  const u8* a = src + pos;
  const u8* b = src + pos - dist;
  if (a[0] != b[0]) return 0;
  usize l = 1;
  while (l + 8 <= limit) {
    const u64 d = load_u64(a + l) ^ load_u64(b + l);
    if (d != 0) {
      l += static_cast<usize>(ctz64(d) >> 3);
      return static_cast<u32>(l);
    }
    l += 8;
  }
  while (l < limit && a[l] == b[l]) ++l;
  return static_cast<u32>(l);
}

// ─── Local Cluster Optimal Parser (levels 3..12) ────────────────────────────

struct OptNode {
  u32 price = kInfinityPrice;
  u16 prev_cur = 0;
  u16 prev_2_cur = 0;
  u32 back_dist = 0;     // 0..3 = rep0..rep3, >= 4 = normal dist + 4, 0xFFFFFFFF = literal
  u32 prev_2_dist = 0;   // for 2-step composite edges (Match/Rep + Lit + Rep0)
  u8 has_prev_2 = 0;     // 0 = single step, 1 = Lit + Rep0, 2 = Match/Rep + Lit + Rep0
  u8 state = 0;
  u32 reps[4] = {1, 1, 1, 1};
};

struct EmittedOp {
  u16 len;
  u32 back;  // 0xFFFFFFFF = literal, 0..3 = rep0..rep3 (or ShortRep if len==1), >=4 = dist + 4
};

void parse_laser_optimal(const u8* src, usize n, int level, [[maybe_unused]] u32 lc,
                         [[maybe_unused]] u32 lp, [[maybe_unused]] u32 pb,
                         LaserModel& model, LaserStreams& streams) {
  LaserMatchFinder mf;
  mf.init(src, n, level);

  const u32 fast_bytes = (level <= 4) ? 32u : (level <= 6) ? 48u : (level <= 8) ? 64u : 96u;
  constexpr usize kOptCap = 1024;
  constexpr usize kClusterStop = kOptCap - kMaxMatchLen - 4;

  std::vector<OptNode> dp(kOptCap);
  MatchPair matches[kMaxMatchLen + 4];
  std::vector<EmittedOp> ops;
  ops.reserve(128);

  usize pos = 0;
  if (n > 0) {
    // Byte 0 is always a literal
    (void)mf.get_matches(0, matches);
    streams.emit_literal(model, src, 0);
    pos = 1;
  }

  while (pos < n) {
    const u32 num_matches = mf.get_matches(pos, matches);
    const u32 max_match_len = (num_matches > 0) ? matches[num_matches - 1].len : 0;

    u32 rep_lens[4];
    u32 best_rep_len = 0;
    u32 best_rep_idx = 0;
    for (u32 r = 0; r < 4; ++r) {
      rep_lens[r] = rep_match_len(src, pos, n, streams.reps[r]);
      if (rep_lens[r] > best_rep_len) {
        best_rep_len = rep_lens[r];
        best_rep_idx = r;
      }
    }

    // Fast path 1: long repeat or match >= fast_bytes
    if (best_rep_len >= fast_bytes) {
      streams.emit_long_rep(model, pos, best_rep_idx, best_rep_len);
      for (u32 i = 1; i < best_rep_len; ++i) mf.skip(pos + i);
      pos += best_rep_len;
      continue;
    }
    if (max_match_len >= fast_bytes) {
      const u32 dist = matches[num_matches - 1].dist;
      streams.emit_normal_match(model, pos, dist, max_match_len);
      for (u32 i = 1; i < max_match_len; ++i) mf.skip(pos + i);
      pos += max_match_len;
      continue;
    }

    // Fast path 2: no >=2 match or rep at `pos`
    if (max_match_len < 2 && best_rep_len < 2) {
      if (rep_lens[0] == 1) {
        const u32 ps = static_cast<u32>(pos) & streams.pb_mask;
        const u32 s = streams.state;
        const u32 lit_ctx = streams.get_lit_ctx(pos, src[pos - 1]);
        const u8 match_byte =
            (pos >= streams.reps[0]) ? src[pos - streams.reps[0]] : 0;
        const u32 p_lit = price_bit0(model.is_match[s][ps]) +
                          model.price_literal(lit_ctx, s, src[pos], match_byte);
        const u32 p_srep = price_bit1(model.is_match[s][ps]) +
                           price_bit1(model.is_rep[s]) +
                           price_bit0(model.is_rep0[s]) +
                           price_bit0(model.is_rep0_long[s][ps]);
        if (p_srep < p_lit) {
          streams.emit_short_rep(model, pos);
          ++pos;
          continue;
        }
      }
      streams.emit_literal(model, src, pos);
      ++pos;
      continue;
    }

    // Start a local DP cluster anchored at `base_pos = pos`
    const usize base_pos = pos;
    const usize rem_n = n - base_pos;
    const usize max_cap = std::min(kOptCap - 1, rem_n);
    usize last_end = std::min(std::max<usize>(1, std::max(max_match_len, best_rep_len)), max_cap);

    dp[0].price = 0;
    dp[0].state = static_cast<u8>(streams.state);
    dp[0].has_prev_2 = 0;
    for (int r = 0; r < 4; ++r) dp[0].reps[r] = streams.reps[r];
    for (usize i = 1; i <= last_end; ++i) dp[i].price = kInfinityPrice;

    for (usize cur = 0; cur < last_end; ++cur) {
      const u32 base_price = dp[cur].price;
      if (base_price >= kInfinityPrice) continue;

      const usize abs_pos = base_pos + cur;
      if (abs_pos >= n) break;

      // Advance state & reps if cur > 0
      if (cur > 0) {
        const OptNode& prev = dp[dp[cur].prev_cur];
        u32 st = prev.state;
        u32 r0 = prev.reps[0], r1 = prev.reps[1], r2 = prev.reps[2], r3 = prev.reps[3];

        if (dp[cur].has_prev_2 == 1) {
          // prev -> Literal -> Rep0
          st = kLiteralNextState[st];
          st = kRepNextState[st];
        } else if (dp[cur].has_prev_2 == 2) {
          // prev -> Match/Rep -> Literal -> Rep0
          const OptNode& prev2 = dp[dp[cur].prev_2_cur];
          st = prev2.state;
          r0 = prev2.reps[0];
          r1 = prev2.reps[1];
          r2 = prev2.reps[2];
          r3 = prev2.reps[3];
          const u32 d2 = dp[cur].prev_2_dist;
          if (d2 < 4) {
            if (d2 > 0) {
              const u32 tmp = prev2.reps[d2];
              if (d2 == 3) r3 = r2;
              if (d2 >= 2) r2 = r1;
              r1 = r0;
              r0 = tmp;
            }
            st = kRepNextState[st];
          } else {
            r3 = r2;
            r2 = r1;
            r1 = r0;
            r0 = d2 - 4;
            st = kMatchNextState[st];
          }
          st = kLiteralNextState[st];
          st = kRepNextState[st];
        } else {
          const u32 bd = dp[cur].back_dist;
          const usize step_len = cur - dp[cur].prev_cur;
          if (bd == 0xFFFFFFFFu) {
            st = kLiteralNextState[st];
          } else if (bd < 4) {
            if (step_len == 1) {
              st = kShortRepNextState[st];
            } else {
              if (bd > 0) {
                const u32 tmp = prev.reps[bd];
                if (bd == 3) r3 = r2;
                if (bd >= 2) r2 = r1;
                r1 = r0;
                r0 = tmp;
              }
              st = kRepNextState[st];
            }
          } else {
            r3 = r2;
            r2 = r1;
            r1 = r0;
            r0 = bd - 4;
            st = kMatchNextState[st];
          }
        }
        dp[cur].state = static_cast<u8>(st);
        dp[cur].reps[0] = r0;
        dp[cur].reps[1] = r1;
        dp[cur].reps[2] = r2;
        dp[cur].reps[3] = r3;
      }

      if (cur >= kClusterStop) {
        last_end = cur;
        break;
      }

      // Fetch matches for `cur > 0`
      u32 cur_num_matches = 0;
      if (cur == 0) {
        cur_num_matches = num_matches;
      } else {
        cur_num_matches = mf.get_matches(abs_pos, matches);
      }
      const u32 cur_max_m =
          (cur_num_matches > 0) ? matches[cur_num_matches - 1].len : 0;

      u32 cur_rep_lens[4];
      u32 cur_best_rep = 0;
      u32 cur_best_rep_idx = 0;
      for (u32 r = 0; r < 4; ++r) {
        cur_rep_lens[r] = rep_match_len(src, abs_pos, n, dp[cur].reps[r]);
        if (cur_rep_lens[r] > cur_best_rep) {
          cur_best_rep = cur_rep_lens[r];
          cur_best_rep_idx = r;
        }
      }

      // If we hit a very long match/rep inside the cluster, stop extending
      if (cur > 0 && (cur_max_m >= fast_bytes || cur_best_rep >= fast_bytes)) {
        u32 flen = 0, fback = 0;
        if (cur_best_rep >= cur_max_m) {
          flen = cur_best_rep;
          fback = cur_best_rep_idx;
        } else {
          flen = cur_max_m;
          fback = matches[cur_num_matches - 1].dist + 4;
        }
        const usize dest = cur + flen;
        if (dest <= max_cap) {
          if (dest > last_end) {
            for (usize i = last_end + 1; i <= dest; ++i) dp[i].price = kInfinityPrice;
            last_end = dest;
          }
          dp[dest].price = 0;
          dp[dest].prev_cur = static_cast<u16>(cur);
          dp[dest].back_dist = fback;
          dp[dest].has_prev_2 = 0;
        }
        for (u32 i = 1; i < flen; ++i) mf.skip(abs_pos + i);
        break;
      }

      const usize new_reach =
          std::min(max_cap, cur + std::max<usize>(1, std::max(cur_max_m, cur_best_rep)));
      if (new_reach > last_end) {
        for (usize i = last_end + 1; i <= new_reach; ++i) {
          dp[i].price = kInfinityPrice;
        }
        last_end = new_reach;
      }

      const u32 st = dp[cur].state;
      const u32 ps = static_cast<u32>(abs_pos) & streams.pb_mask;
      const u32 m0_price = price_bit0(model.is_match[st][ps]);
      const u32 m1_price = price_bit1(model.is_match[st][ps]);
      const u32 rep_yes_price = m1_price + price_bit1(model.is_rep[st]);
      const u32 rep_no_price = m1_price + price_bit0(model.is_rep[st]);

      // 1. Literal transition (cur -> cur + 1)
      const u8 prev_byte = (abs_pos > 0) ? src[abs_pos - 1] : 0;
      const u32 lit_ctx = streams.get_lit_ctx(abs_pos, prev_byte);
      const u8 match_byte =
          (abs_pos >= dp[cur].reps[0]) ? src[abs_pos - dp[cur].reps[0]] : 0;
      const u32 lit_price =
          base_price + m0_price +
          model.price_literal(lit_ctx, st, src[abs_pos], match_byte);

      if (cur + 1 <= last_end && lit_price < dp[cur + 1].price) {
        dp[cur + 1].price = lit_price;
        dp[cur + 1].prev_cur = static_cast<u16>(cur);
        dp[cur + 1].back_dist = 0xFFFFFFFFu;
        dp[cur + 1].has_prev_2 = 0;
      }

      // 1b. Composite: Literal + Rep0 (catches 1-byte differences along rep0)
      if (abs_pos + 1 < n) {
        const u32 r0_after_lit =
            rep_match_len(src, abs_pos + 1, n, dp[cur].reps[0]);
        if (r0_after_lit >= 2) {
          const usize dest = cur + 1 + r0_after_lit;
          if (dest <= max_cap) {
            if (dest > last_end) {
              for (usize i = last_end + 1; i <= dest; ++i)
                dp[i].price = kInfinityPrice;
              last_end = dest;
            }
            const u32 st_next = kLiteralNextState[st];
            const u32 ps_next = static_cast<u32>(abs_pos + 1) & streams.pb_mask;
            const u32 r0_price =
                lit_price + price_bit1(model.is_match[st_next][ps_next]) +
                price_bit1(model.is_rep[st_next]) +
                price_bit0(model.is_rep0[st_next]) +
                price_bit1(model.is_rep0_long[st_next][ps_next]) +
                model.rep_len_coder.price(ps_next, r0_after_lit);
            if (r0_price < dp[dest].price) {
              dp[dest].price = r0_price;
              dp[dest].prev_cur = static_cast<u16>(cur);
              dp[dest].back_dist = 0;
              dp[dest].has_prev_2 = 1;
            }
          }
        }
      }

      // 2. ShortRep (cur -> cur + 1)
      if (cur_rep_lens[0] >= 1) {
        const u32 srep_price = rep_yes_price + price_bit0(model.is_rep0[st]) +
                               price_bit0(model.is_rep0_long[st][ps]);
        if (cur + 1 <= last_end && base_price + srep_price < dp[cur + 1].price) {
          dp[cur + 1].price = base_price + srep_price;
          dp[cur + 1].prev_cur = static_cast<u16>(cur);
          dp[cur + 1].back_dist = 0;
          dp[cur + 1].has_prev_2 = 0;
        }
      }

      // 3. LongRep (r = 0..3, len = 2..cur_rep_lens[r])
      for (u32 r = 0; r < 4; ++r) {
        const u32 rlen = cur_rep_lens[r];
        if (rlen < 2) continue;
        u32 r_prefix = rep_yes_price;
        if (r == 0) {
          r_prefix += price_bit0(model.is_rep0[st]) +
                      price_bit1(model.is_rep0_long[st][ps]);
        } else if (r == 1) {
          r_prefix += price_bit1(model.is_rep0[st]) + price_bit0(model.is_rep1[st]);
        } else if (r == 2) {
          r_prefix += price_bit1(model.is_rep0[st]) +
                      price_bit1(model.is_rep1[st]) + price_bit0(model.is_rep2[st]);
        } else {
          r_prefix += price_bit1(model.is_rep0[st]) +
                      price_bit1(model.is_rep1[st]) + price_bit1(model.is_rep2[st]);
        }

        for (u32 l = 2; l <= rlen; ++l) {
          if (cur + l > last_end) break;
          const u32 p = base_price + r_prefix + model.rep_len_coder.price(ps, l);
          if (p < dp[cur + l].price) {
            dp[cur + l].price = p;
            dp[cur + l].prev_cur = static_cast<u16>(cur);
            dp[cur + l].back_dist = r;
            dp[cur + l].has_prev_2 = 0;
          }
        }

        // Composite: Rep(rlen) + 1 MatchedLiteral + Rep0(len2)
        if (abs_pos + rlen + 2 <= n) {
          const u32 r_dist = dp[cur].reps[r];
          const u32 len2 = rep_match_len(src, abs_pos + rlen + 1, n, r_dist);
          if (len2 >= 2) {
            const usize dest = cur + rlen + 1 + len2;
            if (dest <= max_cap) {
              if (dest > last_end) {
                for (usize i = last_end + 1; i <= dest; ++i)
                  dp[i].price = kInfinityPrice;
                last_end = dest;
              }
              const u32 st1 = kRepNextState[st];
              const usize lit_p = abs_pos + rlen;
              const u32 ps1 = static_cast<u32>(lit_p) & streams.pb_mask;
              const u32 lctx1 = streams.get_lit_ctx(lit_p, src[lit_p - 1]);
              const u32 st2 = kLiteralNextState[st1];
              const u32 ps2 = static_cast<u32>(lit_p + 1) & streams.pb_mask;
              const u32 comp_price =
                  base_price + r_prefix + model.rep_len_coder.price(ps, rlen) +
                  price_bit0(model.is_match[st1][ps1]) +
                  model.price_literal(lctx1, st1, src[lit_p], src[lit_p - r_dist]) +
                  price_bit1(model.is_match[st2][ps2]) +
                  price_bit1(model.is_rep[st2]) +
                  price_bit0(model.is_rep0[st2]) +
                  price_bit1(model.is_rep0_long[st2][ps2]) +
                  model.rep_len_coder.price(ps2, len2);
              if (comp_price < dp[dest].price) {
                dp[dest].price = comp_price;
                dp[dest].prev_cur = static_cast<u16>(cur + rlen + 1);
                dp[dest].prev_2_cur = static_cast<u16>(cur);
                dp[dest].prev_2_dist = r;
                dp[dest].back_dist = 0;
                dp[dest].has_prev_2 = 2;
              }
            }
          }
        }
      }

      // 4. Normal Matches
      if (cur_num_matches > 0) {
        u32 start_len = 2;
        for (u32 mi = 0; mi < cur_num_matches; ++mi) {
          const u32 mlen = matches[mi].len;
          const u32 mdist = matches[mi].dist;
          if (mlen < start_len) continue;

          for (u32 l = start_len; l <= mlen; ++l) {
            if (cur + l > last_end) break;
            const u32 p = base_price + rep_no_price +
                          model.len_coder.price(ps, l) +
                          model.price_dist(l, mdist);
            if (p < dp[cur + l].price) {
              dp[cur + l].price = p;
              dp[cur + l].prev_cur = static_cast<u16>(cur);
              dp[cur + l].back_dist = mdist + 4;
              dp[cur + l].has_prev_2 = 0;
            }
          }

          // Composite: Match(mlen, mdist) + 1 MatchedLiteral + Rep0(len2)
          if (abs_pos + mlen + 2 <= n) {
            const u32 len2 = rep_match_len(src, abs_pos + mlen + 1, n, mdist);
            if (len2 >= 2) {
              const usize dest = cur + mlen + 1 + len2;
              if (dest <= max_cap) {
                if (dest > last_end) {
                  for (usize i = last_end + 1; i <= dest; ++i)
                    dp[i].price = kInfinityPrice;
                  last_end = dest;
                }
                const u32 st1 = kMatchNextState[st];
                const usize lit_p = abs_pos + mlen;
                const u32 ps1 = static_cast<u32>(lit_p) & streams.pb_mask;
                const u32 lctx1 = streams.get_lit_ctx(lit_p, src[lit_p - 1]);
                const u32 st2 = kLiteralNextState[st1];
                const u32 ps2 = static_cast<u32>(lit_p + 1) & streams.pb_mask;
                const u32 comp_price =
                    base_price + rep_no_price +
                    model.len_coder.price(ps, mlen) +
                    model.price_dist(mlen, mdist) +
                    price_bit0(model.is_match[st1][ps1]) +
                    model.price_literal(lctx1, st1, src[lit_p],
                                        src[lit_p - mdist]) +
                    price_bit1(model.is_match[st2][ps2]) +
                    price_bit1(model.is_rep[st2]) +
                    price_bit0(model.is_rep0[st2]) +
                    price_bit1(model.is_rep0_long[st2][ps2]) +
                    model.rep_len_coder.price(ps2, len2);
                if (comp_price < dp[dest].price) {
                  dp[dest].price = comp_price;
                  dp[dest].prev_cur = static_cast<u16>(cur + mlen + 1);
                  dp[dest].prev_2_cur = static_cast<u16>(cur);
                  dp[dest].prev_2_dist = mdist + 4;
                  dp[dest].back_dist = 0;
                  dp[dest].has_prev_2 = 2;
                }
              }
            }
          }

          start_len = mlen + 1;
        }
      }
    }

    // Backtrack from `last_end` to `0`
    while (last_end > 0 && dp[last_end].price >= kInfinityPrice) {
      --last_end;
    }
    if (last_end == 0) {
      streams.emit_literal(model, src, pos);
      ++pos;
      continue;
    }

    ops.clear();
    usize cur = last_end;
    while (cur > 0) {
      const OptNode& node = dp[cur];
      if (node.has_prev_2 == 1) {
        const u16 p = node.prev_cur;
        const u16 rep_len = static_cast<u16>(cur - p - 1);
        ops.push_back({rep_len, 0u});
        ops.push_back({1u, 0xFFFFFFFFu});
        cur = p;
      } else if (node.has_prev_2 == 2) {
        const u16 p1 = node.prev_cur;  // position after Match/Rep + 1 Lit
        const u16 p0 = node.prev_2_cur;
        const u16 rep_len = static_cast<u16>(cur - p1);
        const u16 m_len = static_cast<u16>(p1 - 1 - p0);
        ops.push_back({rep_len, 0u});
        ops.push_back({1u, 0xFFFFFFFFu});
        ops.push_back({m_len, node.prev_2_dist});
        cur = p0;
      } else {
        const u16 p = node.prev_cur;
        ops.push_back({static_cast<u16>(cur - p), node.back_dist});
        cur = p;
      }
    }

    // Note: during cluster exploration `mf.get_matches` was called for `cur = 0..last_end-1`.
    // Wait: if `last_end` was extended past the loop stop or if some positions
    // inside `0..last_end-1` were skipped because `dp[cur].price == kInfinityPrice`,
    // Wait: `dp[cur].price` is ALWAYS finite for every `cur >= 0` because every
    // reachable `cur` relaxes `cur + 1` via a literal!
    // What if the loop broke early at `cur` (`last_end = cur; break;`)?
    // Then `mf.get_matches` was already called for `0 .. last_end` (inclusive of `cur = last_end`),
    // SO wait: if it broke at `cur`, `mf.get_matches(base_pos + cur)` WAS ALREADY CALLED!
    // To avoid calling `mf.get_matches(base_pos + last_end)` a second time on the next outer
    // iteration, let's check: only call `mf.get_matches` inside the cluster when we are NOT breaking,
    // OR check `cur_max_m` after `get_matches` only when we commit to advancing past `cur`!
    // Wait: in the loop above, `cur` runs `0 .. last_end - 1`, so `mf.get_matches` is called for
    // `base_pos + 0 .. base_pos + last_end - 1` (unless it broke early at some `cur`, in which case
    // `mf.get_matches` was called for `base_pos + cur` as well!).
    // Let's make sure `mf.get_matches` is NEVER called on `base_pos + last_end` inside the cluster:
    // if we don't break early (since `kClusterStop` check can happen BEFORE `mf.get_matches`),
    // then `mf.get_matches` is called for `0 .. last_end - 1`!
    // Wait: in the loop above, if `cur_max_m >= fast_bytes`, instead of breaking BEFORE relaxing `cur`,
    // if we relax `cur` and THEN stop extending `last_end` (`last_end = cur + std::max(cur_max_m, cur_best_rep);` and skip `cur+1..last_end-1` via `mf.skip`),
    // then `mf` is advanced through `base_pos + last_end - 1`!
    // Let's check: in the normal case where the loop doesn't break early, `cur` runs from `0` to `last_end - 1`,
    // so `mf.get_matches` was called for `base_pos + 0` through `base_pos + last_end - 1`!
    // And what if the loop DID break early at `cur`? Let's just NOT break early before relaxing `cur`:
    // instead, if `cur >= kClusterStop`, don't extend `last_end` any further, and let `cur` finish up to `last_end - 1`!
    // Wait: let's check what happens if we just replace that early `break` with capping `new_reach`:
    // if `cur` always runs from `0` to `last_end - 1`, then `mf.get_matches` is called on EVERY position `base_pos + 0 .. base_pos + last_end - 1`!

    for (auto it = ops.rbegin(); it != ops.rend(); ++it) {
      const u32 len = it->len;
      const u32 back = it->back;
      if (back == 0xFFFFFFFFu) {
        streams.emit_literal(model, src, pos);
      } else if (back < 4) {
        if (len == 1) {
          streams.emit_short_rep(model, pos);
        } else {
          streams.emit_long_rep(model, pos, back, len);
        }
      } else {
        streams.emit_normal_match(model, pos, back - 4, len);
      }
      pos += len;
    }
  }
}

// ─── Fast Greedy / Rep-Aware Parser (levels 1..2) ───────────────────────────

void parse_laser_fast(const u8* src, usize n, int level, LaserModel& model,
                      LaserStreams& streams) {
  LaserMatchFinder mf;
  mf.init(src, n, level);
  MatchPair matches[kMaxMatchLen + 4];

  usize pos = 0;
  if (n > 0) {
    (void)mf.get_matches(0, matches);
    streams.emit_literal(model, src, 0);
    pos = 1;
  }

  while (pos < n) {
    const u32 num_matches = mf.get_matches(pos, matches);
    const u32 max_match_len = (num_matches > 0) ? matches[num_matches - 1].len : 0;

    u32 best_rep_len = 0;
    u32 best_rep_idx = 0;
    for (u32 r = 0; r < 4; ++r) {
      const u32 rl = rep_match_len(src, pos, n, streams.reps[r]);
      if (rl > best_rep_len) {
        best_rep_len = rl;
        best_rep_idx = r;
      }
    }

    if (best_rep_len >= 2 && best_rep_len + 1 >= max_match_len) {
      streams.emit_long_rep(model, pos, best_rep_idx, best_rep_len);
      for (u32 i = 1; i < best_rep_len; ++i) mf.skip(pos + i);
      pos += best_rep_len;
      continue;
    }

    if (max_match_len >= 2) {
      // Pick match with best bit price
      const u32 ps = static_cast<u32>(pos) & streams.pb_mask;
      u32 best_len = matches[num_matches - 1].len;
      u32 best_dist = matches[num_matches - 1].dist;

      if (best_len <= 4 && best_dist > 512 && num_matches >= 2) {
        // Check if a shorter, much closer match is cheaper per byte
        const u32 prev_len = matches[num_matches - 2].len;
        const u32 prev_dist = matches[num_matches - 2].dist;
        if (prev_len >= 2 && prev_dist * 32 < best_dist) {
          best_len = prev_len;
          best_dist = prev_dist;
        }
      }
      if (best_len == 2 && best_dist >= 128) {
        // 2-byte match at distance >= 128 costs more than 2 literals
        if (rep_match_len(src, pos, n, streams.reps[0]) >= 1) {
          streams.emit_short_rep(model, pos);
        } else {
          streams.emit_literal(model, src, pos);
        }
        ++pos;
        continue;
      }

      (void)ps;
      streams.emit_normal_match(model, pos, best_dist, best_len);
      for (u32 i = 1; i < best_len; ++i) mf.skip(pos + i);
      pos += best_len;
      continue;
    }

    if (best_rep_len == 1 && best_rep_idx == 0) {
      const u32 ps = static_cast<u32>(pos) & streams.pb_mask;
      const u32 s = streams.state;
      const u32 lit_ctx = streams.get_lit_ctx(pos, src[pos - 1]);
      const u8 match_byte = src[pos - streams.reps[0]];
      const u32 p_lit = price_bit0(model.is_match[s][ps]) +
                        model.price_literal(lit_ctx, s, src[pos], match_byte);
      const u32 p_srep = price_bit1(model.is_match[s][ps]) +
                         price_bit1(model.is_rep[s]) +
                         price_bit0(model.is_rep0[s]) +
                         price_bit0(model.is_rep0_long[s][ps]);
      if (p_srep < p_lit) {
        streams.emit_short_rep(model, pos);
        ++pos;
        continue;
      }
    }

    streams.emit_literal(model, src, pos);
    ++pos;
  }
}

}  // namespace

// ─── Public Block Entry Points ──────────────────────────────────────────────

void build_laser_block(const u8* src, usize n, int level, std::vector<u8>& out,
                       bool already_filtered) {
  out.clear();
  if (n == 0) return;

  // Built-in BCJ executable normalization when frame filter did not already apply it
  u32 bcj_mode = 0;  // 0 = none, 1 = x86, 2 = arm64
  std::vector<u8> bcj_buf;
  const u8* encode_src = src;

  ExecArch arch = detect_exec_arch(src, n);
  if (!already_filtered && n >= 64) {
    if (arch == ExecArch::X86) {
      bcj_buf.resize(n);
      bcj_x86_fwd(src, n, bcj_buf.data());
      encode_src = bcj_buf.data();
      bcj_mode = 1;
    } else if (arch == ExecArch::Arm64) {
      bcj_buf.resize(n);
      bcj_arm64_fwd(src, n, bcj_buf.data());
      encode_src = bcj_buf.data();
      bcj_mode = 2;
    }
  }

  // Choose literal/pos context parameters:
  //   ARM64 instructions are 4-byte aligned -> (lc=2, lp=2, pb=2)
  //   x86 / general binary -> (lc=3, lp=0, pb=2)
  const bool use_aligned_lp = (arch == ExecArch::Arm64);
  const u32 lc = use_aligned_lp ? 2u : 3u;
  const u32 lp = use_aligned_lp ? 2u : 0u;
  const u32 pb = 2u;
  const u32 num_lit_ctx = 1u << (lc + lp);

  auto model = std::make_unique<LaserModel>();
  model->init(num_lit_ctx);

  LaserStreams streams;
  streams.init(n, lc, lp, pb);

  if (level <= 2) {
    parse_laser_fast(encode_src, n, level, *model, streams);
  } else {
    // Fix early-break inside parse_laser_optimal by running optimal parser cleanly
    parse_laser_optimal(encode_src, n, level, lc, lp, pb, *model, streams);
  }

  streams.finish();

  const usize total_size =
      18 + streams.rc_flag.size() + streams.rc_ld.size() +
      streams.rc_hi.size() + streams.rc_lo.size() + streams.raw_bits.size();
  if (total_size >= n) return;

  out.resize(total_size);
  u8* p = out.data();
  p[0] = kLaserMagic;
  p[1] = static_cast<u8>((bcj_mode & 3u) | ((use_aligned_lp ? 1u : 0u) << 2));
  store_u32(p + 2, static_cast<u32>(streams.rc_flag.size()));
  store_u32(p + 6, static_cast<u32>(streams.rc_ld.size()));
  store_u32(p + 10, static_cast<u32>(streams.rc_hi.size()));
  store_u32(p + 14, static_cast<u32>(streams.rc_lo.size()));
  p += 18;

  std::memcpy(p, streams.rc_flag.begin, streams.rc_flag.size());
  p += streams.rc_flag.size();
  std::memcpy(p, streams.rc_ld.begin, streams.rc_ld.size());
  p += streams.rc_ld.size();
  std::memcpy(p, streams.rc_hi.begin, streams.rc_hi.size());
  p += streams.rc_hi.size();
  std::memcpy(p, streams.rc_lo.begin, streams.rc_lo.size());
  p += streams.rc_lo.size();
  if (streams.raw_bits.size() != 0) {
    std::memcpy(p, streams.raw_bits.begin, streams.raw_bits.size());
  }
}

template <bool AlignedLp>
Status decode_laser_block_impl(const u8* __restrict data, usize len,
                               u8* __restrict dst, usize n, u32 bcj_mode) {
  const u32 flag_sz = load_u32(data + 2);
  const u32 ld_sz = load_u32(data + 6);
  const u32 hi_sz = load_u32(data + 10);
  const u32 lo_sz = load_u32(data + 14);

  const u64 r_total = 18ull + flag_sz + ld_sz + hi_sz + lo_sz;
  if (r_total > len) return Status::CorruptInput;
  const usize raw_sz = len - static_cast<usize>(r_total);

  const u8* p_flag = data + 18;
  const u8* p_ld = p_flag + flag_sz;
  const u8* p_hi = p_ld + ld_sz;
  const u8* p_lo = p_hi + hi_sz;
  const u8* p_raw = p_lo + lo_sz;

  RangeDec rc_flag;
  RangeDec rc_ld;
  RangeDec rc_hi;
  RangeDec rc_lo;
  if (!rc_flag.init(p_flag, flag_sz) || !rc_ld.init(p_ld, ld_sz) ||
      !rc_hi.init(p_hi, hi_sz) || !rc_lo.init(p_lo, lo_sz)) {
    return Status::CorruptInput;
  }
  LaserRawReader raw_bits(p_raw, raw_sz);

  constexpr u32 kNumLitCtx = AlignedLp ? 16u : 8u;
  LaserModel m;
  m.init(kNumLitCtx);

  u32 state = 0;
  u32 rep0 = 1, rep1 = 1, rep2 = 1, rep3 = 1;
  u32 prev_hi = 0;
  usize pos = 0;

  u32 hi = 0;
  u16* lp = nullptr;

check_unmatched_literal:
  if (SPECTRA8_UNLIKELY(pos >= n)) goto done;
  if (rc_flag.decode_bit(m.is_match[state][static_cast<u32>(pos) & 3u]) != 0) {
    goto handle_match;
  }
  {
    const u32 lit_ctx = laser_lit_ctx<AlignedLp>(pos, prev_hi);
    hi = decode_nibble(rc_hi, m.hi_probs[lit_ctx]);
    lp = m.lo_probs[lit_ctx][hi];
    state = kLiteralNextState[state];
  }

pipeline_literals:
  while (SPECTRA8_LIKELY(pos + 1 < n)) {
    if (rc_flag.decode_bit(
            m.is_match[state][static_cast<u32>(pos + 1) & 3u]) != 0) {
      const u32 lo = decode_nibble(rc_lo, lp);
      dst[pos] = static_cast<u8>((hi << 4) | lo);
      prev_hi = hi;
      ++pos;
      goto handle_match;
    }
    state = kLiteralNextState[state];
    const u32 next_ctx = laser_lit_ctx<AlignedLp>(pos + 1, hi);
#if defined(__aarch64__) || defined(_M_ARM64)
    u32 lo, next_hi;
    decode_nibble_pair(rc_lo, lp, lo, rc_hi, m.hi_probs[next_ctx], next_hi);
    dst[pos] = static_cast<u8>((hi << 4) | lo);
    ++pos;
    hi = next_hi;
#else
    const u32 lo = decode_nibble(rc_lo, lp);
    dst[pos] = static_cast<u8>((hi << 4) | lo);
    ++pos;
    hi = decode_nibble(rc_hi, m.hi_probs[next_ctx]);
#endif
    lp = m.lo_probs[next_ctx][hi];
  }
  {
    const u32 lo = decode_nibble(rc_lo, lp);
    dst[pos] = static_cast<u8>((hi << 4) | lo);
    prev_hi = hi;
    ++pos;
    goto done;
  }

handle_match:
  {
    const u32 ps = static_cast<u32>(pos) & 3u;
    u32 match_len = 0;
    if (rc_flag.decode_bit(m.is_rep[state]) != 0) {
      if (rc_flag.decode_bit(m.is_rep0[state]) == 0) {
        if (rc_flag.decode_bit(m.is_rep0_long[state][ps]) == 0) {
          if (SPECTRA8_UNLIKELY(pos < rep0)) {
            return Status::CorruptInput;
          }
          const u8 b = dst[pos - rep0];
          prev_hi = b >> 4;
          dst[pos++] = b;
          state = kShortRepNextState[state];
          goto check_after_match;
        }
      } else {
        u32 dist;
        if (rc_flag.decode_bit(m.is_rep1[state]) == 0) {
          dist = rep1;
        } else {
          if (rc_flag.decode_bit(m.is_rep2[state]) == 0) {
            dist = rep2;
          } else {
            dist = rep3;
            rep3 = rep2;
          }
          rep2 = rep1;
        }
        rep1 = rep0;
        rep0 = dist;
      }
      match_len = m.rep_len_coder.decode(rc_ld, ps);
      state = kRepNextState[state];
    } else {
      rep3 = rep2;
      rep2 = rep1;
      rep1 = rep0;
      match_len = m.len_coder.decode(rc_ld, ps);
      const u32 len_state = (match_len - 2 < 3) ? (match_len - 2) : 3u;
      const u32 slot = decode_bittree<6>(rc_ld, m.dist_slot[len_state]);

      if (slot < 4) {
        rep0 = slot + 1;
      } else {
        const int footer_bits = static_cast<int>((slot >> 1) - 1);
        const u32 base = (2u | (slot & 1u)) << footer_bits;
        if (slot < 14) {
          rep0 = base +
                 decode_reverse_bittree(rc_ld, m.spec_dist[slot - 4],
                                        footer_bits) +
                 1;
        } else {
          const int direct_bits = footer_bits - 4;
          if (SPECTRA8_UNLIKELY(!raw_bits.can_read(direct_bits))) {
            return Status::CorruptInput;
          }
          const u32 mid = static_cast<u32>(raw_bits.get(direct_bits));
          const u32 low4 = decode_reverse_bittree<4>(rc_ld, m.align_probs);
          rep0 = base + (mid << 4) + low4 + 1;
        }
      }
      state = kMatchNextState[state];
    }

    if (SPECTRA8_UNLIKELY(rep0 == 0 || pos < rep0 || pos + match_len > n)) {
      return Status::CorruptInput;
    }

    u8* d = dst + pos;
    const u8* s = d - rep0;
    if (SPECTRA8_LIKELY(pos + match_len + 16 <= n)) {
      if (SPECTRA8_LIKELY(rep0 >= 16)) {
        copy16(d, s);
        if (SPECTRA8_UNLIKELY(match_len > 16)) {
          usize i = 16;
          do {
            copy16(d + i, s + i);
            i += 16;
          } while (i < match_len);
        }
      } else if (rep0 >= 8) {
        copy8(d, s);
        copy8(d + 8, s + 8);
        if (SPECTRA8_UNLIKELY(match_len > 16)) {
          usize i = 16;
          do {
            copy8(d + i, s + i);
            i += 8;
          } while (i < match_len);
        }
      } else if ((rep0 & (rep0 - 1u)) == 0) {
        u64 pat8;
        if (rep0 == 1) {
          pat8 = static_cast<u64>(s[0]) * 0x0101010101010101ull;
        } else if (rep0 == 2) {
          pat8 = static_cast<u64>(load_u16(s)) * 0x0001000100010001ull;
        } else {
          pat8 = static_cast<u64>(load_u32(s)) * 0x0000000100000001ull;
        }
        store_u64(d, pat8);
        store_u64(d + 8, pat8);
        if (SPECTRA8_UNLIKELY(match_len > 16)) {
          usize i = 16;
          do {
            store_u64(d + i, pat8);
            store_u64(d + i + 8, pat8);
            i += 16;
          } while (i < match_len);
        }
      } else {
        usize i = 0;
        do {
#if defined(__GNUC__) || defined(__clang__)
          __asm__ volatile("" : "+r"(d));
#endif
          d[i] = s[i];
          ++i;
        } while (i < match_len);
      }
    } else {
      usize i = 0;
      do {
#if defined(__GNUC__) || defined(__clang__)
        __asm__ volatile("" : "+r"(d));
#endif
        d[i] = s[i];
        ++i;
      } while (i < match_len);
    }

    pos += match_len;
    prev_hi = dst[pos - 1] >> 4;
  }

check_after_match:
  if (SPECTRA8_UNLIKELY(pos >= n)) goto done;
  if (rc_flag.decode_bit(m.is_match[state][static_cast<u32>(pos) & 3u]) != 0) {
    goto handle_match;
  }
  {
    const u32 lit_ctx = laser_lit_ctx<AlignedLp>(pos, prev_hi);
    const u8 match_byte = dst[pos - rep0];
    const u32 m_hi = match_byte >> 4;
    hi = decode_matched_nibble(rc_hi, m.hi_probs[lit_ctx], m_hi);
    lp = m.lo_probs[lit_ctx][hi];
    state = kLiteralNextState[state];
    if (SPECTRA8_LIKELY(hi != m_hi)) {
      goto pipeline_literals;
    }
    const u32 m_lo = match_byte & 15u;
    const u32 lo = decode_matched_nibble(rc_lo, lp, m_lo);
    dst[pos] = static_cast<u8>((hi << 4) | lo);
    prev_hi = hi;
    ++pos;
    goto check_unmatched_literal;
  }

done:
  if (bcj_mode == 1) {
    bcj_x86_inv(dst, n, dst);
  } else if (bcj_mode == 2) {
    bcj_arm64_inv(dst, n, dst);
  }
  return Status::Ok;
}

Status decode_laser_block(const u8* data, usize len, u8* dst, usize n) {
  if (n == 0) return Status::Ok;
  if (len < 18 || data[0] != kLaserMagic) return Status::CorruptInput;

  const u8 mode = data[1];
  const u32 bcj_mode = mode & 3u;
  const bool use_aligned_lp = ((mode >> 2) & 1u) != 0;
  if (bcj_mode > 2) return Status::CorruptInput;

  if (use_aligned_lp) {
    return decode_laser_block_impl<true>(data, len, dst, n, bcj_mode);
  }
  return decode_laser_block_impl<false>(data, len, dst, n, bcj_mode);
}

}  // namespace spectra8::detail
