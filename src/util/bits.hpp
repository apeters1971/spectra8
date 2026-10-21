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

// Branch-light MSB-first bit writer / reader.
//
// MSB-first packing makes canonical Huffman codes natural: a code of `len` bits
// with canonical value `code` is emitted directly with put(code, len), and the
// decoder can peek the top bits to index a flat lookup table.
#ifndef SPECTRA8_SRC_BITS_HPP
#define SPECTRA8_SRC_BITS_HPP

#include <cstdint>
#include <cstring>
#include <vector>

#include "util/common.hpp"

namespace spectra8::detail {

// Big-endian (MSB-first) 64-bit load. Compilers fold this into a single load +
// byte-swap. Used by BitReader's fast refill path.
inline u64 load_be64(const u8* p) noexcept {
  return (static_cast<u64>(p[0]) << 56) | (static_cast<u64>(p[1]) << 48) |
         (static_cast<u64>(p[2]) << 40) | (static_cast<u64>(p[3]) << 32) |
         (static_cast<u64>(p[4]) << 24) | (static_cast<u64>(p[5]) << 16) |
         (static_cast<u64>(p[6]) << 8) | static_cast<u64>(p[7]);
}

inline void store_be64(u8* p, u64 v) noexcept {
#if defined(__GNUC__) || defined(__clang__)
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  v = __builtin_bswap64(v);
#endif
  std::memcpy(p, &v, 8);
#else
  for (int i = 0; i < 8; ++i) p[i] = static_cast<u8>(v >> (56 - 8 * i));
#endif
}

// Branchless MSB-first bit packer for hot encode loops (same bitstream as
// BitWriter). Pending bits live in the low `n` bits of `acc`; flush() stores
// them as the top of a big-endian 8-byte word and advances by whole bytes,
// leaving at most 7 bits pending, so any sequence of puts totalling <= 57
// bits may run between two flushes with no branch at all. The destination
// must have 8 bytes of slack past the last byte written; `finish()` pads the
// final partial byte with zero bits and returns the end pointer.
struct MsbPacker {
  u8* p;
  u64 acc = 0;
  u32 n = 0;

  explicit MsbPacker(u8* dst) noexcept : p(dst) {}
  void put(u32 bits, u32 len) noexcept {
    acc = (acc << len) | bits;
    n += len;
  }
  void flush() noexcept {
    // (acc << (63 - n)) << 1 == acc << (64 - n), and is defined for n == 0.
    store_be64(p, (acc << (63 - n)) << 1);
    p += n >> 3;
    n &= 7;
  }
  u8* finish() noexcept {
    flush();
    if (n) *p++ = static_cast<u8>(acc << (8 - n));
    n = 0;
    return p;
  }
};

// MSB-first bit writer. Bits accumulate in a 64-bit word and leave in whole
// 32-bit chunks, so a put is a shift-or plus one branch taken every few
// symbols (the byte-at-a-time loop with a push_back per byte was ~7% of the
// encoder). The writer owns the vector's tail between construction and
// flush(): it grows the vector ahead of its cursor and trims it in flush(),
// so nothing else may append to `out` in between. Bits above the pending
// ones in acc_ are stale and never read (the 32-bit truncation below takes
// exactly the pending chunk).
class BitWriter {
 public:
  explicit BitWriter(std::vector<u8>& out) noexcept
      : out_(out), pos_(out.size()) {}

  // Append the low `n` bits of `bits` (n in [0, 32]).
  void put(u32 bits, int n) noexcept {
    const u32 mask = (n == 32) ? 0xFFFFFFFFu : ((1u << n) - 1u);
    acc_ = (acc_ << n) | (bits & mask);
    nbits_ += n;
    if (nbits_ >= 32) {
      nbits_ -= 32;
      if (pos_ + 4 > out_.size()) grow();
      const u32 w = static_cast<u32>(acc_ >> nbits_);
      u8* d = out_.data() + pos_;
      d[0] = static_cast<u8>(w >> 24);
      d[1] = static_cast<u8>(w >> 16);
      d[2] = static_cast<u8>(w >> 8);
      d[3] = static_cast<u8>(w);
      pos_ += 4;
    }
  }

  // Flush remaining bits, zero-padding the final byte, and trim the vector to
  // the bytes written.
  void flush() noexcept {
    if (pos_ + 4 > out_.size()) grow();
    while (nbits_ >= 8) {
      nbits_ -= 8;
      out_[pos_++] = static_cast<u8>(acc_ >> nbits_);
    }
    if (nbits_ > 0) {
      out_[pos_++] = static_cast<u8>(acc_ << (8 - nbits_));
      nbits_ = 0;
    }
    acc_ = 0;
    out_.resize(pos_);
  }

 private:
  void grow() { out_.resize(out_.size() * 2 + 64); }

  std::vector<u8>& out_;
  usize pos_;
  u64 acc_ = 0;
  int nbits_ = 0;
};

// MSB-first bit reader for hot decode loops (same bitstream as BitReader):
// the next bits to decode sit at the TOP of `acc`, `c` counts bits consumed
// since the last refill (whole bytes of it advance `p` at the next refill). A
// refill is then three data-independent operations (advance, mask, one
// big-endian load shifted by c & 7) with no branch on the fill level, and a
// table-driven symbol step is shift -> table load -> shift. The reader's
// state is just (p, c): `acc` is re-derived by the next refill, so it can be
// stored between incremental decode calls. After a refill at least 57 bits
// are valid (c & 7 <= 7 were already consumed).
//
// Bits past the end of the stream read as zero (the encoder pads the last
// byte with zero bits; callers only consume as many bits as were coded).
struct MsbReader {
  const u8* p;
  const u8* end;
  u64 acc;
  u32 c;
  u32 left;  // valid bits at the top of acc; 0 until the first refill

  MsbReader() noexcept : p(nullptr), end(nullptr), acc(0), c(0), left(0) {}
  MsbReader(const u8* data, usize len) noexcept
      : p(data), end(data + len), acc(0), c(0), left(0) {}

  // Whether refill_fast may be used (8 bytes readable at the new position).
  bool has8() const noexcept { return p + (c >> 3) + 8 <= end; }
  void refill_fast() noexcept {
    p += c >> 3;
    c &= 7;
    acc = load_be64(p) << c;
    left = 64 - c;
  }
  void refill_safe() noexcept {
    p += c >> 3;
    c &= 7;
    u8 tmp[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const usize avail = p < end ? static_cast<usize>(end - p) : 0;
    std::memcpy(tmp, p, avail < 8 ? avail : 8);
    acc = load_be64(tmp) << c;
    left = 64 - c;
  }
  // Reload only when the accumulator does not already hold `nbits`.
  // Offset streams typically spend 8-17 bits per new match, so one 64-bit
  // refill covers several sequences; the old per-sequence reload was a
  // load_be64 + shift on every one.
  void ensure(u32 nbits) noexcept {
    if (left >= nbits) return;
    if (__builtin_expect(has8(), 1))
      refill_fast();
    else
      refill_safe();
  }
  // Next `nbits` bits (1..32) without consuming them.
  u32 peek(int nbits) const noexcept {
    return static_cast<u32>(acc >> (64 - nbits));
  }
  // As peek, allowing nbits == 0 (returns 0).
  u32 peek0(int nbits) const noexcept {
    return static_cast<u32>((acc >> 1) >> (63 - nbits));
  }
  void consume(u32 nbits) noexcept {
    acc <<= nbits;
    c += nbits;
    left = (left > nbits) ? left - nbits : 0;
  }
};


class BitReader {
 public:
  // Reads from [data, data+len). Reads beyond the end yield zero bits, so the
  // caller is responsible for only consuming the number of bits actually
  // encoded (tracked out-of-band via element counts).
  BitReader(const u8* data, usize len) noexcept
      : p_(data), end_(data + len) {}

  void refill() noexcept {
    if (nbits_ > 56) return;
    // Fast path: when at least 8 bytes remain, grab them with one big-endian
    // 64-bit load and shift in as many whole bytes as the accumulator can hold
    // (the rest are re-read on the next refill). This replaces the byte-at-a-
    // time loop's ~7 branchy loads with a single load + a couple of shifts.
    if (p_ + 8 <= end_) {
      const u64 next = load_be64(p_);
      const int can = (64 - nbits_) >> 3;  // whole bytes that fit: 1..8
      if (can == 8) {
        acc_ = next;  // nbits_ == 0 here, so this is the whole word
      } else {
        acc_ = (acc_ << (can * 8)) | (next >> ((8 - can) * 8));
      }
      p_ += can;
      nbits_ += can * 8;
      return;
    }
    // Tail path (near the end of the buffer): byte-at-a-time, zero-padding past
    // the end so callers only ever consume bits that were actually encoded.
    while (nbits_ <= 56) {
      u8 b = (p_ < end_) ? *p_++ : 0;
      acc_ = (acc_ << 8) | b;
      nbits_ += 8;
    }
  }

  // Look at the next `n` bits without consuming them (n in [1, 24]).
  u32 peek(int n) noexcept {
    if (nbits_ < n) refill();
    return static_cast<u32>(acc_ >> (nbits_ - n)) & ((1u << n) - 1u);
  }

  // Number of bits currently buffered (valid without a refill).
  int buffered() const noexcept { return nbits_; }

  // Peek the next `n` bits assuming at least `n` are already buffered (no
  // refill). The caller must guarantee buffered() >= n (e.g. by calling
  // refill() and then consuming at most 56 bits before the next refill). Both
  // peek variants mask their result to `n` bits, so consume() does not need to
  // clear the consumed bits out of the accumulator.
  u32 peek_buffered(int n) const noexcept {
    return static_cast<u32>(acc_ >> (nbits_ - n)) & ((1u << n) - 1u);
  }

  void consume(int n) noexcept { nbits_ -= n; }

  // Read and consume the next `n` bits (n in [0, 32]).
  u32 get(int n) noexcept {
    if (n == 0) return 0;
    if (nbits_ < n) refill();
    const u32 mask = (n == 32) ? 0xFFFFFFFFu : ((1u << n) - 1u);
    u32 v = static_cast<u32>(acc_ >> (nbits_ - n)) & mask;
    consume(n);
    return v;
  }

 private:
  const u8* p_;
  const u8* end_;
  u64 acc_ = 0;
  int nbits_ = 0;
};

}  // namespace spectra8::detail

#endif  // SPECTRA8_SRC_BITS_HPP
