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

// Static ANS entropy coder (byte-alphabet rANS, 16-bit renorm, 2/4/8-way
// interleaved). Used for LZ token streams from level 3, literals at level 9+,
// and BWT symbol streams. Named "tans" per the project layout; implemented as
// rANS, which is the modern, easy-to-verify member of the ANS family.
#ifndef SPECTRA8_SRC_TANS_HPP
#define SPECTRA8_SRC_TANS_HPP

#include "util/common.hpp"

namespace spectra8::detail::ans {

// Encode n (>0) bytes from `src`, appending the payload (table + state stream)
// to `out`. The container framing / tag byte is written by the caller.
void encode(std::vector<u8>& out, const u8* src, usize n);

// As above, but reusing a caller-supplied 256-symbol frequency histogram of
// `src` (avoids an extra pass when the caller already has it).
void encode(std::vector<u8>& out, const u8* src, usize n, const u32 counts[256]);

// Estimated payload size in bytes of encode(out, src, n, counts): the exact
// header plus the coded size under the normalized table (rANS codes within
// a few bytes of that). No stream is built.
usize encoded_size(const u32 counts[256], usize n);

// Decode exactly `n` bytes into `dst` from [data, data+len).
Status decode(const u8* data, usize len, u8* dst, usize n);

// Order-1 variant: every symbol is coded from a static table selected by the
// *previous* symbol's context class (kO1Ctx coarse classes). Captures the
// strongest conditional structure of BWT/MTF streams while decoding with
// plain table lookups - no per-symbol model update like the adaptive range
// coder. Encode appends the self-framed payload; decode reads exactly n
// bytes. Used for BWT symbol streams only (full-array decode, no streaming).
void encode_o1(std::vector<u8>& out, const u8* src, usize n);
Status decode_o1(const u8* data, usize len, u8* dst, usize n);

// Context class of the preceding symbol for the order-1 coder. Class count
// and mapping are part of the bitstream format. MTF/BWT residuals concentrate
// on small ranks, so the map splits the low ranks finely and buckets the
// tail: {0},{1},{2},{3},{4..7},{8..15},{16..31},{32+}. Eight static tables
// still decode with one table lookup per symbol; the payload uses hybrid
// dense/sparse frequency headers so the extra classes do not bloat unused
// high-rank alphabets. (Adaptive O1 keeps a coarser 4-class map - see
// arith::o1_class - because thin classes slow its warm-up.)
constexpr int kO1Ctx = 8;
void encode_o1(std::vector<u8>& out, const u8* src, usize n,
               const u32 h[kO1Ctx][256]);
inline int o1_class(u8 prev) noexcept {
  if (prev <= 3) return prev;
  if (prev < 8) return 4;
  if (prev < 16) return 5;
  if (prev < 32) return 6;
  return 7;
}

// Maximum number of interleaved rANS states in an order-0 payload.
constexpr int kMaxWays = 8;

// Incremental rANS decode (2/4/8-way interleaved), source order.
class StreamDecoder {
 public:
  Status init(const u8* data, usize len, usize total_out);
  Status decode(u8* dst, usize count);
  usize remaining() const noexcept { return total_ - done_; }

 private:
  usize total_ = 0;
  usize done_ = 0;
  bool constant_ = false;
  u8 const_sym_ = 0;
  int nway_ = 2;  // interleaved states
  const u8* ptr_[kMaxWays] = {};
  const u8* plast_ = nullptr;  // last in-bounds 16-bit word of the payload
  u32 x_[kMaxWays] = {};
};

}  // namespace spectra8::detail::ans

#endif  // SPECTRA8_SRC_TANS_HPP
