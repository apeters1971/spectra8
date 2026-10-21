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

// Length-limited canonical Huffman coder over a 256-symbol alphabet.
#ifndef SPECTRA8_SRC_HUFFMAN_HPP
#define SPECTRA8_SRC_HUFFMAN_HPP

#include "util/bits.hpp"
#include "util/common.hpp"

namespace spectra8::detail::huff {

// Maximum Huffman code length. Bounds the decode table to 2^kMaxCodeLen.
inline constexpr int kMaxCodeLen = 11;

// Encode n (>0) bytes from `src`, appending the payload (table + bitstream) to
// `out`. The container framing / tag byte is written by the caller.
void encode(std::vector<u8>& out, const u8* src, usize n);

// As above, but reusing a caller-supplied 256-symbol frequency histogram of
// `src` (avoids an extra pass when the caller already has it).
void encode(std::vector<u8>& out, const u8* src, usize n, const u32 freq[256]);

// Size in bytes the payload of encode(out, src, n, freq) will have, within a
// few bytes (per-stream padding is approximated). Builds the code lengths
// only - no bitstream - so callers can pick a coder before encoding.
usize encoded_size(const u32 freq[256], usize n);

// Decode exactly `n` bytes into `dst` from [data, data+len).
Status decode(const u8* data, usize len, u8* dst, usize n);

// Compact table headers (shared with the rANS coder): a canonical prefix
// code over an alphabet of up to kSmallMaxSyms symbols with code lengths
// <= kSmallMaxLen, whose own lengths are sent as 3 bits per symbol. Used to
// code the Huffman code lengths and the rANS frequency classes, which the
// plain 4-bit / 16-bit tables spent 2-4x more on.
inline constexpr int kSmallMaxSyms = 16;
inline constexpr int kSmallMaxLen = 7;
struct SmallCode {
  u8 len[kSmallMaxSyms];
  u8 code[kSmallMaxSyms];
  u16 dec[1 << kSmallMaxLen];  // decoder: sym << 8 | len; 0 = no code
};
// Code for `freq` (nsym <= kSmallMaxSyms). False when the length limit
// cannot be met (then the caller falls back to a plain table).
bool small_code_build(const u32* freq, int nsym, SmallCode& sc);
void small_code_write(BitWriter& w, const SmallCode& sc, int nsym);
inline void small_code_put(BitWriter& w, const SmallCode& sc, int s) {
  w.put(sc.code[s], sc.len[s]);
}
// Reads nsym 3-bit lengths and builds sc.dec; false when the lengths do not
// form a prefix code (over-subscribed).
bool small_code_read(BitReader& r, int nsym, SmallCode& sc);
// Next symbol, or -1 when the bits match no code.
inline int small_code_get(BitReader& r, const SmallCode& sc) {
  const u16 e = sc.dec[r.peek(kSmallMaxLen)];
  if (!(e & 0xFF)) return -1;
  r.consume(e & 0xFF);
  return e >> 8;
}

// Incremental Huffman decode: `init` parses the payload (no container tag byte);
// `decode` appends the next `count` bytes to `dst` (quad mode emits in source
// order: stream0, then stream1, ...).
class StreamDecoder {
 public:
  Status init(const u8* data, usize len, usize total_out);
  Status decode(u8* dst, usize count);
  usize remaining() const noexcept { return total_ - done_; }

 private:
  usize total_ = 0;
  usize done_ = 0;
  bool single_ = true;
  int pair_bits_ = 0;  // pair-table width built for this payload (0: none)
  MsbReader rd_[4];    // per-stream reader state
  int quad_si_ = 0;
  usize quad_sd_ = 0;
  usize quad_lim_[4] = {0, 0, 0, 0};
};

}  // namespace spectra8::detail::huff

#endif  // SPECTRA8_SRC_HUFFMAN_HPP
