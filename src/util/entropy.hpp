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

// Entropy "array" container: packages a byte array using the smallest of
// several encodings (memset / raw / RLE / Huffman / ANS) and frames it with its
// decoded length so the decoder can reconstruct it standalone.
#ifndef SPECTRA8_SRC_ENTROPY_HPP
#define SPECTRA8_SRC_ENTROPY_HPP

#include "util/bits.hpp"
#include "util/common.hpp"
#include "util/huffman.hpp"
#include "util/tans.hpp"

namespace spectra8::detail {

// What an LZ array holds; steers the entropy-coder choice (encoder-only, the
// payload is self-describing either way).
enum class ArrayKind : u8 {
  Literals,  // dense byte alphabet, large: Huffman unless ANS wins clearly
  Tokens     // skewed length/offset codes: ANS from level 3 (sub-bit symbols)
};

// Append a self-describing entropy-coded array of `n` bytes to `out`:
//   varint(n) varint(payload_len) payload
void encode_array(std::vector<u8>& out, const u8* src, usize n, int level,
                  ArrayKind kind = ArrayKind::Literals);

// Same blob layout, body = Crystal static lattice coder (crystal_static.cpp)
// at `level` with record stride `stride` (1 for symbol streams). Appends only
// when the whole blob is smaller than `budget` (0 = no budget); returns
// whether it did. decode_array handles the result like any other tag.
bool encode_array_crystal(std::vector<u8>& out, const u8* src, usize n,
                          int level, int stride, usize budget = 0);

// BWT/MTF symbol stream: Huffman + ANS only (no RLE candidate; MTF is pre-RLE).
void encode_bwt_entropy(std::vector<u8>& out, const u8* src, usize n, int level);

// Decode one array written by encode_array. Appends the decoded bytes to `out`.
Status decode_array(ByteReader& r, std::vector<u8>& out);

// Incremental decode of one encode_array blob (consumes the varint header from
// `r`). Writes the next `count` bytes to `dst` in source order.
class ArrayStreamDecoder {
 public:
  Status begin(ByteReader& r);
  Status decode(u8* dst, usize count);
  usize remaining() const noexcept { return total_ - done_; }

 private:
  enum class Kind : u8 { None, Memset, Raw, Huff, Ans, Rle };

  usize total_ = 0;
  usize done_ = 0;
  Kind kind_ = Kind::None;
  u8 memset_v_ = 0;
  ByteReader raw_br_{nullptr, 0};
  ByteReader rle_br_{nullptr, 0};
  usize rle_lit_remain_ = 0;
  usize rle_run_remain_ = 0;
  u8 rle_run_val_ = 0;
  huff::StreamDecoder huff_;
  ans::StreamDecoder ans_;
};

}  // namespace spectra8::detail

#endif  // SPECTRA8_SRC_ENTROPY_HPP
