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

// Pre-codec byte run-length coding (LZ and BWT). Shuffle/delta filters group identical
// bytes into long runs (especially zeros); this collapses them before LZ/BWT so
// a 32 KiB zero lane becomes a few bytes instead of thousands of literals or
// scattered BWT symbols.
//
// Stream format (LEB128 varints, same as the frame format):
//   0, varint(n), n literal bytes
//   1, u8 value, varint(n)   (n >= kByteRleMinRun)
#ifndef SPECTRA8_SRC_BYTE_RLE_HPP
#define SPECTRA8_SRC_BYTE_RLE_HPP

#include <vector>

#include "util/common.hpp"

namespace spectra8::detail {

inline constexpr u8 kByteRleLitTag = 0;
inline constexpr u8 kByteRleRunTag = 1;
inline constexpr usize kByteRleMinRun = 4;

inline void byte_rle_encode(const u8* src, usize n, std::vector<u8>& out) {
  out.clear();
  usize i = 0;
  while (i < n) {
    usize run = 1;
    while (i + run < n && src[i + run] == src[i]) ++run;
    if (run >= kByteRleMinRun) {
      ByteWriter w(out);
      w.byte(kByteRleRunTag);
      w.byte(src[i]);
      w.varint(run);
      i += run;
      continue;
    }
    const usize start = i;
    while (i < n) {
      usize r = 1;
      while (i + r < n && src[i + r] == src[i] && r < kByteRleMinRun) ++r;
      if (r >= kByteRleMinRun) break;
      ++i;
    }
    const usize lit = i - start;
    ByteWriter w(out);
    w.byte(kByteRleLitTag);
    w.varint(lit);
    w.bytes(src + start, lit);
  }
}

inline Status byte_rle_decode(const u8* data, usize len, u8* dst, usize n) {
  ByteReader r(data, len);
  usize o = 0;
  while (o < n) {
    const u8 tag = r.byte();
    if (r.error) return Status::CorruptInput;
    if (tag == kByteRleRunTag) {
      const u8 v = r.byte();
      const u64 run = r.varint();
      if (r.error || run == 0 || o + run > n) return Status::CorruptInput;
      std::memset(dst + o, v, static_cast<usize>(run));
      o += static_cast<usize>(run);
    } else if (tag == kByteRleLitTag) {
      const u64 lit = r.varint();
      if (r.error || lit == 0 || o + lit > n) return Status::CorruptInput;
      const u8* p = r.take(static_cast<usize>(lit));
      if (!p) return Status::CorruptInput;
      std::memcpy(dst + o, p, static_cast<usize>(lit));
      o += static_cast<usize>(lit);
    } else {
      return Status::CorruptInput;
    }
  }
  if (o != n || r.remaining() != 0) return Status::CorruptInput;
  return Status::Ok;
}

}  // namespace spectra8::detail

#endif  // SPECTRA8_SRC_BYTE_RLE_HPP
