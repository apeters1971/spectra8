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

// bzip2-style RUNA/RUNB encoding of zero runs in an MTF rank stream.
#ifndef SPECTRA8_SRC_BWT_MTF_RLE_HPP
#define SPECTRA8_SRC_BWT_MTF_RLE_HPP

#include <vector>

#include "prism/bwt_decode.hpp"
#include "util/common.hpp"
#include "util/simd.hpp"

namespace spectra8::detail {

inline constexpr u8 kBwtRunA = 0;
inline constexpr u8 kBwtRunB = 1;
inline constexpr u8 kBwtLitEsc = 2;  // followed by rank 254 or 255

// Flush `zpend` consecutive MTF rank-0 symbols (bzip2 compress.c).
inline void bwt_mtf_flush_zeros(u32 zpend, std::vector<u8>& out) {
  if (zpend == 0) return;
  i32 zp = static_cast<i32>(zpend);
  zp--;
  while (true) {
    out.push_back(static_cast<u8>((zp & 1) ? kBwtRunB : kBwtRunA));
    if (zp < 2) break;
    zp = (zp - 2) / 2;
  }
}

// MTF ranks in [0,255]: rank 0 never appears as a literal; runs use RUNA/RUNB.
inline void bwt_mtf_bzip2_encode(const u8* mtf, usize n, std::vector<u8>& out) {
  out.clear();
  // Worst case is one output byte per input (no zero runs); RUNA/RUNB and the
  // rare lit-escape add a few more. Reserve once to avoid push_back growth.
  out.reserve(n + (n >> 6) + 8);
  u32 zpend = 0;
  usize i = 0;
  while (i < n) {
    if (mtf[i] == 0) {
      const usize run = count_eq_run(mtf + i, n - i, 0);
      zpend += static_cast<u32>(run);
      i += run;
      continue;
    }
    bwt_mtf_flush_zeros(zpend, out);
    zpend = 0;
    const u8 r = mtf[i];
    if (r > 252) {
      out.push_back(kBwtLitEsc);
      out.push_back(r);
    } else {
      out.push_back(static_cast<u8>(r + 3));  // ranks 1..252 -> 4..255
    }
    ++i;
  }
  bwt_mtf_flush_zeros(zpend, out);
}

// Expand one RUNA/RUNB sequence (matches bzip2/decompress.c MTF run loop).
inline usize bwt_mtf_bzip2_decode_run(const u8* sym, usize i, usize sym_n,
                                      usize& zeros) {
  const usize start = i;
  i32 es = -1;
  i32 N = 1;
  while (i < sym_n && sym[i] <= kBwtRunB) {
    if (sym[i] == kBwtRunA)
      es += N;
    else
      es += 2 * N;
    N *= 2;
    ++i;
  }
  zeros = (i == start) ? 0 : static_cast<usize>(es + 1);
  return i;
}

// Recover the MTF rank stream (zeros expanded) from a mode-2 symbol block.
// `reserve_hint` should be the final rank count when known (avoids reallocs).
inline void bwt_mtf_bzip2_decode(const u8* sym, usize sym_n, std::vector<u8>& mtf,
                                 usize reserve_hint = 0) {
  mtf.clear();
  mtf.reserve(reserve_hint > sym_n ? reserve_hint : sym_n);
  usize i = 0;
  while (i < sym_n) {
    if (sym[i] <= kBwtRunB) {
      usize run = 0;
      i = bwt_mtf_bzip2_decode_run(sym, i, sym_n, run);
      mtf.insert(mtf.end(), run, 0);
    } else if (sym[i] == kBwtLitEsc) {
      if (i + 1 >= sym_n) return;
      mtf.push_back(sym[i + 1]);
      i += 2;
    } else if (sym[i] >= 4) {
      mtf.push_back(static_cast<u8>(sym[i] - 3));  // 4..255 -> ranks 1..252
      ++i;
    } else {
      return;  // corrupt (reserved byte 3)
    }
  }
}

// Fused mode-2 inverse: expand RUNA/RUNB + invert MTF in a single pass, writing
// the recovered BWT bytes straight into dst[0, n). This replaces the separate
// (expand rank stream -> mtf_inverse) two-pass + temp buffer used by mode 0/1.
//
// MTF state mirrors bwt_decode::mtf_inverse (register-resident list head, see
// bwt_decode::MtfList). A zero-run is a run of rank 0, which under MTF
// inversion just repeats the current front byte (order unchanged), so it
// becomes a single memset. The literal case is tested first: it is the most
// frequent symbol class on every input measured. Returns bytes written.
inline usize bwt_mtf_bzip2_inverse(const u8* sym, usize sym_n, u8* dst, usize n) {
  bwt_decode::MtfList list;
  list.init();

  u8 front = 0;  // rank-0 byte (0 initially)
  usize o = 0;
  usize i = 0;
  while (i < sym_n && o < n) {
    const u8 t = sym[i];
    if (t >= 4) {
      front = list.pull(static_cast<u32>(t) - 3);
      dst[o++] = front;
      ++i;
    } else if (t <= kBwtRunB) {
      usize run = 0;
      i = bwt_mtf_bzip2_decode_run(sym, i, sym_n, run);
      if (run > n - o) run = n - o;
      std::memset(dst + o, front, run);
      o += run;
    } else if (t == kBwtLitEsc) {
      if (i + 1 >= sym_n) break;
      front = list.pull(sym[i + 1]);
      dst[o++] = front;
      i += 2;
    } else {
      break;  // corrupt (reserved byte 3)
    }
  }
  return o;
}

}  // namespace spectra8::detail

#endif  // SPECTRA8_SRC_BWT_MTF_RLE_HPP
