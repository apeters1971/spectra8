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

// Order-0 adaptive range (arithmetic) coder for skewed byte streams (BWT/MTF).
#ifndef SPECTRA8_SRC_ARITH_HPP
#define SPECTRA8_SRC_ARITH_HPP

#include "util/common.hpp"

namespace spectra8::detail::arith {

// Encode `n` bytes; appends payload (no tag) to `out`.
void encode(std::vector<u8>& out, const u8* src, usize n);

// Encode using a precomputed histogram (seeds the adaptive model).
void encode(std::vector<u8>& out, const u8* src, usize n, const u32 freq[256]);

// Decode exactly `n` bytes from [data, data+len).
Status decode(const u8* data, usize len, u8* dst, usize n);

// Adaptive order-0 range coder: no frequency header, the model updates as it
// codes. Better than the static variant on short / piecewise-stationary streams
// (BWT/MTF) where the 512-byte static header dominates.
void encode_adaptive(std::vector<u8>& out, const u8* src, usize n);

// Decode exactly `n` bytes written by encode_adaptive.
Status decode_adaptive(const u8* data, usize len, u8* dst, usize n);

// Predicted encode_adaptive payload size (bytes, no tag). Same model as the
// encoder; a log2 table walk without range renormalize. Used to skip the
// real adaptive encode when AnsO1 already beats this bound.
double estimate_adaptive(const u8* src, usize n);

// Order-1 variant: one adaptive model per coarse class of the previous
// symbol (o1_class). Same per-symbol decode cost as the order-0 coder
// (one model lookup + linear scan + O(1) update), but the conditional
// distributions of BWT/MTF streams - "in a run" vs "churning" - are modeled
// separately, which is worth several percent on them.
//
// Uses a coarser 4-class map than the static AnsO1 coder: adaptive models
// need more samples per class to warm up, and splitting further was measured
// to stick the slow path more often without a matching ratio win.
void encode_adaptive_o1(std::vector<u8>& out, const u8* src, usize n);
Status decode_adaptive_o1(const u8* data, usize len, u8* dst, usize n);

constexpr int kO1Ctx = 4;
inline int o1_class(u8 prev) noexcept {
  if (prev <= 1) return prev;
  return prev < 8 ? 2 : 3;
}

}  // namespace spectra8::detail::arith

#endif  // SPECTRA8_SRC_ARITH_HPP
