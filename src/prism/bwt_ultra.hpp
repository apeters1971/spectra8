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

// BWT Ultra: Bidirectional Context-Mixing & Multi-State Run Model (Mode 5)
//
// Inspired by Substring Enumeration (CSE) and the M03 context-interval model:
// Combines:
//   1. Forward Context (F-column interval): Predicts L[i] conditioned on the
//      true right context symbol fc = F[i] in the original text (reconstructed
//      from the global alphabet cumulative distribution).
//   2. Backward Suffix Context: Order-1 (c1) and Order-2 (c2) Markov contexts
//      along adjacent suffix array rows.
//   3. 8-State Run Automaton: Models BWT cluster dynamics and run lengths.
//   4. Multi-State Interpolated SSE: Secondary Symbol Estimation conditioned
//      on (bit_ctx, run_state).

#ifndef SPECTRA8_SRC_BWT_ULTRA_HPP
#define SPECTRA8_SRC_BWT_ULTRA_HPP

#include <cstddef>
#include <cstdint>
#include <vector>

#include "util/common.hpp"

namespace spectra8::detail {

// Encodes raw BWT L-column src[0, n) using BWT Ultra (Mode 5).
// Appends the compressed stream to out. Returns true on success.
bool bwt_ultra_encode(const std::uint8_t* src, std::size_t n,
                      std::vector<std::uint8_t>& out);

// Decodes BWT Ultra (Mode 5) stream src[0, src_len) into dst[0, n).
// Returns Status::Ok on success, Status::CorruptInput on bad data.
Status bwt_ultra_decode(const std::uint8_t* src, std::size_t src_len,
                        std::uint8_t* dst, std::size_t n);

}  // namespace spectra8::detail

#endif  // SPECTRA8_SRC_BWT_ULTRA_HPP
