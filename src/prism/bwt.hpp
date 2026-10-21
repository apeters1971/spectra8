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

// Burrows-Wheeler Transform + Move-To-Front for the optional BWT block mode.
//
// This is the front half of a bzip-style pipeline (BWT -> MTF -> entropy). The
// entropy stage is the project's existing encode_array (which auto-selects
// ANS/FSE, RLE, Huffman or raw), so only the reversible byte-permutation and
// MTF steps live here.
#ifndef SPECTRA8_SRC_BWT_HPP
#define SPECTRA8_SRC_BWT_HPP

#include "util/common.hpp"

namespace spectra8::detail {

// Largest block for which the inverse uses the packed single-load LF table and
// can run multi-chain parallel inversion (LF target must fit in 24 bits).
inline constexpr usize kBwtPackedMax = usize{1} << 24;

// Largest block supported by BWT mode (256 MiB).
inline constexpr usize kBwtMaxBlock = usize{1} << 28;

// Maximum parallel inverse chains supported by bwt_inverse.
inline constexpr int kBwtMaxChains = 16;

// Default chains emitted on encode for large blocks (<= kBwtMaxChains). Override
// at runtime with env SPECTRA8_BWT_ENCODE_CHAINS=4|8|12|16 (see bwt_encode_chains()).
inline constexpr int kBwtEncodeChains = 16;

// Resolved parallel chain count for encoding (env override, else kBwtEncodeChains).
int bwt_encode_chains() noexcept;

// Forward rotation BWT of src[0, n) into dst[0, n). Returns the primary index
// (the row of the sorted-rotations matrix that holds the original string).
// Requires n >= 1. dst must have room for n bytes.
//
// If num_chains > 1, `anchors` (size num_chains) is filled with the starting
// row of each of the num_chains equal output segments, so the inverse can run
// that many independent LF walks in parallel (memory-level parallelism).
usize bwt_forward(const u8* src, usize n, u8* dst, usize* anchors = nullptr,
                  int num_chains = 1);

// bwt_forward with caller-chosen anchor positions: rows[c] receives the row
// of the sorted matrix whose suffix starts at positions[c] (positions[c] == n
// names the sentinel suffix, row 0). An LF walk started at rows[c] emits
// src[positions[c] - 1], src[positions[c] - 2], ... so a caller can split the
// output into segments of any lengths (e.g. skewed so parallel chains do not
// write to the same cache set). Returns the primary index like bwt_forward.
usize bwt_forward_rows(const u8* src, usize n, u8* dst, const usize* positions,
                       int count, usize* rows);

// Inverse of bwt_forward: reconstruct the original n bytes into dst from the
// BWT bytes and the primary index. Returns CorruptInput if primary > n.
//
// With num_chains > 1 and the matching `anchors` produced by bwt_forward, the
// reconstruction runs num_chains independent LF walks over disjoint output
// segments, keeping several random loads in flight at once.
Status bwt_inverse(const u8* bwt, usize n, usize primary, u8* dst,
                   const usize* anchors = nullptr, int num_chains = 1);

// In-place Move-To-Front transform over a 256-symbol alphabet.
void mtf_forward(u8* buf, usize n);

// In-place inverse Move-To-Front.
void mtf_inverse(u8* buf, usize n);

}  // namespace spectra8::detail

#endif  // SPECTRA8_SRC_BWT_HPP
