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

// Oracle: learned-context compression.
//
// Per tile, k-means clusters the 256 rows of P(next | prev) into K static
// contexts. A per-cluster affine predictor (tiny linear model, Q8 weights)
// is fit on those clusters; the causal residual is then raced against the
// raw bytes through learned rANS, Spectra8 LZ, and Mosaic's mini-BWT last-step.
// Keep-smaller: at least as strong as the BWT/LZ race that beats zstd on
// the official filtered planes, plus a fitted predictor when it helps.
// Decode stays table-driven (rANS / LZ / inverse BWT) plus a few muls.
#ifndef SPECTRA8_SRC_ORACLE_HPP
#define SPECTRA8_SRC_ORACLE_HPP

#include "util/common.hpp"

namespace spectra8::detail {

void build_oracle_block(const u8* src, usize n, int level, std::vector<u8>& payload);

Status decode_oracle_block(const u8* data, usize len, u8* dst, usize n);

}  // namespace spectra8::detail

#endif  // SPECTRA8_SRC_ORACLE_HPP
