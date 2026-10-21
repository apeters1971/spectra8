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

// Laser (BlockType::Laser / Codec::Laser): LZMA2-class binary compressor with
// a 5-stream decoupled branchless range decoder for fast decompression.
//
// Why Laser exists alongside Beam, Prism, and Crystal:
//   Shared libraries (.so / .dylib), executables, object files, and mixed
//   binary blobs combine non-stationary sections (.dynsym, .rela.dyn, .text,
//   .rodata, .eh_frame) where redundancy is short-range, position-aligned
//   (pos & 3), and interrupted by 1-2 byte operand/field differences.
//
//   Laser models that structure directly (12-state LZMA state machine, 4
//   repeat offsets with 1-byte ShortRep and >=2-byte LongRep, matched-literal
//   trees conditioned on dst[pos - rep0], position-aligned length/state
//   models, 64 distance slots with 4-bit alignment trees, built-in x86/ARM64
//   BCJ normalization, and a single-pass local cluster optimal parser using
//   live adaptive bit prices), while breaking LZMA's serial single-stream
//   range-decoder bottleneck into 5 independent streams:
//     1. rc_flag : state transition & match/rep flags
//     2. rc_ld   : match/rep lengths, distance slots, and low-bit trees
//     3. rc_hi   : literal top nibbles (bits 7..4; determines next lit_ctx)
//     4. rc_lo   : literal bottom nibbles (bits 3..0; off the critical path)
//     5. raw_bits: uniform middle distance bits read in one shift-and-mask
#ifndef SPECTRA8_SRC_LASER_HPP
#define SPECTRA8_SRC_LASER_HPP

#include <vector>

#include "util/common.hpp"

namespace spectra8::detail {

void build_laser_block(const u8* src, usize n, int level, std::vector<u8>& out,
                       bool already_filtered = false);

Status decode_laser_block(const u8* data, usize len, u8* dst, usize n);

}  // namespace spectra8::detail

#endif  // SPECTRA8_SRC_LASER_HPP
