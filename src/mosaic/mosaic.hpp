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

// Mosaic: lattice-ROLZ raced with a static-entropy BWT and an LZ last-step.
//
// A new Spectra8 block codec (BlockType::Mosaic). Each block is split into tiles
// (32 KiB at low levels, up to 256 KiB / 1 MiB at higher ones). A tile is
// the smaller of:
//   * ROLZ — matches are an index into a per-context recency list, plus an
//     explicit-offset copy (far hash, lattice multiples of stride 2/4/8,
//     last offsets). Used at low levels and as a small-block fallback.
//   * mini-BWT + static AnsO1 / Huffman, raced with bzip2-style RUNA/RUNB
//     on the MTF ranks and a Spectra8 LZ last-step on the raw L-column.
// No adaptive range coder on the default (L5) path: decode stays
// table-driven (ranked copies, LZ, or a small inverse BWT) around a few
// hundred MB/s, while the BWT/LZ race keeps ratio ahead of zstd on the
// official filtered numeric planes.
#ifndef SPECTRA8_SRC_MOSAIC_HPP
#define SPECTRA8_SRC_MOSAIC_HPP

#include "util/common.hpp"

namespace spectra8::detail {

void build_mosaic_block(const u8* src, usize n, int level, std::vector<u8>& payload);

Status decode_mosaic_block(const u8* data, usize len, u8* dst, usize n);

// Mini-BWT last-step used by Mosaic tiles (and Oracle's keep-smaller race).
void mosaic_encode_tbwt_tile(const u8* src, usize n, int level, std::vector<u8>& out);
Status mosaic_decode_tbwt_tile(ByteReader& r, u8* dst, usize n);

}  // namespace spectra8::detail

#endif  // SPECTRA8_SRC_MOSAIC_HPP
