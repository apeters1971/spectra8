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

// Crystal: record-lattice modelling with an asymmetric split of the work.
//
// A new Spectra8 block codec (BlockType::Crystal) for numeric / columnar data.
// Unlike the LZ, BWT and ROLZ modes it never emits tokens: every byte is
// coded from a probability distribution conditioned on its position in the
// record lattice (the encoder detects the record stride S, 1..16) and on a
// few neighbours -- the previous byte, the byte one record back in the same
// lane, two records back, the linear trend 2*lat1 - lat2.
//
// Three payload kinds share the static coder:
//
//   * Static lattice coder (levels 1-9, crystal_static.cpp). Decompression
//     is the priority: all modelling happens in the encoder, which for each
//     lane chooses one or two context features and a "base" the byte is
//     coded relative to, merges feature values into groups with a dynamic
//     program on an MDL cost (coded bits + table bits), quantizes one
//     frequency table per group and ships the whole model in a compact,
//     adaptively coded header. The decoder does, per byte: quantizer lookup
//     -> static table -> one rANS step, on 8 interleaved streams that run
//     in lockstep and only ever read already-decoded bytes. No adaptation,
//     no mixing, no branches on the data; ~500 MB/s single-threaded.
//     Levels trade encoder search depth (features, group counts, value
//     orders) for ratio; the decoder is the same at every level.
//   * BWT text mode (levels 1-9, crystal_bwt.cpp). Text-like blocks go
//     through BWT -> MTF -> zero-run symbols and the same static coder at
//     stride 1 (MTF ranks are stationary, so static tables fit). Chosen by
//     a cost estimate against the lattice payload, never by a heuristic.
//     Decode = static decoder + fused inverse MTF + 16-chain inverse BWT,
//     350-650 MB/s on text at bzip2/bzip3-class ratios.
//   * Context mixing (levels 10-11, crystal.cpp). The archival mode: bitwise
//     adaptive counters, logistic mixers and SSE stages, a match model, a
//     per-byte "expected byte" flag on record data, 1-4 interleaved streams
//     sharing all model tables. Symmetric and slow (5-20 MB/s), a little
//     smaller than the BWT mode on text (~10%) and than the lattice coder
//     on records (~0-1%).
//
// Both kinds are self-describing (kind byte in the payload), so a decoder
// handles any level's output.
#ifndef SPECTRA8_SRC_CRYSTAL_HPP
#define SPECTRA8_SRC_CRYSTAL_HPP

#include "util/common.hpp"

namespace spectra8::detail {

void build_crystal_block(const u8* src, usize n, int level,
                        std::vector<u8>& payload, u32 profile = 0);

// Record stride S in 1..16 (MDL over per-lane order-0 entropies).
int detect_stride(const u8* src, usize n);

Status decode_crystal_block(const u8* data, usize len, u8* dst, usize n);

// Static lattice coder (fast decode), see crystal_static.cpp. Appends the
// payload body after the (ver, kind) prefix; the decoder gets the body.
void build_crystal_static(const u8* src, usize n, int level, int stride,
                         std::vector<u8>& payload);
Status decode_crystal_static(const u8* data, usize len, u8* dst, usize n);

// Pre-trained static probability profiles for Crystal (zero header overhead, >1 GB/s decode).
// Returns false, leaving `payload` untouched, when the profile is unknown or
// when the block is estimated to be smaller under a per-page model; the
// caller then encodes the block another way. `lane_modes` is set when the
// block carries per-page lane modes (constant / raw lanes skipped by the
// coder): such a block has its own kind (kCryProfileLanes), and a block
// without them keeps the kCryProfile layout.
bool build_crystal_profile(const u8* src, usize n, u32 profile_id,
                          std::vector<u8>& payload, bool& lane_modes);
Status decode_crystal_profile(const u8* data, usize len, u8* dst, usize n, bool lane_modes);
bool train_and_register_profile(u32 profile_id, const u8* sample, usize n, int stride, int level = 5);
bool train_profile_model(const u8* sample, usize n, int stride, std::vector<u8>& out, int level = 5);
bool register_profile_model(u32 profile_id, const u8* model_bytes, usize model_len);
bool export_profile_model(u32 profile_id, std::vector<u8>& out);

// BWT text mode (crystal_bwt.cpp): BWT -> MTF -> zero-run symbols, coded
// with the static lattice coder at stride 1. Same body/prefix convention.
// crystal_bwt_estimate is the detection stage: it runs the transform and
// returns an order-1 cost estimate of the payload (~usize{0} when the block
// is too small); build_crystal_bwt reuses that transform.
usize crystal_bwt_estimate(const u8* src, usize n);
void build_crystal_bwt(const u8* src, usize n, int level,
                      std::vector<u8>& payload);
Status decode_crystal_bwt(const u8* data, usize len, u8* dst, usize n);

}  // namespace spectra8::detail

#endif  // SPECTRA8_SRC_CRYSTAL_HPP
