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

// Internal frame/block codec entry points shared by encoder and decoder.
#ifndef SPECTRA8_SRC_CODEC_HPP
#define SPECTRA8_SRC_CODEC_HPP

#include "util/common.hpp"

namespace spectra8::detail {

// Encoder configuration resolved from the public CompressParams.
struct EncodeParams {
  int level;
  usize block_size;       // power of two within [2^kMinBlockSizeLog, 2^kMaxBlockSizeLog]
  int threads;            // <= 0 => auto
  bool seek_table;        // emit a seek-table trailer
  bool block_checksums;   // store a per-block checksum
  bool content_checksum;  // emit + verify a whole-frame checksum
  bool filters;           // auto-detect + apply pre-LZ byte transforms
  bool filter_fallback;   // when filters: also try raw and keep smaller
  bool filter_rle;        // try byte-RLE on filter/intermediate path
  bool filter_dict;       // allow dictionary-remap filters
  bool filter_sample_full;// detect over whole block (else 16 KiB mid-sample)
  u64 filter_mask;        // transforms detection may nominate (bit = filter id)
  bool gpu;               // use the Metal match finder if available
  bool bwt;               // use the BWT+MTF+entropy block mode instead of LZ
  bool mosaic;              // use the Mosaic (lattice-ROLZ + tiled BWT) block mode
  bool oracle;                // use the Oracle (learned-context) block mode
  bool crystal;            // use the Crystal (context-mixing) block mode
  bool flash;              // use the Flash (byte-aligned LZ77, SIMD decode) block mode
  bool laser = false;      // use the Laser (LZMA2-class 5-stream) block mode
  bool grating = false;    // use the Grating (order-0 integer rANS) block mode
  bool context_oracle;     // PAQ-style context cost modeling / evaluation
  u32 crystal_profile;      // static probability profile for Crystal block mode
};

enum class StatsBlockKind : u8 {
  Stored = 0,
  Memset = 1,
  Compressed = 2,
  Bwt = 3,
  Mosaic = 4,
  Oracle = 5,
  Crystal = 6,
  Flash = 7,
  Reserved = 8,
  Laser = 9,
};

struct DecodeParams {
  int threads;  // <= 0 => auto
};

// A seek-table entry: total serialized size of a block and its decoded size.
struct SeekEntryInternal {
  u64 comp_size;
  u64 regen_size;
};

// Build the compressed-block payload (no block header) for src[0, n).
void build_compressed_block(const u8* src, usize n, int level,
                            std::vector<u8>& payload,
                            bool context_oracle = false);

// True when GPU (Metal) match finding is compiled in and a device is present.
bool gpu_match_available();

// Decode a compressed-block payload into dst[0, n).
Status decode_compressed_block(const u8* data, usize len, u8* dst, usize n);

// Whole-frame compression. Appends a complete Spectra8 frame to `out`.
void frame_compress(const u8* src, usize n, const EncodeParams& p,
                    std::vector<u8>& out);

// Whole-frame decompression into dst (capacity dst_cap). On success sets
// `written`.
Status frame_decompress(const u8* src, usize n, u8* dst, usize dst_cap,
                        const DecodeParams& p, usize& written);

// Parse a seek-table trailer if present. Returns false if absent/invalid.
bool parse_seek_table(const u8* src, usize n,
                      std::vector<SeekEntryInternal>& out);

// Decode a single block by index into dst[0, dst_cap). Uses the seek table if
// present, otherwise scans block headers. On success sets `written`.
Status decode_block_at(const u8* src, usize n, usize index, u8* dst,
                       usize dst_cap, usize& written);

// Regenerated size of every Spectra8 block, from a header scan (no decode).
Status list_block_regens(const u8* src, usize n, std::vector<u64>& out);

}  // namespace spectra8::detail

#endif  // SPECTRA8_SRC_CODEC_HPP
