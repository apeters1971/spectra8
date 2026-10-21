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

// On-disk container constants for the Spectra8 frame format.
//
// Layout (zstd-inspired):
//
//   magic        : 4 bytes  "LHC4" (0x4C 0x48 0x43 0x34)
//   version      : u8       (= kVersion)
//   flags        : u8       (bit0 content-size-present, bit1 has-checksum)
//   block_size_log: u8      (log2 of the uncompressed block size, e.g. 18)
//   content_size : varint   (present iff flags bit0 set)
//   blocks...
//   checksum     : u64 LE   (present iff flags bit1 set)
//
// Each block:
//   block_type   : u8       (low 3 bits type, bit7 = last block)
//   regen_size   : varint   (decompressed bytes produced by this block)
//   ... type-specific payload (see BlockType) ...
#ifndef SPECTRA8_SRC_FORMAT_HPP
#define SPECTRA8_SRC_FORMAT_HPP

#include "util/common.hpp"

namespace spectra8::detail {

inline constexpr u8 kMagic[4] = {'L', 'H', 'C', '4'};
inline constexpr usize kMagicLen = 4;
inline constexpr u8 kVersion = 2;  // 2: Flash token layout (adv:3|wide:1|ml:4, 1/2-byte
                                   //    offset), compact Huffman/rANS table headers,
                                   //    Crystal static lane modes (coded/constant/raw)

inline constexpr u8 kFlagContentSize = 0x01;
inline constexpr u8 kFlagChecksum = 0x02;
inline constexpr u8 kFlagSeekTable = 0x04;       // seek-table trailer present
inline constexpr u8 kFlagBlockChecksums = 0x08;  // per-block checksum present
inline constexpr u8 kFlagFilters = 0x10;         // compressed/BWT blocks may carry
                                                 // a filter byte (transform pre-
                                                 // codec; bit7 = byte-RLE layer).
                                                 // Byte-RLE and the dictionary
                                                 // filters (non-size-preserving)
                                                 // append a varint with the
                                                 // intermediate stream length.
                                                 // Set when filters are enabled or
                                                 // any block uses byte-RLE/filter.

// Uncompressed block size. 256 KiB keeps blocks independent and cache-friendly.
inline constexpr u8 kBlockSizeLog = 18;
inline constexpr usize kBlockSize = usize{1} << kBlockSizeLog;

// Allowed range for a configurable block size (bounds match-finder memory).
inline constexpr int kMinBlockSizeLog = 14;  // 16 KiB
inline constexpr int kMaxBlockSizeLog = 28;  // 256 MiB

// Trailer magic for the seek table footer (last 8 bytes of the frame are
// num_blocks:u32 then this magic:u32).
inline constexpr u8 kSeekMagic[4] = {'L', 'S', 'E', 'K'};

enum class BlockType : u8 {
  Stored = 0,      // regen_size raw bytes follow
  Memset = 1,      // a single byte follows, repeated regen_size times
  Compressed = 2,  // varint(payload_size) then the LZ+entropy payload
  Bwt = 3,         // varint(payload_size) then the BWT+MTF+entropy payload:
                   //   varint(primary_index), u8 num_chains, then (if
                   //   num_chains > 1) num_chains varint anchors (starting rows
                   //   for the parallel inverse), then u8 mode and a symbol
                   //   block.
                   //   mode 0: symbols = the MTF stream coded directly.
                   //   mode 1: RLE0 - symbols (zero-runs collapsed to a single
                   //     0) + an encode_array run-length stream + a varint-
                   //     length run-extra blob.
                   //   mode 2: bzip2 RUNA/RUNB zero-run coding on the MTF
                   //     stream, then one encode_array symbol block.
                   //   mode 3: symbols = the raw BWT L-column (no MTF).
                   //   A symbol block is: u8 seg_mode; seg_mode 0 => one
                   //     encode_array; seg_mode 1 => varint(num_segs) + that
                   //     many encode_array segments (per-segment entropy tables);
                   //     seg_mode 2 => varint(n) varint(len) Spectra8 LZ payload;
                   //     seg_mode 3 => varint(n) varint(len) zstd L5 payload.
  Mosaic = 4,        // varint(payload_size) then a Mosaic payload (tiled ROLZ /
                   // mini-BWT + static entropy). See mosaic.hpp.
  Oracle = 5,          // varint(payload_size) then an Oracle payload (learned
                   // context map + static rANS). See oracle.hpp.
  Crystal = 6,      // varint(payload_size) then a Crystal payload (lattice-
                   // aware context mixing, binary arithmetic coded). See
                   // crystal.hpp.
  Flash = 7,        // varint(payload_size) then a Flash payload (byte-aligned
                   // LZ tokens, direct SIMD decode). See flash.hpp.
  Laser = 8,        // varint(payload_size) then a Laser payload (LZMA2-class
                   // 5-stream decoupled range decoder). See laser.hpp.
  Grating = 9       // varint(payload_size) then a Grating payload (order-0
                   // integer rANS, 4- or 8-way). See grating.hpp.
};

// Low bits of the block-type byte (bits 0..3: types 0..15).
inline constexpr u8 kBlockTypeMask = 0x0F;

inline constexpr u8 kBlockLastFlag = 0x80;

// Laser default block size log2 (21 = 2 MiB default window, or 22 = 4 MiB for L9+).
inline constexpr int kLaserDefaultBlockLog = 21;

// Number of recent offsets cached for repeat-offset coding.
inline constexpr int kNumRecentOffsets = 7;

// Minimum match length for newly coded (non-repeat) offsets.
inline constexpr int kMinMatch = 4;
// Minimum match length accepted for a repeat offset.
inline constexpr int kMinRepMatch = 2;

// Literal transform applied within a compressed block.
enum class LiteralMode : u8 {
  Raw = 0,  // literals stored verbatim
  Sub = 1   // literals stored as delta vs byte at the current recent offset
};

// Compressed-block mode byte: bit 0 is the LiteralMode, the flags below
// select token-context conditioning. The three token streams are decoded in
// the order literal length -> match length -> offset code, so each may be
// conditioned on the ones before it: the match-length tokens split into
// kLlBuckets arrays by literal-length bucket, the offset codes into
// kOcContexts arrays by (match-length bucket, literal run empty). Short
// matches sit at near offsets and matches following an empty literal run are
// mostly repeats, so the conditional token entropies are 3-30% below the
// plain ones on most inputs; the decoder pays one table lookup and one
// cursor increment per token.
inline constexpr u8 kBlockModeLiteralMask = 0x01;
inline constexpr u8 kBlockFlagMlCond = 0x02;
inline constexpr u8 kBlockFlagOcCond = 0x04;
// Bits 3-4: number L (0-3) of low offset bits that are entropy coded instead
// of stored raw. A new offset with k = bit_width - 1 >= L raw bits stores
// its top k - L bits raw and its low L bits as one token in an extra array
// (after the offset-code array). On record-structured data offsets are
// stride multiples, so those bits are nearly constant: 8-20% of an
// unfiltered lattice block was raw zero bits. Offsets with k < L are
// unchanged. Old frames have L = 0.
inline constexpr u8 kBlockOffLowShift = 3;
inline constexpr u8 kBlockOffLowMask = 0x18;
inline constexpr int kMaxOffLowBits = 3;
inline constexpr int kLlBuckets = 4;
inline constexpr int kMlBuckets = 4;
inline constexpr int kOcContexts = 2 * kMlBuckets;

// Bucket of a literal-length token (min(len, 255)): 0, 1-2, 3-8, 9+.
inline constexpr int ll_bucket(u32 tok) noexcept {
  return tok == 0 ? 0 : tok <= 2 ? 1 : tok <= 8 ? 2 : 3;
}
// Bucket of a match-length token (len - kMinRepMatch): lengths 2-3, 4-5,
// 6-8, 9+.
inline constexpr int ml_bucket(u32 tok) noexcept {
  return tok <= 1 ? 0 : tok <= 3 ? 1 : tok <= 6 ? 2 : 3;
}
inline constexpr int oc_context(u32 ml_tok, u32 ll_tok) noexcept {
  return 2 * ml_bucket(ml_tok) + (ll_tok == 0 ? 1 : 0);
}

}  // namespace spectra8::detail

#endif  // SPECTRA8_SRC_FORMAT_HPP
