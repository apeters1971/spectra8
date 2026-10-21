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

// spectra8 - a fast, modern C++ LZ + entropy compressor.
//
// Public C++20 API. Native codecs emit the Spectra8 container (ASCII magic
// "LHC4"); optional external backends emit their own native frames (zstd /
// bzip3 / xz). decompress() auto-detects the frame type by magic.
#ifndef SPECTRA8_SPECTRA8_HPP
#define SPECTRA8_SPECTRA8_HPP

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace spectra8 {

// Result status codes. Returned via Result<T>; never thrown across the API.
enum class Status : std::uint8_t {
  Ok = 0,
  BufferTooSmall,    // destination span cannot hold the result
  BadMagic,          // input magic is not LHC4 / zstd / bzip3 / xz / L4EF
  UnsupportedVersion,
  CorruptInput,      // malformed / truncated stream
  ChecksumMismatch,  // frame content checksum did not match
  InvalidArgument,
  InternalError,
};

[[nodiscard]] std::string_view status_message(Status s) noexcept;

// Lightweight result type (C++20: no std::expected yet).
template <class T>
struct Result {
  Status status{Status::Ok};
  T value{};

  [[nodiscard]] constexpr bool ok() const noexcept { return status == Status::Ok; }
  [[nodiscard]] constexpr explicit operator bool() const noexcept { return ok(); }
};

inline constexpr int kMinLevel = 0;
// Levels 10 and 11 enable the CPU optimal (cost-based) parser, which is much
// slower than the greedy levels 1-9 but targets the best achievable ratio:
//   * Level 10 ("near-optimal") runs a single DP pass over a moderate set of
//     match candidates - a middle ground between level 9 and ultra.
//   * Level 11 ("ultra") runs two refinement passes over more candidates for
//     the best ratio, at the highest cost.
inline constexpr int kMaxLevel = 11;
inline constexpr int kNearOptLevel = 10;
inline constexpr int kUltraLevel = 11;
inline constexpr int kDefaultLevel = 6;

// Block-size log2 bounds (16 KiB .. 256 MiB). 0 selects a default that grows
// with the level: 256 KiB at levels 1-2, 1 MiB at 3-8, 2 MiB at 9, 4 MiB at
// 10-11 (a larger match window is one of the strongest ratio levers). Inputs
// no larger than one block are unaffected. Set block_log explicitly to pin a
// size, e.g. 18 for the former across-the-board default.
inline constexpr int kMinBlockLog = 14;
inline constexpr int kMaxBlockLog = 28;
inline constexpr int kDefaultBlockLog = 18;
// BWT mode is far more effective with large blocks (it sorts contexts within a
// block), so when block_log is left unset it picks a default that grows with
// the compression level (1/2/4/8 MiB for L1-2 / L3-5 / L6-8 / L9+). This is the
// mid (level 6-8) default; blocks stay independent, so this never affects
// parallelism or determinism.
inline constexpr int kBwtDefaultBlockLog = 22;

// Compression backend. Beam/Prism emit Spectra8 frames; Zstd/Bzip3/Lzma emit their
// native frames (interoperable with the stock CLI tools). Auto races the
// available codecs at `level` (or each codec's max effort when
// auto_max_level) and keeps a decode-speed-aware winner (see
// auto_min_gain_pct). Optional backends depend on build-time linking — check
// codec_available() before requesting them.
enum class Codec : std::uint8_t {
  Beam = 0,     // LZ77 + recent-offset cache, Huffman/ANS streams; L11 = "Beam
                // ultra" (cost-based optimal parse, same decoder)
  Lz = Beam,    // legacy name
  Prism = 1,    // block sort (BWT) -> MTF -> zero-run -> entropy stage
  Bwt = Prism,  // legacy name
  Zstd = 2,
  Bzip3 = 3,
  Lzma = 4,
  Auto = 5,
  Mosaic = 6,
  Oracle = 7,
  Crystal = 8,  // static-model lattice/BWT coder: best ratio, fast decode
                // (L10-11: context mixing, symmetric, slow)
  Flash = 9,    // ultra-fast byte-aligned LZ (L1: 4-6 GB/s SIMD decode;
                // L5: optional Huffman / compact extras / rep-offset header)
  Laser = 10,   // LZMA2-class binary compressor with decoupled range decoder
  Grating = 11, // order-0 integer rANS: one spectrum, escape for the tail.
                // No levels; 4 states, or 8 once a column has >= 4096 symbols.
};

// True when the library can compress/decompress with `c`.
// Beam/Prism/Mosaic/Oracle/Crystal/Flash/Auto are always available; Zstd/Bzip3/Lzma depend on
// optional link-time dependencies.
[[nodiscard]] bool codec_available(Codec c) noexcept;

// Bitmask over Codec values (bit = 1u << static_cast<unsigned>(codec)) that
// restricts which flavours Codec::Auto races. Codecs missing from the build
// are skipped anyway; a mask that leaves no available candidate makes
// compress() return InvalidArgument.
using CodecMask = std::uint32_t;
[[nodiscard]] constexpr CodecMask codec_bit(Codec c) noexcept {
  return CodecMask{1} << static_cast<unsigned>(c);
}
// Every codec Auto knows how to race (Oracle is a demo and never raced).
inline constexpr CodecMask kAutoCodecsAll =
    codec_bit(Codec::Zstd) | codec_bit(Codec::Beam) | codec_bit(Codec::Crystal) |
    codec_bit(Codec::Mosaic) | codec_bit(Codec::Prism) | codec_bit(Codec::Laser) |
    codec_bit(Codec::Lzma) | codec_bit(Codec::Bzip3);
// Recommended set for ROOT / RNTuple pages: the three decoders that stay
// above ~300 MB/s on every platform measured (zstd ~2 GB/s, Beam ~0.7 GB/s,
// Crystal 0.3-0.6 GB/s). Prism/Lzma/Bzip3 are excluded for decode speed,
// Mosaic because it is within 0.5% of Crystal at the same speed.
inline constexpr CodecMask kAutoCodecsRoot =
    codec_bit(Codec::Zstd) | codec_bit(Codec::Beam) | codec_bit(Codec::Crystal);

// Pre-trained static probability profiles for the Crystal block coder.
// Bypasses per-block model discovery and table transmission (zero header overhead),
// enabling >1.0 GB/s decompression on columnar physics data and text.
enum class Profile : std::uint8_t {
  None = 0,            // Dynamic per-page model discovery and transmission (default)
  FloatKinematics = 1, // IEEE-754 32-bit floats: pT, energy, mass (4-byte stride, 2D lattice)
  PoissonCounts = 2,   // Discrete multiplicities, track counts, jagged array offsets
  IntMonotonic = 3,    // Monotonically increasing event IDs, timestamps, entry counters
  Text = 4,            // Text / ASCII / JSON / XML tokens
  Custom = 5,          // User-supplied / offline cluster-trained profile
};

// Byte-filter ids as stored on the wire (one byte per filtered block; see the
// README table). Used to build a FilterMask that restricts what the filter
// auto-detection may nominate; the frame format is unchanged.
enum class FilterId : std::uint8_t {
  None = 0,
  Shuffle16 = 1,
  Shuffle32 = 2,
  Shuffle64 = 3,
  Delta16 = 4,
  Delta32 = 5,
  Delta64 = 6,
  ZigzagDelta16 = 7,
  ZigzagDelta32 = 8,
  ZigzagDelta64 = 9,
  DeltaShuffle32 = 10,
  ZigzagDeltaShuffle32 = 11,
  BitShuffle32 = 12,
  Dict32 = 13,
  Dict64 = 14,
  DeltaShuffle64 = 15,
  ZigzagShuffle32 = 16,
  ZigzagShuffle64 = 17,
  BitShuffle64 = 18,
  DeltaBitShuffle32 = 19,
  Dict16 = 20,
  For16 = 21,
  For32 = 22,
  For64 = 23,
  Alp32 = 24,
  Alp64 = 25,
  XorShuffle32 = 26,
  XorShuffle64 = 27,
  Delta2Shuffle32 = 28,
  ZigzagDelta2Shuffle32 = 29,
  Delta2Shuffle64 = 30,
  BcjX86 = 31,
  DigitNorm = 32,
  WordNorm = 33,
  Base64 = 34,
  BitRound32 = 35,
  BitRound64 = 36,
  Lag2Shuffle32 = 37,
  Lag2Shuffle64 = 38,
  XorLag2Shuffle32 = 39,
  BcjArm64 = 40,
};
using FilterMask = std::uint64_t;
[[nodiscard]] constexpr FilterMask filter_bit(FilterId f) noexcept {
  return FilterMask{1} << static_cast<unsigned>(f);
}
inline constexpr FilterMask kFiltersAll = ~FilterMask{0};
// Element-width subsets, for callers that know the column type.
inline constexpr FilterMask kFilters16 = filter_bit(FilterId::Shuffle16) |
                                         filter_bit(FilterId::Delta16) |
                                         filter_bit(FilterId::ZigzagDelta16) |
                                         filter_bit(FilterId::Dict16) |
                                         filter_bit(FilterId::For16);
inline constexpr FilterMask kFilters32 =
    filter_bit(FilterId::Shuffle32) | filter_bit(FilterId::Delta32) |
    filter_bit(FilterId::ZigzagDelta32) | filter_bit(FilterId::DeltaShuffle32) |
    filter_bit(FilterId::ZigzagDeltaShuffle32) | filter_bit(FilterId::BitShuffle32) |
    filter_bit(FilterId::Dict32) | filter_bit(FilterId::ZigzagShuffle32) |
    filter_bit(FilterId::DeltaBitShuffle32) | filter_bit(FilterId::For32) |
    filter_bit(FilterId::Alp32) | filter_bit(FilterId::XorShuffle32) |
    filter_bit(FilterId::Delta2Shuffle32) | filter_bit(FilterId::ZigzagDelta2Shuffle32) |
    filter_bit(FilterId::Lag2Shuffle32) | filter_bit(FilterId::XorLag2Shuffle32);
inline constexpr FilterMask kFilters64 =
    filter_bit(FilterId::Shuffle64) | filter_bit(FilterId::Delta64) |
    filter_bit(FilterId::ZigzagDelta64) | filter_bit(FilterId::Dict64) |
    filter_bit(FilterId::DeltaShuffle64) | filter_bit(FilterId::ZigzagShuffle64) |
    filter_bit(FilterId::BitShuffle64) | filter_bit(FilterId::For64) |
    filter_bit(FilterId::Alp64) | filter_bit(FilterId::XorShuffle64) |
    filter_bit(FilterId::Delta2Shuffle64) | filter_bit(FilterId::Lag2Shuffle64);

// Compression options. Blocks are independent, so threading does not change the
// produced bytes or the ratio; output is deterministic for any thread count.
// Spectra8-only knobs (seek table, checksums, gpu; threads for Zstd/Bzip3/Lzma)
// return InvalidArgument when set for those codecs. block_log selects the
// bzip3 block size when codec == Bzip3. filters wrap external codecs in an
// L4EF frame. Auto rejects seek/crc/gpu; it races candidates with threads=1
// each (params.threads sizes the race pool). Example: codec=Auto, level=5,
// auto_min_gain_pct=1 → race all at level 5. With auto_max_level, each
// candidate uses its own maximum effort instead of params.level.
struct CompressParams {
  Codec codec = Codec::Beam;
  int level = kDefaultLevel;
  int block_log = 0;            // 0 => default; otherwise clamped to [14, 24]
  int threads = 1;             // <= 0 => auto (hardware concurrency); Spectra8 /
                               // Auto race pool (externals stay single-thread)
  // Auto only: size improvement (%) required to prefer a *slower* decoder
  // over a faster one, per doubling of decode time. Nominal decode speeds
  // (MB/s): Zstd 2000, Beam 700, Crystal 300 (L10-11: 8), Mosaic 280,
  // Prism 140, Lzma 70, Bzip3 16; candidates are walked fast→slow. A
  // candidate 2x slower than the champion must be gain% smaller, 4x slower
  // 2*gain%, bzip3 against Crystal (~19x) ~4.2*gain%; one no slower than
  // the champion only has to be smaller. 0 = always take the smallest.
  // Oracle is a learned-context demo and is not raced by Auto.
  int auto_min_gain_pct = 1;
  // Auto only: race each codec at its highest effort instead of params.level
  // (Beam/Prism/Mosaic/Crystal → 11, Zstd → ZSTD_maxCLevel(), Lzma → 9, Bzip3 →
  // block_log 28).
  // Slower encode; usually better ratio. Ignored for non-Auto codecs.
  bool auto_max_level = false;
  // Auto only: which codecs to race (see CodecMask). Default: all available.
  // kAutoCodecsRoot = Zstd | Beam | Crystal is the recommended set for
  // RNTuple pages, where decode speed matters as much as ratio.
  CodecMask auto_codecs = kAutoCodecsAll;
  bool seek_table = false;     // emit a seek-table trailer for random access
  bool block_checksums = false;// store a per-block checksum (parallel verify)
  bool content_checksum = false;// emit + verify a whole-frame checksum (off by
                               // default: skipping it makes decode memcpy-bound)
  bool filters = false;        // auto-detect + apply byte transforms (shuffle/
                               // delta/zigzag) that help numeric/columnar data
  bool filter_fallback = true; // with filters: also compress raw and keep the
                               // smaller (safe default). For Codec::Prism this
                               // includes a raw-BWT trial (not only raw LZ), so
                               // text/source cannot regress vs plain BWT.
                               // When false, only the detected filter path is
                               // encoded (faster).
  bool filter_rle = true;      // try post-filter byte-RLE when it shrinks the
                               // intermediate (Beam/Prism only)
  bool filter_dict = true;     // allow dictionary-remap filters (dict16/32/64)
                               // in the auto-detect path (Beam/Prism only)
  // With filters: the transforms detection may nominate (see FilterMask /
  // FilterId). Default: all. Intersects with filter_dict for the dict filters.
  // E.g. kFilters64 for an 8-byte column. A mask with exactly one FilterId
  // bit forces that transform (no entropy reject). Combine with
  // filter_fallback=false to measure that filter even when it loses.
  FilterMask filter_mask = kFiltersAll;
  bool filter_sample_full = false; // when filters: run detect_filter /
                               // detect_dict_width over the whole block
                               // instead of the default 16 KiB mid-sample
                               // (slower detect, may pick better on
                               // non-stationary pages)
  bool gpu = false;            // use the Metal GPU match finder if available
                               // (Beam only; ignored/rejected with Prism/externals)
  bool context_oracle = false; // enable PAQ-style context cost modeling / evaluation
                               // in Beam Ultra optimal parse and deep model search
  Profile profile = Profile::None; // static probability profile for Crystal coder (zero per-block header tax)
  std::uint32_t profile_id = 0;   // profile ID when profile != Profile::None (0 defaults to static_cast<uint32_t>(profile))
};

struct DecompressParams {
  int threads = 1;  // <= 0 => auto (hardware concurrency); Spectra8 only
};

// One entry of a frame's seek table.
struct SeekEntry {
  std::uint64_t compressed_size;  // total serialized bytes of the block
  std::uint64_t regen_size;       // decompressed size of the block
};

// True if the library was built with Metal support and a usable GPU device is
// present. When false, CompressParams::gpu is ignored (CPU path is used).
// Page consumption on a GPU is CPU decompress + H2D: see page_service.hpp.
[[nodiscard]] bool gpu_available() noexcept;

// Maximum number of bytes compress() may write for an input of `src_size`.
// Conservative across every enabled codec (assumes the smallest Spectra8 block).
[[nodiscard]] std::size_t compress_bound(std::size_t src_size) noexcept;

// Tighter bound for a specific codec. Returns 0 if the codec is unavailable.
[[nodiscard]] std::size_t compress_bound(std::size_t src_size,
                                         Codec codec) noexcept;

// One-shot compression. Writes a complete frame into `dst` (Spectra8 for Beam/Prism,
// native zstd/bzip3/xz for those codecs; Auto emits whichever won). Default
// overload uses Codec::Beam. On success, Result::value holds bytes written.
[[nodiscard]] Result<std::size_t> compress(std::span<const std::byte> src,
                                           std::span<std::byte> dst,
                                           int level = kDefaultLevel) noexcept;

// One-shot compression with full options (codec, threading, block size, …).
[[nodiscard]] Result<std::size_t> compress(std::span<const std::byte> src,
                                           std::span<std::byte> dst,
                                           const CompressParams& params) noexcept;

// One-shot decompression. Auto-detects Spectra8 / zstd / bzip3 / xz by magic.
// `dst` must be large enough to hold the decoded content; query the required
// size with get_frame_content_size() when available. On success, Result::value
// holds the number of bytes written.
[[nodiscard]] Result<std::size_t> decompress(std::span<const std::byte> src,
                                             std::span<std::byte> dst) noexcept;

// One-shot decompression with options (e.g. multithreaded Spectra8 block decode).
[[nodiscard]] Result<std::size_t> decompress(
    std::span<const std::byte> src, std::span<std::byte> dst,
    const DecompressParams& params) noexcept;

// Decode a single Spectra8 block by index (random access). Not supported for
// zstd/bzip3 frames. `dst` must hold at least that block's decompressed size.
// Uses the seek table if present, else scans block headers.
[[nodiscard]] Result<std::size_t> decompress_block(
    std::span<const std::byte> src, std::size_t index,
    std::span<std::byte> dst) noexcept;

// Returns the decompressed content size recorded in the frame header, if it is
// present / recoverable, without decoding the payload. Works for Spectra8 (when
// the content-size flag is set), zstd, bzip3, and xz.
[[nodiscard]] std::optional<std::uint64_t> get_frame_content_size(
    std::span<const std::byte> src) noexcept;

// Returns the Spectra8 frame's seek table if one is present (per-block sizes).
[[nodiscard]] std::optional<std::vector<SeekEntry>> get_seek_table(
    std::span<const std::byte> src);

// Library version, e.g. "2.0.0".
[[nodiscard]] std::string_view version() noexcept;

// --- Compression statistics (aggregated across successful compress() calls) ---

enum class StatsFilterKind : std::uint8_t {
  None = 0,
  Shuffle16 = 1,
  Shuffle32 = 2,
  Shuffle64 = 3,
  Delta16 = 4,
  Delta32 = 5,
  Delta64 = 6,
  ZigzagDelta16 = 7,
  ZigzagDelta32 = 8,
  ZigzagDelta64 = 9,
  DeltaShuffle32 = 10,
  ZigzagDeltaShuffle32 = 11,
  BitShuffle32 = 12,
  Dict32 = 13,
  Dict64 = 14,
  DeltaShuffle64 = 15,
  ZigzagShuffle32 = 16,
  ZigzagShuffle64 = 17,
  BitShuffle64 = 18,
  DeltaBitShuffle32 = 19,
  Dict16 = 20,
  For16 = 21,
  For32 = 22,
  For64 = 23,
  Alp32 = 24,
  Alp64 = 25,
  XorShuffle32 = 26,
  XorShuffle64 = 27,
  Delta2Shuffle32 = 28,
  ZigzagDelta2Shuffle32 = 29,
  Delta2Shuffle64 = 30,
  BcjX86 = 31,
  DigitNorm = 32,
  WordNorm = 33,
  Base64 = 34,
  BitRound32 = 35,
  BitRound64 = 36,
  Lag2Shuffle32 = 37,
  Lag2Shuffle64 = 38,
  XorLag2Shuffle32 = 39,
  BcjArm64 = 40,
  Stored = 41,
  Count = 42,
};

// On-disk container for one compressed page.
enum class StatsWirePath : std::uint8_t {
  RawCodec = 0,     // native zstd/bzip3 (no L4EF wrapper)
  L4efFiltered = 1, // L4EF with a byte transform
  L4efStored = 2,   // L4EF identity/stored fallback
  Spectra8Frame = 3,    // native Spectra8 frame (Beam/Prism)
};

// How filters interacted with this page (one count per compress() call).
enum class StatsPageOutcome : std::uint8_t {
  Plain = 0,         // filters disabled
  NoTransform = 1,   // filters on, but nothing kept on the wire
  RawWon = 2,        // filter fallback kept the unfiltered payload
  FilteredWon = 3,   // a byte transform was kept
  StoredFallback = 4 // stored/L4EF identity (typical bzip3 small-page path)
};

struct CompressStats {
  std::uint64_t num_pages = 0;
  std::uint64_t uncompressed_bytes = 0;
  std::uint64_t compressed_bytes = 0;

  // Page outcomes (each page increments exactly one of these when recorded).
  std::uint64_t plain_pages = 0;
  std::uint64_t no_transform_pages = 0;
  std::uint64_t raw_won_pages = 0;
  std::uint64_t filtered_won_pages = 0;
  std::uint64_t stored_fallback_pages = 0;

  // Wire-format mix (each page increments exactly one of these).
  std::uint64_t wire_raw_codec_pages = 0;
  std::uint64_t wire_l4ef_filtered_pages = 0;
  std::uint64_t wire_l4ef_stored_pages = 0;
  std::uint64_t wire_spectra8_pages = 0;

  // Filter chosen by autodetect (one count per page, including raw-won fallback pages).
  std::uint64_t detected_filter_hits[static_cast<std::size_t>(StatsFilterKind::Count)]{};

  // Dominant on-wire filter per page (not per internal Spectra8 block).
  std::uint64_t l4ef_filter_hits[static_cast<std::size_t>(StatsFilterKind::Count)]{};
  std::uint64_t spectra8_filter_hits[static_cast<std::size_t>(StatsFilterKind::Count)]{};
  std::uint64_t l4ef_rle_pages = 0;
  std::uint64_t spectra8_rle_pages = 0;

  // Codec::Auto winner histogram. Each successful Auto compress() increments
  // auto_selections and exactly one auto_codec_hits[Codec] slot (Beam/Prism/Zstd/
  // Bzip3/Lzma/Mosaic/Crystal/Laser). The Auto and Oracle enum indexes are unused (Oracle is
  // not raced).
  std::uint64_t auto_selections = 0;
  std::uint64_t auto_codec_hits[11]{};

  // Crystal blocks compressed with a static profile (CompressParams::profile):
  // crystal_profile_blocks were coded with the profile tables,
  // crystal_profile_escapes were estimated smaller with a per-block model and
  // written as dynamic blocks instead. Counted per Spectra8 block, not per page.
  std::uint64_t crystal_profile_blocks = 0;
  std::uint64_t crystal_profile_escapes = 0;
};

void reset_compress_stats() noexcept;
[[nodiscard]] CompressStats get_compress_stats() noexcept;
[[nodiscard]] std::string format_compress_stats(const CompressStats& stats);

// Auto-only section of the stats report (winner distribution + fractions).
// Empty string when auto_selections == 0.
[[nodiscard]] std::string format_auto_codec_stats(const CompressStats& stats);

// Thread-local default profile for integrations (e.g. ROOT RNTuple ZipSpectra8).
void set_thread_profile(Profile profile) noexcept;
[[nodiscard]] Profile get_thread_profile() noexcept;

// Train a custom static profile from sample columnar data and register it under profile_id (>= 5).
[[nodiscard]] bool train_and_register_profile(
    std::uint32_t profile_id, std::span<const std::byte> sample, int stride, int level = 5) noexcept;

// Trains a static Crystal profile model from sample columnar data and returns the serialized model bytes.
[[nodiscard]] std::vector<std::uint8_t> train_profile_model(
    std::span<const std::byte> sample, int stride, int level = 5);

// Registers a serialized profile model under a specific profile_id (>= 5).
[[nodiscard]] bool register_profile_model(
    std::uint32_t profile_id, std::span<const std::byte> model_bytes) noexcept;

// Exports an already-registered or builtin profile model to serialized bytes.
[[nodiscard]] std::vector<std::uint8_t> export_profile_model(std::uint32_t profile_id);

}  // namespace spectra8

#endif  // SPECTRA8_SPECTRA8_HPP
