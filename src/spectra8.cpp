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

#include "spectra8/spectra8.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#include "spectra8/spectra8_c.h"
#include "codec.hpp"
#include "util/common.hpp"
#include "crystal/crystal.hpp"
#include "ext/bzip3.hpp"
#include "ext/filter_frame.hpp"
#include "ext/lzma.hpp"
#include "util/transform.hpp"

// The public FilterId mirrors the on-wire detail::Filter ids one to one.
static_assert(static_cast<int>(spectra8::FilterId::None) == static_cast<int>(spectra8::detail::Filter::None));
static_assert(static_cast<int>(spectra8::FilterId::Dict32) == static_cast<int>(spectra8::detail::Filter::Dict32));
static_assert(static_cast<int>(spectra8::FilterId::Dict64) == static_cast<int>(spectra8::detail::Filter::Dict64));
static_assert(static_cast<int>(spectra8::FilterId::Dict16) ==
              static_cast<int>(spectra8::detail::Filter::Dict16));
static_assert(static_cast<int>(spectra8::FilterId::BcjArm64) ==
              static_cast<int>(spectra8::detail::Filter::BcjArm64));
static_assert(static_cast<int>(spectra8::FilterId::BcjArm64) ==
              static_cast<int>(spectra8::detail::Filter::kMaxValue));
static_assert(static_cast<int>(spectra8::StatsFilterKind::Stored) ==
              static_cast<int>(spectra8::detail::Filter::kMaxValue) + 1);
static_assert(spectra8::kFiltersAll == ~spectra8::FilterMask{0});
#include "ext/zstd.hpp"
#include "util/format.hpp"
#include "util/parallel.hpp"
#include "util/stats.hpp"

namespace spectra8 {

using detail::u8;
using detail::u64;
using detail::usize;

std::string_view status_message(Status s) noexcept {
  switch (s) {
    case Status::Ok: return "ok";
    case Status::BufferTooSmall: return "destination buffer too small";
    case Status::BadMagic: return "unrecognized frame magic";
    case Status::UnsupportedVersion: return "unsupported format version";
    case Status::CorruptInput: return "corrupt or truncated input";
    case Status::ChecksumMismatch: return "content checksum mismatch";
    case Status::InvalidArgument: return "invalid argument";
    case Status::InternalError: return "internal error";
  }
  return "unknown status";
}

std::string_view version() noexcept { return "2.0.0"; }

bool gpu_available() noexcept { return detail::gpu_match_available(); }

bool codec_available(Codec c) noexcept {
  switch (c) {
    case Codec::Beam:
    case Codec::Prism:
    case Codec::Mosaic:
    case Codec::Oracle:
    case Codec::Crystal:
    case Codec::Flash:
    case Codec::Laser:
    case Codec::Grating:
    case Codec::Auto:
      return true;
    case Codec::Zstd:
      return detail::zstd_available();
    case Codec::Bzip3:
      return detail::bzip3_available();
    case Codec::Lzma:
      return detail::lzma_available();
  }
  return false;
}

namespace {

std::size_t spectra8_compress_bound(std::size_t src_size) noexcept {
  // Safe for any block size: assume the smallest allowed block.
  const std::size_t min_block = std::size_t{1} << detail::kMinBlockSizeLog;
  const std::size_t num_blocks = src_size / min_block + 1;
  const std::size_t block_overhead =
      num_blocks * (16 + 4 + 8);  // hdr + bcsum + seek entry
  const std::size_t frame_overhead =
      detail::kMagicLen + 3 + 10 + 8 + 8;  // + content chk + footer
  return src_size + block_overhead + frame_overhead + 64;
}

bool looks_like_spectra8(std::span<const std::byte> src) noexcept {
  return src.size() >= detail::kMagicLen &&
         std::memcmp(src.data(), detail::kMagic, detail::kMagicLen) == 0;
}

// Spectra8-only knobs that must not be set for external codecs (filters use L4EF).
bool has_spectra8_only_options(const CompressParams& p) noexcept {
  return p.seek_table || p.block_checksums || p.content_checksum || p.gpu ||
         p.threads != 1;
}

thread_local Profile g_thread_profile = Profile::None;

}  // namespace

std::size_t compress_bound(std::size_t src_size) noexcept {
  std::size_t bound = spectra8_compress_bound(src_size);
  if (detail::zstd_available()) {
    const std::size_t zb = detail::zstd_compress_bound(src_size);
    if (zb > bound) bound = zb;
  }
  if (detail::bzip3_available()) {
    const std::size_t bb = detail::bzip3_compress_bound(src_size);
    if (bb > bound) bound = bb;
  }
  if (detail::lzma_available()) {
    const std::size_t lb = detail::lzma_compress_bound(src_size);
    if (lb > bound) bound = lb;
  }
  return bound;
}

std::size_t compress_bound(std::size_t src_size, Codec codec) noexcept {
  if (!codec_available(codec)) return 0;
  switch (codec) {
    case Codec::Beam:
    case Codec::Prism:
    case Codec::Mosaic:
    case Codec::Oracle:
    case Codec::Crystal:
    case Codec::Flash:
    case Codec::Laser:
    case Codec::Grating:
      return spectra8_compress_bound(src_size);
    case Codec::Zstd:
      return std::max(detail::zstd_compress_bound(src_size),
                      detail::l4ef_compress_bound(src_size, Codec::Zstd));
    case Codec::Bzip3:
      return std::max(detail::bzip3_compress_bound(src_size),
                      detail::l4ef_compress_bound(src_size, Codec::Bzip3));
    case Codec::Lzma:
      return std::max(detail::lzma_compress_bound(src_size),
                      detail::l4ef_compress_bound(src_size, Codec::Lzma));
    case Codec::Auto:
      return compress_bound(src_size);
  }
  return 0;
}

namespace {

// Default BWT block size by level: larger blocks give the BWT more context
// (better ratio) but the inverse-BWT LF walk is random access into an
// O(blocksize) nav table, so once the block exceeds cache the decode throughput
// falls off a cliff (measured: ~240 MB/s at 2 MiB vs ~140 MB/s at 4 MiB, with
// 8 MiB giving no better ratio than 2 MiB on text). So we cap at 2 MiB: it
// keeps the nav walk closer to cache and matches the larger-block ratio, while
// low levels use smaller blocks for more thread parallelism. (bench --bwt-chains
// prints the block-size sweep this is based on.)
int bwt_block_log(int level) {
  if (level <= 2) return 20;  // 1 MiB
  if (level <= 8) return 21;  // 2 MiB
  if (level <= 10) return 22; // 4 MiB (near-optimal)
  return 27;                  // 128 MiB (level 11 / ultra)
}

// Default LZ block size by level. A larger match window is one of the best
// ratio levers on inputs that exceed the block size (dickens L9: 2.75 at
// 256 KiB vs 3.09 at 2 MiB). Levels 3-6 use 1 MiB: with the offset-aware
// parser the bigger window no longer costs ratio on rep-heavy data, the
// row-based finder keeps encode speed level (measured slightly faster than
// 256 KiB blocks), and decode is faster too (fewer blocks, fewer bytes) -
// -2.3% total size on the test corpus at L5. Levels 1-2 gain nothing from a
// bigger window (hash-only finder) and keep 256 KiB blocks for parallelism.
// Blocks > 1 MiB use the binary-tree match finder, which keeps the bigger
// window affordable at the slow levels. Inputs no bigger than the block size
// are unaffected by this choice: they form a single block either way, and
// the parser is tuned by the actual content size, not the nominal block
// size. An explicit block_log always wins.
int lz_block_log(int level) {
  if (level <= 2) return kDefaultBlockLog;  // 256 KiB
  if (level <= 8) return 20;                // 1 MiB
  if (level == 9) return 21;                // 2 MiB
  return 22;                                // 4 MiB (optimal-parser levels)
}

// Crystal: 2 MiB blocks for the static lattice coder (L1-8), 4 MiB at L9.
// The lattice amortises its model header (and the fixed per-block model
// search) over the block, and the BWT text path sorts longer contexts:
// against 1 MiB blocks, 2 MiB gains 3-5% on text (dickens L5 3.72 -> 3.88,
// webster 4.80 -> 5.03) and 1-4% on record data (sao, osdb) at the same
// decode speed; 4 MiB adds another 3-4% on text but its inverse BWT walk
// leaves the cache (decode -10..-20%), so only the top static level takes
// it. The context-mixing tiers (L10-11) keep 1 MiB: a long history while
// the per-thread table arena stays bounded.
int crystal_block_log(int level) {
  if (level <= 8) return 21;
  if (level == 9) return 22;
  return 20;
}

int laser_block_log(int level) {
  if (level >= 9) return 22;  // 4 MiB
  return detail::kLaserDefaultBlockLog;  // 21 -> 2 MiB
}

detail::EncodeParams make_encode_params(const CompressParams& p) {
  int level = p.level;
  if (level < kMinLevel) level = kMinLevel;
  if (level > kMaxLevel) level = kMaxLevel;

  std::uint32_t crystal_profile = static_cast<std::uint32_t>(
      p.profile != Profile::None
          ? (p.profile_id ? p.profile_id : static_cast<std::uint32_t>(p.profile))
          : static_cast<std::uint32_t>(g_thread_profile));

  const bool flash = (p.codec == Codec::Flash) || (p.level == 0);
  const bool laser = !flash && (p.codec == Codec::Laser);
  const bool grating = !flash && !laser && (p.codec == Codec::Grating);
  const bool bwt = !flash && !laser && !grating && (p.codec == Codec::Prism);
  const bool mosaic = !flash && !laser && !grating && (p.codec == Codec::Mosaic);
  const bool oracle = !flash && !laser && !grating && (p.codec == Codec::Oracle);
  const bool crystal = (!flash && !laser && !grating && (p.codec == Codec::Crystal)) || (crystal_profile != 0);
  int blog = (p.block_log == 0)
                 ? (flash ? 18
                        : laser ? laser_block_log(level)
                        : grating ? 20
                        : bwt ? bwt_block_log(level)
                        : crystal ? crystal_block_log(level) : lz_block_log(level))
                 : p.block_log;
  if (blog < kMinBlockLog) blog = kMinBlockLog;
  if (blog > kMaxBlockLog) blog = kMaxBlockLog;

  const bool context_oracle = p.context_oracle || (std::getenv("SPECTRA8_CONTEXT_ORACLE") != nullptr);
  return detail::EncodeParams{level,
                              std::size_t{1} << blog,
                              p.threads,
                              p.seek_table,
                              p.block_checksums,
                              p.content_checksum,
                              p.filters,
                              p.filter_fallback,
                              p.filter_rle,
                              p.filter_dict,
                              p.filter_sample_full,
                              p.filter_mask,
                              p.gpu && !bwt && !mosaic && !oracle && !crystal && !flash && !laser &&
                                  !grating,
                              bwt,
                              mosaic,
                              oracle,
                              crystal,
                              flash,
                              laser,
                              grating,
                              context_oracle,
                              crystal_profile};
}

Result<std::size_t> compress_spectra8(std::span<const std::byte> src,
                                  std::span<std::byte> dst,
                                  const detail::EncodeParams& ep) {
  std::vector<u8> out;
  out.reserve(spectra8_compress_bound(src.size()));
  detail::frame_compress(reinterpret_cast<const u8*>(src.data()), src.size(),
                         ep, out);
  if (out.size() > dst.size()) return {Status::BufferTooSmall, 0};
  std::memcpy(dst.data(), out.data(), out.size());
  detail::stats_set_last_observe(detail::observe_spectra8_frame(
      std::span<const std::byte>(reinterpret_cast<const std::byte*>(out.data()), out.size()),
      ep.filters));
  return {Status::Ok, out.size()};
}

}  // namespace

Result<std::size_t> compress(std::span<const std::byte> src,
                             std::span<std::byte> dst, int level) noexcept {
  CompressParams p;
  p.level = level;
  return compress(src, dst, p);
}

namespace {

void set_plain_observe() noexcept {
  detail::CompressObserveInfo info;
  info.wire_path = StatsWirePath::RawCodec;
  info.outcome = StatsPageOutcome::Plain;
  detail::stats_set_last_observe(info);
}

detail::CompressObserveInfo default_observe_for_output(
    std::span<const std::byte> out, const CompressParams& params) noexcept {
  if (looks_like_spectra8(out))
    return detail::observe_spectra8_frame(out, params.filters);
  if (detail::looks_like_l4ef(out)) return detail::observe_l4ef_frame(out);
  detail::CompressObserveInfo info;
  info.wire_path = StatsWirePath::RawCodec;
  info.outcome = params.filters ? StatsPageOutcome::NoTransform : StatsPageOutcome::Plain;
  return info;
}

// Trial one concrete codec for Auto (no stats aggregation; caller observes once).
Result<std::size_t> auto_trial_compress(std::span<const std::byte> src,
                                        std::span<std::byte> dst, Codec codec,
                                        const CompressParams& base) noexcept {
  CompressParams p = base;
  p.codec = codec;
  p.threads = 1;
  p.seek_table = false;
  p.block_checksums = false;
  p.content_checksum = false;
  p.gpu = false;
  if (codec == Codec::Zstd || codec == Codec::Lzma) p.block_log = 0;

  if (base.auto_max_level) {
    switch (codec) {
      case Codec::Beam:
      case Codec::Prism:
      case Codec::Mosaic:
      case Codec::Oracle:
      case Codec::Crystal:
      case Codec::Laser:
        p.level = kMaxLevel;
        break;
      case Codec::Zstd:
        p.level = detail::zstd_max_level();
        break;
      case Codec::Lzma:
        p.level = 9;
        break;
      case Codec::Bzip3:
        // Highest legal bzip3 block_log (clamped in bzip3_compress).
        p.block_log = 28;
        break;
      default:
        break;
    }
  }

  switch (codec) {
    case Codec::Beam:
    case Codec::Prism:
    case Codec::Mosaic:
    case Codec::Oracle:
    case Codec::Crystal:
    case Codec::Laser:
      return compress_spectra8(src, dst, make_encode_params(p));
    case Codec::Zstd:
      if (p.filters) return detail::compress_filtered_frame(src, dst, Codec::Zstd, p);
      return detail::zstd_compress(src, dst, p.level);
    case Codec::Bzip3:
      if (p.filters)
        return detail::compress_filtered_frame(src, dst, Codec::Bzip3, p);
      return detail::bzip3_compress(src, dst, p.block_log);
    case Codec::Lzma:
      if (p.filters) return detail::compress_filtered_frame(src, dst, Codec::Lzma, p);
      return detail::lzma_compress(src, dst, p.level);
    default:
      return {Status::InvalidArgument, 0};
  }
}

// True when a trial frame roundtrips to `src` (defense against backends that
// report success with a tiny/corrupt payload — e.g. some bzip3 edge cases).
bool auto_trial_roundtrips(std::span<const std::byte> src,
                           std::span<const std::byte> frame) noexcept {
  if (frame.empty()) return src.empty();
  if (!(looks_like_spectra8(frame) || detail::looks_like_l4ef(frame) ||
        detail::looks_like_zstd(frame) || detail::looks_like_bzip3(frame) ||
        detail::looks_like_xz(frame)))
    return false;
  std::vector<std::byte> out(src.size());
  auto d = decompress(frame, out);
  return d.ok() && d.value == src.size() &&
         std::memcmp(out.data(), src.data(), src.size()) == 0;
}

// Nominal single-thread decode speed (MB/s) of each Auto candidate, used only
// to scale the min-gain bar: it is the ratio between two candidates that
// matters, not the absolute figures (measured on an EPYC 7316 / Apple M-class,
// filtered numeric records; text is slower for all in about the same
// proportion). Crystal L10-11 is bitwise context mixing, the slowest decoder.
double auto_nominal_decode_mbps(Codec c, bool crystal_cm) noexcept {
  switch (c) {
    case Codec::Zstd: return 2000.0;
    case Codec::Beam: return 700.0;
    case Codec::Crystal: return crystal_cm ? 8.0 : 300.0;
    case Codec::Mosaic: return 280.0;
    case Codec::Laser: return 180.0;
    case Codec::Prism: return 140.0;
    case Codec::Lzma: return 70.0;
    case Codec::Bzip3: return 16.0;
    default: return 300.0;
  }
}

// Size win (%) a candidate must have over the current champion to replace it:
// auto_min_gain_pct per doubling of decode time. A 2x slower decoder needs
// the plain gain, 4x needs twice, bzip3 against Crystal (~19x) needs ~4.2x,
// and a candidate no slower than the champion needs only to be smaller.
double auto_required_gain_pct(int gain_pct, double champ_mbps, double cand_mbps) noexcept {
  if (cand_mbps >= champ_mbps) return 0.0;
  return static_cast<double>(gain_pct) * std::log2(champ_mbps / cand_mbps);
}

// Race Zstd/Beam/Mosaic/Laser/Prism/Lzma/Bzip3/Crystal at params.level (or each codec's max
// effort when auto_max_level). Walk candidates fastest→slowest decode and only
// step to a slower codec when its size win clears the speed-scaled gain bar.
Result<std::size_t> compress_auto(std::span<const std::byte> src,
                                  std::span<std::byte> dst,
                                  const CompressParams& params) noexcept {
  if (params.seek_table || params.block_checksums || params.content_checksum ||
      params.gpu)
    return {Status::InvalidArgument, 0};

  int gain = params.auto_min_gain_pct;
  if (gain < 0) gain = 0;
  if (gain > 90) gain = 90;

  // Fast→slow decode order. Crystal L1-9 is the static lattice coder (table
  // lookups, ~500 MB/s); L10-11 is context mixing, the slowest decoder.
  static constexpr Codec kOrderFast[] = {Codec::Zstd, Codec::Beam,   Codec::Crystal,
                                         Codec::Mosaic, Codec::Laser, Codec::Prism,  Codec::Lzma,
                                         Codec::Bzip3};
  static constexpr Codec kOrderCm[] = {Codec::Zstd, Codec::Beam,   Codec::Mosaic,
                                       Codec::Laser, Codec::Prism,  Codec::Lzma, Codec::Bzip3,
                                       Codec::Crystal};
  const bool crystal_cm = params.auto_max_level || params.level > 9;
  const std::span<const Codec> kOrder =
      crystal_cm ? std::span<const Codec>(kOrderCm) : std::span<const Codec>(kOrderFast);

  // Executables (ELF / Mach-O / PE header, or an x86 / AArch64 instruction
  // stream, see detail::detect_exec_arch) are a race between zstd, laser, and lzma.
  // Machine code's redundancy is short-range and position-dependent; the record
  // coders have no model for it and finish 10-25 % behind lzma/laser at a fraction
  // of zstd's encode speed, bzip3 behind on both axes. Racing only
  // zstd, laser, and lzma keeps the result and brings the encode down to fast binaries.
  // The gain rule below still decides between them, and excluding lzma/laser from
  // auto_codecs disables the shortcut.
  CodecMask allowed = params.auto_codecs;
  if ((allowed & (codec_bit(Codec::Lzma) | codec_bit(Codec::Laser))) &&
      detail::detect_exec_arch(reinterpret_cast<const u8*>(src.data()), src.size()) !=
          detail::ExecArch::None) {
    allowed &= codec_bit(Codec::Zstd) | codec_bit(Codec::Laser) | codec_bit(Codec::Lzma);
  }
  struct Trial {
    Codec codec = Codec::Beam;
    std::vector<std::byte> buf;
    Result<std::size_t> result{Status::InternalError, 0};
    bool verified = false;
  };
  std::vector<Trial> trials;
  trials.reserve(5);
  for (Codec c : kOrder) {
    if (!codec_available(c)) continue;
    if (!(allowed & codec_bit(c))) continue;
    Trial t;
    t.codec = c;
    t.buf.resize(compress_bound(src.size(), c));
    trials.push_back(std::move(t));
  }
  if (trials.empty()) return {Status::InvalidArgument, 0};

  const int race_threads = detail::resolve_threads(params.threads);
  detail::parallel_for(trials.size(), race_threads, [&](usize i) {
    trials[i].result =
        auto_trial_compress(src, trials[i].buf, trials[i].codec, params);
    if (trials[i].result.ok()) {
      const auto frame = std::span<const std::byte>(
          trials[i].buf.data(), trials[i].result.value);
      trials[i].verified = auto_trial_roundtrips(src, frame);
      if (!trials[i].verified)
        trials[i].result = {Status::InternalError, 0};
    }
  });

  // trials[] is already in decode-speed order. Seed with the first success,
  // then accept a slower codec only if it clears the min-gain bar.
  int best = -1;
  for (int i = 0; i < static_cast<int>(trials.size()); ++i) {
    if (!trials[static_cast<usize>(i)].result.ok()) continue;
    if (best < 0) {
      best = i;
      continue;
    }
    const std::size_t cur = trials[static_cast<usize>(best)].result.value;
    const std::size_t cand = trials[static_cast<usize>(i)].result.value;
    const double need = auto_required_gain_pct(
        gain, auto_nominal_decode_mbps(trials[static_cast<usize>(best)].codec, crystal_cm),
        auto_nominal_decode_mbps(trials[static_cast<usize>(i)].codec, crystal_cm));
    // Need cand <= cur * (100 - need) / 100; with need == 0, strictly smaller
    // (a tie keeps the faster decoder).
    if (need <= 0.0) {
      if (cand < cur) best = i;
    } else if (static_cast<double>(cand) * 100.0 <=
               static_cast<double>(cur) * (100.0 - std::min(need, 99.0))) {
      best = i;
    }
  }
  if (best < 0) return {Status::InternalError, 0};

  auto& w = trials[static_cast<usize>(best)];
  if (w.result.value > dst.size()) return {Status::BufferTooSmall, 0};
  std::memcpy(dst.data(), w.buf.data(), w.result.value);
  const auto out = std::span<const std::byte>(dst.data(), w.result.value);
  auto info = default_observe_for_output(out, params);
  info.has_auto_winner = true;
  info.auto_winner = w.codec;
  detail::stats_set_last_observe(info);
  return {Status::Ok, w.result.value};
}

}  // namespace

Result<std::size_t> compress(std::span<const std::byte> src,
                             std::span<std::byte> dst,
                             const CompressParams& params) noexcept {
  if (!codec_available(params.codec)) return {Status::InvalidArgument, 0};

  detail::stats_reset_last_observe();

  Result<std::size_t> result{Status::InvalidArgument, 0};
  switch (params.codec) {
    case Codec::Beam:
    case Codec::Prism:
    case Codec::Mosaic:
    case Codec::Oracle:
    case Codec::Crystal:
    case Codec::Flash:
    case Codec::Laser:
    case Codec::Grating:
      result = compress_spectra8(src, dst, make_encode_params(params));
      break;
    case Codec::Zstd:
      if (has_spectra8_only_options(params) || params.block_log != 0)
        return {Status::InvalidArgument, 0};
      if (params.filters)
        result = detail::compress_filtered_frame(src, dst, Codec::Zstd, params);
      else {
        result = detail::zstd_compress(src, dst, params.level);
        if (result.ok()) set_plain_observe();
      }
      break;
    case Codec::Bzip3:
      if (params.seek_table || params.block_checksums || params.content_checksum ||
          params.gpu || params.threads != 1)
        return {Status::InvalidArgument, 0};
      if (params.filters)
        result = detail::compress_filtered_frame(src, dst, Codec::Bzip3, params);
      else {
        result = detail::bzip3_compress(src, dst, params.block_log);
        if (result.ok()) {
          const auto out = std::span<const std::byte>(dst.data(), result.value);
          if (detail::looks_like_l4ef(out)) {
            detail::CompressObserveInfo info;
            info.wire_path = StatsWirePath::L4efStored;
            info.outcome = StatsPageOutcome::StoredFallback;
            info.page_filter = StatsFilterKind::Stored;
            detail::stats_set_last_observe(info);
          } else {
            set_plain_observe();
          }
        }
      }
      break;
    case Codec::Lzma:
      if (has_spectra8_only_options(params) || params.block_log != 0)
        return {Status::InvalidArgument, 0};
      if (params.filters)
        result = detail::compress_filtered_frame(src, dst, Codec::Lzma, params);
      else {
        result = detail::lzma_compress(src, dst, params.level);
        if (result.ok()) set_plain_observe();
      }
      break;
    case Codec::Auto:
      result = compress_auto(src, dst, params);
      break;
    default:
      return {Status::InvalidArgument, 0};
  }

  if (result.ok()) {
    const auto out = std::span<const std::byte>(dst.data(), result.value);
    auto info = detail::stats_take_last_observe();
    if (!info.explicit_set) info = default_observe_for_output(out, params);
    detail::stats_observe_compress(src.size(), out, info);
  }
  return result;
}

Result<std::size_t> decompress(std::span<const std::byte> src,
                               std::span<std::byte> dst) noexcept {
  return decompress(src, dst, DecompressParams{});
}

Result<std::size_t> decompress(std::span<const std::byte> src,
                               std::span<std::byte> dst,
                               const DecompressParams& params) noexcept {
  if (looks_like_spectra8(src)) {
    usize written = 0;
    detail::DecodeParams dp{params.threads};
    Status st = detail::frame_decompress(
        reinterpret_cast<const u8*>(src.data()), src.size(),
        reinterpret_cast<u8*>(dst.data()), dst.size(), dp, written);
    if (st != Status::Ok) return {st, 0};
    return {Status::Ok, written};
  }
  if (detail::looks_like_l4ef(src)) {
    return detail::decompress_l4ef_frame(src, dst);
  }
  if (detail::looks_like_zstd(src)) {
    if (!detail::zstd_available()) return {Status::UnsupportedVersion, 0};
    return detail::zstd_decompress(src, dst);
  }
  if (detail::looks_like_bzip3(src)) {
    if (!detail::bzip3_available()) return {Status::UnsupportedVersion, 0};
    return detail::bzip3_decompress(src, dst);
  }
  if (detail::looks_like_xz(src)) {
    if (!detail::lzma_available()) return {Status::UnsupportedVersion, 0};
    return detail::lzma_decompress(src, dst);
  }
  return {Status::BadMagic, 0};
}

Result<std::size_t> decompress_block(std::span<const std::byte> src,
                                     std::size_t index,
                                     std::span<std::byte> dst) noexcept {
  if (!looks_like_spectra8(src)) return {Status::InvalidArgument, 0};
  usize written = 0;
  Status st = detail::decode_block_at(
      reinterpret_cast<const u8*>(src.data()), src.size(), index,
      reinterpret_cast<u8*>(dst.data()), dst.size(), written);
  if (st != Status::Ok) return {st, 0};
  return {Status::Ok, written};
}

std::optional<std::vector<SeekEntry>> get_seek_table(
    std::span<const std::byte> src) {
  if (!looks_like_spectra8(src)) return std::nullopt;
  std::vector<detail::SeekEntryInternal> internal;
  if (!detail::parse_seek_table(reinterpret_cast<const u8*>(src.data()),
                                src.size(), internal))
    return std::nullopt;
  std::vector<SeekEntry> out;
  out.reserve(internal.size());
  for (const auto& e : internal)
    out.push_back(SeekEntry{e.comp_size, e.regen_size});
  return out;
}

std::optional<std::uint64_t> get_frame_content_size(
    std::span<const std::byte> src) noexcept {
  if (detail::looks_like_l4ef(src)) return detail::l4ef_content_size(src);
  if (looks_like_spectra8(src)) {
    detail::ByteReader r(reinterpret_cast<const u8*>(src.data()), src.size());
    const u8* mg = r.take(detail::kMagicLen);
    if (!mg || std::memcmp(mg, detail::kMagic, detail::kMagicLen) != 0)
      return std::nullopt;
    u8 ver = r.byte();
    if (r.error || ver != detail::kVersion) return std::nullopt;
    u8 flags = r.byte();
    (void)r.byte();  // block_size_log
    if (r.error) return std::nullopt;
    if ((flags & detail::kFlagContentSize) == 0) return std::nullopt;
    u64 cs = r.varint();
    if (r.error) return std::nullopt;
    return cs;
  }
  if (detail::looks_like_zstd(src)) return detail::zstd_content_size(src);
  if (detail::looks_like_bzip3(src)) return detail::bzip3_content_size(src);
  if (detail::looks_like_xz(src)) return detail::lzma_content_size(src);
  return std::nullopt;
}

}  // namespace spectra8

// ---------------------------------------------------------------------------
// C ABI
// ---------------------------------------------------------------------------
extern "C" {

static spectra8_status to_c(spectra8::Status s) {
  return static_cast<spectra8_status>(static_cast<int>(s));
}

static spectra8::Codec from_c(spectra8_codec c) {
  return static_cast<spectra8::Codec>(static_cast<int>(c));
}

size_t spectra8_compress_bound(size_t src_size) {
  return spectra8::compress_bound(src_size);
}

size_t spectra8_compress_bound_codec(size_t src_size, spectra8_codec codec) {
  return spectra8::compress_bound(src_size, from_c(codec));
}

int spectra8_codec_available(spectra8_codec codec) {
  return spectra8::codec_available(from_c(codec)) ? 1 : 0;
}

spectra8_status spectra8_compress(const void* src, size_t src_size, void* dst,
                                    size_t dst_capacity, int level,
                                    size_t* written) {
  auto r = spectra8::compress(
      std::span<const std::byte>(static_cast<const std::byte*>(src), src_size),
      std::span<std::byte>(static_cast<std::byte*>(dst), dst_capacity), level);
  if (written) *written = r.value;
  return to_c(r.status);
}

spectra8_status spectra8_compress_ex(const void* src, size_t src_size,
                                       void* dst, size_t dst_capacity,
                                       spectra8_codec codec, int level,
                                       size_t* written) {
  spectra8::CompressParams p;
  p.codec = from_c(codec);
  p.level = level;
  auto r = spectra8::compress(
      std::span<const std::byte>(static_cast<const std::byte*>(src), src_size),
      std::span<std::byte>(static_cast<std::byte*>(dst), dst_capacity), p);
  if (written) *written = r.value;
  return to_c(r.status);
}

static spectra8::CompressParams options_from_c(
    const spectra8_compress_options* options) {
  spectra8::CompressParams p;
  if (!options) return p;
  p.codec = from_c(options->codec);
  if (options->level != 0 || p.codec == spectra8::Codec::Flash) p.level = options->level;
  p.block_log = options->block_log;
  // C zero-init yields 0 (= always take the smallest). C++ default is 1%.
  p.auto_min_gain_pct = options->auto_min_gain_pct;
  if (options->auto_max_level) p.auto_max_level = true;
  if (options->filters) p.filters = true;
  if (options->filter_fallback == 0) p.filter_fallback = false;
  if (options->filter_rle == 0) p.filter_rle = false;
  if (options->filter_dict == 0) p.filter_dict = false;
  if (options->filter_sample_full) p.filter_sample_full = true;
  if (options->auto_codecs != 0) p.auto_codecs = options->auto_codecs;
  if (options->filter_mask != 0) p.filter_mask = options->filter_mask;
  if (options->context_oracle) p.context_oracle = true;
  if (options->profile != SPECTRA8_PROFILE_NONE) {
    p.profile = static_cast<spectra8::Profile>(options->profile);
    p.profile_id = options->profile_id ? options->profile_id : static_cast<unsigned>(options->profile);
  }
  return p;
}

spectra8_status spectra8_compress_with_options(
    const void* src, size_t src_size, void* dst, size_t dst_capacity,
    const spectra8_compress_options* options, size_t* written) {
  auto r = spectra8::compress(
      std::span<const std::byte>(static_cast<const std::byte*>(src), src_size),
      std::span<std::byte>(static_cast<std::byte*>(dst), dst_capacity),
      options_from_c(options));
  if (written) *written = r.value;
  return to_c(r.status);
}

spectra8_status spectra8_decompress(const void* src, size_t src_size,
                                      void* dst, size_t dst_capacity,
                                      size_t* written) {
  auto r = spectra8::decompress(
      std::span<const std::byte>(static_cast<const std::byte*>(src), src_size),
      std::span<std::byte>(static_cast<std::byte*>(dst), dst_capacity));
  if (written) *written = r.value;
  return to_c(r.status);
}

int spectra8_frame_content_size(const void* src, size_t src_size,
                                 uint64_t* size) {
  auto v = spectra8::get_frame_content_size(
      std::span<const std::byte>(static_cast<const std::byte*>(src), src_size));
  if (!v) return 0;
  if (size) *size = *v;
  return 1;
}

const char* spectra8_status_message(spectra8_status s) {
  static thread_local std::string buf;
  buf = spectra8::status_message(static_cast<spectra8::Status>(s));
  return buf.c_str();
}

const char* spectra8_version(void) { return "2.0.0"; }

void spectra8_set_thread_profile(spectra8_profile profile) {
  spectra8::set_thread_profile(static_cast<spectra8::Profile>(profile));
}

spectra8_profile spectra8_get_thread_profile(void) {
  return static_cast<spectra8_profile>(spectra8::get_thread_profile());
}

int spectra8_train_and_register_profile(unsigned profile_id, const void* sample, size_t sample_size, int stride) {
  return spectra8_train_and_register_profile_level(profile_id, sample, sample_size, stride, 5);
}

int spectra8_train_and_register_profile_level(unsigned profile_id, const void* sample, size_t sample_size, int stride, int level) {
  if (!sample || sample_size == 0) return 0;
  return spectra8::train_and_register_profile(
      profile_id,
      std::span<const std::byte>(static_cast<const std::byte*>(sample), sample_size),
      stride, level) ? 1 : 0;
}

int spectra8_train_profile_model(const void* sample, size_t sample_size, int stride,
                                  void* out_model, size_t* out_model_size) {
  return spectra8_train_profile_model_level(sample, sample_size, stride, 5, out_model, out_model_size);
}

int spectra8_train_profile_model_level(const void* sample, size_t sample_size, int stride, int level,
                                        void* out_model, size_t* out_model_size) {
  if (!sample || sample_size == 0 || !out_model_size) return 0;
  auto model = spectra8::train_profile_model(
      std::span<const std::byte>(static_cast<const std::byte*>(sample), sample_size), stride, level);
  if (model.empty()) return 0;
  if (!out_model) {
    *out_model_size = model.size();
    return 1;
  }
  if (*out_model_size < model.size()) return 0;
  std::memcpy(out_model, model.data(), model.size());
  *out_model_size = model.size();
  return 1;
}

int spectra8_register_profile_model(unsigned profile_id, const void* model_bytes, size_t model_size) {
  if (!model_bytes || model_size == 0) return 0;
  return spectra8::register_profile_model(
      profile_id,
      std::span<const std::byte>(static_cast<const std::byte*>(model_bytes), model_size)) ? 1 : 0;
}

int spectra8_export_profile_model(unsigned profile_id, void* out_model, size_t* out_model_size) {
  if (!out_model_size) return 0;
  auto model = spectra8::export_profile_model(profile_id);
  if (model.empty()) return 0;
  if (!out_model) {
    *out_model_size = model.size();
    return 1;
  }
  if (*out_model_size < model.size()) return 0;
  std::memcpy(out_model, model.data(), model.size());
  *out_model_size = model.size();
  return 1;
}

}  // extern "C"

namespace spectra8 {

void set_thread_profile(Profile profile) noexcept {
  g_thread_profile = profile;
}

Profile get_thread_profile() noexcept {
  return g_thread_profile;
}

bool train_and_register_profile(
    std::uint32_t profile_id, std::span<const std::byte> sample, int stride, int level) noexcept {
  return detail::train_and_register_profile(
      profile_id, reinterpret_cast<const detail::u8*>(sample.data()), sample.size(), stride, level);
}

std::vector<std::uint8_t> train_profile_model(
    std::span<const std::byte> sample, int stride, int level) {
  std::vector<std::uint8_t> out;
  if (!detail::train_profile_model(
          reinterpret_cast<const detail::u8*>(sample.data()), sample.size(), stride, out, level)) {
    out.clear();
  }
  return out;
}

bool register_profile_model(
    std::uint32_t profile_id, std::span<const std::byte> model_bytes) noexcept {
  return detail::register_profile_model(
      profile_id, reinterpret_cast<const detail::u8*>(model_bytes.data()), model_bytes.size());
}

std::vector<std::uint8_t> export_profile_model(std::uint32_t profile_id) {
  std::vector<std::uint8_t> out;
  if (!detail::export_profile_model(profile_id, out)) {
    out.clear();
  }
  return out;
}

}  // namespace spectra8
