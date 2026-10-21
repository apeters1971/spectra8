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

#include "util/stats.hpp"

#include <atomic>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <string>

#include "util/format.hpp"
#include "util/transform.hpp"

namespace spectra8 {
namespace {

using namespace detail;

std::atomic<std::uint64_t> g_num_pages{0};
std::atomic<std::uint64_t> g_uncompressed_bytes{0};
std::atomic<std::uint64_t> g_compressed_bytes{0};

std::atomic<std::uint64_t> g_plain_pages{0};
std::atomic<std::uint64_t> g_no_transform_pages{0};
std::atomic<std::uint64_t> g_raw_won_pages{0};
std::atomic<std::uint64_t> g_filtered_won_pages{0};
std::atomic<std::uint64_t> g_stored_fallback_pages{0};

std::atomic<std::uint64_t> g_wire_raw_codec_pages{0};
std::atomic<std::uint64_t> g_wire_l4ef_filtered_pages{0};
std::atomic<std::uint64_t> g_wire_l4ef_stored_pages{0};
std::atomic<std::uint64_t> g_wire_spectra8_pages{0};

std::atomic<std::uint64_t> g_detected_filter_hits[static_cast<std::size_t>(StatsFilterKind::Count)]{};
std::atomic<std::uint64_t> g_l4ef_filter_hits[static_cast<std::size_t>(StatsFilterKind::Count)]{};
std::atomic<std::uint64_t> g_spectra8_filter_hits[static_cast<std::size_t>(StatsFilterKind::Count)]{};
std::atomic<std::uint64_t> g_l4ef_rle_pages{0};
std::atomic<std::uint64_t> g_spectra8_rle_pages{0};

std::atomic<std::uint64_t> g_auto_selections{0};
std::atomic<std::uint64_t> g_auto_codec_hits[11]{};

std::atomic<std::uint64_t> g_crystal_profile_blocks{0};
std::atomic<std::uint64_t> g_crystal_profile_escapes{0};

struct FrameHeader {
  bool has_bcsum = false;
  bool has_filters = false;
};

struct BlockScan {
  Filter filter = Filter::None;
  bool filter_rle = false;
  usize regen = 0;
};

StatsFilterKind filter_to_stats(Filter f) noexcept {
  if (f == Filter::None) return StatsFilterKind::None;
  const auto v = static_cast<std::uint8_t>(f);
  if (v >= static_cast<std::uint8_t>(StatsFilterKind::Count)) return StatsFilterKind::None;
  return static_cast<StatsFilterKind>(v);
}

Status parse_spectra8_header(ByteReader& r, FrameHeader& h) {
  const u8* mg = r.take(kMagicLen);
  if (!mg || std::memcmp(mg, kMagic, kMagicLen) != 0) return Status::BadMagic;
  u8 ver = r.byte();
  if (r.error || ver != kVersion) return Status::CorruptInput;
  u8 flags = r.byte();
  (void)r.byte();
  if (r.error) return Status::CorruptInput;
  h.has_bcsum = (flags & kFlagBlockChecksums) != 0;
  h.has_filters = (flags & kFlagFilters) != 0;
  if ((flags & kFlagContentSize) != 0) (void)r.varint();
  return r.error ? Status::CorruptInput : Status::Ok;
}

Status scan_block_filters(ByteReader& r, bool has_bcsum, bool has_filters, BlockScan& b,
                          bool& last) {
  u8 bt = r.byte();
  if (r.error) return Status::CorruptInput;
  last = (bt & kBlockLastFlag) != 0;
  const BlockType type = static_cast<BlockType>(bt & kBlockTypeMask);
  b.regen = static_cast<usize>(r.varint());
  if (r.error) return Status::CorruptInput;
  if (has_bcsum && !r.take(4)) return Status::CorruptInput;

  b.filter = Filter::None;
  b.filter_rle = false;
  switch (type) {
    case BlockType::Stored:
      if (!r.take(b.regen)) return Status::CorruptInput;
      break;
    case BlockType::Memset:
      (void)r.byte();
      break;
    case BlockType::Compressed:
    case BlockType::Bwt:
    case BlockType::Mosaic:
    case BlockType::Oracle:
    case BlockType::Crystal:
    case BlockType::Flash:
    case BlockType::Laser:
    case BlockType::Grating:
      if (has_filters) {
        const u8 fb = r.byte();
        if (r.error || !filter_is_valid(fb)) return Status::CorruptInput;
        b.filter = filter_id_from_byte(fb);
        b.filter_rle = filter_byte_has_rle(fb);
        if (b.filter_rle || filter_is_framed(b.filter)) (void)r.varint();
      }
      {
        const usize plen = static_cast<usize>(r.varint());
        if (!r.take(plen)) return Status::CorruptInput;
      }
      break;
    default:
      return Status::CorruptInput;
  }
  return Status::Ok;
}

void record_page_filter(StatsWirePath wire, const CompressObserveInfo& info) noexcept {
  if (info.page_filter != StatsFilterKind::None &&
      info.page_filter != StatsFilterKind::Stored &&
      info.outcome != StatsPageOutcome::Plain) {
    g_detected_filter_hits[static_cast<std::size_t>(info.page_filter)].fetch_add(
        1, std::memory_order_relaxed);
  }

  if (wire == StatsWirePath::L4efFiltered && info.page_filter != StatsFilterKind::None) {
    g_l4ef_filter_hits[static_cast<std::size_t>(info.page_filter)].fetch_add(
        1, std::memory_order_relaxed);
    if (info.page_used_rle)
      g_l4ef_rle_pages.fetch_add(1, std::memory_order_relaxed);
  } else if (wire == StatsWirePath::Spectra8Frame && info.page_filter != StatsFilterKind::None) {
    g_spectra8_filter_hits[static_cast<std::size_t>(info.page_filter)].fetch_add(
        1, std::memory_order_relaxed);
    if (info.page_used_rle)
      g_spectra8_rle_pages.fetch_add(1, std::memory_order_relaxed);
  }
}

std::string format_bytes(std::uint64_t n) {
  const char* suffix = "B";
  double v = static_cast<double>(n);
  if (v >= 1024.0 * 1024.0) {
    v /= 1024.0 * 1024.0;
    suffix = "MiB";
  } else if (v >= 1024.0) {
    v /= 1024.0;
    suffix = "KiB";
  }
  std::ostringstream os;
  os.setf(std::ios::fixed);
  os.precision(v >= 100.0 ? 0 : (v >= 10.0 ? 1 : 2));
  os << v << ' ' << suffix;
  return os.str();
}

const char* filter_name(StatsFilterKind kind) noexcept {
  switch (kind) {
    case StatsFilterKind::None: return "none";
    case StatsFilterKind::Shuffle16: return "shuffle16";
    case StatsFilterKind::Shuffle32: return "shuffle32";
    case StatsFilterKind::Shuffle64: return "shuffle64";
    case StatsFilterKind::Delta16: return "delta16";
    case StatsFilterKind::Delta32: return "delta32";
    case StatsFilterKind::Delta64: return "delta64";
    case StatsFilterKind::ZigzagDelta16: return "zigzag16";
    case StatsFilterKind::ZigzagDelta32: return "zigzag32";
    case StatsFilterKind::ZigzagDelta64: return "zigzag64";
    case StatsFilterKind::DeltaShuffle32: return "deltashuf32";
    case StatsFilterKind::ZigzagDeltaShuffle32: return "zigdeltashuf32";
    case StatsFilterKind::BitShuffle32: return "bitshuffle32";
    case StatsFilterKind::DeltaShuffle64: return "deltashuf64";
    case StatsFilterKind::ZigzagShuffle32: return "zigshuf32";
    case StatsFilterKind::ZigzagShuffle64: return "zigshuf64";
    case StatsFilterKind::BitShuffle64: return "bitshuffle64";
    case StatsFilterKind::DeltaBitShuffle32: return "deltabitsh32";
    case StatsFilterKind::Dict16: return "dict16";
    case StatsFilterKind::For16: return "for16";
    case StatsFilterKind::For32: return "for32";
    case StatsFilterKind::For64: return "for64";
    case StatsFilterKind::Alp32: return "alp32";
    case StatsFilterKind::Alp64: return "alp64";
    case StatsFilterKind::Dict32: return "dict32";
    case StatsFilterKind::Dict64: return "dict64";
    case StatsFilterKind::XorShuffle32: return "xorshuf32";
    case StatsFilterKind::XorShuffle64: return "xorshuf64";
    case StatsFilterKind::Delta2Shuffle32: return "d2shuf32";
    case StatsFilterKind::ZigzagDelta2Shuffle32: return "zzd2shuf32";
    case StatsFilterKind::Delta2Shuffle64: return "d2shuf64";
    case StatsFilterKind::BcjX86: return "bcj_x86";
    case StatsFilterKind::DigitNorm: return "digit_norm";
    case StatsFilterKind::WordNorm: return "word_norm";
    case StatsFilterKind::Base64: return "base64";
    case StatsFilterKind::BitRound32: return "bitround32";
    case StatsFilterKind::BitRound64: return "bitround64";
    case StatsFilterKind::Lag2Shuffle32: return "lag2shuf32";
    case StatsFilterKind::Lag2Shuffle64: return "lag2shuf64";
    case StatsFilterKind::XorLag2Shuffle32: return "xorlag2shuf32";
    case StatsFilterKind::BcjArm64: return "bcj_arm64";
    case StatsFilterKind::Stored: return "stored";
    case StatsFilterKind::Count: break;
  }
  return "?";
}

void append_filter_histogram(std::ostringstream& os, const char* title, std::uint64_t denom,
                             const std::uint64_t* hits) {
  if (denom == 0) return;
  os << "  " << title << ":\n";
  for (std::size_t i = 0; i < static_cast<std::size_t>(StatsFilterKind::Count); ++i) {
    if (hits[i] == 0) continue;
    const double pct = 100.0 * static_cast<double>(hits[i]) / static_cast<double>(denom);
    os << "    " << filter_name(static_cast<StatsFilterKind>(i)) << ": " << hits[i] << " ("
       << std::fixed << std::setprecision(1) << pct << "%)\n";
  }
}

void append_outcome_line(std::ostringstream& os, const char* label, std::uint64_t count,
                         std::uint64_t denom) {
  if (count == 0) return;
  const double pct = denom ? 100.0 * static_cast<double>(count) / static_cast<double>(denom) : 0.0;
  os << "    " << label << ": " << count << " (" << std::fixed << std::setprecision(1) << pct
     << "%)\n";
}

const char* auto_codec_name(Codec c) noexcept {
  switch (c) {
    case Codec::Beam: return "beam";
    case Codec::Prism: return "prism";
    case Codec::Zstd: return "zstd";
    case Codec::Bzip3: return "bzip3";
    case Codec::Lzma: return "lzma";
    case Codec::Auto: return "auto";
    case Codec::Mosaic: return "mosaic";
    case Codec::Oracle: return "oracle";
    case Codec::Crystal: return "crystal";
    case Codec::Flash: return "flash";
    case Codec::Laser: return "laser";
    case Codec::Grating: return "grating";
  }
  return "?";
}

void append_auto_codec_stats(std::ostringstream& os, const CompressStats& stats) {
  if (stats.auto_selections == 0) return;
  os << "  auto codec selection (" << stats.auto_selections << " call"
     << (stats.auto_selections == 1 ? "" : "s") << "):\n";
  // Decode-speed order (fast → slow), matching Codec::Auto preference.
  static constexpr Codec kOrder[] = {Codec::Zstd, Codec::Beam,    Codec::Mosaic,
                                     Codec::Laser, Codec::Prism,  Codec::Lzma,  Codec::Bzip3,
                                     Codec::Crystal};
  for (Codec c : kOrder) {
    const auto idx = static_cast<std::size_t>(c);
    if (idx >= 11) continue;
    const std::uint64_t n = stats.auto_codec_hits[idx];
    if (n == 0) continue;
    const double pct =
        100.0 * static_cast<double>(n) / static_cast<double>(stats.auto_selections);
    os << "    " << std::left << std::setw(6) << auto_codec_name(c) << std::right
       << " " << std::setw(6) << n << " (" << std::fixed << std::setprecision(1) << pct
       << "%)\n";
  }
}

}  // namespace

void reset_compress_stats() noexcept {
  g_num_pages.store(0, std::memory_order_relaxed);
  g_uncompressed_bytes.store(0, std::memory_order_relaxed);
  g_compressed_bytes.store(0, std::memory_order_relaxed);
  g_plain_pages.store(0, std::memory_order_relaxed);
  g_no_transform_pages.store(0, std::memory_order_relaxed);
  g_raw_won_pages.store(0, std::memory_order_relaxed);
  g_filtered_won_pages.store(0, std::memory_order_relaxed);
  g_stored_fallback_pages.store(0, std::memory_order_relaxed);
  g_wire_raw_codec_pages.store(0, std::memory_order_relaxed);
  g_wire_l4ef_filtered_pages.store(0, std::memory_order_relaxed);
  g_wire_l4ef_stored_pages.store(0, std::memory_order_relaxed);
  g_wire_spectra8_pages.store(0, std::memory_order_relaxed);
  g_l4ef_rle_pages.store(0, std::memory_order_relaxed);
  g_spectra8_rle_pages.store(0, std::memory_order_relaxed);
  for (auto& c : g_detected_filter_hits) c.store(0, std::memory_order_relaxed);
  for (auto& c : g_l4ef_filter_hits) c.store(0, std::memory_order_relaxed);
  for (auto& c : g_spectra8_filter_hits) c.store(0, std::memory_order_relaxed);
  g_auto_selections.store(0, std::memory_order_relaxed);
  for (auto& c : g_auto_codec_hits) c.store(0, std::memory_order_relaxed);
  g_crystal_profile_blocks.store(0, std::memory_order_relaxed);
  g_crystal_profile_escapes.store(0, std::memory_order_relaxed);
}

CompressStats get_compress_stats() noexcept {
  CompressStats s;
  s.num_pages = g_num_pages.load(std::memory_order_relaxed);
  s.uncompressed_bytes = g_uncompressed_bytes.load(std::memory_order_relaxed);
  s.compressed_bytes = g_compressed_bytes.load(std::memory_order_relaxed);
  s.plain_pages = g_plain_pages.load(std::memory_order_relaxed);
  s.no_transform_pages = g_no_transform_pages.load(std::memory_order_relaxed);
  s.raw_won_pages = g_raw_won_pages.load(std::memory_order_relaxed);
  s.filtered_won_pages = g_filtered_won_pages.load(std::memory_order_relaxed);
  s.stored_fallback_pages = g_stored_fallback_pages.load(std::memory_order_relaxed);
  s.wire_raw_codec_pages = g_wire_raw_codec_pages.load(std::memory_order_relaxed);
  s.wire_l4ef_filtered_pages = g_wire_l4ef_filtered_pages.load(std::memory_order_relaxed);
  s.wire_l4ef_stored_pages = g_wire_l4ef_stored_pages.load(std::memory_order_relaxed);
  s.wire_spectra8_pages = g_wire_spectra8_pages.load(std::memory_order_relaxed);
  s.l4ef_rle_pages = g_l4ef_rle_pages.load(std::memory_order_relaxed);
  s.spectra8_rle_pages = g_spectra8_rle_pages.load(std::memory_order_relaxed);
  for (std::size_t i = 0; i < static_cast<std::size_t>(StatsFilterKind::Count); ++i) {
    s.detected_filter_hits[i] = g_detected_filter_hits[i].load(std::memory_order_relaxed);
    s.l4ef_filter_hits[i] = g_l4ef_filter_hits[i].load(std::memory_order_relaxed);
    s.spectra8_filter_hits[i] = g_spectra8_filter_hits[i].load(std::memory_order_relaxed);
  }
  s.auto_selections = g_auto_selections.load(std::memory_order_relaxed);
  for (std::size_t i = 0; i < 11; ++i)
    s.auto_codec_hits[i] = g_auto_codec_hits[i].load(std::memory_order_relaxed);
  s.crystal_profile_blocks = g_crystal_profile_blocks.load(std::memory_order_relaxed);
  s.crystal_profile_escapes = g_crystal_profile_escapes.load(std::memory_order_relaxed);
  return s;
}

std::string format_compress_stats(const CompressStats& stats) {
  std::ostringstream os;
  os << "  compress pages:     " << stats.num_pages << '\n';
  os << "  uncompressed total: " << format_bytes(stats.uncompressed_bytes) << '\n';
  os << "  compressed total:   " << format_bytes(stats.compressed_bytes) << '\n';
  if (stats.num_pages > 0) {
    const double avg_page =
        static_cast<double>(stats.uncompressed_bytes) / static_cast<double>(stats.num_pages);
    os << "  avg page size:      " << format_bytes(static_cast<std::uint64_t>(avg_page + 0.5))
       << '\n';
  }
  if (stats.uncompressed_bytes > 0) {
    const double ratio = static_cast<double>(stats.compressed_bytes) /
                         static_cast<double>(stats.uncompressed_bytes);
    os << "  avg ratio:          " << ratio << " (compressed/uncompressed)\n";
  }

  if (stats.num_pages > 0) {
    os << "  page outcomes (one row per page):\n";
    append_outcome_line(os, "filtered won", stats.filtered_won_pages, stats.num_pages);
    append_outcome_line(os, "raw won (fallback)", stats.raw_won_pages, stats.num_pages);
    append_outcome_line(os, "stored fallback", stats.stored_fallback_pages, stats.num_pages);
    append_outcome_line(os, "no transform", stats.no_transform_pages, stats.num_pages);
    append_outcome_line(os, "plain (filters off)", stats.plain_pages, stats.num_pages);

    os << "  wire format:\n";
    append_outcome_line(os, "L4EF filtered", stats.wire_l4ef_filtered_pages, stats.num_pages);
    append_outcome_line(os, "L4EF stored", stats.wire_l4ef_stored_pages, stats.num_pages);
    append_outcome_line(os, "Spectra8 frame", stats.wire_spectra8_pages, stats.num_pages);
    append_outcome_line(os, "raw codec", stats.wire_raw_codec_pages, stats.num_pages);
  }
  if (const auto total = stats.crystal_profile_blocks + stats.crystal_profile_escapes; total > 0) {
    os << "  crystal profile blocks:\n";
    append_outcome_line(os, "coded with profile", stats.crystal_profile_blocks, total);
    append_outcome_line(os, "escaped to dynamic", stats.crystal_profile_escapes, total);
  }

  {
    std::uint64_t detected = 0;
    for (std::size_t i = 0; i < static_cast<std::size_t>(StatsFilterKind::Count); ++i)
      detected += stats.detected_filter_hits[i];
    append_filter_histogram(os, "filters detected (autodetect, all pages)", detected,
                            stats.detected_filter_hits);
  }

  append_filter_histogram(os, "L4EF filters kept on wire (filtered-won pages only)",
                          stats.wire_l4ef_filtered_pages, stats.l4ef_filter_hits);
  if (stats.l4ef_rle_pages > 0)
    os << "    (+RLE on " << stats.l4ef_rle_pages << " L4EF page(s))\n";

  {
    std::uint64_t spectra8_filtered = 0;
    for (std::size_t i = 0; i < static_cast<std::size_t>(StatsFilterKind::Count); ++i)
      spectra8_filtered += stats.spectra8_filter_hits[i];
    append_filter_histogram(os, "Spectra8 filters kept on wire (dominant filter per page)", spectra8_filtered,
                            stats.spectra8_filter_hits);
  }
  if (stats.spectra8_rle_pages > 0)
    os << "    (+RLE on " << stats.spectra8_rle_pages << " Spectra8 page(s))\n";

  append_auto_codec_stats(os, stats);
  return os.str();
}

std::string format_auto_codec_stats(const CompressStats& stats) {
  std::ostringstream os;
  append_auto_codec_stats(os, stats);
  return os.str();
}

namespace detail {

thread_local CompressObserveInfo g_last_observe{};

void stats_reset_last_observe() noexcept { g_last_observe = {}; }

void stats_set_last_observe(const CompressObserveInfo& info) noexcept {
  g_last_observe = info;
  g_last_observe.explicit_set = true;
}

CompressObserveInfo stats_take_last_observe() noexcept {
  const auto info = g_last_observe;
  g_last_observe = {};
  return info;
}

CompressObserveInfo observe_spectra8_frame(std::span<const std::byte> frame,
                                       bool filters_enabled) noexcept {
  CompressObserveInfo info;
  info.wire_path = StatsWirePath::Spectra8Frame;
  if (!filters_enabled) {
    info.outcome = StatsPageOutcome::Plain;
    return info;
  }

  ByteReader r(reinterpret_cast<const u8*>(frame.data()), frame.size());
  FrameHeader h;
  if (parse_spectra8_header(r, h) != Status::Ok) {
    info.outcome = StatsPageOutcome::NoTransform;
    return info;
  }

  StatsFilterKind dominant = StatsFilterKind::None;
  usize dominant_regen = 0;
  bool any_rle = false;
  bool any_filter = false;

  bool last = false;
  while (!last && !r.error) {
    BlockScan b;
    if (scan_block_filters(r, h.has_bcsum, h.has_filters, b, last) != Status::Ok) break;
    if (b.filter_rle) any_rle = true;
    if (b.filter != Filter::None) {
      any_filter = true;
      if (b.regen >= dominant_regen) {
        dominant_regen = b.regen;
        dominant = filter_to_stats(b.filter);
      }
    }
  }

  info.page_filter = dominant;
  info.page_used_rle = any_rle;
  info.outcome = any_filter ? StatsPageOutcome::FilteredWon : StatsPageOutcome::NoTransform;
  return info;
}

void stats_observe_crystal_profile_block(bool escaped) noexcept {
  (escaped ? g_crystal_profile_escapes : g_crystal_profile_blocks).fetch_add(1, std::memory_order_relaxed);
}

void stats_observe_compress(std::size_t uncompressed_size, std::span<const std::byte> compressed,
                            const CompressObserveInfo& info) noexcept {
  (void)compressed;
  g_num_pages.fetch_add(1, std::memory_order_relaxed);
  g_uncompressed_bytes.fetch_add(uncompressed_size, std::memory_order_relaxed);
  g_compressed_bytes.fetch_add(compressed.size(), std::memory_order_relaxed);

  switch (info.outcome) {
    case StatsPageOutcome::Plain: g_plain_pages.fetch_add(1, std::memory_order_relaxed); break;
    case StatsPageOutcome::NoTransform:
      g_no_transform_pages.fetch_add(1, std::memory_order_relaxed);
      break;
    case StatsPageOutcome::RawWon: g_raw_won_pages.fetch_add(1, std::memory_order_relaxed); break;
    case StatsPageOutcome::FilteredWon:
      g_filtered_won_pages.fetch_add(1, std::memory_order_relaxed);
      break;
    case StatsPageOutcome::StoredFallback:
      g_stored_fallback_pages.fetch_add(1, std::memory_order_relaxed);
      break;
  }

  switch (info.wire_path) {
    case StatsWirePath::RawCodec:
      g_wire_raw_codec_pages.fetch_add(1, std::memory_order_relaxed);
      break;
    case StatsWirePath::L4efFiltered:
      g_wire_l4ef_filtered_pages.fetch_add(1, std::memory_order_relaxed);
      break;
    case StatsWirePath::L4efStored:
      g_wire_l4ef_stored_pages.fetch_add(1, std::memory_order_relaxed);
      break;
    case StatsWirePath::Spectra8Frame: g_wire_spectra8_pages.fetch_add(1, std::memory_order_relaxed); break;
  }

  record_page_filter(info.wire_path, info);

  if (info.has_auto_winner) {
    const auto idx = static_cast<std::size_t>(info.auto_winner);
    if (idx < 11 && info.auto_winner != Codec::Auto) {
      g_auto_selections.fetch_add(1, std::memory_order_relaxed);
      g_auto_codec_hits[idx].fetch_add(1, std::memory_order_relaxed);
    }
  }
}

}  // namespace detail
}  // namespace spectra8
