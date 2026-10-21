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

#include "ext/filter_frame.hpp"

#include <algorithm>
#include <cstring>
#include <vector>

#include "util/byte_rle.hpp"
#include "util/pack_filter.hpp"
#include "ext/bzip3.hpp"
#include "ext/lzma.hpp"
#include "ext/zstd.hpp"
#include "util/format.hpp"
#include "util/stats.hpp"
#include "util/transform.hpp"

namespace spectra8::detail {

namespace {

struct ExternalStage {
  Filter filter = Filter::None;
  int dict_w = 0;
  bool use_rle = false;
  usize regen_size = 0;
  usize intermediate_len = 0;
  const u8* codec_src = nullptr;
  usize codec_len = 0;
};

bool prepare_external_stage(const u8* src, usize n, bool allow_dict, bool try_rle,
                            bool full_sample, u64 mask, ExternalStage& st,
                            std::vector<u8>& filt, std::vector<u8>& rle) {
  st = ExternalStage{};
  st.regen_size = n;
  st.codec_src = src;
  st.codec_len = n;

  Filter forced = Filter::None;
  if (singleton_forced_filter(mask, forced)) {
    if (filter_is_framed(forced)) {
      if (!allow_dict && filter_is_dict(forced)) return false;
      filt.clear();
      if (!framed_filter_fwd(forced, src, n, filt)) return false;
      st.filter = forced;
      st.dict_w = filter_is_dict(forced) ? dict_filter_width(forced) : 0;
      st.intermediate_len = filt.size();
      st.codec_src = filt.data();
      st.codec_len = filt.size();
      return true;
    }
    filt.resize(n);
    apply_filter(forced, src, n, filt.data());
    st.filter = forced;
    st.codec_src = filt.data();
    st.codec_len = n;
    if (try_rle) {
      byte_rle_encode(filt.data(), n, rle);
      if (rle.size() < n) {
        st.use_rle = true;
        st.intermediate_len = rle.size();
        st.codec_src = rle.data();
        st.codec_len = rle.size();
      }
    }
    return true;
  }

  const Filter framed =
      detect_framed_filter(src, n, full_sample, mask, allow_dict);
  if (filter_is_dict(framed)) {
    filt.clear();
    if (framed_filter_fwd(framed, src, n, filt)) {
      st.filter = framed;
      st.dict_w = dict_filter_width(framed);
      st.intermediate_len = filt.size();
      st.codec_src = filt.data();
      st.codec_len = filt.size();
      return true;
    }
  }

  st.filter = detect_filter(src, n, full_sample, mask);
  std::vector<u8> pack;
  const bool pack_ok = (filter_is_for(framed) || filter_is_alp(framed) ||
                        filter_is_digit_norm(framed) || filter_is_word_norm(framed) ||
                        filter_is_base64(framed)) &&
                       framed_filter_fwd(framed, src, n, pack);
  if (st.filter == Filter::None && !pack_ok) return false;
  if (pack_ok) {
    bool use_pack = st.filter == Filter::None;
    if (!use_pack) {
      filt.resize(n);
      apply_filter(st.filter, src, n, filt.data());
      use_pack = framed_beats_byte(filt.data(), n, st.filter, pack);
    }
    if (use_pack) {
      filt.swap(pack);
      st.filter = framed;
      st.dict_w = 0;
      st.intermediate_len = filt.size();
      st.codec_src = filt.data();
      st.codec_len = filt.size();
      return true;
    }
  }
  if (st.filter == Filter::None) return false;

  filt.resize(n);
  apply_filter(st.filter, src, n, filt.data());
  st.codec_src = filt.data();
  st.codec_len = n;

  if (try_rle) {
    byte_rle_encode(filt.data(), n, rle);
    if (rle.size() < n) {
      st.use_rle = true;
      st.intermediate_len = rle.size();
      st.codec_src = rle.data();
      st.codec_len = rle.size();
    }
  }
  return true;
}

StatsFilterKind stage_filter_kind(Filter f) noexcept {
  if (f == Filter::None) return StatsFilterKind::None;
  const auto v = static_cast<std::uint8_t>(f);
  if (v >= static_cast<std::uint8_t>(StatsFilterKind::Count)) return StatsFilterKind::None;
  return static_cast<StatsFilterKind>(v);
}

void set_l4ef_observe(const ExternalStage& stage, StatsPageOutcome outcome,
                      StatsWirePath wire) noexcept {
  CompressObserveInfo info;
  info.wire_path = wire;
  info.outcome = outcome;
  info.page_filter = stage_filter_kind(stage.filter);
  info.page_used_rle = stage.use_rle;
  stats_set_last_observe(info);
}

void set_raw_codec_observe(StatsPageOutcome outcome, const ExternalStage* stage = nullptr) noexcept {
  CompressObserveInfo info;
  info.wire_path = StatsWirePath::RawCodec;
  info.outcome = outcome;
  if (stage) {
    info.page_filter = stage_filter_kind(stage->filter);
    info.page_used_rle = stage->use_rle;
  }
  stats_set_last_observe(info);
}

Result<std::size_t> backend_compress(Codec codec, std::span<const std::byte> src,
                                     std::span<std::byte> dst, int level,
                                     int block_log) noexcept {
  switch (codec) {
    case Codec::Zstd:
      return zstd_compress(src, dst, level);
    case Codec::Bzip3:
      return bzip3_compress_payload(src, dst, block_log);
    case Codec::Lzma:
      return lzma_compress(src, dst, level);
    default:
      return {Status::InvalidArgument, 0};
  }
}

Result<std::size_t> backend_decompress(Codec codec, std::span<const std::byte> src,
                                         std::span<std::byte> dst) noexcept {
  switch (codec) {
    case Codec::Zstd:
      if (!zstd_available()) return {Status::UnsupportedVersion, 0};
      return zstd_decompress(src, dst);
    case Codec::Bzip3:
      if (!bzip3_available()) return {Status::UnsupportedVersion, 0};
      if (looks_like_bzip3(src)) return bzip3_decompress(src, dst);
      // Identity payload kept when native bzip3 could not roundtrip (small /
      // incompressible input). The outer L4EF header carries the regen size.
      if (src.size() > dst.size()) return {Status::BufferTooSmall, 0};
      std::memcpy(dst.data(), src.data(), src.size());
      return {Status::Ok, src.size()};
    case Codec::Lzma:
      if (!lzma_available()) return {Status::UnsupportedVersion, 0};
      return lzma_decompress(src, dst);
    default:
      return {Status::CorruptInput, 0};
  }
}

Status invert_external_stage(const ExternalStage& st, const u8* codec_out, usize codec_len,
                             u8* dst) {
  thread_local std::vector<u8> filtered;
  if (filter_is_framed(st.filter)) {
    return framed_filter_inv(st.filter, codec_out, codec_len, dst, st.regen_size);
  }
  if (st.use_rle) {
    if (st.filter == Filter::None) {
      return byte_rle_decode(codec_out, codec_len, dst, st.regen_size);
    }
    filtered.resize(st.regen_size);
    Status s = byte_rle_decode(codec_out, codec_len, filtered.data(), st.regen_size);
    if (s != Status::Ok) return s;
    invert_filter(st.filter, filtered.data(), st.regen_size, dst);
    return Status::Ok;
  }
  invert_filter(st.filter, codec_out, st.regen_size, dst);
  return Status::Ok;
}

void write_l4ef_header(std::vector<u8>& out, Codec codec, u8 filter_byte,
                       usize regen_size, usize intermediate_len, usize payload_size) {
  ByteWriter w(out);
  w.bytes(kL4efMagic, kMagicLen);
  w.byte(kL4efVersion);
  w.byte(static_cast<u8>(codec));
  w.byte(filter_byte);
  w.varint(regen_size);
  w.varint(intermediate_len);
  w.varint(payload_size);
}

bool read_l4ef_header(ByteReader& r, Codec& codec, u8& filter_byte, usize& regen_size,
                      usize& intermediate_len, usize& payload_size) {
  const u8* mg = r.take(kMagicLen);
  if (!mg || std::memcmp(mg, kL4efMagic, kMagicLen) != 0) return false;
  u8 ver = r.byte();
  if (r.error || ver != kL4efVersion) return false;
  u8 codec_b = r.byte();
  filter_byte = r.byte();
  if (r.error) return false;
  codec = static_cast<Codec>(codec_b);
  regen_size = r.varint();
  intermediate_len = r.varint();
  payload_size = r.varint();
  return !r.error;
}

Result<std::size_t> compress_stage_to_vec(Codec codec, const ExternalStage& stage, int level,
                                          int block_log, std::vector<std::byte>& out) {
  thread_local std::vector<u8> payload;
  const std::size_t bound =
      (codec == Codec::Zstd)    ? zstd_compress_bound(stage.codec_len)
      : (codec == Codec::Lzma)  ? lzma_compress_bound(stage.codec_len)
                                : bzip3_compress_bound(stage.codec_len);
  payload.resize(bound);
  const auto backend = backend_compress(
      codec,
      std::span<const std::byte>(reinterpret_cast<const std::byte*>(stage.codec_src),
                                 stage.codec_len),
      std::span<std::byte>(reinterpret_cast<std::byte*>(payload.data()), payload.size()),
      level, block_log);
  if (!backend.ok()) return backend;
  payload.resize(backend.value);

  thread_local std::vector<u8> wrapped;
  wrapped.clear();
  write_l4ef_header(wrapped, codec, pack_filter_byte(stage.filter, stage.use_rle),
                    stage.regen_size, stage.intermediate_len, payload.size());
  wrapped.insert(wrapped.end(), payload.begin(), payload.end());
  out.assign(reinterpret_cast<std::byte*>(wrapped.data()),
             reinterpret_cast<std::byte*>(wrapped.data()) + wrapped.size());
  return {Status::Ok, wrapped.size()};
}

}  // namespace

Result<std::size_t> compress_stored_l4ef(std::span<const std::byte> src,
                                         std::span<std::byte> dst, Codec codec) noexcept {
  thread_local std::vector<u8> wrapped;
  wrapped.clear();
  ByteWriter w(wrapped);
  w.bytes(kL4efMagic, kMagicLen);
  w.byte(kL4efVersion);
  w.byte(static_cast<u8>(codec));
  w.byte(kL4efFilterStored);
  w.varint(src.size());
  w.varint(0);
  w.varint(src.size());
  wrapped.insert(wrapped.end(), reinterpret_cast<const u8*>(src.data()),
                 reinterpret_cast<const u8*>(src.data()) + src.size());
  if (wrapped.size() > dst.size()) return {Status::BufferTooSmall, 0};
  std::memcpy(dst.data(), wrapped.data(), wrapped.size());
  CompressObserveInfo info;
  info.wire_path = StatsWirePath::L4efStored;
  info.outcome = StatsPageOutcome::StoredFallback;
  info.page_filter = StatsFilterKind::Stored;
  stats_set_last_observe(info);
  return {Status::Ok, wrapped.size()};
}

bool looks_like_l4ef(std::span<const std::byte> src) noexcept {
  return src.size() >= kMagicLen &&
         std::memcmp(src.data(), kL4efMagic, kMagicLen) == 0;
}

CompressObserveInfo observe_l4ef_frame(std::span<const std::byte> frame) noexcept {
  CompressObserveInfo info;
  info.wire_path = StatsWirePath::L4efFiltered;
  info.outcome = StatsPageOutcome::FilteredWon;
  ByteReader r(reinterpret_cast<const u8*>(frame.data()), frame.size());
  Codec codec;
  u8 fb = 0;
  usize regen = 0, inter = 0, payload = 0;
  if (!read_l4ef_header(r, codec, fb, regen, inter, payload)) return info;
  if (fb == kL4efFilterStored) {
    info.wire_path = StatsWirePath::L4efStored;
    info.outcome = StatsPageOutcome::StoredFallback;
    info.page_filter = StatsFilterKind::Stored;
    return info;
  }
  if (filter_is_valid(fb)) {
    info.page_filter = stage_filter_kind(filter_id_from_byte(fb));
    info.page_used_rle = filter_byte_has_rle(fb);
  }
  return info;
}

std::size_t l4ef_compress_bound(std::size_t src_size, Codec codec) noexcept {
  std::size_t payload_bound = src_size + src_size / 8 + 64;
  if (codec == Codec::Zstd && zstd_available())
    payload_bound = zstd_compress_bound(src_size);
  else if (codec == Codec::Bzip3 && bzip3_available())
    payload_bound = bzip3_compress_bound(src_size);
  else if (codec == Codec::Lzma && lzma_available())
    payload_bound = lzma_compress_bound(src_size);
  return kMagicLen + 3 + 30 + payload_bound;
}

Result<std::size_t> compress_filtered_frame(std::span<const std::byte> src,
                                            std::span<std::byte> dst, Codec codec,
                                            const CompressParams& params) noexcept {
  const u8* in = reinterpret_cast<const u8*>(src.data());
  const usize n = src.size();

  thread_local std::vector<u8> filt, rle;
  ExternalStage st;
  const bool has_stage = prepare_external_stage(
      in, n, params.filter_dict, params.filter_rle, params.filter_sample_full,
      params.filter_mask, st,
      filt, rle);

  auto try_raw = [&]() -> Result<std::size_t> {
    return backend_compress(codec, src, dst, params.level, params.block_log);
  };

  // A BCJ comes from the executable detector, not from the entropy proxy,
  // and on code it never loses materially to the raw block (41 x86-64 and
  // 34 arm64 libraries: raw won on 3 of each by 0.005 % in total). The raw
  // trial would double the encode time of the one codec that matters for
  // executables (lzma), so it is skipped for BCJ stages.
  const bool bcj_stage = has_stage && (st.filter == Filter::BcjX86 || st.filter == Filter::BcjArm64);
  const bool fallback = params.filter_fallback && !bcj_stage;

  if (!has_stage) {
    if (params.filters && !params.filter_fallback) return {Status::InvalidArgument, 0};
    auto raw = try_raw();
    if (raw.ok()) set_raw_codec_observe(StatsPageOutcome::NoTransform);
    return raw;
  }

  ExternalStage stage_no_rle = st;
  if (st.use_rle) {
    stage_no_rle.use_rle = false;
    stage_no_rle.intermediate_len = 0;
    stage_no_rle.codec_src = filt.data();
    stage_no_rle.codec_len = n;
  }

  thread_local std::vector<std::byte> filtered_buf, alt_buf;
  Result<std::size_t> best_filtered{Status::InternalError, 0};
  const ExternalStage* winning_stage = &st;
  if (st.use_rle) {
    auto r = compress_stage_to_vec(codec, st, params.level, params.block_log, filtered_buf);
    best_filtered = r;
    auto r2 =
        compress_stage_to_vec(codec, stage_no_rle, params.level, params.block_log, alt_buf);
    if (r2.ok() && (!best_filtered.ok() || r2.value < best_filtered.value)) {
      best_filtered = r2;
      filtered_buf.swap(alt_buf);
      winning_stage = &stage_no_rle;
    }
  } else {
    best_filtered =
        compress_stage_to_vec(codec, st, params.level, params.block_log, filtered_buf);
  }

  if (!best_filtered.ok()) return best_filtered;

  if (!fallback) {
    if (best_filtered.value > dst.size()) return {Status::BufferTooSmall, 0};
    std::memcpy(dst.data(), filtered_buf.data(), best_filtered.value);
    set_l4ef_observe(*winning_stage, StatsPageOutcome::FilteredWon, StatsWirePath::L4efFiltered);
    return best_filtered;
  }

  const bool filtered_fits = best_filtered.value <= dst.size();
  auto raw = try_raw();
  if (raw.ok()) {
    if (!filtered_fits || raw.value <= best_filtered.value) {
      set_raw_codec_observe(StatsPageOutcome::RawWon, winning_stage);
      return raw;
    }
  } else if (!filtered_fits) {
    return raw;
  }

  if (!filtered_fits) return {Status::BufferTooSmall, 0};

  std::memcpy(dst.data(), filtered_buf.data(), best_filtered.value);
  set_l4ef_observe(*winning_stage, StatsPageOutcome::FilteredWon, StatsWirePath::L4efFiltered);
  return best_filtered;
}

Result<std::size_t> decompress_l4ef_frame(std::span<const std::byte> src,
                                          std::span<std::byte> dst) noexcept {
  ByteReader r(reinterpret_cast<const u8*>(src.data()), src.size());
  Codec codec = Codec::Beam;
  u8 filter_byte = 0;
  usize regen_size = 0;
  usize intermediate_len = 0;
  usize payload_size = 0;
  if (!read_l4ef_header(r, codec, filter_byte, regen_size, intermediate_len, payload_size))
    return {Status::CorruptInput, 0};
  if (regen_size > dst.size()) return {Status::BufferTooSmall, 0};
  if (r.remaining() < payload_size) return {Status::CorruptInput, 0};

  if (filter_byte == kL4efFilterStored) {
    if (regen_size != payload_size) return {Status::CorruptInput, 0};
    std::memcpy(dst.data(), r.p, payload_size);
    return {Status::Ok, regen_size};
  }

  const auto payload =
      std::span<const std::byte>(reinterpret_cast<const std::byte*>(r.p), payload_size);

  thread_local std::vector<u8> decoded;
  decoded.resize(std::max({intermediate_len, regen_size, payload_size}));
  const auto dr = backend_decompress(
      codec, payload,
      std::span<std::byte>(reinterpret_cast<std::byte*>(decoded.data()), decoded.size()));
  if (!dr.ok()) return dr;

  ExternalStage st;
  st.regen_size = regen_size;
  st.intermediate_len = intermediate_len;
  st.filter = filter_id_from_byte(filter_byte);
  st.use_rle = filter_byte_has_rle(filter_byte);
  st.dict_w = filter_is_dict(st.filter) ? dict_filter_width(st.filter) : 0;
  if (filter_is_framed(st.filter) && filter_byte_has_rle(filter_byte))
    return {Status::CorruptInput, 0};

  Status inv =
      invert_external_stage(st, decoded.data(), dr.value, reinterpret_cast<u8*>(dst.data()));
  if (inv != Status::Ok) return {inv, 0};
  return {Status::Ok, regen_size};
}

std::optional<std::uint64_t> l4ef_content_size(std::span<const std::byte> src) noexcept {
  ByteReader r(reinterpret_cast<const u8*>(src.data()), src.size());
  Codec codec = Codec::Beam;
  u8 filter_byte = 0;
  usize regen_size = 0;
  usize intermediate_len = 0;
  usize payload_size = 0;
  if (!read_l4ef_header(r, codec, filter_byte, regen_size, intermediate_len, payload_size))
    return std::nullopt;
  return static_cast<std::uint64_t>(regen_size);
}

}  // namespace spectra8::detail
