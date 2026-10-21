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

#include "ext/bzip3.hpp"

#include <cstring>
#include <vector>

#ifdef SPECTRA8_HAVE_BZIP3
#include <libbz3.h>
#endif

namespace spectra8::detail {

inline constexpr char kBz3Magic[] = "BZ3v1";
inline constexpr std::size_t kBz3MagicLen = 5;

#ifdef SPECTRA8_HAVE_BZIP3
Result<std::size_t> compress_stored_l4ef(std::span<const std::byte> src,
                                         std::span<std::byte> dst, Codec codec) noexcept;
#endif

namespace {

#ifdef SPECTRA8_HAVE_BZIP3

inline constexpr std::uint32_t kBz3DefaultBlock = 1u << 20;
inline constexpr std::uint32_t kBz3MinBlock = 65u * 1024u;
inline constexpr std::uint32_t kBz3MaxBlock = 511u * 1024u * 1024u;

[[nodiscard]] std::uint32_t block_size_from_log(int block_log) noexcept {
  if (block_log <= 0) return kBz3DefaultBlock;
  if (block_log < 16) block_log = 16;
  if (block_log > 28) block_log = 28;
  std::uint32_t bs = std::uint32_t{1} << block_log;
  if (bs < kBz3MinBlock) bs = kBz3MinBlock;
  if (bs > kBz3MaxBlock) bs = kBz3MaxBlock;
  return bs;
}

[[nodiscard]] Result<std::size_t> bzip3_identity_payload(
    std::span<const std::byte> src, std::span<std::byte> dst) noexcept {
  if (src.size() > dst.size()) return {Status::BufferTooSmall, 0};
  std::memcpy(dst.data(), src.data(), src.size());
  return {Status::Ok, src.size()};
}

[[nodiscard]] Result<std::size_t> try_bzip3_pack(std::uint32_t block_size,
                                                 std::span<const std::byte> src,
                                                 std::span<std::byte> dst) noexcept {
  size_t out_sz = dst.size();
  const int rc =
      bz3_compress(block_size, reinterpret_cast<const uint8_t*>(src.data()),
                   reinterpret_cast<uint8_t*>(dst.data()), src.size(), &out_sz);
  if (rc != BZ3_OK) {
    if (rc == BZ3_ERR_OUT_OF_BOUNDS) return {Status::BufferTooSmall, 0};
    return {Status::InternalError, 0};
  }
  if (src.empty()) return {Status::Ok, out_sz};

  // bz3_compress can return BZ3_OK with a tiny/unusable frame on some inputs
  // (seen on ~1 MiB float columns). Reject unless a full roundtrip matches.
  std::vector<uint8_t> probe(src.size());
  size_t probe_sz = probe.size();
  if (bz3_decompress(reinterpret_cast<const uint8_t*>(dst.data()),
                     probe.data(), out_sz, &probe_sz) != BZ3_OK)
    return {Status::InternalError, 0};
  if (probe_sz != src.size() ||
      std::memcmp(probe.data(), src.data(), src.size()) != 0)
    return {Status::InternalError, 0};
  return {Status::Ok, out_sz};
}

[[nodiscard]] Result<std::size_t> bzip3_compress_native(std::span<const std::byte> src,
                                                        std::span<std::byte> dst,
                                                        int block_log) noexcept {
  const std::uint32_t preferred = block_size_from_log(block_log);
  const std::uint32_t candidates[] = {
      preferred,
      768u * 1024u,
      8u << 20,
      64u << 20,
  };
  std::uint32_t last = 0;
  for (std::uint32_t block_size : candidates) {
    if (block_size == last) continue;
    last = block_size;
    auto r = try_bzip3_pack(block_size, src, dst);
    if (r.status == Status::Ok || r.status == Status::BufferTooSmall) return r;
  }
  return {Status::InternalError, 0};
}

#endif  // SPECTRA8_HAVE_BZIP3

}  // namespace

bool looks_like_bzip3(std::span<const std::byte> src) noexcept {
  return src.size() >= kBz3MagicLen &&
         std::memcmp(src.data(), kBz3Magic, kBz3MagicLen) == 0;
}

#ifdef SPECTRA8_HAVE_BZIP3

bool bzip3_available() noexcept { return true; }

std::size_t bzip3_compress_bound(std::size_t src_size) noexcept {
  // libbz3's bz3_bound() can undershoot its own output on tiny inputs (seen:
  // 359-byte input, bound 398, frame 399), and bz3_compress writes the whole
  // frame regardless -- a one-byte heap overflow whose lost tail byte only
  // shows up as a CRC error at decode time. Pad the bound so the frame always
  // fits; try_bzip3_pack still passes the full span to bz3_compress.
  return bz3_bound(src_size) + 256;
}

Result<std::size_t> bzip3_compress_payload(std::span<const std::byte> src,
                                           std::span<std::byte> dst,
                                           int block_log) noexcept {
  auto r = bzip3_compress_native(src, dst, block_log);
  if (r.ok()) return r;
  return bzip3_identity_payload(src, dst);
}

Result<std::size_t> bzip3_compress(std::span<const std::byte> src,
                                   std::span<std::byte> dst,
                                   int block_log) noexcept {
  auto r = bzip3_compress_native(src, dst, block_log);
  if (r.ok()) return r;
  // Prefer a snifferable L4EF stored frame over a bare identity copy: Auto
  // (and decompress() magic dispatch) cannot recover raw identity bytes.
  auto stored = compress_stored_l4ef(src, dst, Codec::Bzip3);
  if (stored.ok()) return stored;
  return bzip3_identity_payload(src, dst);
}

Result<std::size_t> bzip3_decompress(std::span<const std::byte> src,
                                     std::span<std::byte> dst) noexcept {
  size_t out_sz = dst.size();
  const int rc =
      bz3_decompress(reinterpret_cast<const uint8_t*>(src.data()),
                     reinterpret_cast<uint8_t*>(dst.data()), src.size(),
                     &out_sz);
  if (rc != BZ3_OK) {
    // BZ3_ERR_DATA_SIZE_TOO_SMALL only exists from libbz3 1.4 on; EPEL 9
    // ships 1.3.1, where an undersized output buffer reports OUT_OF_BOUNDS.
    if (rc == BZ3_ERR_OUT_OF_BOUNDS
#ifdef BZ3_ERR_DATA_SIZE_TOO_SMALL
        || rc == BZ3_ERR_DATA_SIZE_TOO_SMALL
#endif
    )
      return {Status::BufferTooSmall, 0};
    return {Status::CorruptInput, 0};
  }
  return {Status::Ok, out_sz};
}

std::optional<std::uint64_t> bzip3_content_size(
    std::span<const std::byte> src) noexcept {
  if (!looks_like_bzip3(src) || src.size() < kBz3MagicLen + 8)
    return std::nullopt;
  const auto* p = reinterpret_cast<const uint8_t*>(src.data());
  const std::size_t n = src.size();
  std::size_t off = kBz3MagicLen;
  auto read_u32 = [&](std::uint32_t& v) -> bool {
    if (off + 4 > n) return false;
    v = static_cast<std::uint32_t>(p[off]) |
        (static_cast<std::uint32_t>(p[off + 1]) << 8) |
        (static_cast<std::uint32_t>(p[off + 2]) << 16) |
        (static_cast<std::uint32_t>(p[off + 3]) << 24);
    off += 4;
    return true;
  };
  std::uint32_t block_size = 0, nblocks = 0;
  if (!read_u32(block_size) || !read_u32(nblocks)) return std::nullopt;
  (void)block_size;
  std::uint64_t total = 0;
  for (std::uint32_t i = 0; i < nblocks; ++i) {
    std::uint32_t comp = 0, orig = 0;
    if (!read_u32(comp) || !read_u32(orig)) return std::nullopt;
    total += orig;
    if (off + comp > n) return std::nullopt;
    off += comp;
  }
  return total;
}

#else  // !SPECTRA8_HAVE_BZIP3

bool bzip3_available() noexcept { return false; }

std::size_t bzip3_compress_bound(std::size_t) noexcept { return 0; }

Result<std::size_t> bzip3_compress_payload(std::span<const std::byte>,
                                           std::span<std::byte>, int) noexcept {
  return {Status::InvalidArgument, 0};
}

Result<std::size_t> bzip3_compress(std::span<const std::byte>,
                                   std::span<std::byte>, int) noexcept {
  return {Status::InvalidArgument, 0};
}

Result<std::size_t> bzip3_decompress(std::span<const std::byte>,
                                     std::span<std::byte>) noexcept {
  return {Status::UnsupportedVersion, 0};
}

std::optional<std::uint64_t> bzip3_content_size(
    std::span<const std::byte>) noexcept {
  return std::nullopt;
}

#endif

}  // namespace spectra8::detail
