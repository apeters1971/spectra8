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

#include "prism/lzp.hpp"

#include <cstring>
#include <vector>

#include "util/common.hpp"
#include "util/simd.hpp"

namespace spectra8::detail {
namespace {

inline constexpr std::size_t kLzpDictBits = 19;  // 512K entries = 2 MiB
inline constexpr std::uint32_t kLzpDictSize = 1u << kLzpDictBits;
inline constexpr std::uint32_t kLzpDictMask = kLzpDictSize - 1;

[[nodiscard]] inline std::uint32_t lzp_hash(std::uint32_t ctx) noexcept {
  return (ctx ^ (ctx >> 15) ^ (ctx >> 3)) & kLzpDictMask;
}

[[nodiscard]] inline std::uint32_t load_ctx4(const std::uint8_t* p) noexcept {
  return static_cast<std::uint32_t>(p[0]) |
         (static_cast<std::uint32_t>(p[1]) << 8) |
         (static_cast<std::uint32_t>(p[2]) << 16) |
         (static_cast<std::uint32_t>(p[3]) << 24);
}

}  // namespace

bool lzp_compress(const std::uint8_t* src, std::size_t n, std::uint8_t* dst,
                  std::size_t dst_cap, std::size_t& out_len) noexcept {
  if (n < kLzpMinBlock || dst_cap < n) return false;

  thread_local std::vector<std::int32_t> lut;
  lut.assign(kLzpDictSize, -1);

  const std::uint8_t* const in_beg = src;
  const std::uint8_t* in = src;
  const std::uint8_t* const in_end = src + n;

  std::uint8_t* out = dst;
  std::uint8_t* const out_end = dst + dst_cap;
  std::uint8_t* const out_eob = dst + (n > 32 ? n - 32 : dst_cap);

  // First 4 bytes are emitted as literals.
  for (int i = 0; i < 4; ++i) *out++ = *in++;

  std::uint32_t ctx = load_ctx4(in - 4);
  const std::uint8_t* heur = in;

  while (in < in_end - kLzpMinMatch - 8 && out < out_eob) {
    const std::uint32_t idx = lzp_hash(ctx);
    const std::int32_t val = lut[idx];
    lut[idx] = static_cast<std::int32_t>(in - in_beg);

    if (val >= 0) {
      const std::uint8_t* const ref = in_beg + val;
      // Fast check: verify 4 bytes at offset (kLzpMinMatch - 4) and first 4 bytes.
      if (std::memcmp(in + kLzpMinMatch - 4, ref + kLzpMinMatch - 4, 4) == 0 &&
          std::memcmp(in, ref, 4) == 0) {
        if (heur > in && *heur != *(ref + (heur - in))) goto not_found;

        std::size_t len = 4;
        while (in + len + 8 <= in_end) {
          std::uint64_t diff = 0;
          std::memcpy(&diff, in + len, 8);
          std::uint64_t diff_ref = 0;
          std::memcpy(&diff_ref, ref + len, 8);
          if (diff != diff_ref) {
#if defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
            len += static_cast<std::size_t>(ctz64(diff ^ diff_ref)) >> 3;
#else
            while (in[len] == ref[len]) ++len;
#endif
            break;
          }
          len += 8;
        }

        if (len < kLzpMinMatch) {
          if (heur < in + len) heur = in + len;
          goto not_found;
        }

        in += len;
        ctx = load_ctx4(in - 4);
        *out++ = kLzpMatchMarker;

        len -= kLzpMinMatch;
        while (len >= 254 && out < out_eob) {
          *out++ = 254;
          len -= 254;
        }
        if (out >= out_eob) return false;
        *out++ = static_cast<std::uint8_t>(len);
        continue;
      }
    }

not_found:
    const std::uint8_t next = *in++;
    *out++ = next;
    ctx = (ctx << 8) | next;
    if (next == kLzpMatchMarker && val >= 0) {
      if (out >= out_eob) return false;
      *out++ = 255;
    }
  }

  // Tail bytes
  while (in < in_end && out < out_end - 2) {
    const std::uint32_t idx = lzp_hash(ctx);
    const std::int32_t val = lut[idx];
    lut[idx] = static_cast<std::int32_t>(in - in_beg);

    const std::uint8_t next = *in++;
    *out++ = next;
    ctx = (ctx << 8) | next;
    if (next == kLzpMatchMarker && val >= 0) {
      *out++ = 255;
    }
  }

  if (in < in_end) return false;
  const std::size_t comp_sz = static_cast<std::size_t>(out - dst);
  if (comp_sz >= n) return false;
  out_len = comp_sz;
  return true;
}

Status lzp_decompress(const std::uint8_t* src, std::size_t src_len,
                      std::uint8_t* dst, std::size_t orig_len) noexcept {
  if (orig_len < 4 || src_len < 4) return Status::CorruptInput;

  thread_local std::vector<std::int32_t> lut;
  lut.assign(kLzpDictSize, -1);

  const std::uint8_t* in = src;
  const std::uint8_t* const in_end = src + src_len;

  std::uint8_t* const out_beg = dst;
  std::uint8_t* out = dst;
  std::uint8_t* const out_end = dst + orig_len;

  for (int i = 0; i < 4; ++i) *out++ = *in++;

  std::uint32_t ctx = load_ctx4(out - 4);

  while (in < in_end && out < out_end) {
    const std::uint32_t idx = lzp_hash(ctx);
    const std::int32_t val = lut[idx];
    lut[idx] = static_cast<std::int32_t>(out - out_beg);

    const std::uint8_t c = *in++;
    if (c == kLzpMatchMarker && val >= 0) {
      if (in >= in_end) return Status::CorruptInput;
      const std::uint8_t next = *in++;
      if (next != 255) {
        std::size_t len = kLzpMinMatch;
        std::uint8_t step = next;
        while (step == 254) {
          len += 254;
          if (in >= in_end) return Status::CorruptInput;
          step = *in++;
        }
        len += step;

        if (out + len > out_end) return Status::CorruptInput;
        const std::uint8_t* const ref = out_beg + val;
        if (ref + len > out) {
          // Overlapping copy byte by byte
          for (std::size_t k = 0; k < len; ++k) out[k] = ref[k];
        } else {
          std::memcpy(out, ref, len);
        }
        out += len;
        ctx = load_ctx4(out - 4);
        continue;
      }
      // Escaped literal kLzpMatchMarker
      *out++ = kLzpMatchMarker;
      ctx = (ctx << 8) | kLzpMatchMarker;
    } else {
      *out++ = c;
      ctx = (ctx << 8) | c;
    }
  }

  if (out != out_end || in != in_end) return Status::CorruptInput;
  return Status::Ok;
}

}  // namespace spectra8::detail
