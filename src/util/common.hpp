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

// Internal shared types and byte-stream helpers.
#ifndef SPECTRA8_SRC_COMMON_HPP
#define SPECTRA8_SRC_COMMON_HPP

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "spectra8/spectra8.hpp"

namespace spectra8::detail {

using u8 = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;
using i32 = std::int32_t;
using i64 = std::int64_t;
using usize = std::size_t;

// ---------------------------------------------------------------------------
// Unaligned little-endian load/store helpers.
// ---------------------------------------------------------------------------
inline u16 load_u16(const u8* p) noexcept {
  u16 v;
  std::memcpy(&v, p, sizeof(v));
  return v;
}
inline u32 load_u32(const u8* p) noexcept {
  u32 v;
  std::memcpy(&v, p, sizeof(v));
  return v;
}
inline u64 load_u64(const u8* p) noexcept {
  u64 v;
  std::memcpy(&v, p, sizeof(v));
  return v;
}
inline void store_u16(u8* p, u16 v) noexcept { std::memcpy(p, &v, sizeof(v)); }
inline void store_u32(u8* p, u32 v) noexcept { std::memcpy(p, &v, sizeof(v)); }
inline void store_u64(u8* p, u64 v) noexcept { std::memcpy(p, &v, sizeof(v)); }

// ---------------------------------------------------------------------------
// Append-only byte writer over a std::vector<u8>.
// ---------------------------------------------------------------------------
struct ByteWriter {
  std::vector<u8>& out;

  explicit ByteWriter(std::vector<u8>& o) noexcept : out(o) {}

  void byte(u8 v) { out.push_back(v); }
  void bytes(const u8* p, usize n) { out.insert(out.end(), p, p + n); }
  void bytes(const std::vector<u8>& v) { out.insert(out.end(), v.begin(), v.end()); }

  // Unsigned LEB128 varint.
  void varint(u64 v) {
    while (v >= 0x80) {
      out.push_back(static_cast<u8>(v) | 0x80);
      v >>= 7;
    }
    out.push_back(static_cast<u8>(v));
  }

  usize size() const noexcept { return out.size(); }
};

// ---------------------------------------------------------------------------
// Bounds-checked byte reader. Any out-of-range read flips `error`.
// ---------------------------------------------------------------------------
struct ByteReader {
  const u8* p;
  const u8* end;
  bool error = false;

  ByteReader(const u8* data, usize n) noexcept : p(data), end(data + n) {}

  usize remaining() const noexcept { return static_cast<usize>(end - p); }

  u8 byte() {
    if (p >= end) {
      error = true;
      return 0;
    }
    return *p++;
  }

  // Reads `n` raw bytes, returning the pointer to them (or nullptr on overrun).
  const u8* take(usize n) {
    if (remaining() < n) {
      error = true;
      return nullptr;
    }
    const u8* r = p;
    p += n;
    return r;
  }

  u64 varint() {
    u64 v = 0;
    int shift = 0;
    for (;;) {
      if (p >= end || shift > 63) {
        error = true;
        return 0;
      }
      u8 b = *p++;
      v |= static_cast<u64>(b & 0x7F) << shift;
      if ((b & 0x80) == 0) break;
      shift += 7;
    }
    return v;
  }
};

template <class T>
inline constexpr T min_of(T a, T b) noexcept {
  return a < b ? a : b;
}
template <class T>
inline constexpr T max_of(T a, T b) noexcept {
  return a > b ? a : b;
}

}  // namespace spectra8::detail

#endif  // SPECTRA8_SRC_COMMON_HPP
