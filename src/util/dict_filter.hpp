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

// Dictionary-remap filter (Filter::Dict16 / Dict32 / Dict64): replace each
// W-byte element of a block with a dense id into the sorted set of distinct
// element values, and feed the (much shorter) id stream to the normal
// LZ/BWT + entropy stage. Physics-style numeric columns routinely hold only a
// few hundred distinct values per 128 KiB block, so the whole information
// content is the id stream (<= 2 bytes/element) plus a small dictionary -
// structure the byte-oriented pipeline cannot reach through shuffle/delta
// transforms alone.
//
// Unlike the transforms in transform.hpp this filter is NOT size-preserving:
// the filtered representation is a self-framed stream (see layout below) whose
// length depends on the block content, so the block header stores the stream
// length the same way the byte-RLE layer stores its intermediate length (a
// varint after the filter byte; see encoder.cpp / decoder.cpp).
//
// Filtered stream layout (all varints LEB128):
//   varint  elem_count            ne = n / W (>= 1)
//   varint  dict_size             d in [1, 65536]
//   varint  x d                   sorted dictionary, delta coded:
//                                   first = v[0], then v[i] - v[i-1] (>= 1)
//   ne bytes                      id low byte plane   (id = sorted rank)
//   ne bytes                      id high byte plane  (present iff d > 256)
//   n - ne*W bytes                raw tail (block length not a multiple of W)
//
// Ids are sorted ranks, so the value ordering survives the remap: clustered
// numeric distributions produce smooth id distributions with a near-constant
// high plane, which the downstream LZ + entropy stage codes well. The inverse
// is a single table gather (dictionary of <= 64K u32/u64 entries; typical
// blocks use a few hundred, i.e. L1-resident).
#ifndef SPECTRA8_SRC_DICT_FILTER_HPP
#define SPECTRA8_SRC_DICT_FILTER_HPP

#include <algorithm>
#include <cstring>
#include <vector>

#include "util/common.hpp"
#include "util/transform.hpp"

namespace spectra8::detail {

// Hard cap on dictionary entries: ids must fit two byte planes.
inline constexpr usize kDictMaxSyms = usize{1} << 16;

namespace dict_detail {

// splitmix64 finalizer: cheap, well-mixed hash for the open-addressing table.
inline u64 mix(u64 x) noexcept {
  x ^= x >> 33;
  x *= 0xFF51AFD7ED558CCDull;
  x ^= x >> 33;
  x *= 0xC4CEB9FE1A85EC53ull;
  x ^= x >> 33;
  return x;
}

inline constexpr u32 kEmpty = 0xFFFFFFFFu;

// Per-thread scratch for the element hash table and the sorted dictionary.
struct Scratch {
  std::vector<u64> keys;
  std::vector<u32> ids;     // kEmpty = free slot
  std::vector<u64> sorted;  // distinct values, ascending
};

inline Scratch& scratch() {
  static thread_local Scratch s;
  return s;
}

// Slot count for up to `max_distinct` entries at <= 0.5 load factor.
inline usize table_slots(usize max_distinct) noexcept {
  usize slots = 64;
  while (slots < 2 * max_distinct) slots <<= 1;
  return slots;
}

}  // namespace dict_detail

// Build the self-framed dictionary stream for src[0, n) with w-byte (2, 4 or 8)
// little-endian elements, appending nothing on failure. Returns false when the
// block holds more than kDictMaxSyms distinct elements (graceful bail: the
// encoder falls back to the other candidates) or when the stream would not be
// smaller than the raw block.
inline bool dict_filter_fwd(const u8* src, usize n, int w,
                            std::vector<u8>& out) {
  out.clear();
  if (w != 2 && w != 4 && w != 8) return false;
  const usize ne = n / static_cast<usize>(w);
  if (ne == 0) return false;
  const usize tail = n - ne * static_cast<usize>(w);

  auto& s = dict_detail::scratch();
  const usize slots = dict_detail::table_slots(min_of(ne, kDictMaxSyms));
  const usize mask = slots - 1;
  s.keys.resize(slots);
  s.ids.assign(slots, dict_detail::kEmpty);
  s.sorted.clear();

  // Pass 1: collect distinct values; bail as soon as the dictionary overflows.
  for (usize e = 0; e < ne; ++e) {
    const u64 v = load_le(src + e * static_cast<usize>(w), w);
    usize h = dict_detail::mix(v) & mask;
    for (;;) {
      const u32 id = s.ids[h];
      if (id == dict_detail::kEmpty) {
        if (s.sorted.size() >= kDictMaxSyms) return false;
        s.ids[h] = static_cast<u32>(s.sorted.size());
        s.keys[h] = v;
        s.sorted.push_back(v);
        break;
      }
      if (s.keys[h] == v) break;
      h = (h + 1) & mask;
    }
  }
  const usize d = s.sorted.size();
  std::sort(s.sorted.begin(), s.sorted.end());

  // Re-point every occupied slot at its value's sorted rank, so pass 2 is a
  // single O(1) lookup per element.
  for (usize h = 0; h < slots; ++h) {
    if (s.ids[h] == dict_detail::kEmpty) continue;
    s.ids[h] = static_cast<u32>(
        std::lower_bound(s.sorted.begin(), s.sorted.end(), s.keys[h]) -
        s.sorted.begin());
  }

  ByteWriter wr(out);
  wr.varint(ne);
  wr.varint(d);
  u64 prev = 0;
  for (usize i = 0; i < d; ++i) {
    wr.varint(s.sorted[i] - prev);
    prev = s.sorted[i];
  }

  // Pass 2: emit the id byte planes (low plane, then high plane iff d > 256).
  const bool two = d > 256;
  const usize lo_off = out.size();
  out.resize(out.size() + ne * (two ? 2 : 1) + tail);
  u8* lo = out.data() + lo_off;
  u8* hi = lo + ne;  // valid only when `two`
  for (usize e = 0; e < ne; ++e) {
    const u64 v = load_le(src + e * static_cast<usize>(w), w);
    usize h = dict_detail::mix(v) & mask;
    while (s.ids[h] == dict_detail::kEmpty || s.keys[h] != v)
      h = (h + 1) & mask;
    const u32 id = s.ids[h];
    lo[e] = static_cast<u8>(id);
    if (two) hi[e] = static_cast<u8>(id >> 8);
  }
  if (tail)
    std::memcpy(out.data() + lo_off + ne * (two ? 2 : 1),
                src + ne * static_cast<usize>(w), tail);
  return out.size() < n;
}

// Expand a dictionary stream back to the n raw block bytes. Validates every
// framing field; a corrupt stream can never read or write out of bounds.
inline Status dict_filter_inv(const u8* data, usize len, u8* dst, usize n,
                              int w) {
  if (w != 2 && w != 4 && w != 8) return Status::CorruptInput;
  ByteReader r(data, len);
  const u64 ne64 = r.varint();
  const u64 d64 = r.varint();
  if (r.error) return Status::CorruptInput;
  const usize ne = n / static_cast<usize>(w);
  const usize tail = n - ne * static_cast<usize>(w);
  if (ne64 != ne || ne == 0) return Status::CorruptInput;
  if (d64 == 0 || d64 > kDictMaxSyms || d64 > ne64) return Status::CorruptInput;
  const usize d = static_cast<usize>(d64);

  static thread_local std::vector<u64> dict;
  dict.resize(d);
  const u64 vmax = width_mask(w);
  u64 prev = 0;
  for (usize i = 0; i < d; ++i) {
    const u64 delta = r.varint();
    if (r.error) return Status::CorruptInput;
    if (i > 0 && delta == 0) return Status::CorruptInput;  // must be ascending
    if (delta > vmax - prev) return Status::CorruptInput;  // width overflow
    prev += delta;
    dict[i] = prev;
  }

  const bool two = d > 256;
  const u8* lo = r.take(ne);
  const u8* hi = two ? r.take(ne) : nullptr;
  const u8* tp = tail ? r.take(tail) : nullptr;
  if (r.error || !lo || (two && !hi) || (tail && !tp) || r.remaining() != 0)
    return Status::CorruptInput;

  for (usize e = 0; e < ne; ++e) {
    const u32 id = two ? (static_cast<u32>(lo[e]) | (static_cast<u32>(hi[e]) << 8))
                       : static_cast<u32>(lo[e]);
    if (id >= d) return Status::CorruptInput;
    store_le(dst + e * static_cast<usize>(w), dict[id], w);
  }
  if (tail) std::memcpy(dst + ne * static_cast<usize>(w), tp, tail);
  return Status::Ok;
}

// Cheap detector: count distinct elements on a sample (middle 16 KiB by
// default, like detect_filter; whole block when `full_sample`) for w = 2, 4 and 8.
// A width qualifies when the sample shows at least 2x element repetition; among
// qualifying widths the one with the lowest estimated id cost per input byte
// (log2(distinct) / w) wins. Returns the element width (2, 4 or 8) or 0 when a
// dictionary looks hopeless. The full block may still overflow 64K distinct
// values - dict_filter_fwd bails gracefully, and the encoder's keep-smaller
// trials make a wrong guess here a pure encode-time cost, never a size
// regression.
inline int detect_dict_width(const u8* src, usize n, bool full_sample = false,
                             u64 mask = ~u64{0}) {
  if (n < 64) return 0;
  const bool allow16 = (mask >> static_cast<unsigned>(Filter::Dict16)) & 1u;
  const bool allow32 = (mask >> static_cast<unsigned>(Filter::Dict32)) & 1u;
  const bool allow64 = (mask >> static_cast<unsigned>(Filter::Dict64)) & 1u;
  if (!allow16 && !allow32 && !allow64) return 0;
  constexpr usize kSample = usize{1} << 14;
  const usize sn = full_sample ? n : min_of(n, kSample);
  // Align the sample to the widest candidate element so sampled elements
  // coincide with real ones (a misaligned sample straddles element boundaries
  // and looks spuriously high-cardinality). Filters always start at block
  // offset 0, so alignment relative to the block start is what matters.
  const usize start =
      ((!full_sample && n > sn) ? (n - sn) / 2 : 0) & ~usize{7};
  const u8* s = src + start;

  auto& sc = dict_detail::scratch();
  double best_score = 1e30;
  int best_w = 0;
  for (int w : {2, 4, 8}) {
    if ((w == 2 && !allow16) || (w == 4 && !allow32) || (w == 8 && !allow64))
      continue;
    const usize se = sn / static_cast<usize>(w);
    if (se < 16) continue;
    const usize slots = dict_detail::table_slots(se);
    const usize mask = slots - 1;
    sc.keys.resize(slots);
    sc.ids.assign(slots, dict_detail::kEmpty);
    usize distinct = 0;
    for (usize e = 0; e < se; ++e) {
      const u64 v = load_le(s + e * static_cast<usize>(w), w);
      usize h = dict_detail::mix(v) & mask;
      for (;;) {
        if (sc.ids[h] == dict_detail::kEmpty) {
          sc.ids[h] = 0;
          sc.keys[h] = v;
          ++distinct;
          break;
        }
        if (sc.keys[h] == v) break;
        h = (h + 1) & mask;
      }
    }
    if (distinct * 2 > se) continue;  // not enough repetition to bother
    // Two id planes are as wide as a 2-byte element, so a dict16 of more than
    // 256 symbols cannot shrink the block (the forward pass would decline).
    // Skipping it here lets a frame-of-reference or decimal frame still run.
    const int planes = distinct > 256 ? 2 : 1;
    if (planes >= w) continue;
    const double score =
        std::log2(static_cast<double>(distinct) + 1.0) / static_cast<double>(w);
    if (score < best_score) {
      best_score = score;
      best_w = w;
    }
  }
  return best_w;
}

}  // namespace spectra8::detail

#endif  // SPECTRA8_SRC_DICT_FILTER_HPP
