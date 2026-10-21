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

// Flash codec (BlockType::Flash): ultra-fast byte-aligned LZ77 tier designed
// for 4–6 GB/s decompression and 800–1200 MB/s ingestion (NVMe line-rate).
//
// Encoder level selects the bitstream, not the decoder:
//   * level < 5 (L1): original interleaved-literal stream. Decode stays on
//     the SIMD copy fast path (~4 GB/s).
//   * level >= 5 (L5): optional 0x1F 0x00 0x00 flags header enabling extra
//     ratio tools when they shrink the block: 4-way Huffman on the literal
//     bank, compact (varint) length extras, and offset-0 as "repeat last".
//     Incompressible / already-well-matched L5 blocks keep the original
//     layout and the original decode fast path.
//
// Decode is format-driven (header present => L5 tools; no header => L1
// path), so old L1 streams and new L5 streams both remain valid.
//
// Default stream layout: a run of sequences, each
//
//   token      : u8   bits 7..5 adv = literal length + wide (7 = extended)
//                     bit  4    wide: offset width (0 = 1 byte, 1 = 2 bytes LE)
//                     bits 3..0 match length - 4 (15 = extended)
//   [lit ext]  : 255-run bytes while adv >= 7 (literal length = adv - wide)
//   literals
//   offset     : 1 or 2 bytes (1..65535)
//   [match ext]: 255-run bytes while match length - 4 >= 15
//
// and a final literal-only token (no offset follows, wide = 0; the decoder
// knows it is last because the input is exhausted after the literals). Short
// offsets (<= 255) are the common case on struct/record data (~40% of matches
// on the verdict corpus) and the one-byte form buys ~3.5% ratio over a fixed
// 16-bit offset. The top field stores literal length + wide rather than the
// literal length itself: for a non-extended token the next token is then at
// ip + adv + 1, so the loop-carried input pointer depends on a single field
// extraction (load, shift, add) instead of two fields and two adds.
//
// Optional header (never a valid default sequence: wide offset 0 is rejected):
//
//   0x1F 0x00 0x00 flags
//     bit 0  Huffman literals: varint(lit_n) varint(huff_n) huff[huff_n],
//            then sequences with literal bytes omitted (lengths still in
//            the token). Decoder Huffman-decodes the bank first.
//     bit 1  Compact extras: a 255 extra byte is followed by a LEB128
//            (extra - 255) instead of more 255-run bytes.
//     bit 2  Repeat offset: offset 0 means "same as the previous match".
//            A repeat is stored as a 1-byte 0 (wide = 0).

#ifndef SPECTRA8_SRC_FLASH_HPP
#define SPECTRA8_SRC_FLASH_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "util/common.hpp"
#include "util/huffman.hpp"
#include "util/simd.hpp"

namespace spectra8::detail {

struct FlashStats {
  usize in_len = 0;
  usize out_len = 0;
  usize lit_bytes = 0;
  usize huff_lit_bytes = 0;
  usize token_bytes = 0;
  usize offset_bytes = 0;
  usize ext_bytes = 0;
  usize num_matches = 0;
  usize num_short_offsets = 0;
  usize num_wide_offsets = 0;
  usize num_rep0 = 0;
  usize num_rep1 = 0;
};
inline thread_local FlashStats g_flash_last_stats;
FlashStats get_last_flash_stats() noexcept;

namespace flash_internal {

static constexpr usize kHashLogMax = 14;
static constexpr usize kHashLogMin = 10;
static constexpr usize kTableSize = 1 << kHashLogMax;
static constexpr usize kMaxOffset = 65535;
static constexpr usize kLitBits = 3;
static constexpr usize kLitMax = (1u << kLitBits) - 1;  // 7: extended
static constexpr usize kMlMax = 15;                     // extended
static constexpr u8 kOffWideFlag = 0x10;
// Optional-header sentinel: token 0x1F (0 lits, wide, ml-ext) + offset 0.
static constexpr u8 kHdrTok = 0x1F;
static constexpr u8 kFlagHuffLits = 0x01;
static constexpr u8 kFlagCompactExt = 0x02;
static constexpr u8 kFlagRepOff = 0x04;
static constexpr u8 kFlagMask = 0x07;

struct FlashSeq {
  u32 lit_len;
  u32 match_len;  // 0 = final literal-only token
  u32 offset;
  u8 is_rep;
};

inline u32 hash4(u32 v, u32 shift) noexcept {
  return (v * 2654435761u) >> shift;
}

// Table size follows the input: one slot per ~8 input bytes, 1 Ki..16 Ki
// slots. Each slot is a u64 packing the two most recent positions for that
// hash (low 32 = newest). The table is zeroed per call, so a validity check
// against stale entries is a plain `pos < cur` that never mispredicts;
// sizing the table to the input keeps that memset small for the short
// filtered sub-streams.
inline usize hash_log_for(usize n) noexcept {
  usize lg = kHashLogMin;
  while (lg < kHashLogMax && (usize{8} << lg) < n) ++lg;
  return lg;
}

inline usize varint_len(usize v) noexcept {
  usize n = 1;
  while (v >= 0x80) {
    v >>= 7;
    ++n;
  }
  return n;
}

inline bool put_varint_raw(u8*& op, u8* op_end, usize v) noexcept {
  while (v >= 0x80) {
    if (op >= op_end) return false;
    *op++ = static_cast<u8>(v) | 0x80;
    v >>= 7;
  }
  if (op >= op_end) return false;
  *op++ = static_cast<u8>(v);
  return true;
}

inline bool get_varint_raw(const u8*& ip, const u8* ip_end, usize& v) noexcept {
  v = 0;
  int shift = 0;
  for (;;) {
    if (ip >= ip_end || shift > 63) return false;
    const u8 b = *ip++;
    v |= static_cast<usize>(b & 0x7F) << shift;
    if ((b & 0x80) == 0) return true;
    shift += 7;
  }
}

inline usize ext_bytes_classic(usize extra) noexcept {
  return extra / 255 + 1;
}

inline usize ext_bytes_compact(usize extra) noexcept {
  if (extra < 255) return 1;
  return 1 + varint_len(extra - 255);
}

inline usize match_forward(const u8* ip, const u8* match, const u8* ip_end) noexcept {
  const u8* const start = ip;
  ip += 4;
  match += 4;
  while (ip + 8 <= ip_end) {
    const u64 x = load_u64(ip) ^ load_u64(match);
    if (x != 0) {
#if defined(__GNUC__) || defined(__clang__)
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
      ip += static_cast<usize>(__builtin_ctzll(x) >> 3);
#else
      ip += static_cast<usize>(__builtin_clzll(x) >> 3);
#endif
#else
      while (ip < ip_end && *ip == *match) {
        ++ip;
        ++match;
      }
      return static_cast<usize>(ip - start);
#endif
      return static_cast<usize>(ip - start);
    }
    ip += 8;
    match += 8;
  }
  while (ip < ip_end && *ip == *match) {
    ++ip;
    ++match;
  }
  return static_cast<usize>(ip - start);
}

// Overlapping-match copy for offsets 1..15: after the first 8 bytes the
// source pointer is adjusted so that further 8-byte copies reproduce the
// pattern (same trick as LZ4 / the Beam decoder). Writes exactly up to
// round_up(len, 8) past `op`; the caller guarantees that slack.
static constexpr u32 kInc32[8] = {0, 1, 2, 1, 0, 4, 4, 4};
static constexpr int kDec64[8] = {0, 0, 0, -1, -4, 1, 2, 3};

inline void copy_overlap_wild(u8* op, const u8* m, usize off, usize len) noexcept {
  u8* const end = op + len;
  if (off < 8) {
    op[0] = m[0];
    op[1] = m[1];
    op[2] = m[2];
    op[3] = m[3];
    m += kInc32[off];
    std::memcpy(op + 4, m, 4);
    m -= kDec64[off];
  } else {
    copy8(op, m);
    m += 8;
  }
  op += 8;
  if (end - op > 24) {
    // Long run (zero fill, repeated record): after 16 pattern bytes are in
    // place, step 16 at a time from `op - d2`, d2 the smallest multiple of
    // the period that is >= 16 (so a 16-byte load only reads bytes already
    // written) and <= 16 + off (so it never reads before the match source).
    copy8(op, m);
    op += 8;
    const usize d2 = off * ((16 + off - 1) / off);  // 16..30
    const u8* m2 = op - d2;
    do {
      copy16(op, m2);
      op += 16;
      m2 += 16;
    } while (op < end);
    return;
  }
  while (op < end) {
    copy8(op, m);
    op += 8;
    m += 8;
  }
}

inline void copy_overlap_safe(u8* op, const u8* m, usize len) noexcept {
  for (usize k = 0; k < len; ++k) op[k] = m[k];
}

}  // namespace flash_internal

// Encodes `src[0..src_len)` into `dst[0..max_dst)`.
// `level < 5`: original Flash sequence format (no 0x1F header).
// `level >= 5`: optional Huffman / compact extras / rep-offset header.
// Returns number of bytes written, or 0 if output exceeds max_dst.
inline usize encode_flash_block(const u8* src, usize src_len, u8* dst, usize max_dst,
                                int level = 1) noexcept {
  using namespace flash_internal;
  u8* op = dst;
  u8* const op_end = dst + max_dst;

  usize total_ext_bytes = 0;
  usize num_short_offsets = 0;
  usize num_wide_offsets = 0;

  auto put_ext = [&](usize extra, bool compact) -> bool {
    if (!compact) {
      while (extra >= 255) {
        if (op >= op_end) return false;
        *op++ = 255;
        extra -= 255;
        ++total_ext_bytes;
      }
      if (op >= op_end) return false;
      *op++ = static_cast<u8>(extra);
      ++total_ext_bytes;
      return true;
    }
    if (extra < 255) {
      if (op >= op_end) return false;
      *op++ = static_cast<u8>(extra);
      ++total_ext_bytes;
      return true;
    }
    if (op >= op_end) return false;
    *op++ = 255;
    ++total_ext_bytes;
    const usize before = static_cast<usize>(op - dst);
    if (!put_varint_raw(op, op_end, extra - 255)) return false;
    total_ext_bytes += static_cast<usize>(op - dst) - before;
    return true;
  };

  auto put_tail = [&](const u8* anchor, usize last_lits) -> usize {
    const usize lit_nibble = min_of(last_lits, kLitMax);
    if (op >= op_end) return 0;
    *op++ = static_cast<u8>(lit_nibble << 5);
    if (lit_nibble == kLitMax && !put_ext(last_lits - kLitMax, false)) return 0;
    if (static_cast<usize>(op_end - op) < last_lits) return 0;
    std::memcpy(op, anchor, last_lits);
    op += last_lits;
    return static_cast<usize>(op - dst);
  };

  if (src_len < 13) return put_tail(src, src_len);

  const bool ratio_mode = level >= 5;

  alignas(64) static thread_local u64 table[kTableSize];
  const usize hash_log = hash_log_for(src_len);
  const u32 hash_shift = static_cast<u32>(32 - hash_log);
  std::memset(table, 0, sizeof(u64) << hash_log);

  thread_local std::vector<u8> lit_bank;
  thread_local std::vector<FlashSeq> seqs;
  lit_bank.clear();
  seqs.clear();
  lit_bank.reserve(src_len / 2 + 8);
  seqs.reserve(src_len / 6 + 8);

  const u8* ip = src;
  const u8* anchor = src;
  const u8* const ip_end = src + src_len;
  const u8* const mflimit = ip_end - 8;

  usize step = 1;
  usize search_counter = 1 << 6;

  usize total_lits = 0;
  usize total_matches = 0;
  usize num_rep0 = 0;
  usize num_rep1 = 0;
  u32 last_diff = 0;
  u32 prev_diff = 0;
  u32 lit_freq[256] = {0};

  auto consider = [&](u32 cand, u32 cur, const u8* ip0, u32& best_off, usize& best_ml,
                      const u8*& best_m) {
    if (cand >= cur) return;
    const u32 d = cur - cand;
    if (d == 0 || d > kMaxOffset) return;
    const u8* m = ip0 - d;
    if (load_u32(m) != load_u32(ip0)) return;
    const usize ml = match_forward(ip0, m, ip_end);
    if (ml > best_ml || (ml == best_ml && d < best_off)) {
      best_ml = ml;
      best_off = d;
      best_m = m;
    }
  };

  while (ip < mflimit) {
    const u32 h = hash4(load_u32(ip), hash_shift);
    const u32 cur = static_cast<u32>(ip - src);
    const u64 slot = table[h];
    const u32 pos = static_cast<u32>(slot);
    const u32 pos2 = static_cast<u32>(slot >> 32);
    table[h] = (slot << 32) | cur;

    u32 diff = 0;
    usize match_len = 0;
    const u8* match = nullptr;
    consider(pos, cur, ip, diff, match_len, match);
    consider(pos2, cur, ip, diff, match_len, match);

    if (match_len >= 4) {
      while (ip > anchor && match > src && ip[-1] == match[-1]) {
        --ip;
        --match;
        ++match_len;
      }

      usize lit_len = static_cast<usize>(ip - anchor);
      const u8* const match_start = ip;
      ip += match_len;

      if (match_len <= 32 && match_start + 1 < mflimit) {
        const u8* const ip1 = match_start + 1;
        const u32 h1 = hash4(load_u32(ip1), hash_shift);
        const u32 cur1 = static_cast<u32>(ip1 - src);
        const u64 slot1 = table[h1];
        u32 d1 = 0;
        usize ml1 = 0;
        const u8* m1 = nullptr;
        consider(static_cast<u32>(slot1), cur1, ip1, d1, ml1, m1);
        consider(static_cast<u32>(slot1 >> 32), cur1, ip1, d1, ml1, m1);
        if (ml1 > match_len + 1 ||
            (ml1 >= match_len && d1 <= 255 && diff > 255)) {
          const usize fwd1 = ml1;
          ip = ip1;
          match = m1;
          while (ip > anchor && match > src && ip[-1] == match[-1]) {
            --ip;
            --match;
          }
          diff = d1;
          match_len = fwd1 + static_cast<usize>(ip1 - ip);
          lit_len = static_cast<usize>(ip - anchor);
          ip = ip1 + fwd1;
        }
      }

      if (lit_len) {
        lit_bank.insert(lit_bank.end(), anchor, anchor + lit_len);
        if (ratio_mode) {
          for (usize i = 0; i < lit_len; ++i) ++lit_freq[anchor[i]];
        }
      }
      total_lits += lit_len;
      total_matches += 1;

      const u8 is_rep = (last_diff != 0 && diff == last_diff) ? 1 : 0;
      if (is_rep) {
        ++num_rep0;
      } else if (diff == prev_diff) {
        ++num_rep1;
      }
      prev_diff = last_diff;
      last_diff = diff;
      seqs.push_back(FlashSeq{static_cast<u32>(lit_len), static_cast<u32>(match_len),
                              diff, is_rep});

      anchor = ip;
      if (ip >= mflimit) break;

      const u32 t0 = static_cast<u32>((ip - 3) - src);
      const u32 t1 = static_cast<u32>((ip - 2) - src);
      const u32 t2 = static_cast<u32>((ip - 1) - src);
      auto stamp = [&](const u8* p, u32 posi) {
        const u32 hh = hash4(load_u32(p), hash_shift);
        table[hh] = (table[hh] << 32) | posi;
      };
      stamp(ip - 3, t0);
      stamp(ip - 2, t1);
      stamp(ip - 1, t2);
      step = 1;
      search_counter = 1 << 6;
      continue;
    }

    ip += step;
    if (--search_counter == 0) {
      search_counter = 1 << 6;
      if (step < 8) ++step;
    }
  }

  const usize tail_lits = static_cast<usize>(ip_end - anchor);
  if (tail_lits) {
    lit_bank.insert(lit_bank.end(), anchor, ip_end);
    if (ratio_mode) {
      for (usize i = 0; i < tail_lits; ++i) ++lit_freq[anchor[i]];
    }
  }
  total_lits += tail_lits;
  seqs.push_back(FlashSeq{static_cast<u32>(tail_lits), 0, 0, 0});

  u8 flags = 0;
  const usize huff_margin = max_of(usize{768}, total_lits / 12);
  if (ratio_mode) {
    usize ext_classic = 0;
    usize ext_compact = 0;
    usize wide_rep = 0;
    for (const FlashSeq& s : seqs) {
      if (s.match_len == 0) {
        if (s.lit_len >= kLitMax) {
          const usize extra = s.lit_len - kLitMax;
          ext_classic += ext_bytes_classic(extra);
          ext_compact += ext_bytes_compact(extra);
        }
        continue;
      }
      const bool wide = !s.is_rep && s.offset > 255;
      const usize adv = s.lit_len + (wide ? 1 : 0);
      if (adv >= kLitMax) {
        const usize extra = adv - kLitMax;
        ext_classic += ext_bytes_classic(extra);
        ext_compact += ext_bytes_compact(extra);
      }
      if (s.match_len - 4 >= kMlMax) {
        const usize extra = s.match_len - 4 - kMlMax;
        ext_classic += ext_bytes_classic(extra);
        ext_compact += ext_bytes_compact(extra);
      }
      if (s.is_rep && s.offset > 255) ++wide_rep;
    }

    const usize huff_est =
        total_lits > 32 ? huff::encoded_size(lit_freq, total_lits) : total_lits;
    const usize huff_over = 4 + varint_len(total_lits) + varint_len(huff_est) + huff_est;
    // Huffman the literal bank only when literals dominate the block and the
    // size win covers the extra decode pass (table build + 4-way Huffman).
    // Tiny wins (a few dozen bytes) destroy GB/s decode for almost no ratio.
    // Only Huffman residual planes (almost no matches). Mixing Huffman with
    // thousands of LZ sequences makes decode a two-pass loop and tanks GB/s.
    if (total_matches <= 24 && huff_over + huff_margin < total_lits)
      flags |= kFlagHuffLits;
    if (ext_compact + ((flags ? 0 : 4) + 8) < ext_classic) flags |= kFlagCompactExt;
    if (wide_rep >= 8) flags |= kFlagRepOff;
  }
  const bool compact = (flags & kFlagCompactExt) != 0;
  const bool use_rep = (flags & kFlagRepOff) != 0;
  const bool use_huff = (flags & kFlagHuffLits) != 0;

  thread_local std::vector<u8> huff_buf;
  if (use_huff) {
    huff_buf.clear();
    huff::encode(huff_buf, lit_bank.data(), total_lits, lit_freq);
    const usize actual_over =
        4 + varint_len(total_lits) + varint_len(huff_buf.size()) + huff_buf.size();
    if (actual_over + huff_margin >= total_lits) {
      flags = static_cast<u8>(flags & ~kFlagHuffLits);
    }
  }
  const bool huff_lits = (flags & kFlagHuffLits) != 0;

  op = dst;
  total_ext_bytes = 0;
  num_short_offsets = 0;
  num_wide_offsets = 0;

  if (flags) {
    if (static_cast<usize>(op_end - op) < 4) return 0;
    *op++ = kHdrTok;
    *op++ = 0;
    *op++ = 0;
    *op++ = flags;
  }
  if (huff_lits) {
    if (!put_varint_raw(op, op_end, total_lits)) return 0;
    if (!put_varint_raw(op, op_end, huff_buf.size())) return 0;
    if (static_cast<usize>(op_end - op) < huff_buf.size()) return 0;
    std::memcpy(op, huff_buf.data(), huff_buf.size());
    op += huff_buf.size();
  }

  const u8* lp = lit_bank.data();
  for (const FlashSeq& s : seqs) {
    if (s.match_len == 0) {
      const usize lit_nibble = min_of(static_cast<usize>(s.lit_len), kLitMax);
      if (op >= op_end) return 0;
      *op++ = static_cast<u8>(lit_nibble << 5);
      if (lit_nibble == kLitMax &&
          !put_ext(s.lit_len - kLitMax, compact))
        return 0;
      if (!huff_lits) {
        if (static_cast<usize>(op_end - op) < s.lit_len) return 0;
        std::memcpy(op, lp, s.lit_len);
        op += s.lit_len;
      }
      lp += s.lit_len;
      break;
    }

    const bool is_rep = use_rep && s.is_rep;
    const bool wide = !is_rep && s.offset > 255;
    const usize adv = s.lit_len + (wide ? 1 : 0);
    const usize adv_field = min_of(adv, kLitMax);
    const usize match_nibble = min_of(static_cast<usize>(s.match_len - 4), kMlMax);
    if (op >= op_end) return 0;
    *op++ = static_cast<u8>((adv_field << 5) | (wide ? kOffWideFlag : 0) | match_nibble);
    if (adv_field == kLitMax && !put_ext(adv - kLitMax, compact)) return 0;
    if (!huff_lits) {
      if (static_cast<usize>(op_end - op) < s.lit_len + 2) return 0;
      std::memcpy(op, lp, s.lit_len);
      op += s.lit_len;
    } else if (static_cast<usize>(op_end - op) < 2) {
      return 0;
    }
    lp += s.lit_len;

    if (is_rep) {
      ++num_short_offsets;
      *op++ = 0;
    } else if (wide) {
      ++num_wide_offsets;
      store_u16(op, static_cast<u16>(s.offset));
      op += 2;
    } else {
      ++num_short_offsets;
      *op++ = static_cast<u8>(s.offset);
    }
    if (match_nibble == kMlMax && !put_ext(s.match_len - 4 - kMlMax, compact))
      return 0;
  }

  const usize out_len = static_cast<usize>(op - dst);
  g_flash_last_stats.in_len = src_len;
  g_flash_last_stats.out_len = out_len;
  g_flash_last_stats.lit_bytes = total_lits;
  g_flash_last_stats.huff_lit_bytes = huff_lits ? huff_buf.size() : total_lits;
  g_flash_last_stats.token_bytes = total_matches + 1;
  g_flash_last_stats.offset_bytes = num_wide_offsets * 2 + num_short_offsets;
  g_flash_last_stats.ext_bytes = total_ext_bytes;
  g_flash_last_stats.num_matches = total_matches;
  g_flash_last_stats.num_short_offsets = num_short_offsets;
  g_flash_last_stats.num_wide_offsets = num_wide_offsets;
  g_flash_last_stats.num_rep0 = num_rep0;
  g_flash_last_stats.num_rep1 = num_rep1;
  return out_len;
}

// Decompresses flash payload into dst (which must have capacity at least regen_len).
// Returns Status::Ok on success, Status::CorruptInput on corrupt/truncated input.
inline Status decode_flash_block(const u8* src, usize src_len, u8* dst, usize regen_len) noexcept {
  using namespace flash_internal;
  if (regen_len == 0) {
    // Empty block: nothing, or the encoder's single empty literal token.
    return (src_len == 0 || (src_len == 1 && src[0] == 0)) ? Status::Ok
                                                            : Status::CorruptInput;
  }
  if (src_len == 0) return Status::CorruptInput;

  const u8* ip = src;
  const u8* const ip_end = src + src_len;
  u8* op = dst;
  u8* const op_start = dst;
  u8* const op_end = dst + regen_len;

  u8 flags = 0;
  if (src_len >= 4 && src[0] == kHdrTok && src[1] == 0 && src[2] == 0) {
    flags = src[3];
    if (flags & ~kFlagMask) return Status::CorruptInput;
    ip += 4;
  }
  const bool huff_lits = (flags & kFlagHuffLits) != 0;
  const bool compact = (flags & kFlagCompactExt) != 0;
  const bool use_rep = (flags & kFlagRepOff) != 0;

  thread_local std::vector<u8> lit_bank;
  const u8* litp = nullptr;
  const u8* lit_end = nullptr;
  if (huff_lits) {
    usize lit_n = 0, huff_n = 0;
    if (!get_varint_raw(ip, ip_end, lit_n) || !get_varint_raw(ip, ip_end, huff_n))
      return Status::CorruptInput;
    if (huff_n > static_cast<usize>(ip_end - ip)) return Status::CorruptInput;
    lit_bank.resize(lit_n + 16);
    if (lit_n > 0) {
      if (huff::decode(ip, huff_n, lit_bank.data(), lit_n) != Status::Ok)
        return Status::CorruptInput;
    }
    ip += huff_n;
    litp = lit_bank.data();
    lit_end = litp + lit_n;
  }
  usize last_off = 0;

  if (!huff_lits) {
    constexpr usize kFastIn = 16 + 2;
    constexpr usize kFastOut = 32;
    while (ip < ip_end) {
      const u8 token = *ip++;
      const usize adv = token >> 5;
      usize match_len = (token & 0x0F) + 4;
      const usize wide = (token >> 4) & 1;
      usize lit_len = adv - wide;

      if (adv != kLitMax && adv >= wide && (token & 0x0F) != kMlMax &&
          static_cast<usize>(ip_end - ip) >= kFastIn &&
          static_cast<usize>(op_end - op) >= kFastOut) {
        copy16(op, ip);
        op += lit_len;
        const u32 v = load_u16(ip + lit_len);
        usize offset = wide ? v : (v & 0xFFu);
        ip += adv + 1;
        if (use_rep && offset == 0) offset = last_off;
        if (offset == 0 || offset > static_cast<usize>(op - op_start)) {
          return Status::CorruptInput;
        }
        last_off = offset;
        const u8* m = op - offset;
        if (offset >= 16) {
          copy16(op, m);
          copy8(op + 16, m + 16);
        } else {
          copy_overlap_wild(op, m, offset, match_len);
        }
        op += match_len;
        continue;
      }

      if (adv == kLitMax) {
        if (compact) {
          if (ip >= ip_end) return Status::CorruptInput;
          const u8 b = *ip++;
          if (b != 255) {
            lit_len += b;
          } else {
            usize extra = 0;
            if (!get_varint_raw(ip, ip_end, extra)) return Status::CorruptInput;
            lit_len += 255 + extra;
          }
        } else {
          u8 b;
          do {
            if (ip >= ip_end) return Status::CorruptInput;
            b = *ip++;
            lit_len += b;
          } while (b == 255);
        }
      } else if (adv < wide) {
        return Status::CorruptInput;
      }

      if (static_cast<usize>(op_end - op) < lit_len ||
          static_cast<usize>(ip_end - ip) < lit_len) {
        return Status::CorruptInput;
      }

      if (static_cast<usize>(op_end - op) >= lit_len + 16 &&
          static_cast<usize>(ip_end - ip) >= lit_len + 16) {
        u8* const lend = op + lit_len;
        const u8* const slit = ip;
        do {
          copy16(op, ip);
          op += 16;
          ip += 16;
        } while (op < lend);
        op = lend;
        ip = slit + lit_len;
      } else {
        std::memcpy(op, ip, lit_len);
        op += lit_len;
        ip += lit_len;
      }

      if (ip == ip_end) {
        return op == op_end ? Status::Ok : Status::CorruptInput;
      }

      if (static_cast<usize>(ip_end - ip) < 2) return Status::CorruptInput;
      const u32 v = load_u16(ip);
      usize offset = wide ? v : (v & 0xFFu);
      ip += 1 + wide;
      if (use_rep && offset == 0) offset = last_off;

      if (offset == 0 || offset > static_cast<usize>(op - op_start)) {
        return Status::CorruptInput;
      }
      last_off = offset;

      if ((token & 0x0F) == kMlMax) {
        if (compact) {
          if (ip >= ip_end) return Status::CorruptInput;
          const u8 b = *ip++;
          if (b != 255) {
            match_len += b;
          } else {
            usize extra = 0;
            if (!get_varint_raw(ip, ip_end, extra)) return Status::CorruptInput;
            match_len += 255 + extra;
          }
        } else {
          u8 b;
          do {
            if (ip >= ip_end) return Status::CorruptInput;
            b = *ip++;
            match_len += b;
          } while (b == 255);
        }
      }

      if (static_cast<usize>(op_end - op) < match_len) {
        return Status::CorruptInput;
      }

      const u8* m = op - offset;
      if (static_cast<usize>(op_end - op) >= match_len + 16) {
        if (offset >= 16) {
          u8* const mend = op + match_len;
          do {
            copy16(op, m);
            op += 16;
            m += 16;
          } while (op < mend);
          op = mend;
        } else {
          copy_overlap_wild(op, m, offset, match_len);
          op += match_len;
        }
      } else {
        copy_overlap_safe(op, m, match_len);
        op += match_len;
      }
    }
    return op == op_end ? Status::Ok : Status::CorruptInput;
  }

  auto read_ext = [&](usize& len) -> Status {
    if (!compact) {
      u8 b;
      do {
        if (ip >= ip_end) return Status::CorruptInput;
        b = *ip++;
        len += b;
      } while (b == 255);
      return Status::Ok;
    }
    if (ip >= ip_end) return Status::CorruptInput;
    const u8 b = *ip++;
    if (b != 255) {
      len += b;
      return Status::Ok;
    }
    usize extra = 0;
    if (!get_varint_raw(ip, ip_end, extra)) return Status::CorruptInput;
    len += 255 + extra;
    return Status::Ok;
  };

  // Fast path bounds: a short sequence (<= 6 literals, <= 18 match bytes)
  // reads at most 16 + 2 input bytes (or 2 offset bytes when literals live
  // in the Huffman bank) and writes at most 6 + 24 output bytes.
  const usize kFastIn = huff_lits ? 2 : 16 + 2;
  constexpr usize kFastOut = 32;

  while (ip < ip_end) {
    const u8 token = *ip++;
    const usize adv = token >> 5;
    usize match_len = (token & 0x0F) + 4;
    const usize wide = (token >> 4) & 1;
    usize lit_len = adv - wide;  // adv < wide is rejected below

    if (adv != kLitMax && adv >= wide && (token & 0x0F) != kMlMax &&
        static_cast<usize>(ip_end - ip) >= kFastIn &&
        static_cast<usize>(op_end - op) >= kFastOut &&
        (!huff_lits || static_cast<usize>(lit_end - litp) >= lit_len + 16)) {
      const u8* lsrc = huff_lits ? litp : ip;
      copy16(op, lsrc);
      op += lit_len;
      if (huff_lits) {
        litp += lit_len;
        const u32 v = load_u16(ip);
        usize offset = wide ? v : (v & 0xFFu);
        ip += 1 + wide;
        if (use_rep && offset == 0) offset = last_off;
        if (offset == 0 || offset > static_cast<usize>(op - op_start)) {
          return Status::CorruptInput;
        }
        last_off = offset;
        const u8* m = op - offset;
        if (offset >= 16) {
          copy16(op, m);
          copy8(op + 16, m + 16);
        } else {
          copy_overlap_wild(op, m, offset, match_len);
        }
        op += match_len;
        continue;
      }
      const u32 v = load_u16(ip + lit_len);
      usize offset = wide ? v : (v & 0xFFu);
      ip += adv + 1;
      if (use_rep && offset == 0) offset = last_off;
      if (offset == 0 || offset > static_cast<usize>(op - op_start)) {
        return Status::CorruptInput;
      }
      last_off = offset;
      const u8* m = op - offset;
      if (offset >= 16) {
        copy16(op, m);
        copy8(op + 16, m + 16);
      } else {
        copy_overlap_wild(op, m, offset, match_len);
      }
      op += match_len;
      continue;
    }

    if (adv == kLitMax) {
      if (Status st = read_ext(lit_len); st != Status::Ok) return st;
    } else if (adv < wide) {
      return Status::CorruptInput;
    }

    if (static_cast<usize>(op_end - op) < lit_len) return Status::CorruptInput;
    if (huff_lits) {
      if (static_cast<usize>(lit_end - litp) < lit_len) return Status::CorruptInput;
      if (static_cast<usize>(op_end - op) >= lit_len + 16 &&
          static_cast<usize>(lit_end - litp) >= lit_len + 16) {
        u8* const lend = op + lit_len;
        const u8* const slit = litp;
        do {
          copy16(op, litp);
          op += 16;
          litp += 16;
        } while (op < lend);
        op = lend;
        litp = slit + lit_len;
      } else {
        std::memcpy(op, litp, lit_len);
        op += lit_len;
        litp += lit_len;
      }
    } else {
      if (static_cast<usize>(ip_end - ip) < lit_len) return Status::CorruptInput;
      if (static_cast<usize>(op_end - op) >= lit_len + 16 &&
          static_cast<usize>(ip_end - ip) >= lit_len + 16) {
        u8* const lend = op + lit_len;
        const u8* const slit = ip;
        do {
          copy16(op, ip);
          op += 16;
          ip += 16;
        } while (op < lend);
        op = lend;
        ip = slit + lit_len;
      } else {
        std::memcpy(op, ip, lit_len);
        op += lit_len;
        ip += lit_len;
      }
    }

    if (ip == ip_end) {
      if (huff_lits && litp != lit_end) return Status::CorruptInput;
      return op == op_end ? Status::Ok : Status::CorruptInput;
    }

    if (static_cast<usize>(ip_end - ip) < 2) return Status::CorruptInput;
    const u32 v = load_u16(ip);
    usize offset = wide ? v : (v & 0xFFu);
    ip += 1 + wide;
    if (use_rep && offset == 0) offset = last_off;

    if (offset == 0 || offset > static_cast<usize>(op - op_start)) {
      return Status::CorruptInput;
    }
    last_off = offset;

    if ((token & 0x0F) == kMlMax) {
      if (Status st = read_ext(match_len); st != Status::Ok) return st;
    }

    if (static_cast<usize>(op_end - op) < match_len) {
      return Status::CorruptInput;
    }

    const u8* m = op - offset;
    if (static_cast<usize>(op_end - op) >= match_len + 16) {
      if (offset >= 16) {
        u8* const mend = op + match_len;
        do {
          copy16(op, m);
          op += 16;
          m += 16;
        } while (op < mend);
        op = mend;
      } else {
        copy_overlap_wild(op, m, offset, match_len);
        op += match_len;
      }
    } else {
      copy_overlap_safe(op, m, match_len);
      op += match_len;
    }
  }

  if (huff_lits && litp != lit_end) return Status::CorruptInput;
  return op == op_end ? Status::Ok : Status::CorruptInput;
}

inline void build_flash_block(const u8* src, usize n, std::vector<u8>& payload,
                              int level = 1) {
  payload.resize(n + 80);
  const usize comp_len =
      encode_flash_block(src, n, payload.data(), payload.size(), level);
  if (comp_len == 0 || comp_len >= n) {
    payload.clear();
  } else {
    payload.resize(comp_len);
  }
}

}  // namespace spectra8::detail

#endif  // SPECTRA8_SRC_FLASH_HPP
