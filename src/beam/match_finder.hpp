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

// Hash-chain match finder operating within a single block.
#ifndef SPECTRA8_SRC_MATCH_FINDER_HPP
#define SPECTRA8_SRC_MATCH_FINDER_HPP

#include <algorithm>
#include <vector>

#include "util/common.hpp"
#include "beam/lz.hpp"

#if defined(__aarch64__)
#include <arm_neon.h>
#elif defined(__SSE2__)
#include <emmintrin.h>
#endif

namespace spectra8::detail {

struct MfParams {
  int hash_log;
  int max_chain;   // chain steps, or tree depth budget when use_bt is set
  int nice_len;
  int lazy;
  int hash_bytes = 4;  // bytes hashed per chain entry (4 for small blocks,
                       // 5/6 for large ones: popular 4-grams otherwise create
                       // huge collision chains that max_chain burns down on)
  bool use_bt = false;  // binary-tree finder (large blocks at high levels):
                        // each hash bucket is a tree of positions ordered by
                        // suffix, so a comparison discards a whole subtree
                        // instead of a single chain candidate
  bool use_row = false;  // row-based tag finder (zstd row-hash style): the
                         // hash selects a small row of recent positions plus
                         // an 8-bit tag; one SIMD compare over the row's tags
                         // filters candidates with no pointer chasing
  int row_log = 4;       // log2 slots per row (4 => 16, 5 => 32, 6 => 64)
  // When false, short matches use the same stride as long ones (L5).
  bool dense_short_insert = true;
};

MfParams params_for_level(int level, usize block_size);

inline u32 hash4(u32 x, int bits) noexcept {
  return (x * 2654435761u) >> (32 - bits);
}

// Multiplicative hash over the low `bytes` (5 or 6) of a 64-bit load.
inline u32 hash_bytes_n(u64 x, int bytes, int bits) noexcept {
  x &= (u64{1} << (bytes * 8)) - 1;
  return static_cast<u32>((x * 0x9E3779B185EBCA87ull) >> (64 - bits));
}

// Bitmask of the 16 lanes of `eq` whose byte is 0xFF (result of a byte
// compare): bit i set iff tags[i] matched.
inline u32 tag_mask16(const u8* tags, u8 tag) noexcept {
#if defined(__aarch64__)
  static const u8 kBits[16] = {1, 2, 4, 8, 16, 32, 64, 128,
                               1, 2, 4, 8, 16, 32, 64, 128};
  const uint8x16_t eq = vceqq_u8(vld1q_u8(tags), vdupq_n_u8(tag));
  const uint8x16_t v = vandq_u8(eq, vld1q_u8(kBits));
  return static_cast<u32>(vaddv_u8(vget_low_u8(v))) |
         (static_cast<u32>(vaddv_u8(vget_high_u8(v))) << 8);
#elif defined(__SSE2__)
  const __m128i eq = _mm_cmpeq_epi8(
      _mm_loadu_si128(reinterpret_cast<const __m128i*>(tags)),
      _mm_set1_epi8(static_cast<char>(tag)));
  return static_cast<u32>(_mm_movemask_epi8(eq));
#else
  u32 m = 0;
  for (int i = 0; i < 16; ++i)
    if (tags[i] == tag) m |= u32{1} << i;
  return m;
#endif
}

// Rotate the low `e` bits of `m` right by `r` (0 <= r < e, e a power of two).
inline u64 ror_bits(u64 m, u32 r, u32 e) noexcept {
  if (r == 0) return m;
  const u64 full = (e >= 64) ? ~u64{0} : ((u64{1} << e) - 1);
  return ((m >> r) | (m << (e - r))) & full;
}

class MatchFinder {
 public:
  void init(const u8* base, usize size, MfParams p) {
    base_ = base;
    size_ = size;
    params_ = p;
    // Only the hash heads must be reset each block: chains start at head_[h]
    // (set on insert) and follow prev_[cand] (or the two tree_ child slots),
    // which are always written when `cand` is inserted. So neither prev_ nor
    // tree_ is ever read before being written this block, and their previous
    // contents are irrelevant - no need to clear them. All buffers are
    // grown-only and reused across blocks to avoid per-block (re)allocation
    // and an O(size) memset.
    if (!p.use_row) {
      const usize hsize = usize{1} << p.hash_log;
      if (head_.size() < hsize) head_.resize(hsize);
      if (p.use_bt) {
        // BT walks stale head_ links; those must be -1 at block start.
        std::fill_n(head_.begin(), hsize, i32{-1});
      } else {
        bump_epoch(chain_epoch_, head_epoch_, hsize);
      }
    }
    if (p.use_bt) {
      if (tree_.size() < 2 * size) tree_.resize(2 * size);
      bt_next_insert_ = 0;
      bt_cached_pos_ = ~usize{0};
      // Speculative child prefetch only pays when tree nodes actually miss:
      // data + tree working set past L2 (~4 MB here). On cache-resident
      // blocks the extra prefetch instructions are pure overhead in the
      // walk's serial dependency chain.
      bt_prefetch_ = size * (2 * sizeof(i32) + 1) > (usize{4} << 20);
    } else if (p.use_row) {
      // Epoch stamps replace the O(entries) idx fill: a slot is live only
      // when its stamp matches this block. At L5/1 MiB that skips a 4 MiB
      // memset every block; stale row_head_ is fine (inserts overwrite).
      const usize entries = usize{1} << p.hash_log;
      const usize rows = entries >> p.row_log;
      if (row_tags_.size() < entries) row_tags_.resize(entries);
      if (row_idx_.size() < entries) row_idx_.resize(entries);
      if (row_head_.size() < rows) row_head_.resize(rows);
      bump_epoch(row_gen_, row_epoch_, entries);
      row_memo_pos_ = ~usize{0};
    } else {
      if (prev_.size() < size) prev_.resize(size);
    }
  }

  // Hash key for position `pos`. Reads 8 bytes when a long hash is active, so
  // the last few positions fall back to the 4-byte hash; insert and find use
  // the same rule, and every candidate is verified with match_len anyway, so
  // the mixed keys near the block end only cost a few tail candidates.
  u32 hash_at(usize pos) const noexcept {
    if (params_.hash_bytes > 4 && pos + 8 <= size_)
      return hash_bytes_n(load_u64(base_ + pos), params_.hash_bytes,
                          params_.hash_log);
    return hash4(load_u32(base_ + pos), params_.hash_log);
  }

  void insert(usize pos) noexcept {
    if (params_.use_bt) {
      bt_insert(pos);
      return;
    }
    if (params_.use_row) {
      row_insert(pos);
      return;
    }
    if (pos + 4 > size_) return;
    u32 h = hash_at(pos);
    prev_[pos] = (head_epoch_[h] == chain_epoch_) ? head_[h] : i32{-1};
    head_[h] = static_cast<i32>(pos);
    head_epoch_[h] = chain_epoch_;
  }

  // Insert match-interior positions sparsely (zstd-style): keep the first
  // few and the end dense, stride through the long interior. `start` itself
  // must already have been inserted by the caller. Never inserts a position
  // twice (chain insert is not idempotent).
  void insert_match(usize start, usize len) noexcept {
    if (len <= 1) return;
    const usize end = start + len;
    // Short matches: insert everything at ratio levels (these dominate
    // count). L5 uses the same stride as long matches: 4-7 byte hits are
    // most of the parse and the extra inserts rarely change the next 8
    // chain/row probes.
    if (len <= 8 && params_.dense_short_insert) {
      for (usize k = start + 1; k < end; ++k) insert(k);
      return;
    }
    insert(start + 1);
    if (len == 2) return;  // start + 2 is past the match (and would be inserted twice)
    insert(start + 2);
    const usize tail0 = end - 2;
    const usize tail1 = end - 1;
    constexpr usize kStride = 3;
    for (usize k = start + 3; k < tail0; k += kStride) insert(k);
    if (tail0 > start + 2) insert(tail0);
    if (tail1 > start + 2) insert(tail1);
  }

  // Find the longest non-repeat match for `pos`. Sets best_len/best_off
  // (best_off is the distance, >= 1). best_len is 0 if nothing found.
  void find(usize pos, int& best_len, i32& best_off) noexcept {
    best_len = 0;
    best_off = 0;
    if (params_.use_bt) {
      bt_query(pos);
      if (bt_cached_count_ > 0) {
        best_len = bt_cache_len_[bt_cached_count_ - 1];
        best_off = bt_cache_off_[bt_cached_count_ - 1];
      }
      return;
    }
    if (params_.use_row) {
      int l = 0;
      i32 o = 0;
      if (row_search(pos, &l, &o, 1) > 0) {
        best_len = l;
        best_off = o;
      }
      return;
    }
    if (pos + 4 > size_) return;
    u32 h = hash_at(pos);
    i32 cand = (head_epoch_[h] == chain_epoch_) ? head_[h] : i32{-1};
    int chain = params_.max_chain;
    const u8* p = base_ + pos;
    const u8* end = base_ + size_;
    while (cand >= 0 && chain-- > 0) {
      // The walk is a serial pointer chase (cand -> prev_[cand]) with two
      // random accesses per step; on blocks larger than L2 every step is a
      // cache miss. Load the next link first and prefetch its two cache lines
      // so the following iteration's misses overlap with this candidate's
      // compare (the +best_len line may be slightly stale if best_len grows,
      // but cache-line granularity keeps it useful).
#if defined(__x86_64__) || defined(_M_X64)
      const i32 next = prev_[cand];
      if (next >= 0) {
        __builtin_prefetch(base_ + next + best_len);
        __builtin_prefetch(&prev_[next]);
      }
#else
      const i32 next = prev_[cand];
#endif
      const u8* q = base_ + cand;
      // Hash collisions fail the first 4 bytes; skip the full compare.
      if (load_u32(q) != load_u32(p)) {
        cand = next;
        continue;
      }
      // Short-circuit: to beat best_len a candidate must at least match the byte
      // at index best_len, so probe it before the full (vectorized) compare.
      // (When best_len == 0 the probe is skipped via short-circuit evaluation.)
      if (best_len == 0 || (p + best_len < end && q[best_len] == p[best_len])) {
        usize l = match_len_from4(p, q, end);
        if (static_cast<int>(l) > best_len) {
          best_len = static_cast<int>(l);
          best_off = static_cast<i32>(pos - static_cast<usize>(cand));
          if (static_cast<int>(l) >= params_.nice_len) break;
        }
      }
      cand = next;
    }
  }

  // Collect candidate matches for `pos` into out_len/out_off, one entry per
  // strictly-increasing length found while walking the chain (so each entry is
  // a distinct offset achieving a new best length). Returns the number written
  // (<= max_out); the longest match is always among them. Used by the optimal
  // parser, which prices several (length, offset) choices per position.
  int find_all(usize pos, int* out_len, i32* out_off, int max_out) noexcept {
    if (max_out <= 0) return 0;
    if (params_.use_bt) {
      bt_query(pos);
      const int cnt = bt_cached_count_;
      if (cnt <= max_out) {
        for (int i = 0; i < cnt; ++i) {
          out_len[i] = bt_cache_len_[i];
          out_off[i] = bt_cache_off_[i];
        }
        return cnt;
      }
      // Overflow: keep the shortest candidates first and the longest (last
      // recorded) in the final slot, matching the chain finder's behavior.
      for (int i = 0; i < max_out - 1; ++i) {
        out_len[i] = bt_cache_len_[i];
        out_off[i] = bt_cache_off_[i];
      }
      out_len[max_out - 1] = bt_cache_len_[cnt - 1];
      out_off[max_out - 1] = bt_cache_off_[cnt - 1];
      return max_out;
    }
    if (params_.use_row) return row_search(pos, out_len, out_off, max_out);
    int count = 0;
    int best_len = 0;
    if (pos + 4 > size_) return 0;
    u32 h = hash_at(pos);
    i32 cand = (head_epoch_[h] == chain_epoch_) ? head_[h] : i32{-1};
    int chain = params_.max_chain;
    const u8* p = base_ + pos;
    const u8* end = base_ + size_;
    while (cand >= 0 && chain-- > 0) {
      // Same next-link prefetch as find(): overlap the following iteration's
      // cache misses with this candidate's compare.
#if defined(__x86_64__) || defined(_M_X64)
      const i32 next = prev_[cand];
      if (next >= 0) {
        __builtin_prefetch(base_ + next + best_len);
        __builtin_prefetch(&prev_[next]);
      }
#else
      const i32 next = prev_[cand];
#endif
      const u8* q = base_ + cand;
      if (load_u32(q) != load_u32(p)) {
        cand = next;
        continue;
      }
      if (best_len == 0 || (p + best_len < end && q[best_len] == p[best_len])) {
        usize l = match_len_from4(p, q, end);
        if (static_cast<int>(l) > best_len) {
          best_len = static_cast<int>(l);
          i32 off = static_cast<i32>(pos - static_cast<usize>(cand));
          if (count < max_out) {
            out_len[count] = best_len;
            out_off[count] = off;
            ++count;
          } else {
            // Keep the longest candidate in the final slot.
            out_len[max_out - 1] = best_len;
            out_off[max_out - 1] = off;
          }
          if (best_len >= params_.nice_len) break;
        }
      }
      cand = next;
    }
    return count;
  }

  const MfParams& params() const noexcept { return params_; }

 private:
  // ---------------------------------------------------------------------
  // Row-based tag finder (zstd row-hash family). The hash table is an array
  // of rows of 2^row_log slots; a wide hash yields the row index plus an
  // 8-bit tag stored next to each slot's position. A search does one SIMD
  // compare over the row's tags, turning the survivors into a bitmask - so
  // filtering candidates costs no pointer chasing and touches exactly two
  // cache lines per row, versus ~2 dependent misses per candidate for the
  // chain finder. Rows are circular buffers: insertion moves the row head
  // backwards, and reading slots head, head+1, ... visits candidates newest
  // to oldest (the same order a chain walk yields).
  // ---------------------------------------------------------------------

  // Row index and tag for position `pos`. Uses the same byte width rules as
  // hash_at; the tag takes the 8 hash bits just below the row index bits.
  //
  // The tag hashes one byte more than the row does. Candidates in a row share
  // the row's hash_bytes-gram; on text most of them are the same common
  // 4-gram ("the ", " and") continuing differently, and with a tag drawn from
  // the same bytes every one of them survives the tag filter and costs a
  // random probe into the block (dickens at L5: 34 probes per search, ~3 of
  // which were real matches). Taking the tag from hash_bytes + 1 bytes makes
  // the filter select candidates that agree on the next byte too, so the
  // probes mostly hit matches of >= hash_bytes + 1. Matches of exactly
  // hash_bytes bytes have a different tag and are found by a short fallback
  // probe over the newest entries (row_search); such matches only pay off at
  // small offsets anyway, and the newest entries are the smallest offsets.
  //
  // In the last hash_bytes positions of the block the tag falls back to the
  // row hash; a search there can only miss.
  void row_hash_at(usize pos, u32& row, u8& tag) const noexcept {
    const int hb = params_.hash_bytes;
    u64 x;
    if (hb > 4 && pos + 8 <= size_)
      x = load_u64(base_ + pos) & ((u64{1} << (hb * 8)) - 1);
    else
      x = load_u32(base_ + pos);
    const u64 h = x * 0x9E3779B185EBCA87ull;
    const int rbits = params_.hash_log - params_.row_log;
    row = static_cast<u32>(h >> (64 - rbits));
    if (pos + hb < size_) {
      // One more byte: fold it into a second hash for the tag.
      const u64 x2 = x ^ (static_cast<u64>(base_[pos + hb]) << 56);
      tag = static_cast<u8>((x2 * 0xC2B2AE3D27D4EB4Full) >> 56);
    } else {
      tag = static_cast<u8>(h >> (64 - rbits - 8));
    }
  }

  // Newest row entries probed for an exact hash_bytes-long match when the
  // tag pass found nothing longer (see row_hash_at).
  static constexpr int kRowShortProbes = 4;

  void row_insert(usize pos) noexcept {
    if (pos + 4 > size_) return;
    u32 row;
    u8 tag;
    // The lazy parser inserts a position right after searching it; reuse that
    // search's hash (and still-hot row lines) instead of rehashing.
    if (pos == row_memo_pos_) {
      row = row_memo_row_;
      tag = row_memo_tag_;
    } else {
      row_hash_at(pos, row, tag);
    }
    const u32 mask = (u32{1} << params_.row_log) - 1;
    const usize start = static_cast<usize>(row) << params_.row_log;
    const u32 head = (row_head_[row] - 1u) & mask;
    row_head_[row] = static_cast<u8>(head);
    row_tags_[start + head] = tag;
    row_idx_[start + head] = static_cast<i32>(pos);
    row_epoch_[start + head] = row_gen_;
  }

  // Search with find_all semantics: record each strictly-improving
  // (length, offset) into out_len/out_off (longest kept in the last slot on
  // overflow); returns the count. max_out == 1 degenerates to find().
  int row_search(usize pos, int* out_len, i32* out_off, int max_out) noexcept {
    if (pos + 4 > size_) return 0;
    u32 row;
    u8 tag;
    row_hash_at(pos, row, tag);
    row_memo_pos_ = pos;
    row_memo_row_ = row;
    row_memo_tag_ = tag;
    const u32 e = u32{1} << params_.row_log;
    const u32 mask = e - 1;
    const usize start = static_cast<usize>(row) << params_.row_log;
    const u8* tags = row_tags_.data() + start;
    const i32* idx = row_idx_.data() + start;
    const u32 head = row_head_[row] & mask;

    // Tag bitmask over the row, rotated so bit 0 is the newest entry and set
    // bits enumerate candidates newest-first.
    u64 m = 0;
    for (u32 c = 0; c < e; c += 16)
      m |= static_cast<u64>(tag_mask16(tags + c, tag)) << c;
    m = ror_bits(m, head, e);

    // Verify candidates inline, extracting (and prefetching) one candidate
    // ahead: the next candidate's data loads overlap the current compare, but
    // a nice_len break on the first candidate - the common case inside runs -
    // wastes at most one extraction instead of a whole row's worth.
    auto next_valid = [&]() -> i32 {
      while (m != 0) {
        const u32 j = static_cast<u32>(__builtin_ctzll(m));
        m &= m - 1;
        const u32 slot = (head + j) & mask;
        if (row_epoch_[start + slot] != row_gen_) continue;
        const i32 c = idx[slot];
        if (c >= 0 && static_cast<usize>(c) < pos) return c;
      }
      return -1;
    };

    // The parse is a serial chain (the next search position depends on this
    // search's result), so every cache miss on the row is paid in full. The
    // next search is most often at pos+1: prefetch that row's head, tags and
    // first index line now, so they are in L1 when it comes.
    if (pos + 1 + 8 <= size_) {
      u32 nrow;
      u8 ntag;
      row_hash_at(pos + 1, nrow, ntag);
      const usize nstart = static_cast<usize>(nrow) << params_.row_log;
      __builtin_prefetch(&row_head_[nrow]);
      __builtin_prefetch(row_tags_.data() + nstart);
      __builtin_prefetch(row_idx_.data() + nstart + (row_head_[nrow] & mask));
    }

    int count = 0;
    int best_len = 0;
    int budget = min_of(params_.max_chain, static_cast<int>(e));
    const u8* p = base_ + pos;
    const u8* end = base_ + size_;
    i32 cand = next_valid();
    while (cand >= 0 && budget-- > 0) {
      const i32 nxt = next_valid();
      if (nxt >= 0) __builtin_prefetch(base_ + nxt);
      const u8* q = base_ + cand;
      if (best_len == 0 || (p + best_len < end && q[best_len] == p[best_len])) {
        usize l = match_len(p, q, end);
        if (static_cast<int>(l) > best_len) {
          best_len = static_cast<int>(l);
          const i32 off = static_cast<i32>(pos - static_cast<usize>(cand));
          if (count < max_out) {
            out_len[count] = best_len;
            out_off[count] = off;
            ++count;
          } else {
            out_len[max_out - 1] = best_len;
            out_off[max_out - 1] = off;
          }
          if (best_len >= params_.nice_len) break;
        }
      }
      cand = nxt;
    }

    // Nothing of >= hash_bytes + 1: look for a hash_bytes-long match among
    // the newest entries regardless of tag (a row collision fails the
    // compare, an equal gram is a match).
    if (best_len == 0 && params_.hash_bytes == 4 && pos + 8 <= size_) {
      const u32 want = load_u32(p);
      for (u32 j = 0; j < static_cast<u32>(kRowShortProbes) && j < e; ++j) {
        const u32 slot = (head + j) & mask;
        if (row_epoch_[start + slot] != row_gen_) continue;
        const i32 c = idx[slot];
        if (c < 0 || static_cast<usize>(c) >= pos) continue;
        if (load_u32(base_ + c) == want) {
          const usize l = match_len(p, base_ + c, end);
          out_len[0] = static_cast<int>(l);
          out_off[0] = static_cast<i32>(pos - static_cast<usize>(c));
          return 1;
        }
      }
    }
    return count;
  }

  // ---------------------------------------------------------------------
  // Binary-tree finder (zstd DUBT / btlazy2 family). tree_[2*p] / tree_[2*p+1]
  // are p's smaller / larger child (positions whose suffix compares below /
  // above p's). The most recently sorted position is the subtree root, so a
  // search visits candidates from nearest to farthest while each suffix
  // comparison discards a subtree.
  //
  // Insertion is deferred (zstd's "deferred update binary tree"): insert()
  // just prepends the position to a per-bucket unsorted list in O(1) - vital
  // because parsers used to insert every match-interior position (now sparse
  // via insert_match), and paying a tree walk for each made the BT slower
  // than hash chains on long-match data (xml/nci). head_[h] points at the
  // newest entry; the unsorted prefix of the list (marked kBtUnsorted in the
  // smaller-child slot) is batch-sorted into the tree only when a query
  // actually hits that bucket.
  //
  // Search and root insertion share one walk, so find()/find_all() insert
  // their position as a side effect; a later insert() for it is skipped via
  // bt_next_insert_. The parsers may query the same position twice before
  // moving on (the lazy-shift path), so the last result is cached.
  // ---------------------------------------------------------------------

  static constexpr int kBtMaxCands = 16;
  static constexpr i32 kBtUnsorted = -2;   // "not yet in the tree" marker
  static constexpr int kBtMaxPending = 256;

  // Batch-sort the bucket's unsorted prefix into its tree; returns the root.
  i32 bt_sort_pending(u32 h) noexcept {
    i32 cur = head_[h];
    i32 pend[kBtMaxPending];
    int npend = 0;
    const int pend_cap = params_.max_chain < kBtMaxPending ? params_.max_chain
                                                           : kBtMaxPending;
    while (cur >= 0 && tree_[2 * static_cast<usize>(cur)] == kBtUnsorted &&
           npend < pend_cap) {
      pend[npend++] = cur;
      cur = tree_[2 * static_cast<usize>(cur) + 1];
    }
    if (cur >= 0 && tree_[2 * static_cast<usize>(cur)] == kBtUnsorted) {
      // Over budget: make the cut node a childless sorted node and drop every
      // older entry (zstd does the same). The walk's skipped-byte-compare
      // trick assumes correct BST order for everything reachable, so the tail
      // must become unreachable - keeping a stale pointer to it would report
      // overlong matches and corrupt output.
      tree_[2 * static_cast<usize>(cur)] = -1;
      tree_[2 * static_cast<usize>(cur) + 1] = -1;
    }
    // `cur` is the sorted root (or terminator). Insert pending entries oldest
    // first so newer positions - the cheaper offsets - end up nearer the root.
    // The whole batch shares one positioning budget, measured in *bytes
    // compared* rather than nodes visited: on self-similar data every
    // positioning compare scans ~nice_len near-identical bytes, so a node
    // budget still allowed one sort to cost pending * depth * nice_len byte
    // compares (the BT was ~10x slower than chains on run-heavy blocks,
    // ~2000x zstd on the worst input). When the budget runs dry the subtree
    // assembled so far is dropped and each remaining entry supersedes the
    // last as a childless root, leaving the newest - the cheapest offset -
    // reachable. Text-like data (short compares) rarely hits the cap.
    i32 root = cur;
    int budget = 32 * params_.max_chain;  // bytes
    for (int i = npend - 1; i >= 0; --i) {
      if (budget <= 0) root = -1;  // drop the tree built so far, keep newest
      budget -= bt_walk(static_cast<usize>(pend[i]), root, /*record=*/false,
                        params_.max_chain);
      root = pend[i];
    }
    return root;
  }

  // Insert `pos` on top of the tree rooted at `root` and (when `record`)
  // collect each strictly-improving (length, offset) candidate met on the
  // walk into the cache arrays. `pos` becomes the new root; the caller
  // updates head_ if this bucket's list head should move. Descends at most
  // `depth` nodes. Returns the number of recorded candidates when `record`,
  // otherwise the number of bytes compared (the sort budget consumed).
  int bt_walk(usize pos, i32 root, bool record, int depth) noexcept {
    i32 cand = root;
    i32* smaller = &tree_[2 * pos];
    i32* larger = &tree_[2 * pos + 1];
    usize common_lo = 0;  // guaranteed common prefix with the smaller side
    usize common_hi = 0;  // ... and with the larger side
    int bytes = 0;
    int count = 0;
    int best = 0;
    const u8* p = base_ + pos;
    const u8* end = base_ + size_;
    // Positioning compares (insertion walks) never need to look past
    // nice_len: reaching it drops the subtree below. Capping the scan there
    // keeps run-heavy data from re-scanning a multi-KB run once per pending
    // insert. Record walks scan fully - the parser wants the true length -
    // but they break on the first nice_len hit, so at most one long scan.
    const u8* scan_end = end;
    if (!record && p + params_.nice_len < end) scan_end = p + params_.nice_len;
    while (cand >= 0 && depth-- > 0) {
      // The descent is a serial chain: compare -> pick child -> load child.
      // Both children sit in one cache line of tree_, so load them now and
      // prefetch both possible next candidates' data and tree cells; the
      // misses overlap this node's suffix compare, whichever side wins.
      const i32 kid_lo = tree_[2 * static_cast<usize>(cand)];
      const i32 kid_hi = tree_[2 * static_cast<usize>(cand) + 1];
      const usize pml = min_of(common_lo, common_hi);
      if (bt_prefetch_) {  // only when the block outgrows cache (see init)
        if (kid_lo >= 0) {
          __builtin_prefetch(base_ + static_cast<usize>(kid_lo) + pml);
          __builtin_prefetch(&tree_[2 * static_cast<usize>(kid_lo)]);
        }
        if (kid_hi >= 0) {
          __builtin_prefetch(base_ + static_cast<usize>(kid_hi) + pml);
          __builtin_prefetch(&tree_[2 * static_cast<usize>(kid_hi)]);
        }
      }
      const u8* q = base_ + cand;
      usize ml = pml;
      ml += match_len(p + ml, q + ml, record ? end : scan_end);
      bytes += static_cast<int>(ml - pml) + 1;
      if (record && static_cast<int>(ml) > best) {
        best = static_cast<int>(ml);
        const i32 off = static_cast<i32>(pos - static_cast<usize>(cand));
        if (count < kBtMaxCands) {
          bt_cache_len_[count] = best;
          bt_cache_off_[count] = off;
          ++count;
        } else {
          bt_cache_len_[kBtMaxCands - 1] = best;
          bt_cache_off_[kBtMaxCands - 1] = off;
        }
      }
      if (p + ml >= scan_end || static_cast<int>(ml) >= params_.nice_len) {
        // Suffix ran into the block end (ordering undefined) or the match is
        // long enough: stop and drop the remaining subtree (zstd does the
        // same; re-linking it correctly would need more comparisons than the
        // dropped candidates are worth). Applies to insertion walks too -
        // on self-similar data every pending insert otherwise burns the full
        // depth budget comparing near-identical suffixes, which made the BT
        // an order of magnitude slower than chains on run-heavy blocks.
        break;
      }
      if (q[ml] < p[ml]) {
        // cand's suffix sorts below pos's: link it as the smaller child and
        // descend into its larger subtree (candidates between cand and pos).
        *smaller = cand;
        common_lo = ml;
        smaller = &tree_[2 * static_cast<usize>(cand) + 1];
        cand = kid_hi;
      } else {
        *larger = cand;
        common_hi = ml;
        larger = &tree_[2 * static_cast<usize>(cand)];
        cand = kid_lo;
      }
    }
    *smaller = -1;
    *larger = -1;
    return record ? count : bytes;
  }

  // O(1) deferred insert: prepend to the bucket's unsorted list. The next
  // slot lives in the larger-child slot; the smaller-child slot holds the
  // "unsorted" marker until a query sorts this prefix into the tree.
  void bt_insert(usize pos) noexcept {
    if (pos < bt_next_insert_) return;  // already inserted by a query
    bt_next_insert_ = pos + 1;
    if (pos + 4 > size_) return;
    const u32 h = hash_at(pos);
    tree_[2 * pos] = kBtUnsorted;
    tree_[2 * pos + 1] = head_[h];
    head_[h] = static_cast<i32>(pos);
  }

  // Query `pos` (with insertion), memoizing the candidate list. Re-queries of
  // the same position (lazy parser shift) hit the cache; a query for a
  // position that was already inserted without being cached cannot re-insert
  // (it would self-link), so it reports no candidates - the parsers never do
  // this (queries precede insertion), it is only a safety net.
  void bt_query(usize pos) noexcept {
    if (pos == bt_cached_pos_) return;
    bt_cached_pos_ = pos;
    bt_cached_count_ = 0;
    if (pos < bt_next_insert_ || pos + 4 > size_) return;
    bt_next_insert_ = pos + 1;
    const u32 h = hash_at(pos);
    // Fast path for run-heavy data: if the newest pending (unsorted) entry
    // already matches to nice_len, every finder would stop right there - so
    // answer from it and leave the bucket unsorted, deferring the batch sort
    // (the expensive part: each pending insert re-scans a long run) until a
    // query actually needs to descend. pos joins the pending list. Inside
    // self-similar runs this makes the BT behave like a chain that breaks on
    // its first candidate, instead of paying up to max_chain long compares.
    const i32 top = head_[h];
    if (top >= 0 && tree_[2 * static_cast<usize>(top)] == kBtUnsorted) {
      const usize l = match_len(base_ + pos, base_ + top, base_ + size_);
      if (static_cast<int>(l) >= params_.nice_len) {
        bt_cache_len_[0] = static_cast<int>(l);
        bt_cache_off_[0] = static_cast<i32>(pos - static_cast<usize>(top));
        bt_cached_count_ = 1;
        tree_[2 * pos] = kBtUnsorted;
        tree_[2 * pos + 1] = top;
        head_[h] = static_cast<i32>(pos);
        return;
      }
    }
    const i32 root = bt_sort_pending(h);
    bt_cached_count_ = bt_walk(pos, root, /*record=*/true, params_.max_chain);
    head_[h] = static_cast<i32>(pos);
  }

  static void bump_epoch(u32& epoch, std::vector<u32>& stamps, usize n) {
    if (stamps.size() < n) stamps.assign(n, 0);
    ++epoch;
    if (epoch == 0) {
      std::fill_n(stamps.begin(), n, 0);
      epoch = 1;
    }
  }

  const u8* base_ = nullptr;
  usize size_ = 0;
  MfParams params_{};
  std::vector<i32> head_;
  std::vector<u32> head_epoch_;
  u32 chain_epoch_ = 1;
  std::vector<i32> prev_;
  std::vector<i32> tree_;
  std::vector<u8> row_tags_;
  std::vector<i32> row_idx_;
  std::vector<u32> row_epoch_;
  u32 row_gen_ = 1;
  std::vector<u8> row_head_;
  usize row_memo_pos_ = ~usize{0};
  u32 row_memo_row_ = 0;
  u8 row_memo_tag_ = 0;
  usize bt_next_insert_ = 0;
  usize bt_cached_pos_ = ~usize{0};
  bool bt_prefetch_ = false;
  int bt_cached_count_ = 0;
  int bt_cache_len_[kBtMaxCands];
  i32 bt_cache_off_[kBtMaxCands];
};

}  // namespace spectra8::detail

#endif  // SPECTRA8_SRC_MATCH_FINDER_HPP
