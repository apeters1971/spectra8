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

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <span>
#include <utility>
#include <vector>

#include "util/bits.hpp"
#include "util/byte_rle.hpp"
#include "prism/bwt.hpp"
#include "prism/bwt_cm.hpp"
#include "prism/bwt_ultra.hpp"
#include "prism/bwt_mtf_rle.hpp"
#include "util/checksum.hpp"
#include "codec.hpp"
#include "util/pack_filter.hpp"
#include "util/entropy.hpp"
#include "ext/zstd.hpp"
#include "util/format.hpp"
#include "prism/lzp.hpp"
#include "oracle/oracle.hpp"
#include "crystal/crystal.hpp"
#include "mosaic/mosaic.hpp"
#include "flash/flash.hpp"
#include "laser/laser.hpp"
#include "grating/grating.hpp"
#include "beam/lz.hpp"
#include "beam/match_finder.hpp"
#include "util/parallel.hpp"
#include "util/simd.hpp"
#include "util/tans.hpp"
#include "util/transform.hpp"

#ifdef SPECTRA8_HAVE_GPU
#include "gpu/gpu_match.hpp"
#endif

namespace spectra8::detail {

FlashStats get_last_flash_stats() noexcept {
  return g_flash_last_stats;
}

namespace {

// Append a varint-length-prefixed raw blob.
void put_blob(std::vector<u8>& out, const std::vector<u8>& blob) {
  ByteWriter w(out);
  w.varint(blob.size());
  w.bytes(blob);
}

// Encode a literal-length / match-length value into a stream byte (+ escape).
void put_len(u32 value, std::vector<u8>& stream, std::vector<u8>& extra) {
  if (value >= 255) {
    stream.push_back(255);
    ByteWriter w(extra);
    w.varint(value - 255);
  } else {
    stream.push_back(static_cast<u8>(value));
  }
}

struct ParseResult {
  std::vector<Sequence> seqs;
  std::vector<u8> literals;  // raw literal bytes, in order
  usize last_literals = 0;
};

// Highest level handled by the single-probe "fast" parser below.
constexpr int kFastLevelMax = 4;  // 1-2 fast, 3-4 double-fast

// 2-3 byte repeat matches are a decode/encode tax on small blocks (16 KiB
// shuffle planes, 128 KiB verdict pages): they buy almost no size vs zstd.
// L5 never emits them (zstd L5 minMatch is 4/5). Larger blocks at other
// levels keep kMinRepMatch = 2.
constexpr usize kSmallBlock = usize{256} << 10;
inline int min_rep_for_block(usize n, int level = 0) noexcept {
  if (level == 5 || n < kSmallBlock) return kMinMatch;
  return kMinRepMatch;
}

// zstd-style "fast" strategy for the lowest levels: a single hash *table* (one
// slot per hash, no chain), the most-recent repeat offset, and a skip step that
// accelerates over incompressible regions. Produces a valid (looser) parse far
// faster than the hash-chain parser; the bitstream is unchanged.
// 8-byte hash for the double-fast long table.
inline u32 hash8(u64 v, int bits) noexcept {
  return static_cast<u32>((v * 0xCF1BBCDCBB9ACBB7ull) >> (64 - bits));
}

// kDouble = false: one 4-byte hash table (levels 1-2, zstd "fast").
// kDouble = true: zstd "dfast" (levels 3-4): a long (8-byte) table is probed
// first and wins outright; a short (4-byte) hit is checked against a long
// hit one position later before being taken. Both parsers extend matches
// backwards into the literal run (the skip step can overshoot a match
// start) and share the repeat-offset check and bitstream.
template <bool kDouble>
ParseResult parse_block_fast_impl(const u8* src, usize n, int level) {
  ParseResult res;
  res.seqs.reserve(n / 6 + 8);
  res.literals.reserve(n / 2 + 8);
  const int min_rep = min_rep_for_block(n);
  const MfParams pp = params_for_level(level, n);
  const int hlog = pp.hash_log;
  const usize hsize = usize{1} << hlog;
  static thread_local std::vector<i32> htab;
  static thread_local std::vector<i32> ltab;
  if (htab.size() < hsize) htab.resize(hsize);
  std::fill_n(htab.begin(), hsize, i32{-1});
  if (kDouble) {
    if (ltab.size() < hsize) ltab.resize(hsize);
    std::fill_n(ltab.begin(), hsize, i32{-1});
  }

  // Larger skip strength = more thorough (slower to accelerate); level 1 skips
  // a touch more aggressively than level 2.
  const int skip_strength = (level <= 1) ? 5 : (level <= 2 ? 6 : 7);

  RecentOffsets reps;
  reps.init();
  usize anchor = 0;
  usize ip = 0;
  constexpr usize kNeed = kDouble ? 8 : 4;  // bytes loaded at ip

  while (ip + kNeed <= n) {
    const u32 cur = load_u32(src + ip);
    const u32 h = hash4(cur, hlog);
    const i32 cand = htab[h];
    htab[h] = static_cast<i32>(ip);
    u64 cur8 = 0;
    i32 lcand = -1;
    if (kDouble) {
      cur8 = load_u64(src + ip);
      const u32 lh = hash8(cur8, hlog);
      lcand = ltab[lh];
      ltab[lh] = static_cast<i32>(ip);
    }

    int mlen = 0;
    i32 moff = 0;
    int rep_idx = -1;

    // Repeat-offset candidates (same order as the chain parser).
    for (int ri = 0; ri < kNumRecentOffsets; ++ri) {
      const i32 ro = reps.r[ri];
      if (ro <= 0 || static_cast<usize>(ro) > ip) continue;
      if (load_u32(src + ip - ro) != cur) continue;
      const int l =
          static_cast<int>(match_len(src + ip, src + ip - ro, src + n));
      if (l >= min_rep && l > mlen) {
        mlen = l;
        moff = ro;
        rep_idx = ri;
      }
    }
    usize mpos = ip;  // start of the chosen match (ip or ip + 1)
    bool found_new = false;
    if (kDouble && lcand >= 0 && load_u64(src + lcand) == cur8) {
      const int l = static_cast<int>(match_len(src + ip, src + lcand, src + n));
      if (l > mlen) {
        mlen = l;
        moff = static_cast<i32>(ip - static_cast<usize>(lcand));
        rep_idx = -1;
        found_new = true;
      }
    }
    if (!found_new && cand >= 0 && load_u32(src + cand) == cur) {
      int l = static_cast<int>(match_len(src + ip, src + cand, src + n));
      i32 o = static_cast<i32>(ip - static_cast<usize>(cand));
      if (kDouble && ip + 1 + 8 <= n) {
        // A long match one byte later beats a short one here.
        const u64 nxt8 = load_u64(src + ip + 1);
        const u32 lh = hash8(nxt8, hlog);
        const i32 lc = ltab[lh];
        ltab[lh] = static_cast<i32>(ip + 1);
        if (lc >= 0 && load_u64(src + lc) == nxt8) {
          const int l2 = static_cast<int>(match_len(src + ip + 1, src + lc, src + n));
          if (l2 > l) {
            l = l2;
            o = static_cast<i32>(ip + 1 - static_cast<usize>(lc));
            mpos = ip + 1;
          }
        }
      }
      if (l >= kMinMatch && l > mlen) {
        mlen = l;
        moff = o;
        rep_idx = -1;
      } else {
        mpos = ip;
      }
    }

    if (mlen < ((rep_idx >= 0) ? min_rep : kMinMatch)) {
      // No usable match: advance, accelerating across incompressible data.
      ip += 1 + ((ip - anchor) >> skip_strength);
      continue;
    }

    // Backward extension into the literal run.
    while (mpos > anchor && mpos > static_cast<usize>(moff) &&
           src[mpos - 1] == src[mpos - 1 - static_cast<usize>(moff)]) {
      --mpos;
      ++mlen;
    }

    const usize lit_len = mpos - anchor;
    res.literals.insert(res.literals.end(), src + anchor, src + mpos);
    res.seqs.push_back({static_cast<u32>(lit_len), static_cast<u32>(mlen),
                        static_cast<u32>(moff)});
    if (rep_idx >= 0)
      reps.use_rep(rep_idx);
    else
      reps.use_new(moff);

    // Seed a couple of in-match positions so later matches can reference them.
    const usize mend = mpos + static_cast<usize>(mlen);
    if (mpos + 1 + kNeed <= n) {
      htab[hash4(load_u32(src + mpos + 1), hlog)] = static_cast<i32>(mpos + 1);
      if (kDouble)
        ltab[hash8(load_u64(src + mpos + 1), hlog)] = static_cast<i32>(mpos + 1);
    }
    if (mend >= 2 && mend + kNeed <= n) {
      htab[hash4(load_u32(src + mend - 2), hlog)] = static_cast<i32>(mend - 2);
      if (kDouble)
        ltab[hash8(load_u64(src + mend - 2), hlog)] = static_cast<i32>(mend - 2);
    }

    ip = mend;
    anchor = ip;
  }

  res.last_literals = n - anchor;
  res.literals.insert(res.literals.end(), src + anchor, src + n);
  return res;
}

ParseResult parse_block_fast(const u8* src, usize n, int level) {
  return level <= 2 ? parse_block_fast_impl<false>(src, n, level)
                    : parse_block_fast_impl<true>(src, n, level);
}

ParseResult parse_block(const u8* src, usize n, int level) {
  if (level <= kFastLevelMax) return parse_block_fast(src, n, level);

  ParseResult res;
  res.seqs.reserve(n / 6 + 8);
  res.literals.reserve(n / 2 + 8);
  const int min_rep = min_rep_for_block(n, level);
  // zstd L5: 3 reps; minMatch 5 on pages >256 KiB (the >256 KB clevels table).
  const int nrep = (level == 5) ? 3 : kNumRecentOffsets;
  const int min_new =
      (level == 5 && n > kSmallBlock) ? 5 : kMinMatch;
  MfParams pp = params_for_level(level, n);
  // Reused per worker thread: its head_/prev_ buffers are grown once and then
  // reused across blocks, so only the (small) hash-head table is reset per
  // block instead of also zero-filling the O(block) prev_ chain array.
  static thread_local MatchFinder mf;
  mf.init(src, n, pp);

  RecentOffsets reps;
  reps.init();

  usize anchor = 0;
  usize pos = 0;

  // Search acceleration over matchless stretches (zstd's kSearchStrength):
  // while no match has been found since the last one, the step grows with the
  // length of the literal run, so incompressible regions cost O(sqrt) searches
  // instead of one per position. The first match resets the run (anchor), so
  // compressible data - even right after a long literal run - is still parsed
  // position by position; what can be lost is a match *starting* deep inside a
  // matchless stretch, whose start may be overstepped. Accelerating only
  // after a 256-byte run keeps that loss at ~0.02% on the test corpus at
  // every level (faster ramps saved no additional time: the win comes from
  // long incompressible regions, where even this ramp reaches large steps),
  // so one conservative constant serves levels 3-9.
  constexpr int accel_log = 8;

  // Offset-aware match pricing (zstd's lazy "gain" heuristic). A match is
  // scored in quarter-bytes as 4*len minus its offset cost: a repeat offset
  // is a cheap token (kRepCost), a new offset costs its bit width (raw bits
  // plus the rarer offset-code symbol). This replaces a near-constant length
  // margin that let a marginally longer far match displace a rep - on
  // rep-heavy data (samba, mozilla, record-structured binary) that cost
  // 2-11% of the output and made larger blocks compress *worse*, since more
  // history only offered more expensive offsets. Constants were swept on a
  // 16-dataset mix at levels 3-9 (kLazyBonus 0..6, kRepCost 0..2,
  // kLitThresh 4..24): the total shrank ~2.3% at 256 KiB blocks and ~3.7% at
  // 1 MiB; text-like inputs move by <0.1%.
  constexpr int kRepCost = 1;
  // Bonus for keeping the current match in the lazy step (the shifted
  // alternative costs an extra literal).
  constexpr int kLazyBonus = 1;
  // A new-offset match with no rep alternative must also beat literals:
  // 8*len >= bit_width(off) + kLitThresh (i.e. its length and offset tokens
  // and raw bits must not exceed ~8 bits per byte replaced). Rejects 4-byte
  // matches at offsets >= 2^14 and 5-byte ones at >= 2^22.
  constexpr int kLitThresh = 18;

  // Skip the hash search once a repeat is already this long: a new offset
  // would have to be longer still to win the gain test. zstd's greedy/lazy
  // parsers do the same with targetLength. 16 is nearly ratio-neutral on
  // the L5 corpus (+0.03%) and avoids a full row search inside runs.
  constexpr int kHashSkipLen = 16;
  // Once r[0] is this long, later repeat slots are not checked. Swept at
  // L5: 8 vs checking all seven is +0.03% size (a later slot almost never
  // beats an 8-byte r[0] after the gain test).
  constexpr int kRep0Enough = 8;

  auto find_best = [&](usize p, int& len, i32& off, int& rep_idx) {
    len = 0;
    off = 0;
    rep_idx = -1;
    // r[0] first: it is the current match offset and the common hit inside
    // runs. A long enough r[0] skips the other six slots (each a random
    // match_len) and, at kHashSkipLen, the hash search too.
    {
      const i32 ro = reps.r[0];
      if (ro > 0 && static_cast<usize>(ro) <= p) {
        const u8* q = src + p - ro;
        const bool hit = (min_rep >= 4)
                             ? (p + 4 <= n && load_u32(src + p) == load_u32(q))
                             : (p + 2 <= n && load_u16(src + p) == load_u16(q));
        if (hit) {
          const int l = static_cast<int>(match_len(src + p, q, src + n));
          if (l >= min_rep) {
            len = l;
            off = ro;
            rep_idx = 0;
          }
        }
      }
    }
    if (len < kRep0Enough) {
      for (int i = 1; i < nrep; ++i) {
        i32 ro = reps.r[i];
        if (ro <= 0 || static_cast<usize>(ro) > p) continue;
        const u8* q = src + p - ro;
        const bool hit = (min_rep >= 4)
                             ? (p + 4 <= n && load_u32(src + p) == load_u32(q))
                             : (p + 2 <= n && load_u16(src + p) == load_u16(q));
        if (!hit) continue;
        const int l = static_cast<int>(match_len(src + p, q, src + n));
        if (l >= min_rep && l > len) {
          len = l;
          off = ro;
          rep_idx = i;
        }
      }
    }
    // New match: must pay for its offset, against the rep alternative when
    // there is one, else against literals (see kLitThresh). Skipped once
    // a repeat is already long enough that a new offset cannot win.
    if (len < kHashSkipLen) {
      int new_len;
      i32 new_off;
      mf.find(p, new_len, new_off);
      if (new_len >= min_new) {
        const int nbits = bit_width(static_cast<u32>(new_off));
        const bool take_new = (rep_idx < 0)
                                  ? (new_len * 8 >= nbits + kLitThresh)
                                  : (new_len * 4 - nbits > len * 4 - kRepCost);
        if (take_new) {
          len = new_len;
          off = new_off;
          rep_idx = -1;
        }
      }
    }
  };

  // Result of the lazy probe at pos+1 when the parser shifted there: the
  // probe was computed in exactly the state the next iteration would see (no
  // sequence emitted in between, pos already inserted), so it is reused
  // instead of searching the same position twice.
  bool have_next = false;
  int next_len = 0, next_rep = -1;
  i32 next_off = 0;

  while (pos < n) {
    int len, rep_idx;
    i32 off;
    if (have_next) {
      len = next_len;
      off = next_off;
      rep_idx = next_rep;
      have_next = false;
    } else {
      find_best(pos, len, off, rep_idx);
    }

    int min_needed = (rep_idx >= 0) ? min_rep : min_new;
    if (len < min_needed) {
      mf.insert(pos);
      pos += 1 + ((pos - anchor) >> accel_log);
      continue;
    }

    // 1-step lazy evaluation: prefer a clearly better match at pos+1.
    // Skip the probe once the current match is already solid - the chance of
    // a useful +1 shift is tiny and the extra find dominates. A repeat-offset
    // match is skipped from a much shorter length: the probe's gain formula
    // undervalues keeping the rep (the shifted match also breaks the rep
    // chain for the *following* sequences), so probing on reps lost ratio
    // as well as time. Swept on the 16-dataset corpus at L5: (32, 6) versus
    // the earlier (16, 16) is -0.45% total (samba -11%, xml -5%, nci -2.6%,
    // nothing worse than +0.2%) at about the same encode time (fewer
    // probes on reps, more on long new-offset matches).
    // Skip the pos+1 probe once the current match is already solid. Repeats
    // skip from 6: the gain formula undervalues keeping the rep (the shift
    // also breaks the chain for later sequences). New offsets skip from 16
    // when the match is near, and from 32 when it is far (bit_width >= 16):
    // +1 often finds a closer offset, and that is cheaper than 16+ raw bits.
    // Swept at L5 vs the older (32, 6) on every new offset: this is +0.14%
    // size and ~20% faster on samba / nci; mozilla / dickens stay within
    // noise (they are hash-search bound, not lazy-probe bound).
    constexpr int kLazySkipLen = 16;
    constexpr int kLazyRepSkip = 6;
    constexpr int kLazyFarBits = 16;
    constexpr int kLazyFarCap = 32;
    const int lazy_lim =
        (rep_idx >= 0)
            ? kLazyRepSkip
            : (bit_width(static_cast<u32>(off)) >= kLazyFarBits ? kLazyFarCap
                                                                : kLazySkipLen);
    if (pp.lazy && len < lazy_lim && pos + 1 < n) {
      mf.insert(pos);
      int len2, rep2;
      i32 off2;
      find_best(pos + 1, len2, off2, rep2);
      const int g1 =
          len * 4 - (rep_idx >= 0 ? kRepCost : bit_width(static_cast<u32>(off))) +
          kLazyBonus;
      const int g2 =
          len2 * 4 - (rep2 >= 0 ? kRepCost : bit_width(static_cast<u32>(off2)));
      if (g2 > g1) {
        // Emit a literal and shift to pos+1, carrying the probe's result.
        ++pos;
        // (pos was already inserted above)
        have_next = true;
        next_len = len2;
        next_off = off2;
        next_rep = rep2;
        continue;
      }
      if (pp.lazy >= 2 && pos + 2 < n && len2 < lazy_lim) {
        // 2-step lazy evaluation (levels 8-9): test pos+2 to avoid local minima
        int len3, rep3;
        i32 off3;
        find_best(pos + 2, len3, off3, rep3);
        const int g3 =
            len3 * 4 - (rep3 >= 0 ? kRepCost : bit_width(static_cast<u32>(off3)));
        if (g3 > g1 + 4 && g3 > g2 + 2) {
          // Emit a literal at pos, then carry len2 to pos+1
          ++pos;
          have_next = true;
          next_len = len2;
          next_off = off2;
          next_rep = rep2;
          continue;
        }
      }
      // else fall through and emit the match at pos.
    } else {
      mf.insert(pos);
    }

    usize lit_len = pos - anchor;
    res.literals.insert(res.literals.end(), src + anchor, src + pos);
    res.seqs.push_back({static_cast<u32>(lit_len), static_cast<u32>(len),
                        static_cast<u32>(off)});

    if (rep_idx >= 0)
      reps.use_rep(rep_idx);
    else
      reps.use_new(off);

    // Sparse match-interior inserts (start already inserted above).
    mf.insert_match(pos, static_cast<usize>(len));
    pos += static_cast<usize>(len);
    anchor = pos;
  }

  res.last_literals = n - anchor;
  res.literals.insert(res.literals.end(), src + anchor, src + n);
  return res;
}

// ---------------------------------------------------------------------------
// Optimal (cost-based) parser for the "ultra" level.
//
// Rather than greedily taking the longest match at each position, the optimal
// parser runs a forward shortest-path over the block: every byte position is a
// node and an edge is either "emit one literal" or "emit a match of length L at
// offset O". Edge weights are estimated bit-costs derived from histograms of an
// initial greedy parse, then refined over further passes (each priced from the
// previous pass's parse) until the estimated size stops improving. This is
// the same family as zstd's btultra / LZMA's optimal parse.
//
// Cost model notes / approximations (kept deliberately simple for a first cut):
//   * Literal byte cost  = order-0 price of that byte in the literal stream.
//   * Match cost         = matchlen-token price + offcode-token price
//                          (+ raw offset bits) + an amortized litlen-token cost.
//   * Only the most-recent offset (rep0) is modelled along the DP path; reps
//     1..6 are still recovered for free by the serializer, so the estimate is
//     conservative, never wrong.
// ---------------------------------------------------------------------------

constexpr int kOptMaxCands = 8;    // candidate offsets gathered per position (ultra)
constexpr int kNearOptCands = 6;   // candidates for the lighter near-optimal level
constexpr int kOptLenCap = 64;     // enumerate every length up to this cap
// Maximum DP passes (the first priced from the greedy parse, each further one
// from the previous pass's parse); the loop stops early once a pass gains
// less than kOptMinGain of the estimated size. Measured on the 16-dataset
// corpus at 1 MiB blocks: ultra went from 2 fixed passes to convergence at
// -1.3% total (samba -17%, json -10%, bin-i32 -3%) for ~1.6x the encode
// time; near-optimal from 1 pass to at most 3 at -1.1% for ~1.3x.
constexpr int kOptPasses = 8;
constexpr int kNearOptPasses = 3;
constexpr double kOptMinGain = 0.0005;

struct OptPrices {
  float lit[256];       // per-byte literal price
  float matchlen[256];  // matchlen-token price
  float offcode[256];   // offcode-token price (rep index or new-offset class)
  float litlen[256];    // per-sequence litlen-token price, indexed by run length
};

// Convert a frequency histogram to per-symbol bit-prices (Laplace-smoothed).
void build_price_table(const u32* freq, int n, float* out) {
  u64 total = 0;
  for (int i = 0; i < n; ++i) total += freq[i];
  if (total == 0) {
    for (int i = 0; i < n; ++i) out[i] = 8.0f;
    return;
  }
  const double denom = static_cast<double>(total) + 0.5 * n;
  for (int i = 0; i < n; ++i) {
    double p = (static_cast<double>(freq[i]) + 0.5) / denom;
    out[i] = static_cast<float>(-std::log2(p));
  }
}

// Token histograms of a parse, as the serializer would code it (literal
// bytes, matchlen / offcode / litlen tokens) plus the raw offset bits.
struct ParseHist {
  u32 lit[256] = {0}, ml[256] = {0}, oc[256] = {0}, ll[256] = {0};
  u64 raw_bits = 0;
};

void gather_hist(const ParseResult& pr, ParseHist& h) {
  for (u8 b : pr.literals) ++h.lit[b];
  RecentOffsets reps;
  reps.init();
  for (const Sequence& s : pr.seqs) {
    ++h.ll[min_of<u32>(s.lit_len, 255)];
    ++h.ml[min_of<u32>(s.match_len - kMinRepMatch, 255)];
    int idx = reps.find(static_cast<i32>(s.offset));
    if (idx >= 0) {
      ++h.oc[idx];
      reps.use_rep(idx);
    } else {
      int k = bit_width(s.offset) - 1;
      ++h.oc[min_of(kNumRecentOffsets + k, 255)];
      h.raw_bits += static_cast<u64>(k);
      reps.use_new(static_cast<i32>(s.offset));
    }
  }
}

// Derive a price table from an existing parse (greedy or a prior DP pass).
void build_prices_from(const ParseResult& pr, OptPrices& op) {
  ParseHist h;
  gather_hist(pr, h);
  build_price_table(h.lit, 256, op.lit);
  build_price_table(h.ml, 256, op.matchlen);
  build_price_table(h.oc, 256, op.offcode);
  build_price_table(h.ll, 256, op.litlen);
}

// PAQ-style context model for literal cost evaluation in optimal parse.
// Models P(byte | order-1 context) and P(byte | record-lane context),
// blended with order-0 token prices via confidence weighting.
struct ContextModel {
  bool active = false;
  int stride = 1;
  // 256 rows x 256 cols of literal bit prices: -log2(P(c | c1))
  std::vector<float> lit_ctx;
  // 256 rows x 256 cols of lane bit prices: -log2(P(c | lat1)) if stride > 1
  std::vector<float> lit_lat;

  void init(const u8* src, usize n, int S, const OptPrices& op) {
    if (n < 64) return;
    active = true;
    stride = S;
    lit_ctx.assign(256 * 256, 0.0f);

    std::vector<u32> cond(256 * 256, 0);
    std::vector<u32> row_tot(256, 0);
    u8 prev = 0;
    for (usize i = 0; i < n; ++i) {
      const u8 c = src[i];
      if (i > 0) {
        ++cond[static_cast<usize>(prev) * 256 + c];
        ++row_tot[prev];
      }
      prev = c;
    }

    for (int c1 = 0; c1 < 256; ++c1) {
      const u32 rt = row_tot[c1];
      float* out_row = &lit_ctx[static_cast<usize>(c1) * 256];
      if (rt == 0) {
        for (int c = 0; c < 256; ++c) out_row[c] = op.lit[c];
        continue;
      }
      const float lambda = 0.35f * (static_cast<float>(rt) / (static_cast<float>(rt) + 64.0f));
      const float inv_denom = 1.0f / (static_cast<float>(rt) + 64.0f);
      for (int c = 0; c < 256; ++c) {
        const float p1 = (static_cast<float>(cond[static_cast<usize>(c1) * 256 + c]) + 0.25f) * inv_denom;
        const float p0 = std::exp2(-op.lit[c]);
        const float p = (1.0f - lambda) * p0 + lambda * p1;
        const float min_cost = 0.70f * op.lit[c];
        out_row[c] = std::max(min_cost, -std::log2(std::max(p, 1e-7f)));
      }
    }

    if (stride > 1 && n >= static_cast<usize>(stride * 2)) {
      lit_lat.assign(256 * 256, 0.0f);
      std::vector<u32> lat_cond(256 * 256, 0);
      std::vector<u32> lat_row(256, 0);
      for (usize i = stride; i < n; ++i) {
        const u8 lat = src[i - stride];
        const u8 c = src[i];
        ++lat_cond[static_cast<usize>(lat) * 256 + c];
        ++lat_row[lat];
      }
      for (int lat = 0; lat < 256; ++lat) {
        const u32 rt = lat_row[lat];
        float* out_row = &lit_lat[static_cast<usize>(lat) * 256];
        if (rt == 0) {
          for (int c = 0; c < 256; ++c) out_row[c] = op.lit[c];
          continue;
        }
        const float lambda = 0.35f * (static_cast<float>(rt) / (static_cast<float>(rt) + 64.0f));
        const float inv_denom = 1.0f / (static_cast<float>(rt) + 64.0f);
        for (int c = 0; c < 256; ++c) {
          const float p_lat = (static_cast<float>(lat_cond[static_cast<usize>(lat) * 256 + c]) + 0.25f) * inv_denom;
          const float p0 = std::exp2(-op.lit[c]);
          const float p = (1.0f - lambda) * p0 + lambda * p_lat;
          const float min_cost = 0.70f * op.lit[c];
          out_row[c] = std::max(min_cost, -std::log2(std::max(p, 1e-7f)));
        }
      }
    }
  }

  [[nodiscard]] inline float lit_cost(const u8* src, usize i, const OptPrices& op) const noexcept {
    if (!active) return op.lit[src[i]];
    const u8 c1 = (i > 0) ? src[i - 1] : 0;
    float cost = lit_ctx[(static_cast<usize>(c1) << 8) | src[i]];
    if (stride > 1 && i >= static_cast<usize>(stride)) {
      const u8 lat1 = src[i - stride];
      const float lat_cost = lit_lat[(static_cast<usize>(lat1) << 8) | src[i]];
      cost = 0.5f * (cost + lat_cost);
    }
    return cost;
  }
};

// Order-0 entropy estimate (bits) of a parse's coded size: the sum of the
// token streams' entropies plus the raw offset bits. Cheap, and tracks the
// real entropy-coded size closely enough to compare DP passes.
double estimate_parse_bits(const ParseResult& pr) {
  ParseHist h;
  gather_hist(pr, h);
  auto entropy = [](const u32* f) {
    double total = 0;
    for (int i = 0; i < 256; ++i) total += f[i];
    double bits = 0;
    for (int i = 0; i < 256; ++i)
      if (f[i]) bits -= f[i] * std::log2(f[i] / total);
    return bits;
  };
  return entropy(h.lit) + entropy(h.ml) + entropy(h.oc) + entropy(h.ll) +
         static_cast<double>(h.raw_bits);
}

// Candidate sources for the optimal parser. Each offers:
//   reset()                    - called once at the start of every DP pass
//   candidates(i, lens, offs, cap) - fill up to `cap` (length, offset) pairs
//                                for position i; return the count
//   inserted(i)                - advance internal state past position i
// The CPU source walks a hash chain; the GPU source reads precomputed
// per-position top-K candidates (and re-extends their true length on the CPU,
// since the kernel caps the reported length).

struct CpuCandidateSource {
  const u8* src;
  usize n;
  MfParams pp;
  MatchFinder mf;
  CpuCandidateSource(const u8* s, usize nn, MfParams p) : src(s), n(nn), pp(p) {}
  void reset() { mf.init(src, n, pp); }
  int candidates(usize i, int* lens, i32* offs, int cap) {
    return mf.find_all(i, lens, offs, cap);
  }
  void inserted(usize i) { mf.insert(i); }
};

struct GpuCandidateSource {
  const u8* src;
  usize n;
  const u16* hk_len;  // n * k entries, candidate s of position i at [i*k + s]
  const u32* hk_off;
  int k;
  GpuCandidateSource(const u8* s, usize nn, const u16* hl, const u32* ho, int kk)
      : src(s), n(nn), hk_len(hl), hk_off(ho), k(kk) {}
  void reset() {}
  void inserted(usize) {}
  int candidates(usize i, int* lens, i32* offs, int cap) {
    if (i + 4 > n) return 0;
    int cnt = 0;
    const usize b = i * static_cast<usize>(k);
    for (int s = 0; s < k && cnt < cap; ++s) {
      const u32 ho = hk_off[b + s];
      if (hk_len[b + s] < kMinMatch || ho == 0 || static_cast<usize>(ho) > i)
        continue;
      usize l = match_len(src + i, src + i - ho, src + n);
      if (static_cast<int>(l) < kMinMatch) continue;
      bool dup = false;
      for (int t = 0; t < cnt; ++t)
        if (offs[t] == static_cast<i32>(ho)) {
          dup = true;
          break;
        }
      if (dup) continue;
      lens[cnt] = static_cast<int>(l);
      offs[cnt] = static_cast<i32>(ho);
      ++cnt;
    }
    return cnt;
  }
};

// Core cost-based optimal parse. Candidate discovery is delegated to `cand`
// (CPU hash chain or GPU top-K); everything else (rep-offset matching, pricing,
// DP, reconstruction, 2-pass refinement) is identical regardless of source.
template <class CandSource>
ParseResult optimal_parse_core(const u8* src, usize n, int level,
                               CandSource& cand, const ParseResult& greedy,
                               bool context_oracle = false) {
  // Initial statistics come from the caller-supplied greedy/lazy parse (the
  // caller also serializes it as the fallback candidate, so it is computed
  // exactly once per block).
  if (n < 64) return greedy;  // not worth the DP machinery
  const int min_rep = min_rep_for_block(n);

  OptPrices op;
  build_prices_from(greedy, op);

  const int stride = context_oracle ? detect_stride(src, n) : 1;
  ContextModel cm;
  if (context_oracle) {
    cm.init(src, n, stride, op);
  }

  // Ultra (level 11) iterates price refinement to convergence (up to
  // kOptPasses) with a wider candidate set; the lighter near-optimal level
  // (10) stops after kNearOptPasses and gathers fewer candidates.
  const int passes = (level >= kUltraLevel) ? kOptPasses : kNearOptPasses;
  const int cand_cap = (level >= kUltraLevel) ? kOptMaxCands : kNearOptCands;

  const float kInf = 1e30f;
  std::vector<float> cost(n + 1);
  std::vector<i32> from(n + 1);
  std::vector<i32> mlen(n + 1);   // 0 => arrived via a literal
  std::vector<u32> moff(n + 1);
  std::vector<RecentOffsets> rstate(n + 1);  // recent-offset cache on best path
  std::vector<i32> litrun(n + 1); // literal-run length ending at this position

  std::vector<usize> mstart;
  std::vector<int> mlens;
  std::vector<u32> moffs;

  ParseResult best;
  ParseResult best_kept;  // best pass so far (by estimated coded bits)
  double best_kept_bits = 0;
  int clens[kOptMaxCands];
  i32 coffs[kOptMaxCands];

  // Candidate memo: the finder's answers depend only on the position (every
  // position is inserted exactly once per pass, in order), so refinement
  // passes replay the first pass's candidates instead of re-searching -
  // candidate gathering dominates the whole parse (~75% of ultra encode).
  // Both passes visit every position (cost[0] = 0 and the literal edge makes
  // i+1 reachable from any reachable i), so a sequential entry cursor stays
  // aligned with the per-position counts. If the memo outgrows its budget
  // (match-dense multi-MiB blocks), it is dropped and later passes fall back
  // to re-searching.
  const bool memoize = passes > 1;
  constexpr usize kMemoMaxEntries = usize{8} << 20;  // 8M entries = 64 MiB
  std::vector<u8> memo_cnt;
  std::vector<u64> memo_ent;  // len << 32 | offset
  bool memo_ok = memoize;
  if (memoize) memo_cnt.assign(n, 0);

  for (int pass = 0; pass < passes; ++pass) {
    std::fill(cost.begin(), cost.end(), kInf);
    cost[0] = 0.0f;
    from[0] = -1;
    mlen[0] = 0;
    moff[0] = 0;
    rstate[0].init();
    litrun[0] = 0;
    const bool replay = pass > 0 && memo_ok;
    usize memo_pos = 0;
    if (!replay) cand.reset();

    // Repeat-offset scan cache. Inside a long self-similar run every rep
    // offset matches until (nearly) the end of the run, and a naive
    // match_len per position re-scans the whole remainder for up to 7 reps -
    // quadratic in the run length (this alone made shuffled-binary blocks
    // ~10x slower than text). A scan at position s with offset o that hit
    // its first mismatch at e proves src[k] == src[k - o] for k in [s, e),
    // so any later position i < e has length exactly e - i. Entries are
    // (offset, mismatch position); i only grows within a pass, keeping the
    // proof valid until the position passes e, when one rescan renews it.
    // Only long scans are cached: short ones cannot cause the blowup, and
    // keeping churny data (text) out of the cache keeps the linear lookup
    // near-free there. Expired entries are dropped for the same reason.
    struct RepScan {
      i32 off;
      usize end;
    };
    constexpr usize kScanCacheMin = 256;  // cache scans with len >= this
    RepScan scan_cache[8];
    int scan_n = 0;
    int scan_rr = 0;  // round-robin victim
    auto rep_len = [&](usize i, i32 off) -> usize {
      for (int s = 0; s < scan_n; ++s) {
        if (scan_cache[s].off != off) continue;
        if (scan_cache[s].end > i) return scan_cache[s].end - i;
        const usize l = match_len(src + i, src + i - off, src + n);
        if (l >= kScanCacheMin) {
          scan_cache[s].end = i + l;
        } else {  // run over: drop the entry so lookups stay short
          scan_cache[s] = scan_cache[--scan_n];
          if (scan_rr > scan_n) scan_rr = 0;
        }
        return l;
      }
      const usize l = match_len(src + i, src + i - off, src + n);
      if (l >= kScanCacheMin) {
        if (scan_n < 8) {
          scan_cache[scan_n++] = {off, i + l};
        } else {
          scan_cache[scan_rr] = {off, i + l};
          scan_rr = (scan_rr + 1) & 7;
        }
      }
      return l;
    };

    // Relax every length of one (offset) candidate against the DP table. The
    // litlen token is priced from the actual literal-run length reaching i, so
    // zero-literal matches (common in repetitive data) are not overcharged. The
    // full recent-offset cache (`ns`, the state after taking this offset) is
    // propagated so repeat offsets are priced exactly as the serializer codes
    // them.
    auto relax = [&](usize i, i32 off, int minl, int maxl, float oc,
                     const RecentOffsets& ns) {
      const float base = cost[i] + op.litlen[min_of<i32>(litrun[i], 255)] + oc;
      auto try_len = [&](int l) {
        const usize j = i + static_cast<usize>(l);
        float mlc = op.matchlen[min_of(l - kMinRepMatch, 255)];
        if (l - kMinRepMatch >= 255) mlc += 16.0f;  // length-escape bytes
        const float c = base + mlc;
        if (c < cost[j]) {
          cost[j] = c;
          from[j] = static_cast<i32>(i);
          mlen[j] = l;
          moff[j] = static_cast<u32>(off);
          rstate[j] = ns;
          litrun[j] = 0;
        }
      };
      const int cap = min_of(maxl, kOptLenCap);
      for (int l = minl; l <= cap; ++l) try_len(l);
      if (maxl > cap) try_len(maxl);  // also keep the full-length edge
    };

    for (usize i = 0; i < n; ++i) {
      if (cost[i] >= kInf) {
        if (!replay) cand.inserted(i);
        continue;
      }
      const RecentOffsets& reps = rstate[i];
      // Literal step.
      {
        const float lit_cost = cm.lit_cost(src, i, op);
        const float nc = cost[i] + lit_cost;
        if (nc < cost[i + 1]) {
          cost[i + 1] = nc;
          from[i + 1] = static_cast<i32>(i);
          mlen[i + 1] = 0;
          moff[i + 1] = 0;
          rstate[i + 1] = reps;
          litrun[i + 1] = litrun[i] + 1;
        }
      }
      // Repeat-offset matches (all cached offsets): cheap offcode, min length
      // min_rep (2 on large blocks, 4 on small). At equal length, rep edges
      // from the same position differ only in
      // their offcode price, so the cheapest rep dominates every other rep
      // on the lengths it reaches: relax cheapest-first and give each rep
      // only the lengths no cheaper rep covered (exact - it never skips a
      // length that could win; relax() keeps the isolated full-length edge
      // past kOptLenCap unconditionally). On self-similar data all 7 reps
      // reach the end of the run, and this collapses 7 full length sweeps
      // into one plus 6 no-ops.
      {
        int ridx[kNumRecentOffsets];
        int rlen[kNumRecentOffsets];
        int rn = 0;
        for (int idx = 0; idx < kNumRecentOffsets; ++idx) {
          const i32 ro = reps.r[idx];
          if (ro <= 0 || static_cast<usize>(ro) > i) continue;
          const usize l = rep_len(i, ro);
          if (static_cast<int>(l) < min_rep) continue;
          ridx[rn] = idx;
          rlen[rn] = static_cast<int>(l);
          ++rn;
        }
        if (rn == 1) {  // common on text: one usable rep, no pruning needed
          RecentOffsets ns = reps;
          ns.use_rep(ridx[0]);
          relax(i, reps.r[ridx[0]], min_rep, rlen[0], op.offcode[ridx[0]],
                ns);
        } else if (rn > 1) {
          for (int a = 1; a < rn; ++a) {  // insertion sort by offcode price
            const int ii = ridx[a], ll = rlen[a];
            int b = a - 1;
            while (b >= 0 && op.offcode[ridx[b]] > op.offcode[ii]) {
              ridx[b + 1] = ridx[b];
              rlen[b + 1] = rlen[b];
              --b;
            }
            ridx[b + 1] = ii;
            rlen[b + 1] = ll;
          }
          int covered = min_rep - 1;
          for (int a = 0; a < rn; ++a) {
            const int idx = ridx[a];
            const int contig_hi = min_of(rlen[a], kOptLenCap);
            if (contig_hi <= covered && rlen[a] <= kOptLenCap) continue;
            RecentOffsets ns = reps;
            ns.use_rep(idx);
            if (contig_hi > covered)  // contiguous band no cheaper rep priced
              relax(i, reps.r[idx], covered + 1, contig_hi, op.offcode[idx],
                    ns);
            if (rlen[a] > kOptLenCap)  // this rep's own full-length edge
              relax(i, reps.r[idx], rlen[a], rlen[a], op.offcode[idx], ns);
            covered = max_of(covered, contig_hi);
          }
        }
      }
      // New-offset matches from the candidate source (CPU finder or GPU
      // top-K), or the memo of what the source returned in the first pass.
      int nc;
      if (replay) {
        nc = memo_cnt[i];
        for (int c = 0; c < nc; ++c) {
          const u64 e = memo_ent[memo_pos++];
          clens[c] = static_cast<int>(e >> 32);
          coffs[c] = static_cast<i32>(e & 0xFFFFFFFFu);
        }
      } else {
        nc = cand.candidates(i, clens, coffs, cand_cap);
        if (memoize && pass == 0 && memo_ok) {
          if (memo_ent.size() + static_cast<usize>(nc) <= kMemoMaxEntries) {
            memo_cnt[i] = static_cast<u8>(nc);
            for (int c = 0; c < nc; ++c)
              memo_ent.push_back((static_cast<u64>(clens[c]) << 32) |
                                 static_cast<u32>(coffs[c]));
          } else {
            memo_ok = false;
            memo_ent.clear();
            memo_ent.shrink_to_fit();
          }
        }
      }
      for (int c = 0; c < nc; ++c) {
        const i32 off = coffs[c];
        if (reps.find(off) >= 0) continue;  // already covered as a repeat offset
        const int k = bit_width(static_cast<u32>(off)) - 1;
        const float oc =
            op.offcode[min_of(kNumRecentOffsets + k, 255)] + static_cast<float>(k);
        RecentOffsets ns = reps;
        ns.use_new(off);
        relax(i, off, kMinMatch, clens[c], oc, ns);
      }
      if (!replay) cand.inserted(i);
    }

    // Walk the back-pointers to recover the chosen matches (end -> start).
    mstart.clear();
    mlens.clear();
    moffs.clear();
    for (usize j = n; j > 0;) {
      const i32 f = from[j];
      if (f < 0) break;
      if (mlen[j] > 0) {
        mstart.push_back(static_cast<usize>(f));
        mlens.push_back(mlen[j]);
        moffs.push_back(moff[j]);
      }
      j = static_cast<usize>(f);
    }
    std::reverse(mstart.begin(), mstart.end());
    std::reverse(mlens.begin(), mlens.end());
    std::reverse(moffs.begin(), moffs.end());

    best = ParseResult{};
    usize anchor = 0;
    for (usize s = 0; s < mstart.size(); ++s) {
      const usize p = mstart[s];
      const usize lit_len = p - anchor;
      best.literals.insert(best.literals.end(), src + anchor, src + p);
      best.seqs.push_back({static_cast<u32>(lit_len),
                           static_cast<u32>(mlens[s]), moffs[s]});
      anchor = p + static_cast<usize>(mlens[s]);
    }
    best.last_literals = n - anchor;
    best.literals.insert(best.literals.end(), src + anchor, src + n);

    if (passes == 1) return best;

    // Each pass minimizes the cost under the previous pass's prices; the
    // prices then move towards the new parse's statistics. On data whose
    // greedy statistics are far from the optimum (rep-heavy: samba, json)
    // two passes are nowhere near converged (samba: -13% from pass 2 to
    // pass 8), while on some inputs the iteration oscillates and a later
    // pass is worse. So iterate while the estimated coded size keeps
    // improving by more than kOptMinGain and keep the best pass seen.
    const double bits = estimate_parse_bits(best);
    if (pass == 0 || bits < best_kept_bits) {
      const bool improved =
          pass > 0 && bits < best_kept_bits * (1.0 - kOptMinGain);
      best_kept_bits = bits;
      std::swap(best_kept, best);
      if (pass > 0 && !improved) break;
    } else {
      break;
    }
    if (pass + 1 < passes) {
      build_prices_from(best_kept, op);
      if (context_oracle) cm.init(src, n, stride, op);
    }
  }
  return best_kept;
}

ParseResult parse_block_optimal(const u8* src, usize n, int level,
                                const ParseResult& greedy,
                                bool context_oracle) {
  CpuCandidateSource cand(src, n, params_for_level(level, n));
  return optimal_parse_core(src, n, level, cand, greedy, context_oracle);
}

#ifdef SPECTRA8_HAVE_GPU
// Optimal parse driven by GPU-found top-K candidates (k per position).
ParseResult parse_block_optimal_gpu(const u8* src, usize n, int level,
                                    const u16* hk_len, const u32* hk_off,
                                    int k, const ParseResult& greedy,
                                    bool context_oracle = false) {
  GpuCandidateSource cand(src, n, hk_len, hk_off, k);
  return optimal_parse_core(src, n, level, cand, greedy, context_oracle);
}
#endif

// Apply the "sub" literal transform, producing transformed literal bytes.
void make_sub_literals(const u8* src, usize n, const ParseResult& pr,
                       std::vector<u8>& out) {
  out.resize(pr.literals.size());
  RecentOffsets reps;
  reps.init();
  usize pos = 0;
  usize li = 0;
  auto emit = [&](u32 count) {
    for (u32 k = 0; k < count; ++k) {
      i32 off = reps.r[0];
      u8 raw = pr.literals[li];
      out[li] = (pos >= static_cast<usize>(off))
                    ? static_cast<u8>(raw - src[pos - off])
                    : raw;
      ++li;
      ++pos;
    }
  };
  for (const Sequence& s : pr.seqs) {
    emit(s.lit_len);
    int idx = reps.find(static_cast<i32>(s.offset));
    if (idx >= 0)
      reps.use_rep(idx);
    else
      reps.use_new(static_cast<i32>(s.offset));
    pos += s.match_len;
  }
  emit(static_cast<u32>(pr.last_literals));
  (void)n;
}

// Average order-0 entropy (bits per byte) implied by a histogram over `total`
// sampled bytes.
double entropy_bits_per_byte(const u32 freq[256], usize total) {
  if (total == 0) return 0.0;
  const double inv = 1.0 / static_cast<double>(total);
  double bits = 0.0;
  for (int s = 0; s < 256; ++s) {
    if (!freq[s]) continue;
    double p = static_cast<double>(freq[s]) * inv;
    bits -= static_cast<double>(freq[s]) * std::log2(p);
  }
  return bits * inv;
}

// Cheap strided probe for exploitable LZ repetition: returns true as soon as a
// verified >= 8-byte match is found. Only used to guard the "store" fast path
// against high-entropy-but-repetitive data, so it never needs to be exhaustive.
bool has_lz_repeats(const u8* src, usize n) {
  constexpr int kBits = 16;
  static thread_local std::vector<i32> head;
  head.assign(usize{1} << kBits, -1);
  for (usize i = 0; i + 8 <= n; i += 4) {
    u32 h = hash4(load_u32(src + i), kBits);
    i32 c = head[h];
    head[h] = static_cast<i32>(i);
    if (c >= 0 &&
        match_len(src + i, src + static_cast<usize>(c), src + n) >= 8)
      return true;
  }
  return false;
}

// Heuristic incompressibility test. Samples a few windows for an order-0
// entropy estimate (near-free on normal data); only when that estimate is
// near-maximal does it run the strided repeat probe. Returns true only for
// blocks that neither entropy coding nor LZ can meaningfully shrink.
bool block_is_incompressible(const u8* src, usize n) {
  if (n < (usize{1} << 16)) return false;  // not worth it for small blocks
  constexpr usize kWin = 4096;
  const usize starts[3] = {0, (n - kWin) / 2, n - kWin};
  u32 freq[256] = {0};
  for (usize s : starts)
    for (usize i = 0; i < kWin; ++i) ++freq[src[s + i]];
  // Below ~7.85 bits/byte there is enough order-0 structure to be worth coding.
  if (entropy_bits_per_byte(freq, 3 * kWin) < 7.85) return false;
  return !has_lz_repeats(src, n);
}

// True when a block looks like post-shuffle / post-delta / post-dict data:
// long runs of identical bytes (high-byte planes, dense id lanes). On that
// shape BWT+MTF loses its edge to plain LZ (the same structure LZMA milks
// after an external filter). Used only to gate an LZ trial under Codec::Prism.
bool block_looks_run_heavy(const u8* src, usize n) noexcept {
  if (n < 256) return false;
  constexpr usize kSample = usize{1} << 14;
  const usize sn = min_of(n, kSample);
  const usize start = (n > sn) ? (n - sn) / 2 : 0;
  const u8* s = src + start;
  usize same = 0;
  usize long_run = 0;  // bytes belonging to a run of length >= 8
  usize run = 1;
  for (usize i = 1; i < sn; ++i) {
    if (s[i] == s[i - 1]) {
      ++same;
      ++run;
    } else {
      if (run >= 8) long_run += run;
      run = 1;
    }
  }
  if (run >= 8) long_run += run;
  // ~35% adjacent equals *or* ~20% of the sample in long runs — typical after
  // byte-shuffle on numeric columns; natural text is usually well below both.
  return same * 100 >= (sn - 1) * 35 || long_run * 5 >= sn;
}

// GPU path tuning: 64 KiB blocks fit the threadgroup hash table; the kernel
// caps reported match length, and the CPU extends it during the parse.
constexpr int kGpuBlockSizeLog = 16;
[[maybe_unused]] constexpr int kGpuMaxMatch = 64;
// Pipeline granularity: the GPU finds matches for one segment (this many 64 KiB
// blocks, ~1 MiB) while the CPU entropy-codes the previous segment.
[[maybe_unused]] constexpr usize kGpuSegmentBlocks = 16;

// Greedy parse driven by GPU per-position match hints (block-relative length
// and back-distance per byte). The recent-offset model and CPU match-length
// extension still run, so the output remains a valid Spectra8 block; only candidate
// discovery moved to the GPU.
ParseResult parse_block_from_hints(const u8* src, usize n, const u16* hint_len,
                                   const u32* hint_off) {
  ParseResult res;
  const int min_rep = min_rep_for_block(n);
  RecentOffsets reps;
  reps.init();
  usize anchor = 0;
  usize pos = 0;
  while (pos < n) {
    int len = 0;
    int rep_idx = -1;
    i32 off = 0;
    // Repeat-offset candidates (cheap; preserves the recent-offset model).
    for (int i = 0; i < kNumRecentOffsets; ++i) {
      i32 ro = reps.r[i];
      if (ro <= 0 || static_cast<usize>(ro) > pos) continue;
      usize l = match_len(src + pos, src + pos - ro, src + n);
      if (static_cast<int>(l) >= min_rep && static_cast<int>(l) > len) {
        len = static_cast<int>(l);
        off = ro;
        rep_idx = i;
      }
    }
    // GPU-found new match, with the true length recomputed on the CPU.
    if (hint_len[pos] >= kMinMatch) {
      i32 ho = static_cast<i32>(hint_off[pos]);
      if (ho > 0 && static_cast<usize>(ho) <= pos) {
        usize l = match_len(src + pos, src + pos - ho, src + n);
        if (static_cast<int>(l) >= kMinMatch) {
          bool take_new = (rep_idx < 0) ? true : (static_cast<int>(l) > len + 1);
          if (take_new) {
            len = static_cast<int>(l);
            off = ho;
            rep_idx = -1;
          }
        }
      }
    }
    int min_needed = (rep_idx >= 0) ? min_rep : kMinMatch;
    if (len < min_needed) {
      ++pos;
      continue;
    }
    usize lit_len = pos - anchor;
    res.literals.insert(res.literals.end(), src + anchor, src + pos);
    res.seqs.push_back({static_cast<u32>(lit_len), static_cast<u32>(len),
                        static_cast<u32>(off)});
    if (rep_idx >= 0)
      reps.use_rep(rep_idx);
    else
      reps.use_new(off);
    pos += static_cast<usize>(len);
    anchor = pos;
  }
  res.last_literals = n - anchor;
  res.literals.insert(res.literals.end(), src + anchor, src + n);
  return res;
}

// Entropy-code and serialize an already-parsed block into `payload`.
void serialize_parse_result(const u8* src, usize n, int level,
                            const ParseResult& pr, std::vector<u8>& payload,
                            bool context_oracle = false) {
  const usize num_seq = pr.seqs.size();
  std::vector<u8> lit_len_stream, match_len_stream, off_code_stream;
  std::vector<u8> lit_len_extra, match_len_extra;
  std::vector<u8> off_extra_bytes;
  lit_len_stream.reserve(num_seq);
  match_len_stream.reserve(num_seq);
  off_code_stream.reserve(num_seq);

  // Low offset bits (see kBlockOffLowShift): pick L in 0..kMaxOffLowBits by
  // the order-0 entropy of the low bits of the new offsets versus their raw
  // cost, then write the raw bit stream with those bits removed.
  std::vector<u8> off_low_stream;
  int off_low = 0;
  {
    // Token streams are written by index into pre-sized buffers (the
    // push_back capacity checks were a measurable share of serialize at
    // L1); the rare >= 255 escapes append to the extra blobs.
    lit_len_stream.resize(num_seq);
    match_len_stream.resize(num_seq);
    off_code_stream.resize(num_seq);
    std::vector<u32> new_offs;
    new_offs.reserve(num_seq);
    RecentOffsets reps;
    reps.init();
    // Low-bit histograms of the new offsets by bit-width class, gathered in
    // the same pass: k == 1 and k == 2 offsets only qualify for L <= k, so
    // the per-L tables are sums over classes.
    u32 h_k1[2] = {}, h_k2[4] = {}, h_k3[8] = {};
    const Sequence* seqs = pr.seqs.data();
    u8* ll = lit_len_stream.data();
    u8* ml = match_len_stream.data();
    u8* oc = off_code_stream.data();
    for (usize i = 0; i < num_seq; ++i) {
      const Sequence& s = seqs[i];
      if (__builtin_expect(s.lit_len >= 255, 0)) {
        ll[i] = 255;
        ByteWriter(lit_len_extra).varint(s.lit_len - 255);
      } else {
        ll[i] = static_cast<u8>(s.lit_len);
      }
      const u32 mlv = s.match_len - kMinRepMatch;
      if (__builtin_expect(mlv >= 255, 0)) {
        ml[i] = 255;
        ByteWriter(match_len_extra).varint(mlv - 255);
      } else {
        ml[i] = static_cast<u8>(mlv);
      }

      const int idx = reps.find(static_cast<i32>(s.offset));
      if (idx >= 0) {
        oc[i] = static_cast<u8>(idx);
        reps.use_rep(idx);
      } else {
        const u32 o = s.offset;
        const int k = bit_width(o) - 1;
        oc[i] = static_cast<u8>(kNumRecentOffsets + k);
        new_offs.push_back(o);
        reps.use_new(static_cast<i32>(o));
        if (k >= 3)
          ++h_k3[o & 7u];
        else if (k == 2)
          ++h_k2[o & 3u];
        else if (k == 1)
          ++h_k1[o & 1u];
      }
    }

    if (new_offs.size() >= 256) {
      // Saving in bytes for each L: raw bits removed minus the token stream's
      // entropy, minus one array header. Requires a real gain (>= 0.5% of the
      // sequence count in bytes): the decoder pays one token per new offset.
      static_assert(kMaxOffLowBits == 3, "histogram classes are unrolled for 3");
      double best_gain = static_cast<double>(num_seq) / 200.0 + 48.0;
      for (int L = 1; L <= kMaxOffLowBits; ++L) {
        u32 h[1 << kMaxOffLowBits] = {};
        const u32 m = (1u << L) - 1u;
        for (int v = 0; v < 8; ++v) h[v & m] += h_k3[v];
        if (L <= 2)
          for (int v = 0; v < 4; ++v) h[v & m] += h_k2[v];
        if (L <= 1)
          for (int v = 0; v < 2; ++v) h[v & m] += h_k1[v];
        usize cnt = 0;
        for (int v = 0; v < (1 << L); ++v) cnt += h[v];
        if (cnt == 0) continue;
        double bits = 0.0;
        for (int v = 0; v < (1 << L); ++v)
          if (h[v])
            bits -= static_cast<double>(h[v]) *
                    std::log2(static_cast<double>(h[v]) / static_cast<double>(cnt));
        const double gain = (static_cast<double>(cnt) * L - bits) / 8.0;
        if (gain > best_gain) {
          best_gain = gain;
          off_low = L;
        }
      }
    }

    BitWriter off_bits(off_extra_bytes);
    if (off_low > 0) {
      off_low_stream.reserve(new_offs.size());
      const u32 low_mask = (1u << off_low) - 1u;
      for (u32 o : new_offs) {
        const int k = bit_width(o) - 1;
        if (k >= off_low) {
          off_low_stream.push_back(static_cast<u8>(o & low_mask));
          const int kk = k - off_low;
          const u32 mask = (kk == 0) ? 0 : ((1u << kk) - 1u);
          off_bits.put((o >> off_low) & mask, kk);
        } else {
          const u32 mask = (k == 0) ? 0 : ((1u << k) - 1u);
          off_bits.put(o & mask, k);
        }
      }
    } else {
      for (u32 o : new_offs) {
        const int k = bit_width(o) - 1;
        const u32 mask = (k == 0) ? 0 : ((1u << k) - 1u);
        off_bits.put(o & mask, k);
      }
    }
    off_bits.flush();
  }

  // Choose the cheaper literal model.
  LiteralMode lit_mode = LiteralMode::Raw;
  std::vector<u8> lit_raw_enc;
  encode_array(lit_raw_enc, pr.literals.data(), pr.literals.size(), level);
  std::vector<u8>* chosen_lit_enc = &lit_raw_enc;
  std::vector<u8> lit_sub_enc;
  // L5 on small pages: sub is a second literal transform + Huffman for a
  // rare win after the external verdict filter already ran. Skip it.
  if (level >= 3 && !pr.literals.empty() &&
      !(level == 5 && n <= kSmallBlock)) {
    std::vector<u8> sub;
    make_sub_literals(src, n, pr, sub);
    // Huffman cannot beat its entropy bound. If the sub model is not
    // cheaper on entropy, skip building its tree (the second encode was
    // a measurable L5 cost on files that stay on raw literals).
    u32 fr[256] = {}, fs[256] = {};
    for (u8 b : pr.literals) ++fr[b];
    for (u8 b : sub) ++fs[b];
    if (entropy_bits_per_byte(fs, sub.size()) <
        entropy_bits_per_byte(fr, pr.literals.size()))
      encode_array(lit_sub_enc, sub.data(), sub.size(), level);
    // Sub literals decode slower than raw ones (each byte depends on an
    // earlier output byte; only far offsets vectorize), so require a
    // material win - ~1% of the coded literal array - before taking them.
    // Where Sub matters (record-structured binary, images) it wins 5-6%; the
    // sub-1% cases traded ~8% decode speed for a few hundred bytes.
    if (!lit_sub_enc.empty() &&
        lit_sub_enc.size() + (lit_sub_enc.size() >> 7) < lit_raw_enc.size()) {
      chosen_lit_enc = &lit_sub_enc;
      lit_mode = LiteralMode::Sub;
    }
  }

  // Crystal static coder on the literal bytes (same array blob, tag
  // kCrystal; decode_array handles it). Literals after a good parse are the
  // hard-to-predict residue, but on record data / text they still carry
  // order-1/2 and lattice structure that order-0 Huffman cannot see: -2..-10%
  // of the literal array on the verdict planes, -8% on text. Its decoder
  // runs at ~600 MB/s against multi-GB/s Huffman, so require a real win on
  // the whole block: 4% of the payload from L5, 1.5% from L9 (token
  // streams + offset bits are estimated from their raw sizes here, the
  // literal array is exact).
  std::vector<u8> lit_cry_enc;
  const usize min_lit_size = context_oracle ? 512 : 4096;
  if (level >= 5 && pr.literals.size() >= min_lit_size) {
    const usize cur = chosen_lit_enc->size();
    const usize block_est = cur + num_seq * 2 + off_extra_bytes.size() +
                            lit_len_extra.size() + match_len_extra.size();
    const usize min_gain = context_oracle ? 4 : (level >= 9 ? (block_est * 3) / 200 : block_est / 25);
    // Cheap screen before the (slow) Crystal model search: the order-1
    // conditional entropy of the literals bounds what its C1 context can buy
    // over order-0 Huffman. Crystal L1 realises only part of that bound (16
    // context groups, table headers): on the verdict planes and mozilla
    // every block that ended up taking Crystal had a bound of >= 3.3x the
    // required gain, and the median winner realised 0.7-0.85 of it. Require
    // 2x; below that the search (~1.2 ms per 128 KiB block) never paid off.
    // At L9+ the trial runs Crystal L5, whose text pipeline (BWT) is not
    // bounded by order-1 at all: on enwik8 the 2x screen cost 2 % of the L9
    // and L11 ratio, so those levels keep the original 0.87x screen (which
    // never rejected a winning block there).
    bool plausible = false;
    const int stride = detect_stride(pr.literals.data(), pr.literals.size());
    const int cry_level = (context_oracle || level >= 9) ? 5 : 1;
    const double screen = cry_level == 1 ? 0.5 : 1.15;
    if (min_gain < cur) {
      if (context_oracle) {
        plausible = true;
      } else {
        const u8* L = pr.literals.data();
        const usize ln = pr.literals.size();
        static thread_local std::vector<u32> h1;
        // x*log2(x) by table for the counts a 128 KiB block can produce;
        // the std::log2 calls were most of the screen's cost.
        static thread_local std::vector<float> xl;
        constexpr usize kXl = usize{1} << 17;
        if (xl.empty()) {
          xl.resize(kXl + 1);
          xl[0] = 0.0f;
          for (usize x = 1; x <= kXl; ++x)
            xl[x] = static_cast<float>(x * std::log2(static_cast<double>(x)));
        }
        // Counts are bounded by ln; the table covers every block up to
        // 128 KiB and the rare larger literal array falls back to log2.
        const bool tab = ln <= kXl;
        auto xlogx = [&](u32 x) -> double {
          return tab ? xl[x] : x * std::log2(static_cast<double>(x ? x : 1));
        };
        u32 h0[256] = {};
        for (usize i = 0; i < ln; ++i) ++h0[L[i]];
        double b0 = xlogx(static_cast<u32>(ln));
        for (int c = 0; c < 256; ++c) b0 -= xlogx(h0[c]);
        // Bytes saved by conditioning on the byte `d` back, versus order-0.
        auto cond_gain = [&](usize d) {
          h1.assign(256 * 256, 0);
          for (usize i = d; i < ln; ++i)
            ++h1[static_cast<usize>(L[i - d]) * 256 + L[i]];
          double b1 = 0.0;
          for (int c = 0; c < 256; ++c) {
            // Context c occurs h0[c] times (minus the last d positions);
            // an unused context has an empty row.
            if (!h0[c]) continue;
            const u32* row = &h1[static_cast<usize>(c) * 256];
            u32 rt = 0;
            float sx0 = 0.0f, sx1 = 0.0f;
            if (tab) {
              for (int s = 0; s < 256; s += 2) {
                rt += row[s] + row[s + 1];
                sx0 += xl[row[s]];
                sx1 += xl[row[s + 1]];
              }
            } else {
              for (int s = 0; s < 256; ++s) {
                rt += row[s];
                sx0 += static_cast<float>(xlogx(row[s]));
              }
            }
            b1 += xlogx(rt) - (static_cast<double>(sx0) + sx1);
          }
          return (b0 - b1) / 8.0;
        };
        double g = cond_gain(1);
        if (stride > 1) g = std::max(g, cond_gain(static_cast<usize>(stride)));
        plausible = g * screen >= static_cast<double>(min_gain);
      }
    }
    if (plausible) {
      const usize budget = cur - min_gain;
      if (encode_array_crystal(lit_cry_enc, pr.literals.data(),
                               pr.literals.size(), cry_level, stride, budget)) {
        chosen_lit_enc = &lit_cry_enc;
        lit_mode = LiteralMode::Raw;
      }
    }
  }

  // Token-context conditioning (see kBlockFlagMlCond / kBlockFlagOcCond):
  // split the match-length and offset-code streams by context and keep the
  // split when the summed conditional entropies plus the extra table headers
  // beat the plain stream's entropy. Each extra context array is another
  // decode_array (full table build); on a 128 KiB block that tax was ~30%
  // of decode for a few dozen bytes. Skip the bucket/entropy work unless
  // there are enough tokens to amortize the tables.
  constexpr usize kMinCondSeq = 4096;
  constexpr double kCondArrayTax = 24.0;
  bool ml_cond = false;
  bool oc_cond = false;
  std::vector<u8> ml_ctx[kLlBuckets], oc_ctx[kOcContexts];
  const usize min_cond_thresh = context_oracle ? 512 : kMinCondSeq;
  if (num_seq >= min_cond_thresh && level >= 6) {
    for (usize i = 0; i < num_seq; ++i) {
      const u32 lt = lit_len_stream[i];
      const u32 mt = match_len_stream[i];
      ml_ctx[ll_bucket(lt)].push_back(static_cast<u8>(mt));
      oc_ctx[oc_context(mt, lt)].push_back(off_code_stream[i]);
    }
    auto array_cost = [](const std::vector<u8>& a) {
      if (a.empty()) return 2.0;
      u32 h[256] = {0};
      for (u8 b : a) ++h[b];
      int distinct = 0;
      double bits = 0.0;
      const double dn = static_cast<double>(a.size());
      for (int i = 0; i < 256; ++i)
        if (h[i]) {
          ++distinct;
          bits -= static_cast<double>(h[i]) *
                  std::log2(static_cast<double>(h[i]) / dn);
        }
      // payload + table header (num_syms, packed lengths) + array framing
      return bits / 8.0 + 2.0 + 0.5 * distinct + 4.0;
    };
    double ml_plain = array_cost(match_len_stream), ml_split = 0.0;
    for (const auto& a : ml_ctx) ml_split += array_cost(a);
    double oc_plain = array_cost(off_code_stream), oc_split = 0.0;
    for (const auto& a : oc_ctx) oc_split += array_cost(a);
    const double cond_tax = context_oracle ? 8.0 : kCondArrayTax;
    ml_cond = ml_split + cond_tax * (kLlBuckets - 1) < ml_plain;
    oc_cond = oc_split + cond_tax * (kOcContexts - 1) < oc_plain;
  }

  ByteWriter w(payload);
  w.byte(static_cast<u8>(static_cast<u8>(lit_mode) |
                         (ml_cond ? kBlockFlagMlCond : 0) |
                         (oc_cond ? kBlockFlagOcCond : 0) |
                         static_cast<u8>(off_low << kBlockOffLowShift)));
  w.varint(num_seq);
  w.varint(pr.last_literals);
  encode_array(payload, lit_len_stream.data(), lit_len_stream.size(), level,
               ArrayKind::Tokens);
  if (ml_cond) {
    for (const auto& a : ml_ctx)
      encode_array(payload, a.data(), a.size(), level, ArrayKind::Tokens);
  } else {
    encode_array(payload, match_len_stream.data(), match_len_stream.size(),
                 level, ArrayKind::Tokens);
  }
  if (oc_cond) {
    for (const auto& a : oc_ctx)
      encode_array(payload, a.data(), a.size(), level, ArrayKind::Tokens);
  } else {
    encode_array(payload, off_code_stream.data(), off_code_stream.size(),
                 level, ArrayKind::Tokens);
  }
  if (off_low > 0)
    encode_array(payload, off_low_stream.data(), off_low_stream.size(), level,
                 ArrayKind::Tokens);
  put_blob(payload, lit_len_extra);
  put_blob(payload, match_len_extra);
  put_blob(payload, off_extra_bytes);
  payload.insert(payload.end(), chosen_lit_enc->begin(), chosen_lit_enc->end());

  // Low match coverage (< 60% of the block) on record data: the parse's
  // short matches (2-3 byte repeats at L9+, stride copies) shred the lattice
  // that the Crystal literal coder needs, and the block as one literal run
  // can be much smaller (gau-i16 at L9: 623 -> 581 KB). Serialize that
  // variant too and keep the smaller payload; the recursion ends because
  // the all-literal parse has no sequences.
  if (level >= 5 && num_seq > 0 && n >= 4096 &&
      n - pr.literals.size() < (n * 3) / 5 &&
      detect_stride(src, n) > 1) {
    ParseResult flat;
    flat.literals.assign(src, src + n);
    flat.last_literals = n;
    std::vector<u8> alt;
    serialize_parse_result(src, n, level, flat, alt);
    if (alt.size() < payload.size()) payload.swap(alt);
  }
}

// Blocks smaller than this never use BWT (the primary-index + array framing
// would dominate); they fall back to a stored block.
constexpr usize kBwtMinBlock = 64;

// Per-segment table-selection granularity for the BWT/MTF symbol stream. BWT
// output is piecewise stationary (similar contexts cluster), so coding fixed
// segments independently lets each pick a table tuned to its local statistics.
constexpr usize kBwtSegSize = usize{1} << 16;  // 64 KiB

// Blocks at or below this skip per-segment entropy trials: at ~128 KiB
// the extra encode and a second table cost more than they buy, and the
// 32 KiB-histogram estimate often ranks the worse split. (Encoder-only;
// the decoder is segment-size agnostic.)
constexpr usize kBwtSmallBlock = usize{1} << 17;  // 128 KiB

// Order-0 entropy of a histogram, in bytes.
double hist_entropy_bytes(const u32 hist[256], usize n) {
  if (n == 0) return 0.0;
  double bits = 0.0;
  const double dn = static_cast<double>(n);
  for (int s = 0; s < 256; ++s)
    if (hist[s])
      bits -= static_cast<double>(hist[s]) *
              std::log2(static_cast<double>(hist[s]) / dn);
  return bits / 8.0;
}

// Per-segment framing + coder overhead used by the segmentation estimate:
// varints + tag (~7 bytes), the range coder's 5-byte flush, and the adaptive
// model's learning cost, which scales with how much distribution there is to
// learn. The 8 bytes/distinct-symbol term was calibrated on silesia (swept
// 2/4/6/8; 8 tracked the real coder best). The estimate only picks a
// segmentation - mis-ranking costs a fraction of a percent of ratio.
double seg_overhead_bytes(int distinct) {
  return 12.0 + 8.0 * static_cast<double>(distinct);
}

// Estimated coded size (bytes) of sym[0, n) split into `seg_size` segments
// (0 = whole stream), from per-32KiB histograms merged on the fly.
double estimate_segmented(const std::vector<std::array<u32, 256>>& h32,
                          usize n, usize seg_size) {
  constexpr usize k32 = kBwtSegSize >> 1;
  if (seg_size == 0) seg_size = n;
  const usize step = (seg_size + k32 - 1) / k32;  // 32K histograms per segment
  double est = 0.0;
  u32 merged[256];
  for (usize b = 0; b < h32.size(); b += step) {
    std::memset(merged, 0, sizeof(merged));
    const usize hi = min_of(b + step, h32.size());
    usize seg_n = 0;
    for (usize k = b; k < hi; ++k)
      for (int s = 0; s < 256; ++s) merged[s] += h32[k][s];
    for (int s = 0; s < 256; ++s) seg_n += merged[s];
    int distinct = 0;
    for (int s = 0; s < 256; ++s) distinct += merged[s] != 0;
    est += hist_entropy_bytes(merged, seg_n) + seg_overhead_bytes(distinct);
  }
  return est;
}

// Chosen segmentation for one symbol stream: best segment size, an optional
// too-close-to-call runner-up, and the best estimated coded size.
struct SegChoice {
  usize seg = 0;
  usize second = ~usize{0};
  double est = 0.0;
};

// Rank candidate segment sizes (0 = whole stream) by estimated coded size.
// Returns the best; when the runner-up estimate is within ~1% (estimate noise:
// the model ignores adaptation across segment boundaries), it is reported too
// so the caller can encode both and keep the actual smaller.
SegChoice pick_segmentation(const u8* sym, usize n, int level) {
  constexpr usize k32 = kBwtSegSize >> 1;
  constexpr usize k128 = kBwtSegSize << 1;
  constexpr usize k256 = kBwtSegSize << 2;

  // Per-32KiB histograms: one cheap pass over the stream, then every candidate
  // segmentation is estimated by merging them (no extra passes over the data).
  std::vector<std::array<u32, 256>> h32((n + k32 - 1) / k32);
  for (auto& h : h32) h.fill(0);
  for (usize i = 0; i < n; ++i) ++h32[i / k32][sym[i]];

  SegChoice out;
  double best = estimate_segmented(h32, n, 0);
  usize second_seg = ~usize{0};
  double second_est = 0.0;
  const usize cand[] = {k256, k128, kBwtSegSize, k32};
  const int max_try = level <= 5 ? 2 : (level <= 7 ? 3 : 4);

  // Official 128 KiB: keep whole-stream as the primary. The 32 KiB
  // histogram often ranks a worse 64/32 KiB split (poi-i16 / gau-i16);
  // L6+ still races that split when it estimates smaller. Larger blocks
  // may promote a split to primary.
  if (n >= kBwtSmallBlock && level > 3) {
    int tried = 0;
    for (usize ci = 0; ci < 4 && tried < max_try; ++ci) {
      if (n <= cand[ci]) continue;
      ++tried;
      const double e = estimate_segmented(h32, n, cand[ci]);
      if (n == kBwtSmallBlock) {
        if (e < best && (second_seg == ~usize{0} || e < second_est)) {
          second_seg = cand[ci];
          second_est = e;
        }
        continue;
      }
      if (e < best) {
        second_seg = out.seg;
        second_est = best;
        best = e;
        out.seg = cand[ci];
      } else if (second_seg == ~usize{0} || e < second_est) {
        second_seg = cand[ci];
        second_est = e;
      }
    }
  }
  // 128 KiB: L6+ races a split that estimates smaller than whole-stream.
  // Larger blocks: race a 1%-close runner-up (all levels).
  if (second_seg != ~usize{0} &&
      ((n == kBwtSmallBlock && level > 5 && second_est < best) ||
       (n != kBwtSmallBlock && second_est <= best * 1.01)))
    out.second = second_seg;
  out.est = best;
  return out;
}

// Frame an LZ or zstd payload as a BWT symbol stream (seg_mode 2/3).
void frame_bwt_fast(std::vector<u8>& out, u8 kind, usize n,
                    const u8* body, usize body_n) {
  out.push_back(kind);
  ByteWriter w(out);
  w.varint(n);
  w.varint(body_n);
  w.bytes(body, body_n);
}

// Spectra8 LZ L5 plus zstd L5. Either can beat the adaptive range coder on
// post-MTF streams and both decode much faster. Last-step LZ stays at L5
// even when the BWT block is L9: an L9 parse of the MTF stream was the
// encode-cost cliff and did not shrink official frames vs L5 LZ.
void encode_bwt_fast_bodies(const u8* sym, usize n, int /*level*/,
                            std::vector<u8>& lz, std::vector<u8>& zs) {
  lz.clear();
  zs.clear();
  if (n < 16) return;
  constexpr int kFastLevel = 5;
  const ParseResult pr = parse_block(sym, n, kFastLevel);
  serialize_parse_result(sym, n, kFastLevel, pr, lz);
  if (!zstd_available()) return;
  thread_local std::vector<u8> zbuf;
  zbuf.resize(zstd_compress_bound(n));
  const auto zr = zstd_compress(
      std::span<const std::byte>(reinterpret_cast<const std::byte*>(sym), n),
      std::span<std::byte>(reinterpret_cast<std::byte*>(zbuf.data()),
                           zbuf.size()),
      5);
  if (zr.ok() && zr.value > 0) zs.assign(zbuf.data(), zbuf.data() + zr.value);
}

// Frame the smaller of LZ / zstd. Returns false when neither produced a body.
bool frame_best_bwt_fast(std::vector<u8>& out, const u8* sym, usize n,
                         int level) {
  thread_local std::vector<u8> lz, zs;
  encode_bwt_fast_bodies(sym, n, level, lz, zs);
  const std::vector<u8>* body = nullptr;
  u8 kind = 0;
  if (!lz.empty() && (zs.empty() || lz.size() <= zs.size())) {
    body = &lz;
    kind = 2;
  } else if (!zs.empty()) {
    body = &zs;
    kind = 3;
  }
  if (!body) return false;
  frame_bwt_fast(out, kind, n, body->data(), body->size());
  return true;
}

// Whole-stream last step after BWT+MTF/RLE: race the existing adaptive
// entropy coder against Spectra8 LZ L5 and zstd L5. LZ/zstd decode several
// times faster; they are kept when they are not larger.
bool encode_bwt_stream_raced(std::vector<u8>& out, const u8* sym, usize n,
                             int level) {
  thread_local std::vector<u8> ent, fast;
  ent.clear();
  encode_bwt_entropy(ent, sym, n, level);
  fast.clear();
  if (frame_best_bwt_fast(fast, sym, n, level) && fast.size() <= 1 + ent.size()) {
    out.insert(out.end(), fast.begin(), fast.end());
    return true;
  }
  out.push_back(0);
  out.insert(out.end(), ent.begin(), ent.end());
  return false;
}

// One segmented (or whole-stream, seg_size 0) encode of sym[0, n).
bool encode_bwt_symbols_seg(std::vector<u8>& out, const u8* sym, usize n,
                            int level, usize seg_size) {
  if (seg_size == 0 || n <= seg_size)
    return encode_bwt_stream_raced(out, sym, n, level);
  out.push_back(1);
  ByteWriter w(out);
  w.varint((n + seg_size - 1) / seg_size);
  for (usize off = 0; off < n; off += seg_size)
    encode_bwt_entropy(out, sym + off, min_of(seg_size, n - off), level);
  return false;
}

// Entropy-code a post-MTF/RLE symbol stream as either
//   seg_mode 0: one whole-stream encode, or
//   seg_mode 1: varint(num_segs) + one encode per fixed-size segment.
// Self-framed (leading seg_mode byte). The segmentation is chosen up front
// from per-segment entropy estimates (pick_segmentation) instead of encoding
// every candidate in full and keeping the smallest, which made the range coder
// the top encode hotspot. Only when the top two estimates are within ~1% are
// both encoded (the estimate cannot reliably separate them). The decoder
// concatenates the self-framed arrays, so segment choice is purely an encoder
// size/speed knob.
bool encode_bwt_symbols_best(std::vector<u8>& out, const u8* sym, usize n,
                             int level) {
  const SegChoice c = pick_segmentation(sym, n, level);
  const usize base = out.size();
  bool used_fast = encode_bwt_symbols_seg(out, sym, n, level, c.seg);
  if (c.second != ~usize{0}) {
    std::vector<u8> alt;
    const bool alt_fast = encode_bwt_symbols_seg(alt, sym, n, level, c.second);
    if (alt.size() < out.size() - base) {
      out.resize(base);
      out.insert(out.end(), alt.begin(), alt.end());
      used_fast = alt_fast;
    }
  }
  // Segmented entropy (mode 1) never races LZ/zstd. On 1 MiB official
  // blocks the primary is often a split; still try a whole-stream fast
  // last step so zstd L5 / Spectra8 LZ can displace it when they are smaller.
  if (c.seg != 0 && c.second != 0) {
    thread_local std::vector<u8> fast;
    fast.clear();
    if (frame_best_bwt_fast(fast, sym, n, level)) {
      const usize cur = out.size() - base;
      if (fast.size() <= cur) {
        out.resize(base);
        out.insert(out.end(), fast.begin(), fast.end());
        used_fast = true;
      }
    }
  }
  return used_fast;
}

// Collapse each maximal zero-run in the MTF stream to a single 0 in `sym`, and
// push the run length into a (byte-stream + varint-escape) length code. This is
// the RLE0 stage: long zero-runs cost ~log instead of ~length under entropy
// coding, which the static order-0 coder cannot achieve on its own.
void bwt_rle0_split(const u8* mtf, usize n, std::vector<u8>& sym,
                    std::vector<u8>& runlen, std::vector<u8>& run_extra) {
  // Upper bounds: one sym byte per input; runlen is shorter than n.
  sym.reserve(n);
  runlen.reserve(n >> 2);
  usize i = 0;
  while (i < n) {
    if (mtf[i] == 0) {
      const usize run = count_eq_run(mtf + i, n - i, 0);
      sym.push_back(0);
      put_len(static_cast<u32>(run), runlen, run_extra);
      i += run;
    } else {
      const usize run = count_neq_run(mtf + i, n - i, 0);
      sym.insert(sym.end(), mtf + i, mtf + i + run);
      i += run;
    }
  }
}

// Build a BWT + MTF + RLE0 + entropy payload (no block header) for src[0, n).
// Keeps the smaller of (depending on level):
//   mode 0: the MTF stream coded directly (no RLE0), and
//   mode 1: RLE0 (zero-runs collapsed) + run-length side stream.
// Both route the symbol stream through encode_bwt_symbols (level-dependent
// per-segment tables).
// Blocks at least this large emit parallel-inverse anchors (multi-chain LF
// walks). The per-block cost is just a handful of anchor varints, so even
// ~128 KiB blocks (a common target size) benefit: the single-chain inverse is a
// latency-bound pointer chase, and overlapping several walks hides that latency.
constexpr usize kBwtParallelMin = 64 * 1024;

// Cheap coded-size estimate of one stream, used only to rank the mode
// candidates below (the winner margin is validated with real encodes).
// When the entropy stage has its order-1 coders available (L7+), estimates
// *order-1* entropy over the coder's context classes: with the order-0
// bound, the plain-MTF mode - whose run structure order-1 coding exploits
// but RLE0 pre-digests - was systematically over-estimated and pruned.
// Below L7 the order-0 bound matches what the stage can actually achieve.
double bwt_stream_est(const u8* s, usize n, bool o1) {
  if (!o1) {
    u32 h[256];
    hist_u8(s, n, h);
    int distinct = 0;
    for (int i = 0; i < 256; ++i) distinct += h[i] != 0;
    return hist_entropy_bytes(h, n) + seg_overhead_bytes(distinct);
  }
  u32 h[ans::kO1Ctx][256] = {};
  u8 prev = 0;
  usize i = 0;
  for (; i + 4 <= n; i += 4) {
    const u8 a = s[i], b = s[i + 1], c = s[i + 2], d = s[i + 3];
    ++h[ans::o1_class(prev)][a];
    ++h[ans::o1_class(a)][b];
    ++h[ans::o1_class(b)][c];
    ++h[ans::o1_class(c)][d];
    prev = d;
  }
  for (; i < n; ++i) {
    ++h[ans::o1_class(prev)][s[i]];
    prev = s[i];
  }
  double est = 0.0;
  int distinct = 0;
  u32 tot[256] = {};
  for (int c = 0; c < ans::kO1Ctx; ++c) {
    usize cn = 0;
    for (int i = 0; i < 256; ++i) cn += h[c][i];
    if (cn) est += hist_entropy_bytes(h[c], cn);
    for (int i = 0; i < 256; ++i) tot[i] += h[c][i];
  }
  for (int i = 0; i < 256; ++i) distinct += tot[i] != 0;
  return est + seg_overhead_bytes(distinct);
}

void build_bwt_single(const u8* src, usize n, int level,
                      std::vector<u8>& payload, bool* used_fast = nullptr,
                      bool has_lzp = false) {
  const bool small = n <= kBwtSmallBlock;
  // Order-1 estimates track AnsO1 (L5+) / adaptive O1 (L7+). Below L5 the
  // stage is order-0-only and the estimate must match or plain-MTF is
  // over-ranked.
  const bool o1_est = level >= 5;
  // With AnsO1 (L5+), plain MTF is competitive: the in-run context class
  // captures most of what RLE0 pre-digests. The order-1 estimate prunes it
  // when it cannot compete. Below L5 keep the older L6-on-large-blocks trial.
  const bool try_plain = level >= 5 || (level >= 6 && !small);

  thread_local std::vector<u8> tbuf;
  tbuf.resize(n);
  // Emit anchors for a 16-chain parallel inverse when the block is large enough.
  const bool parallel = (n >= kBwtParallelMin) && (n + 1 <= kBwtMaxBlock);
  const int num_chains = parallel ? bwt_encode_chains() : 1;
  usize anchors[kBwtMaxChains] = {0};
  const usize primary = bwt_forward(src, n, tbuf.data(),
                                    parallel ? anchors : nullptr, num_chains);
  // Skip-MTF (mode 3): entropy-code the L-column directly. On official
  // numeric planes this beats MTF+adaptive by 1–3% and needs no inverse MTF.
  // Kept as a raced candidate (binary still wants MTF+zstd).
  const bool try_raw = level >= 5;
  thread_local std::vector<u8> lcol;
  if (try_raw) {
    lcol.resize(n);
    std::memcpy(lcol.data(), tbuf.data(), n);
  }

  // Mode 4: BWT Context Mixer on raw L-column (no MTF).
  // Excels on natural text and source code.
  std::vector<u8> cm;
  const bool try_cm = level >= 7 && try_raw;
  if (try_cm) {
    cm.push_back(4);  // mode 4
    if (!bwt_cm_encode(lcol.data(), n, cm)) {
      cm.clear();
    }
  }

  // Mode 5: BWT Ultra (Bidirectional Context & Multi-State Run Model).
  // Activated at level 11 (Ultra mode).
  const bool try_ultra = (level >= 11) && try_raw;
  if (try_ultra) {
    std::vector<u8> ultra;
    ultra.push_back(5);  // mode 5
    if (bwt_ultra_encode(lcol.data(), n, ultra)) {
      if (cm.empty() || ultra.size() < cm.size()) {
        cm.swap(ultra);
      }
    }
  }

  // Fast path for LZP blocks: CM dominates on structured text / code.
  // Avoid computing expensive order-1 Shannon entropy estimates and all MTF/ANS passes.
  if (has_lzp && !cm.empty()) {
    ByteWriter w(payload);
    w.varint(primary);
    w.byte(static_cast<u8>(num_chains));
    if (num_chains > 1)
      for (int c = 0; c < num_chains; ++c) w.varint(anchors[c]);
    const u8 mode_byte = cm[0] | 0x80;
    w.byte(mode_byte);
    w.varint(n);  // intermediate uncompressed BWT length
    w.bytes(cm.data() + 1, cm.size() - 1);
    if (used_fast) *used_fast = false;
    return;
  }

  const double est_raw =
      try_raw ? bwt_stream_est(tbuf.data(), n, o1_est) : 1e30;

  // Fast path for non-LZP data: if CM beats the raw stream order-1 entropy bound
  if (!cm.empty() && cm.size() <= est_raw) {
    ByteWriter w(payload);
    w.varint(primary);
    w.byte(static_cast<u8>(num_chains));
    if (num_chains > 1)
      for (int c = 0; c < num_chains; ++c) w.varint(anchors[c]);
    payload.insert(payload.end(), cm.begin(), cm.end());
    if (used_fast) *used_fast = false;
    return;
  }

  mtf_forward(tbuf.data(), n);

  // Mode candidates share the expensive entropy encode, so rank them first
  // with cheap estimates and fully encode only the best plus any within 2%.
  // The final choice among the encoded candidates still compares real sizes.
  thread_local std::vector<u8> sym, runlen, run_extra, brle;
  sym.clear();
  runlen.clear();
  run_extra.clear();
  brle.clear();
  bwt_rle0_split(tbuf.data(), n, sym, runlen, run_extra);
  bwt_mtf_bzip2_encode(tbuf.data(), n, brle);

  const double est_rle0 = bwt_stream_est(sym.data(), sym.size(), o1_est) +
                          bwt_stream_est(runlen.data(), runlen.size(), o1_est) +
                          static_cast<double>(run_extra.size());
  const double est_bzip2 = bwt_stream_est(brle.data(), brle.size(), o1_est);
  const double est_plain =
      try_plain ? bwt_stream_est(tbuf.data(), n, o1_est) : 1e30;
  const double est_best =
      min_of(est_rle0, min_of(est_bzip2, min_of(est_plain, est_raw)));
  const double cut = est_best * 1.02;
  // L5 on official 128 KiB blocks: runner-up modes encode whole-stream
  // only. The estimated winner still goes through pick_segmentation.
  // That recovers binary when the estimate ranks the wrong mode, without
  // a second segmentation encode on every candidate. Larger blocks keep
  // the full per-mode segmentation race.
  const bool l5_cheap = level <= 5 && n <= kBwtSmallBlock;

  auto encode_syms = [&](std::vector<u8>& dst, const u8* p, usize pn,
                         bool full) -> bool {
    if (full) return encode_bwt_symbols_best(dst, p, pn, level);
    return encode_bwt_symbols_seg(dst, p, pn, level, 0);
  };

  std::vector<u8> rle0;
  bool rle0_fast = false;
  if (est_rle0 <= cut) {
    const bool full = !l5_cheap || est_rle0 <= est_best;
    rle0.push_back(1);  // mode 1
    rle0_fast |= encode_syms(rle0, sym.data(), sym.size(), full);
    rle0_fast |= encode_syms(rle0, runlen.data(), runlen.size(), full);
    ByteWriter rw(rle0);
    rw.varint(run_extra.size());
    rw.bytes(run_extra);
  }

  std::vector<u8> bzip2;
  bool bzip2_fast = false;
  if (est_bzip2 <= cut) {
    const bool full = !l5_cheap || est_bzip2 <= est_best;
    bzip2.push_back(2);  // mode 2
    bzip2_fast = encode_syms(bzip2, brle.data(), brle.size(), full);
  }

  std::vector<u8> plain;
  bool plain_fast = false;
  if (try_plain && est_plain <= cut) {
    const bool full = !l5_cheap || est_plain <= est_best;
    plain.push_back(0);  // mode 0
    plain_fast = encode_syms(plain, tbuf.data(), n, full);
  }

  std::vector<u8> raw;
  bool raw_fast = false;
  if (try_raw && est_raw <= cut) {
    const bool full = !l5_cheap || est_raw <= est_best;
    raw.push_back(3);  // mode 3: L-column, no MTF
    raw_fast = encode_syms(raw, lcol.data(), n, full);
  }

  ByteWriter w(payload);
  w.varint(primary);
  w.byte(static_cast<u8>(num_chains));
  if (num_chains > 1)
    for (int c = 0; c < num_chains; ++c) w.varint(anchors[c]);
  const std::vector<u8>* chosen = nullptr;
  bool chosen_fast = false;
  auto consider = [&](const std::vector<u8>& cand, bool fast) {
    if (cand.empty()) return;
    if (!chosen || cand.size() < chosen->size()) {
      chosen = &cand;
      chosen_fast = fast;
    }
  };
  consider(rle0, rle0_fast);
  consider(bzip2, bzip2_fast);
  consider(plain, plain_fast);
  consider(raw, raw_fast);
  consider(cm, false);

  if (has_lzp) {
    const u8 mode_byte = (*chosen)[0] | 0x80;
    w.byte(mode_byte);
    w.varint(n);  // intermediate uncompressed BWT length
    w.bytes(chosen->data() + 1, chosen->size() - 1);
  } else {
    payload.insert(payload.end(), chosen->begin(), chosen->end());
  }
  if (used_fast) *used_fast = chosen_fast;
}

void build_bwt_block(const u8* src, usize n, int level,
                     std::vector<u8>& payload, bool* used_fast = nullptr) {
  // If LZP pre-pass is promising (blocks >= 1 KiB, level >= 5):
  thread_local std::vector<u8> lzp_buf;
  usize lzp_len = 0;
  bool lzp_ok = false;
  if (n >= 1024 && level >= 5) {
    lzp_buf.resize(n);
    if (lzp_compress(src, n, lzp_buf.data(), lzp_buf.size(), lzp_len)) {
      if (lzp_len * 200 <= n * 199) {  // >= 0.5% reduction
        lzp_ok = true;
      }
    }
  }

  if (!lzp_ok) {
    build_bwt_single(src, n, level, payload, used_fast, false);
    return;
  }

  // Build LZP-transformed BWT payload
  std::vector<u8> lzp_payload;
  bool lzp_fast = false;
  build_bwt_single(lzp_buf.data(), lzp_len, level, lzp_payload, &lzp_fast, true);

  // At level >= 7, also try raw BWT without LZP if LZP gain was marginal (< 3%)
  if (level >= 7 && lzp_len * 100 > n * 97) {
    std::vector<u8> raw_payload;
    bool raw_fast = false;
    build_bwt_single(src, n, level, raw_payload, &raw_fast, false);
    if (raw_payload.size() <= lzp_payload.size()) {
      payload.swap(raw_payload);
      if (used_fast) *used_fast = raw_fast;
      return;
    }
  }

  payload.swap(lzp_payload);
  if (used_fast) *used_fast = lzp_fast;
}

}  // namespace

void build_compressed_block(const u8* src, usize n, int level,
                            std::vector<u8>& payload,
                            bool context_oracle) {
  if (level < kNearOptLevel) {
    ParseResult pr = parse_block(src, n, level);
    serialize_parse_result(src, n, level, pr, payload);
    return;
  }
  // Near-optimal (10) and ultra (11): serialize the optimal parse, but fall
  // back to the greedy parse for this block if it happens to code smaller. The
  // optimal parser minimizes an estimated bit-cost, so on a few inputs the true
  // entropy-coded size can edge out the greedy parse; this guarantees the
  // optimal levels never lose to the baseline. The level-9 parse is computed
  // once and shared: it seeds the optimal parser's price tables and serves as
  // the fallback candidate.
  ParseResult greedy = parse_block(src, n, 9);
  ParseResult opt_std = parse_block_optimal(src, n, level, greedy, false);
  serialize_parse_result(src, n, level, opt_std, payload, context_oracle);

  if (context_oracle) {
    ParseResult opt_ctx = parse_block_optimal(src, n, level, greedy, true);
    std::vector<u8> ctx_payload;
    serialize_parse_result(src, n, level, opt_ctx, ctx_payload, context_oracle);
    if (ctx_payload.size() < payload.size()) {
      payload.swap(ctx_payload);
    }
  }

  std::vector<u8> greedy_payload;
  serialize_parse_result(src, n, level, greedy, greedy_payload, context_oracle);
  if (greedy_payload.size() < payload.size()) payload.swap(greedy_payload);
}

bool gpu_match_available() {
#ifdef SPECTRA8_HAVE_GPU
  return gpu_backend_available();
#else
  return false;
#endif
}

namespace {

// Same as build_compressed_block, but candidate matches come from the GPU.
void build_compressed_block_from_hints(const u8* src, usize n, int level,
                                       const u16* hint_len, const u32* hint_off,
                                       std::vector<u8>& payload) {
  ParseResult pr = parse_block_from_hints(src, n, hint_len, hint_off);
  serialize_parse_result(src, n, level, pr, payload);
}

#ifdef SPECTRA8_HAVE_GPU
// Ultra + GPU: run the optimal parse over GPU-found top-K candidates, with the
// same greedy safety net as the CPU ultra path so it never loses to greedy.
void build_compressed_block_gpu_topk(const u8* src, usize n, int level,
                                     const u16* hk_len, const u32* hk_off, int k,
                                     std::vector<u8>& payload,
                                     bool context_oracle = false) {
  ParseResult greedy = parse_block(src, n, 9);
  ParseResult opt_std = parse_block_optimal_gpu(src, n, level, hk_len, hk_off, k, greedy, false);
  serialize_parse_result(src, n, level, opt_std, payload, context_oracle);

  if (context_oracle) {
    ParseResult opt_ctx = parse_block_optimal_gpu(src, n, level, hk_len, hk_off, k, greedy, true);
    std::vector<u8> ctx_payload;
    serialize_parse_result(src, n, level, opt_ctx, ctx_payload, context_oracle);
    if (ctx_payload.size() < payload.size()) {
      payload.swap(ctx_payload);
    }
  }

  std::vector<u8> greedy_payload;
  serialize_parse_result(src, n, level, greedy, greedy_payload, context_oracle);
  if (greedy_payload.size() < payload.size()) payload.swap(greedy_payload);
}
#endif

// Serialize a complete block (type byte + regen size + optional per-block
// checksum + filter byte + payload) into `out`. When `hint_len`/`hint_off` are
// non-null they supply GPU-found per-position matches for this block; `topk`
// > 0 means they hold `topk` candidates per position (for the optimal parser)
// instead of one.
// Filtered intermediate: optional byte-RLE on top of the forward transform.
// When RLE wins, LZ/BWT compress the smaller RLE stream; decode expands back to
// the filtered block (block bytes) before the inverse transform.
struct FilteredStage {
  Filter filter = Filter::None;
  bool use_rle = false;
  usize intermediate_len = 0;
  std::vector<u8> work;
};

// Byte-RLE is only worth a trial when it removes a material share of the
// block. A stage marked use_rle costs a second full LZ/BWT compression, and
// the LZ parser already codes runs well: on the test corpus almost every
// plain block RLE-shrank by a fraction of a percent (a few short runs), the
// trial ran on all of them and won on ~2% - by tens of bytes - while doubling
// encode time. The blocks where RLE actually wins (run-heavy images, filtered
// shuffle/delta planes) shrink by 20-30% or more.
constexpr usize kRleTrialMaxNum = 9;  // trial only if rle <= block * 9/10
constexpr usize kRleTrialDen = 10;

// Optional forward filter, then optional byte-RLE when the RLE stream is
// materially smaller (see kRleTrialMaxNum).
void prepare_byte_stage(const u8* src, usize block, Filter filter, bool try_rle,
                        FilteredStage& st) {
  const u8* base = src;
  thread_local std::vector<u8> filtered;
  if (filter != Filter::None) {
    filtered.resize(block);
    apply_filter(filter, src, block, filtered.data());
    base = filtered.data();
  }

  st.filter = filter;
  if (try_rle) {
    std::vector<u8> rle;
    byte_rle_encode(base, block, rle);
    if (rle.size() <= (block / kRleTrialDen) * kRleTrialMaxNum) {
      st.use_rle = true;
      st.work.swap(rle);
      st.intermediate_len = st.work.size();
      return;
    }
  }

  st.use_rle = false;
  if (filter != Filter::None)
    st.work.assign(filtered.data(), filtered.data() + block);
  else
    st.work.assign(src, src + block);
  st.intermediate_len = block;
}

// Detect both filter families for a block: the size-preserving transforms
// (detect_filter) and one self-framed candidate (dict, or else for/alp).
// True when either nominated a candidate.
bool detect_block_filters(const u8* src, usize n, Filter& filter, Filter& framed,
                          bool allow_dict, bool full_sample, u64 mask) {
  Filter forced = Filter::None;
  if (singleton_forced_filter(mask, forced)) {
    if (filter_is_framed(forced)) {
      if (!allow_dict && filter_is_dict(forced)) {
        filter = Filter::None;
        framed = Filter::None;
        return false;
      }
      filter = Filter::None;
      framed = forced;
      return true;
    }
    filter = forced;
    framed = Filter::None;
    return true;
  }
  filter = detect_filter(src, n, full_sample, mask);
  framed = detect_framed_filter(src, n, full_sample, mask, allow_dict);
  return filter != Filter::None || framed != Filter::None;
}

// When `filters` is set, a byte transform is auto-detected for compressed
// blocks (and recorded as a 1-byte filter id before the payload).
void serialize_block(const u8* bsrc, usize block, bool last, int level,
                     bool block_csum, const u16* hint_len, const u32* hint_off,
                     [[maybe_unused]] int topk, bool filters, bool filter_fallback,
                     bool try_filter_rle, bool allow_filter_dict,
                     bool filter_sample_full, u64 filter_mask, bool bwt,
                     bool mosaic, bool oracle, bool crystal, bool flash,
                     bool laser, bool grating, bool context_oracle, u32 crystal_profile,
                     std::vector<u8>& out, u8* wrote_transform = nullptr) {
  bool all_eq = block > 0;
  for (usize i = 1; i < block; ++i)
    if (bsrc[i] != bsrc[0]) {
      all_eq = false;
      break;
    }

  BlockType type;
  Filter filter = Filter::None;
  Filter framed = Filter::None;
  bool block_filter_rle = false;
  usize filter_intermediate = 0;
  std::vector<u8> payload;
  if (all_eq) {
    type = BlockType::Memset;
  } else if (!mosaic && !oracle && !crystal && !flash && !laser && !grating &&
             block_is_incompressible(bsrc, block)) {
    // Fast path: skip the LZ + entropy pipeline entirely and store verbatim.
    type = BlockType::Stored;
  } else if (flash) {
    if (filters && detect_block_filters(bsrc, block, filter, framed,
                                        allow_filter_dict, filter_sample_full,
                                        filter_mask)) {
      if (filter != Filter::None && framed != Filter::None) {
        FilteredStage stage;
        prepare_byte_stage(bsrc, block, filter, try_filter_rle, stage);
        build_flash_block(stage.work.data(), stage.intermediate_len, payload, level);
        block_filter_rle = stage.use_rle;
        filter_intermediate = stage.use_rle ? stage.intermediate_len : 0;

        thread_local std::vector<u8> framed_stream;
        if (framed_filter_fwd(framed, bsrc, block, framed_stream)) {
          std::vector<u8> framed_payload;
          build_flash_block(framed_stream.data(), framed_stream.size(), framed_payload,
                            level);
          if (payload.empty() || (!framed_payload.empty() && framed_payload.size() < payload.size())) {
            payload.swap(framed_payload);
            filter = framed;
            block_filter_rle = false;
            filter_intermediate = framed_stream.size();
          }
        }
      } else if (filter != Filter::None) {
        FilteredStage stage;
        prepare_byte_stage(bsrc, block, filter, try_filter_rle, stage);
        build_flash_block(stage.work.data(), stage.intermediate_len, payload, level);
        block_filter_rle = stage.use_rle;
        filter_intermediate = stage.use_rle ? stage.intermediate_len : 0;
      } else if (framed != Filter::None) {
        thread_local std::vector<u8> framed_stream;
        if (framed_filter_fwd(framed, bsrc, block, framed_stream)) {
          build_flash_block(framed_stream.data(), framed_stream.size(), payload, level);
          filter = framed;
          filter_intermediate = framed_stream.size();
        }
      }
      if (filter_fallback) {
        std::vector<u8> raw_payload;
        build_flash_block(bsrc, block, raw_payload, level);
        if (!raw_payload.empty() && (payload.empty() || raw_payload.size() < payload.size())) {
          payload.swap(raw_payload);
          filter = Filter::None;
          block_filter_rle = false;
          filter_intermediate = 0;
        }
      }
    } else {
      build_flash_block(bsrc, block, payload, level);
    }
    type = (!payload.empty() && payload.size() < block) ? BlockType::Flash : BlockType::Stored;
    if (type == BlockType::Stored) {
      filter = Filter::None;
      block_filter_rle = false;
      filter_intermediate = 0;
    }
  } else if (laser) {
    if (filters && detect_block_filters(bsrc, block, filter, framed,
                                        allow_filter_dict, filter_sample_full,
                                        filter_mask)) {
      if (filter != Filter::None) {
        FilteredStage stage;
        prepare_byte_stage(bsrc, block, filter, try_filter_rle, stage);
        build_laser_block(stage.work.data(), stage.intermediate_len, level, payload,
                          /*already_filtered=*/true);
        block_filter_rle = stage.use_rle;
        filter_intermediate = stage.use_rle ? stage.intermediate_len : 0;
      } else if (framed != Filter::None) {
        thread_local std::vector<u8> framed_stream;
        if (framed_filter_fwd(framed, bsrc, block, framed_stream)) {
          build_laser_block(framed_stream.data(), framed_stream.size(), level, payload,
                            /*already_filtered=*/true);
          filter = framed;
          filter_intermediate = framed_stream.size();
        }
      }
      if (filter_fallback) {
        std::vector<u8> raw_payload;
        build_laser_block(bsrc, block, level, raw_payload, /*already_filtered=*/false);
        if (!raw_payload.empty() && (payload.empty() || raw_payload.size() < payload.size())) {
          payload.swap(raw_payload);
          filter = Filter::None;
          block_filter_rle = false;
          filter_intermediate = 0;
        }
      }
    } else {
      build_laser_block(bsrc, block, level, payload, /*already_filtered=*/false);
    }
    type = (!payload.empty() && payload.size() < block) ? BlockType::Laser : BlockType::Stored;
    if (type == BlockType::Stored) {
      payload.clear();
      filter = Filter::None;
      block_filter_rle = false;
      filter_intermediate = 0;
    }
  } else if (crystal_profile != 0) {
    build_crystal_block(bsrc, block, level, payload, crystal_profile);
    type = (!payload.empty() && payload.size() < block) ? BlockType::Crystal : BlockType::Stored;
    if (type == BlockType::Stored) {
      payload.clear();
    }
    filter = Filter::None;
    block_filter_rle = false;
    filter_intermediate = 0;
  } else if (mosaic || oracle || crystal || grating) {
    auto try_alt_at = [&](const u8* p, usize len, int lv, std::vector<u8>& dst) {
      dst.clear();
      if (mosaic)
        build_mosaic_block(p, len, lv, dst);
      else if (oracle)
        build_oracle_block(p, len, lv, dst);
      else if (grating)
        build_grating_block(p, len, dst);
      else
        build_crystal_block(p, len, lv, dst);
    };
    auto try_alt = [&](const u8* p, usize len, std::vector<u8>& dst) {
      try_alt_at(p, len, level, dst);
    };
    const BlockType alt_type = mosaic ? BlockType::Mosaic
                                      : oracle ? BlockType::Oracle
                                      : grating ? BlockType::Grating
                                                : BlockType::Crystal;
    // These coders model record strides and lanes themselves, so an external
    // shuffle/delta rarely helps (on NanoAOD pages the raw encode won 92-96%
    // of the time) while the filtered trials double or triple encode time.
    // Order of work when filters are on:
    //   1. raw encode (needed anyway as the keep-smaller fallback);
    //   2. a raw payload under 2% of the block ends it - nothing left to win;
    //   3. otherwise a 16 KiB mid-block window is encoded raw and filtered at
    //      level 1 (the comparison is relative; the search cost is not
    //      linear in the window) and the full filtered trial runs only when
    //      the window says the filter pays (> 2%). Below 64 KiB the window
    //      would be most of the block: full trial.
    // With filter_fallback off the raw encode is skipped and step 3 alone
    // decides (filtered-only when the screen passes, raw otherwise).
    constexpr usize kScreenMin = 64 * 1024, kScreenWin = 16 * 1024;
    const int screen_level = min_of(level, 1);
    std::vector<u8> screen_raw;  // raw window encode, shared by both screens
    auto filter_screen = [&](Filter f, Filter framed_f) -> bool {
      if (block < kScreenMin) return true;
      const usize off = ((block - kScreenWin) / 2) & ~static_cast<usize>(63);
      const u8* w = bsrc + off;
      std::vector<u8> alt;
      if (screen_raw.empty()) try_alt_at(w, kScreenWin, screen_level, screen_raw);
      if (f != Filter::None) {
        FilteredStage stage;
        prepare_byte_stage(w, kScreenWin, f, try_filter_rle, stage);
        try_alt_at(stage.work.data(), stage.intermediate_len, screen_level, alt);
      } else {
        thread_local std::vector<u8> ds;
        if (!framed_filter_fwd(framed_f, w, kScreenWin, ds)) return false;
        try_alt_at(ds.data(), ds.size(), screen_level, alt);
      }
      return alt.size() * 100 < screen_raw.size() * 98;
    };
    bool have_raw = false;
    Filter forced = Filter::None;
    const bool force_filter = singleton_forced_filter(filter_mask, forced);
    if (filters && detect_block_filters(bsrc, block, filter, framed,
                                        allow_filter_dict, filter_sample_full,
                                        filter_mask)) {
      if (filter_fallback) {
        try_alt(bsrc, block, payload);
        have_raw = true;
        if (payload.size() * 50 < block) {
          filter = Filter::None;
          framed = Filter::None;
        }
      }
      // A singleton mask is a caller-forced transform (brute-force measurement).
      // Do not entropy-screen it back to raw or every forced row matches none.
      if (!force_filter) {
        if (filter != Filter::None && !filter_screen(filter, Filter::None))
          filter = Filter::None;
        if (framed != Filter::None && !filter_screen(Filter::None, framed))
          framed = Filter::None;
      }
    }
    if (filters && (filter != Filter::None || framed != Filter::None)) {
      std::vector<u8> best;
      Filter best_f = Filter::None;
      bool best_rle = false;
      usize best_inter = 0;
      if (filter != Filter::None) {
        FilteredStage stage;
        prepare_byte_stage(bsrc, block, filter, try_filter_rle, stage);
        try_alt(stage.work.data(), stage.intermediate_len, best);
        best_f = filter;
        best_rle = stage.use_rle;
        best_inter = stage.use_rle ? stage.intermediate_len : 0;
      }
      if (framed != Filter::None) {
        thread_local std::vector<u8> framed_stream;
        if (framed_filter_fwd(framed, bsrc, block, framed_stream)) {
          std::vector<u8> dp;
          try_alt(framed_stream.data(), framed_stream.size(), dp);
          if (best.empty() || dp.size() < best.size()) {
            best.swap(dp);
            best_f = framed;
            best_rle = false;
            best_inter = framed_stream.size();
          }
        }
      }
      if (best.empty()) {
        if (!have_raw) try_alt(bsrc, block, payload);
        filter = Filter::None;
      } else if (have_raw && payload.size() <= best.size()) {
        filter = Filter::None;  // raw keeps: same or smaller
      } else {
        payload.swap(best);
        filter = best_f;
        block_filter_rle = best_rle;
        filter_intermediate = best_inter;
      }
    } else {
      if (!have_raw) try_alt(bsrc, block, payload);
      filter = Filter::None;
    }
    type = (!payload.empty() && payload.size() < block) ? alt_type : BlockType::Stored;
    if (type == BlockType::Stored) {
      filter = Filter::None;
      block_filter_rle = false;
      filter_intermediate = 0;
    }
  } else if (hint_len != nullptr) {
#ifdef SPECTRA8_HAVE_GPU
    if (topk > 0)
      build_compressed_block_gpu_topk(bsrc, block, level, hint_len, hint_off,
                                      topk, payload, context_oracle);
    else
#endif
      build_compressed_block_from_hints(bsrc, block, level, hint_len, hint_off,
                                        payload);
    type = (payload.size() < block) ? BlockType::Compressed : BlockType::Stored;
  } else if (filters && detect_block_filters(bsrc, block, filter, framed,
                                             allow_filter_dict,
                                             filter_sample_full, filter_mask)) {
    // Two filtered candidates compete here:
    //   A: size-preserving transform + optional byte-RLE on the filtered
    //      bytes (shuffle output is often long zero/identical lanes);
    //   B: dictionary remap (self-framed variable-length stream, see
    //      dict_filter.hpp), which may bail on high-cardinality blocks.
    // Try LZ and (when requested) BWT on each intermediate, keep the overall
    // smaller, then compare against the raw-input fallback as before.
    const bool try_bwt = bwt && block >= kBwtMinBlock;
    std::vector<u8> filt_best;
    BlockType filt_type = BlockType::Compressed;
    Filter filt_filter = Filter::None;
    bool filt_rle = false;
    usize filt_inter = 0;

    // `bwt_margin`: how much smaller the BWT candidate must be to displace the
    // LZ one. The transform path keeps the plain size comparison it always
    // had; the dictionary stream requires a 3% win, because its LZ decode is a
    // plain table gather (multi-GB/s) while a BWT inverse of the small id
    // stream is a serial pointer chase - a hair of ratio is not worth a
    // decode-speed cliff.
    auto compress_intermediate = [&](const u8* p, usize len,
                                     std::vector<u8>& out,
                                     double bwt_margin) -> BlockType {
      if (try_bwt && len >= kBwtMinBlock) {
        build_bwt_block(p, len, level, out);
        if (block_looks_run_heavy(p, len)) {
          std::vector<u8> lz;
          build_compressed_block(p, len, level, lz, context_oracle);
          if (lz.size() <= static_cast<double>(out.size()) * bwt_margin) {
            out.swap(lz);
            return BlockType::Compressed;
          }
        }
        return BlockType::Bwt;
      }
      build_compressed_block(p, len, level, out, context_oracle);
      return BlockType::Compressed;
    };

    if (filter != Filter::None) {
      FilteredStage stage;
      prepare_byte_stage(bsrc, block, filter, try_filter_rle, stage);
      filt_type = compress_intermediate(stage.work.data(),
                                        stage.intermediate_len, filt_best, 1.0);
      filt_filter = filter;
      filt_rle = stage.use_rle;
      filt_inter = stage.use_rle ? stage.intermediate_len : 0;
    }

    if (framed != Filter::None) {
      thread_local std::vector<u8> framed_stream;
      if (framed_filter_fwd(framed, bsrc, block, framed_stream)) {
        std::vector<u8> framed_payload;
        const double margin = filter_is_dict(framed) ? 1.03 : 1.0;
        const BlockType framed_type = compress_intermediate(
            framed_stream.data(), framed_stream.size(), framed_payload, margin);
        if (filt_best.empty() || framed_payload.size() < filt_best.size()) {
          filt_best.swap(framed_payload);
          filt_type = framed_type;
          filt_filter = framed;
          filt_rle = false;
          filt_inter = framed_stream.size();
        }
      }
    }

    if (filt_best.empty()) {
      // Only the dictionary was nominated and it bailed: plain path.
      build_compressed_block(bsrc, block, level, payload, context_oracle);
      filter = Filter::None;
      type = payload.size() < block ? BlockType::Compressed : BlockType::Stored;
    } else {
      // Keep-smaller fallback. For LZ-only requests, compare against raw LZ.
      // For BWT requests, also try raw BWT (+ optional byte-RLE): a mediocre
      // filter can beat LZ yet lose to plain BWT (common on text/source), and
      // historically we only compared against LZ — so `--bwt --filter` could
      // regress vs `--bwt` alone. Decode cost is unchanged: we emit whichever
      // payload actually wins.
      payload.swap(filt_best);
      filter = filt_filter;
      block_filter_rle = filt_rle;
      filter_intermediate = filt_inter;
      type = payload.size() < block ? filt_type : BlockType::Stored;

      if (filter_fallback) {
        if (!try_bwt) {
          std::vector<u8> plain_lz;
          build_compressed_block(bsrc, block, level, plain_lz, context_oracle);
          if (plain_lz.size() <= payload.size()) {
            payload.swap(plain_lz);
            filter = Filter::None;
            block_filter_rle = false;
            filter_intermediate = 0;
            type = payload.size() < block ? BlockType::Compressed
                                          : BlockType::Stored;
          }
        } else {
          std::vector<u8> plain_bwt;
          build_bwt_block(bsrc, block, level, plain_bwt);
          bool plain_bwt_rle = false;
          usize plain_bwt_inter = 0;
          if (try_filter_rle) {
            FilteredStage stage;
            prepare_byte_stage(bsrc, block, Filter::None, true, stage);
            if (stage.use_rle) {
              std::vector<u8> rle_payload;
              build_bwt_block(stage.work.data(), stage.intermediate_len, level,
                              rle_payload);
              if (rle_payload.size() < plain_bwt.size()) {
                plain_bwt.swap(rle_payload);
                plain_bwt_rle = true;
                plain_bwt_inter = stage.intermediate_len;
              }
            }
          }
          if (plain_bwt.size() < payload.size()) {
            payload.swap(plain_bwt);
            filter = Filter::None;
            block_filter_rle = plain_bwt_rle;
            filter_intermediate = plain_bwt_inter;
            type = payload.size() < block ? BlockType::Bwt : BlockType::Stored;
          }
        }
      }

      if (type == BlockType::Stored) {
        filter = Filter::None;
        block_filter_rle = false;
        filter_intermediate = 0;
      }
    }
  } else if (bwt) {
    if (block < kBwtMinBlock) {
      type = BlockType::Stored;
    } else {
      // Strategy A (default): BWT → MTF → RLE/entropy. Wins on raw text and
      // unfiltered data where locality exists without long exact matches.
      bool bwt_fast = false;
      build_bwt_block(bsrc, block, level, payload, &bwt_fast);
      FilteredStage stage;
      if (try_filter_rle) {
        prepare_byte_stage(bsrc, block, Filter::None, true, stage);
        if (stage.use_rle) {
          std::vector<u8> rle_payload;
          bool rle_fast = false;
          build_bwt_block(stage.work.data(), stage.intermediate_len, level,
                          rle_payload, &rle_fast);
          if (rle_payload.size() < payload.size()) {
            payload.swap(rle_payload);
            bwt_fast = rle_fast;
            block_filter_rle = true;
            filter_intermediate = stage.intermediate_len;
          }
        }
      }
      type = (payload.size() < block) ? BlockType::Bwt : BlockType::Stored;

      // Strategy B: when the block already looks prefiltered (run-heavy
      // shuffle/delta/dict planes — as in verdict external +flt staging),
      // also try the LZ path that milks long exact matches. Keep the smaller
      // payload; ties prefer LZ (much faster decode than inverse BWT). Emits
      // BlockType::Compressed inside a Codec::Prism request when LZ wins — the
      // frame stays ordinary Spectra8; only the per-block type changes.
      if (type == BlockType::Bwt && block_looks_run_heavy(bsrc, block)) {
        std::vector<u8> lz;
        build_compressed_block(bsrc, block, level, lz, context_oracle);
        bool lz_rle = false;
        usize lz_inter = 0;
        if (try_filter_rle) {
          FilteredStage lz_stage;
          prepare_byte_stage(bsrc, block, Filter::None, true, lz_stage);
          if (lz_stage.use_rle) {
            std::vector<u8> rle_payload;
            build_compressed_block(lz_stage.work.data(),
                                   lz_stage.intermediate_len, level, rle_payload, context_oracle);
            if (rle_payload.size() <= lz.size()) {
              lz.swap(rle_payload);
              lz_rle = true;
              lz_inter = lz_stage.intermediate_len;
            }
          }
        }
        // A last-step LZ/zstd can make BWT a hair smaller than whole-block
        // LZ while still requiring inverse BWT. Demand a 3% win in that
        // case (same margin as dict-stream BWT). Entropy last-step keeps
        // the old keep-smaller rule.
        const bool lz_wins =
            bwt_fast
                ? (static_cast<double>(payload.size()) * 1.03 >=
                   static_cast<double>(lz.size()))
                : (lz.size() <= payload.size());
        if (lz_wins) {
          payload.swap(lz);
          block_filter_rle = lz_rle;
          filter_intermediate = lz_inter;
          type = payload.size() < block ? BlockType::Compressed
                                        : BlockType::Stored;
        }
      }

      if (type == BlockType::Stored) block_filter_rle = false;
    }
  } else {
    build_compressed_block(bsrc, block, level, payload, context_oracle);
    if (try_filter_rle) {
      FilteredStage stage;
      prepare_byte_stage(bsrc, block, Filter::None, true, stage);
      if (stage.use_rle) {
        std::vector<u8> rle_payload;
        build_compressed_block(stage.work.data(), stage.intermediate_len, level,
                               rle_payload, context_oracle);
        if (rle_payload.size() < payload.size()) {
          payload.swap(rle_payload);
          block_filter_rle = true;
          filter_intermediate = stage.intermediate_len;
        }
      }
    }
    type = (payload.size() < block) ? BlockType::Compressed : BlockType::Stored;
    if (type == BlockType::Stored) block_filter_rle = false;
  }

  ByteWriter w(out);
  w.byte(static_cast<u8>(type) | (last ? kBlockLastFlag : 0));
  w.varint(block);
  if (block_csum) {
    u8 h[4];
    store_u32(h, static_cast<u32>(content_hash(bsrc, block)));
    w.bytes(h, 4);
  }
  switch (type) {
    case BlockType::Memset:
      w.byte(bsrc[0]);
      break;
    case BlockType::Compressed:
    case BlockType::Bwt:
    case BlockType::Mosaic:
    case BlockType::Oracle:
    case BlockType::Crystal:
    case BlockType::Flash:
    case BlockType::Laser:
    case BlockType::Grating:
      // Only blocks that actually used a transform (or byte-RLE) write the
      // filter byte here. If any block in the frame did, frame_compress
      // splices a Filter::None byte into the others, so a filters-enabled
      // frame where nothing transformed stays byte-identical to filters-off.
      if (filter != Filter::None || block_filter_rle) {
        w.byte(pack_filter_byte(filter, block_filter_rle));
        // Byte-RLE and the dictionary filter both produce an intermediate
        // stream whose length differs from the block size; the decoder needs
        // it to size the LZ/BWT decode buffer.
        if (block_filter_rle || filter_is_framed(filter)) w.varint(filter_intermediate);
      }
      w.varint(payload.size());
      w.bytes(payload);
      break;
    case BlockType::Stored:
      w.bytes(bsrc, block);
      break;
  }
  if (wrote_transform) {
    *wrote_transform =
        (type == BlockType::Compressed || type == BlockType::Bwt ||
         type == BlockType::Mosaic || type == BlockType::Oracle ||
         type == BlockType::Crystal || type == BlockType::Flash ||
         type == BlockType::Laser || type == BlockType::Grating) &&
        (filter != Filter::None || block_filter_rle)
            ? 1
            : 0;
  }
}

[[maybe_unused]] inline void encode_one_block(const u8* bsrc, usize block, bool last, int level,
                                          const EncodeParams& p, std::vector<u8>& out,
                                          u8* wrote_transform = nullptr) {
  serialize_block(bsrc, block, last, level, p.block_checksums, nullptr, nullptr, 0,
                  p.filters, p.filter_fallback, p.filter_rle, p.filter_dict,
                  p.filter_sample_full, p.filter_mask, p.bwt, p.mosaic, p.oracle,
                  p.crystal, p.flash, p.laser, p.grating, p.context_oracle,
                  p.crystal_profile, out, wrote_transform);
}

}  // namespace

void frame_compress(const u8* src, usize n, const EncodeParams& p,
                    std::vector<u8>& out) {
  // The GPU match finder only kicks in when requested, available, and the input
  // is large enough to amortize the host<->device transfer.
  // Filters re-arrange block bytes before LZ, so GPU match hints (found on the
  // raw bytes) would not apply; filtering therefore uses the CPU path.
  bool use_gpu = p.gpu && !p.filters && !p.bwt && !p.mosaic && !p.oracle &&
                 !p.crystal && !p.flash && !p.laser && !p.grating &&
                 gpu_match_available() &&
                 n >= (usize{1} << 16);

  int bslog = kBlockSizeLog;
  // Derive the block-size log from the requested size (already clamped).
  {
    int l = kMinBlockSizeLog;
    while ((usize{1} << l) < p.block_size && l < kMaxBlockSizeLog) ++l;
    bslog = l;
  }
  // The GPU kernel sizes its threadgroup hash for a fixed block size.
  if (use_gpu) bslog = kGpuBlockSizeLog;
  const usize block_size = usize{1} << bslog;

  u8 flags = kFlagContentSize;
  if (p.content_checksum) flags |= kFlagChecksum;
  if (p.seek_table) flags |= kFlagSeekTable;
  if (p.block_checksums) flags |= kFlagBlockChecksums;
  ByteWriter w(out);
  w.bytes(kMagic, kMagicLen);
  w.byte(kVersion);
  w.byte(flags);
  w.byte(static_cast<u8>(bslog));
  w.varint(n);

  // Block boundaries (a single empty block for empty input).
  std::vector<std::pair<usize, usize>> blocks;
  if (n == 0) {
    blocks.emplace_back(0, 0);
  } else {
    for (usize off = 0; off < n; off += block_size)
      blocks.emplace_back(off, min_of(block_size, n - off));
  }
  const usize num_blocks = blocks.size();
  std::vector<std::vector<u8>> bufs(num_blocks);
  std::vector<u8> block_transform(num_blocks, 0);
  bool encoded = false;

#ifdef SPECTRA8_HAVE_GPU
  // Pipelined GPU path: the GPU finds matches for segment s+1 while the CPU
  // entropy-codes segment s. Double-buffered hint storage ping-pongs between
  // segments. Any GPU failure degrades that segment to the CPU match finder.
  if (use_gpu && n > 0) {
    // The optimal levels (near-optimal 10 and ultra 11) run the cost-based
    // parser over GPU top-K candidates (kGpuTopK entries per position); greedy
    // levels use one best match per position. `stride` is the per-position
    // entry count for both the GPU buffers and the parser.
    const bool optimal = p.level >= kNearOptLevel;
    const int topk = optimal ? kGpuTopK : 0;
    const usize stride = optimal ? static_cast<usize>(kGpuTopK) : usize{1};

    const usize seg_bytes = kGpuSegmentBlocks * block_size;
    const usize num_segs = (n + seg_bytes - 1) / seg_bytes;
    // Buffers only ever hold one segment, capped at the frame size.
    const usize hint_cap = min_of(seg_bytes, n);

    std::vector<u16> hl_buf[2];
    std::vector<u32> ho_buf[2];
    for (int k = 0; k < 2; ++k) {
      hl_buf[k].resize(hint_cap * stride);
      ho_buf[k].resize(hint_cap * stride);
    }
    auto seg_lo = [&](usize s) { return s * seg_bytes; };
    auto seg_hi = [&](usize s) { return min_of(s * seg_bytes + seg_bytes, n); };
    auto launch = [&](usize a, usize b) {
      return optimal
                 ? gpu_find_topk_begin(src + a, b - a, block_size, kGpuMaxMatch)
                 : gpu_find_begin(src + a, b - a, block_size, kGpuMaxMatch);
    };
    auto collect = [&](GpuJob* j, int buf) {
      return optimal
                 ? gpu_find_topk_end(j, hl_buf[buf].data(), ho_buf[buf].data())
                 : gpu_find_end(j, hl_buf[buf].data(), ho_buf[buf].data());
    };

    GpuJob* job = launch(seg_lo(0), seg_hi(0));

    for (usize s = 0; s < num_segs; ++s) {
      const usize a = seg_lo(s);
      const usize b = seg_hi(s);
      const int buf = static_cast<int>(s & 1);

      bool have = (job != nullptr) && collect(job, buf);
      job = nullptr;

      // Kick off the next segment before encoding this one so the GPU overlaps
      // the CPU entropy coding below.
      if (s + 1 < num_segs) job = launch(seg_lo(s + 1), seg_hi(s + 1));

      const u16* hlb = have ? hl_buf[buf].data() : nullptr;
      const u32* hob = have ? ho_buf[buf].data() : nullptr;
      const usize first = a / block_size;
      const usize count = (b - a + block_size - 1) / block_size;
      parallel_for(count, p.threads, [&](usize j) {
        const usize i = first + j;
        const bool last = (i + 1 == num_blocks);
        const usize boff = blocks[i].first;
        const usize hoff = (boff - a) * stride;
        serialize_block(src + boff, blocks[i].second, last, p.level,
                        p.block_checksums, hlb ? hlb + hoff : nullptr,
                        hob ? hob + hoff : nullptr, topk, /*filters=*/false,
                        /*filter_fallback=*/true, /*try_filter_rle=*/true,
                        /*allow_filter_dict=*/true, /*filter_sample_full=*/false,
                        /*filter_mask=*/~u64{0}, /*bwt=*/false, /*mosaic=*/false, /*oracle=*/false,
                        /*crystal=*/false, /*flash=*/false, /*laser=*/false, /*grating=*/false,
                        p.context_oracle, /*crystal_profile=*/0,
                        bufs[i], &block_transform[i]);
      });
    }
    encoded = true;
  }
#endif

  // CPU path: compress every block independently (optionally in parallel).
  // Output is identical regardless of thread count.
  if (!encoded) {
    parallel_for(num_blocks, p.threads, [&](usize i) {
      bool last = (i + 1 == num_blocks);
      serialize_block(src + blocks[i].first, blocks[i].second, last, p.level,
                      p.block_checksums, nullptr, nullptr, 0, p.filters,
                      p.filter_fallback, p.filter_rle, p.filter_dict,
                      p.filter_sample_full, p.filter_mask, p.bwt, p.mosaic,
                      p.oracle, p.crystal, p.flash, p.laser, p.grating, p.context_oracle,
                      p.crystal_profile, bufs[i], &block_transform[i]);
    });
  }

  // kFlagFilters is frame-global, but blocks are serialized independently: a
  // block only wrote its filter byte when it actually used a transform. If
  // any block did, the decoder will expect the byte on *every* compressed/BWT
  // block of the frame, so splice a Filter::None byte into the blocks that
  // skipped it. When no block transformed, the flag stays clear and the frame
  // is byte-identical to a filters-off encode.
  bool any_transform = false;
  for (u8 bt : block_transform) any_transform |= bt != 0;
  if (any_transform) {
    out[5] |= kFlagFilters;
    for (usize i = 0; i < num_blocks; ++i) {
      if (block_transform[i] || bufs[i].empty()) continue;
      std::vector<u8>& b = bufs[i];
      const BlockType t = static_cast<BlockType>(b[0] & kBlockTypeMask);
      if (t != BlockType::Compressed && t != BlockType::Bwt &&
          t != BlockType::Mosaic && t != BlockType::Oracle && t != BlockType::Crystal &&
          t != BlockType::Flash && t != BlockType::Laser && t != BlockType::Grating)
        continue;
      usize pos = 1;
      while (pos < b.size() && (b[pos] & 0x80)) ++pos;  // regen-size varint
      ++pos;
      if (p.block_checksums) pos += 4;
      b.insert(b.begin() + pos, pack_filter_byte(Filter::None, false));
    }
  }

  std::vector<SeekEntryInternal> seek(num_blocks);
  for (usize i = 0; i < num_blocks; ++i) {
    seek[i] = {static_cast<u64>(bufs[i].size()),
               static_cast<u64>(blocks[i].second)};
    out.insert(out.end(), bufs[i].begin(), bufs[i].end());
  }

  // Optional whole-content checksum (off by default).
  if (p.content_checksum) {
    u8 cs[8];
    store_u64(cs, content_hash(src, n));
    w.bytes(cs, 8);
  }

  // Optional seek-table trailer: fixed 8-byte entries + an 8-byte footer.
  if (p.seek_table) {
    for (const SeekEntryInternal& e : seek) {
      u8 b[8];
      store_u32(b, static_cast<u32>(e.comp_size));
      store_u32(b + 4, static_cast<u32>(e.regen_size));
      w.bytes(b, 8);
    }
    u8 footer[8];
    store_u32(footer, static_cast<u32>(num_blocks));
    footer[4] = kSeekMagic[0];
    footer[5] = kSeekMagic[1];
    footer[6] = kSeekMagic[2];
    footer[7] = kSeekMagic[3];
    w.bytes(footer, 8);
  }
}

}  // namespace spectra8::detail
