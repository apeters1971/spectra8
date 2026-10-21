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

#include "beam/match_finder.hpp"

namespace spectra8::detail {

MfParams params_for_level(int level, usize block_size) {
  // hash_log scales with the block size but is capped per level. Levels >= 6
  // let the hash table grow with the block (up to 2^22 slots): capping a
  // multi-MiB block at 2^18 buckets floods every chain with hash collisions,
  // which is simultaneously much slower (max_chain steps are wasted on
  // collision candidates) and worse at finding real matches. For the default
  // 256 KiB block the caps below are unchanged.
  int size_log = 16;
  while ((usize{1} << size_log) < block_size && size_log < 22) ++size_log;

  MfParams p{};
  switch (level) {
    case 1: p = {min_of(size_log, 14), 4, 16, false}; break;
    case 2: p = {min_of(size_log, 15), 8, 24, false}; break;
    // Levels 3-4 run the double-fast parser (two single-slot hash tables);
    // only hash_log is used, capped so both tables stay cache resident.
    case 3: p = {min_of(size_log, 16), 0, 0, false}; break;
    case 4: p = {min_of(size_log, 18), 0, 0, false}; break;
    // zstd L5: greedy, searchLog 3 (8 attempts). Lazy is L6+ in zstd.
    case 5:
      p = {min_of(size_log, 17), 8, 64, false};
      p.dense_short_insert = false;
      break;
    case 6: p = {min_of(size_log, 20), 128, 96, 1}; break;
    case 7: p = {min_of(size_log, 21), 320, 128, 1}; break;
    case 8: p = {min_of(size_log, 22), 640, 192, 2}; break;
    case 9: p = {min_of(size_log, 22), 1280, 256, 2}; break;
    // Level 10 ("near-optimal"): the optimal parser drives candidate selection
    // (lazy matching off). A moderate chain keeps a single DP pass quick.
    case 10: p = {min_of(size_log, 21), 96, 96, false}; break;
    // Level 11 ("ultra"): deeper chain (more candidate offsets per position),
    // bounded so the two-pass cost-based parse stays tractable.
    default: p = {min_of(size_log, 22), 256, 128, false}; break;
  }
  if (p.hash_log < 12) p.hash_log = 12;
  // Large blocks hash more bytes per chain entry. With a 4-byte hash, popular
  // 4-grams ("the ", runs of spaces/zeros) accumulate thousands of chain
  // entries across a multi-MiB block and the finder spends its whole max_chain
  // budget wading through them; a 5/6-byte key splits those chains by the next
  // byte(s). Short (4-byte) matches at new offsets are mostly lost, but on big
  // blocks they are marginal (their offsets are expensive) and the rep-offset
  // path still finds short repeats. Not applied to levels 1-4 (their parsers
  // have their own single-slot tables) or to the default 256 KiB block.
  if (level >= 5) {
    if (block_size >= (usize{1} << 22))
      p.hash_bytes = 6;
    else if (block_size > (usize{1} << 20))
      p.hash_bytes = 5;
    // zstd L5 on >256 KiB uses minMatch 5, so a 5-byte hash.
    else if (level == 5 && block_size > (usize{1} << 18))
      p.hash_bytes = 5;
  }
  // Blocks above 256 KiB at high levels switch from hash chains to the
  // binary-tree finder: chain quality collapses with window size (every step
  // tests one candidate), while a tree comparison discards a whole subtree.
  // The depth budget is much smaller than the chain budget for the same
  // search quality (zstd's btlazy2/btopt use 2^4..2^10 comparisons).
  // Level 9 and the optimal-parse levels (10/11) use the tree at *every*
  // block size: these are the ratio levels, the tree's candidates are never
  // worse (its sorted history reaches longer matches within budget - e.g.
  // -4% output on 128 KiB int16 blocks), and on the small-block benchmark
  // corpus it is also faster (L9 32 -> 40 MB/s; ultra's find_all gathering
  // was ~75% of encode with chains). At L7-8 small blocks keep chains: they
  // are cache-resident there and BT costs up to 2x on run-heavy binary,
  // which those speed-leaning levels should not pay.
  if ((level >= 7 && block_size > (usize{1} << 18)) || level >= 9) {
    p.use_bt = true;
    p.use_row = false;
    p.max_chain = max_of(64, p.max_chain / 5);
  }
  // Blocks above 256 KiB at levels 3-6 use the row-based tag finder instead
  // of chains: one SIMD compare over a row of recent same-hash positions
  // replaces the chain walk's dependent cache misses, which dominate once the
  // block outgrows L2 (measured ~3x faster at L5/L6 with 4 MiB blocks for
  // ~0.1-1% ratio). At 256 KiB and below chains stay: they are cache-resident
  // and nearly collision-free there (hash_log ~ positions), so a chain search
  // usually verifies a single candidate and the row's fixed per-search work
  // only slows it down. Levels 7+ keep the binary-tree finder, which buys the
  // best ratio at these block sizes for these levels' deeper budgets.
  if (level >= 5 && level <= 6 && block_size > (usize{1} << 18)) {
    p.use_row = true;
    // hash_log is total slots (rows << row_log), sized like the block so
    // capacity matches the chain finder's prev_ array. L5 only verifies
    // 8 hits (searchLog 3); 16-entry rows match zstd L5 and skip scanning
    // 48 unused tags. L3/L4/L6 keep 64-entry rows for history/ratio.
    p.row_log = (level == 5) ? 4 : 6;
    p.hash_log = min_of(size_log, 22);
    if (p.hash_log < 12 + p.row_log) p.hash_log = 12 + p.row_log;
  }
  return p;
}

}  // namespace spectra8::detail
