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

#include "mosaic/mosaic.hpp"

#include <algorithm>
#include <cstring>
#include <vector>

#include "prism/bwt.hpp"
#include "prism/bwt_mtf_rle.hpp"
#include "codec.hpp"
#include "crystal/crystal.hpp"
#include "util/entropy.hpp"
#include "util/tans.hpp"

namespace spectra8::detail {
namespace {

constexpr u8 kMosaicVer = 2;
constexpr u8 kKindStore = 0;
constexpr u8 kKindMemset = 1;
constexpr u8 kKindRolz = 2;
constexpr u8 kKindTbwt = 3;
constexpr u8 kKindCrystal = 4;  // Crystal static lattice coder on the raw tile

// encode_array payload tag (must match entropy.cpp).
constexpr u8 kTagAnsO1 = 7;

constexpr int kCtxBits = 12;
constexpr int kCtxN = 1 << kCtxBits;
constexpr int kSlots = 8;
constexpr int kMinRolz = 2;
constexpr int kMinCopy = 4;
// Stored extras are u8, so a ROLZ/copy length must fit in 255+min.
constexpr int kMaxCopy = 255 + kMinRolz;
constexpr int kFarBits = 15;
constexpr int kFarN = 1 << kFarBits;

inline u32 mosaic_ctx(const u8* s, usize i) noexcept {
  if (i == 0) return 0;
  u32 h = s[i - 1];
  if (i >= 2) h = h * 131u + s[i - 2];
  if (i >= 3) h = h * 31u + s[i - 3];
  return h & (kCtxN - 1);
}

inline u32 far_hash(const u8* s, usize i, usize n) noexcept {
  u32 v = s[i];
  if (i + 1 < n) v |= static_cast<u32>(s[i + 1]) << 8;
  if (i + 2 < n) v |= static_cast<u32>(s[i + 2]) << 16;
  if (i + 3 < n) v |= static_cast<u32>(s[i + 3]) << 24;
  return (v * 0x9E3779B1u) >> (32 - kFarBits);
}

inline usize match_len(const u8* a, const u8* b, usize max) noexcept {
  usize i = 0;
  while (i + 8 <= max) {
    const u64 diff = load_u64(a + i) ^ load_u64(b + i);
    if (diff != 0) return i + (static_cast<usize>(ctz64(diff)) >> 3);
    i += 8;
  }
  while (i < max && a[i] == b[i]) ++i;
  return i;
}

struct RolzTab {
  u32 slot[kCtxN][kSlots]{};
  u8 wr[kCtxN]{};
  u8 cnt[kCtxN]{};
  u32 far[kFarN]{};

  void reset() {
    std::memset(wr, 0, sizeof(wr));
    std::memset(cnt, 0, sizeof(cnt));
    std::memset(far, 0xFF, sizeof(far));
  }

  void push(u32 c, u32 pos) {
    slot[c][wr[c]] = pos;
    wr[c] = static_cast<u8>((wr[c] + 1) & (kSlots - 1));
    if (cnt[c] < kSlots) ++cnt[c];
  }

  void push_far(u32 h, u32 pos) { far[h] = pos; }
};

struct DecTab {
  u32 slot[kCtxN][kSlots]{};
  u8 wr[kCtxN]{};
  u8 cnt[kCtxN]{};

  void reset() {
    std::memset(wr, 0, sizeof(wr));
    std::memset(cnt, 0, sizeof(cnt));
  }

  void push(u32 c, u32 pos) {
    slot[c][wr[c]] = pos;
    wr[c] = static_cast<u8>((wr[c] + 1) & (kSlots - 1));
    if (cnt[c] < kSlots) ++cnt[c];
  }
};

// Encoder search depth handed to the Crystal static coder for a symbol
// stream: the coder's own levels are much slower per level than Mosaic's,
// so map Mosaic L4-6 -> Crystal L2, L7-8 -> L3, L9+ -> L5.
int crystal_level_for(int level) noexcept {
  return level >= 9 ? 5 : level >= 7 ? 3 : 2;
}

void encode_mosaic_symbols(std::vector<u8>& out, const u8* src, usize n,
                           int level) {
  if (n >= 256) {
    thread_local std::vector<u8> body, blob;
    body.clear();
    body.push_back(kTagAnsO1);
    ans::encode_o1(body, src, n);
    blob.clear();
    ByteWriter w(blob);
    w.varint(n);
    w.varint(body.size());
    w.bytes(body);
    // Race the order-1 rANS against the Crystal static coder at stride 1
    // (order-1/2 contexts merged by MDL, per-group static tables). Decode of
    // either is table-driven; the Crystal path is the smaller one on almost
    // every BWT symbol stream.
    if (encode_array_crystal(out, src, n, crystal_level_for(level), 1,
                             blob.size()))
      return;
    out.insert(out.end(), blob.begin(), blob.end());
    return;
  }
  encode_array(out, src, n, 5, ArrayKind::Tokens);
}

inline int stride_from_flags(u8 xf) noexcept {
  if ((xf & 1) == 0) return 0;
  const int code = (xf >> 1) & 3;
  return code == 1 ? 2 : code == 2 ? 4 : code == 3 ? 8 : 0;
}

void undo_xor_stride(u8* d, usize n, int stride) {
  if (stride <= 0) return;
  const usize s = static_cast<usize>(stride);
  for (usize i = s; i < n; ++i) d[i] = static_cast<u8>(d[i] ^ d[i - s]);
}

int detect_stride(const u8* s, usize n) {
  if (n < 64) return 0;
  const usize lim = min_of(n, usize{8192});
  usize eq2 = 0, eq4 = 0, eq8 = 0;
  for (usize i = 8; i < lim; ++i) {
    if (s[i] == s[i - 2]) ++eq2;
    if (s[i] == s[i - 4]) ++eq4;
    if (s[i] == s[i - 8]) ++eq8;
  }
  const usize span = lim - 8;
  usize best = eq2;
  int stride = 2;
  if (eq4 > best) {
    best = eq4;
    stride = 4;
  }
  if (eq8 > best) {
    best = eq8;
    stride = 8;
  }
  // Need a real lattice, not chance equality (~1/256 for noise).
  if (best * 5 < span) return 0;
  return stride;
}

void push_range(RolzTab& tab, const u8* src, usize n, usize i, usize len,
                bool use_far) {
  for (usize k = 0; k < len; ++k) {
    const usize p = i + k;
    tab.push(mosaic_ctx(src, p), static_cast<u32>(p));
    if (use_far) tab.push_far(far_hash(src, p, n), static_cast<u32>(p));
  }
}

void encode_rolz_tile(const u8* src, usize n, int level, std::vector<u8>& out) {
  thread_local RolzTab tab;
  tab.reset();
  std::vector<u8> kinds, lits, slots, rlens, clens, coffs;
  kinds.reserve(n);
  lits.reserve(n / 2);

  const int nsearch = (level <= 2) ? 4 : kSlots;
  const bool use_far = level >= 3;
  const bool use_lat = level >= 3;
  const int lazy = (level >= 9) ? 2 : (level >= 5) ? 1 : 0;
  const int stride = use_lat ? detect_stride(src, n) : 0;
  const int lat_max = (level >= 7) ? 8 : 4;

  u32 rep1 = 1, rep2 = 1;

  auto find_at = [&](usize i, int& best_s, usize& best_l, usize& copy_l,
                     u32& copy_off) {
    best_s = -1;
    best_l = 0;
    copy_l = 0;
    copy_off = 0;
    const u32 c = mosaic_ctx(src, i);
    const usize max = min_of(n - i, static_cast<usize>(kMaxCopy));
    const int cnt = tab.cnt[c];
    const int last = tab.wr[c];
    for (int j = 0; j < cnt && j < nsearch; ++j) {
      const int idx = (last + kSlots - 1 - j) & (kSlots - 1);
      const u32 p = tab.slot[c][idx];
      if (p >= i) continue;
      const usize ml = match_len(src + p, src + i, max);
      if (ml > best_l) {
        best_l = ml;
        best_s = j;
      }
    }

    auto consider_off = [&](u32 off) {
      if (off == 0 || off > i) return;
      const usize ml = match_len(src + (i - off), src + i, max);
      if (ml > copy_l) {
        copy_l = ml;
        copy_off = off;
      }
    };
    consider_off(rep1);
    consider_off(rep2);
    if (use_far && max >= static_cast<usize>(kMinCopy)) {
      const u32 h = far_hash(src, i, n);
      const u32 p = tab.far[h];
      if (p < i && i - p < (1u << 24)) consider_off(static_cast<u32>(i - p));
    }
    if (stride > 0) {
      const u32 s = static_cast<u32>(stride);
      for (int k = 1; k <= lat_max; ++k) {
        const u32 off = s * static_cast<u32>(k);
        if (off > i) break;
        consider_off(off);
      }
    }
  };

  auto commit_rolz = [&](usize i, int s, usize len) {
    kinds.push_back(1);
    slots.push_back(static_cast<u8>(s));
    rlens.push_back(static_cast<u8>(len - kMinRolz));
    const u32 c = mosaic_ctx(src, i);
    const int idx = (tab.wr[c] + kSlots - 1 - s) & (kSlots - 1);
    const u32 p = tab.slot[c][idx];
    const u32 off = static_cast<u32>(i - p);
    rep2 = rep1;
    rep1 = off;
    push_range(tab, src, n, i, len, use_far);
  };

  auto commit_copy = [&](usize i, usize len, u32 off) {
    kinds.push_back(2);
    clens.push_back(static_cast<u8>(len - kMinCopy));
    coffs.push_back(static_cast<u8>(off));
    coffs.push_back(static_cast<u8>(off >> 8));
    coffs.push_back(static_cast<u8>(off >> 16));
    if (off != rep1) {
      rep2 = rep1;
      rep1 = off;
    }
    push_range(tab, src, n, i, len, use_far);
  };

  auto commit_lit = [&](usize i) {
    kinds.push_back(0);
    lits.push_back(src[i]);
    tab.push(mosaic_ctx(src, i), static_cast<u32>(i));
    if (use_far) tab.push_far(far_hash(src, i, n), static_cast<u32>(i));
  };

  usize i = 0;
  while (i < n) {
    int best_s = -1;
    usize best_l = 0, copy_l = 0;
    u32 copy_off = 0;
    find_at(i, best_s, best_l, copy_l, copy_off);

    const bool take_rolz = best_l >= static_cast<usize>(kMinRolz) &&
                           best_l + 1 >= copy_l;
    const bool take_copy = !take_rolz && copy_l >= static_cast<usize>(kMinCopy);

    if ((take_rolz || take_copy) && lazy > 0 && i + 1 < n) {
      int s2 = -1;
      usize l2 = 0, c2 = 0;
      u32 o2 = 0;
      find_at(i + 1, s2, l2, c2, o2);
      const usize cur = take_rolz ? best_l : copy_l;
      const usize nxt = (l2 + 1 >= c2) ? l2 : c2;
      if (nxt > cur) {
        commit_lit(i);
        ++i;
        continue;
      }
    }

    if (take_rolz) {
      commit_rolz(i, best_s, best_l);
      i += best_l;
    } else if (take_copy) {
      commit_copy(i, copy_l, copy_off);
      i += copy_l;
    } else {
      commit_lit(i);
      ++i;
    }
  }

  const int elvl = 5;
  encode_array(out, kinds.data(), kinds.size(), elvl, ArrayKind::Tokens);
  encode_array(out, lits.data(), lits.size(), elvl, ArrayKind::Tokens);
  encode_array(out, slots.data(), slots.size(), elvl, ArrayKind::Tokens);
  encode_array(out, rlens.data(), rlens.size(), elvl, ArrayKind::Tokens);
  encode_array(out, clens.data(), clens.size(), elvl, ArrayKind::Tokens);
  encode_array(out, coffs.data(), coffs.size(), elvl, ArrayKind::Tokens);
}

Status decode_rolz_tile(ByteReader& r, u8* dst, usize n) {
  std::vector<u8> kinds, lits, slots, rlens, clens, coffs;
  if (Status s = decode_array(r, kinds); s != Status::Ok) return s;
  if (Status s = decode_array(r, lits); s != Status::Ok) return s;
  if (Status s = decode_array(r, slots); s != Status::Ok) return s;
  if (Status s = decode_array(r, rlens); s != Status::Ok) return s;
  if (Status s = decode_array(r, clens); s != Status::Ok) return s;
  if (Status s = decode_array(r, coffs); s != Status::Ok) return s;

  thread_local DecTab tab;
  tab.reset();
  usize li = 0, si = 0, ri = 0, ci = 0, oi = 0, o = 0;

  auto copy_bytes = [&](usize p, usize ml) {
    if (o - p >= ml)
      std::memcpy(dst + o, dst + p, ml);
    else
      for (usize t = 0; t < ml; ++t) dst[o + t] = dst[p + t];
    for (usize t = 0; t < ml; ++t)
      tab.push(mosaic_ctx(dst, o + t), static_cast<u32>(o + t));
    o += ml;
  };

  for (u8 k : kinds) {
    if (o > n) return Status::CorruptInput;
    if (k == 0) {
      if (li >= lits.size() || o >= n) return Status::CorruptInput;
      dst[o] = lits[li++];
      tab.push(mosaic_ctx(dst, o), static_cast<u32>(o));
      ++o;
      continue;
    }
    if (k == 1) {
      if (si >= slots.size() || ri >= rlens.size()) return Status::CorruptInput;
      const u32 c = mosaic_ctx(dst, o);
      const int j = slots[si++];
      const usize ml = static_cast<usize>(rlens[ri++]) + kMinRolz;
      if (j >= tab.cnt[c] || o + ml > n) return Status::CorruptInput;
      const int idx = (tab.wr[c] + kSlots - 1 - j) & (kSlots - 1);
      const u32 p = tab.slot[c][idx];
      if (p >= o) return Status::CorruptInput;
      copy_bytes(p, ml);
      continue;
    }
    if (k != 2) return Status::CorruptInput;
    if (ci >= clens.size() || oi + 2 >= coffs.size()) return Status::CorruptInput;
    const usize ml = static_cast<usize>(clens[ci++]) + kMinCopy;
    const u32 off = static_cast<u32>(coffs[oi] | (coffs[oi + 1] << 8) |
                                     (coffs[oi + 2] << 16));
    oi += 3;
    if (off == 0 || off > o || o + ml > n) return Status::CorruptInput;
    copy_bytes(o - off, ml);
  }
  if (o != n || li != lits.size() || si != slots.size() || ri != rlens.size() ||
      ci != clens.size() || oi != coffs.size())
    return Status::CorruptInput;
  return Status::Ok;
}

void write_tbwt_header(std::vector<u8>& dst, u8 flags, usize primary, int chains,
                       const usize* anchors) {
  ByteWriter w(dst);
  w.byte(flags);
  w.varint(primary);
  w.byte(static_cast<u8>(chains));
  if (chains > 1)
    for (int c = 0; c < chains; ++c) w.varint(anchors[c]);
}

void encode_tbwt_tile(const u8* src, usize n, int level, std::vector<u8>& out) {
  thread_local std::vector<u8> lcol, mtf, brle, raw, lz_raw, bzip, cry, lz_scratch;
  lcol.resize(n);
  usize anchors[kBwtMaxChains] = {};
  const int chains = (n >= 16 * 1024) ? bwt_encode_chains() : 1;
  const usize primary =
      bwt_forward(src, n, lcol.data(), chains > 1 ? anchors : nullptr, chains);

  auto pack_ent = [&](const u8* sym, usize sn, u8 flags, std::vector<u8>& dst) {
    dst.clear();
    write_tbwt_header(dst, flags, primary, chains, anchors);
    encode_mosaic_symbols(dst, sym, sn, level);
  };
  auto pack_lz = [&](const u8* sym, u8 flags, std::vector<u8>& dst) {
    lz_scratch.clear();
    build_compressed_block(sym, n, 5, lz_scratch);
    dst.clear();
    write_tbwt_header(dst, static_cast<u8>(flags | 2), primary, chains, anchors);
    dst.insert(dst.end(), lz_scratch.begin(), lz_scratch.end());
  };
  auto pack_bzip = [&](std::vector<u8>& dst) {
    dst.clear();
    write_tbwt_header(dst, 4, primary, chains, anchors);
    encode_mosaic_symbols(dst, brle.data(), brle.size(), level);
  };

  // Crystal's BWT text pipeline (its own BWT -> MTF -> zero-run symbols ->
  // static coder, crystal_bwt.cpp) as a whole-tile candidate: flag 8, no
  // primary/anchors of ours (the payload is self-contained).
  auto pack_cry = [&](std::vector<u8>& dst) {
    dst.clear();
    dst.push_back(8);
    build_crystal_bwt(src, n, crystal_level_for(level), dst);
  };

  // Keep-smaller, except that the Crystal candidate decodes 3-5x faster
  // than the MTF/RLE and raw-column paths (fused inverse MTF, 16-chain
  // inverse BWT), so a rival has to beat it by more than 1% to be taken.
  auto take_best = [&]() {
    const std::vector<u8>* best = nullptr;
    if (!lz_raw.empty()) best = &lz_raw;
    if (!raw.empty() && (!best || raw.size() < best->size())) best = &raw;
    if (!bzip.empty() && (!best || bzip.size() < best->size())) best = &bzip;
    if (!cry.empty() && (!best || cry.size() <= best->size() + best->size() / 100))
      best = &cry;
    if (best) out.insert(out.end(), best->begin(), best->end());
  };

  raw.clear();
  lz_raw.clear();
  bzip.clear();
  cry.clear();
  if (level >= 5 && n >= 4096) pack_cry(cry);

  pack_lz(lcol.data(), 0, lz_raw);
  mtf = lcol;
  mtf_forward(mtf.data(), n);
  bwt_mtf_bzip2_encode(mtf.data(), n, brle);
  pack_bzip(bzip);
  // The raw L column through a stride-1 context coder is the strongest
  // candidate on text and on most numeric planes (no MTF/RLE information
  // loss), so it is raced from L5 on; the MTF variants stay L7+.
  if (level >= 5) pack_ent(lcol.data(), n, 0, raw);
  if (level >= 7) {
    std::vector<u8> with_mtf, lz_mtf;
    pack_ent(mtf.data(), n, 1, with_mtf);
    pack_lz(mtf.data(), 1, lz_mtf);
    if (!with_mtf.empty() && (raw.empty() || with_mtf.size() < raw.size()))
      raw.swap(with_mtf);
    if (!lz_mtf.empty() && (lz_raw.empty() || lz_mtf.size() < lz_raw.size()))
      lz_raw.swap(lz_mtf);
  }
  take_best();
}

Status decode_tbwt_tile(ByteReader& r, u8* dst, usize n) {
  const u8 flags = r.byte();
  if (flags == 8) return decode_crystal_bwt(r.p, r.remaining(), dst, n);
  const u64 primary = r.varint();
  const u8 chains = r.byte();
  if (r.error || chains == 0 || chains > kBwtMaxChains) return Status::CorruptInput;
  usize anchors[kBwtMaxChains] = {};
  if (chains > 1)
    for (int c = 0; c < chains; ++c) anchors[c] = static_cast<usize>(r.varint());
  std::vector<u8> sym;
  if (flags & 4) {
    if (Status s = decode_array(r, sym); s != Status::Ok) return s;
    thread_local std::vector<u8> lcol;
    lcol.resize(n);
    if (bwt_mtf_bzip2_inverse(sym.data(), sym.size(), lcol.data(), n) != n)
      return Status::CorruptInput;
    return bwt_inverse(lcol.data(), n, static_cast<usize>(primary), dst,
                       chains > 1 ? anchors : nullptr, chains);
  }
  if (flags & 2) {
    sym.resize(n);
    if (Status s = decode_compressed_block(r.p, r.remaining(), sym.data(), n);
        s != Status::Ok)
      return s;
  } else {
    if (Status s = decode_array(r, sym); s != Status::Ok) return s;
    if (sym.size() != n) return Status::CorruptInput;
  }
  if (flags & 1) mtf_inverse(sym.data(), n);
  return bwt_inverse(sym.data(), n, static_cast<usize>(primary), dst,
                     chains > 1 ? anchors : nullptr, chains);
}

int tile_log_for(int level, usize n) {
  int log = (level >= 5) ? 20 : (level >= 4) ? 18 : 15;
  if (n < (usize{1} << log)) {
    int l = 8;
    while ((usize{1} << (l + 1)) <= n && l < log) ++l;
    log = l;
  }
  if (log < 8) log = 8;
  return log;
}

bool all_same(const u8* s, usize n) {
  if (n == 0) return true;
  for (usize i = 1; i < n; ++i)
    if (s[i] != s[0]) return false;
  return true;
}

void emit_store(const u8* src, usize n, std::vector<u8>& out) {
  out.push_back(kKindStore);
  ByteWriter w(out);
  w.varint(n);
  w.bytes(src, n);
}

void emit_coded(u8 kind, u8 xflags, const std::vector<u8>& body,
                std::vector<u8>& out) {
  out.push_back(kind);
  ByteWriter w(out);
  w.varint(body.size() + 1);
  w.byte(xflags);
  w.bytes(body);
}

void encode_one_tile(const u8* src, usize n, int level, std::vector<u8>& out) {
  if (n == 0) return;
  if (all_same(src, n)) {
    out.push_back(kKindMemset);
    ByteWriter w(out);
    w.varint(1);
    w.byte(src[0]);
    return;
  }
  if (n < 24) {
    emit_store(src, n, out);
    return;
  }

  const u8* mid = src;
  const u8 xflags = 0;

  u8 kind = kKindRolz;
  std::vector<u8> body;
  if (level >= 4 && n >= 256) {
    encode_tbwt_tile(mid, n, level, body);
    kind = kKindTbwt;
    // Record-shaped tiles (numeric planes, catalogues): the Crystal lattice
    // coder on the untransformed bytes, raced keep-smaller with a 1% bias
    // for its faster decode against the BWT paths (no inverse BWT at all).
    if (level >= 5 && n >= 4096) {
      std::vector<u8> cry;
      build_crystal_static(mid, n, crystal_level_for(level),
                           detect_stride(mid, n), cry);
      if (cry.size() <= body.size() + body.size() / 100) {
        body.swap(cry);
        kind = kKindCrystal;
      }
    }
  } else {
    encode_rolz_tile(mid, n, level, body);
  }

  if (n + 2 <= body.size() + 2) {
    emit_store(src, n, out);
    return;
  }
  emit_coded(kind, xflags, body, out);
}

}  // namespace

void mosaic_encode_tbwt_tile(const u8* src, usize n, int level,
                           std::vector<u8>& out) {
  out.clear();
  encode_tbwt_tile(src, n, level, out);
}

Status mosaic_decode_tbwt_tile(ByteReader& r, u8* dst, usize n) {
  return decode_tbwt_tile(r, dst, n);
}

void build_mosaic_block(const u8* src, usize n, int level, std::vector<u8>& payload) {
  payload.clear();
  if (n == 0) {
    payload.push_back(kMosaicVer);
    payload.push_back(8);
    return;
  }
  const int tlog = tile_log_for(level, n);
  const usize tile = usize{1} << tlog;
  ByteWriter w(payload);
  w.byte(kMosaicVer);
  w.byte(static_cast<u8>(tlog));
  for (usize off = 0; off < n; off += tile)
    encode_one_tile(src + off, min_of(tile, n - off), level, payload);
}

Status decode_mosaic_block(const u8* data, usize len, u8* dst, usize n) {
  if (n == 0) return Status::Ok;
  if (!data || len < 2) return Status::CorruptInput;
  ByteReader r(data, len);
  const u8 ver = r.byte();
  const u8 tlog = r.byte();
  if (r.error || ver != kMosaicVer || tlog < 8 || tlog > 20) return Status::CorruptInput;
  const usize tile = usize{1} << tlog;
  usize o = 0;
  while (o < n) {
    const usize tn = min_of(tile, n - o);
    const u8 kind = r.byte();
    const u64 plen = r.varint();
    if (r.error) return Status::CorruptInput;
    const u8* p = r.take(static_cast<usize>(plen));
    if (!p) return Status::CorruptInput;
    if (kind == kKindStore) {
      if (plen != tn) return Status::CorruptInput;
      std::memcpy(dst + o, p, tn);
    } else if (kind == kKindMemset) {
      if (plen != 1) return Status::CorruptInput;
      std::memset(dst + o, p[0], tn);
    } else if (kind == kKindRolz || kind == kKindTbwt || kind == kKindCrystal) {
      if (plen < 1) return Status::CorruptInput;
      const u8 xflags = p[0];
      ByteReader tr(p + 1, static_cast<usize>(plen) - 1);
      Status s = (kind == kKindRolz)   ? decode_rolz_tile(tr, dst + o, tn)
                 : (kind == kKindTbwt) ? decode_tbwt_tile(tr, dst + o, tn)
                                       : decode_crystal_static(tr.p, tr.remaining(),
                                                               dst + o, tn);
      if (s != Status::Ok) return s;
      const int stride = stride_from_flags(xflags);
      if (stride) undo_xor_stride(dst + o, tn, stride);
    } else {
      return Status::CorruptInput;
    }
    o += tn;
  }
  return r.error ? Status::CorruptInput : Status::Ok;
}

}  // namespace spectra8::detail
