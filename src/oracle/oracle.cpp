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

#include "oracle/oracle.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include "codec.hpp"
#include "crystal/crystal.hpp"
#include "util/entropy.hpp"
#include "mosaic/mosaic.hpp"
#include "util/tans.hpp"

namespace spectra8::detail {
namespace {

constexpr u8 kOracleVer = 2;
constexpr u8 kKindStore = 0;
constexpr u8 kKindMemset = 1;
constexpr u8 kKindLearned = 2;
constexpr u8 kKindArray = 3;
constexpr u8 kKindLz = 4;
constexpr u8 kKindTbwt = 5;
constexpr u8 kKindPred = 6;

constexpr u8 kInnerLearned = 0;
constexpr u8 kInnerArray = 1;
constexpr u8 kInnerLz = 2;
constexpr u8 kInnerTbwt = 3;

constexpr int kMaxCtx = 16;
constexpr int kNumSymbols = 256;
constexpr int kScaleBits = 12;
constexpr u32 kM = 1u << kScaleBits;
constexpr u32 kRansL16 = 1u << 16;

int ctx_count(int level) noexcept { return level >= 5 ? 16 : 8; }

int kmeans_iters(int level) noexcept {
  if (std::getenv("SPECTRA8_CONTEXT_ORACLE") != nullptr) return 16;
  if (level <= 3) return 2;
  if (level <= 8) return 4;
  return 8;
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

void normalize(const u32 counts[kNumSymbols], u64 total, u32 freq[kNumSymbols],
               int num_syms) {
  std::fill(freq, freq + kNumSymbols, u32{0});
  u32 sum = 0;
  for (int s = 0; s < num_syms; ++s) {
    if (counts[s] == 0) continue;
    u32 f = static_cast<u32>((static_cast<u64>(counts[s]) * kM) / total);
    if (f == 0) f = 1;
    freq[s] = f;
    sum += f;
  }
  if (sum < kM) {
    int best = 0;
    for (int s = 1; s < num_syms; ++s)
      if (freq[s] > freq[best]) best = s;
    freq[best] += (kM - sum);
  } else if (sum > kM) {
    u32 excess = sum - kM;
    while (excess > 0) {
      int best = -1;
      for (int s = 0; s < num_syms; ++s)
        if (freq[s] > 1 && (best < 0 || freq[s] > freq[best])) best = s;
      if (best < 0) break;
      const u32 take = min_of(excess, freq[best] - 1);
      freq[best] -= take;
      excess -= take;
    }
  }
}

inline void put_u16(std::vector<u8>& out, u16 v) {
  out.push_back(static_cast<u8>(v & 0xFF));
  out.push_back(static_cast<u8>((v >> 8) & 0xFF));
}

inline void rans_put16(u32& x, u8*& ptr, u32 f, u32 start) {
  constexpr u32 x_max_base = (kRansL16 >> kScaleBits) << 16;
  const u64 x_max = static_cast<u64>(x_max_base) * f;
  if (x >= x_max) {
    ptr -= 2;
    ptr[0] = static_cast<u8>(x);
    ptr[1] = static_cast<u8>(x >> 8);
    x >>= 16;
  }
  x = ((x / f) << kScaleBits) + (x % f) + start;
}

inline Status fill_dsym(u32* dsym, const u8* syms, const u16* freqs, int n) {
  u32 c0 = 0;
  for (int i = 0; i < n; ++i) {
    const u16 f = freqs[i];
    if (!f) continue;
    const u32 packed = static_cast<u32>(syms[i]) |
                       ((static_cast<u32>(f) - 1) << 8) | (c0 << 20);
    for (u32 k = c0; k < c0 + f; ++k) dsym[k] = packed;
    c0 += f;
  }
  return c0 == kM ? Status::Ok : Status::CorruptInput;
}

// Fit a K-class map: k-means on the 256 rows of P(next | prev).
void learn_assign(const u8* src, usize n, int k, int iters, u8 assign[256]) {
  thread_local u32 cond[256][256];
  std::memset(cond, 0, sizeof(cond));
  u32 row[256] = {};
  u8 prev = 0;
  for (usize i = 0; i < n; ++i) {
    ++cond[prev][src[i]];
    ++row[prev];
    prev = src[i];
  }

  for (int b = 0; b < 256; ++b) {
    if (k == 8)
      assign[b] = static_cast<u8>(ans::o1_class(static_cast<u8>(b)));
    else
      assign[b] = static_cast<u8>(b >> 4);
  }

  u32 mean[kMaxCtx][256];
  u32 pop[kMaxCtx];
  for (int it = 0; it < iters; ++it) {
    std::memset(mean, 0, sizeof(mean));
    std::memset(pop, 0, sizeof(pop));
    for (int b = 0; b < 256; ++b) {
      if (!row[b]) continue;
      const int c = assign[b];
      pop[c] += row[b];
      for (int s = 0; s < 256; ++s) mean[c][s] += cond[b][s];
    }
    bool moved = false;
    for (int b = 0; b < 256; ++b) {
      if (!row[b]) {
        assign[b] = 0;
        continue;
      }
      int best = assign[b];
      u64 best_d = ~u64{0};
      for (int c = 0; c < k; ++c) {
        if (!pop[c]) continue;
        u64 d = 0;
        for (int s = 0; s < 256; ++s) {
          const u64 left = static_cast<u64>(cond[b][s]) * pop[c];
          const u64 right = static_cast<u64>(mean[c][s]) * row[b];
          d += left > right ? left - right : right - left;
        }
        if (d < best_d) {
          best_d = d;
          best = c;
        }
      }
      if (static_cast<u8>(best) != assign[b]) {
        assign[b] = static_cast<u8>(best);
        moved = true;
      }
    }
    if (!moved) break;
  }
}

void hist_from_assign(const u8* src, usize n, const u8 assign[256], int k,
                      u32 h[][kNumSymbols]) {
  std::memset(h, 0, static_cast<usize>(k) * sizeof(u32) * kNumSymbols);
  u8 prev = 0;
  for (usize i = 0; i < n; ++i) {
    ++h[assign[prev]][src[i]];
    prev = src[i];
  }
}

void encode_learned_rans(std::vector<u8>& out, const u8* src, usize n, int k,
                         const u8 assign[256], const u32 h[][kNumSymbols]) {
  {
    u32 tot[kNumSymbols] = {};
    for (int c = 0; c < k; ++c)
      for (int s = 0; s < kNumSymbols; ++s) tot[s] += h[c][s];
    int distinct = 0, sym = 0;
    for (int s = 0; s < kNumSymbols; ++s)
      if (tot[s]) {
        ++distinct;
        sym = s;
      }
    if (distinct <= 1) {
      out.push_back(0);
      out.push_back(static_cast<u8>(sym));
      return;
    }
  }

  out.push_back(1);
  u32 freq[kMaxCtx][kNumSymbols];
  u32 cum[kMaxCtx][kNumSymbols + 1];
  std::memset(freq, 0, sizeof(freq));
  for (int c = 0; c < k; ++c) {
    u64 total = 0;
    int max_sym = 0;
    int n_present = 0;
    for (int s = 0; s < kNumSymbols; ++s)
      if (h[c][s]) {
        total += h[c][s];
        max_sym = s + 1;
        ++n_present;
      }
    if (max_sym == 0) {
      put_u16(out, 0);
      continue;
    }
    normalize(h[c], total, freq[c], max_sym);
    cum[c][0] = 0;
    for (int s = 0; s < max_sym; ++s) cum[c][s + 1] = cum[c][s] + freq[c][s];
    for (int s = max_sym; s < kNumSymbols; ++s) {
      freq[c][s] = 0;
      cum[c][s + 1] = cum[c][s];
    }
    const int dense_bytes = 2 * max_sym;
    const int sparse_bytes = 3 * n_present;
    if (sparse_bytes < dense_bytes) {
      put_u16(out, static_cast<u16>(0x8000u | static_cast<u16>(n_present)));
      for (int s = 0; s < max_sym; ++s)
        if (freq[c][s]) {
          out.push_back(static_cast<u8>(s));
          put_u16(out, static_cast<u16>(freq[c][s]));
        }
    } else {
      put_u16(out, static_cast<u16>(max_sym));
      for (int s = 0; s < max_sym; ++s)
        put_u16(out, static_cast<u16>(freq[c][s]));
    }
  }

  std::vector<u8> streams[2];
  for (usize phase = 0; phase < 2 && phase < n; ++phase) {
    std::vector<u8> buf(2 * (n / 2 + 1) + 64);
    u8* bend = buf.data() + buf.size();
    u8* ptr = bend;
    u32 x = kRansL16;
    usize hi = n - 1 - ((n - 1 - phase) % 2);
    for (usize idx = hi;; idx -= 2) {
      const int c = assign[idx > 0 ? src[idx - 1] : u8{0}];
      const u8 s = src[idx];
      rans_put16(x, ptr, freq[c][s], cum[c][s]);
      if (idx < 2) break;
    }
    *--ptr = static_cast<u8>(x >> 24);
    *--ptr = static_cast<u8>(x >> 16);
    *--ptr = static_cast<u8>(x >> 8);
    *--ptr = static_cast<u8>(x >> 0);
    streams[phase].assign(ptr, bend);
  }
  {
    ByteWriter w(out);
    w.varint(streams[0].size());
  }
  out.insert(out.end(), streams[0].begin(), streams[0].end());
  out.insert(out.end(), streams[1].begin(), streams[1].end());
}

Status decode_learned_rans(const u8* data, usize len, u8* dst, usize n, int k,
                           const u8 assign[256]) {
  ByteReader r(data, len);
  const u8 mode = r.byte();
  if (r.error) return Status::CorruptInput;
  if (mode == 0) {
    const u8 sym = r.byte();
    if (r.error) return Status::CorruptInput;
    std::memset(dst, sym, n);
    return Status::Ok;
  }
  if (mode != 1) return Status::CorruptInput;

  thread_local std::vector<u32> dtab;
  dtab.resize(static_cast<usize>(k) * kM);
  for (int c = 0; c < k; ++c) {
    u32* dsym = dtab.data() + static_cast<usize>(c) * kM;
    const u16 hdr =
        static_cast<u16>(r.byte()) | (static_cast<u16>(r.byte()) << 8);
    if (r.error) return Status::CorruptInput;
    if (hdr == 0) {
      std::fill_n(dsym, kM, (u32{kM - 1} << 8));
      continue;
    }
    if (hdr & 0x8000u) {
      const int n_present = hdr & 0x7FFF;
      if (n_present <= 0 || n_present > kNumSymbols) return Status::CorruptInput;
      u8 syms[kNumSymbols];
      u16 freqs[kNumSymbols];
      u32 bitmap[8] = {};
      for (int i = 0; i < n_present; ++i) {
        syms[i] = r.byte();
        freqs[i] =
            static_cast<u16>(r.byte()) | (static_cast<u16>(r.byte()) << 8);
        if (r.error) return Status::CorruptInput;
        const u32 bit = u32{1} << (syms[i] & 31);
        const u32 word = static_cast<u32>(syms[i] >> 5);
        if (bitmap[word] & bit) return Status::CorruptInput;
        bitmap[word] |= bit;
      }
      if (fill_dsym(dsym, syms, freqs, n_present) != Status::Ok)
        return Status::CorruptInput;
    } else {
      const int num_syms = hdr;
      if (num_syms <= 0 || num_syms > kNumSymbols) return Status::CorruptInput;
      u8 syms[kNumSymbols];
      u16 freqs[kNumSymbols];
      int n_present = 0;
      for (int s = 0; s < num_syms; ++s) {
        const u16 f =
            static_cast<u16>(r.byte()) | (static_cast<u16>(r.byte()) << 8);
        if (r.error) return Status::CorruptInput;
        if (!f) continue;
        syms[n_present] = static_cast<u8>(s);
        freqs[n_present] = f;
        ++n_present;
      }
      if (n_present == 0) return Status::CorruptInput;
      if (fill_dsym(dsym, syms, freqs, n_present) != Status::Ok)
        return Status::CorruptInput;
    }
  }

  const usize len0 = static_cast<usize>(r.varint());
  if (r.error || len0 > r.remaining()) return Status::CorruptInput;
  const u8* p0 = r.p;
  const u8* e0 = p0 + len0;
  const u8* p1 = e0;
  const u8* e1 = r.p + r.remaining();
  if (n == 0) return Status::Ok;
  if (len0 < 4 || (n > 1 && static_cast<usize>(e1 - p1) < 4))
    return Status::CorruptInput;

  u32 x0 = static_cast<u32>(p0[0]) | (static_cast<u32>(p0[1]) << 8) |
           (static_cast<u32>(p0[2]) << 16) | (static_cast<u32>(p0[3]) << 24);
  p0 += 4;
  u32 x1 = 0;
  if (n > 1) {
    x1 = static_cast<u32>(p1[0]) | (static_cast<u32>(p1[1]) << 8) |
         (static_cast<u32>(p1[2]) << 16) | (static_cast<u32>(p1[3]) << 24);
    p1 += 4;
  }

  const u32* dsym = dtab.data();
  constexpr u32 mask = kM - 1;
  auto step = [&](u32& x, const u8*& p, const u8* pend, int ctx) -> u8 {
    const u32 e = dsym[(static_cast<u32>(ctx) << kScaleBits) | (x & mask)];
    const u32 f = ((e >> 8) & 0xFFFu) + 1;
    const u32 start = (e >> 20) & 0xFFFu;
    x = f * (x >> kScaleBits) + (x & mask) - start;
    if (x < kRansL16 && p + 2 <= pend) {
      x = (x << 16) | static_cast<u32>(p[0]) | (static_cast<u32>(p[1]) << 8);
      p += 2;
    }
    return static_cast<u8>(e & 0xFF);
  };

  u8 prev = 0;
  usize i = 0;
  for (; i + 2 <= n; i += 2) {
    prev = step(x0, p0, e0, assign[prev]);
    dst[i] = prev;
    prev = step(x1, p1, e1, assign[prev]);
    dst[i + 1] = prev;
  }
  if (i < n) dst[i] = step(x0, p0, e0, assign[prev]);
  return Status::Ok;
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

void emit_kind(u8 kind, const std::vector<u8>& body, std::vector<u8>& out) {
  out.push_back(kind);
  ByteWriter w(out);
  w.varint(body.size());
  w.bytes(body);
}

struct Affine {
  i32 a = 256;  // Q8: identity pred = prev
  i32 b = 0;
};

inline u8 predict_affine(u8 prev, Affine w) noexcept {
  i32 p = (w.a * static_cast<i32>(prev) + w.b) >> 8;
  if (p < 0) p = 0;
  if (p > 255) p = 255;
  return static_cast<u8>(p);
}

void fit_affine(const u8* s, usize n, const u8 assign[256], int k, Affine w[]) {
  double sx[kMaxCtx] = {};
  double sy[kMaxCtx] = {};
  double sxx[kMaxCtx] = {};
  double sxy[kMaxCtx] = {};
  u32 cnt[kMaxCtx] = {};
  u8 prev = 0;
  for (usize i = 0; i < n; ++i) {
    const int c = assign[prev];
    const double x = static_cast<double>(prev);
    const double y = static_cast<double>(s[i]);
    sx[c] += x;
    sy[c] += y;
    sxx[c] += x * x;
    sxy[c] += x * y;
    ++cnt[c];
    prev = s[i];
  }
  for (int c = 0; c < k; ++c) {
    if (cnt[c] < 8) {
      w[c] = Affine{};
      continue;
    }
    const double det = static_cast<double>(cnt[c]) * sxx[c] - sx[c] * sx[c];
    double A = 0, B = sy[c] / static_cast<double>(cnt[c]);
    if (std::fabs(det) > 1e-6) {
      A = (static_cast<double>(cnt[c]) * sxy[c] - sx[c] * sy[c]) / det;
      B = (sy[c] - A * sx[c]) / static_cast<double>(cnt[c]);
    }
    i32 ia = static_cast<i32>(std::lround(A * 256.0));
    i32 ib = static_cast<i32>(std::lround(B * 256.0));
    if (ia < -32768) ia = -32768;
    if (ia > 32767) ia = 32767;
    if (ib < -32768) ib = -32768;
    if (ib > 32767) ib = 32767;
    w[c].a = ia;
    w[c].b = ib;
  }
}

void residual_affine(const u8* s, usize n, const u8 assign[256],
                     const Affine w[], u8* res) {
  u8 prev = 0;
  for (usize i = 0; i < n; ++i) {
    const u8 pred = (i == 0) ? u8{0} : predict_affine(prev, w[assign[prev]]);
    res[i] = static_cast<u8>(s[i] - pred);
    prev = s[i];
  }
}

void undo_affine(u8* d, usize n, const u8 assign[256], const Affine w[]) {
  u8 prev = 0;
  for (usize i = 0; i < n; ++i) {
    const u8 pred = (i == 0) ? u8{0} : predict_affine(prev, w[assign[prev]]);
    d[i] = static_cast<u8>(d[i] + pred);
    prev = d[i];
  }
}

void residual_xor(const u8* s, usize n, int stride, u8* res) {
  const usize st = static_cast<usize>(stride);
  if (st == 0 || st >= n) {
    std::memcpy(res, s, n);
    return;
  }
  std::memcpy(res, s, st);
  for (usize i = st; i < n; ++i) res[i] = static_cast<u8>(s[i] ^ s[i - st]);
}

void undo_xor(u8* d, usize n, int stride) {
  const usize st = static_cast<usize>(stride);
  if (st == 0) return;
  for (usize i = st; i < n; ++i) d[i] = static_cast<u8>(d[i] ^ d[i - st]);
}

void put_i16(std::vector<u8>& out, i32 v) {
  const u16 x = static_cast<u16>(static_cast<std::int16_t>(v));
  out.push_back(static_cast<u8>(x & 0xFF));
  out.push_back(static_cast<u8>(x >> 8));
}

i32 get_i16(const u8* p) {
  return static_cast<i32>(
      static_cast<std::int16_t>(static_cast<u16>(p[0]) | (static_cast<u16>(p[1]) << 8)));
}

void encode_learned_body(const u8* src, usize n, int level, std::vector<u8>& out) {
  out.clear();
  if (n < 512) return;
  const int k = ctx_count(level);
  u8 assign[256];
  learn_assign(src, n, k, kmeans_iters(level), assign);
  u32 h[kMaxCtx][kNumSymbols];
  hist_from_assign(src, n, assign, k, h);
  out.push_back(static_cast<u8>(k));
  out.insert(out.end(), assign, assign + 256);
  encode_learned_rans(out, src, n, k, assign, h);
}

void encode_array_body(const u8* src, usize n, int level, std::vector<u8>& out) {
  out.clear();
  encode_array(out, src, n, level, ArrayKind::Tokens);
}

void encode_lz_body(const u8* src, usize n, std::vector<u8>& out) {
  out.clear();
  build_compressed_block(src, n, 5, out);
}

struct Best {
  u8 kind = kKindStore;
  std::vector<u8> body;
};

void keep_if_better(Best& best, u8 kind, std::vector<u8>& body) {
  if (body.empty()) return;
  if (best.body.empty() || body.size() < best.body.size()) {
    best.kind = kind;
    best.body.swap(body);
  }
}

void pack_pred_affine(const u8 assign[256], int k, const Affine w[], u8 inner,
                      const std::vector<u8>& payload, std::vector<u8>& out) {
  out.clear();
  out.push_back(static_cast<u8>(inner | (0 << 2)));
  out.push_back(static_cast<u8>(k));
  out.insert(out.end(), assign, assign + 256);
  for (int c = 0; c < k; ++c) {
    put_i16(out, w[c].a);
    put_i16(out, w[c].b);
  }
  out.insert(out.end(), payload.begin(), payload.end());
}

void pack_pred_xor(int stride, u8 inner, const std::vector<u8>& payload,
                   std::vector<u8>& out) {
  const u8 pcode = stride == 2 ? 1 : stride == 4 ? 2 : 3;
  out.clear();
  out.push_back(static_cast<u8>(inner | (pcode << 2)));
  out.insert(out.end(), payload.begin(), payload.end());
}

Status decode_inner(u8 inner, const u8* p, usize len, u8* dst, usize n, int level_hint) {
  (void)level_hint;
  if (inner == kInnerLz) {
    if (len < 1) return Status::CorruptInput;
    return decode_compressed_block(p, len, dst, n);
  }
  if (inner == kInnerTbwt) {
    if (len < 1) return Status::CorruptInput;
    ByteReader r(p, len);
    return mosaic_decode_tbwt_tile(r, dst, n);
  }
  if (inner == kInnerArray) {
    ByteReader r(p, len);
    std::vector<u8> sym;
    if (Status s = decode_array(r, sym); s != Status::Ok) return s;
    if (sym.size() != n) return Status::CorruptInput;
    std::memcpy(dst, sym.data(), n);
    return Status::Ok;
  }
  if (inner != kInnerLearned || len < 258) return Status::CorruptInput;
  const int k = p[0];
  if (k != 8 && k != 16) return Status::CorruptInput;
  u8 assign[256];
  std::memcpy(assign, p + 1, 256);
  for (int i = 0; i < 256; ++i)
    if (assign[i] >= static_cast<u8>(k)) return Status::CorruptInput;
  return decode_learned_rans(p + 257, len - 257, dst, n, k, assign);
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
  if (n < 48) {
    emit_store(src, n, out);
    return;
  }

  Best best;
  thread_local std::vector<u8> tmp, tmp2, res;

  encode_array_body(src, n, level, tmp);
  keep_if_better(best, kKindArray, tmp);

  // Crystal static lattice coder on the raw tile (record stride detected).
  // Same kKindArray blob layout, so the decoder needs nothing new; this is
  // the strongest static candidate on numeric planes and beats the k-means
  // learned rANS below (whose 8/16 contexts it generalises).
  const int cry_level = level >= 9 ? 5 : level >= 7 ? 3 : 2;
  const int stride = detect_stride(src, n);
  thread_local std::vector<u8> cry_raw;
  cry_raw.clear();
  if (encode_array_crystal(cry_raw, src, n, cry_level, stride)) {
    tmp = cry_raw;
    keep_if_better(best, kKindArray, tmp);
  }

  const int k = ctx_count(level);
  u8 assign[256];
  bool have_assign = false;
  if (n >= 512) {
    learn_assign(src, n, k, kmeans_iters(level), assign);
    have_assign = true;
    u32 h[kMaxCtx][kNumSymbols];
    hist_from_assign(src, n, assign, k, h);
    tmp.clear();
    tmp.push_back(static_cast<u8>(k));
    tmp.insert(tmp.end(), assign, assign + 256);
    encode_learned_rans(tmp, src, n, k, assign, h);
    keep_if_better(best, kKindLearned, tmp);
  }

  if (n >= 64) {
    encode_lz_body(src, n, tmp);
    keep_if_better(best, kKindLz, tmp);
  }

  if (level >= 4 && n >= 256) {
    mosaic_encode_tbwt_tile(src, n, level, tmp);
    keep_if_better(best, kKindTbwt, tmp);
  }

  // Per-cluster affine predictor. L5 races LZ on the residual (cheap, often
  // the winner on numeric planes). Extra residual BWT / xor-stride LZ only
  // at L7+ so the default path stays one BWT + two LZ.
  if (have_assign) {
    Affine w[kMaxCtx];
    fit_affine(src, n, assign, k, w);
    res.resize(n);
    residual_affine(src, n, assign, w, res.data());
    encode_lz_body(res.data(), n, tmp2);
    pack_pred_affine(assign, k, w, kInnerLz, tmp2, tmp);
    keep_if_better(best, kKindPred, tmp);
    tmp2.clear();
    if (encode_array_crystal(tmp2, res.data(), n, cry_level, stride)) {
      pack_pred_affine(assign, k, w, kInnerArray, tmp2, tmp);
      keep_if_better(best, kKindPred, tmp);
    }
    if (level >= 7) {
      encode_learned_body(res.data(), n, level, tmp2);
      if (!tmp2.empty()) {
        pack_pred_affine(assign, k, w, kInnerLearned, tmp2, tmp);
        keep_if_better(best, kKindPred, tmp);
      }
      if (n >= 256) {
        mosaic_encode_tbwt_tile(res.data(), n, level, tmp2);
        pack_pred_affine(assign, k, w, kInnerTbwt, tmp2, tmp);
        keep_if_better(best, kKindPred, tmp);
      }
    }
  }

  if (level >= 7 && n >= 64) {
    res.resize(n);
    for (int stride : {2, 4, 8}) {
      residual_xor(src, n, stride, res.data());
      encode_lz_body(res.data(), n, tmp2);
      pack_pred_xor(stride, kInnerLz, tmp2, tmp);
      keep_if_better(best, kKindPred, tmp);
    }
  }

  // The plain Crystal tile decodes without the serial predictor / inverse
  // BWT / LZ passes the other winners need, so it is kept unless a rival is
  // more than 1% smaller.
  if (!cry_raw.empty() && best.kind != kKindArray &&
      cry_raw.size() <= best.body.size() + best.body.size() / 100) {
    best.kind = kKindArray;
    best.body = cry_raw;
  }

  if (best.body.empty() || best.body.size() + 2 >= n + 2) {
    emit_store(src, n, out);
    return;
  }
  emit_kind(best.kind, best.body, out);
}

}  // namespace

void build_oracle_block(const u8* src, usize n, int level, std::vector<u8>& payload) {
  payload.clear();
  if (n == 0) {
    payload.push_back(kOracleVer);
    payload.push_back(8);
    return;
  }
  const int tlog = tile_log_for(level, n);
  ByteWriter w(payload);
  w.byte(kOracleVer);
  w.byte(static_cast<u8>(tlog));
  const usize tile = usize{1} << tlog;
  for (usize off = 0; off < n; off += tile)
    encode_one_tile(src + off, min_of(tile, n - off), level, payload);
}

Status decode_oracle_block(const u8* data, usize len, u8* dst, usize n) {
  if (n == 0) return Status::Ok;
  if (!data || len < 2) return Status::CorruptInput;
  ByteReader r(data, len);
  const u8 ver = r.byte();
  const u8 tlog = r.byte();
  if (r.error || (ver != 1 && ver != 2) || tlog < 8 || tlog > 20)
    return Status::CorruptInput;
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
    } else if (kind == kKindArray) {
      if (Status s = decode_inner(kInnerArray, p, static_cast<usize>(plen),
                                  dst + o, tn, 5);
          s != Status::Ok)
        return s;
    } else if (kind == kKindLearned) {
      if (Status s = decode_inner(kInnerLearned, p, static_cast<usize>(plen),
                                  dst + o, tn, 5);
          s != Status::Ok)
        return s;
    } else if (kind == kKindLz) {
      if (Status s = decode_inner(kInnerLz, p, static_cast<usize>(plen), dst + o,
                                  tn, 5);
          s != Status::Ok)
        return s;
    } else if (kind == kKindTbwt) {
      if (Status s = decode_inner(kInnerTbwt, p, static_cast<usize>(plen),
                                  dst + o, tn, 5);
          s != Status::Ok)
        return s;
    } else if (kind == kKindPred) {
      if (plen < 1) return Status::CorruptInput;
      const u8 flags = p[0];
      const u8 inner = flags & 3;
      const u8 pcode = (flags >> 2) & 3;
      const u8* ip = p + 1;
      usize ilen = static_cast<usize>(plen) - 1;
      u8 assign[256];
      Affine w[kMaxCtx];
      int k = 0;
      if (pcode == 0) {
        if (ilen < 1 + 256) return Status::CorruptInput;
        k = ip[0];
        if (k != 8 && k != 16) return Status::CorruptInput;
        std::memcpy(assign, ip + 1, 256);
        ip += 1 + 256;
        ilen -= 1 + 256;
        const usize wbytes = static_cast<usize>(k) * 4;
        if (ilen < wbytes) return Status::CorruptInput;
        for (int c = 0; c < k; ++c) {
          w[c].a = get_i16(ip + static_cast<usize>(c) * 4);
          w[c].b = get_i16(ip + static_cast<usize>(c) * 4 + 2);
        }
        for (int i = 0; i < 256; ++i)
          if (assign[i] >= static_cast<u8>(k)) return Status::CorruptInput;
        ip += wbytes;
        ilen -= wbytes;
      }
      if (Status s = decode_inner(inner, ip, ilen, dst + o, tn, 5);
          s != Status::Ok)
        return s;
      if (pcode == 0)
        undo_affine(dst + o, tn, assign, w);
      else {
        const int stride = pcode == 1 ? 2 : pcode == 2 ? 4 : 8;
        undo_xor(dst + o, tn, stride);
      }
    } else {
      return Status::CorruptInput;
    }
    o += tn;
  }
  return r.error ? Status::CorruptInput : Status::Ok;
}

}  // namespace spectra8::detail
