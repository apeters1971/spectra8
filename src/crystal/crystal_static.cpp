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

// Crystal static lattice coder: the fast-decode mode (levels 1-9).
//
// All modelling happens in the encoder. For every lane of the record lattice
// it picks a base the byte is coded relative to and one or two context
// features (previous byte(s), byte one/two records back, the linear trend,
// a record-repeat flag), merges feature values into groups by dynamic
// programming on an MDL cost (coded bits + table bits), quantizes one
// frequency table per group and writes the model in a compact header. The
// decoder does, per byte: group lookup -> static table -> one rANS step, on
// 16 interleaved streams in lockstep (8 for small blocks). No adaptation, no mixing.
//
// Payload body (after the 2-byte (ver, kind) prefix written by crystal.cpp):
//   u8 stride, u8 K (streams), varint model_len, model (HdrEnc bit stream:
//   per lane a mode - coded: fa/fb/base, group counts, group maps;
//   constant: the value; raw: nothing - then the coded lanes' frequency
//   tables), the raw lanes' bytes verbatim (lane by lane), varint stream
//   sizes (K-1), the K rANS streams. Constant and raw lanes (LaneMode) are
//   filled before the rANS pass and skipped by it.
#include "crystal/crystal.hpp"
#include "util/transform.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace spectra8::detail {
namespace {

constexpr int kPrec = 12;
constexpr u32 kM = u32{1} << kPrec;
// Interleaved rANS streams per block: 16 from 64 KiB (two independent
// 8-wide groups for the x86-64 vector body, whose gather chain is latency
// bound with a single group), halving down to 1 for small blocks. Costs
// ~6 bytes per extra stream; the decoder takes any k <= kMaxStreams.
constexpr int kMaxStreams = 16;
[[maybe_unused]] constexpr int kV8Width = 8;  // streams per vector group
constexpr int kMaxLanes = 16;
constexpr int kMaxGroups = 64;     // groups per feature
constexpr int kMaxCtxLane = 64;    // ka * kb
constexpr u32 kRansL = u32{1} << 16;

// kRecEq: 1 while every earlier byte of the current record equals the byte
// one record back (whole-record repeats), else 0. Maintained per stream.
enum Feat : u8 { kNone = 0, kC1, kC2, kLat1, kLat2, kTrend, kRecEq, kFeatCount };

inline u32 feat_at(const u8* buf, usize pos, usize start, usize off) noexcept {
  return pos >= start + off ? buf[pos - off] : 0u;
}

inline u32 feat_value(Feat f, const u8* buf, usize pos, usize start, usize S,
                      u32 rec_eq = 0) noexcept {
  switch (f) {
    case kC1: return feat_at(buf, pos, start, 1);
    case kC2: return feat_at(buf, pos, start, 2);
    case kLat1: return feat_at(buf, pos, start, S);
    case kLat2: return feat_at(buf, pos, start, 2 * S);
    case kTrend:
      return (2 * feat_at(buf, pos, start, S) - feat_at(buf, pos, start, 2 * S)) & 255u;
    case kRecEq: return rec_eq;
    default: return 0;
  }
}

// Value order for the interval quantizers: unsigned or two's-complement.
inline u32 perm_value(u32 v, bool signed_order) noexcept {
  return signed_order ? (v ^ 0x80u) : v;
}

// ---------------------------------------------------------------------------
// Adaptive binary coder for the model header (size matters, speed does not).
// ---------------------------------------------------------------------------
struct HdrEnc {
  u32 x1 = 0, x2 = 0xFFFFFFFFu;
  std::vector<u8>* out = nullptr;
  void bit(u16& p, int y) {
    const u32 xmid = x1 + static_cast<u32>((static_cast<u64>(x2 - x1) * p) >> 16);
    if (y)
      x2 = xmid;
    else
      x1 = xmid + 1;
    int np = static_cast<int>(p) + (((y << 16) - static_cast<int>(p)) >> 5);
    p = static_cast<u16>(np < 64 ? 64 : np > 65472 ? 65472 : np);
    while (((x1 ^ x2) & 0xFF000000u) == 0) {
      out->push_back(static_cast<u8>(x2 >> 24));
      x1 <<= 8;
      x2 = (x2 << 8) | 255;
    }
  }
  void tree(u16* t, int nbits, u32 v) {
    u32 node = 1;
    for (int i = nbits - 1; i >= 0; --i) {
      const int y = (v >> i) & 1;
      bit(t[node], y);
      node = node * 2 + static_cast<u32>(y);
    }
  }
  void raw(int nbits, u32 v) {
    for (int i = nbits - 1; i >= 0; --i) {
      u16 p = 32768;
      bit(p, (v >> i) & 1);
    }
  }
  void flush() {
    for (int i = 0; i < 4; ++i) {
      out->push_back(static_cast<u8>(x1 >> 24));
      x1 <<= 8;
    }
  }
};

struct HdrDec {
  u32 x1 = 0, x2 = 0xFFFFFFFFu, x = 0;
  const u8* p = nullptr;
  const u8* end = nullptr;
  void init(const u8* d, usize n) {
    p = d;
    end = d + n;
    for (int i = 0; i < 4; ++i) x = (x << 8) | next();
  }
  u8 next() noexcept { return p < end ? *p++ : u8{0}; }
  // Branch-free on the decision: the mantissa bits of the frequencies are
  // close to random and a mispredict per bit would dominate header time.
  int bit(u16& p_) {
    const u32 xmid = x1 + static_cast<u32>((static_cast<u64>(x2 - x1) * p_) >> 16);
    const int y = x <= xmid;
    const u32 m = 0u - static_cast<u32>(y);  // all ones if y
    x2 = (xmid & m) | (x2 & ~m);
    x1 = ((xmid + 1) & ~m) | (x1 & m);
    int np = static_cast<int>(p_) + (((y << 16) - static_cast<int>(p_)) >> 5);
    p_ = static_cast<u16>(np < 64 ? 64 : np > 65472 ? 65472 : np);
    while (((x1 ^ x2) & 0xFF000000u) == 0) {
      x1 <<= 8;
      x2 = (x2 << 8) | 255;
      x = (x << 8) | next();
    }
    return y;
  }
  u32 tree(u16* t, int nbits) {
    u32 node = 1;
    for (int i = 0; i < nbits; ++i) node = node * 2 + static_cast<u32>(bit(t[node]));
    return node - (u32{1} << nbits);
  }
  // Same arithmetic as bit() with p = 32768 (the encoder codes raw bits
  // through the adaptive path with a fresh p), minus the model update.
  u32 raw(int nbits) {
    u32 v = 0;
    for (int i = 0; i < nbits; ++i) {
      const u32 xmid = x1 + ((x2 - x1) >> 1);
      const u32 y = x <= xmid;
      const u32 m = 0u - y;
      x2 = (xmid & m) | (x2 & ~m);
      x1 = ((xmid + 1) & ~m) | (x1 & m);
      v = (v << 1) | y;
      while (((x1 ^ x2) & 0xFF000000u) == 0) {
        x1 <<= 8;
        x2 = (x2 << 8) | 255;
        x = (x << 8) | next();
      }
    }
    return v;
  }
};

// Plain LSB-first bit stream for the incompressible parts of the header
// (frequency mantissas). Arithmetic-coding them costs the same bits but ~5
// cycles each on the range-coder chain; this is ~1 cycle per bit.
struct SideBitsEnc {
  std::vector<u8> bytes;
  u64 acc = 0;
  int nacc = 0;
  void put(int nbits, u32 v) {
    acc |= static_cast<u64>(v) << nacc;
    nacc += nbits;
    while (nacc >= 8) {
      bytes.push_back(static_cast<u8>(acc));
      acc >>= 8;
      nacc -= 8;
    }
  }
  void flush() {
    if (nacc > 0) bytes.push_back(static_cast<u8>(acc));
    acc = 0;
    nacc = 0;
  }
};

struct SideBitsDec {
  const u8* p = nullptr;
  const u8* end = nullptr;
  u64 acc = 0;
  int nacc = 0;
  void init(const u8* d, usize n) {
    p = d;
    end = d + n;
  }
  u32 get(int nbits) {  // nbits <= 24; reads zeros past the end
    while (nacc < nbits) {
      acc |= static_cast<u64>(p < end ? *p++ : u8{0}) << nacc;
      nacc += 8;
    }
    const u32 v = static_cast<u32>(acc) & ((u32{1} << nbits) - 1);
    acc >>= nbits;
    nacc -= nbits;
    return v;
  }
};

// Frequency table coding: per symbol a presence flag (context: presence in
// the reference table and of the previous symbol), then the bit length
// (context: bit length in the reference table) and raw mantissa bits.
struct TableModels {
  u16 nz[4];
  u16 len[kPrec + 2][16];
  u16 empty[2];
  TableModels() {
    std::fill(nz, nz + 4, u16{32768});
    for (auto& row : len) std::fill(row, row + 16, u16{32768});
    empty[0] = empty[1] = 32768;
  }
};

inline int bit_length(u32 v) noexcept { return v ? 32 - __builtin_clz(v) : 0; }

void encode_table(HdrEnc& e, SideBitsEnc& side, TableModels& m, const u16* f,
                  const u16* ref) {
  int prev_nz = 0;
  for (int s = 0; s < 256; ++s) {
    const int rnz = ref ? (ref[s] > 0) : 0;
    const int nz = f[s] > 0;
    e.bit(m.nz[rnz * 2 + prev_nz], nz);
    prev_nz = nz;
    if (!nz) continue;
    const int rl = ref ? bit_length(ref[s]) : 0;
    const int l = bit_length(f[s]);  // 1..kPrec+1 (kM itself)
    e.tree(m.len[rl], 4, static_cast<u32>(l));
    if (l > 1) side.put(l - 1, f[s] & ((u32{1} << (l - 1)) - 1));
  }
}

bool decode_table(HdrDec& d, SideBitsDec& side, TableModels& m, u16* f, const u16* ref) {
  int prev_nz = 0;
  u32 sum = 0;
  for (int s = 0; s < 256; ++s) {
    const int rnz = ref ? (ref[s] > 0) : 0;
    const int nz = d.bit(m.nz[rnz * 2 + prev_nz]);
    prev_nz = nz;
    if (!nz) {
      f[s] = 0;
      continue;
    }
    const int rl = ref ? bit_length(ref[s]) : 0;
    const int l = static_cast<int>(d.tree(m.len[rl], 4));
    if (l < 1 || l > kPrec + 1) return false;
    u32 v = u32{1} << (l - 1);
    if (l > 1) v |= side.get(l - 1);
    if (v > kM) return false;
    f[s] = static_cast<u16>(v);
    sum += v;
  }
  return sum == kM;
}

// Counts -> frequencies summing to kM, every present symbol >= 1.
void quantize(const u32* cnt, u16* f) {
  u64 total = 0;
  int present = 0;
  for (int s = 0; s < 256; ++s) {
    total += cnt[s];
    present += cnt[s] != 0;
  }
  std::fill(f, f + 256, u16{0});
  if (total == 0 || present == 0) {
    f[0] = static_cast<u16>(kM);
    return;
  }
  u32 sum = 0;
  for (int s = 0; s < 256; ++s) {
    if (!cnt[s]) continue;
    u32 v = static_cast<u32>((static_cast<u64>(cnt[s]) * kM) / total);
    if (v < 1) v = 1;
    if (v > kM - static_cast<u32>(present - 1)) v = kM - static_cast<u32>(present - 1);
    f[s] = static_cast<u16>(v);
    sum += v;
  }
  // Fix the remainder where it costs least (cnt * log ratio).
  // Cache the marginal gains/losses per symbol and update only the chosen symbol
  // each step, cutting transcendentals from O(S * kM) to O(S + |kM - sum|).
  if (sum < kM) {
    double g[256];
    for (int s = 0; s < 256; ++s)
      g[s] = cnt[s] ? cnt[s] * std::log2((f[s] + 1.0) / f[s]) : -1.0;
    while (sum < kM) {
      int best = -1;
      double bg = -1.0;
      for (int s = 0; s < 256; ++s) {
        if (g[s] > bg) {
          bg = g[s];
          best = s;
        }
      }
      if (best < 0) break;
      ++f[best];
      ++sum;
      g[best] = cnt[best] * std::log2((f[best] + 1.0) / f[best]);
    }
  } else if (sum > kM) {
    double loss[256];
    for (int s = 0; s < 256; ++s)
      loss[s] = (f[s] > 1) ? cnt[s] * std::log2(static_cast<double>(f[s]) / (f[s] - 1.0)) : 1e300;
    while (sum > kM) {
      int best = -1;
      double bl = 1e300;
      for (int s = 0; s < 256; ++s) {
        if (loss[s] < bl) {
          bl = loss[s];
          best = s;
        }
      }
      if (best < 0) break;
      --f[best];
      --sum;
      loss[best] = (f[best] > 1) ? cnt[best] * std::log2(static_cast<double>(f[best]) / (f[best] - 1.0)) : 1e300;
    }
  }
}

// ---------------------------------------------------------------------------
// Model description shared by encoder and decoder.
// ---------------------------------------------------------------------------
// How a lane's bytes travel. A coded lane goes through the rANS streams. A
// constant lane (every byte of the lane equal) ships one value byte in the
// model; a raw lane (coded size >= its byte count, i.e. the lane is noise)
// ships its bytes verbatim in a section after the model. Neither has
// contexts, tables or stream symbols: the decoder fills them before the
// rANS pass and the lockstep loop only shifts their bytes into the byte
// history (3 instructions per byte instead of ~30, one gather instead of
// five in the vector body). A profile's model codes every lane (it serves
// pages its sample did not see); a profile block carries the modes of its
// own page instead (LaneModes, kCryProfileLanes).
enum LaneMode : u8 { kLaneCoded = 0, kLaneConst = 1, kLaneRaw = 2 };

struct LaneModel {
  Feat fa = kNone, fb = kNone;
  Feat rbase = kNone;  // coded symbol = (byte - base) & 255
  int ka = 1, kb = 1;
  u8 qa[256] = {}, qb[256] = {};  // raw value -> group
  u32 base = 0;                   // first context index
  u8 mode = kLaneCoded;
  u8 value = 0;  // kLaneConst: the lane's byte
  int nctx() const noexcept { return mode == kLaneCoded ? ka * kb : 0; }
};

struct Model {
  int stride = 1;
  LaneModel lane[kMaxLanes];
  u32 nctx = 0;
  std::vector<u16> freq;  // nctx * 256
};

// ---------------------------------------------------------------------------
// Encoder analysis.
// ---------------------------------------------------------------------------
constexpr double kTableBits = 24.0;     // fixed cost of one table
constexpr double kSymbolBits = 7.5;     // per present symbol
constexpr double kBoundaryBits = 9.0;   // per group boundary
// A lane goes raw when its estimated coded size is at least this fraction
// of its byte count: 0 = never larger than coded.
constexpr double kRawLaneSlack = 0.0;

// x * log2(x) for the analysis sample counts (sample capped at kAnalysisMax
// symbols per lane, so the table covers every count).
constexpr usize kAnalysisMax = usize{1} << 17;

struct XLogX {
  std::vector<float> t;
  XLogX() : t(kAnalysisMax + 1) {
    t[0] = 0;
    for (usize x = 1; x <= kAnalysisMax; ++x)
      t[x] = static_cast<float>(x * std::log2(static_cast<double>(x)));
  }
};

const XLogX& xlogx() {
  static const XLogX tab;
  return tab;
}

struct Scratch {
  std::vector<u32> cnt;      // [nv][nsym]
  std::vector<double> cost;  // [nv+1][nv+1] interval costs
  std::vector<u32> pre;      // [ng][nv+1][nsym] prefix histograms (refinement)
  std::vector<u32> binh;     // [nv][ng][nsym] bin histograms (pre-binning)
  std::vector<double> dp;    // [nv+1]
  std::vector<u8> from;      // [nv+1]
  std::vector<double> dpk;   // [kmax+1][nv+1] (constrained fallback)
  std::vector<u8> fromk;
  std::vector<u32> cnt2;     // [ka][nv][nsym]
  std::vector<u32> cnt_pick;  // search_pairs: table of the picked second feature
  std::vector<u32> pair_cnt[kFeatCount];  // search_pairs: tables of the second features
  std::vector<u32> nz_start; // CSR of the nonzero count entries
  std::vector<u8> nz_sym;
  std::vector<u32> nz_cnt;
};

Scratch& scratch() {
  thread_local Scratch s;
  return s;
}

// Entropy + table header bits of a histogram over nsym dense symbols
// (counts <= kAnalysisMax).
inline double xlogx_of(u32 x, const float* xl) {
  return x <= kAnalysisMax ? xl[x] : x * std::log2(static_cast<double>(x));
}

inline double table_cost(const u32* h, int nsym, const float* xl) {
  u32 n = 0;
  int present = 0;
  double sum_xlx = 0;
  for (int s = 0; s < nsym; ++s) {
    n += h[s];
    present += h[s] != 0;
    sum_xlx += xlogx_of(h[s], xl);
  }
  if (present == 0) return 1.0;  // "empty" flag
  const double b = xlogx_of(n, xl) - sum_xlx;
  return b + kTableBits + kSymbolBits * present;
}

// Interval costs over nv dense values: cost[i * (nv + 1) + j] is the table
// cost of merging values i..j-1 (added to what is there when `accumulate`).
// Each interval is extended one value at a time, touching only the symbols
// that value co-occurs with, so the work is O(nv * nnz) rather than
// O(nv^2 * nsym). cnt is [nv][nsym].
//
// The intervals starting at R consecutive values advance together: past
// the first R values they add the same entries, so one pass over the CSR
// data feeds R independent histogram/sum chains (each chain alone is bound
// by the latency of its histogram update and the gathered xlogx loads).
// Per row the arithmetic is the one-row sequence, so the costs are the
// same to the bit whatever R.
constexpr int kIntervalRows = 4;

template <int R>
void interval_rows(const Scratch& sc, int i, int nv, int nsym, const float* xl, double* cost,
                   usize W, bool accumulate) {
  u32 h[R][256];
  for (int r = 0; r < R; ++r) std::memset(h[r], 0, sizeof(u32) * static_cast<usize>(nsym));
  u32 n[R] = {};
  int present[R] = {};
  // Two accumulators per row, even and odd entries of each value's list.
  double sum0[R] = {}, sum1[R] = {};
  auto add_one = [&](int r, u32 e) {
    const u32 s2 = sc.nz_sym[e];
    const u32 c = sc.nz_cnt[e];
    const u32 old = h[r][s2];
    present[r] += old == 0;
    sum0[r] += xl[old + c] - xl[old];
    h[r][s2] = old + c;
    n[r] += c;
  };
  auto add_two = [&](int r, u32 e) {
    const u32 sa = sc.nz_sym[e], sb = sc.nz_sym[e + 1];  // distinct symbols
    const u32 ca = sc.nz_cnt[e], cb = sc.nz_cnt[e + 1];
    const u32 oa = h[r][sa], ob = h[r][sb];
    present[r] += (oa == 0) + (ob == 0);
    sum0[r] += xl[oa + ca] - xl[oa];
    sum1[r] += xl[ob + cb] - xl[ob];
    h[r][sa] = oa + ca;
    h[r][sb] = ob + cb;
    n[r] += ca + cb;
  };
  auto emit = [&](int r, int j) {
    const double v = present[r] == 0
                         ? 1.0
                         : (xl[n[r]] - (sum0[r] + sum1[r])) + kTableBits + kSymbolBits * present[r];
    double& slot = cost[static_cast<usize>(i + r) * W + static_cast<usize>(j + 1)];
    slot = accumulate ? slot + v : v;
  };
  // Row r starts at value i + r: its first R - 1 - r values go alone.
  for (int r = 0; r < R; ++r)
    for (int j = i + r; j < i + R - 1; ++j) {
      u32 e = sc.nz_start[j];
      const u32 end = sc.nz_start[j + 1];
      for (; e + 1 < end; e += 2) add_two(r, e);
      if (e < end) add_one(r, e);
      emit(r, j);
    }
  for (int j = i + R - 1; j < nv; ++j) {
    u32 e = sc.nz_start[j];
    const u32 end = sc.nz_start[j + 1];
    for (; e + 1 < end; e += 2)
      for (int r = 0; r < R; ++r) add_two(r, e);
    if (e < end)
      for (int r = 0; r < R; ++r) add_one(r, e);
    for (int r = 0; r < R; ++r) emit(r, j);
  }
}

void interval_costs(const u32* cnt, int nv, int nsym, double* cost, bool accumulate) {
  const float* xl = xlogx().t.data();
  Scratch& sc = scratch();
  // CSR of the nonzero entries per value row.
  sc.nz_start.resize(static_cast<usize>(nv) + 1);
  sc.nz_sym.clear();
  sc.nz_cnt.clear();
  for (int v = 0; v < nv; ++v) {
    sc.nz_start[v] = static_cast<u32>(sc.nz_sym.size());
    const u32* row = cnt + static_cast<usize>(v) * nsym;
    for (int s2 = 0; s2 < nsym; ++s2)
      if (row[s2]) {
        sc.nz_sym.push_back(static_cast<u8>(s2));
        sc.nz_cnt.push_back(row[s2]);
      }
  }
  sc.nz_start[nv] = static_cast<u32>(sc.nz_sym.size());
  const usize W = static_cast<usize>(nv) + 1;
  int i = 0;
  for (; i + kIntervalRows <= nv; i += kIntervalRows)
    interval_rows<kIntervalRows>(sc, i, nv, nsym, xl, cost, W, accumulate);
  for (; i < nv; ++i) interval_rows<1>(sc, i, nv, nsym, xl, cost, W, accumulate);
}

// Segment the nv dense values into groups minimising the sum of interval
// costs + kBoundaryBits per boundary. First an unconstrained O(nv^2) pass
// (the group count falls out of the optimisation); only when that exceeds
// kmax the exact constrained DP over k <= kmax is run. Fills bounds (k-1
// dense indices) and returns the total cost.
double segment(const double* cost, int nv, int kmax, int& k_out, u8* bounds) {
  Scratch& sc = scratch();
  const usize W = static_cast<usize>(nv) + 1;
  sc.dp.resize(W);
  sc.from.resize(W);
  sc.dp[0] = 0;
  for (int j = 1; j <= nv; ++j) {
    double best = 1e300;
    int bi = 0;
    const double* col = cost + static_cast<usize>(j);
    for (int i = 0; i < j; ++i) {
      const double c = sc.dp[i] + col[static_cast<usize>(i) * W];
      if (c < best) {
        best = c;
        bi = i;
      }
    }
    sc.dp[j] = best + kBoundaryBits;
    sc.from[j] = static_cast<u8>(bi);
  }
  int k = 0;
  for (int j = nv; j > 0; j = sc.from[j]) ++k;
  if (k <= kmax) {
    int j = nv;
    for (int g = k; g >= 1; --g) {
      const int i = sc.from[j];
      if (g > 1) bounds[g - 2] = static_cast<u8>(i);
      j = i;
    }
    k_out = k;
    return sc.dp[nv] - kBoundaryBits;
  }
  // Constrained: dpk[g][j] = best cost of covering values [0, j) with g groups.
  const usize K = static_cast<usize>(kmax) + 1;
  sc.dpk.assign(K * W, 1e300);
  sc.fromk.assign(K * W, 0);
  sc.dpk[0] = 0;
  for (int g = 1; g <= kmax; ++g) {
    double* dpg = sc.dpk.data() + static_cast<usize>(g) * W;
    const double* dpp = sc.dpk.data() + static_cast<usize>(g - 1) * W;
    u8* fr = sc.fromk.data() + static_cast<usize>(g) * W;
    for (int j = g; j <= nv; ++j) {
      double best = 1e300;
      int bi = 0;
      const double* col = cost + static_cast<usize>(j);
      for (int i = g - 1; i < j; ++i) {
        const double c = dpp[i] + col[static_cast<usize>(i) * W];
        if (c < best) {
          best = c;
          bi = i;
        }
      }
      dpg[j] = best + kBoundaryBits;
      fr[j] = static_cast<u8>(bi);
    }
  }
  double best = 1e300;
  int bk = 1;
  for (int g = 1; g <= kmax; ++g) {
    const double c = sc.dpk[static_cast<usize>(g) * W + static_cast<usize>(nv)];
    if (c < best) {
      best = c;
      bk = g;
    }
  }
  int j = nv;
  for (int g = bk; g >= 1; --g) {
    const int i = sc.fromk[static_cast<usize>(g) * W + static_cast<usize>(j)];
    if (g > 1) bounds[g - 2] = static_cast<u8>(i);
    j = i;
  }
  k_out = bk;
  return best - kBoundaryBits;
}

struct LaneStats {
  // Feature values and raw bytes of the lane (analysis sample).
  std::vector<u8> raw;
  std::vector<u8> fv[kFeatCount];
  // Symbols relative to a base, as dense alphabet indices (0..nsym-1, in
  // symbol order), with their histogram; computed once per base.
  std::vector<u8> sym_of[kFeatCount];
  u32 hist_of[kFeatCount][256];
  int nsym_of[kFeatCount] = {};
  // The current base.
  const u8* sym = nullptr;
  const u32* hist = nullptr;
  int nsym = 0;
  Feat base = kNone;

  usize size() const noexcept { return raw.size(); }

  void set_base(Feat rbase) {
    base = rbase;
    if (nsym_of[rbase] == 0) compute(rbase);
    sym = sym_of[rbase].data();
    hist = hist_of[rbase];
    nsym = nsym_of[rbase];
  }

  void compute(Feat rbase) {
    const usize n = raw.size();
    std::vector<u8>& out = sym_of[rbase];
    out.resize(n);
    const u8* r = raw.data();
    u8* s = out.data();
    if (rbase == kNone) {
      std::memcpy(s, r, n);
    } else {
      const u8* b = fv[rbase].data();
      for (usize i = 0; i < n; ++i) s[i] = static_cast<u8>(r[i] - b[i]);
    }
    // Four interleaved histograms: a run of one symbol would otherwise
    // serialise on the store-to-load forwarding latency.
    u32 h[4][256] = {};
    usize i = 0;
    for (; i + 4 <= n; i += 4) {
      ++h[0][s[i]];
      ++h[1][s[i + 1]];
      ++h[2][s[i + 2]];
      ++h[3][s[i + 3]];
    }
    for (; i < n; ++i) ++h[0][s[i]];
    u8 dense[256];
    u32* hd = hist_of[rbase];
    int ns = 0;
    for (int v = 0; v < 256; ++v) {
      const u32 c = h[0][v] + h[1][v] + h[2][v] + h[3][v];
      if (!c) continue;
      hd[ns] = c;
      dense[v] = static_cast<u8>(ns++);
    }
    nsym_of[rbase] = ns;
    if (ns == 256) return;  // identity map
    for (i = 0; i < n; ++i) s[i] = dense[s[i]];
  }
};

// Dense value axis for a feature in a given order: dense index for each raw
// value (only values that occur), and the raw value of each dense index.
struct DenseAxis {
  int nv = 0;
  u8 dense_of[256];  // raw value -> dense index (valid where used[v])
  bool used[256];
  u8 raw_of[256];    // dense index -> raw value, in permuted order
};

// Dense boundaries -> raw value map (values never seen join group 0).
void map_from_dense(const DenseAxis& ax, const u8* bounds, int k, u8* q) {
  std::fill(q, q + 256, u8{0});
  int g = 0;
  for (int d = 0; d < ax.nv; ++d) {
    while (g + 1 < k && d >= bounds[g]) ++g;
    q[ax.raw_of[d]] = static_cast<u8>(g);
  }
}

// Count tables of K features in one pass over the sample, each
// t[j][(g * 256 + v) * nsym + s] over the a-groups g = qa[a-value] (one
// group, g = 0, without qa). Histogramming is bound by the store-to-load
// forwarding latency whenever consecutive elements hit the same cell,
// which on a low-entropy lane is most of them; K independent tables give
// K independent chains per element. Out of line, with K <= kCountGroup:
// the loop holds 2K pointers, which is what x86-64 has registers for.
constexpr int kCountGroup = 4;

template <int K, bool kGrouped>
__attribute__((noinline)) void count_k(const u8* sym, usize n, int nsym, const u8* av,
                                       const u8* qa, const u8* const* fv_in, u32* const* t_in) {
  const u8* fv[K];
  u32* t[K];
  for (int j = 0; j < K; ++j) {
    fv[j] = fv_in[j];
    t[j] = t_in[j];
  }
  const usize gstep = static_cast<usize>(256) * nsym;
  for (usize i = 0; i < n; ++i) {
    usize s = sym[i];
    if constexpr (kGrouped) s += qa[av[i]] * gstep;
    for (int j = 0; j < K; ++j) ++t[j][static_cast<usize>(fv[j][i]) * nsym + s];
  }
}

// Counts the tables of nf features (given the a-groups of fa / qa / ng, or
// ungrouped with qa == nullptr) in groups of features whose tables
// together stay within ~512 KiB; `done(first, count)` is called after each
// group with the range of `feats` whose tables (tabs[feat]) are complete,
// so the caller can consume and release them before the next group.
template <class Done>
void count_features(const LaneStats& ls, const Feat* feats, int nf, Feat fa, const u8* qa, int ng,
                    std::vector<u32>* tabs, Done&& done) {
  const int nsym = ls.nsym;
  const usize per = static_cast<usize>(ng) * 256 * nsym;
  const usize budget =
      min_of<usize>(kCountGroup, max_of<usize>(1, (usize{512} << 10) / (per * 4)));
  const int group = nf > 0 && static_cast<usize>(nf) < budget ? nf : static_cast<int>(budget);
  const u8* av = qa ? ls.fv[fa].data() : nullptr;
  for (int f0 = 0; f0 < nf; f0 += group) {
    const int k = min_of(group, nf - f0);
    const u8* fv[kCountGroup];
    u32* t[kCountGroup];
    for (int j = 0; j < k; ++j) {
      tabs[feats[f0 + j]].assign(per, 0);
      t[j] = tabs[feats[f0 + j]].data();
      fv[j] = ls.fv[feats[f0 + j]].data();
    }
    const u8* sym = ls.sym;
    const usize n = ls.size();
    if (qa) {
      switch (k) {
        case 1: count_k<1, true>(sym, n, nsym, av, qa, fv, t); break;
        case 2: count_k<2, true>(sym, n, nsym, av, qa, fv, t); break;
        case 3: count_k<3, true>(sym, n, nsym, av, qa, fv, t); break;
        default: count_k<4, true>(sym, n, nsym, av, qa, fv, t); break;
      }
    } else {
      switch (k) {
        case 1: count_k<1, false>(sym, n, nsym, av, qa, fv, t); break;
        case 2: count_k<2, false>(sym, n, nsym, av, qa, fv, t); break;
        case 3: count_k<3, false>(sym, n, nsym, av, qa, fv, t); break;
        default: count_k<4, false>(sym, n, nsym, av, qa, fv, t); break;
      }
    }
    done(f0, k);
  }
}

// Ungrouped tables of nf features into tabs[feat].
void count_singles(const LaneStats& ls, const Feat* feats, int nf, std::vector<u32>* tabs) {
  count_features(ls, feats, nf, kNone, nullptr, 1, tabs, [](int, int) {});
}

// Full-resolution conditional entropy (bits) of the counts: a cheap proxy
// for ranking features before running the expensive grouping search.
double cond_entropy(const u32* cnt, int ng, int nsym) {
  const float* xl = xlogx().t.data();
  double bits = 0;
  for (usize r = 0; r < static_cast<usize>(ng) * 256; ++r) {
    const u32* row = cnt + r * nsym;
    u32 n = 0;
    double sx = 0;
    for (int s2 = 0; s2 < nsym; ++s2) {
      n += row[s2];
      sx += xl[row[s2]];
    }
    bits += xl[n] - sx;
  }
  return bits;
}

// L1 distance between the normalised symbol distributions of two raw values
// (summed over a-groups); 0 when either side has no data.
double row_distance(const u32* cnt, int ng, int nsym, u32 va, u32 vb) {
  double d = 0;
  for (int g = 0; g < ng; ++g) {
    const u32* ra = cnt + (static_cast<usize>(g) * 256 + va) * nsym;
    const u32* rb = cnt + (static_cast<usize>(g) * 256 + vb) * nsym;
    u32 na = 0, nb = 0;
    for (int s2 = 0; s2 < nsym; ++s2) {
      na += ra[s2];
      nb += rb[s2];
    }
    if (!na || !nb) continue;
    for (int s2 = 0; s2 < nsym; ++s2)
      d += std::fabs(ra[s2] / static_cast<double>(na) - rb[s2] / static_cast<double>(nb));
  }
  return d;
}

// The two value orders differ only in which seam is adjacent: 0x7F|0x80
// (unsigned) or 0xFF|0x00 (two's complement). Prefer the order whose seam
// joins the more similar distributions.
bool prefer_signed(const u32* cnt, int ng, int nsym) {
  return row_distance(cnt, ng, nsym, 0xFF, 0x00) < row_distance(cnt, ng, nsym, 0x7F, 0x80);
}

// Group the values of one feature axis (given the a-groups) in one order:
// total MDL cost, group count and raw-value -> group map.
double eval_axis(const u32* cnt, int ng, int nsym, bool signed_order, int kmax, int max_bins,
                 int& k, u8* q) {
  Scratch& sc = scratch();
  DenseAxis ax;
  std::fill(ax.used, ax.used + 256, false);
  for (int g = 0; g < ng; ++g)
    for (u32 v = 0; v < 256; ++v) {
      if (ax.used[v]) continue;
      const u32* row = cnt + (static_cast<usize>(g) * 256 + v) * nsym;
      for (int s2 = 0; s2 < nsym; ++s2)
        if (row[s2]) {
          ax.used[v] = true;
          break;
        }
    }
  ax.nv = 0;
  for (u32 pv = 0; pv < 256; ++pv) {
    const u32 v = perm_value(pv, signed_order);  // inverse == itself
    if (!ax.used[v]) continue;
    ax.dense_of[v] = static_cast<u8>(ax.nv);
    ax.raw_of[ax.nv] = static_cast<u8>(v);
    ++ax.nv;
  }
  if (ax.nv == 0) {
    k = 1;
    std::fill(q, q + 256, u8{0});
    return 1.0;
  }
  // Pre-binning: when the axis has more distinct values than max_bins,
  // greedily merge the adjacent (in the chosen order) pair of values whose
  // merge raises the table cost the least, until max_bins remain; the DP
  // then works on bins. Merges the DP would make anyway cost nothing, and
  // the search gets ~(nv/bins)^2 cheaper. Boundaries are refined at full
  // resolution afterwards.
  u8 bin_of[256];  // dense value -> bin
  int nb = 0;
  if (ax.nv <= max_bins) {
    for (int d = 0; d < ax.nv; ++d) bin_of[d] = static_cast<u8>(d);
    nb = ax.nv;
  } else {
    const float* xl = xlogx().t.data();
    // Per-bin histograms (all a-groups concatenated) and per-group totals.
    const usize hrow = static_cast<usize>(ng) * nsym;
    sc.binh.resize(static_cast<usize>(ax.nv) * hrow);
    u32 tot[256][kMaxLanes > 8 ? kMaxLanes : 8];
    for (int d = 0; d < ax.nv; ++d)
      for (int g = 0; g < ng; ++g) {
        u32* dst = sc.binh.data() + static_cast<usize>(d) * hrow + static_cast<usize>(g) * nsym;
        std::memcpy(dst, cnt + (static_cast<usize>(g) * 256 + ax.raw_of[d]) * nsym,
                    sizeof(u32) * static_cast<usize>(nsym));
        u32 n = 0;
        for (int s2 = 0; s2 < nsym; ++s2) n += dst[s2];
        tot[d][g] = n;
      }
    // Cost change of merging bins i and j: only symbols present in both
    // rows change the entropy term; an empty row loses its 1-bit cost.
    auto merge_delta = [&](int i, int j) {
      const u32* hi = sc.binh.data() + static_cast<usize>(i) * hrow;
      const u32* hj = sc.binh.data() + static_cast<usize>(j) * hrow;
      double delta = -kTableBits;
      for (int g = 0; g < ng; ++g) {
        const u32 ni = tot[i][g], nj = tot[j][g];
        if (ni == 0 || nj == 0) {
          delta -= (ni == 0 && nj == 0) ? 0.0 : 1.0;
          if (ni == 0 && nj == 0) delta += 1.0;  // both empty: one 1-bit row remains
          continue;
        }
        const u32* a = hi + static_cast<usize>(g) * nsym;
        const u32* b2 = hj + static_cast<usize>(g) * nsym;
        double both = 0;
        int nboth = 0;
        // xl[0] == 0, so a symbol missing on one side contributes exactly
        // 0 to the sum: no data-dependent branch.
        for (int s2 = 0; s2 < nsym; ++s2) {
          const u32 x = a[s2], y = b2[s2];
          both += xl[x + y] - xl[x] - xl[y];
          nboth += (x != 0) & (y != 0);
        }
        delta += (xl[ni + nj] - xl[ni] - xl[nj]) - both - kSymbolBits * nboth;
      }
      return delta;
    };
    int next[256], prev[256];  // linked list of live bins (by first dense value)
    double merge[256];         // cost delta of merging with next
    for (int d = 0; d < ax.nv; ++d) {
      next[d] = d + 1 < ax.nv ? d + 1 : -1;
      prev[d] = d - 1;
    }
    for (int d = 0; d < ax.nv; ++d) merge[d] = next[d] < 0 ? 1e300 : merge_delta(d, next[d]);
    int live = ax.nv;
    while (live > max_bins) {
      int bi = -1;
      double bc = 1e300;
      for (int d = 0; d >= 0; d = next[d])
        if (merge[d] < bc) {
          bc = merge[d];
          bi = d;
        }
      const int j = next[bi];
      u32* hi = sc.binh.data() + static_cast<usize>(bi) * hrow;
      const u32* hj = sc.binh.data() + static_cast<usize>(j) * hrow;
      for (usize t = 0; t < hrow; ++t) hi[t] += hj[t];
      for (int g = 0; g < ng; ++g) tot[bi][g] += tot[j][g];
      next[bi] = next[j];
      if (next[j] >= 0) prev[next[j]] = bi;
      merge[bi] = next[bi] < 0 ? 1e300 : merge_delta(bi, next[bi]);
      if (prev[bi] >= 0) merge[prev[bi]] = merge_delta(prev[bi], bi);
      --live;
    }
    for (int d = 0; d >= 0; d = next[d]) {
      const int end = next[d] < 0 ? ax.nv : next[d];
      for (int e = d; e < end; ++e) bin_of[e] = static_cast<u8>(nb);
      ++nb;
    }
  }
  const int nv = nb;
  const usize per = static_cast<usize>(nv) * nsym;
  sc.cnt2.assign(static_cast<usize>(ng) * per, 0);
  for (int g = 0; g < ng; ++g)
    for (int d = 0; d < ax.nv; ++d) {
      u32* dst = sc.cnt2.data() + static_cast<usize>(g) * per + static_cast<usize>(bin_of[d]) * nsym;
      const u32* src = cnt + (static_cast<usize>(g) * 256 + ax.raw_of[d]) * nsym;
      for (int s2 = 0; s2 < nsym; ++s2) dst[s2] += src[s2];
    }
  const usize ncost = static_cast<usize>(nv + 1) * (nv + 1);
  sc.cost.assign(ncost, 0.0);
  for (int g = 0; g < ng; ++g)
    interval_costs(sc.cnt2.data() + static_cast<usize>(g) * per, nv, nsym, sc.cost.data(), true);
  u8 bin_bounds[kMaxGroups];
  double c = segment(sc.cost.data(), nv, kmax, k, bin_bounds);
  // Bin boundaries -> dense value boundaries.
  u8 bounds[kMaxGroups];
  for (int g = 0; g + 1 < k; ++g) {
    int d = 0;
    while (d < ax.nv && bin_of[d] < bin_bounds[g]) ++d;
    bounds[g] = static_cast<u8>(d);
  }
  if (nb < ax.nv && k > 1) {
    // Refine every boundary at full value resolution within its two
    // adjacent bins, using prefix histograms over dense values.
    const float* xl = xlogx().t.data();
    const usize prow = static_cast<usize>(ax.nv + 1) * nsym;
    sc.pre.assign(static_cast<usize>(ng) * prow, 0);
    for (int g = 0; g < ng; ++g) {
      u32* pre = sc.pre.data() + static_cast<usize>(g) * prow;
      for (int d = 0; d < ax.nv; ++d) {
        const u32* row = cnt + (static_cast<usize>(g) * 256 + ax.raw_of[d]) * nsym;
        for (int s2 = 0; s2 < nsym; ++s2)
          pre[static_cast<usize>(d + 1) * nsym + s2] = pre[static_cast<usize>(d) * nsym + s2] + row[s2];
      }
    }
    u32 h[256];
    auto group_cost = [&](int lo, int hi) {  // dense values [lo, hi)
      double t = 0;
      for (int g = 0; g < ng; ++g) {
        const u32* a = sc.pre.data() + static_cast<usize>(g) * prow + static_cast<usize>(lo) * nsym;
        const u32* b2 = sc.pre.data() + static_cast<usize>(g) * prow + static_cast<usize>(hi) * nsym;
        for (int s2 = 0; s2 < nsym; ++s2) h[s2] = b2[s2] - a[s2];
        t += table_cost(h, nsym, xl);
      }
      return t;
    };
    for (int sweep = 0; sweep < 2; ++sweep) {
      for (int g = 0; g + 1 < k; ++g) {
        const int lo = g == 0 ? 0 : bounds[g - 1];
        const int hi = g + 2 < k ? bounds[g + 1] : ax.nv;
        const int cur = bounds[g];
        int best_d = cur;
        double best_c = group_cost(lo, cur) + group_cost(cur, hi);
        for (int d = lo + 1; d < hi; ++d) {
          if (d == cur || bin_of[d] + 1 < bin_of[cur] || bin_of[d] > bin_of[cur] + 1) continue;
          const double cc = group_cost(lo, d) + group_cost(d, hi);
          if (cc < best_c) {
            best_c = cc;
            best_d = d;
          }
        }
        bounds[g] = static_cast<u8>(best_d);
      }
    }
    c = kBoundaryBits * (k - 1);
    for (int g = 0; g < k; ++g)
      c += group_cost(g == 0 ? 0 : bounds[g - 1], g + 1 < k ? bounds[g] : ax.nv);
  }
  map_from_dense(ax, bounds, k, q);
  return c;
}

// Evaluate a feature axis in the order(s) the level allows.
double eval_orders(const u32* cnt, int ng, int nsym, bool both_orders, int kmax, int max_bins,
                   int& k, u8* q, bool& signed_out) {
  if (!both_orders) {
    signed_out = prefer_signed(cnt, ng, nsym);
    return eval_axis(cnt, ng, nsym, signed_out, kmax, max_bins, k, q);
  }
  double best = 1e300;
  u8 qt[256];
  for (int o = 0; o < 2; ++o) {
    int kt;
    const double c = eval_axis(cnt, ng, nsym, o == 1, kmax, max_bins, kt, qt);
    if (c < best) {
      best = c;
      k = kt;
      signed_out = o == 1;
      std::memcpy(q, qt, 256);
    }
  }
  return best;
}

struct Candidate {
  double cost = 1e300;
  LaneModel lm;
  bool signed_a = false;  // value order used for fa (needed to rebuild qa)
};

struct SearchCfg {
  int level;
  int S;
  Feat singles[kFeatCount];
  int ns = 0;
  int kmax;
  int max_bins;      // pre-binning limit for the grouping DP (256 = exact)
  int top_singles;   // single-feature candidates that get the DP (by proxy)
  bool both_orders;
  SearchCfg(int level_, int S_, bool deep = false) : level(level_), S(S_) {
    singles[ns++] = kNone;
    singles[ns++] = kC1;
    if (S > 1) singles[ns++] = kLat1;
    if (level >= 4 || deep) {
      singles[ns++] = kC2;
      if (S > 1) singles[ns++] = kLat2;
      singles[ns++] = kTrend;
      if (S > 1) singles[ns++] = kRecEq;
    }
    kmax = (level <= 3 && !deep) ? 16 : (level <= 6 && !deep) ? 32 : kMaxGroups;
    max_bins = (level <= 3 && !deep) ? 64 : (level <= 6 && !deep) ? 128 : 256;
    top_singles = (level <= 3 && !deep) ? 2 : (level <= 6 && !deep) ? 5 : 64;
    both_orders = (level >= 7) || deep;
  }
};

// Per-base proxy scan: count tables for every single feature plus the
// conditional entropy of each, so the DP only runs on promising ones.
struct BaseScan {
  Feat rbase = kNone;
  int nsym = 0;  // dense alphabet size for this base
  double none_cost = 1e300;
  std::vector<u32> cnt[kFeatCount];
  double proxy[kFeatCount];
  bool have[kFeatCount];  // cnt[f] / proxy[f] (none_cost for kNone) are for this lane
};

// Per-lane scans, one per base feature. The quick level-1 pass that
// choose_lane_model runs first shares them with the full search, so a
// (base, feature) table is counted once per lane.
struct LaneScans {
  BaseScan by_base[kFeatCount];
  void reset() {
    for (BaseScan& bs : by_base) {
      std::fill(bs.have, bs.have + kFeatCount, false);
      std::fill(bs.proxy, bs.proxy + kFeatCount, 1e300);
      bs.none_cost = 1e300;
    }
  }
};

// Count tables and proxies for the cfg.singles of one base that this lane
// does not have yet. Sets the lane's base.
void scan_base(LaneStats& ls, const SearchCfg& cfg, Feat rbase, BaseScan& bs) {
  Feat todo[kFeatCount];
  int nt = 0;
  bool need_none = false;
  for (int i = 0; i < cfg.ns; ++i) {
    const Feat f = cfg.singles[i];
    if (bs.have[f]) continue;
    if (f == kNone)
      need_none = true;
    else
      todo[nt++] = f;
  }
  if (!need_none && nt == 0) return;
  if (ls.base != rbase || ls.nsym == 0) ls.set_base(rbase);
  bs.rbase = rbase;
  const int nsym = ls.nsym;
  bs.nsym = nsym;
  if (need_none) {
    std::vector<u32>& c0 = bs.cnt[kNone];
    c0.assign(ls.hist, ls.hist + nsym);
    bs.none_cost = table_cost(c0.data(), nsym, xlogx().t.data());
    bs.have[kNone] = true;
  }
  count_singles(ls, todo, nt, bs.cnt);
  for (int i = 0; i < nt; ++i) {
    bs.proxy[todo[i]] = cond_entropy(bs.cnt[todo[i]].data(), 1, nsym);
    bs.have[todo[i]] = true;
  }
}

// Single-feature search over all bases: the exact no-context cost per base,
// then the grouping DP for the cfg.top_singles (base, feature) pairs with
// the lowest conditional entropy.
void search_singles(const SearchCfg& cfg, BaseScan* const* scans, int nbases, Candidate& best) {
  for (int b = 0; b < nbases; ++b) {
    if (scans[b]->none_cost < best.cost) {
      best.cost = scans[b]->none_cost;
      best.lm = LaneModel{};
      best.lm.rbase = scans[b]->rbase;
      best.lm.fa = kNone;
      best.lm.ka = 1;
      std::fill(best.lm.qa, best.lm.qa + 256, u8{0});
    }
  }
  constexpr int kMaxBases = 6;
  bool done[kMaxBases][kFeatCount] = {};
  u8 q[256];
  for (int t = 0; t < cfg.top_singles; ++t) {
    int bb = -1, bf = -1;
    double bp = 1e300;
    for (int b = 0; b < nbases; ++b)
      for (int f = 1; f < kFeatCount; ++f)
        if (!done[b][f] && scans[b]->proxy[f] < bp) {
          bp = scans[b]->proxy[f];
          bb = b;
          bf = f;
        }
    if (bb < 0) break;
    // Grouping the axis never lowers the entropy term below the full-
    // resolution conditional entropy, and costs at least one table: when
    // that bound cannot beat the best so far, neither can this feature
    // nor the ones after it (they come in proxy order).
    if (bp + kTableBits >= best.cost) break;
    done[bb][bf] = true;
    int k;
    bool sg = false;
    const double c = eval_orders(scans[bb]->cnt[bf].data(), 1, scans[bb]->nsym, cfg.both_orders,
                                 cfg.kmax, cfg.max_bins, k, q, sg);
    if (c < best.cost) {
      best.cost = c;
      best.lm = LaneModel{};
      best.lm.rbase = scans[bb]->rbase;
      best.lm.fa = static_cast<Feat>(bf);
      best.lm.ka = k;
      best.signed_a = sg;
      std::memcpy(best.lm.qa, q, 256);
    }
  }
}

// Pair search: the best single (capped to 8 groups) plus a second feature.
// Below level 7 the second feature is picked by conditional entropy and
// only that one gets the grouping search. `bs` is the scan of the best
// single's base (its cnt[fa] is the first axis's table).
void search_pairs(LaneStats& ls, const SearchCfg& cfg, const BaseScan& bs, Candidate& best) {
  if (best.lm.fa == kNone) return;
  Scratch& sc = scratch();
  const Feat fa = best.lm.fa;
  const Feat rbase = best.lm.rbase;
  if (ls.base != rbase || ls.nsym == 0) ls.set_base(rbase);
  const int nsym = ls.nsym;
  int ka;
  u8 qa[256];
  eval_axis(bs.cnt[fa].data(), 1, nsym, best.signed_a, 8, cfg.max_bins, ka, qa);
  Feat seconds[kFeatCount];
  int np = 0;
  for (int i = 0; i < cfg.ns; ++i)
    if (cfg.singles[i] != kNone && cfg.singles[i] != fa) seconds[np++] = cfg.singles[i];
  if (np == 0) return;
  std::vector<u32>* tabs = sc.pair_cnt;
  std::vector<u32>& cnt_pick = sc.cnt_pick;
  const int bins_b = cfg.max_bins < 64 ? cfg.max_bins : 64;
  if (!cfg.both_orders) {
    // Rank by conditional entropy, keep the best second feature only (and
    // its table).
    double bp = 1e300;
    Feat pick = seconds[0];
    count_features(ls, seconds, np, fa, qa, ka, tabs, [&](int f0, int k) {
      for (int i = f0; i < f0 + k; ++i) {
        const double h = cond_entropy(tabs[seconds[i]].data(), ka, nsym);
        if (h < bp) {
          bp = h;
          pick = seconds[i];
          cnt_pick.swap(tabs[seconds[i]]);
        }
      }
    });
    // Same bound as in search_singles: the grouped pair costs at least
    // the full-resolution conditional entropy plus a table.
    if (bp + kTableBits >= best.cost) return;
    int kb;
    u8 qb[256];
    bool sg;
    // The b axis gets at most kMaxCtxLane / ka groups; 64 bins are plenty.
    const double c = eval_orders(cnt_pick.data(), ka, nsym, false, kMaxCtxLane / ka, bins_b, kb,
                                 qb, sg);
    if (c < best.cost) {
      best.cost = c;
      best.lm = LaneModel{};
      best.lm.rbase = rbase;
      best.lm.fa = fa;
      best.lm.ka = ka;
      std::memcpy(best.lm.qa, qa, 256);
      best.lm.fb = pick;
      best.lm.kb = kb;
      std::memcpy(best.lm.qb, qb, 256);
    }
    return;
  }
  // Both orders: every second feature gets the grouping search.
  count_features(ls, seconds, np, fa, qa, ka, tabs, [&](int f0, int k) {
    for (int i = f0; i < f0 + k; ++i) {
      const u32* cnt_b = tabs[seconds[i]].data();
      if (cond_entropy(cnt_b, ka, nsym) + kTableBits >= best.cost) continue;
      int kb;
      u8 qb[256];
      bool sg;
      const double c = eval_orders(cnt_b, ka, nsym, true, kMaxCtxLane / ka, bins_b, kb, qb, sg);
      if (c < best.cost) {
        best.cost = c;
        best.lm = LaneModel{};
        best.lm.rbase = rbase;
        best.lm.fa = fa;
        best.lm.ka = ka;
        std::memcpy(best.lm.qa, qa, 256);
        best.lm.fb = seconds[i];
        best.lm.kb = kb;
        std::memcpy(best.lm.qb, qb, 256);
      }
    }
  });
}

void choose_lane_model_in(LaneStats& ls, int level, int S, LaneScans& scans, LaneModel& out) {
  // Levels 4+ add features, deeper grouping and the pair search. On a lane
  // that the level-1 search already finds near-incompressible (>= 7.5
  // bit/symbol: float mantissa bytes, hashes, noise) they cannot recover
  // anything worth their cost, and their DPs are the most expensive there
  // (dense count tables). Run the cheap search first and stop on such lanes;
  // its count tables stay in `scans` for the full search.
  const bool deep = (level >= 11) || (std::getenv("SPECTRA8_CONTEXT_ORACLE") != nullptr);
  if (level >= 4 && !deep) {
    LaneModel quick;
    choose_lane_model_in(ls, 1, S, scans, quick);
    // Only lanes the quick search left without a context qualify; their
    // cost is the no-context table cost of their base.
    if (quick.fa == kNone &&
        scans.by_base[quick.rbase].none_cost >= 7.5 * static_cast<double>(ls.raw.size())) {
      out = quick;
      return;
    }
  }
  const SearchCfg cfg(level, S, deep);
  constexpr int kMaxBases = 6;
  Feat bases[kMaxBases];
  int nb = 0;
  bases[nb++] = kNone;
  bases[nb++] = kC1;
  if ((level >= 4 || deep) && S > 1) bases[nb++] = kLat1;
  if (level >= 6 || deep) bases[nb++] = kTrend;
  if ((level >= 7 || deep) && S > 1) bases[nb++] = kLat2;
  if ((level >= 8 || deep) && S == 1) bases[nb++] = kC2;
  // Lanes that are already (nearly) free without a context - constant high
  // bytes of counters, offsets, flags - skip the search: nothing it could
  // find is worth a table. Bar: 0.02 bit/symbol plus one table.
  {
    BaseScan& bs = scans.by_base[kNone];
    if (!bs.have[kNone]) {
      ls.set_base(kNone);
      bs.rbase = kNone;
      bs.nsym = ls.nsym;
      std::vector<u32>& c0 = bs.cnt[kNone];
      c0.assign(ls.hist, ls.hist + ls.nsym);
      bs.none_cost = table_cost(c0.data(), ls.nsym, xlogx().t.data());
      bs.have[kNone] = true;
    }
    if (bs.none_cost <= 0.02 * static_cast<double>(ls.raw.size()) + kTableBits) {
      out = LaneModel{};
      return;
    }
  }
  BaseScan* bscans[kMaxBases];
  for (int b = 0; b < nb; ++b) {
    bscans[b] = &scans.by_base[bases[b]];
    scan_base(ls, cfg, bases[b], *bscans[b]);
  }
  Candidate best;
  if (level >= 7) {
    // Exhaustive: every single of every base, pairs per base.
    for (int b = 0; b < nb; ++b) {
      Candidate cand;
      search_singles(cfg, bscans + b, 1, cand);
      search_pairs(ls, cfg, *bscans[b], cand);
      if (cand.cost < best.cost) best = cand;
    }
  } else {
    search_singles(cfg, bscans, nb, best);
    if (level >= 4) search_pairs(ls, cfg, scans.by_base[best.lm.rbase], best);
  }
  out = best.lm;
}

void choose_lane_model(LaneStats& ls, int level, int S, LaneModel& out) {
  thread_local LaneScans scans;
  scans.reset();
  choose_lane_model_in(ls, level, S, scans, out);
}

// ---------------------------------------------------------------------------
// Streams.
// ---------------------------------------------------------------------------
int streams_for(usize n) noexcept {
  int k = kMaxStreams;
  while (k > 1 && n / static_cast<usize>(k) < 4096) k >>= 1;
  return k;
}

// The k streams decode in lockstep, so their output pointers keep a fixed
// distance. If that distance is a multiple of 4 KiB (n/8 for any power-of-two
// block) every stream's load and store hits the same L1 set and, on x86, the
// same low 12 address bits, which the store queue treats as a possible
// dependency (4K aliasing). Accept a stream spacing only if the k start
// addresses fall at least one cache line apart modulo 4 KiB.
bool spacing_ok(usize L, int k) noexcept {
  constexpr usize kPage = 4096;
  usize r[kMaxStreams];
  for (int i = 0; i < k; ++i) r[i] = (L * static_cast<usize>(i)) % kPage;
  for (int i = 1; i < k; ++i)
    for (int j = 0; j < i; ++j) {
      const usize d = r[i] > r[j] ? r[i] - r[j] : r[j] - r[i];
      if (min_of(d, kPage - d) < 64) return false;
    }
  return true;
}

// Equal-length streams (multiple of S), spacing shrunk by at most ~128 bytes
// so that it is not congruent to a small fraction of 4 KiB; the remainder
// goes to the last stream.
void split_streams(usize n, int stride, int k, usize* bounds) {
  const usize S = static_cast<usize>(stride);
  usize L = (n / static_cast<usize>(k)) / S * S;
  if (k > 1) {
    // Only shrink: the leftover then lengthens the last stream alone, and
    // the serial tail after the lockstep loop is one stream's excess (at
    // most ~1 KiB) instead of k-1 tails.
    for (usize d = 0; d <= L && d < 4096; d += S)
      if (spacing_ok(L - d, k)) { L -= d; break; }
  }
  bounds[0] = 0;
  for (int i = 1; i < k; ++i) bounds[i] = min_of(n, L * static_cast<usize>(i));
  bounds[k] = n;
}

// Context index of the byte at pos (lane = pos % S) within its stream.
inline u32 context_of(const Model& m, const LaneModel& lm, const u8* buf, usize pos,
                      usize start, u32 rec_eq) noexcept {
  const usize S = static_cast<usize>(m.stride);
  u32 c = lm.base;
  const u32 a = lm.qa[feat_value(lm.fa, buf, pos, start, S, rec_eq)];
  c += a * static_cast<u32>(lm.kb);
  if (lm.fb != kNone) c += lm.qb[feat_value(lm.fb, buf, pos, start, S, rec_eq)];
  return c;
}

// ---------------------------------------------------------------------------
// Header (model) serialization.
// ---------------------------------------------------------------------------
// Layout: varint(arith_len), arith-coded part, plain mantissa bits.
void write_model(const Model& m, std::vector<u8>& out) {
  std::vector<u8> arith;
  SideBitsEnc side;
  HdrEnc e;
  e.out = &arith;
  u16 map_tree[4][128];
  for (auto& t : map_tree) std::fill(t, t + 128, u16{32768});
  for (int l = 0; l < m.stride; ++l) {
    const LaneModel& lm = m.lane[l];
    e.raw(2, lm.mode);
    if (lm.mode == kLaneConst) e.raw(8, lm.value);
    if (lm.mode != kLaneCoded) continue;
    e.raw(3, lm.fa);
    e.raw(3, lm.fb);
    e.raw(3, lm.rbase);
    e.raw(6, static_cast<u32>(lm.ka - 1));
    e.raw(6, static_cast<u32>(lm.kb - 1));
    // Group maps: coded as the group of each value (mostly runs), with the
    // previous value's group as context.
    for (int which = 0; which < 2; ++which) {
      const int k = which == 0 ? lm.ka : lm.kb;
      const u8* q = which == 0 ? lm.qa : lm.qb;
      if (k <= 1) continue;
      const int nb = bit_length(static_cast<u32>(k - 1));
      u32 prev = 0;
      for (int v = 0; v < 256; ++v) {
        const int same = q[v] == prev;
        u16& ps = map_tree[which][1 + (prev & 63)];
        e.bit(ps, same);
        if (!same) e.tree(map_tree[which + 2], nb, q[v]);
        prev = q[v];
      }
    }
  }
  TableModels tm;
  const u16* ref = nullptr;
  for (u32 c = 0; c < m.nctx; ++c) {
    const u16* f = m.freq.data() + static_cast<usize>(c) * 256;
    encode_table(e, side, tm, f, ref);
    ref = f;
  }
  e.flush();
  side.flush();
  ByteWriter w(out);
  w.varint(arith.size());
  w.bytes(arith.data(), arith.size());
  w.bytes(side.bytes.data(), side.bytes.size());
}

bool read_model(Model& m, const u8* d, usize n) {
  ByteReader r(d, n);
  const usize arith_len = static_cast<usize>(r.varint());
  if (r.error || arith_len > r.remaining()) return false;
  HdrDec dec;
  dec.init(r.take(arith_len), arith_len);
  SideBitsDec side;
  side.init(d + (n - r.remaining()), r.remaining());
  u16 map_tree[4][128];
  for (auto& t : map_tree) std::fill(t, t + 128, u16{32768});
  u32 base = 0;
  for (int l = 0; l < m.stride; ++l) {
    LaneModel& lm = m.lane[l];
    lm = LaneModel{};
    const u32 mode = dec.raw(2);
    if (mode > kLaneRaw) return false;
    lm.mode = static_cast<u8>(mode);
    if (mode == kLaneConst) lm.value = static_cast<u8>(dec.raw(8));
    if (mode != kLaneCoded) continue;  // base: the sentinel row, set below
    const u32 fa = dec.raw(3), fb = dec.raw(3), rb = dec.raw(3);
    if (fa >= kFeatCount || fb >= kFeatCount || rb >= kRecEq) return false;
    lm.fa = static_cast<Feat>(fa);
    lm.fb = static_cast<Feat>(fb);
    lm.rbase = static_cast<Feat>(rb);
    lm.ka = static_cast<int>(dec.raw(6)) + 1;
    lm.kb = static_cast<int>(dec.raw(6)) + 1;
    if (lm.ka * lm.kb > kMaxCtxLane) return false;
    if (lm.fb == kNone && lm.kb != 1) return false;
    for (int which = 0; which < 2; ++which) {
      const int k = which == 0 ? lm.ka : lm.kb;
      u8* q = which == 0 ? lm.qa : lm.qb;
      std::fill(q, q + 256, u8{0});
      if (k <= 1) continue;
      const int nb = bit_length(static_cast<u32>(k - 1));
      u32 prev = 0;
      for (int v = 0; v < 256; ++v) {
        u16& ps = map_tree[which][1 + (prev & 63)];
        const int same = dec.bit(ps);
        u32 g = prev;
        if (!same) g = dec.tree(map_tree[which + 2], nb);
        if (g >= static_cast<u32>(k)) return false;
        q[v] = static_cast<u8>(g);
        prev = g;
      }
    }
    lm.base = base;
    base += static_cast<u32>(lm.nctx());
  }
  m.nctx = base;
  for (int l = 0; l < m.stride; ++l)
    if (m.lane[l].mode != kLaneCoded) m.lane[l].base = base;
  m.freq.assign(static_cast<usize>(m.nctx) * 256, 0);
  TableModels tm;
  const u16* ref = nullptr;
  for (u32 c = 0; c < m.nctx; ++c) {
    u16* f = m.freq.data() + static_cast<usize>(c) * 256;
    if (!decode_table(dec, side, tm, f, ref)) return false;
    ref = f;
  }
  return true;
}

// Bytes of lane l in a block of n (lane = position % stride).
inline usize lane_count(usize n, usize S, usize l) noexcept {
  return n > l ? (n - l + S - 1) / S : 0;
}

// The lane modes of one block. For a dynamic block they are part of the
// model; a profile block carries them per page (kCryProfileLanes), since
// the profile itself codes every lane.
struct LaneModes {
  u8 mode[kMaxLanes] = {};
  u8 value[kMaxLanes] = {};  // kLaneConst: the lane's byte
};

inline LaneModes lane_modes_of(const Model& m) noexcept {
  LaneModes lm;
  for (int l = 0; l < m.stride; ++l) {
    lm.mode[l] = m.lane[l].mode;
    lm.value[l] = m.lane[l].value;
  }
  return lm;
}

// Size of the verbatim section: the raw lanes' bytes, lane by lane.
usize raw_section_size(const LaneModes& lm, usize S, usize n) noexcept {
  usize total = 0;
  for (usize l = 0; l < S; ++l)
    if (lm.mode[l] == kLaneRaw) total += lane_count(n, S, l);
  return total;
}

bool all_coded(const LaneModes& lm, usize S) noexcept {
  for (usize l = 0; l < S; ++l)
    if (lm.mode[l] != kLaneCoded) return false;
  return true;
}

bool model_all_coded(const Model& m) noexcept {
  for (int l = 0; l < m.stride; ++l)
    if (m.lane[l].mode != kLaneCoded) return false;
  return true;
}

// 16-byte vectors for the lane interleave below (GCC/clang vector
// extensions; the zips compile to zip1/zip2 on AArch64 and punpckl/h on
// x86-64).
typedef u8 v16u8 __attribute__((vector_size(16)));

inline v16u8 v16_load(const u8* p) noexcept {
  v16u8 v;
  std::memcpy(&v, p, 16);
  return v;
}
inline void v16_store(u8* p, v16u8 v) noexcept { std::memcpy(p, &v, 16); }
inline v16u8 v16_splat(u8 x) noexcept {
  return v16u8{x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x};
}
// Interleave the low / high halves of a and b in W-byte elements.
template <int W>
inline v16u8 v16_zip_lo(v16u8 a, v16u8 b) noexcept {
#if SPECTRA8_TRANSFORM_NEON
  const uint8x16_t va = reinterpret_cast<uint8x16_t>(a);
  const uint8x16_t vb = reinterpret_cast<uint8x16_t>(b);
  if constexpr (W == 1) return reinterpret_cast<v16u8>(vzip1q_u8(va, vb));
  else if constexpr (W == 2) return reinterpret_cast<v16u8>(vzip1q_u16(reinterpret_cast<uint16x8_t>(va), reinterpret_cast<uint16x8_t>(vb)));
  else if constexpr (W == 4) return reinterpret_cast<v16u8>(vzip1q_u32(reinterpret_cast<uint32x4_t>(va), reinterpret_cast<uint32x4_t>(vb)));
  else return reinterpret_cast<v16u8>(vzip1q_u64(reinterpret_cast<uint64x2_t>(va), reinterpret_cast<uint64x2_t>(vb)));
#elif SPECTRA8_TRANSFORM_SSE2
  __m128i va, vb, vr;
  std::memcpy(&va, &a, 16);
  std::memcpy(&vb, &b, 16);
  if constexpr (W == 1) vr = _mm_unpacklo_epi8(va, vb);
  else if constexpr (W == 2) vr = _mm_unpacklo_epi16(va, vb);
  else if constexpr (W == 4) vr = _mm_unpacklo_epi32(va, vb);
  else vr = _mm_unpacklo_epi64(va, vb);
  v16u8 r;
  std::memcpy(&r, &vr, 16);
  return r;
#elif defined(__clang__)
  if constexpr (W == 1)
    return __builtin_shufflevector(a, b, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
  else if constexpr (W == 2)
    return __builtin_shufflevector(a, b, 0, 1, 16, 17, 2, 3, 18, 19, 4, 5, 20, 21, 6, 7, 22, 23);
  else if constexpr (W == 4)
    return __builtin_shufflevector(a, b, 0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23);
  else
    return __builtin_shufflevector(a, b, 0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23);
#else
  v16u8 r;
  u8* out = reinterpret_cast<u8*>(&r);
  const u8* pa = reinterpret_cast<const u8*>(&a);
  const u8* pb = reinterpret_cast<const u8*>(&b);
  for (int i = 0; i < 8 / W; ++i) {
    std::memcpy(out + (2 * i) * W, pa + i * W, W);
    std::memcpy(out + (2 * i + 1) * W, pb + i * W, W);
  }
  return r;
#endif
}
template <int W>
inline v16u8 v16_zip_hi(v16u8 a, v16u8 b) noexcept {
#if SPECTRA8_TRANSFORM_NEON
  const uint8x16_t va = reinterpret_cast<uint8x16_t>(a);
  const uint8x16_t vb = reinterpret_cast<uint8x16_t>(b);
  if constexpr (W == 1) return reinterpret_cast<v16u8>(vzip2q_u8(va, vb));
  else if constexpr (W == 2) return reinterpret_cast<v16u8>(vzip2q_u16(reinterpret_cast<uint16x8_t>(va), reinterpret_cast<uint16x8_t>(vb)));
  else if constexpr (W == 4) return reinterpret_cast<v16u8>(vzip2q_u32(reinterpret_cast<uint32x4_t>(va), reinterpret_cast<uint32x4_t>(vb)));
  else return reinterpret_cast<v16u8>(vzip2q_u64(reinterpret_cast<uint64x2_t>(va), reinterpret_cast<uint64x2_t>(vb)));
#elif SPECTRA8_TRANSFORM_SSE2
  __m128i va, vb, vr;
  std::memcpy(&va, &a, 16);
  std::memcpy(&vb, &b, 16);
  if constexpr (W == 1) vr = _mm_unpackhi_epi8(va, vb);
  else if constexpr (W == 2) vr = _mm_unpackhi_epi16(va, vb);
  else if constexpr (W == 4) vr = _mm_unpackhi_epi32(va, vb);
  else vr = _mm_unpackhi_epi64(va, vb);
  v16u8 r;
  std::memcpy(&r, &vr, 16);
  return r;
#elif defined(__clang__)
  if constexpr (W == 1)
    return __builtin_shufflevector(a, b, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
  else if constexpr (W == 2)
    return __builtin_shufflevector(a, b, 8, 9, 24, 25, 10, 11, 26, 27, 12, 13, 28, 29, 14, 15, 30, 31);
  else if constexpr (W == 4)
    return __builtin_shufflevector(a, b, 8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31);
  else
    return __builtin_shufflevector(a, b, 8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31);
#else
  v16u8 r;
  u8* out = reinterpret_cast<u8*>(&r);
  const u8* pa = reinterpret_cast<const u8*>(&a);
  const u8* pb = reinterpret_cast<const u8*>(&b);
  for (int i = 0; i < 8 / W; ++i) {
    std::memcpy(out + (2 * i) * W, pa + 8 + i * W, W);
    std::memcpy(out + (2 * i + 1) * W, pb + 8 + i * W, W);
  }
  return r;
#endif
}

// 16 records of P lanes: v[l] holds lane l of records 0..15 on entry and
// record block r (records 16r/P..) on exit, i.e. the P*16 output bytes in
// record order. log2(P) zip levels; at level k the vectors are indexed
// (record block, lane group) and each zip halves the records of a pair of
// adjacent lane groups.
//
// Every index is a compile-time constant (pair I of a level: record block
// I / (G/2), lane-group pair I % (G/2)) so the vectors stay in registers;
// with run-time loops the compiler kept the arrays on the stack.
template <int W, int G, usize... I>
inline void v16_level(const v16u8* v, v16u8* t, std::index_sequence<I...>) noexcept {
  constexpr int H = G / 2;
  ((t[(2 * (I / H)) * H + I % H] = v16_zip_lo<W>(v[(I / H) * G + 2 * (I % H)], v[(I / H) * G + 2 * (I % H) + 1]),
    t[(2 * (I / H) + 1) * H + I % H] = v16_zip_hi<W>(v[(I / H) * G + 2 * (I % H)], v[(I / H) * G + 2 * (I % H) + 1])),
   ...);
}

template <int P>
inline void v16_interleave(v16u8* v) noexcept {
  v16u8 t[P];
  constexpr auto pairs = std::make_index_sequence<P / 2>{};
  if constexpr (P >= 2) v16_level<1, P>(v, t, pairs);
  if constexpr (P >= 4) {
    v16_level<2, P / 2>(t, v, pairs);
  } else {
    for (int i = 0; i < P; ++i) v[i] = t[i];
  }
  if constexpr (P >= 8) v16_level<4, P / 4>(v, t, pairs);
  if constexpr (P == 4) {
    // result in v
  } else if constexpr (P == 8) {
    for (int i = 0; i < P; ++i) v[i] = t[i];
  } else if constexpr (P >= 16) {
    v16_level<8, P / 8>(t, v, pairs);
  }
}

// Interleaved fill of the first 16 * (nrec / 16) records: raw lanes from
// their section bytes, constant lanes from their value, coded lanes zero
// (the rANS pass overwrites them). Returns the records written.
template <int P>
usize fill_interleaved(const LaneModes& lm, const u8* raw, u8* dst, usize n) {
  const usize S = static_cast<usize>(P), nrec = n / S;  // complete records
  const u8* src[P];
  v16u8 fixed[P];
  for (int l = 0; l < P; ++l) {
    src[l] = nullptr;
    fixed[l] = v16_splat(lm.mode[l] == kLaneConst ? lm.value[l] : u8{0});
    if (lm.mode[l] == kLaneRaw) {  // section layout: lane by lane
      src[l] = raw;
      raw += lane_count(n, S, static_cast<usize>(l));
    }
  }
  usize rec = 0;
  for (; rec + 16 <= nrec; rec += 16) {
    v16u8 v[P];
    for (int l = 0; l < P; ++l) v[l] = src[l] ? v16_load(src[l] + rec) : fixed[l];
    v16_interleave<P>(v);
    u8* out = dst + rec * static_cast<usize>(P);
    for (int l = 0; l < P; ++l) v16_store(out + 16 * static_cast<usize>(l), v[l]);
  }
  return rec;
}

// Writes the constant and raw lanes of the block (the coded lanes are
// written by the rANS pass afterwards). With a raw lane and a power-of-two
// stride the records go 16 at a time through a byte interleave (the strided
// byte scatter is store-bound at ~2 stores per cycle and was 14 % of the
// decode time of a float block with 6 raw lanes of 8); otherwise constant
// lanes go as one periodic S-byte pattern over the whole block and raw
// lanes as the scatter.
void fill_skipped_lanes(const LaneModes& lm, usize S, const u8* raw, u8* dst, usize n) {
  bool any_const = false;
  int nraw = 0;
  u8 pattern[kMaxLanes] = {};
  for (usize l = 0; l < S; ++l) {
    if (lm.mode[l] == kLaneConst) {
      any_const = true;
      pattern[l] = lm.value[l];
    }
    nraw += lm.mode[l] == kLaneRaw;
  }
  usize rec = 0;  // records already complete
  // One raw lane: the scatter (one store per 8..16 output bytes) is cheaper
  // than interleaving the whole record.
  if (nraw >= 2 && n >= 16 * S) {
    switch (S) {
      case 2: rec = fill_interleaved<2>(lm, raw, dst, n); break;
      case 4: rec = fill_interleaved<4>(lm, raw, dst, n); break;
      case 8: rec = fill_interleaved<8>(lm, raw, dst, n); break;
      case 16: rec = fill_interleaved<16>(lm, raw, dst, n); break;
      default: break;
    }
  }
  const usize from = rec * S;  // first byte not yet written
  if (any_const && from < n) {
    u8* d = dst + from;
    const usize left = n - from;
    const usize head = min_of(left, S);
    std::memcpy(d, pattern, head);
    for (usize done = head; done < left; done *= 2)  // doubling copy of the period
      std::memcpy(d + done, d, min_of(done, left - done));
  }
  for (usize l = 0; l < S; ++l) {
    if (lm.mode[l] != kLaneRaw) continue;
    const usize cnt = lane_count(n, S, l);
    u8* p = dst + l;
    for (usize i = rec; i < cnt; ++i) p[i * S] = raw[i];
    raw += cnt;
  }
}

// The verbatim section of a block: the raw lanes' bytes, lane by lane.
void write_raw_section(const LaneModes& lm, usize S, const u8* src, usize n,
                       std::vector<u8>& payload) {
  for (usize l = 0; l < S; ++l) {
    if (lm.mode[l] != kLaneRaw) continue;
    const usize cnt = lane_count(n, S, l);
    const usize at = payload.size();
    payload.resize(at + cnt);
    u8* out = payload.data() + at;
    const u8* p = src + l;
    for (usize i = 0; i < cnt; ++i) out[i] = p[i * S];
  }
}

// ---------------------------------------------------------------------------
// rANS.
// ---------------------------------------------------------------------------
// Encoder symbol with a precomputed reciprocal so the encode step needs no
// division: q = x / f via Granlund-Montgomery (round-down variant, exact
// for every 32-bit x): t = (m_lo * x) >> 32, q = (((x - t) >> 1) + t) >> sh,
// where m_lo = ceil(2^(32+l) / f) - 2^32, l = ceil(log2 f), sh = l - 1.
struct EncSym {
  u32 xmax;       // renormalise when x >= xmax
  u32 m_lo;       // low 32 bits of the 33-bit reciprocal
  u32 bias;       // cum (plus kM - 1 for f == 1, see below)
  u16 cmpl_freq;  // kM - f
  u16 sh;
  void init(u32 f, u32 c) {
    xmax = f < kM ? f << (32 - kPrec) : ~0u;  // f == kM: never renormalise
    cmpl_freq = static_cast<u16>(kM - f);
    if (f < 2) {
      // t = x - 1, q = x - 1; the extra kM - 1 in the bias makes up for it.
      m_lo = ~0u;
      sh = 0;
      bias = c + kM - 1;
    } else {
      const u32 l = 32u - static_cast<u32>(__builtin_clz(f - 1));  // ceil(log2 f), f >= 2
      m_lo = static_cast<u32>(((u64{1} << (32 + l)) + f - 1) / f - (u64{1} << 32));
      sh = static_cast<u16>(l - 1);
      bias = c;
    }
  }
};

// Writes backwards into a caller-provided buffer (end pointer).
struct RansEnc {
  u32 x = kRansL;
  u8* p;  // next free byte (exclusive), moving down
  explicit RansEnc(u8* end) : p(end) {}
  // Renormalisation is branchless: the two low bytes are always stored
  // below p and p moves only when they are needed. The write below p
  // needs two bytes of slack under the final position; the stream slices
  // (2 bytes per symbol plus 8) have four under the flush.
  inline void put(const EncSym& sy) {
    const bool renorm = x >= sy.xmax;
    p[-2] = static_cast<u8>(x);
    p[-1] = static_cast<u8>(x >> 8);
    p -= renorm ? 2 : 0;
    x = renorm ? x >> 16 : x;
    const u32 t = static_cast<u32>((static_cast<u64>(x) * sy.m_lo) >> 32);
    const u32 q = (((x - t) >> 1) + t) >> sy.sh;
    x = x + sy.bias + q * sy.cmpl_freq;
  }
  u8* flush() {
    p -= 4;
    p[0] = static_cast<u8>(x);
    p[1] = static_cast<u8>(x >> 8);
    p[2] = static_cast<u8>(x >> 16);
    p[3] = static_cast<u8>(x >> 24);
    return p;
  }
};

// The decoder reads from a copy of the stream padded with kRansPad zero
// bytes, so refills need no per-symbol bounds check: a refill reads at most
// 2 bytes, the lockstep loop verifies the stream pointers every kCheckSteps
// steps, and the padding covers the bytes a corrupt stream could consume in
// between.
constexpr usize kCheckSteps = 16;
constexpr usize kRansPad = 2 * kCheckSteps + 8;

struct RansDec {
  u32 x = 0;
  const u8* p = nullptr;
  void init(const u8* d) {
    x = static_cast<u32>(d[0]) | (static_cast<u32>(d[1]) << 8) | (static_cast<u32>(d[2]) << 16) |
        (static_cast<u32>(d[3]) << 24);
    p = d + 4;
  }
  inline u32 slot() const noexcept { return x & (kM - 1); }
  inline void advance(u32 f, u32 c, u32 s) noexcept {
    x = f * (x >> kPrec) + s - c;
    // Branch-free refill: r is 0 or 1, shift by 0 or 16 bits.
    const u32 r = x < kRansL;
    const u32 m = 0u - r;
    const u32 v = static_cast<u32>(p[0] | (p[1] << 8));
    x = (((x << 16) | v) & m) | (x & ~m);
    p += 2 * r;
  }
};

// Decoder tables: slot -> symbol, and (freq | cum << 16) per symbol.
//
// The slot -> symbol map exists in two forms. Dense: one byte per slot,
// 4 KiB per context, then (freq | cum << 16) by symbol. Compact: per
// 16-slot bucket the rank (index among the present symbols) at the bucket
// start and a 16-bit mask of the slots inside the bucket where a new
// symbol begins; rank = base + popcount(mask below slot), then a rank-
// indexed table gives (freq - 1 | cum << 12 | sym << 24). 2 KiB per
// context instead of 5, same two dependent loads plus a popcount. The
// compact form is used when the dense tables would not fit the L1 cache:
// this decoder is a random-lookup loop and an L2 hit in every stream's
// dependency chain costs more than the extra ALU work.
constexpr usize kBucketBits = 4;
constexpr usize kBucketSlots = usize{1} << kBucketBits;

// x86-64: the feature selects are variable shifts, which are `shr %cl`
// (3 uops on Intel) unless BMI2's single-uop `shrx` is available, and the
// compact table uses popcnt. When the build does not already target BMI2,
// compile the lockstep loops twice and pick the BMI2+POPCNT version at run
// time; the two are the same C++.
#if defined(__x86_64__) && !defined(__BMI2__) && (defined(__GNUC__) || defined(__clang__))
#define CRY_X86_BMI2_DISPATCH 1
#define CRY_INLINE inline __attribute__((always_inline))
#else
#define CRY_X86_BMI2_DISPATCH 0
#define CRY_INLINE inline
#endif
#if defined(__GNUC__) || defined(__clang__)
#define CRY_NOINLINE __attribute__((noinline))
#else
#define CRY_NOINLINE
#endif

// Dense bytes per context (sym + fc) and the footprint above which the
// compact form is used. Measured: the dense tables win at every table size
// on both Apple silicon (128 KiB L1D) and Zen 3 (32 KiB L1D; compact was
// 15% slower there), so the budget is unlimited and the compact form is
// opt-in via SPECTRA8_CRYSTAL_TABLES=compact for tuning on other machines.
constexpr usize kDenseBytesPerCtx = (usize{1} << kPrec) + 256 * sizeof(u32);
constexpr usize kDenseBudget = ~usize{0};

bool cry_compact_tables(u32 nctx) noexcept {
#if defined(CRY_FORCE_TABLES)  // test hook: 0 = dense, 1 = compact
  (void)nctx;
  return CRY_FORCE_TABLES != 0;
#else
  static const int force = [] {
    const char* e = std::getenv("SPECTRA8_CRYSTAL_TABLES");
    if (!e) return 0;
    if (std::strcmp(e, "dense") == 0) return 1;
    if (std::strcmp(e, "compact") == 0) return 2;
    return 0;
  }();
  if (force) return force == 2;
  return static_cast<usize>(nctx) * kDenseBytesPerCtx > kDenseBudget;
#endif
}

// One table block per context, so the decode loop addresses everything of
// a context from a single base pointer plus the context's byte offset:
//   dense:   [0, 4096)     slot -> symbol (u8)
//            [4096, 5120)  by symbol: freq | cum << 16 (u32)
//   compact: [0, 1024)     per bucket: base rank | mask << 16 (u32)
//            [1024, 2048)  by rank: (freq - 1) | cum << 12 | sym << 24 (u32)
// The context index is pre-scaled to that byte offset in the lane tables
// (DecLane::qa/qb), so the loop does tab + qa[a] + qb[b] with no shifts.
// Matters most on x86-64: 16 GPRs, and the unfused form (two table bases,
// two scaled context indices) spilled the lane descriptors every byte.
constexpr u32 kDenseCtxBytes = (u32{1} << kPrec) + 256 * sizeof(u32);
constexpr u32 kDenseFcOff = u32{1} << kPrec;
constexpr u32 kCompactCtxBytes = 2 * 256 * sizeof(u32);
constexpr u32 kCompactFcrOff = 256 * sizeof(u32);
constexpr u32 ctx_bytes_for(bool compact) noexcept {
  return compact ? kCompactCtxBytes : kDenseCtxBytes;
}

struct DecTables {
  std::vector<u8> tab;        // nctx blocks of ctx_bytes_for(compact)
  std::vector<u8> padded;     // compressed streams, each followed by padding
  std::vector<u32> q_scaled;  // stride * 512 dynamic scratch
};

DecTables& dec_tables() {
  thread_local DecTables t;
  return t;
}

CRY_INLINE u32 load_u32(const u8* p) noexcept {
  u32 v;
  std::memcpy(&v, p, sizeof v);
  return v;
}
inline void store_u32(u8* p, u32 v) noexcept { std::memcpy(p, &v, sizeof v); }

void build_dec_tables(const Model& m, std::vector<u8>& tab, bool compact) {
  const usize cb = ctx_bytes_for(compact);
  tab.assign(static_cast<usize>(m.nctx) * cb, 0);
  u8 rank_of[256];
  std::vector<u8> sym_tmp(compact ? (usize{1} << kPrec) : 0);
  for (u32 c = 0; c < m.nctx; ++c) {
    const u16* f = m.freq.data() + static_cast<usize>(c) * 256;
    u8* blk = tab.data() + static_cast<usize>(c) * cb;
    u8* sym = compact ? sym_tmp.data() : blk;
    u32 cum = 0;
    u32 rank = 0;
    for (int s = 0; s < 256; ++s) {
      if (!compact) store_u32(blk + kDenseFcOff + 4u * static_cast<u32>(s), f[s] | (cum << 16));
      if (f[s]) {
        std::memset(sym + cum, s, f[s]);
        if (compact) {
          rank_of[s] = static_cast<u8>(rank);
          store_u32(blk + kCompactFcrOff + 4u * rank,
                    (f[s] - 1u) | (cum << 12) | (static_cast<u32>(s) << 24));
        }
        ++rank;
      }
      cum += f[s];
    }
    if (!compact) continue;
    for (usize b = 0; b < (usize{1} << (kPrec - kBucketBits)); ++b) {
      const u8* s = sym + b * kBucketSlots;
      u32 mask = 0;
      for (usize i = 1; i < kBucketSlots; ++i) mask |= static_cast<u32>(s[i] != s[i - 1]) << i;
      store_u32(blk + 4 * b, rank_of[s[0]] | (mask << 16));
    }
  }
}

// Per-lane decode descriptor. Feature values are packed into one u64
// (byte i = feature i, feature 0 = 0) so selecting one is a shift. The
// three shift counts share one register (bytes 0/1/2 of `sh`): a shift by
// rotr(sh, 8) or rotr(sh, 16) reads the low 6 bits only, which the ISA
// guarantees (shrx/shr %cl mask to 6 bits, aarch64 lsr likewise), so no
// masking instruction is needed. Two fewer live registers per byte on
// x86-64, where the unpacked form spilled them.
struct DecLane {
  u32 sh;         // 8*fa | 8*fb << 8 | 8*rbase << 16
  u32 mv;         // LaneMode | (kLaneConst: value << 8); 0 = coded
  const u32* q;   // [0,256): (base + group(a) * kb) * ctx_bytes; [256,512): group(b) * ctx_bytes
};

// A lane whose bytes are already in the output (constant or raw): the
// lockstep loop only shifts them into the history.
CRY_INLINE bool lane_skipped(const DecLane& L) noexcept { return (L.mv & 255u) != 0; }

inline DecLane make_dec_lane(const LaneModel& lm, const u32* q) noexcept {
  const u32 mv = lm.mode == kLaneCoded ? 0u : (static_cast<u32>(lm.mode) | (static_cast<u32>(lm.value) << 8));
  return DecLane{(8u * lm.fa) | (8u * lm.fb) << 8 | (8u * lm.rbase) << 16, mv, q};
}

CRY_INLINE u32 lane_sha(u32 sh) noexcept { return sh & 63u; }
CRY_INLINE u32 lane_shb(u32 sh) noexcept { return ((sh >> 8) | (sh << 24)) & 63u; }
CRY_INLINE u32 lane_shr(u32 sh) noexcept { return ((sh >> 16) | (sh << 16)) & 63u; }

struct DecCtx {
  const u8* tab;  // DecTables::tab
  const DecLane* lanes;
  usize S;
  const u8* dst;  // block output (stream 0 starts here); vector body only
  const u8* pad;  // DecTables::padded base; vector body only
};

// Symbol, frequency and cumulative frequency of a slot in the context whose
// table block starts at byte offset `cofs`.
struct SlotInfo {
  u32 sym, freq, cum;
};

template <bool COMPACT>
CRY_INLINE SlotInfo slot_lookup(const DecCtx& dc, u32 cofs, u32 slot) noexcept {
  const u8* blk = dc.tab + cofs;
  if constexpr (COMPACT) {
    const u32 b = load_u32(blk + 4u * (slot >> kBucketBits));
    const u32 below = (2u << (slot & (kBucketSlots - 1))) - 1;  // bits 0..(slot % 16)
    const u32 rank = (b & 255u) + static_cast<u32>(__builtin_popcount((b >> 16) & below));
    const u32 e = load_u32(blk + kCompactFcrOff + 4u * rank);
    return SlotInfo{e >> 24, (e & 0xFFFu) + 1u, (e >> 12) & 0xFFFu};
  } else {
    const u32 sym = blk[slot];
    const u32 fc = load_u32(blk + kDenseFcOff + 4u * sym);
    return SlotInfo{sym, fc & 0xFFFFu, fc >> 16};
  }
}

struct StreamState {
  RansDec d;
  const u8* limit = nullptr;  // end of the stream's real bytes
  u8* out = nullptr;          // dst + stream start
  usize len = 0;
  u32 hist = 0;       // previous bytes of this stream, c1 in the low byte
  u32 rec_eq = 1;
};

// One byte of one stream at lattice step `step` (pos = start + step).
template <bool GUARD, bool TREND, bool REQ, bool COMPACT>
CRY_INLINE void decode_one(StreamState& st, usize step, const DecLane& L,
                          const DecCtx& dc) noexcept {
  u8* p = st.out + step;
  // One negative stride register serves both lattice loads (p + ns, p + 2*ns
  // are single addressing modes on x86-64 and aarch64).
  const std::ptrdiff_t ns = -static_cast<std::ptrdiff_t>(dc.S);
  u32 lat1, lat2;
  if constexpr (GUARD) {
    lat1 = step >= dc.S ? p[ns] : 0u;
    lat2 = step >= 2 * dc.S ? p[2 * ns] : 0u;
  } else {
    lat1 = p[ns];
    lat2 = p[2 * ns];
  }
  u64 packed = (static_cast<u64>(st.hist & 0xFFFFu) << (8 * kC1)) |
               (static_cast<u64>(lat1) << (8 * kLat1)) | (static_cast<u64>(lat2) << (8 * kLat2));
  if constexpr (TREND) packed |= static_cast<u64>((2 * lat1 - lat2) & 255u) << (8 * kTrend);
  if constexpr (REQ) packed |= static_cast<u64>(st.rec_eq) << (8 * kRecEq);
  const u32 a = static_cast<u32>(packed >> lane_sha(L.sh)) & 255u;
  const u32 b = static_cast<u32>(packed >> lane_shb(L.sh)) & 255u;
  const u32 c = L.q[a] + L.q[256 + b];
  const u32 slot = st.d.slot();
  const SlotInfo si = slot_lookup<COMPACT>(dc, c, slot);
  st.d.advance(si.freq, si.cum, slot);
  const u8 byte = static_cast<u8>(si.sym + (static_cast<u32>(packed >> lane_shr(L.sh)) & 255u));
  *p = byte;
  st.hist = (st.hist << 8) | byte;
  if constexpr (REQ) st.rec_eq &= byte == lat1;
}

// One byte of one stream on a skipped lane (constant or raw): the byte is
// already in the output; shift it into the history (and the record-equal
// flag) so the following lanes' features see it.
template <bool GUARD, bool REQ>
CRY_INLINE void skip_one(StreamState& st, usize step, const DecCtx& dc) noexcept {
  u8* p = st.out + step;
  const u8 byte = *p;
  st.hist = (st.hist << 8) | byte;
  if constexpr (REQ) {
    const std::ptrdiff_t ns = -static_cast<std::ptrdiff_t>(dc.S);
    u32 lat1;
    if constexpr (GUARD)
      lat1 = step >= dc.S ? p[ns] : 0u;
    else
      lat1 = p[ns];
    st.rec_eq &= byte == lat1;
  }
}

// One lockstep step (lane L) over k streams, generic byte routine.
template <bool GUARD, bool TREND, bool REQ, bool COMPACT>
CRY_INLINE void decode_step(StreamState* st, int k, usize step, const DecLane& L,
                           const DecCtx& dc) noexcept {
  if (lane_skipped(L)) {
    for (int s = 0; s < k; ++s) skip_one<GUARD, REQ>(st[s], step, dc);
  } else {
    for (int s = 0; s < k; ++s) decode_one<GUARD, TREND, REQ, COMPACT>(st[s], step, L, dc);
  }
}

template <bool TREND, bool REQ, bool COMPACT>
CRY_INLINE bool decode_streams_body(StreamState* st_in, int k, usize shortest,
                                   const DecCtx& dc_in) {
  // Local copies: byte stores into dst could otherwise alias the state and
  // force reloads after every symbol.
  const DecCtx dc = dc_in;
  StreamState st[kMaxStreams];
  for (int s = 0; s < k; ++s) st[s] = st_in[s];
  const usize S = dc.S;
  // All streams start at lane 0 and advance in lockstep: one shared lane
  // counter. Guarded prologue (lattice history not yet available), then
  // the unguarded main part, then per-stream tails.
  u32 lane = 0;
  const usize pro = min_of(shortest, 2 * S);
  usize step = 0;
  auto in_bounds = [&]() {
    for (int s = 0; s < k; ++s)
      if (st[s].d.p > st[s].limit) return false;
    return true;
  };
  for (; step < pro; ++step) {
    decode_step<true, TREND, REQ, COMPACT>(st, k, step, dc.lanes[lane], dc);
    if (++lane == S) {
      lane = 0;
      for (int s = 0; s < k; ++s) st[s].rec_eq = 1;
    }
    if (!in_bounds()) return false;
  }
  while (step < shortest) {
    const usize chunk_end = min_of(shortest, step + kCheckSteps);
    for (; step < chunk_end; ++step) {
      decode_step<false, TREND, REQ, COMPACT>(st, k, step, dc.lanes[lane], dc);
      if (++lane == S) {
        lane = 0;
        if constexpr (REQ)
          for (int s = 0; s < k; ++s) st[s].rec_eq = 1;
      }
    }
    if (!in_bounds()) return false;
  }
  for (int s = 0; s < k; ++s) {
    u32 ln = lane;
    for (usize t2 = step; t2 < st[s].len; ++t2) {
      decode_step<true, TREND, REQ, COMPACT>(st + s, 1, t2, dc.lanes[ln], dc);
      if (++ln == S) {
        ln = 0;
        st[s].rec_eq = 1;
      }
      if (st[s].d.p > st[s].limit) return false;
    }
  }
  return true;
}

// ---------------------------------------------------------------------------
// 8-wide vector lockstep body (x86-64 AVX2). Each group of 8 interleaved rANS
// streams is the 8 lanes of one 256-bit register (16 streams = 2 groups): state x, stream offset, output
// history and record-equality flag are each one vector, and every table
// access of the scalar loop becomes one gather (lane groups x2, slot ->
// symbol, symbol -> freq/cum, refill = 5 per 8 bytes). The lattice bytes
// (S and 2S back in the same stream) come from four per-lane history
// registers holding each stream's last 16 output bytes, so for strides
// <= 8 they cost shifts, not gathers; strides > 8 add two lattice gathers.
// The lane (feature choice) is shared by all streams, so the feature
// selects are scalar branches per step, not per byte.
//
// Why: the scalar loop is instruction-bound (60-80 instructions per byte
// on x86-64 against 48 on aarch64, with 16 GPRs forcing spills), and Zen 3
// runs it at ~22 cycles/byte. Zen 3's gather unit is the vector limit
// (~19 cycles per vpgatherdd, so ~12 cycles per 8 bytes for 5 gathers);
// cores with faster gathers (Intel, Zen 4) gain more.
//
// Over-read discipline (gathers load 4 bytes): lattice bytes are fetched as
// the top byte of the dword ending at the wanted position, so the vector
// loop starts at step 2S+3 (the scalar guarded prologue covers the rest);
// table gathers stay inside the context's 5 KiB block; refill gathers stay
// inside the stream's kRansPad zero padding (checked every kCheckSteps).
//
// The vector API below maps to AVX2 intrinsics on x86-64 and, with
// CRY_V8_EMU, to a scalar emulation so the body's logic is testable on
// other hosts (select it there with SPECTRA8_CRYSTAL_X86=avx2).
// ---------------------------------------------------------------------------
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#define CRY_V8 1
#define CRY_V8_NATIVE 1
#include <immintrin.h>
#define CRY_V8_TARGET __attribute__((target("avx2,bmi2,popcnt")))
#elif defined(CRY_V8_EMU)
#define CRY_V8 1
#define CRY_V8_NATIVE 0
#define CRY_V8_TARGET
#else
#define CRY_V8 0
#endif

#if CRY_V8
#define CRY_V8_FN CRY_V8_TARGET static inline __attribute__((always_inline))

#if CRY_V8_NATIVE
using V8 = __m256i;
CRY_V8_FN V8 v_set1(u32 x) noexcept { return _mm256_set1_epi32(static_cast<int>(x)); }
CRY_V8_FN V8 v_load(const u32* p) noexcept {
  return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
}
CRY_V8_FN void v_store(u32* p, V8 v) noexcept {
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(p), v);
}
CRY_V8_FN V8 v_add(V8 a, V8 b) noexcept { return _mm256_add_epi32(a, b); }
CRY_V8_FN V8 v_sub(V8 a, V8 b) noexcept { return _mm256_sub_epi32(a, b); }
CRY_V8_FN V8 v_and(V8 a, V8 b) noexcept { return _mm256_and_si256(a, b); }
CRY_V8_FN V8 v_or(V8 a, V8 b) noexcept { return _mm256_or_si256(a, b); }
CRY_V8_FN V8 v_mullo(V8 a, V8 b) noexcept { return _mm256_mullo_epi32(a, b); }
CRY_V8_FN V8 v_cmpeq(V8 a, V8 b) noexcept { return _mm256_cmpeq_epi32(a, b); }
CRY_V8_FN V8 v_cmpgt(V8 a, V8 b) noexcept { return _mm256_cmpgt_epi32(a, b); }  // signed
CRY_V8_FN V8 v_blend(V8 a, V8 b, V8 mask) noexcept { return _mm256_blendv_epi8(a, b, mask); }
CRY_V8_FN bool v_any(V8 mask) noexcept {
  return _mm256_movemask_ps(_mm256_castsi256_ps(mask)) != 0;
}
template <int N>
CRY_V8_FN V8 v_srl(V8 a) noexcept { return _mm256_srli_epi32(a, N); }
template <int N>
CRY_V8_FN V8 v_sll(V8 a) noexcept { return _mm256_slli_epi32(a, N); }
CRY_V8_FN V8 v_srl_n(V8 a, u32 n) noexcept {
  return _mm256_srl_epi32(a, _mm_cvtsi32_si128(static_cast<int>(n)));
}
// 8 dwords at base + idx[i] * SC (4 bytes each, little-endian).
template <int SC>
CRY_V8_FN V8 v_gather(const void* base, V8 idx) noexcept {
  return _mm256_i32gather_epi32(static_cast<const int*>(base), idx, SC);
}
// Low byte of each lane, packed into one u64 (lane i -> byte i).
CRY_V8_FN u64 v_pack_bytes(V8 v) noexcept {
  const __m128i lo = _mm256_castsi256_si128(v);
  const __m128i hi = _mm256_extracti128_si256(v, 1);
  const __m128i w = _mm_packus_epi32(lo, hi);  // 8 x u16 (values < 256)
  const __m128i b = _mm_packus_epi16(w, w);    // 16 x u8, low 8 wanted
  return static_cast<u64>(_mm_cvtsi128_si64(b));
}
#else
struct V8 {
  u32 v[8];
};
CRY_V8_FN V8 v_set1(u32 x) noexcept {
  V8 r;
  for (u32& e : r.v) e = x;
  return r;
}
CRY_V8_FN V8 v_load(const u32* p) noexcept {
  V8 r;
  std::memcpy(r.v, p, sizeof r.v);
  return r;
}
CRY_V8_FN void v_store(u32* p, V8 v) noexcept { std::memcpy(p, v.v, sizeof v.v); }
#define CRY_V8_BINOP(name, expr)                    \
  CRY_V8_FN V8 name(V8 a, V8 b) noexcept {          \
    V8 r;                                           \
    for (int i = 0; i < 8; ++i) r.v[i] = (expr);    \
    return r;                                       \
  }
CRY_V8_BINOP(v_add, a.v[i] + b.v[i])
CRY_V8_BINOP(v_sub, a.v[i] - b.v[i])
CRY_V8_BINOP(v_and, a.v[i] & b.v[i])
CRY_V8_BINOP(v_or, a.v[i] | b.v[i])
CRY_V8_BINOP(v_mullo, a.v[i] * b.v[i])
CRY_V8_BINOP(v_cmpeq, a.v[i] == b.v[i] ? ~0u : 0u)
CRY_V8_BINOP(v_cmpgt, static_cast<i32>(a.v[i]) > static_cast<i32>(b.v[i]) ? ~0u : 0u)
#undef CRY_V8_BINOP
CRY_V8_FN V8 v_blend(V8 a, V8 b, V8 mask) noexcept {
  V8 r;
  for (int i = 0; i < 8; ++i) r.v[i] = (mask.v[i] & 0x80000000u) ? b.v[i] : a.v[i];
  return r;
}
CRY_V8_FN bool v_any(V8 mask) noexcept {
  u32 acc = 0;
  for (u32 e : mask.v) acc |= e & 0x80000000u;
  return acc != 0;
}
template <int N>
CRY_V8_FN V8 v_srl(V8 a) noexcept {
  for (u32& e : a.v) e >>= N;
  return a;
}
template <int N>
CRY_V8_FN V8 v_sll(V8 a) noexcept {
  for (u32& e : a.v) e <<= N;
  return a;
}
CRY_V8_FN V8 v_srl_n(V8 a, u32 n) noexcept {
  for (u32& e : a.v) e = n >= 32 ? 0u : e >> n;
  return a;
}
template <int SC>
CRY_V8_FN V8 v_gather(const void* base, V8 idx) noexcept {
  V8 r;
  for (int i = 0; i < 8; ++i)
    r.v[i] = load_u32(static_cast<const u8*>(base) +
                      static_cast<std::ptrdiff_t>(static_cast<i32>(idx.v[i])) * SC);
  return r;
}
CRY_V8_FN u64 v_pack_bytes(V8 v) noexcept {
  u64 r = 0;
  for (int i = 0; i < 8; ++i) r |= static_cast<u64>(v.v[i] & 255u) << (8 * i);
  return r;
}
#endif  // CRY_V8_NATIVE

// Feature `f` (Feat) of the current lane as a vector over the 8 streams.
template <bool TREND, bool REQ>
CRY_V8_FN V8 v_feature(u32 f, V8 hist, V8 lat1, V8 lat2, V8 req) noexcept {
  const V8 m255 = v_set1(255u);
  switch (f) {
    case kC1: return v_and(hist, m255);
    case kC2: return v_and(v_srl<8>(hist), m255);
    case kLat1: return lat1;
    case kLat2: return lat2;
    case kTrend:
      if constexpr (TREND) return v_and(v_sub(v_sll<1>(lat1), lat2), m255);
      return v_set1(0u);
    case kRecEq:
      if constexpr (REQ) return req;
      return v_set1(0u);
    default: return v_set1(0u);
  }
}

// Per-group vector state. h[0..3] hold each stream's previous 16 output
// bytes (h[0] low byte = last byte, h[0] >> 8 = the one before, h[1] low byte
// = 5 bytes back, ...): C1/C2 come from h[0], and for strides <= 8 the
// lattice bytes S and 2S back come from h[] too, so the lattice costs no
// gathers. h[] for strides > 8 (rare) fall back to lattice gathers.
constexpr int kV8Hist = 4;
struct V8Group {
  V8 x, p, out, req;
  V8 h[kV8Hist];
};

// Loop-invariant vectors and scalars of the step.
struct V8Consts {
  V8 m255, m4095, mffff, zero, two, one, off1, off2;
  u32 nh;              // history registers initialised (ceil(2S/4), <= 4)
  u32 i1, s1, i2, s2;  // lattice bytes: h[i1] >> s1, h[i2] >> s2 (LATH)
};

// One lockstep step (one lane) for one group of 8 streams: decodes a byte
// per stream, returns the 8 bytes packed (stream i -> byte i).
template <bool TREND, bool REQ, bool LATH>
CRY_V8_FN u64 v8_step(V8Group& g, u32 fa, u32 fb, u32 fr, const u32* q, usize step,
                      const DecCtx& dc, const V8Consts& K) noexcept {
  V8 lat1, lat2;
  if constexpr (LATH) {
    lat1 = v_and(v_srl_n(g.h[K.i1], K.s1), K.m255);
    lat2 = v_and(v_srl_n(g.h[K.i2], K.s2), K.m255);
  } else {
    const V8 pos = v_add(g.out, v_set1(static_cast<u32>(step)));
    lat1 = v_srl<24>(v_gather<1>(dc.dst, v_sub(pos, K.off1)));
    lat2 = v_srl<24>(v_gather<1>(dc.dst, v_sub(pos, K.off2)));
  }
  const V8 hist = g.h[0];
  const V8 a = v_feature<TREND, REQ>(fa, hist, lat1, lat2, g.req);
  const V8 b = v_feature<TREND, REQ>(fb, hist, lat1, lat2, g.req);
  const V8 c = v_add(v_gather<4>(q, a), v_gather<4>(q + 256, b));
  const V8 slot = v_and(g.x, K.m4095);
  const V8 sym = v_and(v_gather<1>(dc.tab, v_add(c, slot)), K.m255);
  const V8 fc = v_gather<1>(dc.tab + kDenseFcOff, v_add(c, v_sll<2>(sym)));
  // x = freq * (x >> 12) + slot - cum
  V8 x = v_add(v_sub(v_mullo(v_and(fc, K.mffff), v_srl<12>(g.x)), v_srl<16>(fc)), slot);
  // Refill where x < 2^16: shift in the next 2 stream bytes.
  const V8 need = v_cmpeq(v_srl<16>(x), K.zero);
  const V8 fill = v_or(v_sll<16>(x), v_and(v_gather<1>(dc.pad, g.p), K.mffff));
  g.x = v_blend(x, fill, need);
  g.p = v_add(g.p, v_and(need, K.two));
  const V8 r = v_feature<TREND, REQ>(fr, hist, lat1, lat2, g.req);
  const V8 byte = v_and(v_add(sym, r), K.m255);
  // Shift the byte into the history chain (all registers, unconditionally:
  // keeps h[] in registers and costs 6 cheap ops off the gather chain).
  if constexpr (LATH) {
    g.h[3] = v_or(v_sll<8>(g.h[3]), v_srl<24>(g.h[2]));
    g.h[2] = v_or(v_sll<8>(g.h[2]), v_srl<24>(g.h[1]));
    g.h[1] = v_or(v_sll<8>(g.h[1]), v_srl<24>(hist));
  }
  g.h[0] = v_or(v_sll<8>(hist), byte);
  if constexpr (REQ) g.req = v_and(g.req, v_and(v_cmpeq(byte, lat1), K.one));
  return v_pack_bytes(byte);
}

// One lockstep step on a skipped lane (bytes already in the output): no
// table work, only the history chain and the record-equal flag. A raw
// lane's bytes come from one gather (the dword ending at the position, as
// the lattice gathers do; the main loop starts past step 3), a constant
// lane's from the descriptor.
template <bool REQ, bool LATH>
CRY_V8_FN void v8_skip_step(V8Group& g, const DecLane& L, usize step, const DecCtx& dc,
                            const V8Consts& K) noexcept {
  V8 byte;
  if ((L.mv & 255u) == kLaneConst) {
    byte = v_set1(L.mv >> 8);
  } else {
    const V8 pos = v_add(g.out, v_set1(static_cast<u32>(step)));
    byte = v_srl<24>(v_gather<1>(dc.dst, v_sub(pos, v_set1(3u))));
  }
  const V8 hist = g.h[0];
  if constexpr (REQ) {
    V8 lat1;
    if constexpr (LATH) {
      lat1 = v_and(v_srl_n(g.h[K.i1], K.s1), K.m255);
    } else {
      const V8 pos = v_add(g.out, v_set1(static_cast<u32>(step)));
      lat1 = v_srl<24>(v_gather<1>(dc.dst, v_sub(pos, K.off1)));
    }
    g.req = v_and(g.req, v_and(v_cmpeq(byte, lat1), K.one));
  }
  if constexpr (LATH) {
    g.h[3] = v_or(v_sll<8>(g.h[3]), v_srl<24>(g.h[2]));
    g.h[2] = v_or(v_sll<8>(g.h[2]), v_srl<24>(g.h[1]));
    g.h[1] = v_or(v_sll<8>(g.h[1]), v_srl<24>(hist));
  }
  g.h[0] = v_or(v_sll<8>(hist), byte);
}

// Dense tables, exactly G * kV8Width streams. Same contract as
// decode_streams_body: false on a stream running past its bytes. LATH:
// stride <= 8, lattice bytes from the history registers (5 gathers per 8
// bytes instead of 7). The G groups are independent dependency chains
// issued back to back so their gathers overlap; on Zen 3 the gather unit
// itself is the limit (~19 cycles per vpgatherdd), so the gather count is
// what sets the speed.
template <bool TREND, bool REQ, bool LATH, int G>
CRY_V8_TARGET CRY_NOINLINE bool decode_streams_v8(StreamState* st_in, usize shortest,
                                                  const DecCtx& dc_in) {
  constexpr int k = G * kV8Width;
  const DecCtx dc = dc_in;
  StreamState st[k];
  for (int s = 0; s < k; ++s) st[s] = st_in[s];
  const usize S = dc.S;
  u32 lane = 0;
  usize step = 0;
  // Scalar guarded prologue: lattice history (LATH reads 4*ceil(2S/4) <=
  // 2S+3 bytes back) and the 3-byte dword margin of the lattice gathers.
  const usize pro = min_of(shortest, 2 * S + 3);
  for (; step < pro; ++step) {
    decode_step<true, TREND, REQ, false>(st, k, step, dc.lanes[lane], dc);
    if (++lane == S) {
      lane = 0;
      for (int s = 0; s < k; ++s) st[s].rec_eq = 1;
    }
    for (int s = 0; s < k; ++s)
      if (st[s].d.p > st[s].limit) return false;
  }

  // Vector state: offsets are relative to dc.dst / dc.pad (both < 2^31).
  // History register r of stream s = its output bytes step-1-4r .. step-4-4r
  // (low byte = most recent); the prologue wrote at least 2S+3 >= 4*nh.
  const u32 nh = LATH ? static_cast<u32>((2 * S + 3) / 4) : 1u;
  alignas(32) u32 xs[k], ps[k], outs[k], hists[kV8Hist][k], reqs[k], lims[k];
  u8* outp[k];
  for (int s = 0; s < k; ++s) {
    xs[s] = st[s].d.x;
    ps[s] = static_cast<u32>(st[s].d.p - dc.pad);
    outs[s] = static_cast<u32>(st[s].out - dc.dst);
    reqs[s] = st[s].rec_eq;
    lims[s] = static_cast<u32>(st[s].limit - dc.pad);
    outp[s] = st[s].out;
    for (u32 r = 0; r < kV8Hist; ++r) {
      u32 h = 0;
      if (r < nh)
        for (u32 j = 0; j < 4; ++j) h |= static_cast<u32>(outp[s][step - 1 - 4 * r - j]) << (8 * j);
      hists[r][s] = h;
    }
  }
  V8Group grp[G];
  V8 vlim[G];
  for (int g = 0; g < G; ++g) {
    const int o = g * kV8Width;
    grp[g].x = v_load(xs + o);
    grp[g].p = v_load(ps + o);
    grp[g].out = v_load(outs + o);
    grp[g].req = v_load(reqs + o);
    for (int r = 0; r < kV8Hist; ++r) grp[g].h[r] = v_load(hists[r] + o);
    vlim[g] = v_load(lims + o);
  }
  V8Consts K{v_set1(255u),  v_set1(4095u), v_set1(0xFFFFu),
             v_set1(0u),    v_set1(2u),    v_set1(1u),
             v_set1(static_cast<u32>(S + 3)), v_set1(static_cast<u32>(2 * S + 3)),
             nh, 0, 0, 0, 0};
  if (LATH) {
    // byte d back (d >= 1) sits in h[(d-1)/4] at bit 8*((d-1)%4)
    K.i1 = static_cast<u32>((S - 1) / 4);
    K.s1 = static_cast<u32>(8 * ((S - 1) % 4));
    K.i2 = static_cast<u32>((2 * S - 1) / 4);
    K.s2 = static_cast<u32>(8 * ((2 * S - 1) % 4));
  }

  while (step < shortest) {
    const usize chunk_end = min_of(shortest, step + kCheckSteps);
    for (; step < chunk_end; ++step) {
      const DecLane L = dc.lanes[lane];
      if (lane_skipped(L)) {
        for (int g = 0; g < G; ++g) v8_skip_step<REQ, LATH>(grp[g], L, step, dc, K);
      } else {
        const u32 fa = lane_sha(L.sh) >> 3, fb = lane_shb(L.sh) >> 3, fr = lane_shr(L.sh) >> 3;
        u64 bytes[G];
        for (int g = 0; g < G; ++g)
          bytes[g] = v8_step<TREND, REQ, LATH>(grp[g], fa, fb, fr, L.q, step, dc, K);
        for (int g = 0; g < G; ++g)
          for (int i = 0; i < kV8Width; ++i)
            outp[g * kV8Width + i][step] = static_cast<u8>(bytes[g] >> (8 * i));
      }
      if (++lane == S) {
        lane = 0;
        if constexpr (REQ)
          for (int g = 0; g < G; ++g) grp[g].req = K.one;
      }
    }
    for (int g = 0; g < G; ++g)
      if (v_any(v_cmpgt(grp[g].p, vlim[g]))) return false;
  }

  // Back to scalar state for the per-stream tails.
  for (int g = 0; g < G; ++g) {
    const int o = g * kV8Width;
    v_store(xs + o, grp[g].x);
    v_store(ps + o, grp[g].p);
    v_store(hists[0] + o, grp[g].h[0]);
    v_store(reqs + o, grp[g].req);
  }
  for (int s = 0; s < k; ++s) {
    st[s].d.x = xs[s];
    st[s].d.p = dc.pad + ps[s];
    st[s].hist = hists[0][s];
    st[s].rec_eq = reqs[s];
  }
  for (int s = 0; s < k; ++s) {
    u32 ln = lane;
    for (usize t2 = step; t2 < st[s].len; ++t2) {
      decode_step<true, TREND, REQ, false>(st + s, 1, t2, dc.lanes[ln], dc);
      if (++ln == S) {
        ln = 0;
        st[s].rec_eq = 1;
      }
      if (st[s].d.p > st[s].limit) return false;
    }
  }
  return true;
}

// k must be kV8Width or 2 * kV8Width (checked by the caller).
bool decode_streams_v8(StreamState* st, int k, usize shortest, const DecCtx& dc, bool trend,
                       bool req) {
  const bool lath = dc.S <= 2 * kV8Hist;  // 16 history bytes cover 2S
  const int sel = (k == 2 * kV8Width ? 8 : 0) | (trend ? 4 : 0) | (req ? 2 : 0) | (lath ? 1 : 0);
  switch (sel) {
    case 0: return decode_streams_v8<false, false, false, 1>(st, shortest, dc);
    case 1: return decode_streams_v8<false, false, true, 1>(st, shortest, dc);
    case 2: return decode_streams_v8<false, true, false, 1>(st, shortest, dc);
    case 3: return decode_streams_v8<false, true, true, 1>(st, shortest, dc);
    case 4: return decode_streams_v8<true, false, false, 1>(st, shortest, dc);
    case 5: return decode_streams_v8<true, false, true, 1>(st, shortest, dc);
    case 6: return decode_streams_v8<true, true, false, 1>(st, shortest, dc);
    case 7: return decode_streams_v8<true, true, true, 1>(st, shortest, dc);
    case 8: return decode_streams_v8<false, false, false, 2>(st, shortest, dc);
    case 9: return decode_streams_v8<false, false, true, 2>(st, shortest, dc);
    case 10: return decode_streams_v8<false, true, false, 2>(st, shortest, dc);
    case 11: return decode_streams_v8<false, true, true, 2>(st, shortest, dc);
    case 12: return decode_streams_v8<true, false, false, 2>(st, shortest, dc);
    case 13: return decode_streams_v8<true, false, true, 2>(st, shortest, dc);
    case 14: return decode_streams_v8<true, true, false, 2>(st, shortest, dc);
    default: return decode_streams_v8<true, true, true, 2>(st, shortest, dc);
  }
}

// Whether to use the vector body: x86-64 with AVX2 at run time (or the
// emulation when built with CRY_V8_EMU), SPECTRA8_CRYSTAL_X86=scalar disables,
// =avx2 forces (emulation included).
bool use_v8() noexcept {
  static const int mode = [] {
    const char* e = std::getenv("SPECTRA8_CRYSTAL_X86");
    if (e && std::strcmp(e, "scalar") == 0) return 0;
    if (e && std::strcmp(e, "avx2") == 0) return 1;
#if CRY_V8_NATIVE
    return __builtin_cpu_supports("avx2") ? 1 : 0;
#else
    return 0;
#endif
  }();
  return mode == 1;
}
#endif  // CRY_V8

// Run-time ISA dispatch wrapper: BODY is a template<...> functor type whose
// static run() holds the loop; it is compiled once with the baseline ISA
// and, on x86-64 builds without BMI2, once more with target("bmi2,popcnt").
#if CRY_X86_BMI2_DISPATCH
bool x86_has_bmi2() {
  static const bool has = __builtin_cpu_supports("bmi2") && __builtin_cpu_supports("popcnt");
  return has;
}

template <class BODY>
__attribute__((target("bmi2,popcnt"), noinline)) bool run_bmi2(StreamState* st, int k,
                                                                usize shortest,
                                                                const DecCtx& dc) {
  return BODY::run(st, k, shortest, dc);
}
#endif

template <class BODY>
CRY_NOINLINE bool run_baseline(StreamState* st, int k, usize shortest, const DecCtx& dc) {
  return BODY::run(st, k, shortest, dc);
}

template <class BODY>
bool run_dispatch(StreamState* st, int k, usize shortest, const DecCtx& dc) {
#if CRY_X86_BMI2_DISPATCH
  if (x86_has_bmi2()) return run_bmi2<BODY>(st, k, shortest, dc);
#endif
  return run_baseline<BODY>(st, k, shortest, dc);
}

template <bool TREND, bool REQ, bool COMPACT>
struct GenericBody {
  CRY_INLINE static bool run(StreamState* st, int k, usize shortest, const DecCtx& dc) {
    return decode_streams_body<TREND, REQ, COMPACT>(st, k, shortest, dc);
  }
};

// ---------------------------------------------------------------------------
// Lane-specialised lockstep body (stride > 1). decode_one selects the three
// features of the current lane at run time: it packs every feature into one
// u64 (3 inserts, 2 more with Trend/RecEq) and extracts three with variable
// shifts and masks, about 12 of its ~48 instructions per byte on aarch64,
// and on x86-64 the packed word and the shift counts are 2 more live
// registers in a loop that already spills. The lane is the same for all k
// streams of a step, so the step is dispatched once on the lane's (fa, fb,
// rbase) to a byte routine with the selects resolved at compile time: an
// order-0 lane (the most common one on NanoAOD: constant context) then
// costs no feature work at all, a C1 lane one mask, and the lattice bytes
// are loaded only when a feature reads them. The switch costs one indirect
// branch per step (per k bytes), on a pattern periodic in S.
//
// Covered: fa, fb any feature; rbase None, C1 or Lat1 (what the trainer
// produces on columnar data); no RecEq lane (RecEq needs the record
// equality flag maintained on every lane). Other blocks take the generic
// body. 7 * 7 * 3 = 147 byte routines per COMPACT (~60 KiB of text on
// aarch64; only the S routines of a block's lanes are hot).
//
// Measured on NanoAOD pages (CMS OpenData, Apple M-series, pure page
// decode): 515 -> 770 MB/s dynamic, 500 -> 684 MB/s learned profiles.
// Per byte, dense tables: aarch64 40 -> 29..36 instructions, x86-64 57
// (53 BMI2) -> 34..40 with the stack references down from 3 to 0..2. The
// stream loop is deliberately not unrolled: a 4x unroll measured within
// noise on aarch64 and quintuples the text.
// ---------------------------------------------------------------------------
template <int F>
constexpr bool lane_reads_lat1 = F == kLat1 || F == kTrend;
template <int F>
constexpr bool lane_reads_lat2 = F == kLat2 || F == kTrend;

template <int F>
CRY_INLINE u32 lane_feat(u32 hist, u32 lat1, u32 lat2) noexcept {
  if constexpr (F == kC1) return hist & 255u;
  else if constexpr (F == kC2) return (hist >> 8) & 255u;
  else if constexpr (F == kLat1) return lat1;
  else if constexpr (F == kLat2) return lat2;
  else if constexpr (F == kTrend) return (2u * lat1 - lat2) & 255u;
  else return 0u;
}

template <int FA, int FB, int RB, bool COMPACT>
CRY_INLINE void decode_one_lane(StreamState& st, usize step, const DecLane& L,
                               const DecCtx& dc) noexcept {
  u8* p = st.out + step;
  const std::ptrdiff_t ns = -static_cast<std::ptrdiff_t>(dc.S);
  u32 lat1 = 0, lat2 = 0;
  if constexpr (lane_reads_lat1<FA> || lane_reads_lat1<FB> || lane_reads_lat1<RB>) lat1 = p[ns];
  if constexpr (lane_reads_lat2<FA> || lane_reads_lat2<FB> || lane_reads_lat2<RB>) lat2 = p[2 * ns];
  const u32 hist = st.hist;
  const u32 c = L.q[lane_feat<FA>(hist, lat1, lat2)] + L.q[256 + lane_feat<FB>(hist, lat1, lat2)];
  const u32 slot = st.d.slot();
  const SlotInfo si = slot_lookup<COMPACT>(dc, c, slot);
  st.d.advance(si.freq, si.cum, slot);
  const u8 byte = static_cast<u8>(si.sym + lane_feat<RB>(hist, lat1, lat2));
  *p = byte;
  st.hist = (hist << 8) | byte;
}

// rbase kinds covered by the lane body, in dispatch order.
constexpr int kLaneRb[3] = {kNone, kC1, kLat1};
constexpr int lane_rb_index(u32 rb) noexcept { return rb == kNone ? 0 : rb == kC1 ? 1 : rb == kLat1 ? 2 : -1; }
constexpr int lane_code(u32 fa, u32 fb, int rbi) noexcept {
  return (static_cast<int>(fa) * kFeatCount + static_cast<int>(fb)) * 3 + rbi;
}

constexpr int kLaneSkipCode = -1;  // skipped lane (constant or raw)

// True if every lane of the block is covered by the lane body.
inline bool lane_body_covers(const DecLane* lanes, usize S, bool req) noexcept {
  if (req || S < 2 || S > kMaxLanes) return false;
  for (usize l = 0; l < S; ++l)
    if (!lane_skipped(lanes[l]) && lane_rb_index(lane_shr(lanes[l].sh) / 8) < 0) return false;
  return true;
}

template <bool COMPACT>
struct LaneBody {
  CRY_INLINE static bool run(StreamState* st_in, int k, usize shortest, const DecCtx& dc_in) {
    const DecCtx dc = dc_in;
    StreamState st[kMaxStreams];
    for (int s = 0; s < k; ++s) st[s] = st_in[s];
    const usize S = dc.S;
    int code[kMaxLanes];
    for (usize l = 0; l < S; ++l) {
      const u32 sh = dc.lanes[l].sh;
      code[l] = lane_skipped(dc.lanes[l])
                    ? kLaneSkipCode
                    : lane_code(lane_sha(sh) / 8, lane_shb(sh) / 8, lane_rb_index(lane_shr(sh) / 8));
    }
    u32 lane = 0;
    usize step = 0;
    auto in_bounds = [&]() {
      for (int s = 0; s < k; ++s)
        if (st[s].d.p > st[s].limit) return false;
      return true;
    };
    // Guarded prologue (lattice history not yet available): generic step.
    // RecEq is excluded, Trend is packed unconditionally (2S steps).
    const usize pro = min_of(shortest, 2 * S);
    for (; step < pro; ++step) {
      decode_step<true, true, false, COMPACT>(st, k, step, dc.lanes[lane], dc);
      if (++lane == S) lane = 0;
      if (!in_bounds()) return false;
    }
    while (step < shortest) {
      const usize chunk_end = min_of(shortest, step + kCheckSteps);
      for (; step < chunk_end; ++step) {
        const DecLane L = dc.lanes[lane];
        switch (code[lane]) {
          case kLaneSkipCode:
            for (int s = 0; s < k; ++s) skip_one<false, false>(st[s], step, dc);
            break;
#define CRY_LANE_CASE(A, B, R)                                                      \
  case lane_code(A, B, R):                                                          \
    for (int s = 0; s < k; ++s)                                                     \
      decode_one_lane<A, B, kLaneRb[R], COMPACT>(st[s], step, L, dc);               \
    break;
#define CRY_LANE_RB(A, B) CRY_LANE_CASE(A, B, 0) CRY_LANE_CASE(A, B, 1) CRY_LANE_CASE(A, B, 2)
#define CRY_LANE_FB(A)                                                              \
  CRY_LANE_RB(A, kNone) CRY_LANE_RB(A, kC1) CRY_LANE_RB(A, kC2) CRY_LANE_RB(A, kLat1) \
  CRY_LANE_RB(A, kLat2) CRY_LANE_RB(A, kTrend) CRY_LANE_RB(A, kRecEq)
          CRY_LANE_FB(kNone) CRY_LANE_FB(kC1) CRY_LANE_FB(kC2) CRY_LANE_FB(kLat1)
          CRY_LANE_FB(kLat2) CRY_LANE_FB(kTrend) CRY_LANE_FB(kRecEq)
#undef CRY_LANE_FB
#undef CRY_LANE_RB
#undef CRY_LANE_CASE
          default: __builtin_unreachable();
        }
        if (++lane == S) lane = 0;
      }
      if (!in_bounds()) return false;
    }
    // Per-stream tails: generic guarded step.
    for (int s = 0; s < k; ++s) {
      u32 ln = lane;
      for (usize t2 = step; t2 < st[s].len; ++t2) {
        decode_step<true, true, false, COMPACT>(st + s, 1, t2, dc.lanes[ln], dc);
        if (++ln == S) ln = 0;
        if (st[s].d.p > st[s].limit) return false;
      }
    }
    return true;
  }
};

template <bool COMPACT>
bool decode_streams(StreamState* st, int k, usize shortest, const DecCtx& dc, bool trend,
                    bool req) {
  if (lane_body_covers(dc.lanes, dc.S, req)) return run_dispatch<LaneBody<COMPACT>>(st, k, shortest, dc);
  switch ((trend ? 2 : 0) | (req ? 1 : 0)) {
    case 0: return run_dispatch<GenericBody<false, false, COMPACT>>(st, k, shortest, dc);
    case 1: return run_dispatch<GenericBody<false, true, COMPACT>>(st, k, shortest, dc);
    case 2: return run_dispatch<GenericBody<true, false, COMPACT>>(st, k, shortest, dc);
    default: return run_dispatch<GenericBody<true, true, COMPACT>>(st, k, shortest, dc);
  }
}

// ---------------------------------------------------------------------------
// Stride 1 (byte planes after a filter, i.e. most blocks). One lane, and the
// lattice features coincide with the byte history: Lat1 = C1, Lat2 = C2,
// RecEq is constant. Every feature is then an ALU op on the history
// register: no lattice loads, no packed word, no variable shifts, no lane
// counter. Specialised on the lane's (fa, fb, rbase) so the selects are
// resolved at compile time.
// ---------------------------------------------------------------------------
enum : int { kS1None = 0, kS1C1 = 1, kS1C2 = 2, kS1Trend = 3 };

template <int F>
CRY_INLINE u32 s1_feat(u32 hist) noexcept {
  if constexpr (F == kS1C1) return hist & 255u;
  if constexpr (F == kS1C2) return (hist >> 8) & 255u;
  if constexpr (F == kS1Trend) return (2 * (hist & 255u) - ((hist >> 8) & 255u)) & 255u;
  return 0u;
}

// Context byte offset of the lane for history `hist`. An order-0 lane
// (no feature) has one context, c0, hoisted out of the loop: the byte
// stores into the output could alias the lane table, so the compiler
// cannot hoist the two loads itself.
template <int FA, int FB>
CRY_INLINE u32 s1_ctx(const DecLane& L, u32 hist, u32 c0) noexcept {
  if constexpr (FA == kS1None && FB == kS1None) {
    (void)L;
    (void)hist;
    return c0;
  } else {
    return L.q[s1_feat<FA>(hist)] + L.q[256 + s1_feat<FB>(hist)];
  }
}

template <int FA, int FB, int RB, bool COMPACT>
CRY_INLINE void decode_one_s1(StreamState& st, usize step, const DecLane& L, u32 c0,
                             const DecCtx& dc) noexcept {
  const u32 hist = st.hist;
  const u32 c = s1_ctx<FA, FB>(L, hist, c0);
  const u32 slot = st.d.slot();
  const SlotInfo si = slot_lookup<COMPACT>(dc, c, slot);
  st.d.advance(si.freq, si.cum, slot);
  const u8 byte = static_cast<u8>(si.sym + s1_feat<RB>(hist));
  st.out[step] = byte;
  st.hist = (hist << 8) | byte;
}

template <int FA, int FB, int RB, bool COMPACT>
struct S1Body {
  CRY_INLINE static bool run(StreamState* st_in, int k, usize shortest, const DecCtx& dc_in) {
    const DecCtx dc = dc_in;
    const DecLane L = dc.lanes[0];
    const u32 cc = L.q[0] + L.q[256];  // the order-0 context (see s1_ctx)
    StreamState st[kMaxStreams];
    for (int s = 0; s < k; ++s) st[s] = st_in[s];
    usize step = 0;
    while (step < shortest) {
      const usize chunk_end = min_of(shortest, step + kCheckSteps);
      for (; step < chunk_end; ++step) {
        int s = 0;
        for (; s + 4 <= k; s += 4) {
          const u32 hist0 = st[s + 0].hist;
          const u32 hist1 = st[s + 1].hist;
          const u32 hist2 = st[s + 2].hist;
          const u32 hist3 = st[s + 3].hist;
          const u32 c0 = s1_ctx<FA, FB>(L, hist0, cc);
          const u32 c1 = s1_ctx<FA, FB>(L, hist1, cc);
          const u32 c2 = s1_ctx<FA, FB>(L, hist2, cc);
          const u32 c3 = s1_ctx<FA, FB>(L, hist3, cc);
          const u32 slot0 = st[s + 0].d.slot();
          const u32 slot1 = st[s + 1].d.slot();
          const u32 slot2 = st[s + 2].d.slot();
          const u32 slot3 = st[s + 3].d.slot();
          const SlotInfo si0 = slot_lookup<COMPACT>(dc, c0, slot0);
          const SlotInfo si1 = slot_lookup<COMPACT>(dc, c1, slot1);
          const SlotInfo si2 = slot_lookup<COMPACT>(dc, c2, slot2);
          const SlotInfo si3 = slot_lookup<COMPACT>(dc, c3, slot3);
          st[s + 0].d.advance(si0.freq, si0.cum, slot0);
          st[s + 1].d.advance(si1.freq, si1.cum, slot1);
          st[s + 2].d.advance(si2.freq, si2.cum, slot2);
          st[s + 3].d.advance(si3.freq, si3.cum, slot3);
          const u8 byte0 = static_cast<u8>(si0.sym + s1_feat<RB>(hist0));
          const u8 byte1 = static_cast<u8>(si1.sym + s1_feat<RB>(hist1));
          const u8 byte2 = static_cast<u8>(si2.sym + s1_feat<RB>(hist2));
          const u8 byte3 = static_cast<u8>(si3.sym + s1_feat<RB>(hist3));
          st[s + 0].out[step] = byte0;
          st[s + 1].out[step] = byte1;
          st[s + 2].out[step] = byte2;
          st[s + 3].out[step] = byte3;
          st[s + 0].hist = (hist0 << 8) | byte0;
          st[s + 1].hist = (hist1 << 8) | byte1;
          st[s + 2].hist = (hist2 << 8) | byte2;
          st[s + 3].hist = (hist3 << 8) | byte3;
        }
        for (; s < k; ++s) decode_one_s1<FA, FB, RB, COMPACT>(st[s], step, L, cc, dc);
      }
      for (int s = 0; s < k; ++s)
        if (st[s].d.p > st[s].limit) return false;
    }
    for (int s = 0; s < k; ++s) {
      for (usize t2 = step; t2 < st[s].len; ++t2) {
        decode_one_s1<FA, FB, RB, COMPACT>(st[s], t2, L, cc, dc);
        if (st[s].d.p > st[s].limit) return false;
      }
    }
    return true;
  }
};

// Maps a lane feature to its stride-1 equivalent; -1 if none.
int s1_code(Feat f) noexcept {
  switch (f) {
    case kNone: return kS1None;
    case kC1:
    case kLat1: return kS1C1;
    case kC2:
    case kLat2: return kS1C2;
    case kTrend: return kS1Trend;
    default: return -1;
  }
}

template <bool COMPACT>
bool decode_streams_stride1(StreamState* st, int k, usize shortest, const DecCtx& dc,
                            const LaneModel& lm, bool* handled) {
  const int fa = s1_code(lm.fa), fb = lm.kb > 1 ? s1_code(lm.fb) : kS1None;
  const int rb = s1_code(lm.rbase);
  *handled = fa >= 0 && fb >= 0 && rb >= 0;
  if (!*handled) return false;
#define CRY_S1_CASE(A, B, R) \
  case (A) * 16 + (B) * 4 + (R): \
    return run_dispatch<S1Body<A, B, R, COMPACT>>(st, k, shortest, dc);
#define CRY_S1_ROW(A, B) \
  CRY_S1_CASE(A, B, 0) CRY_S1_CASE(A, B, 1) CRY_S1_CASE(A, B, 2) CRY_S1_CASE(A, B, 3)
#define CRY_S1_BLOCK(A) \
  CRY_S1_ROW(A, 0) CRY_S1_ROW(A, 1) CRY_S1_ROW(A, 2) CRY_S1_ROW(A, 3)
  switch (fa * 16 + fb * 4 + rb) {
    CRY_S1_BLOCK(0) CRY_S1_BLOCK(1) CRY_S1_BLOCK(2) CRY_S1_BLOCK(3)
    default: *handled = false; return false;
  }
#undef CRY_S1_BLOCK
#undef CRY_S1_ROW
#undef CRY_S1_CASE
}

// The lattice detector is tuned for the CM; for static tables a wrong
// stride costs real bytes (S sets of tables, meaningless lattice features).
// Keep S > 1 only if per-lane order-0 tables beat one shared table on a
// sample, including the extra table cost.
int sanitize_stride(const u8* src, usize n, int stride) {
  if (stride <= 1) return 1;
  const usize S = static_cast<usize>(stride);
  const usize sample = min_of(n, usize{1} << 18) / S * S;
  if (sample < 4096) return 1;
  std::vector<u32> lane_cnt(S * 256, 0);
  u32 all_cnt[256] = {};
  for (usize i = 0; i < sample; i += S)
    for (usize l = 0; l < S; ++l) ++lane_cnt[l * 256 + src[i + l]];
  for (usize l = 0; l < S; ++l)
    for (int v = 0; v < 256; ++v) all_cnt[v] += lane_cnt[l * 256 + v];
  const float* xl = xlogx().t.data();
  double lanes = 0;
  for (usize l = 0; l < S; ++l) lanes += table_cost(lane_cnt.data() + l * 256, 256, xl);
  const double shared = table_cost(all_cnt, 256, xl);
  return lanes < shared ? stride : 1;
}

// ---------------------------------------------------------------------------
// Symbol index pass: for every byte of the block, the coder's table index
// idx[pos] = context * 256 + symbol under model m, counted into cnt and,
// when given, summed per context into ctot. Each stream is walked lane by
// lane past its guarded prologue; a lane's three features are then plain
// byte loads through per-lane pointers (src shifted by the feature's
// offset, or a block-wide trend / record-equal buffer), its first-feature
// map carries the context base and multiplier, so the inner loop is the
// same for every model. The lane goes in two interleaved halves: the
// count increments then form two independent chains (one chain is bound
// by the store-to-load forwarding latency whenever consecutive bytes
// repeat context and symbol, which a low-entropy lane does most of the
// time).
// ---------------------------------------------------------------------------
struct IndexScratch {
  std::vector<u8> trend;  // (2 * src[pos - S] - src[pos - 2S]) & 255
  std::vector<u8> req;    // record-equal flag at pos (lanes before pos's lane)
};

IndexScratch& index_scratch() {
  thread_local IndexScratch s;
  return s;
}

// Feature source for a lane: the value at pos is p[pos - off] (p null: 0).
struct FeatSrc {
  const u8* p = nullptr;
  usize off = 0;
};

inline FeatSrc feature_source(Feat f, const u8* src, usize S, const IndexScratch& sc) noexcept {
  switch (f) {
    case kC1: return {src, 1};
    case kC2: return {src, 2};
    case kLat1: return {src, S};
    case kLat2: return {src, 2 * S};
    case kTrend: return {sc.trend.data(), 0};
    case kRecEq: return {sc.req.data(), 0};
    default: return {};
  }
}

// One lane of one stream over records [r0, r1): pos = start + r * S + lane.
// qas[v] = base + qa[v] * kb, qbs[v] = qb[v] (all 0 without a second
// feature); fr is the base feature's source (kRb) or unused. Out of line
// with every source as a pointer advanced by one index: x86-64 has the
// registers for this loop and for little else.
template <bool kRb>
__attribute__((noinline)) void index_lane(const u32* qas, const u32* qbs, const u8* src,
                                          FeatSrc fa, FeatSrc fb, FeatSrc fr, usize start,
                                          usize S, usize lane, usize r0, usize r1, u32* idx,
                                          u32* cnt) {
  if (r1 <= r0) return;
  const usize nr = r1 - r0, half = nr / 2;
  const usize p0 = start + r0 * S + lane, p1 = p0 + half * S;
  const u8* a0 = fa.p + (p0 - fa.off);
  const u8* a1 = fa.p + (p1 - fa.off);
  const u8* b0 = fb.p + (p0 - fb.off);
  const u8* b1 = fb.p + (p1 - fb.off);
  const u8* c0 = kRb ? fr.p + (p0 - fr.off) : nullptr;
  const u8* c1 = kRb ? fr.p + (p1 - fr.off) : nullptr;
  const u8* s0 = src + p0;
  const u8* s1 = src + p1;
  u32* i0 = idx + p0;
  u32* i1 = idx + p1;
  auto step = [&](const u8* a, const u8* b, const u8* c, const u8* sp, u32* ip, usize d) {
    u32 t = sp[d];
    if constexpr (kRb) t -= c[d];
    const u32 i = (qas[a[d]] + qbs[b[d]]) * 256 + (t & 255u);
    ip[d] = i;
    ++cnt[i];
  };
  usize d = 0;
  for (usize i = 0; i < half; ++i, d += S) {
    step(a0, b0, c0, s0, i0, d);
    step(a1, b1, c1, s1, i1, d);
  }
  if (nr & 1) step(a1, b1, c1, s1, i1, d);  // record r1 - 1 = p1 + half * S
}

void index_block(const Model& m, const u8* src, const usize* bounds, int k, u32* idx, u32* cnt,
                 u32* ctot) {
  const usize S = static_cast<usize>(m.stride);
  const usize n = bounds[k];
  IndexScratch& sc = index_scratch();
  bool uses_trend = false, uses_req = false;
  for (int l = 0; l < m.stride; ++l) {
    const LaneModel& lm = m.lane[l];
    uses_trend |= lm.fa == kTrend || lm.fb == kTrend || lm.rbase == kTrend;
    uses_req |= lm.fa == kRecEq || lm.fb == kRecEq;
  }
  if (uses_trend) {
    // Only positions past a stream's prologue are read; all lie >= 2S.
    if (sc.trend.size() < n) sc.trend.resize(n);
    u8* tr = sc.trend.data();
    for (usize pos = 2 * S; pos < n; ++pos)
      tr[pos] = static_cast<u8>(2 * src[pos - S] - src[pos - 2 * S]);
  }
  if (uses_req) {
    // The record-equal chain of the byte-major definition: reset at each
    // record, compared against 0 before the stream's second record.
    if (sc.req.size() < n) sc.req.resize(n);
    u8* rq = sc.req.data();
    for (int s = 0; s < k; ++s) {
      const usize start = bounds[s], end = bounds[s + 1];
      u32 rec_eq = 1;
      usize l = 0;
      for (usize pos = start; pos < end; ++pos) {
        if (l == 0) rec_eq = 1;
        rq[pos] = static_cast<u8>(rec_eq);
        rec_eq &= src[pos] == feat_at(src, pos, start, S);
        if (++l == S) l = 0;
      }
    }
  }
  // Per-lane tables and sources.
  u32 qas[kMaxLanes][256], qbs[kMaxLanes][256];
  FeatSrc pa[kMaxLanes], pb[kMaxLanes], pr[kMaxLanes];
  for (int l = 0; l < m.stride; ++l) {
    const LaneModel& lm = m.lane[l];
    for (int v = 0; v < 256; ++v) {
      qas[l][v] = lm.base + lm.qa[v] * static_cast<u32>(lm.kb);
      qbs[l][v] = lm.fb != kNone ? lm.qb[v] : 0u;
    }
    pa[l] = feature_source(lm.fa, src, S, sc);
    pb[l] = feature_source(lm.fb, src, S, sc);
    pr[l] = feature_source(lm.rbase, src, S, sc);
    // A feature of kind "none" reads any valid byte: its map is constant.
    if (!pa[l].p) {
      std::fill(qas[l], qas[l] + 256, qas[l][0]);
      pa[l] = {src, 0};
    }
    if (!pb[l].p) pb[l] = {src, 0};
  }
  for (int s = 0; s < k; ++s) {
    const usize start = bounds[s], end = bounds[s + 1];
    u32 rec_eq = 1;
    usize pos = start;
    // Guarded prologue: the first two records plus two bytes, byte by byte.
    const usize pro = min_of(end, start + 2 * S + 2);
    for (; pos < pro; ++pos) {
      if ((pos - start) % S == 0) rec_eq = 1;
      const LaneModel& lm = m.lane[(pos - start) % S];
      const u32 c = context_of(m, lm, src, pos, start, rec_eq);
      const u32 t = (src[pos] - feat_value(lm.rbase, src, pos, start, S)) & 255u;
      const u32 i = c * 256 + t;
      idx[pos] = i;
      ++cnt[i];
      rec_eq &= src[pos] == feat_at(src, pos, start, S);
    }
    if (pos >= end) continue;
    // Steady state, lane by lane. Record r0 holds pos; its lanes before
    // l0 are done.
    const usize r0 = (pos - start) / S, l0 = (pos - start) % S;
    for (usize l = 0; l < S; ++l) {
      const usize first = l >= l0 ? r0 : r0 + 1;
      const usize last = end - start > l ? (end - start - l + S - 1) / S : 0;
      if (pr[l].p)
        index_lane<true>(qas[l], qbs[l], src, pa[l], pb[l], pr[l], start, S, l, first, last, idx,
                         cnt);
      else
        index_lane<false>(qas[l], qbs[l], src, pa[l], pb[l], FeatSrc{}, start, S, l, first, last,
                          idx, cnt);
    }
  }
  if (ctot)
    for (u32 c = 0; c < m.nctx; ++c) {
      const u32* row = cnt + static_cast<usize>(c) * 256;
      u32 tot = 0;
      for (int t = 0; t < 256; ++t) tot += row[t];
      ctot[c] = tot;
    }
}

// ---------------------------------------------------------------------------
// Stream encoding: each stream is coded backwards into its own slice of
// buf (2 bytes per symbol plus the flush); streams go four at a time in
// lockstep so four rANS states advance in parallel instead of one
// dependency chain per stream.
// ---------------------------------------------------------------------------
template <int N>
void encode_lockstep(const EncSym* esym, const u32* idx, RansEnc* enc, usize* pos, usize steps) {
  for (usize i = 0; i < steps; ++i)
    for (int j = 0; j < N; ++j) enc[j].put(esym[idx[--pos[j]]]);
}

void encode_streams(const EncSym* esym, const u32* idx, const usize* bounds, int k, u8* buf,
                    const u8** stream_ptr, usize* stream_len) {
  for (int s0 = 0; s0 < k; s0 += 4) {
    const int ns = min_of(4, k - s0);
    RansEnc enc[4] = {RansEnc(nullptr), RansEnc(nullptr), RansEnc(nullptr), RansEnc(nullptr)};
    u8* end[4];
    usize pos[4], lo[4];
    usize steps = ~usize{0};
    for (int j = 0; j < ns; ++j) {
      const int s = s0 + j;
      const usize cap = (bounds[s + 1] - bounds[s]) * 2 + 8;
      end[j] = buf + bounds[s] * 2 + static_cast<usize>(s) * 8 + cap;
      enc[j] = RansEnc(end[j]);
      lo[j] = bounds[s];
      pos[j] = bounds[s + 1];
      steps = min_of(steps, pos[j] - lo[j]);
    }
    switch (ns) {
      case 4: encode_lockstep<4>(esym, idx, enc, pos, steps); break;
      case 3: encode_lockstep<3>(esym, idx, enc, pos, steps); break;
      case 2: encode_lockstep<2>(esym, idx, enc, pos, steps); break;
      default: encode_lockstep<1>(esym, idx, enc, pos, steps); break;
    }
    for (int j = 0; j < ns; ++j) {
      while (pos[j] > lo[j]) enc[j].put(esym[idx[--pos[j]]]);
      stream_ptr[s0 + j] = enc[j].flush();
      stream_len[s0 + j] = static_cast<usize>(end[j] - stream_ptr[s0 + j]);
    }
  }
}

}  // namespace

void train_model_from_sample(const u8* src, usize n, int stride, int level, Model& m,
                            std::vector<u32>* out_idx = nullptr,
                            bool for_profile = false) {
  // A profile keeps the column's element stride: sanitize_stride trades
  // the per-lane tables against their header cost on this one block,
  // while a profile ships its header once, and merging the lanes would
  // spread each lane's pseudo-counts (below) over the others' symbols.
  if (!for_profile) stride = sanitize_stride(src, n, stride);
  m.stride = stride;
  const int k = streams_for(n);
  usize bounds[kMaxStreams + 1];
  split_streams(n, stride, k, bounds);
  const usize S = static_cast<usize>(stride);

  // Gather per-lane feature values and symbols (with stream-start rules).
  std::vector<LaneStats> stats(static_cast<usize>(stride));
  // Analysis sample: up to kAnalysisMax symbols per lane, spread over the
  // streams (each stream contributes its leading part).
  const usize per_stream = (kAnalysisMax / static_cast<usize>(k)) * S;  // <= kAnalysisMax per lane
  usize sample = 0;
  for (int s = 0; s < k; ++s) sample += min_of(bounds[s + 1], bounds[s] + per_stream) - bounds[s];
  for (int l = 0; l < stride; ++l) {
    stats[l].raw.resize(sample / S + 1);
    for (int f = 1; f < kFeatCount; ++f) stats[l].fv[f].resize(sample / S + 1);
  }
  usize fill[kMaxLanes] = {};
  // Per stream: de-interleave the lanes, then derive every feature from
  // the lanes' raw bytes. The byte at pos - 1 is the previous lane of the
  // record (lane S - 1 of the record before for lane 0), pos - 2 the lane
  // before that, pos - S and pos - 2S the lane's own earlier records. The
  // first records of a stream read as 0 before the stream start (feat_at).
  // kRecEq per record: 1 while every earlier lane of the record repeats the
  // previous record, computed lane by lane.
  std::vector<u8> req_prev, req_cur;
  for (int s = 0; s < k; ++s) {
    const usize start = bounds[s];
    const usize lim = min_of(bounds[s + 1], start + per_stream);
    const usize len = lim - start;
    const usize nrec = (len + S - 1) / S;
    usize seg[kMaxLanes], cnt[kMaxLanes];  // this stream's segment per lane
    for (usize l = 0; l < S; ++l) {
      seg[l] = fill[l];
      cnt[l] = len > l ? (len - l + S - 1) / S : 0;
      u8* raw = stats[l].raw.data() + seg[l];
      const u8* p = src + start + l;
      for (usize r = 0; r < cnt[l]; ++r) raw[r] = p[r * S];
    }
    req_prev.assign(nrec, 1);  // lane 0: always 1
    req_cur.resize(nrec);
    for (usize l = 0; l < S; ++l) {
      LaneStats& ls = stats[l];
      const usize nr = cnt[l];
      const u8* raw = ls.raw.data() + seg[l];
      u8* fc1 = ls.fv[kC1].data() + seg[l];
      u8* fc2 = ls.fv[kC2].data() + seg[l];
      u8* fl1 = ls.fv[kLat1].data() + seg[l];
      u8* fl2 = ls.fv[kLat2].data() + seg[l];
      u8* ftr = ls.fv[kTrend].data() + seg[l];
      u8* frq = ls.fv[kRecEq].data() + seg[l];
      // Own lane, one and two records back.
      auto shifted = [&](u8* dst, const u8* from, usize from_n, usize d) {
        const usize z = min_of(d, nr);
        std::memset(dst, 0, z);
        if (nr > z) std::memcpy(dst + z, from, min_of(nr - z, from_n));
      };
      shifted(fl1, raw, nr, 1);
      shifted(fl2, raw, nr, 2);
      // Previous bytes: lane l - 1 (l - 2) of the same record, wrapping
      // into the record before (two records before for S == 1).
      const usize la = (l + S - 1) % S, lb = (l + S - 2) % S;
      const usize da = l >= 1 ? 0 : 1;
      const usize db = l >= 2 ? 0 : (S == 1 ? 2 : 1);
      shifted(fc1, stats[la].raw.data() + seg[la], cnt[la], da);
      shifted(fc2, stats[lb].raw.data() + seg[lb], cnt[lb], db);
      for (usize r = 0; r < nr; ++r) ftr[r] = static_cast<u8>(2 * fl1[r] - fl2[r]);
      for (usize r = 0; r < nr; ++r) {
        frq[r] = req_prev[r];
        req_cur[r] = static_cast<u8>(req_prev[r] & (raw[r] == fl1[r]));
      }
      fill[l] += nr;
      req_prev.swap(req_cur);
    }
  }
  for (int l = 0; l < stride; ++l) {
    stats[l].raw.resize(fill[l]);
    for (int f = 1; f < kFeatCount; ++f) stats[l].fv[f].resize(fill[l]);
  }
  // Lane modes (dynamic blocks only; a profile serves pages its sample did
  // not see). Constant lanes are known before the search; raw lanes need
  // the final counts and are decided after the index pass below.
  if (!for_profile)
    for (int l = 0; l < stride; ++l) {
      const usize cnt_l = lane_count(n, S, static_cast<usize>(l));
      const u8 v = cnt_l ? src[l] : 0;
      usize i = 1;
      while (i < cnt_l && src[static_cast<usize>(l) + i * S] == v) ++i;
      if (i >= cnt_l) {
        m.lane[l] = LaneModel{};
        m.lane[l].mode = kLaneConst;
        m.lane[l].value = v;
      }
    }
  // Context bases: coded lanes in order; the skipped lanes point at the
  // sentinel row past the last context (see build_crystal_static).
  auto assign_bases = [&] {
    u32 base = 0;
    for (int l = 0; l < stride; ++l)
      if (m.lane[l].mode == kLaneCoded) {
        m.lane[l].base = base;
        base += static_cast<u32>(m.lane[l].nctx());
      }
    m.nctx = base;
    for (int l = 0; l < stride; ++l)
      if (m.lane[l].mode != kLaneCoded) m.lane[l].base = base;
  };
  for (int l = 0; l < stride; ++l)
    if (m.lane[l].mode == kLaneCoded) choose_lane_model(stats[l], level, stride, m.lane[l]);
  assign_bases();

  // Final counts per context (and the coder's symbol indices). One extra
  // row for the skipped lanes' bytes.
  std::vector<u32> cnt((static_cast<usize>(m.nctx) + 1) * 256, 0);
  std::vector<u32> idx_local;
  std::vector<u32>& idx = out_idx ? *out_idx : idx_local;
  idx.resize(n);
  index_block(m, src, bounds, k, idx.data(), cnt.data(), nullptr);
  if (!for_profile) {
    // Raw lanes: a coded lane whose size under its own tables (entropy of
    // the counts plus the table headers) is not below its byte count is
    // noise to the model; shipping it verbatim costs nothing in size and
    // takes it out of the rANS loop. Strict: never larger than coded.
    const float* xl = xlogx().t.data();
    bool changed = false;
    for (int l = 0; l < stride; ++l) {
      LaneModel& lm = m.lane[l];
      if (lm.mode != kLaneCoded) continue;
      double bits = 0;
      for (u32 c = lm.base; c < lm.base + static_cast<u32>(lm.nctx()); ++c)
        bits += table_cost(cnt.data() + static_cast<usize>(c) * 256, 256, xl);
      const double raw_bits = 8.0 * static_cast<double>(lane_count(n, S, static_cast<usize>(l)));
      if (bits < raw_bits * (1.0 - kRawLaneSlack)) continue;
      lm = LaneModel{};
      lm.mode = kLaneRaw;
      changed = true;
    }
    if (changed) {
      assign_bases();
      cnt.assign((static_cast<usize>(m.nctx) + 1) * 256, 0);
      index_block(m, src, bounds, k, idx.data(), cnt.data(), nullptr);
    }
  }
  if (for_profile) {
    // A profile codes pages outside its sample, so symbols the sample did
    // not show need a frequency too. The smallest one, 1/kM, is a real
    // cost: 255 unseen symbols take 6% of a context's mass, 0.09 bits per
    // byte, which on a constant lane or a low-entropy count column is 10%
    // of the page. A page that meets a symbol at frequency 0 is not codable
    // with the profile and takes the dynamic path (build_crystal_profile's
    // escape), which costs it the per-page model header, ~1.5 KiB. So:
    //  - a minor context (under kMinorContext observations in the sample,
    //    e.g. the stream starts) is one where the pseudo-counts cost next
    //    to nothing on a page (0.09 bits per use) against the header of an
    //    escape: every unseen symbol gets one;
    //  - elsewhere the pseudo-count goes only to the unseen symbols inside
    //    the sample's support, the ring interval spanned by the seen
    //    symbols plus a margin of kPriorMargin on both sides (the gaps of a
    //    count's tail, the next values of a growing one, the 0xFF of a
    //    negative delta next to the 0x00 of a positive one).
    // The seen counts are scaled up against the pseudo-counts so they keep
    // their proportions.
    constexpr int kPriorMargin = 2;
    constexpr usize kMinorContext = 4096;
    const usize scale = min_of<usize>(256, usize{0xFFFFFF00u} / (n ? n : 1));
    for (u32 c = 0; c < m.nctx; ++c) {
      u32* h = cnt.data() + static_cast<usize>(c) * 256;
      int seen[256];
      int ns = 0;
      usize total = 0;
      for (int s = 0; s < 256; ++s) {
        if (!h[s]) continue;
        seen[ns++] = s;
        total += h[s];
        h[s] *= static_cast<u32>(scale);
      }
      for (int s = 0; s < 256; ++s)
        if (!h[s]) h[s] = 1;
      if (total < kMinorContext) continue;  // includes a context the sample never used: uniform
      // The support's complement is the largest run of unseen symbols
      // between two seen ones on the ring; its interior beyond the margin
      // stays at frequency 0.
      int gap_start = 0, gap_len = 0;
      for (int i = 0; i < ns; ++i) {
        const int a = seen[i], b = seen[(i + 1) % ns];
        const int len = (b - a - 1 + 256) & 255;
        if (len > gap_len) {
          gap_len = len;
          gap_start = a + 1;
        }
      }
      for (int j = kPriorMargin; j < gap_len - kPriorMargin; ++j) h[(gap_start + j) & 255] = 0;
    }
  }
  m.freq.assign(static_cast<usize>(m.nctx) * 256, 0);
  for (u32 c = 0; c < m.nctx; ++c)
    quantize(cnt.data() + static_cast<usize>(c) * 256, m.freq.data() + static_cast<usize>(c) * 256);
}

// Payload after the 2-byte (ver, kind) prefix:
//   u8 stride, u8 K, varint model_len, model, varint stream sizes (K-1),
//   streams.
void build_crystal_static(const u8* src, usize n, int level, int stride,
                         std::vector<u8>& payload) {
  Model m;
  std::vector<u32> idx;
  train_model_from_sample(src, n, stride, level, m, &idx, false);
  stride = m.stride;  // sanitized by the training
  // No coded lane: the streams carry nothing, one 4-byte flush suffices.
  const int k = m.nctx ? streams_for(n) : 1;
  usize bounds[kMaxStreams + 1];
  split_streams(n, stride, k, bounds);

  // Cumulative frequencies for encoding.
  std::vector<u16> cum(static_cast<usize>(m.nctx) * 256);
  for (u32 c = 0; c < m.nctx; ++c) {
    u32 acc = 0;
    for (int s = 0; s < 256; ++s) {
      cum[static_cast<usize>(c) * 256 + s] = static_cast<u16>(acc);
      acc += m.freq[static_cast<usize>(c) * 256 + s];
    }
  }

  payload.push_back(static_cast<u8>(stride));
  payload.push_back(static_cast<u8>(k));
  std::vector<u8> model_bytes;
  write_model(m, model_bytes);
  ByteWriter w(payload);
  w.varint(model_bytes.size());
  w.bytes(model_bytes);
  // Verbatim section: the raw lanes' bytes, lane by lane.
  write_raw_section(lane_modes_of(m), static_cast<usize>(stride), src, n, payload);

  // Encoder symbols (only for symbols that occur). The bytes of the
  // skipped lanes index the sentinel row past the last context, whose
  // symbols have the full frequency kM: a put leaves the coder state
  // unchanged (x = kM * (x >> kPrec) + slot, no renormalisation), so the
  // stream loop needs no special case for them.
  std::vector<EncSym> esym((static_cast<usize>(m.nctx) + 1) * 256);
  for (usize i = 0; i < static_cast<usize>(m.nctx) * 256; ++i)
    if (m.freq[i]) esym[i].init(m.freq[i], cum[i]);
  for (usize i = static_cast<usize>(m.nctx) * 256; i < esym.size(); ++i) esym[i].init(kM, 0);

  // Each stream is written backwards into its own slice of one buffer; a
  // symbol costs at most 2 bytes plus the 4-byte flush.
  std::vector<u8> buf(n * 2 + static_cast<usize>(k) * 8);
  const u8* stream_ptr[kMaxStreams];
  usize stream_len[kMaxStreams];
  encode_streams(esym.data(), idx.data(), bounds, k, buf.data(), stream_ptr, stream_len);
  for (int s = 0; s + 1 < k; ++s) w.varint(stream_len[s]);
  for (int s = 0; s < k; ++s) w.bytes(stream_ptr[s], stream_len[s]);
}

static Status decode_crystal_streams(const Model& m, bool compact, const u8* tab,
                                     const DecLane* lanes, bool trend, bool req,
                                     int k, ByteReader& r, u8* dst, usize n) {
  const int stride = m.stride;
  usize lens[kMaxStreams];
  usize sum = 0;
  for (int s = 0; s + 1 < k; ++s) {
    lens[s] = static_cast<usize>(r.varint());
    if (r.error) return Status::CorruptInput;
    sum += lens[s];
  }
  if (sum > r.remaining()) return Status::CorruptInput;
  lens[k - 1] = r.remaining() - sum;

  DecTables& t = dec_tables();

  usize bounds[kMaxStreams + 1];
  split_streams(n, stride, k, bounds);
  // Copy the streams into one padded buffer. A stream of a valid block is
  // never over-read; a corrupt one may read into the padding of the next
  // stream, which is harmless (each stream gets its own zero padding).
  usize total = 0;
  for (int s = 0; s < k; ++s) total += lens[s] + kRansPad;
  if (t.padded.size() < total) {
    t.padded.resize(total);
  }
  StreamState st[kMaxStreams];
  usize shortest = n;
  usize off = 0;
  for (int s = 0; s < k; ++s) {
    const u8* p = r.take(lens[s]);
    if (!p) return Status::CorruptInput;
    std::memcpy(t.padded.data() + off, p, lens[s]);
    std::memset(t.padded.data() + off + lens[s], 0, kRansPad);
    st[s].d.init(t.padded.data() + off);
    st[s].limit = t.padded.data() + off + lens[s];
    off += lens[s] + kRansPad;
    st[s].out = dst + bounds[s];
    st[s].len = bounds[s + 1] - bounds[s];
    shortest = min_of(shortest, st[s].len);
  }
  const usize S = static_cast<usize>(stride);
  const DecCtx dc{tab, lanes, S, dst, t.padded.data()};
  bool ok;
#if CRY_V8
  if (!compact && (k == kV8Width || k == 2 * kV8Width) && use_v8())
    return decode_streams_v8(st, k, shortest, dc, trend, req) ? Status::Ok : Status::CorruptInput;
#endif
  if (stride == 1) {
    bool handled;
    ok = compact ? decode_streams_stride1<true>(st, k, shortest, dc, m.lane[0], &handled)
                 : decode_streams_stride1<false>(st, k, shortest, dc, m.lane[0], &handled);
    if (handled) return ok ? Status::Ok : Status::CorruptInput;
  }
  ok = compact ? decode_streams<true>(st, k, shortest, dc, trend, req)
               : decode_streams<false>(st, k, shortest, dc, trend, req);
  if (!ok) return Status::CorruptInput;
  return Status::Ok;
}

// raw: the verbatim section (the raw lanes' bytes), null when every lane is
// coded (always for profiles).
static Status decode_crystal_model_streams(const Model& m, int k, ByteReader& r, u8* dst,
                                           usize n, const u8* raw) {
  const int stride = m.stride;
  if (!model_all_coded(m)) {
    fill_skipped_lanes(lane_modes_of(m), static_cast<usize>(stride), raw, dst, n);
    if (m.nctx == 0) return Status::Ok;  // nothing coded: the streams are empty flushes
  }
  DecTables& t = dec_tables();
  const bool compact = cry_compact_tables(m.nctx);
  build_dec_tables(m, t.tab, compact);

  const u32 cb = ctx_bytes_for(compact);
  t.q_scaled.resize(static_cast<usize>(stride) * 512);
  DecLane lanes[kMaxLanes];
  bool trend = false, req = false;
  for (int l = 0; l < stride && l < kMaxLanes; ++l) {
    const LaneModel& lm = m.lane[l];
    u32* qa = t.q_scaled.data() + static_cast<usize>(l) * 512;
    u32* qb = qa + 256;
    for (int v = 0; v < 256; ++v) {
      qa[v] = (lm.base + lm.qa[v] * static_cast<u32>(lm.kb)) * cb;
      qb[v] = lm.qb[v] * cb;
    }
    lanes[l] = make_dec_lane(lm, qa);
    trend |= lm.fa == kTrend || lm.fb == kTrend || lm.rbase == kTrend;
    req |= lm.fa == kRecEq || lm.fb == kRecEq;
  }
  return decode_crystal_streams(m, compact, t.tab.data(), lanes, trend, req, k, r, dst, n);
}

Status decode_crystal_static(const u8* data, usize len, u8* dst, usize n) {
  ByteReader r(data, len);
  const int stride = r.byte();
  const int k = r.byte();
  if (r.error || stride < 1 || stride > kMaxLanes || k < 1 || k > kMaxStreams)
    return Status::CorruptInput;
  const usize model_len = static_cast<usize>(r.varint());
  if (r.error || model_len > r.remaining()) return Status::CorruptInput;
  const u8* model_bytes = r.take(model_len);
  Model m;
  m.stride = stride;
  if (!read_model(m, model_bytes, model_len)) return Status::CorruptInput;
  const usize raw_len = raw_section_size(lane_modes_of(m), static_cast<usize>(stride), n);
  const u8* raw = nullptr;
  if (raw_len) {
    raw = r.take(raw_len);
    if (!raw) return Status::CorruptInput;
  }
  return decode_crystal_model_streams(m, k, r, dst, n, raw);
}

// ---------------------------------------------------------------------------
// Pre-trained static probability profiles (zero per-block header tax).
// ---------------------------------------------------------------------------
//
// A profile may carry an element transform (a delta over the column's
// W-byte little-endian elements, plain or zig-zag), chosen when the profile
// is trained and applied to every page coded with it: the pages of an
// offset / index column drift (each page starts where the previous one
// ended), so the byte values of the sample say nothing about the byte
// values of a later page, while their first differences (the collection
// sizes) are stationary and are what the profile models. The transform is
// part of the model bytes (high bits of the leading stride byte, 0 for
// older models), so a reader needs nothing beyond the model.
struct StaticProfileHolder {
  Model model;
  Filter transform = Filter::None;
  std::vector<EncSym> esym;

  // Precomputed decoder tables & structures (read-only after finalize_profile)
  bool compact = false;
  bool trend = false;
  bool req = false;
  std::vector<u8> dec_tab;
  std::vector<u32> q_scaled;
  std::array<DecLane, kMaxLanes> lanes{};

  StaticProfileHolder() = default;
  StaticProfileHolder(const StaticProfileHolder&) = delete;
  StaticProfileHolder& operator=(const StaticProfileHolder&) = delete;
  StaticProfileHolder(StaticProfileHolder&& o) noexcept
      : model(std::move(o.model)),
        transform(o.transform),
        esym(std::move(o.esym)),
        compact(o.compact),
        trend(o.trend),
        req(o.req),
        dec_tab(std::move(o.dec_tab)),
        q_scaled(std::move(o.q_scaled)),
        lanes(o.lanes) {
    for (int l = 0; l < model.stride && l < kMaxLanes; ++l) {
      lanes[l].q = q_scaled.data() + static_cast<usize>(l) * 512;
    }
  }
  StaticProfileHolder& operator=(StaticProfileHolder&& o) noexcept {
    if (this != &o) {
      model = std::move(o.model);
      transform = o.transform;
      esym = std::move(o.esym);
      compact = o.compact;
      trend = o.trend;
      req = o.req;
      dec_tab = std::move(o.dec_tab);
      q_scaled = std::move(o.q_scaled);
      lanes = o.lanes;
      for (int l = 0; l < model.stride && l < kMaxLanes; ++l) {
        lanes[l].q = q_scaled.data() + static_cast<usize>(l) * 512;
      }
    }
    return *this;
  }
};

static void finalize_profile(StaticProfileHolder& h) {
  const Model& m = h.model;
  if (m.stride < 1 || m.stride > kMaxLanes) return;
  std::vector<u16> cum(static_cast<usize>(m.nctx) * 256);
  for (u32 c = 0; c < m.nctx; ++c) {
    u32 acc = 0;
    for (int s = 0; s < 256; ++s) {
      cum[static_cast<usize>(c) * 256 + s] = static_cast<u16>(acc);
      acc += m.freq[static_cast<usize>(c) * 256 + s];
    }
  }
  // One sentinel row past the last context for the bytes of a page's
  // skipped lanes (kCryProfileLanes): full frequency, a put leaves the
  // coder state unchanged, as in build_crystal_static.
  h.esym.resize((static_cast<usize>(m.nctx) + 1) * 256);
  for (usize i = 0; i < static_cast<usize>(m.nctx) * 256; ++i) {
    if (m.freq[i]) h.esym[i].init(m.freq[i], cum[i]);
  }
  for (usize i = static_cast<usize>(m.nctx) * 256; i < h.esym.size(); ++i) h.esym[i].init(kM, 0);

  // Precomputed decoder tables & structures
  h.compact = cry_compact_tables(m.nctx);
  build_dec_tables(m, h.dec_tab, h.compact);

  const usize stride = static_cast<usize>(m.stride);
  const u32 cb = ctx_bytes_for(h.compact);
  h.q_scaled.resize(stride * 512);
  h.trend = false;
  h.req = false;
  for (int l = 0; l < m.stride && l < kMaxLanes; ++l) {
    const LaneModel& lm = m.lane[l];
    u32* qa = h.q_scaled.data() + static_cast<usize>(l) * 512;
    u32* qb = qa + 256;
    for (int v = 0; v < 256; ++v) {
      qa[v] = (lm.base + lm.qa[v] * static_cast<u32>(lm.kb)) * cb;
      qb[v] = lm.qb[v] * cb;
    }
    h.lanes[l] = make_dec_lane(lm, qa);
    h.trend |= lm.fa == kTrend || lm.fb == kTrend || lm.rbase == kTrend;
    h.req |= lm.fa == kRecEq || lm.fb == kRecEq;
  }
}

static void init_float_kinematics(StaticProfileHolder& h) {
  Model& m = h.model;
  m.stride = 4;
  m.nctx = 4;
  m.freq.assign(4 * 256, 0);

  // Lane 0: Mantissa LSB (bits 0..7) -> uniform noise
  m.lane[0].fa = kNone;
  m.lane[0].fb = kNone;
  m.lane[0].rbase = kNone;
  m.lane[0].ka = 1;
  m.lane[0].kb = 1;
  m.lane[0].base = 0;
  {
    u32 cnt[256];
    std::fill(cnt, cnt + 256, 100u);
    quantize(cnt, m.freq.data() + 0 * 256);
  }

  // Lane 1: Mantissa mid (bits 8..15) -> delta relative to lat1
  m.lane[1].fa = kNone;
  m.lane[1].fb = kNone;
  m.lane[1].rbase = kLat1;
  m.lane[1].ka = 1;
  m.lane[1].kb = 1;
  m.lane[1].base = 1;
  {
    u32 cnt[256];
    for (int s = 0; s < 256; ++s) {
      int d = s <= 128 ? s : 256 - s;
      cnt[s] = static_cast<u32>(1 + 30000.0 / (1.0 + d * 4.0));
    }
    quantize(cnt, m.freq.data() + 1 * 256);
  }

  // Lane 2: Mantissa upper (bits 16..22) + exp LSB (bit 23) -> sharper delta
  m.lane[2].fa = kNone;
  m.lane[2].fb = kNone;
  m.lane[2].rbase = kLat1;
  m.lane[2].ka = 1;
  m.lane[2].kb = 1;
  m.lane[2].base = 2;
  {
    u32 cnt[256];
    for (int s = 0; s < 256; ++s) {
      int d = s <= 128 ? s : 256 - s;
      cnt[s] = static_cast<u32>(1 + 60000.0 / (1.0 + d * 8.0));
    }
    quantize(cnt, m.freq.data() + 2 * 256);
  }

  // Lane 3: Exp upper (bits 24..30) + sign (bit 31) -> very sharp delta (mostly 0)
  m.lane[3].fa = kNone;
  m.lane[3].fb = kNone;
  m.lane[3].rbase = kLat1;
  m.lane[3].ka = 1;
  m.lane[3].kb = 1;
  m.lane[3].base = 3;
  {
    u32 cnt[256];
    for (int s = 0; s < 256; ++s) {
      int d = s <= 128 ? s : 256 - s;
      cnt[s] = static_cast<u32>(1 + 120000.0 / (1.0 + d * 16.0));
    }
    quantize(cnt, m.freq.data() + 3 * 256);
  }
}

static void init_poisson_counts(StaticProfileHolder& h) {
  Model& m = h.model;
  m.stride = 4;
  m.nctx = 4;
  m.freq.assign(4 * 256, 0);

  // Lane 0: LSB -> Poisson / discrete count distribution
  m.lane[0].fa = kNone;
  m.lane[0].fb = kNone;
  m.lane[0].rbase = kNone;
  m.lane[0].ka = 1;
  m.lane[0].kb = 1;
  m.lane[0].base = 0;
  {
    u32 cnt[256];
    std::fill(cnt, cnt + 256, 1u);
    cnt[0] += 30000u;
    for (int s = 1; s <= 60; ++s) {
      double p = std::exp(-4.0) * std::pow(4.0, s) / std::tgamma(s + 1.0);
      cnt[s] += static_cast<u32>(60000.0 * p);
    }
    quantize(cnt, m.freq.data() + 0 * 256);
  }

  // Lanes 1, 2, 3: High bytes of small integers -> 0x00 with probability > 99.9%
  for (int l = 1; l < 4; ++l) {
    m.lane[l].fa = kNone;
    m.lane[l].fb = kNone;
    m.lane[l].rbase = kNone;
    m.lane[l].ka = 1;
    m.lane[l].kb = 1;
    m.lane[l].base = static_cast<u32>(l);
    u32 cnt[256];
    std::fill(cnt, cnt + 256, 1u);
    cnt[0] = 100000u;
    quantize(cnt, m.freq.data() + l * 256);
  }
}

static void init_int_monotonic(StaticProfileHolder& h) {
  Model& m = h.model;
  m.stride = 4;
  m.nctx = 4;
  m.freq.assign(4 * 256, 0);

  for (int l = 0; l < 4; ++l) {
    m.lane[l].fa = kNone;
    m.lane[l].fb = kNone;
    m.lane[l].rbase = kLat1;
    m.lane[l].ka = 1;
    m.lane[l].kb = 1;
    m.lane[l].base = static_cast<u32>(l);
  }

  // Lane 0: Delta is mostly +1, occasionally 0 or +2..+5
  {
    u32 cnt[256];
    for (int s = 0; s < 256; ++s) {
      int d = s <= 128 ? s : 256 - s;
      cnt[s] = 1 + static_cast<u32>(1000.0 / (1.0 + d));
    }
    cnt[1] += 80000u;
    cnt[0] += 15000u;
    cnt[2] += 5000u;
    cnt[3] += 2000u;
    quantize(cnt, m.freq.data() + 0 * 256);
  }

  // Lanes 1, 2, 3: Delta is 0 (carry happens only every 256 / 65536 rows)
  {
    u32 cnt[256];
    std::fill(cnt, cnt + 256, 1u);
    cnt[0] = 100000u;
    cnt[1] = 400u;
    quantize(cnt, m.freq.data() + 1 * 256);
  }
  for (int l = 2; l < 4; ++l) {
    u32 cnt[256];
    std::fill(cnt, cnt + 256, 1u);
    cnt[0] = 100000u;
    quantize(cnt, m.freq.data() + l * 256);
  }
}

static void init_text(StaticProfileHolder& h) {
  Model& m = h.model;
  m.stride = 1;
  m.nctx = 8;
  m.freq.assign(8 * 256, 0);

  m.lane[0].fa = kC1;
  m.lane[0].fb = kNone;
  m.lane[0].rbase = kNone;
  m.lane[0].ka = 8;
  m.lane[0].kb = 1;
  m.lane[0].base = 0;

  for (int v = 0; v < 256; ++v) {
    if (v <= 32) m.lane[0].qa[v] = 0;
    else if (v == 'a' || v == 'e' || v == 'i' || v == 'o' || v == 'u') m.lane[0].qa[v] = 1;
    else if (v >= 'a' && v <= 'z') m.lane[0].qa[v] = 2;
    else if (v >= 'A' && v <= 'Z') m.lane[0].qa[v] = 3;
    else if (v >= '0' && v <= '9') m.lane[0].qa[v] = 4;
    else if (v == '<' || v == '>' || v == '/' || v == '=' || v == '"' || v == '\'' ||
             v == '{' || v == '}' || v == '[' || v == ']' || v == '(' || v == ')' ||
             v == ';' || v == ':' || v == ',' || v == '.') m.lane[0].qa[v] = 5;
    else if (v < 128) m.lane[0].qa[v] = 6;
    else m.lane[0].qa[v] = 7;
  }

  for (u32 c = 0; c < 8; ++c) {
    u32 cnt[256];
    std::fill(cnt, cnt + 256, 1u);
    const char common_letters[] = "etaoinshrdlcumwfgypbvkjxqz";
    for (int i = 0; common_letters[i]; ++i) {
      cnt[static_cast<u8>(common_letters[i])] += 2000u / (1u + static_cast<unsigned>(i));
    }
    cnt[static_cast<u8>(' ')] += 3000u;
    if (c == 0) {
      cnt[static_cast<u8>('t')] += 2000u;
      cnt[static_cast<u8>('a')] += 1500u;
      cnt[static_cast<u8>('i')] += 1000u;
      cnt[static_cast<u8>('s')] += 1000u;
      cnt[static_cast<u8>('<')] += 1000u;
      cnt[static_cast<u8>('"')] += 1000u;
      for (char ch = 'A'; ch <= 'Z'; ++ch) cnt[static_cast<u8>(ch)] += 500u;
      for (char ch = '0'; ch <= '9'; ++ch) cnt[static_cast<u8>(ch)] += 300u;
    } else if (c == 1) {
      for (const char ch : {'n', 't', 'r', 's', 'l', 'd', 'm', 'p', 'c'}) {
        cnt[static_cast<u8>(ch)] += 2500u;
      }
      cnt[static_cast<u8>(' ')] += 2000u;
    } else if (c == 2) {
      for (const char ch : {'e', 'a', 'i', 'o', 'u'}) {
        cnt[static_cast<u8>(ch)] += 4000u;
      }
      cnt[static_cast<u8>(' ')] += 2000u;
    } else if (c == 3) {
      for (char ch = 'a'; ch <= 'z'; ++ch) cnt[static_cast<u8>(ch)] += 500u;
    } else if (c == 4) {
      for (char ch = '0'; ch <= '9'; ++ch) cnt[static_cast<u8>(ch)] += 2000u;
      cnt[static_cast<u8>('.')] += 1000u;
      cnt[static_cast<u8>(',')] += 800u;
      cnt[static_cast<u8>(' ')] += 1500u;
    } else if (c == 5) {
      cnt[static_cast<u8>(' ')] += 3000u;
      cnt[static_cast<u8>('\n')] += 1000u;
      cnt[static_cast<u8>('"')] += 1000u;
    } else if (c == 7) {
      for (int b = 0x80; b <= 0xBF; ++b) cnt[b] += 3000u;
    }
    quantize(cnt, m.freq.data() + c * 256);
  }
}

static std::shared_mutex g_custom_profiles_mutex;
static std::unordered_map<u32, std::shared_ptr<StaticProfileHolder>> g_custom_profiles;

static std::shared_ptr<const StaticProfileHolder> get_static_profile_holder(u32 profile_id) {
  static const auto builtins = [] {
    auto p = std::make_shared<std::array<StaticProfileHolder, 5>>();
    init_float_kinematics((*p)[1]);
    finalize_profile((*p)[1]);
    init_poisson_counts((*p)[2]);
    finalize_profile((*p)[2]);
    init_int_monotonic((*p)[3]);
    finalize_profile((*p)[3]);
    init_text((*p)[4]);
    finalize_profile((*p)[4]);
    return p;
  }();

  if (profile_id >= 1 && profile_id <= 4) {
    return std::shared_ptr<const StaticProfileHolder>(builtins, &(*builtins)[profile_id]);
  }
  if (profile_id >= 5) {
    std::shared_lock<std::shared_mutex> lock(g_custom_profiles_mutex);
    auto it = g_custom_profiles.find(profile_id);
    if (it != g_custom_profiles.end()) return it->second;
  }
  return nullptr;
}

// Bits of one coded symbol with frequency f out of kM.
static const float* symbol_bits_table() {
  static const std::vector<float> t = [] {
    std::vector<float> v(kM + 1);
    v[0] = 0.f;
    for (u32 f = 1; f <= kM; ++f)
      v[f] = static_cast<float>(kPrec - std::log2(static_cast<double>(f)));
    return v;
  }();
  return t.data();
}

// Leading byte of a profile's model bytes: bits 0-4 the lattice stride
// (1..16), bits 5-7 the element transform (kProfileTransforms index).
constexpr Filter kProfileTransforms[7] = {
    Filter::None,          Filter::Delta16,       Filter::Delta32,      Filter::Delta64,
    Filter::ZigzagDelta16, Filter::ZigzagDelta32, Filter::ZigzagDelta64};

// Element width of a profile transform, 0 for none.
static usize profile_transform_width(Filter f) {
  switch (f) {
    case Filter::Delta16: case Filter::ZigzagDelta16: return 2;
    case Filter::Delta32: case Filter::ZigzagDelta32: return 4;
    case Filter::Delta64: case Filter::ZigzagDelta64: return 8;
    default: return 0;
  }
}

static u8 profile_head_byte(int stride, Filter transform) {
  u8 code = 0;
  for (u8 i = 1; i < 7; ++i)
    if (kProfileTransforms[i] == transform) code = i;
  return static_cast<u8>(stride | (code << 5));
}

// Coded size of a sample under a trained profile: the exact sum of
// -log2 p over the sample (idx as filled by train_model_from_sample).
static double profile_sample_bits(const Model& m, const std::vector<u32>& idx) {
  const float* sbits = symbol_bits_table();
  double bits = 0;
  for (const u32 i : idx) bits += sbits[m.freq[i]];
  return bits;
}

// The transform has to pay for its decode pass (one prefix sum over the
// page, cheap next to the entropy decoder) with at least this fraction of
// the sample's coded size, except on a drifting sample (below).
constexpr double kProfileTransformGain = 0.02;

// A column drifts when the byte distribution of some lane changes along
// the sample: the two halves of the sample share at most half of a lane's
// mass (total variation distance >= 1/2). Offsets, timestamps and event
// numbers move their high lanes this way; floats, counts and small signed
// integers keep the same lane histograms throughout.
static bool sample_drifts(const u8* sample, usize n, usize S) {
  const usize nel = n / S;
  if (nel < 64) return false;
  const usize half = nel / 2;
  std::vector<u32> h1(S * 256, 0), h2(S * 256, 0);
  for (usize e = 0; e < half; ++e)
    for (usize l = 0; l < S; ++l) ++h1[l * 256 + sample[e * S + l]];
  for (usize e = half; e < 2 * half; ++e)
    for (usize l = 0; l < S; ++l) ++h2[l * 256 + sample[e * S + l]];
  for (usize l = 0; l < S; ++l) {
    usize diff = 0;
    for (int s = 0; s < 256; ++s) {
      const u32 a = h1[l * 256 + s], b = h2[l * 256 + s];
      diff += a > b ? a - b : b - a;
    }
    if (diff >= half) return true;  // TV = diff / (2 * half) >= 1/2
  }
  return false;
}

// Trains the profile for a column sample of W-byte elements: the model
// without transform, and for W in {2, 4, 8} the delta and zig-zag delta
// candidates that an order-0 per-lane estimate on the transformed sample
// rates below the plain sample (floats and small counts fail this gate and
// cost one training only). The choice between the trained candidates is by
// the exact coded size of the sample, with one exception: on a drifting
// sample (sample_drifts) the plain model's sample cost is not predictive,
// since it is reached with bases relative to the previous record and
// contexts on byte values that say nothing about later pages; there the
// transform is taken when it costs at most kProfileTransformGain more on
// the sample.
static Filter train_profile(const u8* sample, usize n, int stride, int level, Model& m) {
  std::vector<u32> idx;
  train_model_from_sample(sample, n, stride, level, m, &idx, true);
  if (stride != 2 && stride != 4 && stride != 8) return Filter::None;

  const usize S = static_cast<usize>(stride);
  const float* xl = xlogx().t.data();
  auto lane_cost = [&](const u8* d) {
    std::vector<u32> cnt(S * 256, 0);
    for (usize i = 0; i < n; ++i) ++cnt[(i % S) * 256 + d[i]];
    double c = 0;
    for (usize l = 0; l < S; ++l) c += table_cost(cnt.data() + l * 256, 256, xl);
    return c;
  };
  const double plain_est = lane_cost(sample);
  const bool drifting = sample_drifts(sample, n, S);
  const Filter cand[2] = {stride == 2 ? Filter::Delta16 : stride == 4 ? Filter::Delta32 : Filter::Delta64,
                          stride == 2   ? Filter::ZigzagDelta16
                          : stride == 4 ? Filter::ZigzagDelta32
                                        : Filter::ZigzagDelta64};

  const double plain_bits = profile_sample_bits(m, idx);
  double best_bits = plain_bits;
  Filter best = Filter::None;
  std::vector<u8> xbuf(n);
  for (const Filter f : cand) {
    apply_filter(f, sample, n, xbuf.data());
    std::memset(xbuf.data(), 0, S);  // the first element travels raw, see build_crystal_profile
    if (lane_cost(xbuf.data()) >= plain_est) continue;
    Model mx;
    train_model_from_sample(xbuf.data(), n, stride, level, mx, &idx, true);
    const double bits = profile_sample_bits(mx, idx);
    double bar = best_bits;
    if (best == Filter::None)
      bar = plain_bits * (drifting ? 1.0 + kProfileTransformGain : 1.0 - kProfileTransformGain);
    if (bits < bar) {
      best_bits = bits;
      best = f;
      m = std::move(mx);
    }
  }
  return best;
}

bool train_and_register_profile(u32 profile_id, const u8* sample, usize n, int stride, int level) {
  if (profile_id < 5 || stride < 1 || stride > kMaxLanes || n < static_cast<usize>(stride))
    return false;
  auto h = std::make_shared<StaticProfileHolder>();
  h->transform = train_profile(sample, n, stride, level, h->model);
  finalize_profile(*h);

  std::unique_lock<std::shared_mutex> lock(g_custom_profiles_mutex);
  auto it = g_custom_profiles.find(profile_id);
  if (it != g_custom_profiles.end()) return true;
  g_custom_profiles[profile_id] = std::move(h);
  return true;
}

bool train_profile_model(const u8* sample, usize n, int stride, std::vector<u8>& out, int level) {
  out.clear();
  if (stride < 1 || stride > kMaxLanes || n < static_cast<usize>(stride) || !sample)
    return false;
  Model m;
  const Filter transform = train_profile(sample, n, stride, level, m);
  out.push_back(profile_head_byte(m.stride, transform));
  write_model(m, out);
  return true;
}

bool register_profile_model(u32 profile_id, const u8* model_bytes, usize model_len) {
  if (profile_id < 5 || !model_bytes || model_len < 2)
    return false;
  const int stride = model_bytes[0] & 31;
  const unsigned transform_code = model_bytes[0] >> 5;
  if (stride < 1 || stride > kMaxLanes || transform_code >= 7)
    return false;

  {
    std::shared_lock<std::shared_mutex> lock(g_custom_profiles_mutex);
    if (g_custom_profiles.find(profile_id) != g_custom_profiles.end())
      return true;
  }

  auto h = std::make_shared<StaticProfileHolder>();
  h->model.stride = stride;
  h->transform = kProfileTransforms[transform_code];
  if (!read_model(h->model, model_bytes + 1, model_len - 1))
    return false;
  finalize_profile(*h);

  std::unique_lock<std::shared_mutex> lock(g_custom_profiles_mutex);
  auto it = g_custom_profiles.find(profile_id);
  if (it != g_custom_profiles.end()) return true;
  g_custom_profiles[profile_id] = std::move(h);
  return true;
}

bool export_profile_model(u32 profile_id, std::vector<u8>& out) {
  out.clear();
  auto h = get_static_profile_holder(profile_id);
  if (!h) return false;
  out.push_back(profile_head_byte(h->model.stride, h->transform));
  write_model(h->model, out);
  return true;
}

// Per-thread buffers of build_crystal_profile. `cnt` is nctx * 256 and is
// zero between calls (each call clears the contexts it touched).
struct ProfileScratch {
  std::vector<u32> idx;
  std::vector<u8> buf;
  std::vector<u8> xbuf;  // the page after the profile's transform
  std::vector<u32> cnt;
  std::vector<u32> ctot;  // per-context symbol count
};

static ProfileScratch& profile_scratch() {
  thread_local ProfileScratch s;
  return s;
}

// A profile block is skipped (false) when the page is estimated to be
// smaller as a per-page (dynamic) block: the profile's cost is exact
// (sum of -log2 p over the page under the profile's tables), the dynamic
// cost is the trainer's own estimate for the same contexts (per-context
// entropy plus table headers), so the dynamic coder can only do better
// than the estimate by choosing other contexts. The margin covers the
// decoder's preference for header-free blocks.
constexpr double kProfileEscapeMargin = 0.03;

// Lane modes of a profile page (block kind kCryProfileLanes). The profile
// codes every lane, since it serves pages its sample never saw; the page
// itself may still have lanes the coder should skip:
//  - a lane constant over the page (the high bytes of small values, a
//    delta-transformed counter) is one value byte;
//  - a lane the profile codes at >= 8 bits per byte, or one that meets a
//    symbol outside the profile's support (frequency 0), goes raw. Before
//    this the second case sent the whole page to the dynamic path with its
//    per-page model header.
// Either lane leaves the rANS loop on both sides (section 3 instructions
// per byte instead of ~30 in the decoder). The modes travel in the block
// header: 2 bits per lane, then one value byte per constant lane, then the
// raw lanes' bytes lane by lane (as in a dynamic block).
inline usize lane_mode_bytes(usize S) noexcept { return (2 * S + 7) / 8; }

void write_lane_modes(const LaneModes& lm, usize S, ByteWriter& w) {
  u8 packed[(2 * kMaxLanes + 7) / 8] = {};
  for (usize l = 0; l < S; ++l) packed[l / 4] |= static_cast<u8>(lm.mode[l] << (2 * (l % 4)));
  w.bytes(packed, lane_mode_bytes(S));
  for (usize l = 0; l < S; ++l)
    if (lm.mode[l] == kLaneConst) w.byte(lm.value[l]);
}

bool read_lane_modes(LaneModes& lm, usize S, ByteReader& r) {
  const u8* packed = r.take(lane_mode_bytes(S));
  if (!packed) return false;
  for (usize l = 0; l < S; ++l) {
    const u32 mode = (packed[l / 4] >> (2 * (l % 4))) & 3u;
    if (mode > kLaneRaw) return false;
    lm.mode[l] = static_cast<u8>(mode);
  }
  for (usize l = 0; l < S; ++l)
    if (lm.mode[l] == kLaneConst) {
      lm.value[l] = r.byte();
      if (r.error) return false;
    }
  return true;
}

// Whether lane l of the block is one value, and which.
inline bool lane_constant(const u8* src, usize n, usize S, usize l, u8& value) noexcept {
  const usize cnt = lane_count(n, S, l);
  if (!cnt) return false;
  const u8* p = src + l;
  value = p[0];
  for (usize i = 1; i < cnt; ++i)
    if (p[i * S] != value) return false;
  return true;
}

bool build_crystal_profile(const u8* src, usize n, u32 profile_id,
                          std::vector<u8>& payload, bool& lane_modes) {
  lane_modes = false;
  auto h = get_static_profile_holder(profile_id);
  if (!h) return false;
  const Model& m = h->model;
  const int stride = m.stride;
  int k = streams_for(n);
  const usize S = static_cast<usize>(stride);

  usize bounds[kMaxStreams + 1];
  split_streams(n, stride, k, bounds);

  ProfileScratch& sc = profile_scratch();
  // The transformed page. Its first element is the page's absolute start
  // (the delta against nothing), which the profile cannot know: it travels
  // raw in the block header and is zero for the coder.
  u8 first[8];
  const usize W = profile_transform_width(h->transform);
  if (W) {
    if (sc.xbuf.size() < n) sc.xbuf.resize(n);
    apply_filter(h->transform, src, n, sc.xbuf.data());
    if (n >= W) {
      std::memcpy(first, sc.xbuf.data(), W);
      std::memset(sc.xbuf.data(), 0, W);
    }
    src = sc.xbuf.data();
  }
  if (sc.idx.size() < n) sc.idx.resize(n);
  const usize ncnt = static_cast<usize>(m.nctx) * 256;
  if (sc.cnt.size() < ncnt) sc.cnt.resize(ncnt, 0);
  if (sc.ctot.size() < m.nctx) sc.ctot.resize(m.nctx, 0);
  u32* const idx = sc.idx.data();
  u32* const cnt = sc.cnt.data();
  u32* const ctot = sc.ctot.data();
  index_block(m, src, bounds, k, idx, cnt, ctot);

  // Lane modes of the page. Constant lanes are known from the bytes; a raw
  // lane needs the page's cost under the profile, per lane: the sum over
  // the lane's contexts of -log2 p of its bytes, infinite for a symbol
  // the profile cannot code.
  LaneModes lm;
  for (usize l = 0; l < S; ++l) {
    if (m.lane[l].mode != kLaneCoded) {  // a lane the model itself skips
      lm.mode[l] = m.lane[l].mode;
      lm.value[l] = m.lane[l].value;
    } else if (lane_constant(src, n, S, l, lm.value[l])) {
      lm.mode[l] = kLaneConst;
    }
  }

  // Page cost under the profile against the dynamic estimate (see
  // kProfileEscapeMargin); this pass also returns the scratch counts to
  // zero. Both sides see the same lane modes: a constant lane costs its
  // value byte on either, a raw lane its bytes under the profile and the
  // smaller of its bytes and the trainer's estimate under a per-page
  // model (which may still compress what the profile's tables cannot).
  const float* sbits = symbol_bits_table();
  const float* xl = xlogx().t.data();
  double profile_bits = 0;
  double dynamic_bits = 8.0 * 2 + kTableBits * S;
  for (usize l = 0; l < S; ++l) {
    const LaneModel& lane = m.lane[l];
    double bits = 0, est = 0;
    bool codable = true;
    for (u32 c = lane.base; c < lane.base + static_cast<u32>(lane.nctx()); ++c) {
      if (!ctot[c]) continue;
      ctot[c] = 0;
      u32* hc = cnt + static_cast<usize>(c) * 256;
      const u16* fc = m.freq.data() + static_cast<usize>(c) * 256;
      est += table_cost(hc, 256, xl);
      for (int t = 0; t < 256; ++t) {
        if (!hc[t]) continue;
        if (!fc[t]) codable = false;
        bits += static_cast<double>(hc[t]) * sbits[fc[t]];
        hc[t] = 0;
      }
    }
    const double raw_bits = 8.0 * static_cast<double>(lane_count(n, S, l));
    if (lm.mode[l] == kLaneConst) {
      profile_bits += 8;
      dynamic_bits += 8;
    } else if (lm.mode[l] == kLaneRaw) {
      profile_bits += raw_bits;
      dynamic_bits += raw_bits;
    } else if (!codable || bits >= raw_bits * (1.0 - kRawLaneSlack)) {
      lm.mode[l] = kLaneRaw;
      profile_bits += raw_bits;
      dynamic_bits += min_of(est, raw_bits);
    } else {
      profile_bits += bits;
      dynamic_bits += est;
    }
  }
  const bool with_modes = !all_coded(lm, S);
  if (with_modes) profile_bits += 8.0 * static_cast<double>(lane_mode_bytes(S));
  if (profile_bits > dynamic_bits * (1.0 + kProfileEscapeMargin)) return false;
  lane_modes = with_modes;

  // No coded lane: no streams (k = 0).
  bool any_coded = false;
  for (usize l = 0; l < S; ++l) any_coded |= lm.mode[l] == kLaneCoded;
  if (!any_coded) k = 0;

  if (profile_id < 255) {
    payload.push_back(static_cast<u8>(profile_id));
  } else {
    payload.push_back(255);
    ByteWriter w_id(payload);
    w_id.varint(profile_id - 255);
  }
  payload.push_back(static_cast<u8>(k));
  ByteWriter w(payload);
  if (W && n >= W) w.bytes(first, W);
  if (lane_modes) {
    write_lane_modes(lm, S, w);
    write_raw_section(lm, S, src, n, payload);
    if (!any_coded) return true;
    // The skipped lanes' bytes take the sentinel row (see finalize_profile).
    const u32 sentinel = m.nctx * 256;
    for (usize l = 0; l < S; ++l) {
      if (lm.mode[l] == kLaneCoded) continue;
      for (usize i = l; i < n; i += S) idx[i] = sentinel;
    }
  }

  const auto& esym = h->esym;
  const usize buf_size = n * 2 + static_cast<usize>(k) * 8;
  if (sc.buf.size() < buf_size) sc.buf.resize(buf_size);
  u8* const buf = sc.buf.data();
  const u8* stream_ptr[kMaxStreams];
  usize stream_len[kMaxStreams];
  encode_streams(esym.data(), idx, bounds, k, buf, stream_ptr, stream_len);
  for (int s = 0; s + 1 < k; ++s) w.varint(stream_len[s]);
  for (int s = 0; s < k; ++s) w.bytes(stream_ptr[s], stream_len[s]);
  return true;
}

Status decode_crystal_profile(const u8* data, usize len, u8* dst, usize n, bool lane_modes) {
  ByteReader r(data, len);
  u32 profile_id = r.byte();
  if (profile_id == 255) {
    profile_id = 255 + static_cast<u32>(r.varint());
  }
  const int k = r.byte();
  if (r.error || k < (lane_modes ? 0 : 1) || k > kMaxStreams) return Status::CorruptInput;
  auto h = get_static_profile_holder(profile_id);
  if (!h) return Status::CorruptInput;
  const usize W = profile_transform_width(h->transform);
  const u8* first = nullptr;
  if (W && n >= W) {
    first = r.take(W);
    if (r.error) return Status::CorruptInput;
  }
  Status st;
  if (!lane_modes) {
    st = decode_crystal_streams(h->model, h->compact, h->dec_tab.data(), h->lanes.data(),
                                h->trend, h->req, k, r, dst, n);
  } else {
    // The page's lane modes override the profile's descriptors: the
    // skipped lanes are filled before the rANS pass and the lockstep loop
    // only shifts their bytes into the history (DecLane::mv).
    const Model& m = h->model;
    const usize S = static_cast<usize>(m.stride);
    LaneModes lm;
    if (!read_lane_modes(lm, S, r)) return Status::CorruptInput;
    const usize raw_len = raw_section_size(lm, S, n);
    const u8* raw = nullptr;
    if (raw_len) {
      raw = r.take(raw_len);
      if (!raw) return Status::CorruptInput;
    }
    fill_skipped_lanes(lm, S, raw, dst, n);
    bool any_coded = false;
    DecLane lanes[kMaxLanes];
    bool trend = false, req = false;
    for (usize l = 0; l < S; ++l) {
      lanes[l] = h->lanes[l];
      if (lm.mode[l] != kLaneCoded) {
        lanes[l].mv = static_cast<u32>(lm.mode[l]) | (static_cast<u32>(lm.value[l]) << 8);
        continue;
      }
      any_coded = true;
      const LaneModel& lane = m.lane[l];
      trend |= lane.fa == kTrend || lane.fb == kTrend || lane.rbase == kTrend;
      req |= lane.fa == kRecEq || lane.fb == kRecEq;
    }
    if (!any_coded) {
      st = k == 0 ? Status::Ok : Status::CorruptInput;
    } else if (k == 0) {
      st = Status::CorruptInput;
    } else {
      st = decode_crystal_streams(m, h->compact, h->dec_tab.data(), lanes, trend, req, k, r, dst, n);
    }
  }
  if (st != Status::Ok || !W) return st;
  // The element deltas invert in place: every element reads only itself
  // and the running sum.
  if (first) std::memcpy(dst, first, W);
  invert_filter(h->transform, dst, n, dst);
  return st;
}

}  // namespace spectra8::detail
