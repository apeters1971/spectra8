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

#include "crystal/crystal.hpp"
#include "crystal/crystal_lstm.hpp"
#include "util/stats.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define CRY_NEON 1
#elif defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <tmmintrin.h>
#define CRY_X86_MIX 1
#endif


namespace spectra8::detail {
namespace {

constexpr u8 kCryVer = 1;
constexpr u8 kCryStored = 0;
constexpr u8 kCryCm = 1;      // bitwise CM
constexpr u8 kCryCmFlag = 2;  // bitwise CM behind a per-byte "expected byte" flag
constexpr u8 kCryStatic = 3;  // static lattice coder (fast decode)
constexpr u8 kCryBwt = 4;     // BWT + MTF/RLE front end, static coder back end
constexpr u8 kCryProfile = 5; // static pre-trained probability profile (zero per-block header tax)
constexpr u8 kCryProfileLanes = 6;  // profile block with per-page lane modes (constant / raw lanes)

constexpr int kMaxStreams = 8;
constexpr int kMatchMinLen = 4;
constexpr int kMatchMaxLen = 63;
constexpr int kMatchBypassThreshold = 48;

// ---------------------------------------------------------------------------
// Logistic helpers (12-bit probabilities, stretch domain +-2047 = 8 fractional
// bits of ln(p/(1-p))).
// ---------------------------------------------------------------------------
inline int squash(int d) noexcept {
  static constexpr int t[33] = {1,    2,    3,    6,    10,   16,   27,
                                45,   73,   120,  194,  310,  488,  747,
                                1101, 1546, 2047, 2549, 2994, 3348, 3607,
                                3785, 3901, 3975, 4024, 4050, 4068, 4079,
                                4085, 4089, 4092, 4093, 4094};
  if (d > 2047) return 4095;
  if (d < -2047) return 0;
  const int w = d & 127;
  d = (d >> 7) + 16;
  return (t[d] * (128 - w) + t[d + 1] * w + 64) >> 7;
}

// Direct squash table over the clamped stretch domain (bit-exact with squash()).
struct SquashTable {
  std::int16_t t[4096];
  SquashTable() {
    for (int d = -2048; d < 2048; ++d) t[d + 2048] = static_cast<std::int16_t>(squash(d));
  }
};

const SquashTable& squash_table() {
  static const SquashTable tab;
  return tab;
}

struct StretchTable {
  std::int16_t t[4096];
  StretchTable() {
    int pi = 0;
    for (int x = -2047; x <= 2047; ++x) {
      const int v = squash(x);
      for (int j = pi; j <= v; ++j) t[j] = static_cast<std::int16_t>(x);
      pi = v + 1;
    }
    for (int j = pi; j < 4096; ++j) t[j] = 2047;
  }
};

const StretchTable& stretch_table() {
  static const StretchTable tab;
  return tab;
}

// Reciprocal table for the adaptive-rate counters: dt[n] ~ 1/(n+1.5).
struct RateTable {
  int dt[1024];
  RateTable() {
    for (int i = 0; i < 1024; ++i) dt[i] = 16384 / (i + i + 3);
  }
};

const RateTable& rate_table() {
  static const RateTable tab;
  return tab;
}

// Counter slot: high 22 bits probability of a 1, low 10 bits hit count.
constexpr u32 kSlotInit = u32{1} << 31;

inline void slot_update(u32& e, int y, int limit, const int* dt) noexcept {
  const u32 v = e;
  const int n = static_cast<int>(v & 1023);
  const int p = static_cast<int>(v >> 10);
  // |delta| < 2^19 and dt < 2^13, so the product fits 32 bits: the u32 wrap
  // of a negative step is exactly the two's-complement add we want.
  const u32 step = static_cast<u32>(((y << 22) - p) >> 3) * static_cast<u32>(dt[n]);
  e = v + static_cast<u32>(n < limit) + (step & 0xFFFFFC00u);
}

// ---------------------------------------------------------------------------
// Mixer arithmetic: 16-bit weights (Q14) and 16-bit stretch inputs. The NEON
// and scalar paths are bit-exact so streams are portable across CPUs.
// ---------------------------------------------------------------------------
using i16 = std::int16_t;

// Scalar on purpose: the inputs were just produced by scalar stores and a
// wide vector reload right after would stall on store forwarding.
template <int N>
inline i32 mix_dot(const i16* w, const i16* in) noexcept {
  static_assert(N % 4 == 0);
  i32 s = 0;
  for (int i = 0; i < N; ++i) s += static_cast<i32>(w[i]) * in[i];
  return s;
}

// w += round((in * err * 2) >> 16), saturating.
// NEON is vqrdmulh + vqadd. On x86 the same rounding is pmulhrsw + paddsw
// (checked against the scalar formula for every int16 pair). The caller
// clamps err to +/-32767, so pmulhrsw's saturating corner (both lanes
// -32768) is not reached and the two paths stay bit-exact. A baseline
// x86-64 build (no -mssse3) picks the SSSE3 body at run time.
#if defined(CRY_X86_MIX)
template <int N>
#if !defined(__SSSE3__)
__attribute__((target("ssse3")))
#endif
inline void mix_train_ssse3(i16* w, const i16* in, int err) noexcept {
  const __m128i e = _mm_set1_epi16(static_cast<i16>(err));
  int j = 0;
  for (; j + 8 <= N; j += 8) {
    const __m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(in + j));
    __m128i a = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w + j));
    a = _mm_adds_epi16(a, _mm_mulhrs_epi16(b, e));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(w + j), a);
  }
  for (; j < N; ++j) {
    const i32 t = (static_cast<i32>(in[j]) * err * 2 + 32768) >> 16;
    i32 v = static_cast<i32>(w[j]) + t;
    if (v > 32767) v = 32767;
    if (v < -32768) v = -32768;
    w[j] = static_cast<i16>(v);
  }
}

inline bool cry_have_ssse3() noexcept {
#if defined(__SSSE3__)
  return true;
#else
  static const bool have = __builtin_cpu_supports("ssse3");
  return have;
#endif
}
#endif

template <int N>
inline void mix_train(i16* w, const i16* in, int err) noexcept {
  static_assert(N % 4 == 0);
#if defined(CRY_NEON)
  const int16x8_t e = vdupq_n_s16(static_cast<i16>(err));
  int j = 0;
  for (; j + 8 <= N; j += 8) {
    const int16x8_t b = vld1q_s16(in + j);
    int16x8_t a = vld1q_s16(w + j);
    a = vqaddq_s16(a, vqrdmulhq_s16(b, e));
    vst1q_s16(w + j, a);
  }
  if (j < N) {
    const int16x4_t b = vld1_s16(in + j);
    int16x4_t a = vld1_s16(w + j);
    a = vqadd_s16(a, vqrdmulh_s16(b, vget_low_s16(e)));
    vst1_s16(w + j, a);
  }
#elif defined(CRY_X86_MIX)
  if (cry_have_ssse3()) {
    mix_train_ssse3<N>(w, in, err);
    return;
  }
  for (int i = 0; i < N; ++i) {
    i32 t = (static_cast<i32>(in[i]) * err * 2 + 32768) >> 16;
    if (t > 32767) t = 32767;
    if (t < -32768) t = -32768;
    i32 v = static_cast<i32>(w[i]) + t;
    if (v > 32767) v = 32767;
    if (v < -32768) v = -32768;
    w[i] = static_cast<i16>(v);
  }
#else
  for (int i = 0; i < N; ++i) {
    i32 t = (static_cast<i32>(in[i]) * err * 2 + 32768) >> 16;
    if (t > 32767) t = 32767;
    if (t < -32768) t = -32768;
    i32 v = static_cast<i32>(w[i]) + t;
    if (v > 32767) v = 32767;
    if (v < -32768) v = -32768;
    w[i] = static_cast<i16>(v);
  }
#endif
}

// ---------------------------------------------------------------------------
// Configuration by level.
// ---------------------------------------------------------------------------
struct Cfg {
  int limit;     // counter hit-count horizon
  int lr;        // mixer learning rate (x/16 of the base step)
  int cap_bits;  // max bucket bits for high-order models
};

Cfg cfg_for_tier(int tier) noexcept {
  switch (tier) {
    case 0: return Cfg{255, 2, 16};
    case 1: return Cfg{511, 2, 16};
    default: return Cfg{1023, 2, 17};
  }
}

int tier_for_level(int level) noexcept {
  if (level <= 3) return 0;
  if (level <= 6) return 1;
  return 2;
}

// Number of interleaved streams by level: more streams = more instruction-
// level parallelism in the bit loop. Models are shared, so the ratio cost is
// only the shorter per-stream history for the match model (~1% on text per
// halving, nothing on records, so flag mode always takes 4); the ultra
// levels (10-11) use a single stream for maximum ratio.
int streams_for_level(int level, usize n, bool flag) noexcept {
  int k = flag ? 4 : level > 9 ? 1 : level <= 3 ? 4 : 2;
  while (k > 1 && n / static_cast<usize>(k) < 8192) k >>= 1;
  return k;
}

int ceil_log2(usize n) noexcept {
  int l = 0;
  while ((usize{1} << l) < n) ++l;
  return l;
}

}  // namespace

// Record stride detection (MDL): pick the S in {1..16} minimising the
// per-lane order-0 entropy H(byte | pos mod S) plus the cost of describing
// S lane tables. i.i.d. records land on their element width; byte planes
// and text land on 1.
int detect_stride(const u8* s, usize n) {
  static constexpr int kCand[] = {1, 2, 3, 4, 5, 6, 7, 8, 12, 16};
  const usize lim = min_of(n, usize{1} << 15);  // 32 KiB is plenty
  if (lim < 256) return 1;
  double best = 1e300;
  int best_s = 1;
  // Lane histograms: the ones of S fold exactly into those of S / 2 (lane
  // l of S / 2 is lanes l and l + S / 2 of S), so the chains 16-8-4-2-1
  // and 12-6-3 take one counting pass each, and 5 and 7 one more.
  thread_local std::vector<u32> tabs;
  if (tabs.size() < 17 * 16 * 256) tabs.resize(17 * 16 * 256);  // [S][lane][byte], lane < S
  auto tab = [&](int S) { return tabs.data() + static_cast<usize>(S) * 16 * 256; };
  auto count = [&](int S) {
    u32* c = tab(S);
    const usize us = static_cast<usize>(S);
    std::memset(c, 0, us * 256 * sizeof(u32));
    for (usize i = 0; i < lim; i += us) {
      const usize m = min_of(us, lim - i);
      for (usize l = 0; l < m; ++l) ++c[l * 256 + s[i + l]];
    }
  };
  auto fold = [&](int S) {  // from 2S
    const u32* from = tab(2 * S);
    u32* to = tab(S);
    for (usize l = 0; l < static_cast<usize>(S); ++l)
      for (usize v = 0; v < 256; ++v)
        to[l * 256 + v] = from[l * 256 + v] + from[(l + static_cast<usize>(S)) * 256 + v];
  };
  auto fits = [&](int S) { return static_cast<usize>(S) * 64 <= lim; };
  for (int S = 16; S >= 1; S /= 2)
    if (fits(S)) {
      if (S == 16 || !fits(2 * S)) count(S); else fold(S);
    }
  for (int S = 12; S >= 3; S /= 2)
    if (fits(S)) {
      if (S == 12 || !fits(2 * S)) count(S); else fold(S);
    }
  if (fits(5)) count(5);
  if (fits(7)) count(7);
  for (int S : kCand) {
    if (!fits(S)) break;
    double bits = 0;
    double model = 0;
    for (int l = 0; l < S; ++l) {
      const u32* c = tab(S) + static_cast<usize>(l) * 256;
      double tot = 0;
      int distinct = 0;
      for (int v = 0; v < 256; ++v) {
        tot += c[v];
        distinct += c[v] != 0;
      }
      if (tot <= 0) continue;
      const double lt = std::log2(tot);
      for (int v = 0; v < 256; ++v)
        if (c[v]) bits += c[v] * (lt - std::log2(static_cast<double>(c[v])));
      model += distinct * 0.5 * std::log2(static_cast<double>(lim) / S + 1);
    }
    const double score = bits + model;
    if (score < best * 0.998) {
      best = score;
      best_s = S;
    }
  }
  return best_s;
}

namespace {

// Split [0, n) into k stride-aligned parts.
void split_streams(usize n, int stride, int k, usize bounds[kMaxStreams + 1]) {
  const usize S = static_cast<usize>(stride);
  bounds[0] = 0;
  for (int i = 1; i < k; ++i) {
    usize b = (n * static_cast<usize>(i)) / static_cast<usize>(k);
    b = ((b + S - 1) / S) * S;
    if (b > n) b = n;
    if (b < bounds[i - 1]) b = bounds[i - 1];
    bounds[i] = b;
  }
  bounds[k] = n;
}

// ---------------------------------------------------------------------------
// Binary arithmetic coder (32-bit, carry-less).
// ---------------------------------------------------------------------------
struct ArithEncoder {
  u32 x1 = 0, x2 = 0xFFFFFFFFu;
  std::vector<u8> out;

  // Branch-free on y: the bit value is unpredictable for high-entropy data.
  inline void encode(int y, int p) noexcept {
    const u32 xmid = x1 + ((x2 - x1) >> 12) * static_cast<u32>(p);
    const u32 mask = 0u - static_cast<u32>(y);
    x2 = (xmid & mask) | (x2 & ~mask);
    x1 = ((xmid + 1) & ~mask) | (x1 & mask);
    if (((x1 ^ x2) & 0xFF000000u) == 0) normalize();
  }
  void normalize() noexcept {
    do {
      out.push_back(static_cast<u8>(x2 >> 24));
      x1 <<= 8;
      x2 = (x2 << 8) | 255;
    } while (((x1 ^ x2) & 0xFF000000u) == 0);
  }
  void flush() { out.push_back(static_cast<u8>(x1 >> 24)); }
};

struct ArithDecoder {
  u32 x1 = 0, x2 = 0xFFFFFFFFu, x = 0;
  const u8* p = nullptr;
  const u8* end = nullptr;
  void init(const u8* d, usize n) {
    p = d;
    end = d + n;
    x = 0;
    for (int i = 0; i < 4; ++i) x = (x << 8) | next();
  }
  inline u8 next() noexcept { return p < end ? *p++ : u8{0xFF}; }
  inline int decode(int pr) noexcept {
    const u32 xmid = x1 + ((x2 - x1) >> 12) * static_cast<u32>(pr);
    const u32 y = x <= xmid;
    const u32 mask = 0u - y;
    x2 = (xmid & mask) | (x2 & ~mask);
    x1 = ((xmid + 1) & ~mask) | (x1 & mask);
    if (((x1 ^ x2) & 0xFF000000u) == 0) normalize();
    return static_cast<int>(y);
  }
  void normalize() noexcept {
    do {
      x1 <<= 8;
      x2 = (x2 << 8) | 255;
      x = (x << 8) | next();
    } while (((x1 ^ x2) & 0xFF000000u) == 0);
  }
};

// ---------------------------------------------------------------------------
// Reusable per-thread storage so blocks do not pay for allocation.
// ---------------------------------------------------------------------------
struct Arena {
  std::vector<u32> slots;
  std::vector<i16> mix;
  std::vector<u16> apm;
  std::vector<u32> match_tab;
  std::vector<u32> match_sm;
  std::vector<u32> flag;
  std::vector<i16> flag_mix;
};

Arena& arena() {
  thread_local Arena a;
  return a;
}

// ---------------------------------------------------------------------------
// The shared model plus K interleaved coding streams.
//
// NM hashed context models, optional match model, NMIX mixers, NAPM SSE
// stages. All tables are shared by the streams; each stream carries only its
// position, partial byte, selected slots/weights and coder.
// ---------------------------------------------------------------------------
// FLAG selects the byte-flag mode: each byte is first coded as "equal to the
// expected byte?" and the bitwise stage only runs on a miss (with an
// exclusion input instead of the bitwise match model).
// LSTM adds the experimental cmix-style LSTM byte predictor (crystal_lstm.hpp)
// as two more mixer inputs (and one flag-mixer input in FLAG mode).
template <int NM, bool MATCH, int NMIX, int NAPM, bool FLAG, bool LSTM = false>
class Ensemble {
 public:
  static constexpr bool kFlag = FLAG;
  static constexpr bool kLstm = LSTM;
  static constexpr int NIN = NM + (FLAG ? 1 : (MATCH ? 2 : 0)) + (LSTM ? 2 : 0) + 1;
  static constexpr int NINP = (NIN + 3) & ~3;  // padded to 4 for SIMD
  static constexpr int kApm1Ctx = 256 * 16;
  static constexpr int kApm2Bits = 14;
  // Byte-flag stage ("byte == expected byte?").
  static constexpr int kFlagA = 2048;          // (lane, agree, src, mlen bucket, history)
  static constexpr int kFlagHashBits = 16;
  static constexpr int kFlagSlots = kFlagA + 2 * (1 << kFlagHashBits);
  static constexpr int kFlagMixSets = 256;
  static constexpr int kExSlots = 256 + 1;     // (lane, bit, expected bit) + dummy
  static constexpr int kMatchMargin = 8192;  // of 32768 = 100% hit rate

  struct Stream {
    usize pos = 0, start = 0, end = 0;
    int c0 = 1;
    int bitpos = 0;
    int lane = 0;
    u32 plane = 0;        // pos * 16 / n, maintained incrementally
    usize plane_next = 0; // first pos of the next plane
    u32 rp_hash = 0;
    int pr = 2048;
    int pmix = 2048;
    u32 h[NM];
    u32* base[NM];
    alignas(16) i16 in[NINP];
    i16* wsel[NMIX];
    int apm_idx[2];
    int ex_ctx = kExSlots - 1;
    // Match model (per stream: only matches inside its own range).
    u32 mptr = 0;
    int mlen = 0;
    int mctx = 0;
    int mlen_bucket = 0;
    // Byte flag stage.
    bool in_bits = false;
    u32 expect = 0;
    int fsrc = 0;
    int fpr = 2048;
    int fpmix = 2048;
    u8 fh[16] = {};
    u16 fscore[16][2] = {};  // decayed hit rates of the two candidates per lane
    u32 cand_lat = 0;        // lane candidate (lat1 / c1) of the current byte
    alignas(16) i16 fin[8];
    i16* fwsel = nullptr;
    u32* fslot[3];
    // Experimental LSTM byte predictor (one per stream; owned by the caller).
    LstmByteModel* lstm = nullptr;
  };

  void set_lstm(int i, LstmByteModel* m) noexcept { strm_[i].lstm = m; }

  Ensemble(const Cfg& cfg, usize n, int stride, u8* hist, int nstreams)
      : cfg_(cfg),
        n_(n),
        stride_(stride),
        hist_(hist),
        st_(stretch_table().t),
        sq_(squash_table().t),
        dt_(rate_table().dt),
        k_(nstreams) {
    Arena& a = arena();
    const int nb = ceil_log2(max_of(n, usize{2}));
    usize total = 0;
    for (int i = 0; i < NM; ++i) {
      bits_[i] = model_bits(i, nb);
      total += usize{1} << (bits_[i] + 4);
    }
    if (a.slots.size() < total) a.slots.resize(total);
    std::fill(a.slots.begin(), a.slots.begin() + static_cast<std::ptrdiff_t>(total),
              kSlotInit);
    {
      u32* p = a.slots.data();
      for (int i = 0; i < NM; ++i) {
        tab_[i] = p;
        p += usize{1} << (bits_[i] + 4);
      }
    }
    {
      const usize sets = 512 * NMIX;
      if (a.mix.size() < sets * NINP) a.mix.resize(sets * NINP);
      std::fill(a.mix.begin(), a.mix.begin() + static_cast<std::ptrdiff_t>(sets * NINP),
                static_cast<i16>((1 << 14) / NIN));
      for (int m = 0; m < NMIX; ++m) mix_[m] = a.mix.data() + static_cast<usize>(m) * 512 * NINP;
    }
    if (NAPM > 0) {
      const usize ctxs = kApm1Ctx + (NAPM > 1 ? (usize{1} << kApm2Bits) : 0);
      if (a.apm.size() < ctxs * 33) a.apm.resize(ctxs * 33);
      for (usize c = 0; c < ctxs; ++c)
        for (int j = 0; j < 33; ++j)
          a.apm[c * 33 + j] = static_cast<u16>(squash((j - 16) * 128) * 16);
      apm1_ = a.apm.data();
      apm2_ = a.apm.data() + usize{kApm1Ctx} * 33;
    }
    if (MATCH) {
      int mb = nb;
      if (mb < 10) mb = 10;
      if (mb > 22) mb = 22;
      match_bits_ = mb;
      const usize mt = usize{1} << mb;
      if (a.match_tab.size() < mt) a.match_tab.resize(mt);
      std::fill(a.match_tab.begin(), a.match_tab.begin() + static_cast<std::ptrdiff_t>(mt), 0u);
      match_tab_ = a.match_tab.data();
      if (a.match_sm.size() < 128) a.match_sm.resize(128);
      std::fill(a.match_sm.begin(), a.match_sm.begin() + 128, kSlotInit);
      match_sm_ = a.match_sm.data();
    }
    if (FLAG) {
      const usize fs = kFlagSlots + kExSlots;
      if (a.flag.size() < fs) a.flag.resize(fs);
      std::fill(a.flag.begin(), a.flag.begin() + static_cast<std::ptrdiff_t>(fs), kSlotInit);
      flag_ = a.flag.data();
      ex_ = flag_ + kFlagSlots;
      const usize fm = usize{kFlagMixSets} * 8;
      if (a.flag_mix.size() < fm) a.flag_mix.resize(fm);
      std::fill(a.flag_mix.begin(), a.flag_mix.begin() + static_cast<std::ptrdiff_t>(fm),
                static_cast<i16>((1 << 14) / 4));
      fmix_ = a.flag_mix.data();
    }
    split_streams(n, stride, k_, bounds_);
    for (int s = 0; s < k_; ++s) {
      Stream& S = strm_[s];
      S.start = S.pos = bounds_[s];
      S.end = bounds_[s + 1];
      S.lane = 0;
    }
  }

  int streams() const noexcept { return k_; }
  Stream& stream(int i) noexcept { return strm_[i]; }

  // Byte stage 1: choose the expected byte and predict the flag (s.fpr).
  inline void begin_byte(Stream& s) noexcept {
    const u8* hist = hist_;
    const u32 c1 = back(s, hist, 1);
    const u32 lane = static_cast<u32>(s.lane);
    if (s.lane == 0)
      s.rp_hash = 0;
    else
      s.rp_hash = (s.rp_hash + c1 + 1) * 0x9E3779B1u;

    if (MATCH) {
      if (s.mlen > 0) {
        ++s.mptr;
        if (s.mlen < kMatchMaxLen) ++s.mlen;
      }
      const usize avail = s.pos - s.start;
      if (avail >= static_cast<usize>(kMatchMinLen)) {
        const u32 h = hash64(load_u32(hist + s.pos - 4), 0x1234567u) >> (32 - match_bits_);
        if (s.mlen == 0) {
          // The table is shared by the streams: a candidate is usable once
          // its owner stream has coded it (the streams run in lockstep, so
          // the predicted byte stays behind the owner from then on). The
          // history buffer is zeroed on both sides, so even a stale read
          // is deterministic.
          const u32 cand = match_tab_[h];
          if (cand > 0) {
            int j = 0;
            while (j + 1 < k_ && static_cast<usize>(cand) >= bounds_[j + 1]) ++j;
            const Stream& o = strm_[j];
            if (cand > o.start && static_cast<usize>(cand) + 1 < o.pos) {
              const usize cavail = cand - o.start;
              usize len = 0;
              while (len < static_cast<usize>(kMatchMaxLen) && len < cavail &&
                     len < avail && hist[cand - 1 - len] == hist[s.pos - 1 - len])
                ++len;
              if (len >= static_cast<usize>(kMatchMinLen)) {
                s.mlen = static_cast<int>(len);
                s.mptr = cand;
              }
            }
          }
        }
        match_tab_[h] = static_cast<u32>(s.pos);
      }
    }
    if (!FLAG) {
      s.in_bits = true;
      begin_bits(s);
      return;
    }
    int mlb = 0;
    s.cand_lat = stride_ > 1 ? back(s, hist, static_cast<usize>(stride_)) : c1;
    const u16* sc = s.fscore[lane & 15];
    u32 agree = 0;
    if (MATCH && s.mlen > 0) agree = hist[s.mptr] == s.cand_lat;
    // The match byte replaces the lane candidate only when it is clearly the
    // better guess (text); on records the lane candidate wins and the match
    // enters through the agreement bit.
    if (MATCH && s.mlen > 0 && sc[1] > sc[0] + kMatchMargin) {
      s.expect = hist[s.mptr];
      s.fsrc = 1;
      mlb = s.mlen < 8 ? 1 : s.mlen < 16 ? 2 : 3;
    } else {
      s.expect = s.cand_lat;
      s.fsrc = 0;
    }
    s.mlen_bucket = mlb;
    const u32 fh = s.fh[lane & 15];
    const u32 src = static_cast<u32>(s.fsrc);
    s.fslot[0] = flag_ + ((agree << 10) | ((lane & 15) << 6) | (src << 5) |
                          (static_cast<u32>(mlb) << 3) | (fh & 7));
    s.fslot[1] = flag_ + kFlagA +
                 (hash64(pack(s.expect, lane, src, 1), 0x51u) >> (32 - kFlagHashBits));
    s.fslot[2] = flag_ + kFlagA + (1u << kFlagHashBits) +
                 (hash64(pack(s.expect, c1, lane, src | 2), 0x52u) >> (32 - kFlagHashBits));
    for (int i = 0; i < 3; ++i) s.fin[i] = st_[*s.fslot[i] >> 20];
    s.fin[3] = 256;
    for (int i = 4; i < 8; ++i) s.fin[i] = 0;
    if (LSTM) {
      // P(byte == expected) under the LSTM's byte distribution.
      const float* cum = s.lstm->cum();
      s.fin[4] = st_[prob12(cum[s.expect + 1] - cum[s.expect])];
    }
    s.fwsel = fmix_ + (((agree << 7) | ((lane & 15) << 3) | (src << 2) | (fh & 3)) * 8);
    int d = mix_dot<8>(s.fwsel, s.fin) >> 14;
    if (d > 2047) d = 2047;
    if (d < -2047) d = -2047;
    s.fpmix = sq_[d + 2048];
    s.fpr = s.fpmix < 1 ? 1 : s.fpmix;
  }

  // Byte stage 2: learn the flag; a hit finishes the byte, a miss starts the
  // bitwise stage.
  inline void finish_flag(Stream& s, int hit) noexcept {
    const int limit = cfg_.limit;
    for (int i = 0; i < 3; ++i) slot_update(*s.fslot[i], hit, limit, dt_);
    {
      int err = ((hit << 12) - s.fpmix) * cfg_.lr * 2;
      if (err > 32767) err = 32767;
      if (err < -32767) err = -32767;
      mix_train<8>(s.fwsel, s.fin, err);
    }
    u8& fh = s.fh[s.lane & 15];
    fh = static_cast<u8>(((fh << 1) | hit) & 7);
    if (hit) {
      advance_byte(s, s.expect);
    } else {
      s.in_bits = true;
      begin_bits(s);
    }
  }

  // Candidate scores, once the byte is known.
  inline void score_candidates(Stream& s, u32 byte, int lane) noexcept {
    u16* sc = s.fscore[lane & 15];
    sc[0] = static_cast<u16>(sc[0] - (sc[0] >> 5) + ((byte == s.cand_lat) << 10));
    if (MATCH && s.mlen > 0)
      sc[1] = static_cast<u16>(sc[1] - (sc[1] >> 5) + ((byte == hist_[s.mptr]) << 10));
  }

  // Learn from bit y, advance the stream and predict its next bit.
  inline void update(Stream& s, int y) noexcept {
    if constexpr (!FLAG && MATCH) {
      if (s.mlen >= kMatchBypassThreshold) {
        slot_update(match_sm_[s.mctx], y, 1023, dt_);
        s.c0 = (s.c0 << 1) | y;
        ++s.bitpos;
        if (((hist_[s.mptr] + 256) >> (8 - s.bitpos)) != s.c0) s.mlen = 0;
        if (s.bitpos == 8) {
          s.in_bits = false;
          advance_byte(s, static_cast<u32>(s.c0 & 255));
        } else {
          if (s.bitpos == 4) select_buckets(s, static_cast<u32>(s.c0));
          predict(s);
        }
        return;
      }
    }
    {
      const int idx = slot_index(s.c0, s.bitpos);
      const int limit = cfg_.limit;
      for (int i = 0; i < NM; ++i) slot_update(s.base[i][idx], y, limit, dt_);
    }
    if (FLAG) slot_update(ex_[s.ex_ctx], y, 1023, dt_);  // dummy slot when inactive
    if (!FLAG && MATCH && s.mlen > 0) slot_update(match_sm_[s.mctx], y, 1023, dt_);
    {
      int err = ((y << 12) - s.pmix) * cfg_.lr * 2;
      if (err > 32767) err = 32767;
      if (err < -32767) err = -32767;
      for (int m = 0; m < NMIX; ++m) mix_train<NINP>(s.wsel[m], s.in, err);
    }
    if (NAPM > 0) apm_update(apm1_, s.apm_idx[0], y);
    if (NAPM > 1) apm_update(apm2_, s.apm_idx[1], y);

    s.c0 = (s.c0 << 1) | y;
    ++s.bitpos;
    if (!FLAG && MATCH && s.mlen > 0) {
      if (((hist_[s.mptr] + 256) >> (8 - s.bitpos)) != s.c0) s.mlen = 0;
    }
    if (s.bitpos == 8) {
      s.in_bits = false;
      advance_byte(s, static_cast<u32>(s.c0 & 255));
    } else {
      if (s.bitpos == 4) select_buckets(s, static_cast<u32>(s.c0));
      predict(s);
    }
  }

 private:
  int model_bits(int i, int nb) const noexcept {
    int bits;
    switch (i) {
      case 0: bits = 8; break;
      case 1: bits = 13; break;
      case 2: bits = 16; break;
      default: bits = nb - 2; break;  // ~n/2 buckets of 16 slots
    }
    if (bits > cfg_.cap_bits) bits = cfg_.cap_bits;
    if (bits < 8) bits = 8;
    return bits;
  }

  // Byte k back within the stream's own range (0 beyond its start).
  static inline u32 back(const Stream& s, const u8* hist, usize k) noexcept {
    return s.pos >= s.start + k ? hist[s.pos - k] : 0u;
  }

  static inline u32 hash64(u64 v, u32 salt) noexcept {
    u64 h = (v + salt) * 0x9E3779B97F4A7C15ull;
    h ^= h >> 29;
    h *= 0xBF58476D1CE4E5B9ull;
    return static_cast<u32>(h >> 32);
  }

  static inline u64 pack(u32 a, u32 b, u32 c = 0, u32 d = 0) noexcept {
    return (static_cast<u64>(a)) | (static_cast<u64>(b) << 16) |
           (static_cast<u64>(c) << 32) | (static_cast<u64>(d) << 48);
  }

  inline void advance_byte(Stream& s, u32 byte) noexcept {
    hist_[s.pos] = static_cast<u8>(byte);
    if (FLAG) {
      score_candidates(s, byte, s.lane);
      if (MATCH && s.mlen > 0 && hist_[s.mptr] != byte) s.mlen = 0;
    }
    ++s.pos;
    if (++s.lane == stride_) s.lane = 0;
    if (LSTM) s.lstm->observe(static_cast<int>(byte), s.lane);
  }

  // Float probability -> 12-bit, clamped to the open interval.
  static inline int prob12(float p) noexcept {
    int v = static_cast<int>(p * 4096.f + 0.5f);
    return v < 1 ? 1 : v > 4095 ? 4095 : v;
  }

  // P(next bit = 1 | partial byte c0 at bit bp) from the LSTM byte
  // distribution's prefix sums; `mass` receives the probability of the
  // partial byte itself (how much the LSTM believed in the prefix so far).
  static inline int lstm_bit_p(const float* cum, int c0, int bp, float& mass) noexcept {
    const int lo = (c0 - (1 << bp)) << (8 - bp);
    const int half = 1 << (7 - bp);
    const float a = cum[lo], m = cum[lo + half], b = cum[lo + 2 * half];
    const float tot = b - a;
    mass = tot;
    if (tot <= 1e-12f) return 2048;
    return prob12((b - m) / tot);
  }

  inline void begin_bits(Stream& s) noexcept {
    s.c0 = 1;
    s.bitpos = 0;
    const usize S = static_cast<usize>(stride_);
    const u8* hist = hist_;
    const u32 c1 = back(s, hist, 1), c2 = back(s, hist, 2), c3 = back(s, hist, 3),
              c4 = back(s, hist, 4);
    const u32 lat1 = back(s, hist, S), lat2 = back(s, hist, 2 * S);
    const u32 lane = static_cast<u32>(s.lane);
    while (s.pos >= s.plane_next) {
      ++s.plane;
      s.plane_next = (static_cast<usize>(s.plane) * n_ + 15) / 16;
    }
    const u32 plane = s.plane - 1;

    // Record prefix: bytes of the current record coded so far (maintained in
    // begin_byte). For i.i.d. records this is the only context that carries
    // information; bytes of the previous record are noise.
    const u32 rp1 = s.lane >= 1 ? c1 : 0x100u;
    const u32 rp2 = s.lane >= 2 ? c2 : 0x100u;

    // Context sets are ordered by value so small ensembles keep the best.
    u64 ctx[12];
    if (stride_ > 1) {
      ctx[0] = pack(lane, 0);
      ctx[1] = pack(rp1, lane, 1);
      ctx[2] = pack(rp1, rp2, lane, 2);
      ctx[3] = pack(s.rp_hash & 0xFFFF, s.rp_hash >> 16, lane, 3);
      ctx[4] = pack(lat1, lane, 4);
      ctx[5] = pack(c1, c2, c3, 5);
      if (NM > 6) {
        const u32 c56 = (back(s, hist, 5) << 8) | back(s, hist, 6);
        ctx[6] = pack(c1, c2, c3, c4) ^ 0x6000000000000000ull;
        ctx[7] = pack(c1, c2, c3, c4) ^ (static_cast<u64>(c56) << 40) ^ 0x7;
        ctx[8] = pack(lat1, lat2, lane, 8);
        ctx[9] = pack(lat1, rp1, lane, 9);
        ctx[10] = pack((2 * lat1 - lat2) & 0xFF, rp1, lane, 10);
        ctx[11] = pack(c1, 0, 0, 11);
      }
    } else {
      ctx[0] = pack(plane, 0);
      ctx[1] = pack(c1, 0, 0, 1);
      ctx[2] = pack(c1, c2, 0, 2);
      ctx[3] = pack(c1, c2, c3, 3);
      ctx[4] = pack(c1, c2, c3, c4) ^ 0x4000000000000000ull;
      {
        const u32 c56 = (back(s, hist, 5) << 8) | back(s, hist, 6);
        ctx[5] = pack(c1, c2, c3, c4) ^ (static_cast<u64>(c56) << 40) ^ 0x5;
        if (NM > 6) {
          ctx[6] = pack(c1, c3, 0, 6);
          ctx[7] = pack(c1 >> 2, c2 >> 2, c3 >> 2, (c4 >> 2) | 0x7000);
          ctx[8] = pack(c2, c3, 0, 8);
          ctx[9] = pack(c1, c2, plane, 9);
          ctx[10] = pack(c1, plane, 0, 10);
          ctx[11] = pack(c2, c4, c56 & 0xFF, 11);
        }
      }
    }
    for (int i = 0; i < NM; ++i) s.h[i] = hash64(ctx[i], static_cast<u32>(i) * 0x9E3779B1u);
    select_buckets(s, 0);
    predict(s);
  }

  inline void select_buckets(Stream& s, u32 hi) noexcept {
    for (int i = 0; i < NM; ++i) {
      u32 b = s.h[i] ^ (hi * 0x2F0B4F1Bu);
      b *= 0x9E3779B1u;
      b >>= (32 - bits_[i]);
      s.base[i] = tab_[i] + (static_cast<usize>(b) << 4);
      __builtin_prefetch(s.base[i]);
    }
  }

  static inline int apm_pp(u16* t, u32 cx, int pr, const std::int16_t* st,
                           int& index) noexcept {
    const int sv = (st[pr] + 2048) * 32;
    const int wt = sv & 0xFFF;
    const usize base = static_cast<usize>(cx) * 33 + static_cast<usize>(sv >> 12);
    index = static_cast<int>(base) + (wt >> 11);
    return (t[base] * (4096 - wt) + t[base + 1] * wt) >> 16;
  }

  static inline void apm_update(u16* t, int index, int y) noexcept {
    constexpr int rate = 7;
    const int g = (y << 16) + (y << rate) - y - y;
    t[index] = static_cast<u16>(t[index] + ((g - t[index]) >> rate));
  }

  // Slot within the 16-entry nibble bucket: the partial nibble with a
  // leading 1 (1..15).
  static inline int slot_index(int c0, int bp) noexcept {
    return bp < 4 ? c0 : ((1 << (bp - 4)) | (c0 & ((1 << (bp - 4)) - 1)));
  }

  inline void predict(Stream& s) noexcept {
    const int bp = s.bitpos;
    if constexpr (!FLAG && MATCH) {
      if (s.mlen >= kMatchBypassThreshold) {
        const int expected = (hist_[s.mptr] >> (7 - bp)) & 1;
        s.mctx = s.mlen * 2 + expected;
        int pr = static_cast<int>(match_sm_[s.mctx] >> 20);
        if (pr < 1) pr = 1;
        if (pr > 4095) pr = 4095;
        s.pmix = pr;
        s.pr = pr;
        return;
      }
    }
    const int slot_idx = slot_index(s.c0, bp);
    // Mixer 0 by (lane, bit position); mixer 1 by (lane, bit, flag history).
    // The dot product is accumulated as the inputs are produced so the
    // coder does not wait on the in[] store -> reload.
    const i16* w0 = mix_[0] + static_cast<usize>(((s.lane & 15) << 3) | bp) * NINP;
    s.wsel[0] = const_cast<i16*>(w0);
    i32 acc0 = 0;
    for (int i = 0; i < NM; ++i) {
      const i16 v = st_[s.base[i][slot_idx] >> 20];
      s.in[i] = v;
      acc0 += static_cast<i32>(w0[i]) * v;
    }
    int k = NM;
    if (!FLAG && MATCH) {
      if (s.mlen > 0) {
        const int expected = (hist_[s.mptr] >> (7 - bp)) & 1;
        const int ml = s.mlen;
        s.mctx = ml * 2 + expected;
        const i16 a = st_[match_sm_[s.mctx] >> 20];
        const i16 b = static_cast<i16>((expected ? 1 : -1) * (ml < 32 ? ml * 64 : 2047));
        s.in[k] = a;
        acc0 += static_cast<i32>(w0[k]) * a;
        ++k;
        s.in[k] = b;
        acc0 += static_cast<i32>(w0[k]) * b;
        ++k;
        s.mlen_bucket = ml < 8 ? 1 : ml < 16 ? 2 : 3;
      } else {
        s.in[k++] = 0;
        s.in[k++] = 0;
        s.mlen_bucket = 0;
      }
    }
    // Exclusion: the byte is known not to be the expected one; while the
    // partial byte still follows it, the expected next bit is less likely.
    if (FLAG) {
      const bool on = ((s.expect | 0x100u) >> (8 - bp)) == static_cast<u32>(s.c0);
      const int ebit = static_cast<int>((s.expect >> (7 - bp)) & 1);
      const int ctx = ((s.lane & 15) << 4) | (bp << 1) | ebit;
      s.ex_ctx = on ? ctx : kExSlots - 1;
      const i16 v = st_[ex_[s.ex_ctx] >> 20];
      const i16 ev = on ? v : i16{0};
      s.in[k] = ev;
      acc0 += static_cast<i32>(w0[k]) * ev;
      ++k;
    }
    if (LSTM) {
      // Input 1: the LSTM's conditional bit probability. Input 2: the same
      // only while the LSTM gave the partial byte so far a fair chance
      // (>= 1/8), so the mixer can weight confident predictions separately.
      float mass = 0.f;
      const i16 v = st_[lstm_bit_p(s.lstm->cum(), s.c0, bp, mass)];
      const i16 vc = mass >= 0.125f ? v : i16{0};
      s.in[k] = v;
      acc0 += static_cast<i32>(w0[k]) * v;
      ++k;
      s.in[k] = vc;
      acc0 += static_cast<i32>(w0[k]) * vc;
      ++k;
    }
    s.in[k] = 256;  // bias
    acc0 += static_cast<i32>(w0[k]) * 256;
    ++k;
    for (; k < NINP; ++k) s.in[k] = 0;

    i32 dot = acc0 >> 14;
    if (NMIX > 1) {
      const int st1 = FLAG ? (s.fh[s.lane & 15] & 3) : s.mlen_bucket;
      const usize sel1 = static_cast<usize>(((s.lane & 15) << 5) | (bp << 2) | st1);
      i16* w = mix_[1] + sel1 * NINP;
      s.wsel[1] = w;
      dot = (dot + (mix_dot<NINP>(w, s.in) >> 14)) >> 1;
    }
    int d = dot;
    if (d > 2047) d = 2047;
    if (d < -2047) d = -2047;
    s.pmix = sq_[d + 2048];
    int pr = s.pmix;
    if (NAPM > 0) {
      const u32 cx = static_cast<u32>(s.c0 | ((s.lane & 15) << 8));
      pr = (pr + 3 * apm_pp(apm1_, cx, pr, st_, s.apm_idx[0])) >> 2;
    }
    if (NAPM > 1) {
      const u32 cx = (static_cast<u32>(s.c0 | (back(s, hist_, 1) << 8) | (s.lane << 16)) *
                      0x9E3779B1u) >> (32 - kApm2Bits);
      pr = (pr + 3 * apm_pp(apm2_, cx, pr, st_, s.apm_idx[1])) >> 2;
    }
    if (pr < 1) pr = 1;
    if (pr > 4095) pr = 4095;
    s.pr = pr;
  }

  Cfg cfg_;
  usize n_;
  int stride_;
  u8* hist_;
  const std::int16_t* st_;
  const std::int16_t* sq_;
  const int* dt_;
  int k_;

  u32* tab_[NM] = {};
  int bits_[NM] = {};
  i16* mix_[NMIX] = {};
  u16* apm1_ = nullptr;
  u16* apm2_ = nullptr;
  int match_bits_ = 0;
  u32* match_tab_ = nullptr;
  u32* match_sm_ = nullptr;
  u32* flag_ = nullptr;
  u32* ex_ = nullptr;
  i16* fmix_ = nullptr;

  usize bounds_[kMaxStreams + 1] = {};
  Stream strm_[kMaxStreams];
};

// Level tiers: L1-3 five models; L4-6 six models + match; L7+ twelve models,
// match, two mixers and two SSE stages. B = bitwise, F = byte-flag mode.
using Tier0B = Ensemble<5, false, 1, 1, false>;
using Tier1B = Ensemble<6, true, 1, 1, false>;
using Tier2B = Ensemble<12, true, 2, 2, false>;
using Tier0F = Ensemble<5, false, 1, 0, true>;
using Tier1F = Ensemble<6, true, 1, 0, true>;
using Tier2F = Ensemble<12, true, 2, 2, true>;
// Experimental tier 3: tier 2 plus the cmix-style LSTM byte predictor.
using Tier3B = Ensemble<12, true, 2, 2, false, true>;
using Tier3F = Ensemble<12, true, 2, 2, true, true>;

// LSTM hyper-parameters, carried in the block header so they can be swept
// without touching the decoder.
struct LstmParams {
  int cells = 32;
  int horizon = 10;
  float lr = 0.002f;
  bool lane = true;  // feed the record lane as a second one-hot input
};

// SPECTRA8_CRYSTAL_LSTM=1 or "cells,horizon,lr,lane" turns the experiment on at
// L10-11. Returns false when unset.
bool lstm_params_from_env(LstmParams& p) {
  const char* e = std::getenv("SPECTRA8_CRYSTAL_LSTM");
  if (!e || !*e || (e[0] == '0' && !e[1])) return false;
  int cells = p.cells, horizon = p.horizon, lane = p.lane;
  float lr = p.lr;
  if (std::sscanf(e, "%d,%d,%f,%d", &cells, &horizon, &lr, &lane) >= 1 && cells >= 8) {
    p.cells = cells < 8 ? 8 : cells > 128 ? 128 : cells;
    p.horizon = horizon < 1 ? 1 : horizon > 64 ? 64 : horizon;
    p.lr = lr < 1e-5f ? 1e-5f : lr > 0.5f ? 0.5f : lr;
    p.lane = lane != 0;
  }
  return true;
}

template <class E>
void attach_lstm(E& ens, int k, int stride, const LstmParams* lp,
                 std::vector<std::unique_ptr<LstmByteModel>>& owners) {
  if constexpr (E::kLstm) {
    for (int s = 0; s < k; ++s) {
      owners.push_back(std::make_unique<LstmByteModel>(lp->cells, lp->horizon,
                                                       lp->lane ? stride : 1, lp->lr));
      ens.set_lstm(s, owners.back().get());
    }
  } else {
    (void)ens;
    (void)k;
    (void)stride;
    (void)lp;
    (void)owners;
  }
}

// Byte-flag mode pays off when the lane candidate (byte one record back, or
// the previous byte) is often right; on text it would displace the bitwise
// match model and lose ratio.
bool use_flag_mode(const u8* s, usize n, int stride) noexcept {
  const usize S = static_cast<usize>(stride);
  const usize lim = min_of(n, usize{1} << 15);
  if (lim <= S) return false;
  usize hits = 0;
  for (usize i = S; i < lim; ++i) hits += s[i] == s[i - S];
  return hits * 10 >= (lim - S);  // >= 10%
}

// Encode: K streams in lockstep, one bit of each per inner step.
template <class E>
void encode_streams(const Cfg& cfg, const u8* src, usize n, int stride, int k,
                    std::vector<u8>& payload, const LstmParams* lp = nullptr) {
  thread_local std::vector<u8> hist;
  hist.assign(n, 0);  // the decoder zeroes dst the same way
  E ens(cfg, n, stride, hist.data(), k);
  std::vector<std::unique_ptr<LstmByteModel>> lstms;
  attach_lstm(ens, k, stride, lp, lstms);
  ArithEncoder enc[kMaxStreams];
  for (int s = 0; s < k; ++s) enc[s].out.reserve(n / static_cast<usize>(k) / 2 + 64);

  usize longest = 0;
  for (int s = 0; s < k; ++s)
    longest = max_of(longest, ens.stream(s).end - ens.stream(s).start);

  for (usize step = 0; step < longest; ++step) {
    for (int s = 0; s < k; ++s) {
      typename E::Stream& st = ens.stream(s);
      if (st.pos >= st.end) continue;
      ens.begin_byte(st);
      if constexpr (E::kFlag) {
        const int hit = src[st.pos] == st.expect;
        enc[s].encode(hit, st.fpr);
        ens.finish_flag(st, hit);
      }
    }
    for (int b = 7; b >= 0; --b) {
      for (int s = 0; s < k; ++s) {
        typename E::Stream& st = ens.stream(s);
        if (!st.in_bits) continue;
        const int y = (src[st.pos] >> b) & 1;
        enc[s].encode(y, st.pr);
        ens.update(st, y);
      }
    }
  }
  ByteWriter w(payload);
  for (int s = 0; s < k; ++s) enc[s].flush();
  for (int s = 0; s + 1 < k; ++s) w.varint(enc[s].out.size());
  for (int s = 0; s < k; ++s) w.bytes(enc[s].out);
}

template <class E>
Status decode_streams(const Cfg& cfg, ByteReader& r, u8* dst, usize n, int stride,
                      int k, const LstmParams* lp = nullptr) {
  usize lens[kMaxStreams];
  usize sum = 0;
  for (int s = 0; s + 1 < k; ++s) {
    lens[s] = static_cast<usize>(r.varint());
    if (r.error) return Status::CorruptInput;
    sum += lens[s];
  }
  if (sum > r.remaining()) return Status::CorruptInput;
  lens[k - 1] = r.remaining() - sum;

  std::memset(dst, 0, n);
  E ens(cfg, n, stride, dst, k);
  std::vector<std::unique_ptr<LstmByteModel>> lstms;
  attach_lstm(ens, k, stride, lp, lstms);
  ArithDecoder dec[kMaxStreams];
  for (int s = 0; s < k; ++s) {
    const u8* p = r.take(lens[s]);
    if (!p) return Status::CorruptInput;
    dec[s].init(p, lens[s]);
  }
  usize longest = 0;
  for (int s = 0; s < k; ++s)
    longest = max_of(longest, ens.stream(s).end - ens.stream(s).start);

  for (usize step = 0; step < longest; ++step) {
    for (int s = 0; s < k; ++s) {
      typename E::Stream& st = ens.stream(s);
      if (st.pos >= st.end) continue;
      ens.begin_byte(st);
      if constexpr (E::kFlag) {
        const int hit = dec[s].decode(st.fpr);
        ens.finish_flag(st, hit);
      }
    }
    for (int b = 0; b < 8; ++b) {
      for (int s = 0; s < k; ++s) {
        typename E::Stream& st = ens.stream(s);
        if (!st.in_bits) continue;
        const int y = dec[s].decode(st.pr);
        ens.update(st, y);
      }
    }
  }
  return Status::Ok;
}

}  // namespace

// Payload: u8 ver, u8 kind. kind 0: raw bytes follow. kind 1: u8 tier,
// u8 stride, u8 streams, varint stream sizes (all but last), then the
// arithmetic-coded streams. kind 5: static probability profile; kind 6: the
// same with per-page lane modes (build_crystal_profile).
void build_crystal_block(const u8* src, usize n, int level,
                        std::vector<u8>& payload, u32 profile) {
  payload.clear();
  payload.push_back(kCryVer);
  if (n < 64) {
    payload.push_back(kCryStored);
    payload.insert(payload.end(), src, src + n);
    return;
  }
  if (profile != 0) {
    payload.push_back(kCryProfile);
    const usize hdr = payload.size();
    bool lane_modes = false;
    if (build_crystal_profile(src, n, profile, payload, lane_modes) && payload.size() - hdr < n) {
      if (lane_modes) payload[hdr - 1] = kCryProfileLanes;
      stats_observe_crystal_profile_block(false);
      return;
    }
    // Unknown profile, or a page the profile codes worse than a per-page
    // model (build_crystal_profile's escape): this block takes the dynamic
    // path below. The block kind is per block, so a column may mix the two.
    stats_observe_crystal_profile_block(true);
    payload.resize(hdr - 1);
  }
  const int stride = detect_stride(src, n);
  if (level <= 9) {
    payload.push_back(kCryStatic);
    const usize hdr = payload.size();
    build_crystal_static(src, n, level, stride, payload);
    // Detection stage for the BWT text path: the transform plus an order-1
    // cost estimate. Only a predicted gain of >= kBwtGain over the lattice
    // size is worth the second static encode (and the ~2x slower decode of
    // the inverse BWT); the final choice is by real size.
    constexpr double kBwtGain = 0.03;
    const double bar = (1.0 - kBwtGain) * static_cast<double>(payload.size() - hdr);
    // The estimate is a full suffix sort: ~25% of the encode on pages where
    // it does not win (most numeric columns). Two cheap gates first:
    //  - a lattice payload under 1% of the input leaves nothing worth the
    //    BWT's slower decode;
    //  - for blocks >= 64 KiB, a 32 KiB mid-block window is encoded with
    //    the lattice (level 1) and BWT-estimated, and the full sort runs
    //    only when the window estimate is within 5% of the window lattice
    //    size. Both sides carry their fixed table costs at the same scale,
    //    which a linear scale-up of the estimate alone would not. It is a
    //    ratio test, not a type test: BWT also wins on repetitive numeric
    //    records (floats of small integers) and sparse flag columns.
    const usize lat = payload.size() - hdr;
    bool consider_bwt = lat * 100 >= n;
    constexpr usize kWinMin = 64 * 1024, kWin = 32 * 1024;
    if (consider_bwt && n >= kWinMin) {
      const usize off = ((n - kWin) / 2) / static_cast<usize>(stride) * static_cast<usize>(stride);
      thread_local std::vector<u8> win_lat;
      win_lat.clear();
      build_crystal_static(src + off, kWin, 1, stride, win_lat);
      const usize est_w = crystal_bwt_estimate(src + off, kWin);
      consider_bwt = static_cast<double>(est_w) < 1.05 * static_cast<double>(win_lat.size());
    }
    const usize est = consider_bwt ? crystal_bwt_estimate(src, n) : ~usize{0};
    if (static_cast<double>(est) < bar) {
      thread_local std::vector<u8> alt;
      alt.clear();
      build_crystal_bwt(src, n, level, alt);
      if (static_cast<double>(alt.size()) < bar) {
        payload.resize(hdr);
        payload[hdr - 1] = kCryBwt;
        payload.insert(payload.end(), alt.begin(), alt.end());
      }
    }
    if (payload.size() - hdr >= n) {
      payload.clear();
      payload.push_back(kCryVer);
      payload.push_back(kCryStored);
      payload.insert(payload.end(), src, src + n);
    }
    return;
  }
  const bool flag = use_flag_mode(src, n, stride);
  LstmParams lp;
  const bool lstm = lstm_params_from_env(lp);
  const int tier = lstm ? 3 : tier_for_level(level);
  // The LSTM is a byte-sequence model: one stream so it sees the whole block.
  int k = lstm ? 1 : streams_for_level(level, n, flag);
  if (!lstm) {
    // Experiment knob: SPECTRA8_CRYSTAL_CM_STREAMS=N overrides the stream count
    // (a control for single-stream vs. LSTM comparisons).
    if (const char* e = std::getenv("SPECTRA8_CRYSTAL_CM_STREAMS")) {
      const int v = std::atoi(e);
      if (v >= 1 && v <= kMaxStreams) k = v;
    }
  }
  payload.push_back(flag ? kCryCmFlag : kCryCm);
  payload.push_back(static_cast<u8>(tier));
  payload.push_back(static_cast<u8>(stride));
  payload.push_back(static_cast<u8>(k));
  if (lstm) {
    payload.push_back(static_cast<u8>(lp.cells));
    payload.push_back(static_cast<u8>(lp.horizon));
    const u16 lr = static_cast<u16>(lp.lr * 100000.f + 0.5f);
    payload.push_back(static_cast<u8>(lr & 255));
    payload.push_back(static_cast<u8>(lr >> 8));
    payload.push_back(static_cast<u8>(lp.lane));
    lp.lr = static_cast<float>(lr) / 100000.f;  // what the decoder will see
  }
  const usize hdr = payload.size();
  const Cfg cfg = cfg_for_tier(tier);
  if (lstm) {
    if (flag)
      encode_streams<Tier3F>(cfg, src, n, stride, k, payload, &lp);
    else
      encode_streams<Tier3B>(cfg, src, n, stride, k, payload, &lp);
  } else if (flag) {
    switch (tier) {
      case 0: encode_streams<Tier0F>(cfg, src, n, stride, k, payload); break;
      case 1: encode_streams<Tier1F>(cfg, src, n, stride, k, payload); break;
      default: encode_streams<Tier2F>(cfg, src, n, stride, k, payload); break;
    }
  } else {
    switch (tier) {
      case 0: encode_streams<Tier0B>(cfg, src, n, stride, k, payload); break;
      case 1: encode_streams<Tier1B>(cfg, src, n, stride, k, payload); break;
      default: encode_streams<Tier2B>(cfg, src, n, stride, k, payload); break;
    }
  }
  if (payload.size() - hdr >= n) {
    payload.clear();
    payload.push_back(kCryVer);
    payload.push_back(kCryStored);
    payload.insert(payload.end(), src, src + n);
  }
}

Status decode_crystal_block(const u8* data, usize len, u8* dst, usize n) {
  if (!data || len < 2) return Status::CorruptInput;
  if (data[0] != kCryVer) return Status::CorruptInput;
  if (data[1] == kCryStored) {
    if (len - 2 != n) return Status::CorruptInput;
    std::memcpy(dst, data + 2, n);
    return Status::Ok;
  }
  if (data[1] == kCryStatic) {
    if (n == 0) return Status::Ok;
    return decode_crystal_static(data + 2, len - 2, dst, n);
  }
  if (data[1] == kCryBwt) {
    if (n == 0) return Status::Ok;
    return decode_crystal_bwt(data + 2, len - 2, dst, n);
  }
  if (data[1] == kCryProfile || data[1] == kCryProfileLanes) {
    if (n == 0) return Status::Ok;
    return decode_crystal_profile(data + 2, len - 2, dst, n, data[1] == kCryProfileLanes);
  }
  if ((data[1] != kCryCm && data[1] != kCryCmFlag) || len < 5) return Status::CorruptInput;
  const bool flag = data[1] == kCryCmFlag;
  const int tier = data[2];
  const int stride = data[3];
  const int k = data[4];
  if (tier > 3 || stride < 1 || stride > 16 || k < 1 || k > kMaxStreams)
    return Status::CorruptInput;
  if (n == 0) return Status::Ok;
  ByteReader r(data + 5, len - 5);
  const Cfg cfg = cfg_for_tier(tier);
  if (tier == 3) {
    if (len < 10 || k != 1) return Status::CorruptInput;
    LstmParams lp;
    lp.cells = data[5];
    lp.horizon = data[6];
    lp.lr = static_cast<float>(data[7] | (data[8] << 8)) / 100000.f;
    lp.lane = data[9] != 0;
    if (lp.cells < 8 || lp.cells > 128 || lp.horizon < 1 || lp.horizon > 64)
      return Status::CorruptInput;
    ByteReader r3(data + 10, len - 10);
    if (flag) return decode_streams<Tier3F>(cfg, r3, dst, n, stride, k, &lp);
    return decode_streams<Tier3B>(cfg, r3, dst, n, stride, k, &lp);
  }
  if (flag) {
    switch (tier) {
      case 0: return decode_streams<Tier0F>(cfg, r, dst, n, stride, k);
      case 1: return decode_streams<Tier1F>(cfg, r, dst, n, stride, k);
      default: return decode_streams<Tier2F>(cfg, r, dst, n, stride, k);
    }
  }
  switch (tier) {
    case 0: return decode_streams<Tier0B>(cfg, r, dst, n, stride, k);
    case 1: return decode_streams<Tier1B>(cfg, r, dst, n, stride, k);
    default: return decode_streams<Tier2B>(cfg, r, dst, n, stride, k);
  }
}

}  // namespace spectra8::detail
