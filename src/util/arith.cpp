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

#include "util/arith.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace spectra8::detail::arith {
namespace {

constexpr u32 kTop = 1u << 24;
// Frequency total of the static table. Must fit u16 (header) and stay below
// 2^16 so range/total >= 1 and the reciprocal table (2^16 entries) covers it.
constexpr u32 kTotal = 65535u;
constexpr usize kHeader = 512u;  // 256 x u16 LE frequencies (0 = unused)

void build_cum(const u16 freq[256], u32 cum[257], u32& total) {
  u32 c = 0;
  for (int i = 0; i < 256; ++i) {
    cum[i] = c;
    c += freq[i];
  }
  cum[256] = c;
  total = c;
}

// Scales the histogram to a table with total exactly kTotal, every used
// symbol >= 1. Floor scaling first (sum <= kTotal), then the min-1 bumps can
// push the sum over by at most the number of used symbols (<= 256 < kTotal);
// the difference is settled on the largest entries, never below 1. (The
// former round-up scaling to 65536 could leave the total above 2^16 -- the
// reciprocal table then read out of bounds and the encoder ran away -- and a
// single-symbol input gave freq 65536, i.e. 0 in u16.)
void hist_to_freq(const u32 hist[256], u16 freq[256]) {
  u64 sum = 0;
  for (int i = 0; i < 256; ++i) sum += hist[i];
  if (sum == 0) {
    for (int i = 0; i < 256; ++i) freq[i] = 1;
    return;
  }
  u32 f[256];
  u32 acc = 0;
  for (int i = 0; i < 256; ++i) {
    if (!hist[i]) {
      f[i] = 0;
      continue;
    }
    u32 w = static_cast<u32>((static_cast<u64>(hist[i]) * kTotal) / sum);
    if (w == 0) w = 1;
    f[i] = w;
    acc += w;
  }
  if (acc < kTotal) {
    int best = 0;
    for (int i = 1; i < 256; ++i)
      if (f[i] > f[best]) best = i;
    f[best] += kTotal - acc;
  } else {
    while (acc > kTotal) {
      int best = -1;
      for (int i = 0; i < 256; ++i)
        if (f[i] > 1 && (best < 0 || f[i] > f[best])) best = i;
      if (best < 0) break;  // unreachable: used <= 256 < kTotal
      const u32 take = min_of(acc - kTotal, f[best] - 1);
      f[best] -= take;
      acc -= take;
    }
  }
  for (int i = 0; i < 256; ++i) freq[i] = static_cast<u16>(f[i]);
}

// Exact `x / d` for 16-bit totals (shared by encode and decode).
// With m = floor(2^64/d)+1, floor((x*m)>>64) == floor(x/d) for x < 2^48
// (Granlund & Montgomery). range < 2^32, so this is bit-identical to `/`.
struct RecipTable {
  std::vector<u64> m;
  RecipTable() : m(usize{1} << 16) {
    for (u32 d = 2; d < (u32{1} << 16); ++d) m[d] = ~u64{0} / d + 1;
  }
};

inline const u64* recip_tab() {
  static const RecipTable g;
  return g.m.data();
}

inline u32 div_recip(u32 x, u32 d, const u64* recip) {
  if (d <= 1) return x;  // magic for d==1 overflows u64; x/1 == x
  return static_cast<u32>(
      (static_cast<unsigned __int128>(x) * recip[d]) >> 64);
}

// Canonical LZMA-family range encoder. `low` MUST be 64-bit so the carry out of
// the active 32-bit window is detected in shift_low (a u32 `low` silently drops
// carries and desyncs the decoder on high-entropy streams).
struct RangeEnc {
  u64 low = 0;
  u32 range = 0xFFFFFFFFu;
  u8 cache = 0;
  u64 cache_size = 1;
  std::vector<u8>& out;
  const u64* recip;
  u8* w = nullptr;
  u8* wend = nullptr;

  explicit RangeEnc(std::vector<u8>& o) : out(o), recip(recip_tab()) {
    const usize sz = out.size();
    out.resize(sz + 64);
    w = out.data() + sz;
    wend = out.data() + out.size();
  }

  void ensure(usize extra) {
    const usize pos = static_cast<usize>(w - out.data());
    if (pos + extra <= out.size()) return;
    out.resize(pos + extra);
    w = out.data() + pos;
    wend = out.data() + out.size();
  }

  void emit(u8 b) {
    if (w == wend) ensure(1024);
    *w++ = b;
  }

  void shift_low() {
    if (low < 0xFF000000ULL || low > 0xFFFFFFFFULL) {
      u8 temp = cache;
      do {
        emit(static_cast<u8>(temp + static_cast<u8>(low >> 32)));
        temp = 0xFF;
      } while (--cache_size != 0);
      cache = static_cast<u8>(low >> 24);
    }
    ++cache_size;
    low = (low & 0x00FFFFFFULL) << 8;
  }

  void encode(u32 start, u32 size, u32 total) {
    if (total == 0 || size == 0) return;
    // Same exact reciprocal as the decoder: total is 16-bit (kMaxTotal),
    // range < 2^32, so this is bit-identical to `range /= total` and
    // removes the udiv from every BWT symbol.
    range = div_recip(range, total, recip);
    // MTF/BWT ranks are dominated by symbol 0 (start == 0): skip the mul.
    if (start) low += static_cast<u64>(start) * range;
    range *= size;
    while (range < kTop) {
      shift_low();
      range <<= 8;
    }
  }

  void finish() {
    for (int i = 0; i < 5; ++i) shift_low();
    out.resize(static_cast<usize>(w - out.data()));
  }
};

struct RangeDec {
  u32 code = 0;
  u32 range = 0xFFFFFFFFu;
  u32 r = 1;  // cached range/total between get_count and update_range
  const u8* p = nullptr;
  const u8* end = nullptr;

  explicit RangeDec(const u8* data, usize len) : p(data), end(data + len) {}

  bool init() {
    if (static_cast<usize>(end - p) < 5) return false;
    ++p;  // encoder's leading byte (initial cache) is always 0
    for (int i = 0; i < 4; ++i) code = (code << 8) | *p++;
    return true;
  }

  void normalize() {
    while (range < kTop) {
      code = (code << 8) | (p < end ? *p++ : 0u);
      range <<= 8;
    }
  }


  u32 get_count(u32 total) {
    r = range / total;
    const u32 count = (r == 0) ? 0 : code / r;
    return count >= total ? total - 1 : count;
  }

  void update_range(u32 start, u32 size) {
    code -= r * start;
    range = r * size;
    normalize();
  }

  int decode(const u32 cum[257], u32 total) {
    if (total == 0) return -1;
    const u32 count = get_count(total);
    int lo = 0;
    int hi = 256;
    while (lo + 1 < hi) {
      const int mid = (lo + hi) >> 1;
      if (cum[mid] <= count)
        lo = mid;
      else
        hi = mid;
    }
    const int sym = lo;
    update_range(cum[sym], cum[sym + 1] - cum[sym]);
    return sym;
  }
};

void encode_block(std::vector<u8>& out, const u8* src, usize n,
                  const u16 freq[256]) {
  u32 cum[257];
  u32 total = 0;
  build_cum(freq, cum, total);
  if (total == 0) return;

  out.resize(kHeader);
  for (int i = 0; i < 256; ++i) {
    out[static_cast<usize>(i) * 2] = static_cast<u8>(freq[i]);
    out[static_cast<usize>(i) * 2 + 1] = static_cast<u8>(freq[i] >> 8);
  }

  RangeEnc enc(out);
  enc.ensure(n + 16);
  for (usize i = 0; i < n; ++i) enc.encode(cum[src[i]], cum[src[i] + 1] - cum[src[i]], total);
  enc.finish();
}

// Adaptive order-0 model over a 256-symbol alphabet. BWT+MTF (and the RUNA/RUNB
// run stream) produce strongly skewed output dominated by small symbols, so a
// plain frequency array with a linear cumulative scan is faster than a Fenwick
// tree here: the decoder's symbol search and the encoder's cum() touch only the
// few low symbols that actually occur, and the per-symbol update is O(1). The
// coded bits are identical to the Fenwick version (same increment, rescale and
// cumulative semantics) -- this is purely a speed change.
struct AdaptiveModel {
  // Increment per coded symbol and rescale threshold. kMaxTotal stays below the
  // range coder's kBottom (2^16) so range/total >= 1 always holds.
  static constexpr u32 kInc = 32;
  static constexpr u32 kMaxTotal = (1u << 16) - 256u;

  u32 freq[256];
  u32 group[16];  // sum of freq[g*16 .. g*16+15]; skips the decode scan
  u32 total;

  void init() {
    for (int s = 0; s < 256; ++s) freq[s] = 1;
    for (int g = 0; g < 16; ++g) group[g] = 16;
    total = 256;
  }

  // Sum of frequencies of symbols [0, sym).
  u32 cum(int sym) const {
    u32 s = 0;
    const int gs = sym >> 4;
    for (int g = 0; g < gs; ++g) s += group[g];
    for (int i = gs << 4; i < sym; ++i) s += freq[i];
    return s;
  }

  // Largest sym with cum(sym) <= count; also returns that cumulative start.
  int find(u32 count, u32& start) const {
    u32 acc = 0;
    int s = 0;
    while (acc + freq[s] <= count) {
      acc += freq[s];
      ++s;
    }
    start = acc;
    return s;
  }

  void update(int sym) {
    freq[sym] += kInc;
    group[sym >> 4] += kInc;
    total += kInc;
    if (total >= kMaxTotal) rescale();
  }

  void rescale() {
    total = 0;
    for (int g = 0; g < 16; ++g) {
      u32 s = 0;
      const int base = g << 4;
      for (int k = 0; k < 16; ++k) {
        freq[base + k] = (freq[base + k] >> 1) | 1u;  // keep min 1
        s += freq[base + k];
      }
      group[g] = s;
      total += s;
    }
  }
};

}  // namespace

namespace {

// Same symbol as the linear scan, but skip 16-wide frequency groups when
// the target sits past them. Rank-0 (the BWT/MTF common case) still exits
// after one compare. Bit-identical to walking freq[0..s].
inline void adapt_find(const u32* __restrict freq, const u32* __restrict group,
                       u32 r, u32 code, int& s, u32& acc, u32& f) {
  s = 0;
  acc = 0;
  f = freq[0];
  if (static_cast<u64>(acc + f) * r > code) return;
  acc += f;
  const u32 rest0 = group[0] - freq[0];
  if (static_cast<u64>(acc + rest0) * r <= code) {
    acc += rest0;
    int g = 1;
    while (g < 15 && static_cast<u64>(acc + group[g]) * r <= code) {
      acc += group[g];
      ++g;
    }
    s = g << 4;
    f = freq[s];
  } else {
    s = 1;
    f = freq[1];
  }
  while (s < 255 && static_cast<u64>(acc + f) * r <= code) {
    acc += f;
    f = freq[++s];
  }
}

}  // namespace

double estimate_adaptive(const u8* src, usize n) {
  // log2(1..65535) covers every AdaptiveModel total / frequency.
  static const auto log2tab = []() {
    std::vector<float> t(usize{1} << 16);
    t[0] = 0.0f;
    for (u32 i = 1; i < (u32{1} << 16); ++i)
      t[i] = std::log2(static_cast<float>(i));
    return t;
  }();
  const float* tab = log2tab.data();
  AdaptiveModel m;
  m.init();
  double bits = 0.0;
  for (usize i = 0; i < n; ++i) {
    const int s = src[i];
    bits += static_cast<double>(tab[m.total] - tab[m.freq[s]]);
    m.update(s);
  }
  // Five finish bytes plus a little range-coder slack.
  return bits * 0.125 + 6.0;
}

void encode_adaptive(std::vector<u8>& out, const u8* src, usize n) {
  AdaptiveModel m;
  m.init();
  RangeEnc enc(out);
  enc.ensure(n + 32);
  for (usize i = 0; i < n; ++i) {
    const int sym = src[i];
    const u32 size = m.freq[sym];
    const u32 start = sym ? m.cum(sym) : 0u;
    enc.encode(start, size, m.total);
    m.update(sym);
  }
  enc.finish();
}

Status decode_adaptive(const u8* data, usize len, u8* dst, usize n) {
  if (n == 0) return Status::Ok;
  if (!data || len < 5) return Status::CorruptInput;
  AdaptiveModel m;
  m.init();
  RangeDec dec(data, len);
  if (!dec.init()) return Status::CorruptInput;

  // Hot loop with a single division per symbol. The textbook range decoder
  // needs two (`range/total` then `code/r`); we keep only the first and fold
  // the second into the cumulative scan: the largest symbol s with
  // cum(s) <= floor(code/r) is exactly the largest with cum(s)*r <= code, so we
  // walk the (skewed, small-symbol-dominated) frequency table comparing
  // (acc+freq)*r against code. The capped `s < 255` matches the decoder's
  // count>=total clamp (last symbol is the catch-all). The division itself is
  // replaced by an exact reciprocal multiply (total < 2^16, range < 2^32, so
  // the round-up magic is exact - see RecipTable). Bit-identical to the
  // two-division form, so no re-encode / format change -- pure decode speed.
  const u64* __restrict recip = recip_tab();
  const u32* __restrict freq = m.freq;
  const u32* __restrict group = m.group;
  for (usize i = 0; i < n; ++i) {
    const u32 r = div_recip(dec.range, m.total, recip);
    const u32 code = dec.code;
    int s = 0;
    u32 acc = 0;
    u32 f = 0;
    adapt_find(freq, group, r, code, s, acc, f);
    dec.code = code - r * acc;
    dec.range = r * f;
    dec.normalize();
    m.update(s);
    dst[i] = static_cast<u8>(s);
  }
  return Status::Ok;
}

void encode_adaptive_o1(std::vector<u8>& out, const u8* src, usize n) {
  static thread_local std::vector<AdaptiveModel> models;
  models.resize(kO1Ctx);
  for (auto& m : models) m.init();
  RangeEnc enc(out);
  enc.ensure(n + 32);
  u8 prev = 0;
  for (usize i = 0; i < n; ++i) {
    AdaptiveModel& m = models[o1_class(prev)];
    const int sym = src[i];
    const u32 size = m.freq[sym];
    const u32 start = sym ? m.cum(sym) : 0u;
    enc.encode(start, size, m.total);
    m.update(sym);
    prev = src[i];
  }
  enc.finish();
}

Status decode_adaptive_o1(const u8* data, usize len, u8* dst, usize n) {
  if (n == 0) return Status::Ok;
  if (!data || len < 5) return Status::CorruptInput;
  static thread_local std::vector<AdaptiveModel> models;
  models.resize(kO1Ctx);
  for (auto& m : models) m.init();
  RangeDec dec(data, len);
  if (!dec.init()) return Status::CorruptInput;

  // Same single-division hot loop as decode_adaptive (see there), with the
  // model picked by the previous symbol's class. The extra work per symbol
  // is one predictable index computation, so this decodes at the same speed
  // as the order-0 coder while modeling run/churn contexts separately.
  const u64* __restrict recip = recip_tab();
  u8 prev = 0;
  for (usize i = 0; i < n; ++i) {
    AdaptiveModel& m = models[o1_class(prev)];
    const u32 r = div_recip(dec.range, m.total, recip);
    const u32 code = dec.code;
    int s = 0;
    u32 acc = 0;
    u32 f = 0;
    adapt_find(m.freq, m.group, r, code, s, acc, f);
    dec.code = code - r * acc;
    dec.range = r * f;
    dec.normalize();
    m.update(s);
    prev = static_cast<u8>(s);
    dst[i] = prev;
  }
  return Status::Ok;
}

void encode(std::vector<u8>& out, const u8* src, usize n) {
  u32 hist[256] = {};
  for (usize i = 0; i < n; ++i) ++hist[src[i]];
  encode(out, src, n, hist);
}

void encode(std::vector<u8>& out, const u8* src, usize n, const u32 freq[256]) {
  u32 hist[256] = {};
  for (usize i = 0; i < n; ++i) ++hist[src[i]];
  for (int s = 0; s < 256; ++s)
    if (freq[s]) hist[s] = max_of(hist[s], freq[s]);
  u16 table[256];
  hist_to_freq(hist, table);
  encode_block(out, src, n, table);
}

Status decode(const u8* data, usize len, u8* dst, usize n) {
  if (n == 0) return Status::Ok;
  if (!data || len < kHeader + 5) return Status::CorruptInput;

  u16 freq[256];
  for (int i = 0; i < 256; ++i)
    freq[i] = static_cast<u16>(data[static_cast<usize>(i) * 2] |
                               (static_cast<u16>(data[static_cast<usize>(i) * 2 + 1])
                                << 8));

  u32 cum[257];
  u32 total = 0;
  build_cum(freq, cum, total);
  if (total == 0) return Status::CorruptInput;

  RangeDec dec(data + kHeader, len - kHeader);
  if (!dec.init()) return Status::CorruptInput;

  for (usize i = 0; i < n; ++i) {
    const int sym = dec.decode(cum, total);
    if (sym < 0 || sym > 255 || !freq[sym]) return Status::CorruptInput;
    dst[i] = static_cast<u8>(sym);
  }
  return Status::Ok;
}

}  // namespace spectra8::detail::arith
