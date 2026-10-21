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

#include "grating/grating.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <type_traits>
#include <vector>

#if defined(__x86_64__) && !defined(__BMI2__) && (defined(__GNUC__) || defined(__clang__))
#define GRATING_X86_BMI2 1
#define GRATING_INLINE inline __attribute__((always_inline))
#define GRATING_BMI2_FN __attribute__((target("bmi2,popcnt"), noinline))
#else
#define GRATING_X86_BMI2 0
#if defined(__GNUC__) || defined(__clang__)
#define GRATING_INLINE inline __attribute__((always_inline))
#else
#define GRATING_INLINE inline
#endif
#define GRATING_BMI2_FN
#endif

namespace spectra8::detail {
namespace {

#if GRATING_X86_BMI2
inline bool grating_has_bmi2() noexcept {
  static const bool has = __builtin_cpu_supports("bmi2") && __builtin_cpu_supports("popcnt");
  return has;
}
#endif

constexpr int kScaleBits = 12;
constexpr u32 kM = 1u << kScaleBits;
constexpr u32 kRansL16 = 1u << 16;
constexpr int kMaxKept = 4095;  // one slot stays available for the escape

struct Item {
  u64 sym;
  u32 count;
};

struct Tab {
  u64 sym;
  u32 count;
  u32 freq;
  u32 start;
  bool escape;
};

struct EncSym {
  u32 xmax;
  u32 m_lo;
  u32 bias;
  u16 cmpl_freq;
  u16 sh;
};

template <int W>
GRATING_INLINE u64 load_sym_w(const u8* p) noexcept {
  if constexpr (W == 1) {
    return *p;
  } else if constexpr (W == 2) {
    u16 v;
    std::memcpy(&v, p, 2);
    return v;
  } else if constexpr (W == 4) {
    u32 v;
    std::memcpy(&v, p, 4);
    return v;
  } else {
    u64 v;
    std::memcpy(&v, p, 8);
    return v;
  }
}

template <int W>
GRATING_INLINE void store_sym_w(u8* p, u64 v) noexcept {
  if constexpr (W == 1) {
    *p = static_cast<u8>(v);
  } else if constexpr (W == 2) {
    const u16 u = static_cast<u16>(v);
    std::memcpy(p, &u, 2);
  } else if constexpr (W == 4) {
    const u32 u = static_cast<u32>(v);
    std::memcpy(p, &u, 4);
  } else {
    std::memcpy(p, &v, 8);
  }
}

inline u64 load_sym(const u8* p, int w) noexcept {
  switch (w) {
    case 1: return load_sym_w<1>(p);
    case 2: return load_sym_w<2>(p);
    case 4: return load_sym_w<4>(p);
    default: return load_sym_w<8>(p);
  }
}

inline void store_sym(u8* p, int w, u64 v) noexcept {
  switch (w) {
    case 1: store_sym_w<1>(p, v); break;
    case 2: store_sym_w<2>(p, v); break;
    case 4: store_sym_w<4>(p, v); break;
    default: store_sym_w<8>(p, v); break;
  }
}

inline void init_enc_sym(EncSym& e, u32 f, u32 start) {
  e.xmax = f < kM ? (f << (32 - kScaleBits)) : ~0u;
  e.cmpl_freq = static_cast<u16>(kM - f);
  if (f < 2) {
    e.m_lo = ~0u;
    e.sh = 0;
    e.bias = start + kM - 1;
  } else {
    const u32 l = 32u - static_cast<u32>(__builtin_clz(f - 1));
    e.m_lo = static_cast<u32>(((u64{1} << (32 + l)) + f - 1) / f - (u64{1} << 32));
    e.sh = static_cast<u16>(l - 1);
    e.bias = start;
  }
}

GRATING_INLINE void rans_put16(u32& x, u8*& ptr, const EncSym& e) {
  const bool renorm = x >= e.xmax;
  ptr[-2] = static_cast<u8>(x);
  ptr[-1] = static_cast<u8>(x >> 8);
  ptr -= renorm ? 2 : 0;
  x = renorm ? (x >> 16) : x;
  const u32 t = static_cast<u32>((static_cast<u64>(x) * e.m_lo) >> 32);
  const u32 q = (((x - t) >> 1) + t) >> e.sh;
  x = x + e.bias + q * e.cmpl_freq;
}

GRATING_INLINE void rans_renorm(u32& x, const u8*& p, const u8* plast) {
  const u8* q = p < plast ? p : plast;
  const u32 v = load_u16(q);
  const bool need = x < kRansL16;
  x = need ? (x << 16) | v : x;
  p = need ? q + 2 : p;
}

GRATING_INLINE void rans_renorm_fast(u32& x, const u8*& p) {
  const u32 v = load_u16(p);
  const bool need = x < kRansL16;
  x = need ? (x << 16) | v : x;
  p += need ? 2 : 0;
}

void normalize(std::vector<Tab>& tabs) {
  u64 total = 0;
  for (const Tab& t : tabs) total += t.count;
  u32 sum = 0;
  for (Tab& t : tabs) {
    u32 f = static_cast<u32>((static_cast<u64>(t.count) * kM) / total);
    if (f < 1) f = 1;
    t.freq = f;
    sum += f;
  }
  if (sum < kM) {
    int best = 0;
    for (int i = 1; i < static_cast<int>(tabs.size()); ++i)
      if (tabs[static_cast<usize>(i)].count > tabs[static_cast<usize>(best)].count)
        best = i;
    tabs[static_cast<usize>(best)].freq += kM - sum;
  } else if (sum > kM) {
    // One pass from the largest frequency down. Each symbol keeps at least
    // one slot; the kept set is at most kM, so the excess always fits.
    u32 excess = sum - kM;
    std::vector<usize> order(tabs.size());
    for (usize i = 0; i < order.size(); ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](usize a, usize b) {
      return tabs[a].freq > tabs[b].freq;
    });
    for (usize i : order) {
      if (excess == 0) break;
      u32& f = tabs[i].freq;
      if (f <= 1) continue;
      const u32 take = std::min(excess, f - 1);
      f -= take;
      excess -= take;
    }
  }
  u32 c = 0;
  for (Tab& t : tabs) {
    t.start = c;
    c += t.freq;
  }
}

// Symbols whose natural slot would be empty join the escape. The kept set is
// at most kMaxKept so the escape, when present, still has a slot.
void split_tail(const std::vector<Item>& items, u64 nsyms, std::vector<Tab>& tabs) {
  // A symbol earns a slot when count * scale >= nsyms. The tail below that
  // line is escape. Only the heavy hitters are sorted.
  const u64 thresh = nsyms == 0 ? 1 : (nsyms + kM - 1) / kM;
  std::vector<Item> kept;
  kept.reserve(std::min(items.size(), static_cast<usize>(kMaxKept)));
  for (const Item& it : items)
    if (it.count >= thresh) kept.push_back(it);
  if (kept.size() > static_cast<usize>(kMaxKept)) {
    std::nth_element(kept.begin(), kept.begin() + kMaxKept, kept.end(),
                     [](const Item& a, const Item& b) {
                       if (a.count != b.count) return a.count > b.count;
                       return a.sym < b.sym;
                     });
    kept.resize(kMaxKept);
  }
  std::sort(kept.begin(), kept.end(),
            [](const Item& a, const Item& b) { return a.sym < b.sym; });
  u64 kept_mass = 0;
  for (const Item& it : kept) kept_mass += it.count;
  tabs.clear();
  tabs.reserve(kept.size() + 1);
  for (const Item& it : kept)
    tabs.push_back(Tab{it.sym, it.count, 0, 0, false});
  if (kept_mass < nsyms)
    tabs.push_back(Tab{0, static_cast<u32>(nsyms - kept_mass), 0, 0, true});
  if (tabs.size() >= 2) normalize(tabs);
}

double estimate(const std::vector<Item>& items, u64 nsyms, int width) {
  if (items.size() <= 1) return 2.0 + width;
  std::vector<Tab> tabs;
  split_tail(items, nsyms, tabs);
  if (tabs.size() < 2)
    return static_cast<double>(nsyms * static_cast<u64>(width)) + 16.0;
  double bits = 0;
  u32 esc_count = 0;
  for (const Tab& t : tabs) {
    if (t.escape) {
      esc_count = t.count;
      bits += static_cast<double>(t.count) *
              (kScaleBits - std::log2(static_cast<double>(t.freq)));
    } else {
      bits += static_cast<double>(t.count) *
              (kScaleBits - std::log2(static_cast<double>(t.freq)));
    }
  }
  // Each escaped symbol also ships its raw bytes.
  bits += static_cast<double>(esc_count) * width * 8.0;
  const double header =
      8.0 + static_cast<double>(tabs.size()) * (width + 2);
  return header + bits / 8.0;
}

// Peaked i32 columns live next to zero. Count the signed 16-bit window in a
// flat table and sort only the outliers, instead of sorting every symbol.
void histogram_i32(const u8* src, u64 nsyms, std::vector<Item>& items) {
  thread_local std::vector<u32> cnt(65536, 0);
  u64 dirty[1024] = {};
  thread_local std::vector<u32> spill;
  spill.clear();
  const u8* p = src;
  for (u64 i = 0; i < nsyms; ++i) {
    i32 v;
    std::memcpy(&v, p, 4);
    p += 4;
    const u32 slot = static_cast<u32>(v + 32768);
    if (slot < 65536u) {
      if (cnt[slot]++ == 0) dirty[slot >> 6] |= (u64{1} << (slot & 63u));
    } else {
      spill.push_back(static_cast<u32>(v));
    }
  }
  items.reserve(std::min(static_cast<usize>(nsyms), usize{4096}));
  for (u32 w = 0; w < 1024u; ++w) {
    u64 m = dirty[w];
    while (m) {
      const u32 b = static_cast<u32>(__builtin_ctzll(m));
      const u32 s = (w << 6) | b;
      const i32 v = static_cast<i32>(s) - 32768;
      items.push_back(Item{static_cast<u64>(static_cast<u32>(v)), cnt[s]});
      cnt[s] = 0;
      m &= m - 1;
    }
  }
  if (spill.empty()) return;
  std::sort(spill.begin(), spill.end());
  u32 prev = spill[0];
  u32 run = 1;
  for (usize i = 1; i < spill.size(); ++i) {
    if (spill[i] == prev) {
      if (run != ~u32{0}) ++run;
    } else {
      items.push_back(Item{prev, run});
      prev = spill[i];
      run = 1;
    }
  }
  items.push_back(Item{prev, run});
}

// Peaked i64 columns (IDs, timestamps, counters) cluster in small positive
// values. Count the [0, 65536) window in a flat table and sort only outliers.
void histogram_i64(const u8* src, u64 nsyms, std::vector<Item>& items) {
  thread_local std::vector<u32> cnt(65536, 0);
  u64 dirty[1024] = {};
  thread_local std::vector<u64> spill;
  spill.clear();
  const u8* p = src;
  for (u64 i = 0; i < nsyms; ++i) {
    u64 v;
    std::memcpy(&v, p, 8);
    p += 8;
    if (v < 65536u) {
      const u32 slot = static_cast<u32>(v);
      if (cnt[slot]++ == 0) dirty[slot >> 6] |= (u64{1} << (slot & 63u));
    } else {
      spill.push_back(v);
    }
  }
  items.reserve(std::min(static_cast<usize>(nsyms), usize{4096}));
  for (u32 w = 0; w < 1024u; ++w) {
    u64 m = dirty[w];
    while (m) {
      const u32 b = static_cast<u32>(__builtin_ctzll(m));
      const u32 s = (w << 6) | b;
      items.push_back(Item{s, cnt[s]});
      cnt[s] = 0;
      m &= m - 1;
    }
  }
  if (spill.empty()) return;
  std::sort(spill.begin(), spill.end());
  u64 prev = spill[0];
  u32 run = 1;
  for (usize i = 1; i < spill.size(); ++i) {
    if (spill[i] == prev) {
      if (run != ~u32{0}) ++run;
    } else {
      items.push_back(Item{prev, run});
      prev = spill[i];
      run = 1;
    }
  }
  items.push_back(Item{prev, run});
}

void histogram(const u8* src, usize n, int width, std::vector<Item>& items, u64& nsyms) {
  items.clear();
  nsyms = n / static_cast<usize>(width);
  if (width == 1) {
    u32 cnt0[256] = {}, cnt1[256] = {}, cnt2[256] = {}, cnt3[256] = {};
    usize i = 0;
    for (; i + 4 <= n; i += 4) {
      ++cnt0[src[i]];
      ++cnt1[src[i + 1]];
      ++cnt2[src[i + 2]];
      ++cnt3[src[i + 3]];
    }
    for (; i < n; ++i) ++cnt0[src[i]];
    for (usize s = 0; s < 256; ++s) {
      const u32 c = cnt0[s] + cnt1[s] + cnt2[s] + cnt3[s];
      if (c) items.push_back(Item{static_cast<u64>(s), c});
    }
    return;
  }
  if (width == 2) {
    thread_local std::vector<u32> cnt(65536, 0);
    u64 dirty[1024] = {};
    u64 i = 0;
    for (; i + 4 <= nsyms; i += 4) {
      u64 q;
      std::memcpy(&q, src + i * 2, 8);
      const u32 s0 = static_cast<u16>(q);
      const u32 s1 = static_cast<u16>(q >> 16);
      const u32 s2 = static_cast<u16>(q >> 32);
      const u32 s3 = static_cast<u16>(q >> 48);
      if (cnt[s0]++ == 0) dirty[s0 >> 6] |= (u64{1} << (s0 & 63u));
      if (cnt[s1]++ == 0) dirty[s1 >> 6] |= (u64{1} << (s1 & 63u));
      if (cnt[s2]++ == 0) dirty[s2 >> 6] |= (u64{1} << (s2 & 63u));
      if (cnt[s3]++ == 0) dirty[s3 >> 6] |= (u64{1} << (s3 & 63u));
    }
    for (; i < nsyms; ++i) {
      u16 s;
      std::memcpy(&s, src + i * 2, 2);
      if (cnt[s]++ == 0) dirty[s >> 6] |= (u64{1} << (s & 63u));
    }
    for (u32 w = 0; w < 1024u; ++w) {
      u64 m = dirty[w];
      while (m) {
        const u32 b = static_cast<u32>(__builtin_ctzll(m));
        const u32 s = (w << 6) | b;
        items.push_back(Item{static_cast<u64>(s), cnt[s]});
        cnt[s] = 0;
        m &= m - 1;
      }
    }
    return;
  }
  if (width == 4) {
    histogram_i32(src, nsyms, items);
    return;
  }
  histogram_i64(src, nsyms, items);
}

struct Map1 {
  u32 id[256];
  explicit Map1(u32 esc) { std::fill(id, id + 256, esc); }
  void set(u64 sym, u32 i) { id[static_cast<usize>(sym)] = i; }
  GRATING_INLINE u32 get(u64 sym) const { return id[static_cast<usize>(sym)]; }
};

struct Map2 {
  u32* id = nullptr;
  u32 esc;
  explicit Map2(u32 escape) : esc(escape) {
    thread_local std::vector<u32> buf(65536, ~u32{0});
    id = buf.data();
  }
  void set(u64 sym, u32 i) { id[static_cast<usize>(sym)] = i; }
  void clear(u64 sym) { id[static_cast<usize>(sym)] = ~u32{0}; }
  GRATING_INLINE u32 get(u64 sym) const {
    const u32 v = id[static_cast<usize>(sym)];
    return v == ~u32{0} ? esc : v;
  }
};

// In-window i32 symbols (the peaked core) resolve in a flat table. Outliers
// are a short list: the escape tail is rarely kept.
struct Map4 {
  u32* id = nullptr;
  std::vector<u64> extra_sym;
  std::vector<u32> extra_id;
  u32 esc;
  explicit Map4(u32 escape) : esc(escape) {
    thread_local std::vector<u32> buf(65536, ~u32{0});
    id = buf.data();
  }
  static GRATING_INLINE u32 slot_of(u64 sym) {
    const i32 v = static_cast<i32>(static_cast<u32>(sym));
    return static_cast<u32>(v + 32768);
  }
  void set(u64 sym, u32 i) {
    const u32 slot = slot_of(sym);
    if (slot < 65536u)
      id[slot] = i;
    else {
      extra_sym.push_back(sym);
      extra_id.push_back(i);
    }
  }
  void clear(u64 sym) {
    const u32 slot = slot_of(sym);
    if (slot < 65536u) id[slot] = ~u32{0};
  }
  GRATING_INLINE u32 get(u64 sym) const {
    const u32 slot = slot_of(sym);
    if (slot < 65536u) {
      const u32 v = id[slot];
      return v == ~u32{0} ? esc : v;
    }
    for (usize i = 0; i < extra_sym.size(); ++i)
      if (extra_sym[i] == sym) return extra_id[i];
    return esc;
  }
};

struct MapH {
  struct Slot {
    u64 key = 0;
    u32 id = 0;
    bool used = false;
  };
  u32 small_id[4096];
  std::vector<Slot> tab;
  u32 mask = 0;
  u32 esc = 0;
  explicit MapH(usize n, u32 escape) : esc(escape) {
    std::fill(small_id, small_id + 4096, escape);
    usize cap = 16;
    while (cap < n * 2 + 8) cap <<= 1;
    tab.resize(cap);
    mask = static_cast<u32>(cap - 1);
  }
  void set(u64 sym, u32 i) {
    if (sym < 4096u) {
      small_id[static_cast<usize>(sym)] = i;
      return;
    }
    u32 k = static_cast<u32>(sym * 0x9E3779B97F4A7C15ull) & mask;
    while (tab[k].used) k = (k + 1) & mask;
    tab[k] = Slot{sym, i, true};
  }
  GRATING_INLINE u32 get(u64 sym) const {
    if (sym < 4096u) return small_id[static_cast<usize>(sym)];
    if (tab.empty()) return esc;
    u32 k = static_cast<u32>(sym * 0x9E3779B97F4A7C15ull) & mask;
    while (tab[k].used) {
      if (tab[k].key == sym) return tab[k].id;
      k = (k + 1) & mask;
    }
    return esc;
  }
};

template <int W, int NWay, class Map>
GRATING_INLINE void encode_streams_body(const u8* src, u64 nsyms, const Map& map, const EncSym* enc,
                                        u32 esc_id, u32 esc_count, u8* buf, u8* stream_ptr[NWay],
                                        usize stream_len[NWay], std::vector<u8>& raw) {
  const usize per = 2 * (static_cast<usize>(nsyms) / static_cast<usize>(NWay) + 1) + 64;
  u8* ends[NWay];
  u8* ptrs[NWay];
  u32 x[NWay];
  for (int k = 0; k < NWay; ++k) {
    ends[k] = buf + per * static_cast<usize>(k + 1);
    ptrs[k] = ends[k];
    x[k] = kRansL16;
  }
  raw.resize(static_cast<usize>(esc_count) * static_cast<usize>(W));
  u8* rptr = raw.data() + raw.size();
  u64 idx = nsyms;
  constexpr usize w = static_cast<usize>(NWay);
  if (esc_count == 0) {
    while (idx % w != 0) {
      --idx;
      const u64 sym = load_sym_w<W>(src + idx * static_cast<u64>(W));
      rans_put16(x[idx % w], ptrs[idx % w], enc[map.get(sym)]);
    }
    while (idx > 0) {
      idx -= w;
      const u8* base = src + idx * static_cast<u64>(W);
      for (int k = NWay - 1; k >= 0; --k) {
        const u64 sym = load_sym_w<W>(base + static_cast<usize>(k * W));
        rans_put16(x[k], ptrs[k], enc[map.get(sym)]);
      }
    }
  } else {
    while (idx % w != 0) {
      --idx;
      const u64 sym = load_sym_w<W>(src + idx * static_cast<u64>(W));
      const u32 id = map.get(sym);
      if (__builtin_expect(id == esc_id, 0)) {
        rptr -= W;
        store_sym_w<W>(rptr, sym);
      }
      rans_put16(x[idx % w], ptrs[idx % w], enc[id]);
    }
    while (idx > 0) {
      idx -= w;
      const u8* base = src + idx * static_cast<u64>(W);
      for (int k = NWay - 1; k >= 0; --k) {
        const u64 sym = load_sym_w<W>(base + static_cast<usize>(k * W));
        const u32 id = map.get(sym);
        if (__builtin_expect(id == esc_id, 0)) {
          rptr -= W;
          store_sym_w<W>(rptr, sym);
        }
        rans_put16(x[k], ptrs[k], enc[id]);
      }
    }
  }
  for (int k = 0; k < NWay; ++k) {
    u8* ptr = ptrs[k];
    *--ptr = static_cast<u8>(x[k] >> 24);
    *--ptr = static_cast<u8>(x[k] >> 16);
    *--ptr = static_cast<u8>(x[k] >> 8);
    *--ptr = static_cast<u8>(x[k]);
    stream_ptr[k] = ptr;
    stream_len[k] = static_cast<usize>(ends[k] - ptr);
  }
}

#if GRATING_X86_BMI2
template <int W, int NWay, class Map>
GRATING_BMI2_FN void encode_streams_bmi2(const u8* src, u64 nsyms, const Map& map,
                                         const EncSym* enc, u32 esc_id, u32 esc_count, u8* buf,
                                         u8* stream_ptr[NWay], usize stream_len[NWay],
                                         std::vector<u8>& raw) {
  encode_streams_body<W, NWay, Map>(src, nsyms, map, enc, esc_id, esc_count, buf, stream_ptr,
                                    stream_len, raw);
}
#endif

template <int W, int NWay, class Map>
void encode_streams(const u8* src, u64 nsyms, const Map& map, const EncSym* enc, u32 esc_id,
                    u32 esc_count, u8* buf, u8* stream_ptr[NWay], usize stream_len[NWay],
                    std::vector<u8>& raw) {
#if GRATING_X86_BMI2
  if (grating_has_bmi2()) {
    encode_streams_bmi2<W, NWay, Map>(src, nsyms, map, enc, esc_id, esc_count, buf, stream_ptr,
                                      stream_len, raw);
    return;
  }
#endif
  encode_streams_body<W, NWay, Map>(src, nsyms, map, enc, esc_id, esc_count, buf, stream_ptr,
                                    stream_len, raw);
}

template <int W, class Map>
void write_coded(std::vector<u8>& out, const u8* src, u64 nsyms, int nway,
                 const std::vector<Tab>& tabs, const Map& map) {
  thread_local std::vector<EncSym> enc;
  enc.resize(tabs.size());
  for (usize i = 0; i < tabs.size(); ++i)
    init_enc_sym(enc[i], tabs[i].freq, tabs[i].start);
  u32 esc_count = 0;
  u32 esc_id = 0;
  for (usize i = 0; i < tabs.size(); ++i)
    if (tabs[i].escape) {
      esc_count = tabs[i].count;
      esc_id = static_cast<u32>(i);
    }
  const usize per = 2 * (static_cast<usize>(nsyms) / static_cast<usize>(nway) + 1) + 64;
  thread_local std::vector<u8> buf;
  if (buf.size() < per * static_cast<usize>(nway))
    buf.resize(per * static_cast<usize>(nway));
  u8* stream_ptr[8] = {};
  usize stream_len[8] = {};
  thread_local std::vector<u8> raw;
  if (nway == 8)
    encode_streams<W, 8>(src, nsyms, map, enc.data(), esc_id, esc_count, buf.data(), stream_ptr,
                         stream_len, raw);
  else
    encode_streams<W, 4>(src, nsyms, map, enc.data(), esc_id, esc_count, buf.data(), stream_ptr,
                         stream_len, raw);
  {
    ByteWriter w(out);
    w.varint(raw.size());
    for (int k = 0; k + 1 < nway; ++k) w.varint(stream_len[k]);
  }
  usize total_streams = 0;
  for (int k = 0; k < nway; ++k) total_streams += stream_len[k];
  const usize off0 = out.size();
  out.resize(off0 + total_streams + raw.size());
  u8* dst = out.data() + off0;
  for (int k = 0; k < nway; ++k) {
    std::memcpy(dst, stream_ptr[k], stream_len[k]);
    dst += stream_len[k];
  }
  if (!raw.empty()) std::memcpy(dst, raw.data(), raw.size());
}

void encode_width(std::vector<u8>& out, const u8* src, u64 nsyms, int width,
                  const std::vector<Item>& items) {
  if (items.size() <= 1) {
    out.push_back(static_cast<u8>(width));
    out.push_back(0);
    u64 sym = items.empty() ? 0 : items[0].sym;
    u8 buf[8] = {};
    store_sym(buf, width, sym);
    out.insert(out.end(), buf, buf + width);
    return;
  }
  std::vector<Tab> tabs;
  split_tail(items, nsyms, tabs);
  if (tabs.size() < 2) {
    out.clear();
    return;
  }
  const int nway = nsyms >= 4096 ? 8 : 4;
  out.push_back(static_cast<u8>(width));
  out.push_back(static_cast<u8>(nway));
  u32 n_kept = 0;
  u32 esc_freq = 0;
  for (const Tab& t : tabs) {
    if (t.escape) esc_freq = t.freq;
    else ++n_kept;
  }
  out.push_back(static_cast<u8>(n_kept));
  out.push_back(static_cast<u8>(n_kept >> 8));
  out.push_back(static_cast<u8>(esc_freq));
  out.push_back(static_cast<u8>(esc_freq >> 8));
  for (const Tab& t : tabs) {
    if (t.escape) continue;
    u8 buf[8] = {};
    store_sym(buf, width, t.sym);
    out.insert(out.end(), buf, buf + width);
    out.push_back(static_cast<u8>(t.freq));
    out.push_back(static_cast<u8>(t.freq >> 8));
  }
  const u32 esc_id = n_kept;
  if (width == 1) {
    Map1 map(esc_freq ? esc_id : 0);
    for (usize i = 0; i < tabs.size(); ++i)
      if (!tabs[i].escape) map.set(tabs[i].sym, static_cast<u32>(i));
    write_coded<1>(out, src, nsyms, nway, tabs, map);
  } else if (width == 2) {
    Map2 map(esc_freq ? esc_id : 0);
    for (usize i = 0; i < tabs.size(); ++i)
      if (!tabs[i].escape) map.set(tabs[i].sym, static_cast<u32>(i));
    write_coded<2>(out, src, nsyms, nway, tabs, map);
    for (usize i = 0; i < tabs.size(); ++i)
      if (!tabs[i].escape) map.clear(tabs[i].sym);
  } else if (width == 4) {
    Map4 map(esc_freq ? esc_id : 0);
    for (usize i = 0; i < tabs.size(); ++i)
      if (!tabs[i].escape) map.set(tabs[i].sym, static_cast<u32>(i));
    write_coded<4>(out, src, nsyms, nway, tabs, map);
    for (usize i = 0; i < tabs.size(); ++i)
      if (!tabs[i].escape) map.clear(tabs[i].sym);
  } else {
    MapH map(tabs.size(), esc_freq ? esc_id : 0);
    for (usize i = 0; i < tabs.size(); ++i)
      if (!tabs[i].escape) map.set(tabs[i].sym, static_cast<u32>(i));
    write_coded<8>(out, src, nsyms, nway, tabs, map);
  }
}

// For W <= 4, dsym[s] packs:
//   bits  0..11: freq
//   bits 12..23: start
//   bit      24: is_escape
//   bits 32..63: 32-bit symbol value (so W <= 4 needs no second load)
// For W == 8, bits 24..39 hold the symbol index into values[].
template <int N, int W, bool Esc>
GRATING_INLINE bool rans_run_body(const u64* dsym, const u64* values, u32 escape_id, u32* xs,
                                  const u8** ps, const u8* plast, const u8*& raw,
                                  const u8* raw_end, u8* dst, usize nsyms) {
  u32 x[N];
  const u8* p[N];
  for (int k = 0; k < N; ++k) {
    x[k] = xs[k];
    p[k] = ps[k];
  }
  const u8* rptr = raw;
  const u8* pfast = plast - 16;
  auto emit_val = [&](u8* d, u64 e) {
    if constexpr (W <= 4) {
      store_sym_w<W>(d, e >> 32);
    } else {
      store_sym_w<W>(d, values[static_cast<u32>(e >> 24)]);
    }
  };
  auto is_esc = [&](u64 e) -> bool {
    if constexpr (W <= 4) {
      return (static_cast<u32>(e) & (1u << 24)) != 0;
    } else {
      return static_cast<u32>(e >> 24) == escape_id;
    }
  };
  if constexpr (!Esc) {
    usize i = 0;
    while (i + static_cast<usize>(8 * N) <= nsyms && p[N - 1] < pfast) {
      for (int rep = 0; rep < 8; ++rep, i += static_cast<usize>(N)) {
        u8* base = dst + i * static_cast<usize>(W);
        for (int k = 0; k < N; ++k) {
          constexpr u32 mask = kM - 1;
          const u32 s = x[k] & mask;
          const u64 e = dsym[s];
          const u32 f = static_cast<u32>(e) & mask;
          const u32 start = static_cast<u32>(e >> 12) & mask;
          x[k] = f * (x[k] >> kScaleBits) + s - start;
          rans_renorm_fast(x[k], p[k]);
          emit_val(base + static_cast<usize>(k * W), e);
        }
      }
    }
    for (; i + static_cast<usize>(N) <= nsyms; i += static_cast<usize>(N)) {
      u8* base = dst + i * static_cast<usize>(W);
      for (int k = 0; k < N; ++k) {
        constexpr u32 mask = kM - 1;
        const u32 s = x[k] & mask;
        const u64 e = dsym[s];
        const u32 f = static_cast<u32>(e) & mask;
        const u32 start = static_cast<u32>(e >> 12) & mask;
        x[k] = f * (x[k] >> kScaleBits) + s - start;
        rans_renorm(x[k], p[k], plast);
        emit_val(base + static_cast<usize>(k * W), e);
      }
    }
    for (; i < nsyms; ++i) {
      const int k = static_cast<int>(i % static_cast<usize>(N));
      constexpr u32 mask = kM - 1;
      const u32 s = x[k] & mask;
      const u64 e = dsym[s];
      const u32 f = static_cast<u32>(e) & mask;
      const u32 start = static_cast<u32>(e >> 12) & mask;
      x[k] = f * (x[k] >> kScaleBits) + s - start;
      rans_renorm(x[k], p[k], plast);
      emit_val(dst + i * static_cast<usize>(W), e);
    }
  } else {
    usize i = 0;
    while (i + static_cast<usize>(8 * N) <= nsyms && p[N - 1] < pfast &&
           rptr + static_cast<usize>(8 * N * W) <= raw_end) {
      for (int rep = 0; rep < 8; ++rep, i += static_cast<usize>(N)) {
        u8* base = dst + i * static_cast<usize>(W);
        for (int k = 0; k < N; ++k) {
          constexpr u32 mask = kM - 1;
          const u32 s = x[k] & mask;
          const u64 e = dsym[s];
          const u32 f = static_cast<u32>(e) & mask;
          const u32 start = static_cast<u32>(e >> 12) & mask;
          x[k] = f * (x[k] >> kScaleBits) + s - start;
          rans_renorm_fast(x[k], p[k]);
          u8* d = base + static_cast<usize>(k * W);
          if (__builtin_expect(is_esc(e), 0)) {
            std::memcpy(d, rptr, W);
            rptr += W;
          } else {
            emit_val(d, e);
          }
        }
      }
    }
    for (; i + static_cast<usize>(N) <= nsyms; i += static_cast<usize>(N)) {
      u8* base = dst + i * static_cast<usize>(W);
      if (rptr + static_cast<usize>(N * W) <= raw_end) {
        for (int k = 0; k < N; ++k) {
          constexpr u32 mask = kM - 1;
          const u32 s = x[k] & mask;
          const u64 e = dsym[s];
          const u32 f = static_cast<u32>(e) & mask;
          const u32 start = static_cast<u32>(e >> 12) & mask;
          x[k] = f * (x[k] >> kScaleBits) + s - start;
          rans_renorm(x[k], p[k], plast);
          u8* d = base + static_cast<usize>(k * W);
          if (__builtin_expect(is_esc(e), 0)) {
            std::memcpy(d, rptr, W);
            rptr += W;
          } else {
            emit_val(d, e);
          }
        }
      } else {
        for (int k = 0; k < N; ++k) {
          constexpr u32 mask = kM - 1;
          const u32 s = x[k] & mask;
          const u64 e = dsym[s];
          const u32 f = static_cast<u32>(e) & mask;
          const u32 start = static_cast<u32>(e >> 12) & mask;
          x[k] = f * (x[k] >> kScaleBits) + s - start;
          rans_renorm(x[k], p[k], plast);
          u8* d = base + static_cast<usize>(k * W);
          if (is_esc(e)) {
            if (rptr + W > raw_end) return false;
            std::memcpy(d, rptr, W);
            rptr += W;
          } else {
            emit_val(d, e);
          }
        }
      }
    }
    for (; i < nsyms; ++i) {
      const int k = static_cast<int>(i % static_cast<usize>(N));
      constexpr u32 mask = kM - 1;
      const u32 s = x[k] & mask;
      const u64 e = dsym[s];
      const u32 f = static_cast<u32>(e) & mask;
      const u32 start = static_cast<u32>(e >> 12) & mask;
      x[k] = f * (x[k] >> kScaleBits) + s - start;
      rans_renorm(x[k], p[k], plast);
      u8* d = dst + i * static_cast<usize>(W);
      if (is_esc(e)) {
        if (rptr + W > raw_end) return false;
        std::memcpy(d, rptr, W);
        rptr += W;
      } else {
        emit_val(d, e);
      }
    }
  }
  raw = rptr;
  for (int k = 0; k < N; ++k) {
    xs[k] = x[k];
    ps[k] = p[k];
  }
  return true;
}

#if GRATING_X86_BMI2
template <int N, int W, bool Esc>
GRATING_BMI2_FN bool rans_run_bmi2(const u64* dsym, const u64* values, u32 escape_id, u32* xs,
                                   const u8** ps, const u8* plast, const u8*& raw,
                                   const u8* raw_end, u8* dst, usize nsyms) {
  return rans_run_body<N, W, Esc>(dsym, values, escape_id, xs, ps, plast, raw, raw_end, dst, nsyms);
}
#endif

template <int N, int W, bool Esc>
bool rans_run(const u64* dsym, const u64* values, u32 escape_id, u32* xs, const u8** ps,
              const u8* plast, const u8*& raw, const u8* raw_end, u8* dst, usize nsyms) {
#if GRATING_X86_BMI2
  if (grating_has_bmi2())
    return rans_run_bmi2<N, W, Esc>(dsym, values, escape_id, xs, ps, plast, raw, raw_end, dst,
                                    nsyms);
#endif
  return rans_run_body<N, W, Esc>(dsym, values, escape_id, xs, ps, plast, raw, raw_end, dst, nsyms);
}

}  // namespace

// Rank the four integer widths on a mid-block sample, then histogram and
// encode only the winner. A full histogram of every width is several times
// the cost of the rANS pass on a peaked column.
int select_width(const u8* src, usize n) {
  int best = 1;
  double best_score = 1e300;
  constexpr usize kSample = 8192;
  const bool sampled = n > kSample;
  const usize win = sampled ? kSample : n;
  const usize off = sampled ? (((n - win) / 2) & ~usize{7}) : 0;
  for (int w : {1, 2, 4, 8}) {
    if ((n % static_cast<usize>(w)) != 0 || (win % static_cast<usize>(w)) != 0) continue;
    std::vector<Item> items;
    u64 nsyms = 0;
    histogram(src + off, win, w, items, nsyms);
    // Keep only symbols that repeat in the sample window; scale their counts
    // to the full page so split_tail applies the real keep cutoff.
    if (sampled && items.size() > 1 && nsyms > 0) {
      const u64 full = n / static_cast<usize>(w);
      const double scale = static_cast<double>(full) / static_cast<double>(nsyms);
      std::vector<Item> scaled;
      scaled.reserve(items.size());
      for (const Item& it : items) {
        if (it.count < 2) continue;
        const double c = static_cast<double>(it.count) * scale;
        u32 cnt = static_cast<u32>(std::min(c, static_cast<double>(~u32{0})));
        if (cnt == 0) cnt = 1;
        scaled.push_back(Item{it.sym, cnt});
      }
      if (scaled.size() < 2) continue;
      items.swap(scaled);
      nsyms = full;
    }
    const double score = estimate(items, nsyms, w) / static_cast<double>(n);
    if (score < best_score) {
      best_score = score;
      best = w;
    }
  }
  return best;
}

void build_grating_block(const u8* src, usize n, std::vector<u8>& payload) {
  payload.clear();
  if (n == 0) return;
  auto encode_at = [&](int w) {
    payload.clear();
    std::vector<Item> items;
    u64 nsyms = 0;
    histogram(src, n, w, items, nsyms);
    encode_width(payload, src, nsyms, w, items);
  };
  const int w = select_width(src, n);
  encode_at(w);
  if (!payload.empty()) return;
  // The sample can prefer a width whose full histogram is all escape.
  // Try the other widths before leaving the block to be stored.
  for (int alt : {1, 2, 4, 8}) {
    if (alt == w || (n % static_cast<usize>(alt)) != 0) continue;
    encode_at(alt);
    if (!payload.empty()) return;
  }
}

Status decode_grating_block(const u8* data, usize len, u8* dst, usize n) {
  if (n == 0) return len == 0 ? Status::Ok : Status::CorruptInput;
  ByteReader r(data, len);
  const int width = r.byte();
  const int mode = r.byte();
  if (r.error || (width != 1 && width != 2 && width != 4 && width != 8))
    return Status::CorruptInput;
  if (n % static_cast<usize>(width) != 0) return Status::CorruptInput;
  const u64 nsyms = n / static_cast<usize>(width);

  if (mode == 0) {
    const u8* sym = r.take(static_cast<usize>(width));
    if (!sym || r.remaining() != 0) return Status::CorruptInput;
    for (u64 i = 0; i < nsyms; ++i)
      std::memcpy(dst + i * static_cast<u64>(width), sym, static_cast<usize>(width));
    return Status::Ok;
  }
  if (mode != 4 && mode != 8) return Status::CorruptInput;
  const int nway = mode;
  const u32 n_kept = r.byte() | (static_cast<u32>(r.byte()) << 8);
  const u32 esc_freq = r.byte() | (static_cast<u32>(r.byte()) << 8);
  if (r.error || n_kept == 0 || n_kept > static_cast<u32>(kMaxKept))
    return Status::CorruptInput;

  thread_local std::vector<u64> values;
  values.resize(n_kept + 1);
  values[n_kept] = 0;
  thread_local std::vector<u32> freq;
  freq.resize(n_kept);
  u32 sum = esc_freq;
  for (u32 i = 0; i < n_kept; ++i) {
    const u8* sp = r.take(static_cast<usize>(width));
    if (!sp) return Status::CorruptInput;
    values[i] = load_sym(sp, width);
    const u32 f = r.byte() | (static_cast<u32>(r.byte()) << 8);
    if (r.error || f == 0 || f >= kM) return Status::CorruptInput;
    freq[i] = f;
    sum += f;
  }
  if (sum != kM || (esc_freq == 0 && n_kept < 2)) return Status::CorruptInput;

  thread_local std::vector<u64> dsym;
  if (dsym.size() < kM) dsym.resize(kM);
  u32 cum = 0;
  if (width <= 4) {
    for (u32 i = 0; i < n_kept; ++i) {
      const u64 packed =
          freq[i] | (static_cast<u64>(cum) << 12) | (static_cast<u64>(values[i]) << 32);
      for (u32 k = 0; k < freq[i]; ++k) dsym[cum + k] = packed;
      cum += freq[i];
    }
    if (esc_freq) {
      const u64 packed = esc_freq | (static_cast<u64>(cum) << 12) | (u64{1} << 24);
      for (u32 k = 0; k < esc_freq; ++k) dsym[cum + k] = packed;
    }
  } else {
    for (u32 i = 0; i < n_kept; ++i) {
      const u64 packed = freq[i] | (static_cast<u64>(cum) << 12) | (static_cast<u64>(i) << 24);
      for (u32 k = 0; k < freq[i]; ++k) dsym[cum + k] = packed;
      cum += freq[i];
    }
    if (esc_freq) {
      const u32 id = n_kept;
      const u64 packed = esc_freq | (static_cast<u64>(cum) << 12) | (static_cast<u64>(id) << 24);
      for (u32 k = 0; k < esc_freq; ++k) dsym[cum + k] = packed;
    }
  }

  const usize raw_len = static_cast<usize>(r.varint());
  if (r.error || raw_len > r.remaining() || (esc_freq == 0 && raw_len != 0))
    return Status::CorruptInput;
  usize sizes[8] = {};
  usize prefix = 0;
  for (int k = 0; k + 1 < nway; ++k) {
    sizes[k] = static_cast<usize>(r.varint());
    if (r.error || sizes[k] > r.remaining() - raw_len - prefix) return Status::CorruptInput;
    prefix += sizes[k];
  }
  if (r.remaining() < raw_len + prefix) return Status::CorruptInput;
  sizes[nway - 1] = r.remaining() - raw_len - prefix;

  const u8* streams = r.p;
  const u8* ps[8];
  u32 xs[8];
  const u8* p = streams;
  for (int k = 0; k < nway; ++k) {
    if (sizes[k] < 4) return Status::CorruptInput;
    ps[k] = p;
    xs[k] = load_u32(ps[k]);
    ps[k] += 4;
    p += sizes[k];
  }
  const u8* plast = p - 2;
  const u8* raw = p;
  const u8* raw_end = p + raw_len;
  if (raw_end != r.end) return Status::CorruptInput;

  bool ok = false;
  const bool esc = esc_freq != 0;
  const u32 esc_id = n_kept;
  auto go = [&](auto width_tag, auto esc_tag) {
    constexpr int W = decltype(width_tag)::value;
    constexpr bool E = decltype(esc_tag)::value;
    if (nway == 8)
      ok = rans_run<8, W, E>(dsym.data(), values.data(), esc_id, xs, ps, plast, raw, raw_end,
                             dst, static_cast<usize>(nsyms));
    else
      ok = rans_run<4, W, E>(dsym.data(), values.data(), esc_id, xs, ps, plast, raw, raw_end,
                             dst, static_cast<usize>(nsyms));
  };
  if (width == 1 && !esc) go(std::integral_constant<int, 1>{}, std::false_type{});
  else if (width == 1 && esc) go(std::integral_constant<int, 1>{}, std::true_type{});
  else if (width == 2 && !esc) go(std::integral_constant<int, 2>{}, std::false_type{});
  else if (width == 2 && esc) go(std::integral_constant<int, 2>{}, std::true_type{});
  else if (width == 4 && !esc) go(std::integral_constant<int, 4>{}, std::false_type{});
  else if (width == 4 && esc) go(std::integral_constant<int, 4>{}, std::true_type{});
  else if (width == 8 && !esc) go(std::integral_constant<int, 8>{}, std::false_type{});
  else go(std::integral_constant<int, 8>{}, std::true_type{});
  if (!ok || raw != raw_end) return Status::CorruptInput;
  return Status::Ok;
}

}  // namespace spectra8::detail
