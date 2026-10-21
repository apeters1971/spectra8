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

#include "util/tans.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include "util/bits.hpp"
#include "util/huffman.hpp"

namespace spectra8::detail::ans {
namespace {

constexpr int kNumSymbols = 256;
constexpr int kScaleBits = 12;
constexpr u32 kM = 1u << kScaleBits;  // total frequency
// Lower bound of the normalized state interval: [2^16, 2^32) with 16-bit
// renormalization. With 12-bit frequencies at most one 16-bit word moves per
// symbol in either direction, so the decoder's renorm is one 16-bit load and
// a select instead of a byte loop - about 2x the decode speed of the earlier
// byte-wise renorm from [2^23, 2^31) at the same payload size (+0.2%).
constexpr u32 kRansL16 = 1u << 16;

// Normalize symbol counts so they sum exactly to kM, keeping used symbols >= 1.
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
    // Give the surplus to the most frequent symbol.
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
      u32 take = min_of(excess, freq[best] - 1);
      freq[best] -= take;
      excess -= take;
    }
  }
}

// Frequency table coding (the order-0 table and each order-1 class table):
// every frequency is sent as its bit length (class 0..13, under one
// huff::SmallCode) followed by the bits below the leading one. Against the
// u16-per-symbol table this is ~1 byte per symbol on a dense 256-symbol
// alphabet and 2-4 bits per absent symbol, so a token table shrinks 2-3x
// and a literal table 2x; the decoder's parse is linear in num_syms.
constexpr int kFreqClasses = 14;  // bit lengths 0..13 (13: freq == kM)

inline int freq_class(u32 f) noexcept { return f ? 32 - __builtin_clz(f) : 0; }

void class_hist(u32 cf[kFreqClasses], const u32* freq, int num_syms) {
  for (int s = 0; s < num_syms; ++s) ++cf[freq_class(freq[s])];
}

void put_freqs(BitWriter& w, const huff::SmallCode& sc, const u32* freq, int num_syms) {
  for (int s = 0; s < num_syms; ++s) {
    const int c = freq_class(freq[s]);
    huff::small_code_put(w, sc, c);
    if (c >= 2) w.put(freq[s] & ((1u << (c - 1)) - 1u), c - 1);
  }
}

// Reads num_syms frequencies; `bits` accumulates the bits consumed.
bool get_freqs(BitReader& r, const huff::SmallCode& sc, u32* freq, int num_syms, usize& bits) {
  for (int s = 0; s < num_syms; ++s) {
    const int c = huff::small_code_get(r, sc);
    if (c < 0 || c >= kFreqClasses) return false;
    bits += sc.len[c];
    u32 f = static_cast<u32>(c != 0);
    if (c >= 2) {
      f = (1u << (c - 1)) | r.get(c - 1);
      bits += static_cast<usize>(c - 1);
    }
    freq[s] = f;
  }
  return true;
}

// Order-0 table header: u8 num_syms - 1, then the SmallCode lengths (14 x 3
// bits) and the coded frequencies, zero-padded to a byte.
void write_freq_table(std::vector<u8>& out, const u32* freq, int num_syms) {
  out.push_back(static_cast<u8>(num_syms - 1));
  u32 cf[kFreqClasses] = {};
  class_hist(cf, freq, num_syms);
  huff::SmallCode sc;
  // 14 symbols never need more than 7 bits here: the classes' counts are
  // bounded by 256 symbols, far from the Fibonacci-like skew that would.
  huff::small_code_build(cf, kFreqClasses, sc);
  BitWriter w(out);
  huff::small_code_write(w, sc, kFreqClasses);
  put_freqs(w, sc, freq, num_syms);
  w.flush();
}

bool read_freq_table(ByteReader& r, u32 freq[kNumSymbols], int& num_syms) {
  num_syms = r.byte() + 1;
  if (r.error) return false;
  BitReader br(r.p, r.remaining());
  huff::SmallCode sc;
  if (!huff::small_code_read(br, kFreqClasses, sc)) return false;
  usize bits = static_cast<usize>(kFreqClasses) * 3;
  std::fill(freq, freq + kNumSymbols, u32{0});
  if (!get_freqs(br, sc, freq, num_syms, bits)) return false;
  return r.take((bits + 7) / 8) != nullptr;
}

}  // namespace

void encode(std::vector<u8>& out, const u8* src, usize n) {
  u32 counts[kNumSymbols] = {0};
  for (usize i = 0; i < n; ++i) ++counts[src[i]];
  encode(out, src, n, counts);
}

namespace {

// rANS "put" for one symbol (encode is reverse order, renorm words go
// backward into *ptr). x < 2^32 and x_max = f<<20, so a single 16-bit
// emission always brings x below x_max. x_max is 64-bit because a class with
// a single symbol (order-1 coder) has f == kM, where f<<20 == 2^32.
// Per-symbol encoder constants: the division x / f is replaced by a
// multiply-high with a precomputed reciprocal (Granlund-Montgomery: with
// sh = ceil(log2 f) and rcp = floor(2^(32+sh) / f) + 1, (x * rcp) >> (32+sh)
// equals x / f for every x < 2^32). rcp needs up to 33 bits, so the product
// is formed in 128 bits (one umulh/mulx on 64-bit targets); f == 1 is
// special-cased (sh would be 0 and rcp 2^33).
struct EncSym {
  u64 rcp;
  u32 f;
  u32 start;
  u32 sh;
};

inline void init_enc_sym(EncSym& e, u32 f, u32 start) {
  e.f = f;
  e.start = start;
  if (f <= 1) {
    e.rcp = 0;
    e.sh = 0;
    return;
  }
  u32 sh = 0;
  while ((u32{1} << sh) < f) ++sh;
  e.sh = sh;
  e.rcp = ((u64{1} << (32 + sh)) / f) + 1;
}

inline u32 div_by_freq(u32 x, const EncSym& e) {
#if defined(__SIZEOF_INT128__)
  if (e.f <= 1) return x;
  const unsigned __int128 prod =
      static_cast<unsigned __int128>(x) * static_cast<unsigned __int128>(e.rcp);
  return static_cast<u32>(prod >> (32 + e.sh));
#else
  return x / e.f;
#endif
}

inline void rans_put16(u32& x, u8*& ptr, const EncSym& e) {
  constexpr u32 x_max_base = (kRansL16 >> kScaleBits) << 16;  // = 1<<20
  const u64 x_max = static_cast<u64>(x_max_base) * e.f;
  if (x >= x_max) {
    ptr -= 2;
    ptr[0] = static_cast<u8>(x);
    ptr[1] = static_cast<u8>(x >> 8);
    x >>= 16;
  }
  const u32 q = div_by_freq(x, e);
  x = (q << kScaleBits) + (x - q * e.f) + e.start;
}

// Division form for the order-1 coder (per-context tables, BWT streams only).
inline void rans_put16(u32& x, u8*& ptr, u32 f, u32 start) {
  constexpr u32 x_max_base = (kRansL16 >> kScaleBits) << 16;  // = 1<<20
  const u64 x_max = static_cast<u64>(x_max_base) * f;
  if (x >= x_max) {
    ptr -= 2;
    ptr[0] = static_cast<u8>(x);
    ptr[1] = static_cast<u8>(x >> 8);
    x >>= 16;
  }
  x = ((x / f) << kScaleBits) + (x % f) + start;
}

// Encode `src` as nway independent rANS streams (16-bit renorm): symbol i
// belongs to stream i % nway, and rANS is LIFO, so one backward pass over
// the data feeds all states - they are data-independent, and the pass reads
// the input once instead of once per stream. Stream k's bytes are returned
// in outbuf[k].
void encode_streams(const u8* src, usize n, int nway, const EncSym syms[256],
                    std::vector<u8> outbuf[kMaxWays]) {
  // Worst case per stream: one 16-bit renorm per symbol + state flush.
  const usize per = 2 * (n / static_cast<usize>(nway) + 1) + 64;
  std::vector<u8> buf(per * static_cast<usize>(nway));
  u8* ends[kMaxWays];
  u8* ptrs[kMaxWays];
  u32 x[kMaxWays];
  for (int k = 0; k < nway; ++k) {
    ends[k] = buf.data() + per * static_cast<usize>(k + 1);
    ptrs[k] = ends[k];
    x[k] = kRansL16;
  }
  const usize w = static_cast<usize>(nway);
  usize idx = n;
  // Unaligned tail first (the pass runs backwards), then whole groups of nway
  // symbols starting at multiples of nway, so symbol idx always feeds state
  // idx % nway.
  while (idx % w != 0) {
    --idx;
    rans_put16(x[idx % w], ptrs[idx % w], syms[src[idx]]);
  }
  while (idx > 0) {
    idx -= w;
    const u8* g = src + idx;
    for (int k = nway - 1; k >= 0; --k)
      rans_put16(x[k], ptrs[k], syms[g[k]]);
  }
  for (int k = 0; k < nway; ++k) {
    u8* ptr = ptrs[k];
    *--ptr = static_cast<u8>(x[k] >> 24);
    *--ptr = static_cast<u8>(x[k] >> 16);
    *--ptr = static_cast<u8>(x[k] >> 8);
    *--ptr = static_cast<u8>(x[k] >> 0);
    outbuf[k].assign(ptr, ends[k]);
  }
}

}  // namespace

usize encoded_size(const u32 counts[kNumSymbols], usize n) {
  int num_syms = 1;
  int distinct = 0;
  for (int s = 0; s < kNumSymbols; ++s)
    if (counts[s]) {
      num_syms = s + 1;
      ++distinct;
    }
  if (distinct <= 1) return 2;
  u32 freq[kNumSymbols];
  normalize(counts, n, freq, num_syms);
  double bits = 0.0;
  for (int s = 0; s < num_syms; ++s)
    if (counts[s])
      bits += static_cast<double>(counts[s]) *
              (kScaleBits - std::log2(static_cast<double>(freq[s])));
  const int nway = n >= 8192 ? 8 : n >= 2048 ? 4 : 2;
  // mode + table header, nway state flushes, nway-1 length varints
  // (stream lengths are < 2^14 for the sizes that pick 2/4 ways, mostly
  // < 2^21 otherwise: 2-3 bytes each).
  thread_local std::vector<u8> hdr;
  hdr.clear();
  write_freq_table(hdr, freq, num_syms);
  const usize header = 1 + hdr.size() + 4 * static_cast<usize>(nway) +
                       static_cast<usize>(nway - 1) * (n >= 8192 ? 3 : 2);
  return header + static_cast<usize>(bits / 8.0 + 0.5);
}

void encode(std::vector<u8>& out, const u8* src, usize n, const u32* counts) {
  int num_syms = 1;
  int distinct = 0;
  for (int s = 0; s < kNumSymbols; ++s)
    if (counts[s]) {
      num_syms = s + 1;
      ++distinct;
    }

  // Constant array (single distinct symbol): store the symbol; the decoder
  // memsets. This also keeps every freq used by the dual-stream path <= kM - 1,
  // so the decode table can pack (sym, freq, cum) into one 32-bit word.
  if (distinct <= 1) {
    int sym = 0;
    for (int s = 0; s < kNumSymbols; ++s)
      if (counts[s]) {
        sym = s;
        break;
      }
    out.push_back(0);  // mode 0: constant
    out.push_back(static_cast<u8>(sym));
    return;
  }

  u32 freq[kNumSymbols];
  normalize(counts, n, freq, num_syms);
  u32 cum[kNumSymbols + 1] = {0};
  for (int s = 0; s < num_syms; ++s) cum[s + 1] = cum[s] + freq[s];

  // Independent streams over position classes (i mod nway) sharing one table,
  // so the decoder runs nway rANS states with no data dependency between them
  // (instruction-level parallelism). Eight states decode at ~1.0-1.15 GB/s
  // across 3-6.5 bits/symbol (four: ~0.7 GB/s), close to the Huffman decoder
  // on skewed data and ahead of it on dense data. Each extra state costs a
  // 4-byte flush and a length varint, so small arrays use fewer.
  //   payload mode 1: 2 states, 2: 4 states, 3: 8 states.
  // Header = mode + frequency table (write_freq_table).
  const int nway = n >= 8192 ? 8 : n >= 2048 ? 4 : 2;
  out.push_back(static_cast<u8>(nway == 8 ? 3 : nway == 4 ? 2 : 1));
  write_freq_table(out, freq, num_syms);

  // n >= 2 here and nway > 2 requires n >= 2048, so every stream is
  // non-empty. All but the last stream get a varint length prefix; the last
  // one runs to the end of the payload.
  EncSym syms[kNumSymbols];
  for (int s = 0; s < kNumSymbols; ++s) init_enc_sym(syms[s], freq[s], cum[s]);
  std::vector<u8> streams[kMaxWays];
  encode_streams(src, n, nway, syms, streams);
  {
    ByteWriter w(out);
    for (int k = 0; k + 1 < nway; ++k) w.varint(streams[k].size());
  }
  for (int k = 0; k < nway; ++k)
    out.insert(out.end(), streams[k].begin(), streams[k].end());
}

namespace {

thread_local std::vector<u32> g_ans_dsym;

bool build_ans_table(ByteReader& r, u32 dsym[kM]) {
  u32 freq[kNumSymbols];
  int num_syms;
  if (!read_freq_table(r, freq, num_syms)) return false;
  u32 cum[kNumSymbols + 1] = {0};
  u32 sum = 0;
  for (int s = 0; s < num_syms; ++s) {
    if (freq[s] >= kM) return false;  // a single-symbol table is mode 0
    cum[s + 1] = cum[s] + freq[s];
    sum += freq[s];
  }
  if (sum != kM) return false;

  for (int s = 0; s < num_syms; ++s) {
    if (freq[s] == 0) continue;
    const u32 packed = static_cast<u32>(s) | (freq[s] << 8) | (cum[s] << 20);
    for (u32 k = cum[s]; k < cum[s + 1]; ++k) dsym[k] = packed;
  }
  return true;
}

}  // namespace

Status StreamDecoder::init(const u8* data, usize len, usize total_out) {
  total_ = total_out;
  done_ = 0;
  ByteReader r(data, len);
  const u8 mode = r.byte();
  if (r.error) return Status::CorruptInput;

  if (mode == 0) {
    const_sym_ = r.byte();
    if (r.error) return Status::CorruptInput;
    constant_ = true;
    return Status::Ok;
  }
  constant_ = false;
  switch (mode) {
    case 1: nway_ = 2; break;
    case 2: nway_ = 4; break;
    case 3: nway_ = 8; break;
    default: return Status::CorruptInput;
  }

  g_ans_dsym.resize(kM);
  if (!build_ans_table(r, g_ans_dsym.data())) return Status::CorruptInput;

  // Stream lengths: varint per stream except the last, which runs to the end
  // of the payload.
  usize sizes[kMaxWays] = {0};
  usize prefix = 0;
  for (int k = 0; k + 1 < nway_; ++k) {
    sizes[k] = static_cast<usize>(r.varint());
    if (r.error || sizes[k] > r.remaining() - prefix) return Status::CorruptInput;
    prefix += sizes[k];
  }
  if (r.error || prefix > r.remaining()) return Status::CorruptInput;
  sizes[nway_ - 1] = r.remaining() - prefix;

  const u8* p = r.p;
  for (int k = 0; k < nway_; ++k) {
    if (sizes[k] < 4) return Status::CorruptInput;
    ptr_[k] = p;
    p += sizes[k];
    x_[k] = load_u32(ptr_[k]);
    ptr_[k] += 4;
  }
  // Renormalization loads are clamped to the last 16-bit word of the payload
  // (one shared bound: a stream over-running into its neighbour on corrupt
  // input just decodes bounded garbage, and stays inside the payload).
  plast_ = p - 2;
  return Status::Ok;
}

namespace {

// 16-bit renormalization, branchless: always load the next word (clamped to
// the last in-bounds word of the payload, so corrupt input decodes bounded
// garbage) and select whether to consume it. A branch here looks faster in a
// microbenchmark that repeats one array (the predictor memorizes the renorm
// pattern) but is ~15-25% slower on a stream of different arrays.
inline void rans_renorm(u32& x, const u8*& p, const u8* plast) {
  const u8* q = p < plast ? p : plast;
  const u32 v = load_u16(q);
  const bool need = x < kRansL16;
  x = need ? (x << 16) | v : x;
  p = need ? q + 2 : p;
}

// One order-0 rANS decode step.
inline u8 rans_step(const u32* dsym, u32& x, const u8*& p, const u8* plast) {
  constexpr u32 mask = kM - 1;
  const u32 e = dsym[x & mask];
  const u32 f = (e >> 8) & 0xFFFu;
  const u32 start = (e >> 20) & 0xFFFu;
  x = f * (x >> kScaleBits) + (x & mask) - start;
  rans_renorm(x, p, plast);
  return static_cast<u8>(e & 0xFF);
}

// Decode `count` symbols with N interleaved states held in registers.
// Position i is coded by state i mod N; `i` (the global position) must be a
// multiple of N on entry and count a multiple of N.
template <int N>
void rans_run(const u32* dsym, u32* xs, const u8** ps, const u8* plast,
              u8* dst, usize count) {
  // States and stream pointers live in locals (address never taken, so the
  // byte stores to `dst` cannot alias them and they stay in registers).
  // Symbols are stored one byte at a time on purpose: packing N symbols into
  // a word first costs ~30% (measured; the shifts/ors compete with the
  // decode chains for issue slots and force spills at N = 8).
  u32 x[N];
  const u8* p[N];
  for (int k = 0; k < N; ++k) {
    x[k] = xs[k];
    p[k] = ps[k];
  }
  for (usize w = 0; w < count; w += N) {
    for (int k = 0; k < N; ++k) dst[w + k] = rans_step(dsym, x[k], p[k], plast);
  }
  for (int k = 0; k < N; ++k) {
    xs[k] = x[k];
    ps[k] = p[k];
  }
}

}  // namespace

Status StreamDecoder::decode(u8* dst, usize count) {
  if (done_ + count > total_) return Status::CorruptInput;
  if (constant_) {
    std::memset(dst, const_sym_, count);
    done_ += count;
    return Status::Ok;
  }

  const u32* dsym = g_ans_dsym.data();
  auto step1 = [&](usize k) -> u8 {
    return rans_step(dsym, x_[k], ptr_[k], plast_);
  };

  // Align to a state-0 boundary, run whole rounds with all states live in
  // registers (independent chains = instruction-level parallelism), then
  // finish the tail one symbol at a time.
  usize w = 0;
  usize i = done_;
  const usize m = static_cast<usize>(nway_) - 1;
  while (w < count && (i & m) != 0) {
    dst[w++] = step1(i & m);
    ++i;
  }
  const usize rounds = (count - w) & ~m;
  if (rounds) {
    switch (nway_) {
      case 2: rans_run<2>(dsym, x_, ptr_, plast_, dst + w, rounds); break;
      case 4: rans_run<4>(dsym, x_, ptr_, plast_, dst + w, rounds); break;
      case 8: rans_run<8>(dsym, x_, ptr_, plast_, dst + w, rounds); break;
      default: return Status::CorruptInput;
    }
    w += rounds;
    i += rounds;
  }
  while (w < count) {
    dst[w++] = step1(i & m);
    ++i;
  }
  done_ = i;
  return Status::Ok;
}

Status decode(const u8* data, usize len, u8* dst, usize n) {
  StreamDecoder sd;
  if (Status s = sd.init(data, len, n); s != Status::Ok) return s;
  return sd.decode(dst, n);
}

// ---------------------------------------------------------------------------
// Order-1 context-split coder. Symbol i is coded with the static table of
// o1_class(src[i-1]) (class 0 for i == 0). Two interleaved rANS states cover
// even/odd positions so the state update and renorm of one symbol overlap
// the (serial) context -> table -> symbol chain of the next.
//
// Payload: mode byte (0 = constant + symbol byte; 1 = coded), then one bit
// stream holding one SmallCode (14 x 3 bits) shared by all classes and per
// class 9 bits num_syms (0: class unused) followed by its num_syms coded
// frequencies (put_freqs), zero-padded to a byte; then a varint with the
// even-position stream's length, then the two state streams.
// ---------------------------------------------------------------------------

namespace {

// Packed decode entry: sym | (freq-1) << 8 | cum << 20. freq-1 keeps a
// single-symbol class (freq == kM == 4096) inside 12 bits; cum <= kM - freq
// <= 4095 always fits.
thread_local std::vector<u32> g_o1_dsym;  // kO1Ctx * kM entries

inline void o1_hists(const u8* src, usize n, u32 (&h)[kO1Ctx][kNumSymbols]) {
  std::memset(h, 0, sizeof(h));
  u8 prev = 0;
  usize i = 0;
  for (; i + 4 <= n; i += 4) {
    const u8 a = src[i], b = src[i + 1], c = src[i + 2], d = src[i + 3];
    ++h[o1_class(prev)][a];
    ++h[o1_class(a)][b];
    ++h[o1_class(b)][c];
    ++h[o1_class(c)][d];
    prev = d;
  }
  for (; i < n; ++i) {
    ++h[o1_class(prev)][src[i]];
    prev = src[i];
  }
}

// Fill dsym[0..kM) from a list of (symbol, freq) with freqs summing to kM.
inline Status fill_dsym(u32* dsym, const u8* syms, const u16* freqs, int n) {
  u32 sum = 0;
  u32 c0 = 0;
  for (int i = 0; i < n; ++i) {
    const u16 f = freqs[i];
    if (!f) continue;
    const u32 packed = static_cast<u32>(syms[i]) |
                       ((static_cast<u32>(f) - 1) << 8) | (c0 << 20);
    for (u32 k = c0; k < c0 + f; ++k) dsym[k] = packed;
    c0 += f;
    sum += f;
  }
  if (sum != kM) return Status::CorruptInput;
  return Status::Ok;
}

}  // namespace

void encode_o1(std::vector<u8>& out, const u8* src, usize n) {
  u32 h[kO1Ctx][kNumSymbols];
  o1_hists(src, n, h);
  encode_o1(out, src, n, h);
}

void encode_o1(std::vector<u8>& out, const u8* src, usize n,
               const u32 h[kO1Ctx][kNumSymbols]) {
  // Whole-stream constant: same degenerate mode as the order-0 coder.
  {
    u32 tot[kNumSymbols] = {0};
    for (int c = 0; c < kO1Ctx; ++c)
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
  u32 freq[kO1Ctx][kNumSymbols];
  u32 cum[kO1Ctx][kNumSymbols + 1];
  int max_syms[kO1Ctx];
  u32 cf[kFreqClasses] = {};
  for (int c = 0; c < kO1Ctx; ++c) {
    u64 total = 0;
    int max_sym = 0;
    for (int s = 0; s < kNumSymbols; ++s)
      if (h[c][s]) {
        total += h[c][s];
        max_sym = s + 1;
      }
    max_syms[c] = max_sym;
    cum[c][0] = 0;
    if (max_sym == 0) {
      for (int s = 0; s < kNumSymbols; ++s) {
        freq[c][s] = 0;
        cum[c][s + 1] = 0;
      }
      continue;
    }
    normalize(h[c], total, freq[c], max_sym);
    for (int s = 0; s < max_sym; ++s) cum[c][s + 1] = cum[c][s] + freq[c][s];
    for (int s = max_sym; s < kNumSymbols; ++s) {
      freq[c][s] = 0;
      cum[c][s + 1] = cum[c][s];
    }
    class_hist(cf, freq[c], max_sym);
  }
  {
    huff::SmallCode sc;
    huff::small_code_build(cf, kFreqClasses, sc);
    BitWriter w(out);
    huff::small_code_write(w, sc, kFreqClasses);
    for (int c = 0; c < kO1Ctx; ++c) {
      w.put(static_cast<u32>(max_syms[c]), 9);
      if (max_syms[c]) put_freqs(w, sc, freq[c], max_syms[c]);
    }
    w.flush();
  }

  // Two independent state streams over even/odd positions, encoded in
  // reverse (rANS is LIFO). The context comes from src, so each stream can
  // be built independently exactly like the order-0 substreams.
  std::vector<u8> streams[2];
  for (usize phase = 0; phase < 2 && phase < n; ++phase) {
    // At most one 16-bit renorm per symbol + state flush.
    std::vector<u8> buf(2 * (n / 2 + 1) + 64);
    u8* bend = buf.data() + buf.size();
    u8* ptr = bend;
    u32 x = kRansL16;
    usize hi = n - 1 - ((n - 1 - phase) % 2);
    for (usize idx = hi;; idx -= 2) {
      const int c = o1_class(idx > 0 ? src[idx - 1] : u8{0});
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

Status decode_o1(const u8* data, usize len, u8* dst, usize n) {
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

  g_o1_dsym.resize(static_cast<usize>(kO1Ctx) * kM);
  {
    BitReader br(r.p, r.remaining());
    huff::SmallCode sc;
    if (!huff::small_code_read(br, kFreqClasses, sc)) return Status::CorruptInput;
    usize bits = static_cast<usize>(kFreqClasses) * 3;
    for (int c = 0; c < kO1Ctx; ++c) {
      u32* dsym = g_o1_dsym.data() + static_cast<usize>(c) * kM;
      const int num_syms = static_cast<int>(br.get(9));
      bits += 9;
      if (num_syms > kNumSymbols) return Status::CorruptInput;
      if (num_syms == 0) {
        // Class absent from the (well-formed) stream. Fill with a degenerate
        // entry so a corrupt stream that reaches it decodes bounded garbage
        // instead of reading a stale table from a previous block.
        std::fill_n(dsym, kM, (u32{kM - 1} << 8));
        continue;
      }
      u32 freq[kNumSymbols];
      if (!get_freqs(br, sc, freq, num_syms, bits)) return Status::CorruptInput;
      u8 syms[kNumSymbols];
      u16 freqs[kNumSymbols];
      int n_present = 0;
      for (int s = 0; s < num_syms; ++s) {
        if (!freq[s]) continue;
        if (freq[s] > kM) return Status::CorruptInput;
        syms[n_present] = static_cast<u8>(s);
        freqs[n_present] = static_cast<u16>(freq[s]);
        ++n_present;
      }
      if (n_present == 0) return Status::CorruptInput;
      if (fill_dsym(dsym, syms, freqs, n_present) != Status::Ok)
        return Status::CorruptInput;
    }
    if (!r.take((bits + 7) / 8)) return Status::CorruptInput;
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

  const u32* dsym = g_o1_dsym.data();
  constexpr u32 mask = kM - 1;
  auto step = [&](u32& x, const u8*& p, const u8* pend, int ctx) -> u8 {
    const u32 e = dsym[(static_cast<u32>(ctx) << kScaleBits) | (x & mask)];
    const u32 f = ((e >> 8) & 0xFFFu) + 1;
    const u32 start = (e >> 20) & 0xFFFu;
    x = f * (x >> kScaleBits) + (x & mask) - start;
    rans_renorm(x, p, pend);
    return static_cast<u8>(e & 0xFF);
  };

  u8 prev = 0;
  usize i = 0;
  for (; i + 2 <= n; i += 2) {
    prev = step(x0, p0, e0, o1_class(prev));
    dst[i] = prev;
    prev = step(x1, p1, e1, o1_class(prev));
    dst[i + 1] = prev;
  }
  if (i < n) dst[i] = step(x0, p0, e0, o1_class(prev));
  return Status::Ok;
}

}  // namespace spectra8::detail::ans
