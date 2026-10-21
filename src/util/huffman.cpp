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

#include "util/huffman.hpp"

#include <algorithm>
#include <array>
#include <cstring>

#include "util/bits.hpp"

namespace spectra8::detail::huff {
namespace {

constexpr int kNumSymbols = 256;

// Bitstream layout selector, written after the code-length header.
//   kStreamSingle: one bitstream covering all symbols.
//   kStreamQuad:   four independent bitstreams over four equal output ranges,
//                  decoded interleaved so the CPU runs four dependency chains
//                  in parallel (instruction-level parallelism). A 3-varint
//                  size prefix locates the first three streams; the fourth is
//                  the remainder.
enum StreamMode : u8 { kStreamSingle = 0, kStreamQuad = 1 };

// Below this many symbols the 4-stream jump-table + padding overhead is not
// worth the parallelism, so a single stream is used. Also the lower bound that
// makes the quad output-range split (q = ceil(n/4)) well-formed and in-bounds.
// Four-stream Huffman once the array is large enough for ILP to pay for the
// extra stream overhead (was 4096; 1024 helps literal buffers on big blocks).
constexpr usize kQuadMinSymbols = 1024;

// Compute code lengths (limited to max_len) for the given frequencies.
// `lengths` is filled for symbols [0, num_syms); unused symbols get length 0.
void build_lengths(const u32 freq[kNumSymbols], int num_syms,
                   u8 lengths[kNumSymbols], int max_len_limit = kMaxCodeLen) {
  std::fill(lengths, lengths + kNumSymbols, u8{0});

  // Collect used symbols.
  const int limit = num_syms > kNumSymbols ? kNumSymbols : num_syms;
  int used[kNumSymbols];
  int n = 0;
  for (int s = 0; s < limit; ++s)
    if (freq[s] != 0) used[n++] = s;

  if (n <= 0) return;
  if (n == 1) {
    lengths[used[0]] = 1;
    return;
  }

  // Assign shortest lengths to the most frequent symbols (also the leaf
  // order for the tree build below).
  std::sort(used, used + n, [&](int a, int b) {
    if (freq[a] != freq[b]) return freq[a] > freq[b];
    return a < b;
  });

  // Optimal (unlimited) Huffman to obtain the multiset of code lengths:
  // two-queue construction over the sorted leaves (ascending) and the
  // internal nodes, which come out in non-decreasing weight order, so no
  // heap is needed. Node depth = parent depth + 1, computed top-down since
  // every parent index is larger than its children.
  u64 weight[2 * kNumSymbols];
  int parent[2 * kNumSymbols] = {};
  u8 depth[2 * kNumSymbols] = {};
  for (int i = 0; i < n; ++i) weight[i] = freq[used[n - 1 - i]];  // ascending
  int leaf = 0, node = n, next = n;
  auto take = [&]() -> int {
    // Prefer the leaf on ties: keeps the tree shallower on skewed inputs.
    if (leaf < n && (node >= next || weight[leaf] <= weight[node])) return leaf++;
    return node++;
  };
  for (int step = 0; step < n - 1; ++step) {
    const int a = take();
    const int b = take();
    weight[next] = weight[a] + weight[b];
    parent[a] = next;
    parent[b] = next;
    ++next;
  }
  const int root = 2 * n - 2;
  depth[root] = 0;
  for (int i = root - 1; i >= 0; --i)
    depth[i] = static_cast<u8>(depth[parent[i]] + 1);

  constexpr int kMaxBits = 255;
  std::array<int, kMaxBits + 1> bl_count{};
  int max_len = 0;
  for (int i = 0; i < n; ++i) {
    int d = depth[i];
    if (d == 0) d = 1;  // single-node safety
    ++bl_count[d];
    max_len = max_of(max_len, d);
  }

  // Limit lengths to max_len_limit using the JPEG (Annex K) procedure on
  // the per-length counts. Total count and Kraft equality are preserved.
  for (int i = max_len; i > max_len_limit; --i) {
    while (bl_count[i] > 0) {
      int j = i - 2;
      while (j > 0 && bl_count[j] == 0) --j;
      if (j == 0) break;  // should not happen for a valid tree
      bl_count[i] -= 2;
      bl_count[i - 1] += 1;
      bl_count[j + 1] += 2;
      bl_count[j] -= 1;
    }
  }
  int idx = 0;
  for (int l = 1; l <= max_len_limit; ++l) {
    for (int c = 0; c < bl_count[l] && idx < n; ++c) {
      lengths[used[idx++]] = static_cast<u8>(l);
    }
  }
}

// Code-length header. Byte 0: num_syms - 1; byte 1: layout.
//   kHdrNibbles: ceil(num_syms / 2) bytes of 4-bit lengths.
//   kHdrCoded:   the lengths as a run-length symbol stream under a SmallCode
//                (deflate's scheme): 0..kMaxCodeLen a length, kClRepeat the
//                previous length 3-6 more times (2 bits), kClZeros3 3-10
//                zeros (3 bits), kClZeros11 11-138 zeros (7 bits). The 15
//                code lengths of the SmallCode lead (3 bits each), the
//                stream is zero-padded to a byte.
// A 256-symbol literal table costs 60-90 bytes coded against 130 packed;
// the encoder writes whichever is smaller (small token alphabets stay
// packed).
enum HdrLayout : u8 { kHdrNibbles = 0, kHdrCoded = 1 };
constexpr int kClRepeat = kMaxCodeLen + 1;
constexpr int kClZeros3 = kMaxCodeLen + 2;
constexpr int kClZeros11 = kMaxCodeLen + 3;
constexpr int kClSyms = kMaxCodeLen + 4;
static_assert(kClSyms <= kSmallMaxSyms);

// Run-length symbols of a length array: (symbol, extra bits value).
struct ClStream {
  u8 sym[kNumSymbols];
  u8 extra[kNumSymbols];
  int n = 0;
};

void cl_tokenize(const u8* lengths, int num_syms, ClStream& cs) {
  cs.n = 0;
  int i = 0;
  while (i < num_syms) {
    const int l = lengths[i];
    int run = 1;
    while (i + run < num_syms && lengths[i + run] == l) ++run;
    if (l == 0 && run >= 3) {
      const int r = run > 138 ? 138 : run;
      if (r >= 11) {
        cs.sym[cs.n] = kClZeros11;
        cs.extra[cs.n++] = static_cast<u8>(r - 11);
      } else {
        cs.sym[cs.n] = kClZeros3;
        cs.extra[cs.n++] = static_cast<u8>(r - 3);
      }
      i += r;
      continue;
    }
    cs.sym[cs.n] = static_cast<u8>(l);
    cs.extra[cs.n++] = 0;
    ++i;
    --run;
    while (run >= 3) {
      const int r = run > 6 ? 6 : run;
      cs.sym[cs.n] = kClRepeat;
      cs.extra[cs.n++] = static_cast<u8>(r - 3);
      i += r;
      run -= r;
    }
  }
}

constexpr int cl_extra_bits(int sym) {
  return sym == kClRepeat ? 2 : sym == kClZeros3 ? 3 : sym == kClZeros11 ? 7 : 0;
}

// Appends the header for `lengths` to `out`.
void write_table_header(std::vector<u8>& out, const u8* lengths, int num_syms) {
  out.push_back(static_cast<u8>(num_syms - 1));
  const usize nibble_bytes = static_cast<usize>((num_syms + 1) / 2);
  ClStream cs;
  SmallCode sc;
  usize coded_bits = static_cast<usize>(kClSyms) * 3;
  // The coded form starts with 6 bytes of SmallCode lengths: an alphabet
  // packing into 7 bytes is never worth tokenizing.
  bool coded = nibble_bytes > 7;
  if (coded) {
    cl_tokenize(lengths, num_syms, cs);
    u32 cf[kSmallMaxSyms] = {};
    for (int i = 0; i < cs.n; ++i) ++cf[cs.sym[i]];
    coded = small_code_build(cf, kClSyms, sc);
  }
  if (coded) {
    for (int i = 0; i < cs.n; ++i)
      coded_bits += sc.len[cs.sym[i]] + static_cast<usize>(cl_extra_bits(cs.sym[i]));
    coded = (coded_bits + 7) / 8 < nibble_bytes;
  }
  if (!coded) {
    out.push_back(kHdrNibbles);
    for (int s = 0; s < num_syms; s += 2) {
      const u8 lo = lengths[s];
      const u8 hi = (s + 1 < num_syms) ? lengths[s + 1] : 0;
      out.push_back(static_cast<u8>(lo | (hi << 4)));
    }
    return;
  }
  out.push_back(kHdrCoded);
  BitWriter w(out);
  small_code_write(w, sc, kClSyms);
  for (int i = 0; i < cs.n; ++i) {
    small_code_put(w, sc, cs.sym[i]);
    const int eb = cl_extra_bits(cs.sym[i]);
    if (eb) w.put(cs.extra[i], eb);
  }
  w.flush();
}

// Parses the header at r; fills lengths[0..num_syms) and num_syms. False on
// a malformed header.
bool read_table_header(ByteReader& r, u8 lengths[kNumSymbols], int& num_syms) {
  num_syms = r.byte() + 1;
  const u8 layout = r.byte();
  if (r.error) return false;
  std::fill(lengths, lengths + kNumSymbols, u8{0});
  if (layout == kHdrNibbles) {
    const int packed = (num_syms + 1) / 2;
    const u8* lp = r.take(static_cast<usize>(packed));
    if (!lp) return false;
    for (int s = 0; s < num_syms; ++s) {
      const u8 byte = lp[s >> 1];
      const u8 l = (s & 1) ? (byte >> 4) : (byte & 0x0F);
      if (l > kMaxCodeLen) return false;
      lengths[s] = l;
    }
    return true;
  }
  if (layout != kHdrCoded) return false;
  BitReader br(r.p, r.remaining());
  usize bits = 0;
  SmallCode sc;
  if (!small_code_read(br, kClSyms, sc)) return false;
  bits += static_cast<usize>(kClSyms) * 3;
  int i = 0;
  int prev = 0;
  while (i < num_syms) {
    const int sym = small_code_get(br, sc);
    if (sym < 0) return false;
    bits += sc.len[sym];
    const int eb = cl_extra_bits(sym);
    const int extra = eb ? static_cast<int>(br.get(eb)) : 0;
    bits += static_cast<usize>(eb);
    int run = 1, val = sym;
    if (sym == kClRepeat) {
      run = 3 + extra;
      val = prev;
    } else if (sym == kClZeros3) {
      run = 3 + extra;
      val = 0;
    } else if (sym == kClZeros11) {
      run = 11 + extra;
      val = 0;
    }
    if (i + run > num_syms) return false;
    for (int k = 0; k < run; ++k) lengths[i++] = static_cast<u8>(val);
    prev = val;
  }
  return r.take((bits + 7) / 8) != nullptr;
}

// Generate canonical codes (MSB-first) from lengths.
void canonical_codes(const u8 lengths[kNumSymbols], int num_syms,
                     u32 codes[kNumSymbols]) {
  std::array<int, kMaxCodeLen + 1> bl_count{};
  for (int s = 0; s < num_syms; ++s)
    if (lengths[s]) ++bl_count[lengths[s]];

  std::array<u32, kMaxCodeLen + 1> next_code{};
  u32 code = 0;
  for (int bits = 1; bits <= kMaxCodeLen; ++bits) {
    code = (code + static_cast<u32>(bl_count[bits - 1])) << 1;
    next_code[bits] = code;
  }
  for (int s = 0; s < num_syms; ++s) {
    int l = lengths[s];
    codes[s] = l ? next_code[l]++ : 0;
  }
}

}  // namespace

bool small_code_build(const u32* freq, int nsym, SmallCode& sc) {
  u32 f[kNumSymbols] = {};
  for (int s = 0; s < nsym; ++s) f[s] = freq[s];
  u8 lengths[kNumSymbols];
  build_lengths(f, nsym, lengths, kSmallMaxLen);
  u32 codes[kNumSymbols];
  canonical_codes(lengths, nsym, codes);
  for (int s = 0; s < kSmallMaxSyms; ++s) {
    sc.len[s] = s < nsym ? lengths[s] : 0;
    sc.code[s] = s < nsym ? static_cast<u8>(codes[s]) : 0;
    if (sc.len[s] > kSmallMaxLen) return false;
  }
  return true;
}

void small_code_write(BitWriter& w, const SmallCode& sc, int nsym) {
  for (int s = 0; s < nsym; ++s) w.put(sc.len[s], 3);
}

bool small_code_read(BitReader& r, int nsym, SmallCode& sc) {
  u8 lengths[kNumSymbols] = {};
  for (int s = 0; s < nsym; ++s) lengths[s] = static_cast<u8>(r.get(3));
  u32 codes[kNumSymbols];
  canonical_codes(lengths, nsym, codes);
  std::fill(sc.dec, sc.dec + (1 << kSmallMaxLen), u16{0});
  u32 covered = 0;
  for (int s = 0; s < nsym; ++s) {
    const int l = lengths[s];
    sc.len[s] = static_cast<u8>(l);
    sc.code[s] = static_cast<u8>(codes[s]);
    if (!l) continue;
    const u32 base = codes[s] << (kSmallMaxLen - l);
    const u32 count = 1u << (kSmallMaxLen - l);
    if (base + count > (1u << kSmallMaxLen)) return false;
    covered += count;
    for (u32 k = 0; k < count; ++k)
      sc.dec[base + k] = static_cast<u16>((s << 8) | l);
  }
  // An incomplete code (one symbol: length 1 covering half the table) is
  // fine - unmatched bits decode as -1; over-subscription is caught above.
  return covered <= (1u << kSmallMaxLen);
}

void encode(std::vector<u8>& out, const u8* src, usize n) {
  u32 freq[kNumSymbols] = {0};
  for (usize i = 0; i < n; ++i) ++freq[src[i]];
  encode(out, src, n, freq);
}

usize encoded_size(const u32 freq[kNumSymbols], usize n) {
  int num_syms = 1;
  for (int s = 0; s < kNumSymbols; ++s)
    if (freq[s]) num_syms = s + 1;
  u8 lengths[kNumSymbols];
  build_lengths(freq, num_syms, lengths);
  u64 bits = 0;
  for (int s = 0; s < num_syms; ++s)
    bits += static_cast<u64>(freq[s]) * lengths[s];
  // Table header + mode byte; quad mode adds three stream length varints
  // and pads each of the four streams to a byte.
  thread_local std::vector<u8> hdr;
  hdr.clear();
  write_table_header(hdr, lengths, num_syms);
  usize size = hdr.size() + 1;
  if (n >= kQuadMinSymbols)
    size += 3 * 3 + static_cast<usize>((bits + 4 * 7) / 8);
  else
    size += static_cast<usize>((bits + 7) / 8);
  return size;
}

void encode(std::vector<u8>& out, const u8* src, usize n, const u32* freq) {
  int num_syms = 1;
  for (int s = 0; s < kNumSymbols; ++s)
    if (freq[s]) num_syms = s + 1;

  u8 lengths[kNumSymbols];
  build_lengths(freq, num_syms, lengths);

  u32 codes[kNumSymbols];
  canonical_codes(lengths, num_syms, codes);

  // One table lookup per symbol: code in the low half, length in the high.
  u32 ce[kNumSymbols];
  for (int s = 0; s < kNumSymbols; ++s)
    ce[s] = (s < num_syms) ? (codes[s] | (static_cast<u32>(lengths[s]) << 16)) : 0;

  write_table_header(out, lengths, num_syms);

  // Output bound per stream: kMaxCodeLen bits per symbol plus packer slack.
  auto bound = [](usize cnt) { return (cnt * kMaxCodeLen + 7) / 8 + 16; };

  if (n >= kQuadMinSymbols) {
    // Four independent streams over four contiguous output ranges. All streams
    // share the one code-length header above (same canonical code). They are
    // packed in one interleaved loop so the four accumulator chains overlap
    // (the serial shift-or chain was the encoder's limit), two symbols per
    // branchless flush (<= 22 + 7 pending bits).
    out.push_back(kStreamQuad);
    const usize q = (n + 3) / 4;
    const usize sizes[4] = {q, q, q, n - 3 * q};
    const usize per = bound(q);
    thread_local std::vector<u8> scratch;
    if (scratch.size() < 4 * per) scratch.resize(4 * per);
    const u8* in[4];
    MsbPacker pk[4] = {MsbPacker(scratch.data()), MsbPacker(scratch.data() + per),
                       MsbPacker(scratch.data() + 2 * per), MsbPacker(scratch.data() + 3 * per)};
    in[0] = src;
    for (int t = 1; t < 4; ++t) in[t] = in[t - 1] + sizes[t - 1];
    const usize common = sizes[3] & ~usize{1};  // whole pairs in every stream
    usize i = 0;
    for (; i < common; i += 2) {
      for (int t = 0; t < 4; ++t) {
        const u32 a = ce[in[t][i]], b = ce[in[t][i + 1]];
        pk[t].put(a & 0xFFFFu, a >> 16);
        pk[t].put(b & 0xFFFFu, b >> 16);
        pk[t].flush();
      }
    }
    for (int t = 0; t < 4; ++t) {
      for (usize k = i; k < sizes[t]; ++k) {
        const u32 a = ce[in[t][k]];
        pk[t].put(a & 0xFFFFu, a >> 16);
        pk[t].flush();
      }
    }
    usize len[4];
    for (int t = 0; t < 4; ++t) len[t] = static_cast<usize>(pk[t].finish() - (scratch.data() + t * per));
    ByteWriter w(out);
    w.varint(len[0]);
    w.varint(len[1]);
    w.varint(len[2]);
    usize pos = out.size();
    out.resize(pos + len[0] + len[1] + len[2] + len[3]);
    for (int t = 0; t < 4; ++t) {
      std::memcpy(out.data() + pos, scratch.data() + t * per, len[t]);
      pos += len[t];
    }
  } else {
    out.push_back(kStreamSingle);
    const usize pos = out.size();
    out.resize(pos + bound(n));
    MsbPacker pk(out.data() + pos);
    usize i = 0;
    for (; i + 4 <= n; i += 4) {  // 4 * 11 + 7 <= 57 pending bits
      const u32 a = ce[src[i]], b = ce[src[i + 1]], c = ce[src[i + 2]], d = ce[src[i + 3]];
      pk.put(a & 0xFFFFu, a >> 16);
      pk.put(b & 0xFFFFu, b >> 16);
      pk.put(c & 0xFFFFu, c >> 16);
      pk.put(d & 0xFFFFu, d >> 16);
      pk.flush();
    }
    for (; i < n; ++i) {
      const u32 a = ce[src[i]];
      pk.put(a & 0xFFFFu, a >> 16);
      pk.flush();
    }
    out.resize(static_cast<usize>(pk.finish() - out.data()));
  }
}

namespace {

constexpr int kTableBits = kMaxCodeLen;
constexpr usize kTableSize = usize{1} << kTableBits;

// Pair table index widths. A pair entry exists when two consecutive codes fit
// in the index, so the useful width depends on the alphabet: skewed data
// (tokens, text literals: 3-5 bit codes) pairs nearly every slot at 11 bits;
// dense data (6-8 bit codes: images, executables, floats) pairs almost none
// at 11 but most at 13. The wider table costs 4x the build (32 KiB of
// writes), so it is only used on arrays large enough to amortize that.
constexpr int kPairBitsNarrow = kTableBits;
constexpr int kPairBitsWide = 13;
constexpr usize kPairTableSize = usize{1} << kPairBitsWide;

// Minimum output size for which a pair table is built. Below, the build cost
// dominates and the single-symbol path wins.
constexpr usize kPairMinSymbols = 8192;
constexpr usize kPairWideMinSymbols = 16384;

// Minimum fraction of pair-table slots (in 1/256 units) that must resolve two
// symbols for that table width to be worth it; otherwise the pair loop is the
// single loop with a bigger table.
constexpr int kPairMinCoverage256 = 64;

thread_local std::array<u16, kTableSize> g_huff_table;

// Pair table: one lookup resolves up to TWO consecutive symbols when both
// canonical codes fit within the index width, halving the serial
// peek->load->consume dependency chain per symbol (same idea as zstd's HUF
// "X2" double-symbol table). Decoder-only; the bitstream format is unchanged.
// Entry layout (u32):
//   bits  7..0   first symbol
//   bits 15..8   second symbol (junk when nsyms == 1; written speculatively
//                by the decoder and overwritten by the next step)
//   bits 23..16  total code bits consumed (l0, or l0 + l1)
//   bits 31..24  symbols decoded by this entry (1 or 2)
thread_local std::array<u32, kPairTableSize> g_huff_pair_table;
// Index width the pair table was built for (0: not built for this code).
thread_local int g_pair_bits = 0;

// Fill g_huff_pair_table (index width `pb`) from validated lengths/codes
// (Kraft-complete, all ranges in bounds -- checked by build_decode_table).
// Left-aligned canonical codes are contiguous in (length, symbol) order, so
// for each first symbol the two-symbol subranges are emitted back-to-back
// from the range base and the remainder (second codes longer than the spare
// bits) is filled with the single-symbol fallback: every index is written
// exactly once and the build is O(table size).
void build_pair_table(const u8 lengths[kNumSymbols],
                      const u32 codes[kNumSymbols], int num_syms, int pb) {
  u8 syms_by_len[kMaxCodeLen + 1][kNumSymbols];
  int cnt_by_len[kMaxCodeLen + 1] = {0};
  for (int s = 0; s < num_syms; ++s) {
    const int l = lengths[s];
    if (l) syms_by_len[l][cnt_by_len[l]++] = static_cast<u8>(s);
  }

  u32* tbl = g_huff_pair_table.data();
  for (int l0 = 1; l0 <= kMaxCodeLen; ++l0) {
    for (int i = 0; i < cnt_by_len[l0]; ++i) {
      const u32 s0 = syms_by_len[l0][i];
      const u32 base = codes[s0] << (pb - l0);
      const u32 end = base + (1u << (pb - l0));
      const int rem = pb - l0;
      u32 cur = base;
      for (int l1 = 1; l1 <= rem && l1 <= kMaxCodeLen; ++l1) {
        const u32 sub = 1u << (rem - l1);
        const u32 e01 = s0 | (static_cast<u32>(l0 + l1) << 16) | (2u << 24);
        for (int j = 0; j < cnt_by_len[l1]; ++j) {
          const u32 e = e01 | (static_cast<u32>(syms_by_len[l1][j]) << 8);
          for (u32 k = 0; k < sub; ++k) tbl[cur++] = e;
        }
      }
      const u32 single = s0 | (static_cast<u32>(l0) << 16) | (1u << 24);
      while (cur < end) tbl[cur++] = single;
    }
  }
}

// Fraction (in 1/256) of a pb-bit pair table's slots that resolve two symbols.
int pair_coverage256(const int cnt[kMaxCodeLen + 1], int pb) {
  u64 paired = 0;
  for (int l0 = 1; l0 <= kMaxCodeLen; ++l0)
    for (int l1 = 1; l1 <= kMaxCodeLen && l0 + l1 <= pb; ++l1)
      paired += (static_cast<u64>(cnt[l0]) * cnt[l1]) << (pb - l0 - l1);
  return static_cast<int>((paired * 256) >> pb);
}

// Parse the code-length header, build the single-symbol table (and the pair
// table when worthwhile for `n` symbols), leave `r_after_hdr` at the stream
// mode byte. Returns false on a malformed header.
bool build_decode_table(const u8* data, usize len, ByteReader& r_after_hdr,
                        usize n) {
  ByteReader r(data, len);
  u8 lengths[kNumSymbols];
  int num_syms;
  if (!read_table_header(r, lengths, num_syms)) return false;

  u32 codes[kNumSymbols];
  canonical_codes(lengths, num_syms, codes);

  u16* tbl = g_huff_table.data();
  u32 covered = 0;
  for (int s = 0; s < num_syms; ++s) {
    const int l = lengths[s];
    if (!l) continue;
    const u32 base = codes[s] << (kTableBits - l);
    const u32 count = 1u << (kTableBits - l);
    if (base + count > kTableSize) return false;
    const u16 entry = static_cast<u16>(s | (l << 8));
    u16* dst = tbl + base;
    u32 k = 0;
    if (count >= 4) {
      const u64 pat = static_cast<u64>(entry) | (static_cast<u64>(entry) << 16) |
                      (static_cast<u64>(entry) << 32) |
                      (static_cast<u64>(entry) << 48);
      for (; k + 4 <= count; k += 4) std::memcpy(dst + k, &pat, 8);
    }
    for (; k < count; ++k) dst[k] = entry;
    covered += count;
  }
  if (covered != kTableSize) return false;

  g_pair_bits = 0;
  if (n >= kPairMinSymbols) {
    int cnt[kMaxCodeLen + 1] = {0};
    for (int s = 0; s < num_syms; ++s) ++cnt[lengths[s]];
    cnt[0] = 0;
    int pb = 0;
    if (pair_coverage256(cnt, kPairBitsNarrow) >= kPairMinCoverage256)
      pb = kPairBitsNarrow;
    else if (n >= kPairWideMinSymbols &&
             pair_coverage256(cnt, kPairBitsWide) >= kPairMinCoverage256)
      pb = kPairBitsWide;
    if (pb) {
      build_pair_table(lengths, codes, num_syms, pb);
      g_pair_bits = pb;
    }
  }
  r_after_hdr = r;
  return true;
}

// After a refill at most 7 bits are already consumed, so 57 bits are valid:
// a batch may consume up to 57 bits before the next refill.
constexpr int kBudgetBits = 57;

// One single-symbol step (requires kTableBits valid bits).
inline u8 step1(const u16* tbl, MsbReader& r) {
  const u16 e = tbl[r.peek(kTableBits)];
  r.consume(e >> 8);
  return static_cast<u8>(e);
}

// One pair-table step: write two bytes speculatively, advance by the actual
// symbol count. Caller guarantees >= 2 writable bytes at `p` and PB valid bits.
template <int PB>
inline u8* step2(const u32* tbl2, MsbReader& r, u8* p) {
  const u32 e = tbl2[r.peek(PB)];
  p[0] = static_cast<u8>(e);
  p[1] = static_cast<u8>(e >> 8);
  r.consume((e >> 16) & 0xFF);
  return p + (e >> 24);
}

// Decode `count` symbols from one stream. PB = 0: single-symbol table;
// otherwise the pair table of that width. Fast refills while 8 bytes remain,
// then a zero-padded tail.
template <int PB>
void run(MsbReader& r, u8* dst, usize count) {
  const u16* tbl = g_huff_table.data();
  u8* p = dst;
  u8* const end = dst + count;
  if constexpr (PB != 0) {
    constexpr int kBatch = kBudgetBits / PB;
    const u32* tbl2 = g_huff_pair_table.data();
    while (end - p >= 2 * kBatch && r.has8()) {
      r.refill_fast();
      for (int j = 0; j < kBatch; ++j) p = step2<PB>(tbl2, r, p);
    }
  } else {
    constexpr int kBatch = kBudgetBits / kTableBits;
    while (end - p >= kBatch && r.has8()) {
      r.refill_fast();
      for (int j = 0; j < kBatch; ++j) *p++ = step1(tbl, r);
    }
  }
  // Tail: single symbols, safe refill before each (short) batch.
  constexpr int kTailBatch = kBudgetBits / kTableBits;
  while (p < end) {
    r.refill_safe();
    for (int j = 0; j < kTailBatch && p < end; ++j) *p++ = step1(tbl, r);
  }
}

// Four streams decoded interleaved (four independent dependency chains), each
// stepped by one batch per refill. Streams advance at their own
// (data-dependent) pace; the loop stops as soon as any stream is within one
// batch of its end or its stream data, and `run` finishes each independently.
template <int PB>
void run_quad(MsbReader* r, u8** p, u8* const* e) {
  constexpr int kBatch = kBudgetBits / (PB ? PB : kTableBits);
  constexpr usize kNeed = PB ? 2 * kBatch : kBatch;  // output bytes per batch
  const u16* tbl = g_huff_table.data();
  const u32* tbl2 = g_huff_pair_table.data();
  MsbReader r0 = r[0], r1 = r[1], r2 = r[2], r3 = r[3];
  u8 *p0 = p[0], *p1 = p[1], *p2 = p[2], *p3 = p[3];
  while (static_cast<usize>(e[0] - p0) >= kNeed &&
         static_cast<usize>(e[1] - p1) >= kNeed &&
         static_cast<usize>(e[2] - p2) >= kNeed &&
         static_cast<usize>(e[3] - p3) >= kNeed && r0.has8() && r1.has8() &&
         r2.has8() && r3.has8()) {
    r0.refill_fast();
    r1.refill_fast();
    r2.refill_fast();
    r3.refill_fast();
    for (int j = 0; j < kBatch; ++j) {
      if constexpr (PB != 0) {
        p0 = step2<PB>(tbl2, r0, p0);
        p1 = step2<PB>(tbl2, r1, p1);
        p2 = step2<PB>(tbl2, r2, p2);
        p3 = step2<PB>(tbl2, r3, p3);
      } else {
        *p0++ = step1(tbl, r0);
        *p1++ = step1(tbl, r1);
        *p2++ = step1(tbl, r2);
        *p3++ = step1(tbl, r3);
      }
    }
  }
  r[0] = r0;
  r[1] = r1;
  r[2] = r2;
  r[3] = r3;
  p[0] = p0;
  p[1] = p1;
  p[2] = p2;
  p[3] = p3;
  for (int k = 0; k < 4; ++k)
    run<PB>(r[k], p[k], static_cast<usize>(e[k] - p[k]));
}

void run_any(MsbReader& r, u8* dst, usize count) {
  switch (g_pair_bits) {
    case kPairBitsNarrow: run<kPairBitsNarrow>(r, dst, count); break;
    case kPairBitsWide: run<kPairBitsWide>(r, dst, count); break;
    default: run<0>(r, dst, count); break;
  }
}

// Parse the quad stream header after the mode byte: four readers over the
// four sub-streams. Returns false if the size prefix is inconsistent.
bool init_quad(ByteReader& r, MsbReader out[4]) {
  const u64 b0 = r.varint(), b1 = r.varint(), b2 = r.varint();
  if (r.error) return false;
  const usize avail = r.remaining();
  if (b0 > avail || b1 > avail - b0 || b2 > avail - b0 - b1) return false;
  const u8* s0 = r.p;
  const u8* s1 = s0 + b0;
  const u8* s2 = s1 + b1;
  const u8* s3 = s2 + b2;
  const usize b3 = avail - static_cast<usize>(b0) - static_cast<usize>(b1) -
                   static_cast<usize>(b2);
  out[0] = MsbReader(s0, static_cast<usize>(b0));
  out[1] = MsbReader(s1, static_cast<usize>(b1));
  out[2] = MsbReader(s2, static_cast<usize>(b2));
  out[3] = MsbReader(s3, b3);
  return true;
}

}  // namespace

Status StreamDecoder::init(const u8* data, usize len, usize total_out) {
  total_ = total_out;
  done_ = 0;
  quad_si_ = 0;
  quad_sd_ = 0;
  ByteReader r{nullptr, 0};
  if (!build_decode_table(data, len, r, total_out)) return Status::CorruptInput;
  pair_bits_ = g_pair_bits;

  const u8 mode = r.byte();
  if (r.error) return Status::CorruptInput;

  if (mode == kStreamSingle) {
    single_ = true;
    rd_[0] = MsbReader(r.p, r.remaining());
    return Status::Ok;
  }
  if (mode != kStreamQuad || total_out < kQuadMinSymbols)
    return Status::CorruptInput;
  single_ = false;
  MsbReader q[4];
  if (!init_quad(r, q)) return Status::CorruptInput;
  for (int k = 0; k < 4; ++k) rd_[k] = q[k];

  const usize qn = (total_out + 3) / 4;
  quad_lim_[0] = qn;
  quad_lim_[1] = qn;
  quad_lim_[2] = qn;
  quad_lim_[3] = total_out - 3 * qn;
  return Status::Ok;
}

Status StreamDecoder::decode(u8* dst, usize count) {
  if (done_ + count > total_) return Status::CorruptInput;
  // The tables are thread-local and shared with other decodes on this thread;
  // an incremental decoder that resumes after another array was decoded
  // would see the wrong table. Callers decode one array at a time (see
  // entropy::StreamDecoder), so a mismatch here is a programming error, but
  // guard the pair width anyway.
  if (pair_bits_ != g_pair_bits) return Status::CorruptInput;
  if (single_) {
    run_any(rd_[0], dst, count);
  } else {
    usize w = 0;
    while (w < count) {
      if (quad_si_ >= 4) return Status::CorruptInput;
      const usize left = quad_lim_[quad_si_] - quad_sd_;
      const usize chunk = min_of(count - w, left);
      run_any(rd_[quad_si_], dst + w, chunk);
      quad_sd_ += chunk;
      w += chunk;
      if (quad_sd_ == quad_lim_[quad_si_]) {
        ++quad_si_;
        quad_sd_ = 0;
      }
    }
  }
  done_ += count;
  return Status::Ok;
}

Status decode(const u8* data, usize len, u8* dst, usize n) {
  ByteReader r{nullptr, 0};
  if (!build_decode_table(data, len, r, n)) return Status::CorruptInput;

  const u8 mode = r.byte();
  if (r.error) return Status::CorruptInput;

  if (mode == kStreamSingle) {
    MsbReader br(r.p, r.remaining());
    run_any(br, dst, n);
    return Status::Ok;
  }
  if (mode != kStreamQuad || n < kQuadMinSymbols) return Status::CorruptInput;

  MsbReader rd[4];
  if (!init_quad(r, rd)) return Status::CorruptInput;
  const usize q = (n + 3) / 4;
  u8* p[4] = {dst, dst + q, dst + 2 * q, dst + 3 * q};
  u8* const e[4] = {dst + q, dst + 2 * q, dst + 3 * q, dst + n};
  switch (g_pair_bits) {
    case kPairBitsNarrow: run_quad<kPairBitsNarrow>(rd, p, e); break;
    case kPairBitsWide: run_quad<kPairBitsWide>(rd, p, e); break;
    default: run_quad<0>(rd, p, e); break;
  }
  return Status::Ok;
}

}  // namespace spectra8::detail::huff
