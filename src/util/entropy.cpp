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

#include "util/entropy.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "util/arith.hpp"
#include "crystal/crystal.hpp"
#include "util/huffman.hpp"
#include "util/simd.hpp"
#include "util/tans.hpp"

namespace spectra8::detail {
namespace {

enum Tag : u8 {
  kMemset = 0,
  kRaw = 1,
  kHuffman = 2,
  kAns = 3,
  kRle = 4,
  kArith = 5,
  kArithAd = 6,    // adaptive order-0 range coder (no frequency header)
  kAnsO1 = 7,      // order-1 context-split static rANS (BWT symbol streams)
  kArithAdO1 = 8,  // order-1 context-split adaptive range coder
  kCrystal = 9,    // Crystal static lattice coder body (encode_array_crystal)
};

// Decode-speed bias for BWT symbol streams: the static interleaved ANS coder
// decodes several times faster than the serial adaptive range coder (~800 vs
// ~135 MB/s on an MTF-shaped stream with the 4-state payload), so it is
// chosen whenever its output is at most this factor larger. Low/mid levels
// lean toward decode speed; level 9+ pays (almost) nothing for ratio.
// (Dropping the range coder entirely below L9 was measured: it costs ~10% of
// ratio on shuffled columnar data, where the adaptive model's win is far
// larger than any reasonable bias - the trial stays at every level. A
// multi-lane adaptive coder was likewise tried as an alternative fast path:
// a shared model serializes the lanes, and per-lane models lose too much
// adaptation - static ANS is the better lever.)
inline double ans_bias(int level) {
  if (level >= 9) return 1.02;
  if (level >= 7) return 1.06;
  return 1.15;
}

void rle_encode(std::vector<u8>& out, const u8* src, usize n) {
  usize i = 0;
  while (i < n) {
    usize run = 1;
    while (i + run < n && src[i + run] == src[i] && run < 128) ++run;
    if (run >= 3) {
      out.push_back(static_cast<u8>(257 - run));  // 129..254
      out.push_back(src[i]);
      i += run;
    } else {
      usize start = i;
      while (i < n) {
        usize r = 1;
        while (i + r < n && src[i + r] == src[i] && r < 3) ++r;
        if (r >= 3) break;
        ++i;
        if (i - start == 128) break;
      }
      usize litlen = i - start;
      out.push_back(static_cast<u8>(litlen - 1));  // 0..127
      out.insert(out.end(), src + start, src + i);
    }
  }
}

Status rle_decode_stream(ByteReader& r, u8* dst, usize n, usize& lit_remain,
                         usize& run_remain, u8& run_val) {
  usize o = 0;
  while (o < n) {
    if (lit_remain > 0) {
      const usize take = min_of(lit_remain, n - o);
      const u8* p = r.take(take);
      if (!p) return Status::CorruptInput;
      std::memcpy(dst + o, p, take);
      lit_remain -= take;
      o += take;
      continue;
    }
    if (run_remain > 0) {
      const usize take = min_of(run_remain, n - o);
      std::memset(dst + o, run_val, take);
      run_remain -= take;
      o += take;
      continue;
    }
    u8 h = r.byte();
    if (r.error) return Status::CorruptInput;
    if (h < 128) {
      usize cnt = static_cast<usize>(h) + 1;
      const usize take = min_of(cnt, n - o);
      const u8* p = r.take(take);
      if (!p) return Status::CorruptInput;
      std::memcpy(dst + o, p, take);
      lit_remain = cnt - take;
      o += take;
    } else if (h > 128) {
      usize cnt = static_cast<usize>(257 - h);
      u8 v = r.byte();
      if (r.error) return Status::CorruptInput;
      const usize take = min_of(cnt, n - o);
      std::memset(dst + o, v, take);
      run_remain = cnt - take;
      run_val = v;
      o += take;
    }
    // h == 128: no-op
  }
  return Status::Ok;
}

Status rle_decode(const u8* data, usize len, u8* dst, usize n) {
  ByteReader r(data, len);
  usize lit_remain = 0;
  usize run_remain = 0;
  u8 run_val = 0;
  return rle_decode_stream(r, dst, n, lit_remain, run_remain, run_val);
}

Status decode_payload(const u8* data, usize len, u8* dst, usize n) {
  if (len == 0) return Status::CorruptInput;
  ByteReader r(data, len);
  u8 tag = r.byte();
  switch (tag) {
    case kMemset: {
      u8 v = r.byte();
      if (r.error) return Status::CorruptInput;
      std::memset(dst, v, n);
      return Status::Ok;
    }
    case kRaw: {
      const u8* s = r.take(n);
      if (!s) return Status::CorruptInput;
      std::memcpy(dst, s, n);
      return Status::Ok;
    }
    case kHuffman:
      return huff::decode(r.p, r.remaining(), dst, n);
    case kAns:
      return ans::decode(r.p, r.remaining(), dst, n);
    case kRle:
      return rle_decode(r.p, r.remaining(), dst, n);
    case kArith:
      return arith::decode(r.p, r.remaining(), dst, n);
    case kArithAd:
      return arith::decode_adaptive(r.p, r.remaining(), dst, n);
    case kAnsO1:
      return ans::decode_o1(r.p, r.remaining(), dst, n);
    case kArithAdO1:
      return arith::decode_adaptive_o1(r.p, r.remaining(), dst, n);
    case kCrystal:
      return decode_crystal_static(r.p, r.remaining(), dst, n);
    default:
      return Status::CorruptInput;
  }
}

}  // namespace

void encode_array(std::vector<u8>& out, const u8* src, usize n, int level,
                  ArrayKind kind) {
  std::vector<u8> best;

  // Single pass: byte histogram + adjacency count (how many bytes equal their
  // predecessor). This drives a cheap cost estimate so we only build the
  // entropy coders that can plausibly win, instead of building all of them and
  // discarding the losers.
  u32 freq[256];
  hist_u8(src, n, freq);
  // Adjacent-equal count in its own loop: the compiler vectorizes the
  // compare+sum, while fused with the histogram it was a scalar pass with a
  // store-to-load stall per repeated byte.
  usize adj_eq = 0;
  {
    u32 acc = 0;
    usize i = 1;
    for (; i < n; ++i) acc += static_cast<u32>(src[i] == src[i - 1]);
    adj_eq = acc;
  }
  int distinct = 0;
  for (int s = 0; s < 256; ++s)
    if (freq[s]) ++distinct;

  // Single distinct byte => memset is optimal (2 bytes); nothing can beat it.
  if (n > 0 && distinct == 1) {
    ByteWriter w(out);
    w.varint(n);
    w.varint(2);
    out.push_back(kMemset);
    out.push_back(src[0]);
    return;
  }

  // Order-0 entropy (bits) -> estimated Huffman size. The estimate is a lower
  // bound on the real Huffman size (entropy bound on the payload, and the
  // header estimate never exceeds the real header), so skipping Huffman when
  // its estimate already loses to raw can never discard a true winner.
  double bits = 0.0;
  if (n > 0) {
    const double dn = static_cast<double>(n);
    for (int s = 0; s < 256; ++s)
      if (freq[s]) {
        const double p = static_cast<double>(freq[s]) / dn;
        bits -= static_cast<double>(freq[s]) * std::log2(p);
      }
  }
  const double huff_est = bits / 8.0 + 3.0 + 0.5 * distinct;  // payload + header
  const double raw_size = static_cast<double>(n) + 1.0;
  const bool compressible = (n >= 32) && (huff_est < raw_size);

  // ANS, only when the array looks compressible (if Huffman cannot beat raw,
  // ANS - which targets the same entropy - cannot either).
  //
  // The 8-way rANS decodes at ~0.7x the Huffman decoder's symbol rate, so it
  // has to buy a real size win. Token streams (literal/match lengths, offset
  // codes) are where that happens: they are skewed towards a few symbols and
  // Huffman's one-bit-per-symbol floor costs 5-10% on them, while rANS codes
  // them near entropy - so from level 3 a token array switches when rANS
  // saves >= ~4% of it (sweep on the test corpus: this keeps the 1-2.5%
  // whole-file wins on text/json/samba/columnar data and skips the sub-0.5%
  // ones that cost 10-15% decode speed). Literals are dense and large; they
  // switch only at level 9+ and for a >= ~1.5% win.
  //
  // The choice is made from the two coders' size estimates (Huffman's from
  // its code lengths, rANS's from its normalized table - both within a few
  // bytes of the real payload), so only the winner is built.
  const bool try_ans = compressible && n >= 256 &&
                       (kind == ArrayKind::Tokens ? level >= 3 : level >= 9);
  bool use_ans = false;
  if (try_ans) {
    const usize hs = huff::encoded_size(freq, n);
    const usize as = ans::encoded_size(freq, n);
    const usize margin = (kind == ArrayKind::Tokens) ? (as * 10) / 256 : as >> 6;
    use_ans = as + margin < hs;
  }

  if (compressible) {
    std::vector<u8> h;
    h.push_back(use_ans ? kAns : kHuffman);
    if (use_ans)
      ans::encode(h, src, n, freq);
    else
      huff::encode(h, src, n, freq);
    if (best.empty() || h.size() < best.size()) best.swap(h);
  }

  // RLE: only when the data is "runny" enough to plausibly win (cheap proxy:
  // at least ~12.5% of bytes repeat their predecessor). Token streams (lengths,
  // offset codes) are the usual winners here; literal byte arrays rarely are.
  // RLE output is at least n - adj_eq bytes (every byte outside a run is
  // copied, runs cost two bytes), so skip the pass when that floor already
  // loses to the current best.
  if (n > 0 && adj_eq * 8 >= n &&
      (best.empty() || n - adj_eq + 1 < best.size())) {
    std::vector<u8> rl;
    rl.push_back(kRle);
    rle_encode(rl, src, n);
    if (best.empty() || rl.size() < best.size()) best.swap(rl);
  }

  // Raw fallback (always valid).
  if (best.empty() || (n + 1) < best.size()) {
    best.clear();
    best.push_back(kRaw);
    best.insert(best.end(), src, src + n);
  }

  ByteWriter w(out);
  w.varint(n);
  w.varint(best.size());
  w.bytes(best);
}

bool encode_array_crystal(std::vector<u8>& out, const u8* src, usize n,
                          int level, int stride, usize budget) {
  if (n < 64) return false;
  thread_local std::vector<u8> body;
  body.clear();
  body.push_back(kCrystal);
  build_crystal_static(src, n, level, stride, body);
  // Total blob = varint(n) varint(len) body; compare against the budget with
  // the header counted (it is what the caller's rival blob also carries).
  auto varint_size = [](u64 v) {
    usize k = 1;
    while (v >= 128) {
      v >>= 7;
      ++k;
    }
    return k;
  };
  const usize total = varint_size(n) + varint_size(body.size()) + body.size();
  if (budget && total >= budget) return false;
  ByteWriter w(out);
  w.varint(n);
  w.varint(body.size());
  w.bytes(body);
  return true;
}

void encode_bwt_entropy(std::vector<u8>& out, const u8* src, usize n, int level) {
  std::vector<u8> best;

  u32 freq[256];
  hist_u8(src, n, freq);

  int distinct = 0;
  for (int s = 0; s < 256; ++s)
    if (freq[s]) ++distinct;

  if (n > 0 && distinct == 1) {
    ByteWriter w(out);
    w.varint(n);
    w.varint(2);
    out.push_back(kMemset);
    out.push_back(src[0]);
    return;
  }

  double bits = 0.0;
  if (n > 0) {
    const double dn = static_cast<double>(n);
    for (int s = 0; s < 256; ++s)
      if (freq[s]) {
        const double p = static_cast<double>(freq[s]) / dn;
        bits -= static_cast<double>(freq[s]) * std::log2(p);
      }
  }
  const double huff_est = bits / 8.0 + 3.0 + 0.5 * distinct;
  const double raw_size = static_cast<double>(n) + 1.0;
  const bool compressible = (n >= 32) && (huff_est < raw_size);

  // Order-1 hist once for AnsO1 / adaptive-O1 gating (L5+). Building it before
  // the slow adaptive-O1 encode lets us skip that pass when the static O1
  // estimate cannot beat the current best - L9 encode was spending most of
  // its time racing adaptive O0 + O1 + ANS on every segment.
  const bool try_ans_o1 = level >= 5;
  const bool try_arith_o1 = level >= 7;
  double o1_est = 1e30;
  u32 h1[ans::kO1Ctx][256] = {};
  bool have_h1 = false;
  if ((try_ans_o1 || try_arith_o1) && compressible && n >= 512) {
    have_h1 = true;
    u8 prev = 0;
    usize i = 0;
    for (; i + 4 <= n; i += 4) {
      const u8 a = src[i], b = src[i + 1], c = src[i + 2], d = src[i + 3];
      ++h1[ans::o1_class(prev)][a];
      ++h1[ans::o1_class(a)][b];
      ++h1[ans::o1_class(b)][c];
      ++h1[ans::o1_class(c)][d];
      prev = d;
    }
    for (; i < n; ++i) {
      ++h1[ans::o1_class(prev)][src[i]];
      prev = src[i];
    }
    o1_est = 2.0;  // mode byte + slack
    for (int c = 0; c < ans::kO1Ctx; ++c) {
      u64 cn = 0;
      int cd = 0;
      int max_sym = 0;
      for (int s = 0; s < 256; ++s) {
        cn += h1[c][s];
        if (h1[c][s]) {
          ++cd;
          max_sym = s + 1;
        }
      }
      if (cn == 0) {
        o1_est += 9.0 / 8.0;
        continue;
      }
      // Coded table (tans.cpp put_freqs): ~2 bits per absent symbol, a
      // ~3-bit class code plus the mantissa for a present one.
      const double mant = std::max(0.0, std::log2(4096.0 / cd) - 1.0);
      o1_est += (9.0 + 2.0 * (max_sym - cd) + cd * (3.0 + mant)) / 8.0;
      const double dcn = static_cast<double>(cn);
      for (int s = 0; s < 256; ++s)
        if (h1[c][s]) {
          const double p = static_cast<double>(h1[c][s]) / dcn;
          o1_est -= static_cast<double>(h1[c][s]) * std::log2(p) / 8.0;
        }
    }
  }

  // Arithmetic coding: tight on skewed MTF / RUNA/RUNB streams (BWT blocks).
  // The adaptive (header-less) multisymbol range coder is the sole order-0
  // candidate. On the heavily-skewed post-filter streams this path sees, its
  // cumulative scan averages ~1 step per byte and it issues just one (on Apple
  // Silicon, cheap) division, so it both compresses tightly and decodes fast
  // (~90 MB/s vs ~66 for an LZMA-style binary bit-tree, which pays a fixed
  // 8 bits/byte -- measured worse here). The multisymbol *static* coder is not
  // used: it decodes with a per-symbol division plus binary search, and its
  // 16-bit frequency rescale can round a rare-but-present symbol to 0 on large
  // blocks (the encoder's size==0 guard then drops it, corrupting the stream),
  // so it is both slower and unsafe.
  //
  // Note: an LZMA-style division-free binary bit-tree coder was evaluated as a
  // decode-speed win, but measured ~27% *slower* to decode on these skewed
  // streams (fixed 8 bits/byte vs the multisymbol coder's ~1-step scan + one
  // cheap division), so the adaptive multisymbol coder is kept.
  const bool ans_o1_ok = try_ans_o1 && compressible && n >= 1024;
  auto do_adaptive = [&] {
    if (!(compressible && n >= 64)) return;
    std::vector<u8> ad;
    ad.push_back(kArithAd);
    arith::encode_adaptive(ad, src, n);
    if (best.empty() || ad.size() < best.size()) best.swap(ad);
  };
  auto do_ans_o1 = [&] {
    if (!ans_o1_ok) return;
    const double best_sz =
        best.empty() ? raw_size : static_cast<double>(best.size());
    // Against order-0 adaptive, only a 1% concession (the table-driven decode
    // is several times faster than the adaptive coder).
    const double margin = (!best.empty() && best[0] == kArithAd) ? 1.01 : 1.0;
    if (!best.empty() && o1_est >= best_sz * (margin + 0.03)) return;
    std::vector<u8> o1;
    o1.push_back(kAnsO1);
    if (have_h1)
      ans::encode_o1(o1, src, n, h1);
    else
      ans::encode_o1(o1, src, n);
    if (best.empty() ||
        static_cast<double>(o1.size()) <
            static_cast<double>(best.size()) * margin)
      best.swap(o1);
  };

  // Invert the usual adaptive-then-AnsO1 race when the O1 estimate sits well
  // under O0 Shannon (the adaptive model can beat Shannon, but not by ~16%+
  // on these streams). A cheap adaptive bit-count then decides whether the
  // range encode can be skipped: same 1% AnsO1 concession as the compare
  // below, so the chosen payload stays the same.
  const double shannon = bits / 8.0;
  const bool o1_first = ans_o1_ok && o1_est < shannon * 0.84;
  if (o1_first) {
    const double ad_pred = arith::estimate_adaptive(src, n) + 1.0;
    if (o1_est < ad_pred * 1.03) {
      do_ans_o1();
      if (best.empty() || best[0] != kAnsO1 ||
          static_cast<double>(best.size()) > ad_pred * 1.01)
        do_adaptive();
    } else {
      do_adaptive();
    }
  } else {
    do_adaptive();
    do_ans_o1();
  }

  // Order-1 adaptive range coder (L7+): rare winner once AnsO1 has run, so
  // require the static O1 estimate to look strictly smaller than the current
  // best. Measured on dickens L9: adaptive O1 won ~1% of segments but was the
  // dominant encode-time cost when always tried.
  if (try_arith_o1 && compressible && n >= 512) {
    const double best_sz =
        best.empty() ? raw_size : static_cast<double>(best.size());
    if (o1_est < best_sz * 0.99) {
      std::vector<u8> ad1;
      ad1.push_back(kArithAdO1);
      arith::encode_adaptive_o1(ad1, src, n);
      if (best.empty() || ad1.size() < best.size()) best.swap(ad1);
    }
  }

  // ANS candidate with a decode-speed bias (see ans_bias). Skip the real
  // encode when the O0 entropy estimate cannot beat the current best under
  // that bias (same spirit as the Huffman lower-bound prune below).
  if (compressible && n >= 256) {
    const double best_sz =
        best.empty() ? raw_size : static_cast<double>(best.size());
    const double bias = best.empty() ? 1.0
                        : (best[0] == kArithAd || best[0] == kArithAdO1)
                            ? ans_bias(level)
                        : best[0] == kAnsO1 ? 1.005
                                            : 1.0;
    if (huff_est < best_sz * (bias + 0.03)) {
      std::vector<u8> a;
      a.push_back(kAns);
      ans::encode(a, src, n);
      if (best.empty() ||
          static_cast<double>(a.size()) < static_cast<double>(best.size()) * bias)
        best.swap(a);
    }
  }

  // Huffman: cheap order-0 fallback; only meaningfully competes when the stream
  // is too small/degenerate for the arith path, but kept as a safety net. Two
  // provable lower bounds on its output - the entropy payload (bits/8) and one
  // bit per symbol (n/8), plus the tag byte - let us skip the encode entirely
  // when the arith candidate is already at or below them; the kept output is
  // bit-identical, this only prunes losing work.
  if (compressible) {
    const double huff_lb = max_of(bits, static_cast<double>(n)) / 8.0 + 1.0;
    if (best.empty() || huff_lb < static_cast<double>(best.size())) {
      std::vector<u8> h;
      h.push_back(kHuffman);
      huff::encode(h, src, n, freq);
      if (best.empty() || h.size() < best.size()) best.swap(h);
    }
  }

  if (best.empty() || (n + 1) < best.size()) {
    best.clear();
    best.push_back(kRaw);
    best.insert(best.end(), src, src + n);
  }

  if (std::getenv("SPECTRA8_DEBUG_ENTROPY"))
    std::fprintf(stderr, "bwt-entropy n=%zu tag=%d size=%zu\n", n,
                 best.empty() ? -1 : best[0], best.size());

  ByteWriter w(out);
  w.varint(n);
  w.varint(best.size());
  w.bytes(best);
}

Status decode_array(ByteReader& r, std::vector<u8>& out) {
  u64 n = r.varint();
  u64 plen = r.varint();
  if (r.error) return Status::CorruptInput;
  const u8* pd = r.take(static_cast<usize>(plen));
  if (!pd) return Status::CorruptInput;

  usize base = out.size();
  out.resize(base + static_cast<usize>(n));
  return decode_payload(pd, static_cast<usize>(plen),
                        out.data() + base, static_cast<usize>(n));
}

Status ArrayStreamDecoder::begin(ByteReader& r) {
  total_ = static_cast<usize>(r.varint());
  const u64 plen = r.varint();
  if (r.error) return Status::CorruptInput;
  const u8* pd = r.take(static_cast<usize>(plen));
  if (!pd || plen == 0) return Status::CorruptInput;

  done_ = 0;
  kind_ = Kind::None;
  rle_lit_remain_ = 0;
  rle_run_remain_ = 0;
  rle_run_val_ = 0;
  const u8 tag = pd[0];
  const u8* body = pd + 1;
  const usize blen = static_cast<usize>(plen) - 1;

  switch (tag) {
    case kMemset:
      if (blen < 1) return Status::CorruptInput;
      memset_v_ = body[0];
      kind_ = Kind::Memset;
      return Status::Ok;
    case kRaw:
      raw_br_ = ByteReader(body, blen);
      kind_ = Kind::Raw;
      return Status::Ok;
    case kHuffman: {
      kind_ = Kind::Huff;
      return huff_.init(body, blen, total_);
    }
    case kAns:
      kind_ = Kind::Ans;
      return ans_.init(body, blen, total_);
    case kRle:
      rle_br_ = ByteReader(body, blen);
      kind_ = Kind::Rle;
      return Status::Ok;
    default:
      return Status::CorruptInput;
  }
}

Status ArrayStreamDecoder::decode(u8* dst, usize count) {
  if (done_ + count > total_) return Status::CorruptInput;
  switch (kind_) {
    case Kind::Memset:
      std::memset(dst, memset_v_, count);
      break;
    case Kind::Raw: {
      const u8* p = raw_br_.take(count);
      if (!p) return Status::CorruptInput;
      std::memcpy(dst, p, count);
      break;
    }
    case Kind::Huff:
      if (Status s = huff_.decode(dst, count); s != Status::Ok) return s;
      break;
    case Kind::Ans:
      if (Status s = ans_.decode(dst, count); s != Status::Ok) return s;
      break;
    case Kind::Rle:
      if (Status s = rle_decode_stream(rle_br_, dst, count, rle_lit_remain_,
                                      rle_run_remain_, rle_run_val_);
          s != Status::Ok)
        return s;
      break;
    default:
      return Status::CorruptInput;
  }
  done_ += count;
  return Status::Ok;
}

}  // namespace spectra8::detail
