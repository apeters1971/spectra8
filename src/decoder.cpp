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

#include <atomic>
#include <cstring>
#include <span>
#include <vector>

#include "util/bits.hpp"
#include "util/byte_rle.hpp"
#include "prism/bwt.hpp"
#include "prism/bwt_cm.hpp"
#include "prism/bwt_ultra.hpp"
#include "prism/bwt_mtf_rle.hpp"
#include "util/checksum.hpp"
#include "codec.hpp"
#include "util/pack_filter.hpp"
#include "util/entropy.hpp"
#include "ext/zstd.hpp"
#include "util/format.hpp"
#include "prism/lzp.hpp"
#include "oracle/oracle.hpp"
#include "crystal/crystal.hpp"
#include "mosaic/mosaic.hpp"
#include "flash/flash.hpp"
#include "laser/laser.hpp"
#include "grating/grating.hpp"
#include "beam/lz.hpp"
#include "util/parallel.hpp"
#include "util/simd.hpp"
#include "util/transform.hpp"

namespace spectra8::detail {
namespace {

// Fast overlapping match copy for offset 1 (the most common short offset).
// Offsets 2..7 use the byte loop (m[k] == d[k-off]); off >= 8 is handled inline.
// Exact-length variant for the block tail (no overshoot allowed).
void copy_match_overlap(u8* d, usize off, usize len) noexcept {
  const u8* m = d - off;
  if (off != 1) {
    for (usize k = 0; k < len; ++k) d[k] = m[k];
    return;
  }
  d[0] = m[0];
  usize k = 1;
  for (; k < 8 && k < len; ++k) d[k] = d[k - 1];
  for (; k + 8 <= len; k += 8) copy8(d + k, d + k - 8);
  for (; k < len; ++k) d[k] = d[k - 1];
}

// LZ4-style overlap tables: after copying the first 4 bytes, advance the
// source by inc32[off], copy 4 more, then rewind by dec64[off]. The result is
// that (dst - src) becomes a multiple of the pattern period that is >= 8, so
// the remainder can proceed in 8-byte steps.
constexpr int kInc32[8] = {0, 1, 2, 1, 0, 4, 4, 4};
constexpr int kDec64[8] = {0, 0, 0, -1, -4, 1, 2, 3};

// Overlapping match copy for off < 8 that may overwrite up to 8 bytes past
// d + len. Callers must guarantee slack. Kept out of line so the rare short-
// offset path does not bloat the sequence loop's code.
__attribute__((noinline)) void copy_match_overlap_wild(u8* d, usize off,
                                                       usize len) noexcept {
  const u8* m = d - off;
  d[0] = m[0];
  d[1] = m[1];
  d[2] = m[2];
  d[3] = m[3];
  m += kInc32[off];
  std::memcpy(d + 4, m, 4);
  m -= kDec64[off];
  // Here (d + 8) - m is a multiple of the period `off` and >= 8, so the tail
  // can copy in 8-byte steps reading only already-written bytes.
  u8* dend = d + len;
  d += 8;
  while (d < dend) {
    copy8(d, m);
    d += 8;
    m += 8;
  }
}

}  // namespace

Status decode_compressed_block(const u8* data, usize len, u8* dst, usize n) {
  ByteReader r(data, len);
  u8 lit_mode = r.byte();
  u64 num_seq = r.varint();
  u64 last_lit = r.varint();
  if (r.error) return Status::CorruptInput;
  const bool sub = (lit_mode & kBlockModeLiteralMask) != 0;
  const bool ml_cond = (lit_mode & kBlockFlagMlCond) != 0;
  const bool oc_cond = (lit_mode & kBlockFlagOcCond) != 0;
  const int off_low = (lit_mode & kBlockOffLowMask) >> kBlockOffLowShift;

  // Reused across every block decoded on this thread: decode_array appends from
  // out.size(), so clearing (which keeps capacity) makes these allocation-free
  // after the first block instead of four heap allocations per block.
  thread_local std::vector<u8> lit_len_stream, match_len_stream, off_code_stream,
      off_low_stream, literals;
  lit_len_stream.clear();
  match_len_stream.clear();
  off_code_stream.clear();
  off_low_stream.clear();
  literals.clear();
  if (num_seq > 0) {
    const usize seq_cap = static_cast<usize>(num_seq);
    lit_len_stream.reserve(seq_cap);
    match_len_stream.reserve(seq_cap);
    off_code_stream.reserve(seq_cap);
  }
  // Literal copies below read up to kLitSlack bytes past the current literal
  // position (16-byte wild copies for short runs), so the buffer gets that
  // much padding after the decoded literals.
  constexpr usize kLitSlack = 32;
  literals.reserve(n + kLitSlack);
  if (Status s = decode_array(r, lit_len_stream); s != Status::Ok) return s;
  if (lit_len_stream.size() != num_seq) return Status::CorruptInput;

  // Conditional token streams are stored as one array per context, in bucket
  // order. Tokens were pushed in sequence order, so reconstituting is a
  // per-sequence pop from the bucket selected by the earlier tokens.
  auto take_ctx = [&](std::vector<u8>& dst, int nctx,
                      auto&& bucket_of) -> Status {
    thread_local std::vector<u8> ctx[kOcContexts];
    thread_local usize cur[kOcContexts];
    for (int b = 0; b < nctx; ++b) {
      ctx[b].clear();
      if (Status s = decode_array(r, ctx[b]); s != Status::Ok) return s;
      cur[b] = 0;
    }
    dst.resize(static_cast<usize>(num_seq));
    for (u64 i = 0; i < num_seq; ++i) {
      const int b = bucket_of(static_cast<usize>(i));
      if (b < 0 || b >= nctx || cur[b] >= ctx[b].size())
        return Status::CorruptInput;
      dst[static_cast<usize>(i)] = ctx[b][cur[b]++];
    }
    for (int b = 0; b < nctx; ++b)
      if (cur[b] != ctx[b].size()) return Status::CorruptInput;
    return Status::Ok;
  };

  if (ml_cond) {
    if (Status s = take_ctx(match_len_stream, kLlBuckets,
                            [&](usize i) { return ll_bucket(lit_len_stream[i]); });
        s != Status::Ok)
      return s;
  } else if (Status s = decode_array(r, match_len_stream); s != Status::Ok) {
    return s;
  }
  if (oc_cond) {
    if (Status s = take_ctx(off_code_stream, kOcContexts,
                            [&](usize i) {
                              return oc_context(match_len_stream[i],
                                                lit_len_stream[i]);
                            });
        s != Status::Ok)
      return s;
  } else if (Status s = decode_array(r, off_code_stream); s != Status::Ok) {
    return s;
  }
  if (r.error) return Status::CorruptInput;
  if (match_len_stream.size() != num_seq || off_code_stream.size() != num_seq)
    return Status::CorruptInput;
  if (off_low > 0) {
    if (Status s = decode_array(r, off_low_stream); s != Status::Ok) return s;
    // One token per new offset with k >= off_low; a stray token is a corrupt
    // block, and the loop below checks the cursor before every read. Pad so
    // the read itself never runs off the array.
    off_low_stream.push_back(0);
  }
  const u8* const low_tok = off_low_stream.data();
  const usize low_total = off_low ? off_low_stream.size() - 1 : 0;
  usize low_i = 0;

  u64 ll_extra_len = r.varint();
  const u8* ll_extra = r.take(static_cast<usize>(ll_extra_len));
  u64 ml_extra_len = r.varint();
  const u8* ml_extra = r.take(static_cast<usize>(ml_extra_len));
  u64 off_extra_len = r.varint();
  const u8* off_extra = r.take(static_cast<usize>(off_extra_len));
  if (r.error) return Status::CorruptInput;

  if (Status s = decode_array(r, literals); s != Status::Ok) return s;
  if (r.error) return Status::CorruptInput;
  const usize lit_total = literals.size();
  literals.resize(lit_total + kLitSlack);

  ByteReader ll(ll_extra, static_cast<usize>(ll_extra_len));
  ByteReader ml(ml_extra, static_cast<usize>(ml_extra_len));
  MsbReader ob(off_extra, static_cast<usize>(off_extra_len));

  RecentOffsets reps;
  reps.init();
  usize out_pos = 0;
  usize li = 0;
  const u8* lit_data = literals.data();
  // Raw pointers for the hot loop (vector indexing adds a load per access).
  const u8* const ll_tok = lit_len_stream.data();
  const u8* const ml_tok = match_len_stream.data();
  const u8* const oc_tok = off_code_stream.data();

  // Emit `count` literals at out_pos. Bounds are checked by the caller.
  //
  // Raw literals: one 16-byte wild copy first (the average run is well under
  // 16 bytes on most data), then the rarely-taken remainder; the block tail
  // uses memcpy. The literal buffer carries kLitSlack bytes of padding so the
  // source side never overruns either.
  //
  // Sub literals depend on the byte `off` back. When off >= 16 no byte within
  // a 16-byte chunk depends on another byte of the same chunk, so the chunk
  // can be computed as a straight vector add (the compiler vectorizes the
  // fixed-size inner loop); the scalar loop remains for near offsets, the
  // block start (out_pos < off) and the block tail.
  auto emit_literals = [&](u32 count) {
    if (!sub) {
      if (out_pos + count + 16 <= n) {
        u8* d = dst + out_pos;
        const u8* l = lit_data + li;
        copy16(d, l);
        if (__builtin_expect(count > 16, 0)) {
          u8* const dend = d + count;
          d += 16;
          l += 16;
          do {
            copy16(d, l);
            d += 16;
            l += 16;
          } while (d < dend);
        }
      } else {
        std::memcpy(dst + out_pos, lit_data + li, count);
      }
      li += count;
      out_pos += count;
      return;
    }
    const i32 off = reps.r[0];
    if (off >= 16 && out_pos >= static_cast<usize>(off) &&
        out_pos + count + 16 <= n) {
      u8* d = dst + out_pos;
      const u8* s = d - off;
      const u8* l = lit_data + li;
      for (u32 k = 0; k < count; k += 16) {
        u8 a[16], b[16];
        std::memcpy(a, s + k, 16);
        std::memcpy(b, l + k, 16);
        for (int j = 0; j < 16; ++j) a[j] = static_cast<u8>(a[j] + b[j]);
        std::memcpy(d + k, a, 16);
      }
      li += count;
      out_pos += count;
      return;
    }
    for (u32 k = 0; k < count; ++k) {
      const u8 v = lit_data[li++];
      dst[out_pos] = (out_pos >= static_cast<usize>(off))
                         ? static_cast<u8>(v + dst[out_pos - off])
                         : v;
      ++out_pos;
    }
  };

  // Largest offset code: kNumRecentOffsets repeat slots, then one code per
  // raw-bit count 0..31.
  constexpr u32 kMaxOffCode = kNumRecentOffsets + 31;

  for (u64 i = 0; i < num_seq; ++i) {
    u32 L = ll_tok[i];
    if (__builtin_expect(L == 255, 0)) {
      L += static_cast<u32>(ll.varint());
      if (ll.error) return Status::CorruptInput;
    }
    // Emit before the offset-cache update: sub literals are coded against
    // the previous match's offset (r[0]). Skipping L==0 also skips the
    // raw-literal path's 16-byte wild copy of upcoming literals into the
    // match dest — that write was immediately overwritten and showed up as
    // extra traffic on short-match files.
    if (L) {
      if (li + L > lit_total || out_pos + L > n) return Status::CorruptInput;
      if (!sub) {
        if (out_pos + L + 16 <= n) {
          u8* d = dst + out_pos;
          const u8* l = lit_data + li;
          copy16(d, l);
          if (__builtin_expect(L > 16, 0)) {
            u8* const dend = d + L;
            d += 16;
            l += 16;
            do {
              copy16(d, l);
              d += 16;
              l += 16;
            } while (d < dend);
          }
        } else {
          std::memcpy(dst + out_pos, lit_data + li, L);
        }
        li += L;
        out_pos += L;
      } else {
        emit_literals(L);
      }
    }

    u32 M = ml_tok[i];
    if (__builtin_expect(M == 255, 0)) {
      M += static_cast<u32>(ml.varint());
      if (ml.error) return Status::CorruptInput;
    }
    M += kMinRepMatch;

    // Offset: refill only when the accumulator is short. New offsets are
    // typically 8-17 bits, so one 64-bit load covers several sequences.
    const u32 c = oc_tok[i];
    if (c > kMaxOffCode) return Status::CorruptInput;
    i32 off;
    if (c >= kNumRecentOffsets) {
      const int k = static_cast<int>(c - kNumRecentOffsets);
      if (k >= off_low && off_low > 0) {
        if (low_i >= low_total) return Status::CorruptInput;
        const int kk = k - off_low;
        ob.ensure(static_cast<u32>(kk));
        off = static_cast<i32>((1u << k) | (ob.peek0(kk) << off_low) |
                               low_tok[low_i++]);
        ob.consume(static_cast<u32>(kk));
      } else {
        ob.ensure(static_cast<u32>(k));
        off = static_cast<i32>((1u << k) | ob.peek0(k));
        ob.consume(static_cast<u32>(k));
      }
      reps.use_new(off);
    } else {
      off = reps.r[c];
      reps.use_rep(static_cast<int>(c));
    }
    if (off <= 0 || static_cast<usize>(off) > out_pos || out_pos + M > n)
      return Status::CorruptInput;

    u8* d = dst + out_pos;
    const u8* m = d - static_cast<usize>(off);
    // Short far matches (128 KiB shuffle planes: M is 4-7): an 8-byte copy
    // avoids the 16-byte wild copy's extra stores, which the next sequence
    // immediately overwrote.
    if (off >= 8 && M <= 8 && out_pos + 8 <= n) {
      copy8(d, m);
    } else if (off >= 16 && out_pos + M + kCopySlack <= n) {
      copy16(d, m);
      if (__builtin_expect(M > 16, 0)) {
        u8* const dend = d + M;
        d += 16;
        m += 16;
        do {
          copy16(d, m);
          d += 16;
          m += 16;
        } while (d < dend);
      }
    } else if (off >= 8) {
      usize k2 = 0;
      for (; k2 + 8 <= M; k2 += 8) store_u64(d + k2, load_u64(m + k2));
      for (; k2 < M; ++k2) d[k2] = m[k2];
    } else if (out_pos + M + kCopySlack <= n) {
      copy_match_overlap_wild(d, static_cast<usize>(off),
                              static_cast<usize>(M));
    } else {
      copy_match_overlap(d, static_cast<usize>(off), static_cast<usize>(M));
    }
    out_pos += M;
  }

  if (last_lit > n || li + last_lit > lit_total || out_pos + last_lit > n)
    return Status::CorruptInput;
  if (last_lit) emit_literals(static_cast<u32>(last_lit));

  if (out_pos != n || li != lit_total) return Status::CorruptInput;
  return Status::Ok;
}

// Decode a self-framed BWT symbol block, appending to `out`.
//   0: one entropy array (decode_array)
//   1: varint(num_segs) + that many entropy arrays
//   2: Spectra8 LZ block (varint n, varint len, compressed payload)
//   3: zstd L5 frame (varint n, varint len, zstd payload)
Status decode_bwt_symbols(ByteReader& r, std::vector<u8>& out) {
  u8 seg_mode = r.byte();
  if (r.error) return Status::CorruptInput;
  if (seg_mode == 0) return decode_array(r, out);
  if (seg_mode == 2 || seg_mode == 3) {
    const u64 n = r.varint();
    const u64 plen = r.varint();
    const u8* p = r.take(static_cast<usize>(plen));
    if (r.error || !p) return Status::CorruptInput;
    const usize base = out.size();
    out.resize(base + static_cast<usize>(n));
    if (seg_mode == 2)
      return decode_compressed_block(p, static_cast<usize>(plen),
                                     out.data() + base, static_cast<usize>(n));
    if (!zstd_available()) return Status::UnsupportedVersion;
    const auto zr = zstd_decompress(
        std::span<const std::byte>(reinterpret_cast<const std::byte*>(p),
                                   static_cast<usize>(plen)),
        std::span<std::byte>(reinterpret_cast<std::byte*>(out.data() + base),
                             static_cast<usize>(n)));
    if (!zr.ok() || zr.value != static_cast<usize>(n))
      return Status::CorruptInput;
    return Status::Ok;
  }
  if (seg_mode != 1) return Status::CorruptInput;
  u64 num = r.varint();
  if (r.error || num == 0) return Status::CorruptInput;
  for (u64 s = 0; s < num; ++s)
    if (Status st = decode_array(r, out); st != Status::Ok) return st;
  return Status::Ok;
}

// Decode a BWT block payload into dst[0, n): reverse the last step, then
// (unless mode 3) inverse MTF, then inverse BWT.
Status decode_bwt_block(const u8* data, usize len, u8* dst, usize n) {
  ByteReader r(data, len);
  u64 primary = r.varint();
  u8 num_chains = r.byte();
  if (r.error || num_chains == 0 || num_chains > kBwtMaxChains)
    return Status::CorruptInput;
  usize anchors[kBwtMaxChains] = {0};
  if (num_chains > 1)
    for (int c = 0; c < num_chains; ++c) anchors[c] = static_cast<usize>(r.varint());
  u8 raw_mode = r.byte();
  bool has_lzp = (raw_mode & 0x80) != 0;
  u8 mode = raw_mode & 0x7F;
  usize bwt_n = n;
  if (has_lzp) {
    bwt_n = static_cast<usize>(r.varint());
    if (r.error || bwt_n == 0 || bwt_n >= n) return Status::CorruptInput;
  }
  if (r.error) return Status::CorruptInput;

  thread_local std::vector<u8> mtf;
  mtf.clear();
  mtf.reserve(bwt_n);
  bool mtf_done = false;  // mode 2 and 4 produce post-MTF or raw bytes directly

  if (mode == 0) {
    if (Status s = decode_bwt_symbols(r, mtf); s != Status::Ok) return s;
    if (r.error || mtf.size() != bwt_n) return Status::CorruptInput;
  } else if (mode == 1) {
    thread_local std::vector<u8> sym, runlen;
    sym.clear();
    runlen.clear();
    if (Status s = decode_bwt_symbols(r, sym); s != Status::Ok) return s;
    if (Status s = decode_bwt_symbols(r, runlen); s != Status::Ok) return s;
    u64 rx_len = r.varint();
    const u8* rx = r.take(static_cast<usize>(rx_len));
    if (!rx) return Status::CorruptInput;

    // Fused expand + inverse MTF, one pass, no intermediate rank buffer: each
    // 0 in `sym` is a zero-run (length from `runlen`, value 255 escaping to a
    // varint in the run-extra blob), which under MTF inversion just repeats
    // the current front byte - a memset. Non-zero ranks move-to-front as
    // usual.
    ByteReader re(rx, static_cast<usize>(rx_len));
    bwt_decode::MtfList list;
    list.init();
    u8 front = 0;
    mtf.resize(bwt_n);
    u8* out_p = mtf.data();
    usize o = 0;
    usize ri = 0;
    for (u8 s : sym) {
      if (s != 0) {
        if (o + 1 > bwt_n) return Status::CorruptInput;
        front = list.pull(s);
        out_p[o++] = front;
        continue;
      }
      if (ri >= runlen.size()) return Status::CorruptInput;
      u32 run = runlen[ri++];
      if (run == 255) run += static_cast<u32>(re.varint());
      if (re.error || o + run > bwt_n) return Status::CorruptInput;
      std::memset(out_p + o, front, run);
      o += run;
    }
    if (ri != runlen.size() || o != bwt_n) return Status::CorruptInput;
    mtf_done = true;
  } else if (mode == 2) {
    thread_local std::vector<u8> brle;
    brle.clear();
    if (Status s = decode_bwt_symbols(r, brle); s != Status::Ok) return s;
    // Fused RUNA/RUNB expansion + inverse MTF straight into `mtf` (one pass).
    mtf.resize(bwt_n);
    const usize got = bwt_mtf_bzip2_inverse(brle.data(), brle.size(),
                                            mtf.data(), bwt_n);
    if (got != bwt_n) return Status::CorruptInput;
    mtf_done = true;
  } else if (mode == 3) {
    // Raw L-column: same framing as mode 0, skip inverse MTF.
    if (Status s = decode_bwt_symbols(r, mtf); s != Status::Ok) return s;
    if (r.error || mtf.size() != bwt_n) return Status::CorruptInput;
    mtf_done = true;
  } else if (mode == 4) {
    // Raw L-column with BWT Context-Mixer
    mtf.resize(bwt_n);
    const usize rem = r.remaining();
    const u8* body = r.take(rem);
    if (!body) return Status::CorruptInput;
    if (Status s = bwt_cm_decode(body, rem, mtf.data(), bwt_n); s != Status::Ok)
      return s;
    mtf_done = true;
  } else if (mode == 5) {
    // Raw L-column with BWT Ultra (Bidirectional Context-Mixer)
    mtf.resize(bwt_n);
    const usize rem = r.remaining();
    const u8* body = r.take(rem);
    if (!body) return Status::CorruptInput;
    if (Status s = bwt_ultra_decode(body, rem, mtf.data(), bwt_n); s != Status::Ok)
      return s;
    mtf_done = true;
  } else {
    return Status::CorruptInput;
  }

  if (!mtf_done) mtf_inverse(mtf.data(), bwt_n);

  if (has_lzp) {
    thread_local std::vector<u8> lzp_tmp;
    lzp_tmp.resize(bwt_n);
    Status st = bwt_inverse(mtf.data(), bwt_n, static_cast<usize>(primary),
                            lzp_tmp.data(),
                            num_chains > 1 ? anchors : nullptr, num_chains);
    if (st != Status::Ok) return st;
    return lzp_decompress(lzp_tmp.data(), bwt_n, dst, n);
  }

  return bwt_inverse(mtf.data(), n, static_cast<usize>(primary), dst,
                     num_chains > 1 ? anchors : nullptr, num_chains);
}

namespace {

struct FrameHeader {
  u8 flags = 0;
  int bslog = 0;
  u64 content_size = 0;
  bool has_content_size = false;
  bool has_checksum = false;
  bool has_bcsum = false;
  bool has_filters = false;
  const u8* blocks_start = nullptr;
};

Status parse_header(ByteReader& r, FrameHeader& h) {
  const u8* mg = r.take(kMagicLen);
  if (!mg) return Status::BadMagic;
  if (std::memcmp(mg, kMagic, kMagicLen) != 0) return Status::BadMagic;
  u8 ver = r.byte();
  if (r.error) return Status::CorruptInput;
  if (ver != kVersion) return Status::UnsupportedVersion;
  h.flags = r.byte();
  h.bslog = r.byte();
  if (r.error) return Status::CorruptInput;
  h.has_content_size = (h.flags & kFlagContentSize) != 0;
  h.has_checksum = (h.flags & kFlagChecksum) != 0;
  h.has_bcsum = (h.flags & kFlagBlockChecksums) != 0;
  h.has_filters = (h.flags & kFlagFilters) != 0;
  if (h.has_content_size) {
    h.content_size = r.varint();
    if (r.error) return Status::CorruptInput;
  }
  h.blocks_start = r.p;
  return Status::Ok;
}

struct BlockRef {
  BlockType type;
  usize regen;
  usize out_off;
  const u8* data;   // payload (compressed) or raw bytes (stored)
  usize data_len;
  u8 mval;          // memset value
  bool has_hash;
  u32 hash;
  Filter filter = Filter::None;
  bool filter_rle = false;
  usize filter_intermediate = 0;
};

Status decode_filtered_payload(BlockType codec, const u8* data, usize data_len,
                               const BlockRef& b, u8* dst) {
  thread_local std::vector<u8> scratch, filtered;
  if (filter_is_framed(b.filter)) {
    // LZ/BWT decodes to the self-framed stream (its length came from the
    // block header), which then expands to the raw block bytes.
    scratch.resize(b.filter_intermediate);
    Status st;
    if (codec == BlockType::Compressed)
      st = decode_compressed_block(data, data_len, scratch.data(),
                                   b.filter_intermediate);
    else if (codec == BlockType::Mosaic)
      st = decode_mosaic_block(data, data_len, scratch.data(),
                             b.filter_intermediate);
    else if (codec == BlockType::Oracle)
      st = decode_oracle_block(data, data_len, scratch.data(),
                           b.filter_intermediate);
    else if (codec == BlockType::Crystal)
      st = decode_crystal_block(data, data_len, scratch.data(),
                               b.filter_intermediate);
    else if (codec == BlockType::Flash)
      st = decode_flash_block(data, data_len, scratch.data(),
                              b.filter_intermediate);
    else if (codec == BlockType::Laser)
      st = decode_laser_block(data, data_len, scratch.data(),
                              b.filter_intermediate);
    else if (codec == BlockType::Grating)
      st = decode_grating_block(data, data_len, scratch.data(),
                                b.filter_intermediate);
    else
      st = decode_bwt_block(data, data_len, scratch.data(),
                            b.filter_intermediate);
    if (st != Status::Ok) return st;
    return framed_filter_inv(b.filter, scratch.data(), b.filter_intermediate, dst,
                             b.regen);
  }
  if (b.filter_rle) {
    scratch.resize(b.filter_intermediate);
    Status st;
    if (codec == BlockType::Compressed)
      st = decode_compressed_block(data, data_len, scratch.data(),
                                   b.filter_intermediate);
    else if (codec == BlockType::Mosaic)
      st = decode_mosaic_block(data, data_len, scratch.data(),
                             b.filter_intermediate);
    else if (codec == BlockType::Oracle)
      st = decode_oracle_block(data, data_len, scratch.data(),
                           b.filter_intermediate);
    else if (codec == BlockType::Crystal)
      st = decode_crystal_block(data, data_len, scratch.data(),
                               b.filter_intermediate);
    else if (codec == BlockType::Flash)
      st = decode_flash_block(data, data_len, scratch.data(),
                              b.filter_intermediate);
    else if (codec == BlockType::Laser)
      st = decode_laser_block(data, data_len, scratch.data(),
                              b.filter_intermediate);
    else if (codec == BlockType::Grating)
      st = decode_grating_block(data, data_len, scratch.data(),
                                b.filter_intermediate);
    else
      st = decode_bwt_block(data, data_len, scratch.data(), b.filter_intermediate);
    if (st != Status::Ok) return st;
    if (b.filter == Filter::None) {
      return byte_rle_decode(scratch.data(), b.filter_intermediate, dst,
                             b.regen);
    }
    filtered.resize(b.regen);
    st = byte_rle_decode(scratch.data(), b.filter_intermediate, filtered.data(),
                         b.regen);
    if (st != Status::Ok) return st;
    invert_filter(b.filter, filtered.data(), b.regen, dst);
    return Status::Ok;
  }
  scratch.resize(b.regen);
  Status st;
  if (codec == BlockType::Compressed)
    st = decode_compressed_block(data, data_len, scratch.data(), b.regen);
  else if (codec == BlockType::Mosaic)
    st = decode_mosaic_block(data, data_len, scratch.data(), b.regen);
  else if (codec == BlockType::Oracle)
    st = decode_oracle_block(data, data_len, scratch.data(), b.regen);
  else if (codec == BlockType::Crystal)
    st = decode_crystal_block(data, data_len, scratch.data(), b.regen);
  else if (codec == BlockType::Flash)
    st = decode_flash_block(data, data_len, scratch.data(), b.regen);
  else if (codec == BlockType::Laser)
    st = decode_laser_block(data, data_len, scratch.data(), b.regen);
  else if (codec == BlockType::Grating)
    st = decode_grating_block(data, data_len, scratch.data(), b.regen);
  else
    st = decode_bwt_block(data, data_len, scratch.data(), b.regen);
  if (st != Status::Ok) return st;
  invert_filter(b.filter, scratch.data(), b.regen, dst);
  return Status::Ok;
}

// Read one block (header + payload range) at the reader. Sets `last`.
Status scan_block(ByteReader& r, bool has_bcsum, bool has_filters, usize out_off,
                  BlockRef& br, bool& last) {
  u8 bt = r.byte();
  if (r.error) return Status::CorruptInput;
  last = (bt & kBlockLastFlag) != 0;
  br.type = static_cast<BlockType>(bt & kBlockTypeMask);
  br.regen = static_cast<usize>(r.varint());
  if (r.error) return Status::CorruptInput;
  br.out_off = out_off;
  br.has_hash = has_bcsum;
  if (has_bcsum) {
    const u8* h = r.take(4);
    if (!h) return Status::CorruptInput;
    br.hash = load_u32(h);
  }
  switch (br.type) {
    case BlockType::Stored: {
      const u8* p = r.take(br.regen);
      if (!p) return Status::CorruptInput;
      br.data = p;
      br.data_len = br.regen;
      break;
    }
    case BlockType::Memset: {
      br.mval = r.byte();
      if (r.error) return Status::CorruptInput;
      break;
    }
    case BlockType::Compressed:
    case BlockType::Bwt:
    case BlockType::Mosaic:
    case BlockType::Oracle:
    case BlockType::Crystal:
    case BlockType::Flash:
    case BlockType::Laser:
    case BlockType::Grating: {
      if (has_filters) {
        const u8 fb = r.byte();
        if (r.error || !filter_is_valid(fb)) return Status::CorruptInput;
        br.filter = filter_id_from_byte(fb);
        br.filter_rle = filter_byte_has_rle(fb);
        // Byte-RLE never stacks on the dictionary filter (its stream is
        // already dense); reject the combination outright.
        if (br.filter_rle && filter_is_framed(br.filter))
          return Status::CorruptInput;
        if (br.filter_rle || filter_is_framed(br.filter)) {
          br.filter_intermediate = static_cast<usize>(r.varint());
          // Both intermediates (byte-RLE stream, dictionary stream) are only
          // ever emitted when strictly smaller than the raw block, so a
          // larger value is corruption - and bounding it here keeps a hostile
          // varint from driving a huge scratch allocation. For digit normalization,
          // the intermediate contains both the normalized stream and packed side stream.
          const usize max_inter = (filter_is_digit_norm(br.filter) ||
                                     filter_is_word_norm(br.filter) ||
                                     filter_is_base64(br.filter))
                                      ? (br.regen + (br.regen >> 1) + 256)
                                      : br.regen;
          if (r.error || br.filter_intermediate == 0 ||
              br.filter_intermediate >= max_inter)
            return Status::CorruptInput;
        }
      }
      const usize plen = static_cast<usize>(r.varint());
      const u8* p = r.take(plen);
      if (!p) return Status::CorruptInput;
      br.data = p;
      br.data_len = plen;
      break;
    }
    default:
      return Status::CorruptInput;
  }
  return Status::Ok;
}

Status decode_block_ref(const BlockRef& b, u8* dst) {
  Status st = Status::Ok;
  switch (b.type) {
    case BlockType::Stored:
      std::memcpy(dst + b.out_off, b.data, b.regen);
      break;
    case BlockType::Memset:
      std::memset(dst + b.out_off, b.mval, b.regen);
      break;
    case BlockType::Compressed:
      if (b.filter == Filter::None && !b.filter_rle)
        st = decode_compressed_block(b.data, b.data_len, dst + b.out_off,
                                     b.regen);
      else
        st = decode_filtered_payload(BlockType::Compressed, b.data, b.data_len,
                                     b, dst + b.out_off);
      break;
    case BlockType::Bwt:
      if (b.filter == Filter::None && !b.filter_rle)
        st = decode_bwt_block(b.data, b.data_len, dst + b.out_off, b.regen);
      else
        st = decode_filtered_payload(BlockType::Bwt, b.data, b.data_len, b,
                                     dst + b.out_off);
      break;
    case BlockType::Mosaic:
      if (b.filter == Filter::None && !b.filter_rle)
        st = decode_mosaic_block(b.data, b.data_len, dst + b.out_off, b.regen);
      else
        st = decode_filtered_payload(BlockType::Mosaic, b.data, b.data_len, b,
                                     dst + b.out_off);
      break;
    case BlockType::Oracle:
      if (b.filter == Filter::None && !b.filter_rle)
        st = decode_oracle_block(b.data, b.data_len, dst + b.out_off, b.regen);
      else
        st = decode_filtered_payload(BlockType::Oracle, b.data, b.data_len, b,
                                     dst + b.out_off);
      break;
    case BlockType::Crystal:
      if (b.filter == Filter::None && !b.filter_rle)
        st = decode_crystal_block(b.data, b.data_len, dst + b.out_off, b.regen);
      else
        st = decode_filtered_payload(BlockType::Crystal, b.data, b.data_len, b,
                                     dst + b.out_off);
      break;
    case BlockType::Flash:
      if (b.filter == Filter::None && !b.filter_rle)
        st = decode_flash_block(b.data, b.data_len, dst + b.out_off, b.regen);
      else
        st = decode_filtered_payload(BlockType::Flash, b.data, b.data_len, b,
                                     dst + b.out_off);
      break;
    case BlockType::Laser:
      if (b.filter == Filter::None && !b.filter_rle)
        st = decode_laser_block(b.data, b.data_len, dst + b.out_off, b.regen);
      else
        st = decode_filtered_payload(BlockType::Laser, b.data, b.data_len, b,
                                     dst + b.out_off);
      break;
    case BlockType::Grating:
      if (b.filter == Filter::None && !b.filter_rle)
        st = decode_grating_block(b.data, b.data_len, dst + b.out_off, b.regen);
      else
        st = decode_filtered_payload(BlockType::Grating, b.data, b.data_len, b,
                                     dst + b.out_off);
      break;
  }
  if (st == Status::Ok && b.has_hash) {
    if (static_cast<u32>(content_hash(dst + b.out_off, b.regen)) != b.hash)
      return Status::ChecksumMismatch;
  }
  return st;
}

}  // namespace

Status frame_decompress(const u8* src, usize n, u8* dst, usize dst_cap,
                        const DecodeParams& p, usize& written) {
  ByteReader r(src, n);
  FrameHeader h;
  if (Status s = parse_header(r, h); s != Status::Ok) return s;

  // Sequential pre-scan: cheap, and gives every block's input/output range.
  std::vector<BlockRef> refs;
  usize out_pos = 0;
  for (;;) {
    BlockRef br;
    bool last;
    if (Status s = scan_block(r, h.has_bcsum, h.has_filters, out_pos, br, last);
        s != Status::Ok)
      return s;
    if (out_pos + br.regen > dst_cap) return Status::BufferTooSmall;
    out_pos += br.regen;
    refs.push_back(br);
    if (last) break;
  }

  // Content checksum sits immediately after the last block.
  u64 stored_checksum = 0;
  if (h.has_checksum) {
    const u8* cs = r.take(8);
    if (!cs) return Status::CorruptInput;
    stored_checksum = load_u64(cs);
  }

  // Decode blocks (optionally in parallel); first error wins. Frames made
  // entirely of stored/memset blocks (e.g. random/incompressible input) are
  // pure memcpy/memset: spawning threads for them only adds overhead, so they
  // stay single-threaded regardless of the requested thread count.
  int dec_threads = p.threads;
  if (dec_threads != 1) {
    bool any_cpu_heavy = false;
    for (const BlockRef& b : refs)
      if (b.type == BlockType::Compressed || b.type == BlockType::Bwt ||
          b.type == BlockType::Mosaic || b.type == BlockType::Oracle ||
          b.type == BlockType::Crystal || b.type == BlockType::Flash ||
          b.type == BlockType::Laser || b.type == BlockType::Grating) {
        any_cpu_heavy = true;
        break;
      }
    if (!any_cpu_heavy) dec_threads = 1;
  }
  std::atomic<int> err{static_cast<int>(Status::Ok)};
  parallel_for(refs.size(), dec_threads, [&](usize i) {
    if (err.load(std::memory_order_relaxed) != 0) return;
    Status st = decode_block_ref(refs[i], dst);
    if (st != Status::Ok) {
      int expected = 0;
      err.compare_exchange_strong(expected, static_cast<int>(st));
    }
  });
  if (err.load() != 0) return static_cast<Status>(err.load());

  if (h.has_checksum && content_hash(dst, out_pos) != stored_checksum)
    return Status::ChecksumMismatch;
  if (h.has_content_size && h.content_size != out_pos) return Status::CorruptInput;

  written = out_pos;
  return Status::Ok;
}

bool parse_seek_table(const u8* src, usize n,
                      std::vector<SeekEntryInternal>& out) {
  if (n < 8) return false;
  const u8* end = src + n;
  if (std::memcmp(end - 4, kSeekMagic, 4) != 0) return false;
  u32 num_blocks = load_u32(end - 8);
  usize table_bytes = static_cast<usize>(num_blocks) * 8;
  if (table_bytes + 8 > n) return false;
  const u8* t = end - 8 - table_bytes;
  out.resize(num_blocks);
  for (u32 i = 0; i < num_blocks; ++i) {
    out[i].comp_size = load_u32(t + i * 8);
    out[i].regen_size = load_u32(t + i * 8 + 4);
  }
  return true;
}

Status decode_block_at(const u8* src, usize n, usize index, u8* dst,
                       usize dst_cap, usize& written) {
  ByteReader r(src, n);
  FrameHeader h;
  if (Status s = parse_header(r, h); s != Status::Ok) return s;

  // Fast path: jump using the seek table if present.
  std::vector<SeekEntryInternal> seek;
  if ((h.flags & kFlagSeekTable) && parse_seek_table(src, n, seek)) {
    if (index >= seek.size()) return Status::InvalidArgument;
    usize off = static_cast<usize>(h.blocks_start - src);
    for (usize i = 0; i < index; ++i) off += static_cast<usize>(seek[i].comp_size);
    if (off > n) return Status::CorruptInput;
    ByteReader br_reader(src + off, n - off);
    BlockRef br;
    bool last;
    if (Status s = scan_block(br_reader, h.has_bcsum, h.has_filters, 0, br, last);
        s != Status::Ok)
      return s;
    if (br.regen > dst_cap) return Status::BufferTooSmall;
    if (Status s = decode_block_ref(br, dst); s != Status::Ok) return s;
    written = br.regen;
    return Status::Ok;
  }

  // Fallback: scan block headers up to `index`.
  usize cur = 0;
  for (;;) {
    BlockRef br;
    bool last;
    if (Status s = scan_block(r, h.has_bcsum, h.has_filters, 0, br, last);
        s != Status::Ok)
      return s;
    if (cur == index) {
      if (br.regen > dst_cap) return Status::BufferTooSmall;
      if (Status s = decode_block_ref(br, dst); s != Status::Ok) return s;
      written = br.regen;
      return Status::Ok;
    }
    if (last) return Status::InvalidArgument;  // index out of range
    ++cur;
  }
}

Status list_block_regens(const u8* src, usize n, std::vector<u64>& out) {
  ByteReader r(src, n);
  FrameHeader h;
  if (Status s = parse_header(r, h); s != Status::Ok) return s;
  out.clear();
  for (;;) {
    BlockRef br;
    bool last;
    if (Status s = scan_block(r, h.has_bcsum, h.has_filters, 0, br, last);
        s != Status::Ok)
      return s;
    out.push_back(br.regen);
    if (last) break;
  }
  return Status::Ok;
}

}  // namespace spectra8::detail
