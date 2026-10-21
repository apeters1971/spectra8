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

// API-level and edge-case tests: bounds, error reporting, determinism, the C
// ABI, and coverage of each entropy mode. Returns non-zero on any failure.
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "spectra8/spectra8.hpp"
#include "spectra8/spectra8_c.h"
#include "spectra8/page_service.hpp"

#include "prism/bwt.hpp"  // internal primitives, for direct unit tests
#include "prism/bwt_cm.hpp"
#include "prism/bwt_ultra.hpp"
#include "util/arith.hpp"
#include "util/byte_rle.hpp"
#include "prism/bwt_mtf_rle.hpp"
#include "util/dict_filter.hpp"
#include "util/pack_filter.hpp"
#include "ext/filter_frame.hpp"
#include "flash/flash.hpp"
#include "util/huffman.hpp"
#include "util/tans.hpp"
#include "util/transform.hpp"

namespace {

int g_failures = 0;
int g_checks = 0;

void check(bool cond, const std::string& what) {
  ++g_checks;
  if (!cond) {
    ++g_failures;
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
  }
}

std::vector<std::byte> bytes(const std::vector<uint8_t>& v) {
  std::vector<std::byte> out(v.size());
  if (!v.empty()) std::memcpy(out.data(), v.data(), v.size());
  return out;
}

std::vector<std::byte> compress_ok(const std::vector<uint8_t>& in, int level) {
  auto src = bytes(in);
  std::vector<std::byte> comp(spectra8::compress_bound(src.size()));
  auto r = spectra8::compress(src, comp, level);
  check(r.ok(), "compress ok");
  comp.resize(r.ok() ? r.value : 0);
  return comp;
}

bool roundtrips(const std::vector<uint8_t>& in, int level) {
  auto comp = compress_ok(in, level);
  std::vector<std::byte> out(in.size());
  auto r = spectra8::decompress(comp, out);
  if (!r || r.value != in.size()) return false;
  return in.empty() || std::memcmp(out.data(), in.data(), in.size()) == 0;
}

// ---------------------------------------------------------------------------

void test_compress_bound() {
  std::mt19937_64 rng(1);
  for (size_t sz : {0u, 1u, 100u, 4096u, 300000u, 1100000u}) {
    std::vector<uint8_t> d(sz);
    for (auto& b : d) b = static_cast<uint8_t>(rng());
    auto src = bytes(d);
    std::vector<std::byte> comp(spectra8::compress_bound(sz));
    auto r = spectra8::compress(src, comp, 6);
    check(r.ok(), "bound: compress ok sz=" + std::to_string(sz));
    check(r.value <= spectra8::compress_bound(sz),
          "bound: result within bound sz=" + std::to_string(sz));
  }
}

void test_buffer_too_small() {
  std::vector<uint8_t> d(5000, 0);
  for (size_t i = 0; i < d.size(); ++i) d[i] = static_cast<uint8_t>(i);
  auto src = bytes(d);

  // Compress into a 1-byte buffer -> BufferTooSmall.
  std::vector<std::byte> tiny(1);
  auto cr = spectra8::compress(src, tiny, 6);
  check(cr.status == spectra8::Status::BufferTooSmall,
        "buffer: compress too small reported");

  // Decompress into a buffer smaller than the content -> BufferTooSmall.
  auto comp = compress_ok(d, 6);
  std::vector<std::byte> small_out(d.size() / 2);
  auto dr = spectra8::decompress(comp, small_out);
  check(dr.status == spectra8::Status::BufferTooSmall,
        "buffer: decompress too small reported");
}

void test_bad_inputs() {
  // Random non-frame bytes.
  std::vector<uint8_t> junk(64);
  for (size_t i = 0; i < junk.size(); ++i) junk[i] = static_cast<uint8_t>(i * 7);
  auto j = bytes(junk);
  std::vector<std::byte> out(1024);
  auto dr = spectra8::decompress(j, out);
  check(dr.status == spectra8::Status::BadMagic, "bad: junk -> BadMagic");
  check(!spectra8::get_frame_content_size(j).has_value(),
        "bad: junk content size nullopt");

  // Empty input -> BadMagic.
  std::vector<std::byte> empty;
  auto er = spectra8::decompress(empty, out);
  check(er.status == spectra8::Status::BadMagic, "bad: empty -> BadMagic");
}

void test_corruption_detection() {
  std::mt19937_64 rng(99);
  std::vector<uint8_t> d(40000);
  for (auto& b : d) b = static_cast<uint8_t>(rng());
  // The whole-frame checksum is opt-in, so enable it for this test.
  spectra8::CompressParams cp;
  cp.level = 6;
  cp.content_checksum = true;
  auto src = bytes(d);
  std::vector<std::byte> comp(spectra8::compress_bound(src.size()));
  {
    auto r = spectra8::compress(src, comp, cp);
    check(r.ok(), "compress ok");
    comp.resize(r.ok() ? r.value : 0);
  }

  auto decode_status = [&](const std::vector<std::byte>& c) {
    std::vector<std::byte> out(d.size());
    return spectra8::decompress(c, out).status;
  };

  // Flip magic -> BadMagic.
  {
    auto c = comp;
    c[0] = static_cast<std::byte>(0xFF);
    check(decode_status(c) == spectra8::Status::BadMagic,
          "corrupt: magic -> BadMagic");
  }
  // Flip version -> UnsupportedVersion.
  {
    auto c = comp;
    c[4] = static_cast<std::byte>(0x7E);
    check(decode_status(c) == spectra8::Status::UnsupportedVersion,
          "corrupt: version -> UnsupportedVersion");
  }
  // Flip a checksum byte -> ChecksumMismatch (frame still decodes structurally).
  {
    auto c = comp;
    c.back() = static_cast<std::byte>(static_cast<uint8_t>(c.back()) ^ 0xFF);
    check(decode_status(c) == spectra8::Status::ChecksumMismatch,
          "corrupt: checksum -> ChecksumMismatch");
  }
  // Truncated frame -> CorruptInput (or BadMagic if cut into the header).
  {
    auto c = comp;
    c.resize(c.size() / 2);
    auto st = decode_status(c);
    check(st != spectra8::Status::Ok, "corrupt: truncated not Ok");
  }
}

void test_determinism() {
  std::vector<uint8_t> d;
  for (int i = 0; i < 50000; ++i) d.push_back(static_cast<uint8_t>((i * 13) ^ (i >> 4)));
  for (int lvl = 1; lvl <= 9; ++lvl) {
    auto a = compress_ok(d, lvl);
    auto b = compress_ok(d, lvl);
    check(a == b, "determinism level " + std::to_string(lvl));
  }
}

// Exercise each block / entropy mode through the public API.
void test_entropy_modes() {
  std::mt19937_64 rng(7);

  // memset block: all equal.
  check(roundtrips(std::vector<uint8_t>(200000, 0xAB), 9), "mode: memset");

  // RLE-friendly: long runs of a few values.
  {
    std::vector<uint8_t> d;
    for (int i = 0; i < 4000; ++i) {
      uint8_t v = static_cast<uint8_t>(i & 3);
      d.insert(d.end(), 50, v);
    }
    check(roundtrips(d, 6), "mode: rle-ish");
  }

  // Huffman-friendly: skewed distribution, no long matches.
  {
    std::vector<uint8_t> d(100000);
    for (auto& b : d) {
      uint64_t r = rng();
      b = static_cast<uint8_t>((r % 100 < 80) ? 'e' : ('a' + r % 26));
    }
    check(roundtrips(d, 6), "mode: huffman");
    check(roundtrips(d, 9), "mode: ans (level 9)");
  }

  // Raw fallback: incompressible random.
  {
    std::vector<uint8_t> d(100000);
    for (auto& b : d) b = static_cast<uint8_t>(rng());
    check(roundtrips(d, 1), "mode: raw l1");
    check(roundtrips(d, 9), "mode: raw l9");
  }

  // Small alphabet (DNA-like) with matches.
  {
    std::vector<uint8_t> d(300000);
    const char alpha[4] = {'A', 'C', 'G', 'T'};
    for (auto& b : d) b = static_cast<uint8_t>(alpha[rng() & 3]);
    check(roundtrips(d, 7), "mode: dna");
  }
}

// Pin the entropy coders' payload formats: a fixed input must produce a fixed
// payload (FNV-1a hash) and decode back, for each rANS interleave width (mode
// 1/2/3 by array size), the order-1 rANS coder and the Huffman coder. Catches
// accidental bitstream drift in the coders themselves; update the constants
// deliberately when the format changes.
uint64_t fnv1a(const std::vector<uint8_t>& v) {
  uint64_t h = 1469598103934665603ull;
  for (uint8_t b : v) h = (h ^ b) * 1099511628211ull;
  return h;
}

void test_entropy_golden() {
  using namespace spectra8::detail;
  auto skewed = [](size_t n, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::vector<uint8_t> d(n);
    for (auto& b : d) {
      uint64_t r = rng();
      b = static_cast<uint8_t>((r % 100 < 60) ? (r >> 8) % 4 : (r >> 8) % 64);
    }
    return d;
  };
  struct Case {
    size_t n;
    uint64_t golden;
    const char* what;
  };
  // rANS order-0: 2-way (< 2048 symbols), 4-way (< 8192), 8-way.
  const Case ans_cases[] = {
      {1000, 0xb6c6bd0e2f2935abull, "ans 2-way"},
      {4000, 0x55dbf1182bbca50aull, "ans 4-way"},
      {100000, 0xba220d2fb8db14e4ull, "ans 8-way"},
  };
  for (const Case& c : ans_cases) {
    std::vector<uint8_t> d = skewed(c.n, c.n);
    std::vector<uint8_t> enc;
    ans::encode(enc, d.data(), d.size());
    std::vector<uint8_t> dec(d.size());
    check(ans::decode(enc.data(), enc.size(), dec.data(), dec.size()) ==
                  spectra8::Status::Ok && dec == d,
          std::string("golden roundtrip: ") + c.what);
    const uint64_t h = fnv1a(enc);
    if (c.golden == 0)
      std::printf("golden %-10s n=%zu payload=%zu fnv=0x%016llx\n", c.what, c.n,
                  enc.size(), static_cast<unsigned long long>(h));
    else
      check(h == c.golden, std::string("golden payload: ") + c.what);
  }
  {
    std::vector<uint8_t> d = skewed(50000, 99);
    std::vector<uint8_t> enc;
    ans::encode_o1(enc, d.data(), d.size());
    std::vector<uint8_t> dec(d.size());
    check(ans::decode_o1(enc.data(), enc.size(), dec.data(), dec.size()) ==
                  spectra8::Status::Ok && dec == d,
          "golden roundtrip: ans o1");
    const uint64_t h = fnv1a(enc);
    const uint64_t golden = 0x1ee00e4616cfd04aull;
    if (golden == 0)
      std::printf("golden ans-o1 payload=%zu fnv=0x%016llx\n", enc.size(),
                  static_cast<unsigned long long>(h));
    else
      check(h == golden, "golden payload: ans o1");
  }
  {
    std::vector<uint8_t> d = skewed(50000, 5);
    std::vector<uint8_t> enc;
    huff::encode(enc, d.data(), d.size());
    std::vector<uint8_t> dec(d.size());
    check(huff::decode(enc.data(), enc.size(), dec.data(), dec.size()) ==
                  spectra8::Status::Ok && dec == d,
          "golden roundtrip: huffman");
    const uint64_t h = fnv1a(enc);
    const uint64_t golden = 0x78177cc3df3f17c7ull;
    if (golden == 0)
      std::printf("golden huffman payload=%zu fnv=0x%016llx\n", enc.size(),
                  static_cast<unsigned long long>(h));
    else
      check(h == golden, "golden payload: huffman");
  }
}

void test_all_single_byte_values() {
  for (int v = 0; v < 256; ++v) {
    std::vector<uint8_t> d(1000 + v, static_cast<uint8_t>(v));
    if (!roundtrips(d, 6)) {
      check(false, "single-byte value " + std::to_string(v));
      return;
    }
  }
  check(true, "all single-byte values roundtrip");
}

void test_large_input() {
  std::mt19937_64 rng(2024);
  std::vector<uint8_t> d(6u << 20);  // 6 MiB crossing several blocks
  size_t i = 0;
  while (i < d.size()) {
    if (rng() % 7 == 0) {
      d[i++] = static_cast<uint8_t>(rng());
    } else {
      const char* w = "spectra8 roundtrip ";
      for (size_t k = 0; w[k] && i < d.size(); ++k) d[i++] = static_cast<uint8_t>(w[k]);
    }
  }
  check(roundtrips(d, 6), "large: 6 MiB level 6");
  check(roundtrips(d, 9), "large: 6 MiB level 9");
}

void test_c_abi() {
  std::vector<uint8_t> d(20000);
  for (size_t i = 0; i < d.size(); ++i) d[i] = static_cast<uint8_t>((i * 17) ^ (i >> 3));

  size_t bound = spectra8_compress_bound(d.size());
  std::vector<uint8_t> comp(bound);
  size_t written = 0;
  spectra8_status cs =
      spectra8_compress(d.data(), d.size(), comp.data(), comp.size(),
                         SPECTRA8_DEFAULT_LEVEL, &written);
  check(cs == SPECTRA8_OK, "c-abi: compress ok");
  comp.resize(written);

  uint64_t content = 0;
  int has = spectra8_frame_content_size(comp.data(), comp.size(), &content);
  check(has == 1 && content == d.size(), "c-abi: content size");

  std::vector<uint8_t> out(d.size());
  size_t dwritten = 0;
  spectra8_status ds = spectra8_decompress(comp.data(), comp.size(),
                                             out.data(), out.size(), &dwritten);
  check(ds == SPECTRA8_OK && dwritten == d.size(), "c-abi: decompress ok");
  check(std::memcmp(out.data(), d.data(), d.size()) == 0, "c-abi: data matches");

  check(spectra8_status_message(SPECTRA8_OK) != nullptr, "c-abi: status msg");
  check(spectra8_version() != nullptr && spectra8_version()[0] != '\0',
        "c-abi: version");

  check(spectra8_codec_available(SPECTRA8_CODEC_BEAM) == 1, "c-abi: lz avail");
  check(spectra8_codec_available(SPECTRA8_CODEC_PRISM) == 1, "c-abi: bwt avail");
  check(spectra8_codec_available(SPECTRA8_CODEC_MOSAIC) == 1, "c-abi: mosaic avail");
  check(spectra8_codec_available(SPECTRA8_CODEC_ORACLE) == 1, "c-abi: oracle avail");
  check(spectra8_codec_available(SPECTRA8_CODEC_CRYSTAL) == 1,
        "c-abi: crystal avail");
  check(spectra8_codec_available(SPECTRA8_CODEC_LASER) == 1,
        "c-abi: laser avail");
  check(spectra8_codec_available(SPECTRA8_CODEC_GRATING) == 1,
        "c-abi: grating avail");
  {
    size_t lb = spectra8_compress_bound_codec(d.size(), SPECTRA8_CODEC_LASER);
    std::vector<uint8_t> lc(lb);
    size_t lw = 0;
    check(spectra8_compress_ex(d.data(), d.size(), lc.data(), lc.size(),
                                SPECTRA8_CODEC_LASER, 6, &lw) == SPECTRA8_OK,
          "c-abi: laser compress_ex");
    lc.resize(lw);
    uint64_t lcs = 0;
    check(spectra8_frame_content_size(lc.data(), lc.size(), &lcs) == 1 &&
              lcs == d.size(),
          "c-abi: laser content size");
    std::vector<uint8_t> lo(d.size());
    size_t low = 0;
    check(spectra8_decompress(lc.data(), lc.size(), lo.data(), lo.size(),
                               &low) == SPECTRA8_OK &&
              low == d.size() &&
              std::memcmp(lo.data(), d.data(), d.size()) == 0,
          "c-abi: laser roundtrip");
  }
  if (spectra8_codec_available(SPECTRA8_CODEC_ZSTD)) {
    size_t zb = spectra8_compress_bound_codec(d.size(), SPECTRA8_CODEC_ZSTD);
    std::vector<uint8_t> zc(zb);
    size_t zw = 0;
    check(spectra8_compress_ex(d.data(), d.size(), zc.data(), zc.size(),
                                SPECTRA8_CODEC_ZSTD, 3, &zw) == SPECTRA8_OK,
          "c-abi: zstd compress_ex");
    zc.resize(zw);
    uint64_t zcs = 0;
    check(spectra8_frame_content_size(zc.data(), zc.size(), &zcs) == 1 &&
              zcs == d.size(),
          "c-abi: zstd content size");
    std::vector<uint8_t> zo(d.size());
    size_t odw = 0;
    check(spectra8_decompress(zc.data(), zc.size(), zo.data(), zo.size(),
                               &odw) == SPECTRA8_OK &&
              odw == d.size() &&
              std::memcmp(zo.data(), d.data(), d.size()) == 0,
          "c-abi: zstd roundtrip");
  }
}

void test_version() {
  check(!spectra8::version().empty(), "version non-empty");
}

std::vector<uint8_t> make_multiblock(size_t n) {
  std::mt19937_64 rng(555);
  std::vector<uint8_t> d(n);
  size_t i = 0;
  while (i < n) {
    if (rng() % 6 == 0) {
      d[i++] = static_cast<uint8_t>(rng());
    } else {
      const char* w = "multiblock parallel spectra8 data ";
      for (size_t k = 0; w[k] && i < n; ++k) d[i++] = static_cast<uint8_t>(w[k]);
    }
  }
  return d;
}

void test_threading_determinism() {
  auto d = make_multiblock(2'500'000);  // ~10 blocks at default size
  auto src = bytes(d);

  for (int lvl : {1, 6, 9}) {
    spectra8::CompressParams p1;
    p1.level = lvl;
    p1.threads = 1;
    spectra8::CompressParams p4 = p1;
    p4.threads = 4;

    std::vector<std::byte> c1(spectra8::compress_bound(d.size()));
    std::vector<std::byte> c4(spectra8::compress_bound(d.size()));
    auto r1 = spectra8::compress(src, c1, p1);
    auto r4 = spectra8::compress(src, c4, p4);
    check(r1.ok() && r4.ok(), "threads: compress ok");
    c1.resize(r1.value);
    c4.resize(r4.value);
    check(c1 == c4, "threads: output identical for 1 vs 4 threads L" +
                        std::to_string(lvl));

    // Multithreaded decode matches.
    spectra8::DecompressParams dp;
    dp.threads = 4;
    std::vector<std::byte> out(d.size());
    auto dr = spectra8::decompress(c4, out, dp);
    check(dr.ok() && dr.value == d.size(), "threads: parallel decode size");
    check(std::memcmp(out.data(), d.data(), d.size()) == 0,
          "threads: parallel decode data");
  }
}

void test_block_sizes() {
  auto d = make_multiblock(1'000'000);
  auto src = bytes(d);
  for (int blog : {14, 16, 18, 20, 22}) {
    spectra8::CompressParams p;
    p.level = 6;
    p.block_log = blog;
    p.threads = 0;  // auto
    std::vector<std::byte> comp(spectra8::compress_bound(d.size()));
    auto cr = spectra8::compress(src, comp, p);
    check(cr.ok(), "blocksize: compress blog=" + std::to_string(blog));
    comp.resize(cr.value);
    std::vector<std::byte> out(d.size());
    auto dr = spectra8::decompress(comp, out);
    check(dr.ok() && dr.value == d.size() &&
              std::memcmp(out.data(), d.data(), d.size()) == 0,
          "blocksize: roundtrip blog=" + std::to_string(blog));
  }
}

void test_seek_table_and_random_access() {
  auto d = make_multiblock(1'200'000);
  auto src = bytes(d);
  const int blog = 16;  // 64 KiB blocks
  const size_t bs = size_t{1} << blog;

  spectra8::CompressParams p;
  p.level = 6;
  p.block_log = blog;
  p.seek_table = true;
  p.block_checksums = true;
  std::vector<std::byte> comp(spectra8::compress_bound(d.size()));
  auto cr = spectra8::compress(src, comp, p);
  check(cr.ok(), "seek: compress ok");
  comp.resize(cr.value);

  auto table = spectra8::get_seek_table(comp);
  check(table.has_value(), "seek: table present");
  if (table) {
    size_t expected_blocks = (d.size() + bs - 1) / bs;
    check(table->size() == expected_blocks, "seek: block count");
    uint64_t total_regen = 0;
    for (auto& e : *table) total_regen += e.regen_size;
    check(total_regen == d.size(), "seek: total regen size");

    // Random-access decode of each block matches the corresponding slice.
    bool ok = true;
    for (size_t i = 0; i < table->size(); ++i) {
      std::vector<std::byte> blk((*table)[i].regen_size);
      auto br = spectra8::decompress_block(comp, i, blk);
      if (!br || br.value != (*table)[i].regen_size) {
        ok = false;
        break;
      }
      size_t off = i * bs;
      if (std::memcmp(blk.data(), d.data() + off, blk.size()) != 0) {
        ok = false;
        break;
      }
    }
    check(ok, "seek: random-access block decode matches");

    // Out-of-range index reports an error.
    std::vector<std::byte> dummy(bs);
    auto oob = spectra8::decompress_block(comp, table->size() + 5, dummy);
    check(!oob, "seek: out-of-range index errors");
  }

  // Full decode still works with seek table + per-block checksums present.
  std::vector<std::byte> out(d.size());
  auto dr = spectra8::decompress(comp, out);
  check(dr.ok() && std::memcmp(out.data(), d.data(), d.size()) == 0,
        "seek: full decode with trailer");

  // No seek table -> get_seek_table returns nullopt.
  auto plain = compress_ok(d, 6);
  check(!spectra8::get_seek_table(plain).has_value(),
        "seek: absent when not requested");
}

void test_block_checksum_detection() {
  auto d = make_multiblock(600'000);
  auto src = bytes(d);
  spectra8::CompressParams p;
  p.level = 6;
  p.block_log = 16;
  p.block_checksums = true;
  std::vector<std::byte> comp(spectra8::compress_bound(d.size()));
  auto cr = spectra8::compress(src, comp, p);
  comp.resize(cr.value);

  // Corrupt a byte in the middle (a block body) -> detected.
  comp[comp.size() / 2] =
      static_cast<std::byte>(static_cast<uint8_t>(comp[comp.size() / 2]) ^ 0xFF);
  std::vector<std::byte> out(d.size());
  auto dr = spectra8::decompress(comp, out);
  check(dr.status != spectra8::Status::Ok, "block-crc: corruption detected");
}

void test_incompressible_fast_path() {
  std::mt19937_64 rng(99);
  // One default-sized block (256 KiB) so any repetition stays intra-block.
  const size_t n = 256 * 1024;

  // Pure random: must round-trip and stay ~1.0x (taken by the stored fast path).
  std::vector<uint8_t> r(n);
  for (auto& b : r) b = static_cast<uint8_t>(rng());
  check(roundtrips(r, 6), "incompressible: random roundtrips");

  // High order-0 entropy but highly repetitive (first half copied to second).
  // The repeat guard must keep this on the normal path (~2x), not store it.
  std::vector<uint8_t> dup(n);
  for (size_t i = 0; i < n / 2; ++i) dup[i] = static_cast<uint8_t>(rng());
  std::memcpy(dup.data() + n / 2, dup.data(), n / 2);
  auto dc = compress_ok(dup, 6);
  check(dc.size() < (n * 3) / 4,
        "incompressible: duplicated-random still compresses");
  check(roundtrips(dup, 6), "incompressible: duplicated-random roundtrips");
}

// GPU-assisted compression must produce ordinary, decodable frames that round
// trip and actually shrink compressible data. Skipped when Metal is absent.
void test_gpu_path() {
  if (!spectra8::gpu_available()) {
    std::printf("  [gpu] Metal not available; skipping GPU path test\n");
    return;
  }
  std::mt19937_64 rng(7);
  // ~1.5 MiB of semi-repetitive data (several GPU blocks worth).
  const size_t n = 1536 * 1024;
  std::vector<uint8_t> data(n);
  size_t i = 0;
  while (i < n) {
    if ((rng() & 3) == 0) {
      data[i++] = static_cast<uint8_t>(rng());
    } else {
      size_t run = 4 + (rng() % 40);
      size_t back = 1 + (rng() % 4096);
      for (size_t k = 0; k < run && i < n; ++k, ++i)
        data[i] = (i >= back) ? data[i - back] : static_cast<uint8_t>(rng());
    }
  }

  auto src = bytes(data);
  spectra8::CompressParams gp;
  gp.level = 6;
  gp.gpu = true;
  std::vector<std::byte> comp(spectra8::compress_bound(n));
  auto cr = spectra8::compress(src, comp, gp);
  check(cr.ok(), "gpu: compress ok");
  comp.resize(cr.ok() ? cr.value : 0);

  std::vector<std::byte> out(n);
  auto dr = spectra8::decompress(comp, out);
  check(dr.ok() && dr.value == n, "gpu: decompress size");
  check(dr.ok() && std::memcmp(out.data(), data.data(), n) == 0,
        "gpu: roundtrip bytes match");
  check(comp.size() < n * 3 / 4, "gpu: compressible data shrinks");

  // GPU output must be decodable by the standard (CPU) decoder, which it is by
  // construction; also confirm it is interchangeable with a CPU-produced frame.
  check(roundtrips(data, 6), "gpu: same data roundtrips on CPU path too");
}

void test_page_pump_backend(spectra8::GpuBackend backend, const char* tag) {
  if (!spectra8::gpu_transport_available(backend)) {
    std::printf("  [h2d] %s not available; skip\n", tag);
    return;
  }
  auto d = make_multiblock(300000);
  auto src = bytes(d);
  spectra8::CompressParams p;
  p.level = 5;
  p.seek_table = true;
  p.block_log = 16;
  std::vector<std::byte> frame(spectra8::compress_bound(src.size()));
  auto cr = spectra8::compress(src, frame, p);
  check(cr.ok(), std::string(tag) + ": compress");
  if (!cr.ok()) return;
  frame.resize(cr.value);

  auto sizes = spectra8::get_page_sizes(frame);
  check(sizes.ok() && sizes.value.size() >= 2, std::string(tag) + ": pages>=2");

  spectra8::PagePump pump(backend);
  check(pump.ok(), std::string(tag) + ": pump ok");
  check(pump.bind(frame) == spectra8::Status::Ok, std::string(tag) + ": bind");
  check(pump.page_count() == sizes.value.size(), std::string(tag) + ": count");
  check(pump.total_bytes() == src.size(), std::string(tag) + ": total");

  void* dev = pump.transport().alloc_device(src.size());
  check(dev != nullptr, std::string(tag) + ": device alloc");
  if (!dev) return;
  auto pr = pump.push_all(dev);
  check(pr.ok() && pr.value == src.size(), std::string(tag) + ": push_all");
  std::vector<std::byte> back(src.size());
  check(pump.transport().d2h(back.data(), dev, src.size()) == spectra8::Status::Ok,
        std::string(tag) + ": d2h");
  check(std::memcmp(back.data(), src.data(), src.size()) == 0,
        std::string(tag) + ": bytes match");

  std::vector<std::byte> page(static_cast<std::size_t>(sizes.value[1]));
  void* d1 = pump.transport().alloc_device(page.size());
  check(d1 != nullptr, std::string(tag) + ": page device");
  if (d1) {
    auto one = pump.push(1, d1);
    check(one.ok() && one.value == page.size(), std::string(tag) + ": push(1)");
    check(pump.transport().d2h(page.data(), d1, page.size()) ==
              spectra8::Status::Ok,
          std::string(tag) + ": page d2h");
    std::vector<std::byte> cpu(page.size());
    auto blk = spectra8::decompress_block(frame, 1, cpu);
    check(blk.ok() && std::memcmp(page.data(), cpu.data(), page.size()) == 0,
          std::string(tag) + ": page matches CPU block");
    pump.transport().free_device(d1);
  }

  std::vector<std::byte> tiny(8);
  auto conv = spectra8::decompress_to_device(frame, pump.transport(),
                                              tiny.data(), tiny.size());
  check(conv.status == spectra8::Status::BufferTooSmall,
        std::string(tag) + ": dest too small");

  pump.transport().free_device(dev);

  spectra8_page_pump* cp =
      spectra8_page_pump_create(static_cast<spectra8_gpu_backend>(
                                     static_cast<int>(backend)),
                                 1, 3);
  check(cp != nullptr, std::string(tag) + ": c create");
  if (cp) {
    void* cd = spectra8_page_pump_alloc_device(cp, src.size());
    check(cd != nullptr, std::string(tag) + ": c alloc");
    size_t w = 0;
    check(spectra8_page_pump_push_all(cp, frame.data(), frame.size(), cd,
                                       src.size(), &w) == SPECTRA8_OK &&
              w == src.size(),
          std::string(tag) + ": c push_all");
    spectra8_page_pump_free_device(cp, cd);
    spectra8_page_pump_destroy(cp);
  }
}

void test_page_pump() {
  check(spectra8::gpu_transport_available(spectra8::GpuBackend::Host),
        "h2d: host always available");
  check(spectra8::gpu_transport_available(spectra8::GpuBackend::Default),
        "h2d: default available");
  test_page_pump_backend(spectra8::GpuBackend::Host, "h2d-host");
  const auto def = spectra8::default_gpu_backend();
  if (def != spectra8::GpuBackend::Host)
    test_page_pump_backend(def, "h2d-default");
}

// The optimal-parser levels - near-optimal (10) and ultra (11) - use the CPU
// optimal parser. Verify they roundtrip on several data shapes (including edge
// sizes) and that for compressible data they are at least as small as greedy 9.
void test_ultra_level() {
  check(spectra8::kMaxLevel >= spectra8::kUltraLevel, "ultra: level exposed");
  check(spectra8::kNearOptLevel < spectra8::kUltraLevel,
        "near-opt: below ultra");

  // Edge sizes around the small-input fast path inside the parser (both the
  // near-optimal and ultra levels share the optimal-parser machinery).
  for (size_t sz : {0u, 1u, 3u, 63u, 64u, 65u, 1000u, 70000u}) {
    std::mt19937_64 rng(sz + 1);
    std::vector<uint8_t> d(sz);
    for (auto& b : d) b = static_cast<uint8_t>(rng() % 7);  // low-entropy
    check(roundtrips(d, spectra8::kUltraLevel),
          "ultra: roundtrip sz=" + std::to_string(sz));
    check(roundtrips(d, spectra8::kNearOptLevel),
          "near-opt: roundtrip sz=" + std::to_string(sz));
  }

  // Pure-random data must still roundtrip (stored fast path).
  {
    std::mt19937_64 rng(99);
    std::vector<uint8_t> d(200000);
    for (auto& b : d) b = static_cast<uint8_t>(rng());
    check(roundtrips(d, spectra8::kUltraLevel), "ultra: random roundtrip");
  }

  // Mixed literal/match text: ultra should not be worse than greedy level 9.
  {
    std::mt19937_64 rng(11);
    const size_t n = 1u << 19;  // 512 KiB, multiple blocks
    std::vector<uint8_t> d(n);
    size_t i = 0;
    while (i < n) {
      if ((rng() & 7) == 0) {
        d[i++] = static_cast<uint8_t>('a' + (rng() % 26));
      } else {
        size_t run = 3 + (rng() % 60);
        size_t back = 1 + (rng() % 8192);
        for (size_t k = 0; k < run && i < n; ++k, ++i)
          d[i] = (i >= back) ? d[i - back] : static_cast<uint8_t>('a' + (rng() % 26));
      }
    }
    check(roundtrips(d, spectra8::kUltraLevel), "ultra: text roundtrip");
    check(roundtrips(d, spectra8::kNearOptLevel), "near-opt: text roundtrip");
    auto c9 = compress_ok(d, 9);
    auto cn = compress_ok(d, spectra8::kNearOptLevel);
    auto cu = compress_ok(d, spectra8::kUltraLevel);
    // Both optimal levels carry a greedy-9 fallback, so neither loses to 9.
    check(cn.size() <= c9.size(),
          "near-opt: not larger than level 9 (" + std::to_string(cn.size()) +
              " vs " + std::to_string(c9.size()) + ")");
    check(cu.size() <= c9.size(),
          "ultra: not larger than level 9 (" + std::to_string(cu.size()) +
              " vs " + std::to_string(c9.size()) + ")");
  }
}

void test_filters() {
  std::mt19937_64 rng(7);

  // Compress with/without filters, verify the roundtrip, and return the size.
  auto run = [&](const std::vector<uint8_t>& in, bool filt,
                 bool bwt = false) -> size_t {
    spectra8::CompressParams p;
    p.level = 9;
    p.filters = filt;
    p.codec = bwt ? spectra8::Codec::Prism : spectra8::Codec::Beam;
    auto src = bytes(in);
    std::vector<std::byte> comp(spectra8::compress_bound(src.size()));
    auto r = spectra8::compress(src, comp, p);
    check(r.ok(), "filter: compress ok");
    if (!r.ok()) return 0;
    comp.resize(r.value);
    std::vector<std::byte> out(in.size());
    auto d = spectra8::decompress(comp, out);
    check(d.ok() && d.value == in.size(), "filter: decompress size");
    check(in.empty() || std::memcmp(out.data(), in.data(), in.size()) == 0,
          "filter: roundtrip bytes");
    // Also verify a multithreaded decode path (inverse filter on worker threads).
    std::vector<std::byte> out2(in.size());
    spectra8::DecompressParams dp{0};
    auto d2 = spectra8::decompress(comp, out2, dp);
    check(d2.ok() && d2.value == in.size() &&
              (in.empty() ||
               std::memcmp(out2.data(), in.data(), in.size()) == 0),
          "filter: MT roundtrip bytes");
    return r.value;
  };

  // int32 column: smooth ramp + small noise -> shuffle/zigzag should win big.
  {
    std::vector<uint8_t> d(800000);
    uint32_t v = 100000;
    for (size_t i = 0; i + 4 <= d.size(); i += 4) {
      v += static_cast<uint32_t>(static_cast<int>(rng() % 101) - 50);
      std::memcpy(&d[i], &v, 4);
    }
    size_t off = run(d, false);
    size_t on = run(d, true);
    check(on < off, "filter: int32 column compresses better with filters");
  }

  // float64 column: slowly varying values.
  {
    std::vector<uint8_t> d(800000);
    double x = 1.5;
    for (size_t i = 0; i + 8 <= d.size(); i += 8) {
      x += (static_cast<double>(rng() % 2001) - 1000.0) * 1e-5;
      std::memcpy(&d[i], &x, 8);
    }
    size_t off = run(d, false);
    size_t on = run(d, true);
    check(on <= off, "filter: float64 column not worse with filters");
  }

  // text: must roundtrip; size must not regress materially.
  {
    std::vector<uint8_t> d;
    const char* w[] = {"the ", "quick ", "brown ", "fox ", "jumps ", "over "};
    for (int i = 0; i < 40000; ++i) {
      const char* s = w[rng() % 6];
      d.insert(d.end(), reinterpret_cast<const uint8_t*>(s),
               reinterpret_cast<const uint8_t*>(s) + std::strlen(s));
    }
    size_t off = run(d, false);
    size_t on = run(d, true);
    check(on <= off + off / 50, "filter: text not materially worse");
  }

  // random: the detector should decline, so size must not blow up.
  {
    std::vector<uint8_t> d(400000);
    for (auto& b : d) b = static_cast<uint8_t>(rng());
    size_t off = run(d, false);
    size_t on = run(d, true);
    check(on <= off + 64, "filter: random not hurt by filters");
  }

  // BWT + filters: columnar must win big; text/source must not regress vs
  // plain BWT (keep-smaller must compare against raw BWT, not only raw LZ).
  {
    std::vector<uint8_t> d(800000);
    uint32_t v = 100000;
    for (size_t i = 0; i + 4 <= d.size(); i += 4) {
      v += static_cast<uint32_t>(static_cast<int>(rng() % 101) - 50);
      std::memcpy(&d[i], &v, 4);
    }
    size_t off = run(d, false, true);
    size_t on = run(d, true, true);
    check(on < off, "filter+bwt: int32 column smaller than plain BWT");
  }
  {
    std::vector<uint8_t> d;
    const char* w[] = {"the ", "quick ", "brown ", "fox ", "jumps ", "over "};
    for (int i = 0; i < 40000; ++i) {
      const char* s = w[rng() % 6];
      d.insert(d.end(), reinterpret_cast<const uint8_t*>(s),
               reinterpret_cast<const uint8_t*>(s) + std::strlen(s));
    }
    size_t off = run(d, false, true);
    size_t on = run(d, true, true);
    check(on <= off + off / 50, "filter+bwt: text not worse than plain BWT");
  }

  // Strategy B: pre-shuffled (run-heavy) data under Codec::Prism should be allowed
  // to emit LZ blocks when they win — same exploit LZMA gets after +flt.
  {
    constexpr size_t ne = 256 * 1024;
    std::vector<uint8_t> raw(ne * 4);
    for (size_t i = 0; i < ne; ++i) {
      const uint32_t v = static_cast<uint32_t>(i);
      std::memcpy(&raw[i * 4], &v, 4);
    }
    std::vector<uint8_t> shuf(raw.size());
    for (int b = 0; b < 4; ++b)
      for (size_t i = 0; i < ne; ++i) shuf[static_cast<size_t>(b) * ne + i] =
          raw[i * 4 + static_cast<size_t>(b)];
    size_t bwt_sz = run(shuf, false, true);
    size_t lz_sz = run(shuf, false, false);
    check(bwt_sz <= lz_sz + 32,
          "bwt strategy-B: run-heavy pre-shuffle not worse than LZ");
  }
}

// Dictionary-remap filter: direct fwd/inv unit tests (framing, tails, plane
// count, overflow bail, corruption rejection) plus end-to-end coverage of the
// encoder/decoder plumbing (detection, keep-smaller fallback, BWT payloads,
// multithreaded decode, random access).
void test_dict_filter() {
  using spectra8::detail::detect_dict_width;
  using spectra8::detail::dict_filter_fwd;
  using spectra8::detail::dict_filter_inv;
  using spectra8::detail::u8;
  std::mt19937_64 rng(99);

  // Low-cardinality columnar data: n bytes of w-byte elements drawn from a
  // `card`-value pool, plus a verbatim tail when n % w != 0.
  auto make_column = [&](int w, size_t n, size_t card) {
    std::vector<uint64_t> pool(card);
    for (auto& v : pool) {
      v = rng();
      if (w == 4) v &= 0xFFFFFFFFull;
    }
    std::vector<uint8_t> d(n);
    const size_t ne = n / static_cast<size_t>(w);
    for (size_t e = 0; e < ne; ++e)
      std::memcpy(&d[e * static_cast<size_t>(w)], &pool[rng() % card],
                  static_cast<size_t>(w));
    for (size_t i = ne * static_cast<size_t>(w); i < n; ++i)
      d[i] = static_cast<uint8_t>(rng());
    return d;
  };

  // --- direct fwd/inv roundtrips -------------------------------------------
  auto unit = [&](int w, size_t n, size_t card, const std::string& tag) {
    auto d = make_column(w, n, card);
    std::vector<u8> stream;
    bool ok = dict_filter_fwd(d.data(), n, w, stream);
    check(ok, "dict fwd ok (" + tag + ")");
    if (!ok) return;
    check(stream.size() < n, "dict stream smaller (" + tag + ")");
    std::vector<uint8_t> out(n, 0xCC);
    auto st = dict_filter_inv(stream.data(), stream.size(), out.data(), n, w);
    check(st == spectra8::Status::Ok &&
              std::memcmp(out.data(), d.data(), n) == 0,
          "dict inv roundtrip (" + tag + ")");
  };
  unit(2, 65536, 200, "w2");
  unit(2, 65539, 40, "w2 tail");
  unit(4, 65536, 300, "w4 single-plane-ish");
  unit(4, 65539, 300, "w4 tail");
  unit(4, 65536, 200, "w4 one plane (d<=256)");
  unit(8, 131072, 500, "w8");
  unit(8, 131077, 700, "w8 tail");
  unit(4, 64, 2, "w4 small");
  unit(4, 40000, 1, "w4 constant");
  unit(4, 400000, 50000, "w4 near-64K dict");

  // Tiny blocks can never shrink (framing + dictionary exceed the input);
  // the forward pass must decline rather than emit a larger stream.
  {
    auto d = make_column(4, 8, 2);
    std::vector<u8> stream;
    check(!dict_filter_fwd(d.data(), d.size(), 4, stream),
          "dict fwd declines non-shrinking tiny block");
  }

  // --- overflow bail: > 64K distinct elements ------------------------------
  {
    std::vector<uint8_t> d(512 * 1024);
    for (size_t e = 0; e < d.size() / 4; ++e) {
      uint32_t v = static_cast<uint32_t>(e);  // 131072 distinct values
      std::memcpy(&d[e * 4], &v, 4);
    }
    std::vector<u8> stream;
    check(!dict_filter_fwd(d.data(), d.size(), 4, stream),
          "dict fwd bails on >64K distinct");
  }

  // --- corruption rejection (no crash, CorruptInput) ------------------------
  {
    auto d = make_column(4, 4096, 3);
    std::vector<u8> stream;
    check(dict_filter_fwd(d.data(), d.size(), 4, stream), "dict corrupt: fwd");
    std::vector<uint8_t> out(d.size());
    // Every truncation must be rejected.
    bool all_reject = true;
    for (size_t cut = 0; cut < stream.size(); cut += 7)
      if (dict_filter_inv(stream.data(), cut, out.data(), d.size(), 4) ==
          spectra8::Status::Ok)
        all_reject = false;
    check(all_reject, "dict corrupt: truncations rejected");
    // An id past the dictionary (d == 3, plane byte 0xFF) must be rejected.
    auto bad = stream;
    bad[bad.size() - 1] = 0xFF;  // last id plane byte (no tail in this stream)
    check(dict_filter_inv(bad.data(), bad.size(), out.data(), d.size(), 4) ==
              spectra8::Status::CorruptInput,
          "dict corrupt: id out of range rejected");
  }

  // --- detection ------------------------------------------------------------
  {
    auto d16 = make_column(2, 128 * 1024, 200);
    check(detect_dict_width(d16.data(), d16.size()) == 2, "dict detect w2");
    auto d32 = make_column(4, 128 * 1024, 300);
    check(detect_dict_width(d32.data(), d32.size()) == 4, "dict detect w4");
    auto d64 = make_column(8, 128 * 1024, 300);
    check(detect_dict_width(d64.data(), d64.size()) == 8, "dict detect w8");
    std::vector<uint8_t> rnd(128 * 1024);
    for (auto& b : rnd) b = static_cast<uint8_t>(rng());
    check(detect_dict_width(rnd.data(), rnd.size()) == 0,
          "dict detect declines random");
  }

  // --- end-to-end through the public API ------------------------------------
  auto run = [&](const std::vector<uint8_t>& in, bool filt, bool bwt,
                 const std::string& tag) -> size_t {
    spectra8::CompressParams p;
    p.level = 9;
    p.filters = filt;
    p.codec = bwt ? spectra8::Codec::Prism : spectra8::Codec::Beam;
    auto src = bytes(in);
    std::vector<std::byte> comp(spectra8::compress_bound(src.size()));
    auto r = spectra8::compress(src, comp, p);
    check(r.ok(), "dict api: compress ok (" + tag + ")");
    if (!r.ok()) return 0;
    comp.resize(r.value);
    std::vector<std::byte> out(in.size());
    auto d = spectra8::decompress(comp, out);
    check(d.ok() && d.value == in.size() &&
              (in.empty() ||
               std::memcmp(out.data(), in.data(), in.size()) == 0),
          "dict api: roundtrip (" + tag + ")");
    std::vector<std::byte> out2(in.size());
    spectra8::DecompressParams dp{0};
    auto d2 = spectra8::decompress(comp, out2, dp);
    check(d2.ok() && d2.value == in.size() &&
              (in.empty() ||
               std::memcmp(out2.data(), in.data(), in.size()) == 0),
          "dict api: MT roundtrip (" + tag + ")");
    return r.value;
  };

  // Low-cardinality i32 column with no byte-adjacency structure: no other
  // filter helps here, so a clear (>= 15%) win over the unfiltered encode
  // proves the dict path actually engaged. (The unfiltered Beam block is
  // itself coded by the Crystal lattice coder on such columns now, which
  // is why the bar is 15% and not the 25% of the Huffman-literal days.)
  {
    auto d = make_column(4, 400000, 300);
    size_t off = run(d, false, false, "i32 low-card off");
    size_t on = run(d, true, false, "i32 low-card on");
    check(on * 100 < off * 85, "dict api: i32 low-card clearly smaller");
    size_t on_bwt = run(d, true, true, "i32 low-card bwt");
    check(on_bwt * 100 < off * 85, "dict api: bwt+filters also wins");
  }
  // f64-style column (8-byte elements from a small pool), tail not a
  // multiple of the width.
  {
    auto d = make_column(8, 400005, 500);
    size_t off = run(d, false, false, "f64 low-card off");
    size_t on = run(d, true, false, "f64 low-card on");
    check(on * 100 < off * 85, "dict api: f64 column clearly smaller");
  }
  // Tiny input.
  {
    auto d = make_column(4, 100, 5);
    (void)run(d, true, false, "tiny");
  }
  // Adversarial high-cardinality input: detection may nominate, the forward
  // pass bails, output must not blow up vs filters-off.
  {
    std::vector<uint8_t> d(512 * 1024);
    for (size_t e = 0; e < d.size() / 4; ++e) {
      uint32_t v = static_cast<uint32_t>(rng());
      std::memcpy(&d[e * 4], &v, 4);
    }
    size_t off = run(d, false, false, "high-card off");
    size_t on = run(d, true, false, "high-card on");
    check(on <= off + 64, "dict api: high-cardinality not enlarged");
  }
  // Random access into a dict-filtered multi-block frame (seek table).
  {
    auto d = make_column(4, 100000, 300);
    spectra8::CompressParams p;
    p.level = 5;
    p.block_log = 14;
    p.filters = true;
    p.seek_table = true;
    auto src = bytes(d);
    std::vector<std::byte> comp(spectra8::compress_bound(src.size()));
    auto r = spectra8::compress(src, comp, p);
    check(r.ok(), "dict api: seek compress ok");
    comp.resize(r.ok() ? r.value : 0);
    std::vector<std::byte> blk(16384);
    auto b = spectra8::decompress_block(comp, 1, blk);
    check(b.ok() && b.value == 16384 &&
              std::memcmp(blk.data(), src.data() + 16384, 16384) == 0,
          "dict api: random-access block decode");
  }
}

// Frame-of-reference bit packing and decimal integerization: direct roundtrip,
// outlier patching, and the framed detector on columns the dictionary misses.
void test_pack_filters() {
  using spectra8::detail::Filter;
  using spectra8::detail::alp_filter_fwd;
  using spectra8::detail::alp_filter_inv;
  using spectra8::detail::detect_framed_filter;
  using spectra8::detail::for_filter_fwd;
  using spectra8::detail::for_filter_inv;
  using spectra8::detail::u8;

  auto round = [&](auto fwd, auto inv, int w, const std::vector<uint8_t>& d,
                   const std::string& tag) {
    std::vector<u8> stream;
    check(fwd(d.data(), d.size(), w, stream), "pack fwd " + tag);
    if (stream.empty()) return;
    check(stream.size() < d.size(), "pack shrinks " + tag);
    std::vector<uint8_t> out(d.size(), 0xA5);
    check(inv(stream.data(), stream.size(), out.data(), d.size(), w) ==
                  spectra8::Status::Ok &&
              std::memcmp(out.data(), d.data(), d.size()) == 0,
          "pack roundtrip " + tag);
    bool all_reject = true;
    for (size_t cut = 0; cut < stream.size(); cut += 11)
      if (inv(stream.data(), cut, out.data(), d.size(), w) == spectra8::Status::Ok)
        all_reject = false;
    check(all_reject, "pack truncations rejected " + tag);
  };

  {
    std::vector<uint8_t> d(128 * 1024);
    const size_t ne = d.size() / 4;
    for (size_t e = 0; e < ne; ++e) {
      int32_t v = 100000 + static_cast<int32_t>(e);
      std::memcpy(&d[e * 4], &v, 4);
    }
    round(for_filter_fwd, for_filter_inv, 4, d, "for32 narrow unique");
    check(detect_framed_filter(d.data(), d.size()) == Filter::For32,
          "detect: unique narrow i32 -> for32");
  }
  {
    std::vector<uint8_t> d(64 * 1024 + 3);
    const size_t ne = d.size() / 4;
    for (size_t e = 0; e < ne; ++e) {
      int32_t v = -40000 + static_cast<int32_t>(e % 1000);
      if (e % 128 == 0) v = 2000000000;
      std::memcpy(&d[e * 4], &v, 4);
    }
    d[d.size() - 1] = 0x5A;
    round(for_filter_fwd, for_filter_inv, 4, d, "for32 outliers+tail");
  }
  {
    std::vector<uint8_t> d(8 * 1024);
    const size_t ne = d.size() / 2;
    for (size_t e = 0; e < ne; ++e) {
      int16_t v = static_cast<int16_t>(1000 + static_cast<int>(e));
      std::memcpy(&d[e * 2], &v, 2);
    }
    round(for_filter_fwd, for_filter_inv, 2, d, "for16");
    check(detect_framed_filter(d.data(), d.size()) == Filter::For16,
          "detect: unique narrow i16 -> for16");
  }
  {
    std::vector<uint8_t> d(128 * 1024);
    const size_t ne = d.size() / 8;
    for (size_t e = 0; e < ne; ++e) {
      const double v = static_cast<double>(e) * 0.25;
      std::memcpy(&d[e * 8], &v, 8);
    }
    round(alp_filter_fwd, alp_filter_inv, 8, d, "alp64 quarter");
    check(detect_framed_filter(d.data(), d.size()) == Filter::Alp64,
          "detect: dyadic f64 -> alp64");
  }
  {
    std::vector<uint8_t> d(64 * 1024 + 1);
    const size_t ne = d.size() / 4;
    for (size_t e = 0; e < ne; ++e) {
      const float v = (static_cast<float>(e) + 100000.f) * 0.5f;
      std::memcpy(&d[e * 4], &v, 4);
    }
    round(alp_filter_fwd, alp_filter_inv, 4, d, "alp32 half");
    check(detect_framed_filter(d.data(), d.size()) == Filter::Alp32,
          "detect: dyadic f32 -> alp32");
  }
  {
    std::vector<uint8_t> rnd(128 * 1024);
    std::mt19937_64 rng(1234);
    for (auto& b : rnd) b = static_cast<uint8_t>(rng());
    check(detect_framed_filter(rnd.data(), rnd.size()) == Filter::None,
          "detect: random -> no framed filter");
    std::vector<u8> stream;
    check(!for_filter_fwd(rnd.data(), rnd.size(), 4, stream),
          "for32 declines random");
    check(!alp_filter_fwd(rnd.data(), rnd.size(), 8, stream),
          "alp64 declines random");
  }
  auto api = [&](const std::vector<uint8_t>& in, const std::string& tag) {
    spectra8::CompressParams p;
    p.level = 5;
    p.filters = true;
    auto src = bytes(in);
    std::vector<std::byte> comp(spectra8::compress_bound(src.size()));
    auto r = spectra8::compress(src, comp, p);
    check(r.ok(), "pack api compress " + tag);
    if (!r.ok()) return;
    comp.resize(r.value);
    std::vector<std::byte> out(in.size());
    auto d = spectra8::decompress(comp, out);
    check(d.ok() && d.value == in.size() &&
              std::memcmp(out.data(), in.data(), in.size()) == 0,
          "pack api roundtrip " + tag);
  };
  {
    std::vector<uint8_t> d(96 * 1024);
    for (size_t e = 0; e < d.size() / 8; ++e) {
      const double v = static_cast<double>(e) * 0.25;
      std::memcpy(&d[e * 8], &v, 8);
    }
    api(d, "alp64");
  }
  {
    std::vector<uint8_t> d(96 * 1024);
    for (size_t e = 0; e < d.size() / 4; ++e) {
      int32_t v = 250000 + static_cast<int32_t>(e);
      if (e % 200 == 0) v = -2000000000;
      std::memcpy(&d[e * 4], &v, 4);
    }
    api(d, "for32");
  }
}

// Exhaustive check of the shuffle / bitshuffle kernels against a scalar
// reference: the SIMD fast paths (16-element transpose blocks) and the
// bit_transpose8 bitshuffle path must produce the exact on-disk layout of the
// original scalar loops for every width and for awkward sizes (SIMD remainder,
// non-multiple-of-8 element counts, sub-element tail bytes).
void test_transform_kernels() {
  using spectra8::detail::u8;
  using spectra8::detail::usize;
  std::mt19937_64 rng(101);

  auto ref_shuffle_fwd = [](const u8* in, usize n, u8* out, int w) {
    const usize cnt = n / static_cast<usize>(w);
    for (int p = 0; p < w; ++p)
      for (usize e = 0; e < cnt; ++e)
        out[static_cast<usize>(p) * cnt + e] = in[e * static_cast<usize>(w) + p];
    for (usize i = cnt * static_cast<usize>(w); i < n; ++i) out[i] = in[i];
  };
  auto ref_bitshuffle_fwd = [](const u8* in, usize n, u8* out, int w) {
    const usize cnt = n / static_cast<usize>(w);
    const int bits = w * 8;
    const usize region = cnt * static_cast<usize>(w);
    std::memset(out, 0, region);
    for (int b = 0; b < bits; ++b)
      for (usize e = 0; e < cnt; ++e) {
        const auto v = spectra8::detail::load_le(
            in + e * static_cast<usize>(w), w);
        if ((v >> b) & 1) {
          const usize pos = static_cast<usize>(b) * cnt + e;
          out[pos >> 3] |= static_cast<u8>(1u << (pos & 7));
        }
      }
    for (usize i = region; i < n; ++i) out[i] = in[i];
  };

  const size_t sizes[] = {0,   1,    7,    16,   64,   127,  128,  255, 256,
                          257, 1024, 4093, 4096, 8191, 65536, 100000};
  for (size_t n : sizes) {
    std::vector<u8> in(n), a(n), b(n), back(n);
    for (auto& x : in) x = static_cast<u8>(rng());
    for (int w : {2, 4, 8}) {
      const std::string tag = " (n=" + std::to_string(n) +
                              " w=" + std::to_string(w) + ")";
      ref_shuffle_fwd(in.data(), n, a.data(), w);
      spectra8::detail::shuffle_fwd(in.data(), n, b.data(), w);
      check(a == b, "shuffle_fwd matches scalar reference" + tag);
      spectra8::detail::shuffle_inv(b.data(), n, back.data(), w);
      check(back == in, "shuffle roundtrip" + tag);

      ref_bitshuffle_fwd(in.data(), n, a.data(), w);
      spectra8::detail::bitshuffle_fwd(in.data(), n, b.data(), w);
      check(a == b, "bitshuffle_fwd matches scalar reference" + tag);
      spectra8::detail::bitshuffle_inv(b.data(), n, back.data(), w);
      check(back == in, "bitshuffle roundtrip" + tag);
    }
  }
}

// New composed filters (ids 15-17): delta_shuffle64 for monotonic 64-bit
// offset/index columns, zigzag_shuffle32/64 for signed columns oscillating
// around zero (zig-zag of the *raw* values, no delta). Direct apply/invert
// roundtrips on target shapes and awkward tails, detector behaviour, and
// end-to-end compression.
void test_new_composed_filters() {
  using spectra8::detail::Filter;
  using spectra8::detail::u8;
  std::mt19937_64 rng(202);

  auto roundtrip = [&](Filter f, const std::vector<u8>& in,
                       const std::string& tag) {
    std::vector<u8> fwd(in.size()), back(in.size());
    spectra8::detail::apply_filter(f, in.data(), in.size(), fwd.data());
    spectra8::detail::invert_filter(f, fwd.data(), in.size(), back.data());
    check(back == in, "new filter roundtrip (" + tag + ")");
  };

  auto make_offsets_u64 = [&](size_t n) {  // monotonic, small random deltas
    std::vector<u8> d(n);
    uint64_t v = 0x12345678ULL;
    for (size_t i = 0; i + 8 <= n; i += 8) {
      v += 20 + (rng() % 40);
      std::memcpy(&d[i], &v, 8);
    }
    for (size_t i = (n / 8) * 8; i < n; ++i) d[i] = static_cast<u8>(rng());
    return d;
  };
  auto make_oscillating = [&](size_t n, int w, int64_t amp) {
    std::vector<u8> d(n);
    for (size_t i = 0; i + static_cast<size_t>(w) <= n;
         i += static_cast<size_t>(w)) {
      const int64_t s =
          static_cast<int64_t>(rng() % static_cast<uint64_t>(2 * amp + 1)) -
          amp;
      std::memcpy(&d[i], &s, static_cast<size_t>(w));  // little-endian host
    }
    return d;
  };

  const Filter fs[] = {Filter::DeltaShuffle64, Filter::ZigzagShuffle32,
                       Filter::ZigzagShuffle64, Filter::XorShuffle32,
                       Filter::XorShuffle64,   Filter::Delta2Shuffle32,
                       Filter::ZigzagDelta2Shuffle32, Filter::Delta2Shuffle64,
                       Filter::BcjX86,         Filter::Lag2Shuffle32,
                       Filter::Lag2Shuffle64,  Filter::XorLag2Shuffle32,
                       Filter::BcjArm64};
  const char* fname[] = {"dshuf64", "zzshuf32", "zzshuf64", "xorshuf32",
                         "xorshuf64", "d2shuf32", "zzd2shuf32", "d2shuf64",
                         "bcj_x86", "lag2shuf32", "lag2shuf64", "xorlag2shuf32",
                         "bcj_arm64"};
  static_assert(sizeof(fs) / sizeof(fs[0]) == sizeof(fname) / sizeof(fname[0]));
  // Random data at awkward lengths: trailing bytes that don't fill a whole
  // element must pass through verbatim, like every other size-preserving
  // transform.
  const size_t sizes[] = {0, 1, 7, 8, 9, 15, 16, 63,
                          64, 65, 4093, 4096, 100003};
  for (int fi = 0; fi < static_cast<int>(sizeof(fs) / sizeof(fs[0])); ++fi) {
    for (size_t n : sizes) {
      std::vector<u8> d(n);
      for (auto& x : d) x = static_cast<u8>(rng());
      roundtrip(fs[fi], d,
                std::string(fname[fi]) + " random n=" + std::to_string(n));
    }
    roundtrip(fs[fi], make_offsets_u64(65543),
              std::string(fname[fi]) + " u64 offsets");
    roundtrip(fs[fi], make_oscillating(65539, 4, 1000),
              std::string(fname[fi]) + " i32 oscillating");
    roundtrip(fs[fi], make_oscillating(65541, 8, 1000000),
              std::string(fname[fi]) + " i64 oscillating");
  }

  // Zig-zag mapping sanity: {0,-1,1,-2,2} -> {0,1,2,3,4}, then transposed so
  // byte plane 0 holds the small codes and the high planes are all zero.
  {
    const int32_t vals[] = {0, -1, 1, -2, 2};
    std::vector<u8> d(sizeof(vals));
    std::memcpy(d.data(), vals, sizeof(vals));
    std::vector<u8> fwd(d.size());
    spectra8::detail::apply_filter(Filter::ZigzagShuffle32, d.data(),
                                    d.size(), fwd.data());
    const u8 expect[20] = {0, 1, 2, 3, 4};  // plane 0, planes 1-3 zero
    check(std::memcmp(fwd.data(), expect, 20) == 0,
          "zzshuf32 zig-zag mapping matches spec");
  }
  // Second-order delta sanity: linear ramp {10, 20, 30, 40, 50} produces
  // residuals {10, 0, 0, 0, 0}, so all higher planes are zero.
  {
    const uint32_t ramp[] = {10, 20, 30, 40, 50};
    std::vector<u8> d(sizeof(ramp));
    std::memcpy(d.data(), ramp, sizeof(ramp));
    std::vector<u8> fwd(d.size());
    spectra8::detail::apply_filter(Filter::Delta2Shuffle32, d.data(),
                                    d.size(), fwd.data());
    const u8 expect[20] = {10, 0, 0, 0, 0};  // plane 0 has {10,0,0,0,0}, planes 1-3 all zero
    check(std::memcmp(fwd.data(), expect, 20) == 0,
          "d2shuf32 second-order delta cancels linear slope");
  }
  // Lag-2 sanity: an interleaved pair of constants {7, 9, 7, 9, 7, 9} gives
  // residuals {7, 9, 0, 0, 0, 0} (x[i] - x[i-2]); the XOR variant the same here.
  {
    const uint32_t pairs[] = {7, 9, 7, 9, 7, 9};
    std::vector<u8> d(sizeof(pairs));
    std::memcpy(d.data(), pairs, sizeof(pairs));
    std::vector<u8> fwd(d.size());
    const u8 expect[24] = {7, 9, 0, 0, 0, 0};
    spectra8::detail::apply_filter(Filter::Lag2Shuffle32, d.data(), d.size(), fwd.data());
    check(std::memcmp(fwd.data(), expect, 24) == 0, "lag2shuf32 cancels a period-2 pattern");
    spectra8::detail::apply_filter(Filter::XorLag2Shuffle32, d.data(), d.size(), fwd.data());
    check(std::memcmp(fwd.data(), expect, 24) == 0, "xorlag2shuf32 cancels a period-2 pattern");
    const uint64_t pairs64[] = {7, 9, 7, 9};
    std::vector<u8> d64(sizeof(pairs64));
    std::memcpy(d64.data(), pairs64, sizeof(pairs64));
    std::vector<u8> fwd64(d64.size());
    const u8 expect64[32] = {7, 9, 0, 0};
    spectra8::detail::apply_filter(Filter::Lag2Shuffle64, d64.data(), d64.size(), fwd64.data());
    check(std::memcmp(fwd64.data(), expect64, 32) == 0, "lag2shuf64 cancels a period-2 pattern");
  }
  // BCJ x86 branch target normalizer sanity:
  // CALL instruction (0xE8) at offset 10 with relative target 0x100.
  // Absolute address written = 0x100 + (10 + 5) = 0x10F.
  {
    std::vector<u8> code(32, 0x90);
    code[10] = 0xE8;
    code[11] = 0x00; code[12] = 0x01; code[13] = 0x00; code[14] = 0x00;
    std::vector<u8> fwd(code.size());
    spectra8::detail::apply_filter(Filter::BcjX86, code.data(), code.size(), fwd.data());
    uint32_t target = 0;
    std::memcpy(&target, fwd.data() + 11, 4);
    check(target == 0x100 + 15, "bcj_x86 forward target displacement normalizes to absolute");
    std::vector<u8> inv(code.size());
    spectra8::detail::invert_filter(Filter::BcjX86, fwd.data(), fwd.size(), inv.data());
    check(inv == code, "bcj_x86 roundtrip restores relative displacement");
    // An 0xE8 inside data (displacement top byte neither 0x00 nor 0xFF) is
    // left alone, so two copies of the same data at different offsets stay
    // identical after the transform.
    std::vector<u8> data(64, 0x11);
    data[3] = 0xE8; data[4] = 0x12; data[5] = 0x34; data[6] = 0x56; data[7] = 0x78;
    std::copy(data.begin(), data.begin() + 32, data.begin() + 32);
    std::vector<u8> dfwd(data.size());
    spectra8::detail::apply_filter(Filter::BcjX86, data.data(), data.size(), dfwd.data());
    check(dfwd == data, "bcj_x86 leaves out-of-range displacements alone");
  }
  // BCJ ARM64: BL imm26 and ADRP imm21 become absolute, everything else is
  // untouched; the inverse restores the words bit-exactly.
  {
    auto put = [](std::vector<u8>& v, size_t off, uint32_t w) { std::memcpy(v.data() + off, &w, 4); };
    auto get = [](const std::vector<u8>& v, size_t off) { uint32_t w; std::memcpy(&w, v.data() + off, 4); return w; };
    std::vector<u8> code(64, 0);
    for (size_t i = 0; i < 64; i += 4) put(code, i, 0xD503201Fu);  // NOP
    put(code, 8, 0x94000000u | 0x100u);           // BL +0x400 bytes (imm26 = 0x100 words)
    put(code, 20, 0x90000000u | (1u << 29) | (5u << 5) | 3u);  // ADRP x3, page +(5*4+1)
    put(code, 40, 0x94000000u | 0x100u);          // same target form, different pc
    put(code, 48, 0xF9400000u);                   // LDR x0,[x0] (no immediate rewrite)
    std::vector<u8> fwd(64), inv(64);
    spectra8::detail::apply_filter(Filter::BcjArm64, code.data(), code.size(), fwd.data());
    check((get(fwd, 8) & 0x03FFFFFFu) == 0x100u + 2u, "bcj_arm64 BL imm26 becomes absolute (+pc/4)");
    check((get(fwd, 40) & 0x03FFFFFFu) == 0x100u + 10u, "bcj_arm64 BL at another pc shifts by its own pc");
    check((get(fwd, 20) & 0x9F00001Fu) == (0x90000000u | 3u), "bcj_arm64 ADRP keeps opcode and Rd");
    check(get(fwd, 48) == 0xF9400000u && get(fwd, 0) == 0xD503201Fu, "bcj_arm64 leaves other words alone");
    spectra8::detail::invert_filter(Filter::BcjArm64, fwd.data(), fwd.size(), inv.data());
    check(inv == code, "bcj_arm64 roundtrip is exact");
    // ADRP whose page immediate is outside the signed 18-bit range is skipped
    // on both sides (xz rule), so the roundtrip stays exact across a block
    // where pc pushes the sum over the edge.
    std::vector<u8> big(1 << 16, 0);
    for (size_t i = 0; i < big.size(); i += 4) put(big, i, 0x90000000u | (0x1FFFCu << 3) | 7u);  // imm ~ +2^17-ish pages
    std::vector<u8> bf(big.size()), bi(big.size());
    spectra8::detail::apply_filter(Filter::BcjArm64, big.data(), big.size(), bf.data());
    spectra8::detail::invert_filter(Filter::BcjArm64, bf.data(), bf.size(), bi.data());
    check(bi == big, "bcj_arm64 roundtrip exact at the ADRP range edge");
  }
  // Executable detection: file headers, instruction streams, and the
  // negatives the byte filters are for.
  {
    using spectra8::detail::ExecArch;
    std::vector<u8> elf(4096, 0);
    elf[0] = 0x7F; elf[1] = 'E'; elf[2] = 'L'; elf[3] = 'F'; elf[4] = 2; elf[5] = 1;
    elf[18] = 0x3E;
    check(spectra8::detail::detect_exec_arch(elf.data(), elf.size()) == ExecArch::X86,
          "detect_exec: ELF e_machine x86-64");
    elf[18] = 0xB7;
    check(spectra8::detail::detect_exec_arch(elf.data(), elf.size()) == ExecArch::Arm64,
          "detect_exec: ELF e_machine AArch64");
    std::vector<u8> macho(4096, 0);
    const uint32_t mh = 0xFEEDFACFu, cpu_arm64 = 0x0100000Cu, cpu_x86 = 0x01000007u;
    std::memcpy(macho.data(), &mh, 4);
    std::memcpy(macho.data() + 4, &cpu_arm64, 4);
    check(spectra8::detail::detect_exec_arch(macho.data(), macho.size()) == ExecArch::Arm64,
          "detect_exec: Mach-O arm64");
    std::memcpy(macho.data() + 4, &cpu_x86, 4);
    check(spectra8::detail::detect_exec_arch(macho.data(), macho.size()) == ExecArch::X86,
          "detect_exec: Mach-O x86_64");
    // Headerless x86-64 code: REX-prefixed instructions with calls to a few
    // nearby targets (the shape of a .text page).
    std::vector<u8> x86(1 << 16);
    {
      static const u8 ops[] = {0x48, 0x89, 0xE5, 0x48, 0x83, 0xEC, 0x20, 0x48, 0x8B, 0x45,
                               0xF8, 0x41, 0x56, 0x4C, 0x8D, 0x3D, 0x10, 0x00, 0x00, 0x00};
      size_t i = 0;
      while (i + 5 <= x86.size()) {
        if (rng() % 4 == 0) {
          x86[i] = 0xE8;
          const uint32_t rel = static_cast<uint32_t>(rng() % 0x8000) - 0x4000u;
          std::memcpy(x86.data() + i + 1, &rel, 4);
          i += 5;
        } else {
          const size_t k = rng() % sizeof(ops);
          const size_t len = 1 + rng() % 3;
          for (size_t j = 0; j < len && i < x86.size(); ++j) x86[i++] = ops[(k + j) % sizeof(ops)];
        }
      }
      while (i < x86.size()) x86[i++] = 0x90;
    }
    check(spectra8::detail::detect_exec_arch(x86.data(), x86.size()) == ExecArch::X86,
          "detect_exec: headerless x86-64 instruction stream");
    check(spectra8::detail::detect_filter(x86.data(), x86.size()) == Filter::BcjX86,
          "detect: x86 code -> bcj_x86");
    // Headerless AArch64: STP/ADRP/BL/LDP/RET function bodies.
    std::vector<u8> a64(1 << 16);
    {
      size_t i = 0;
      auto put = [&](uint32_t w) { std::memcpy(a64.data() + i, &w, 4); i += 4; };
      while (i + 32 <= a64.size()) {
        put(0xA9BF7BFDu);                                    // stp x29,x30,[sp,#-16]!
        put(0x90000000u | ((rng() % 64) << 5) | (rng() % 31));  // adrp
        put(0x91000000u | static_cast<uint32_t>(rng() % 4096) << 10);  // add imm
        put(0x94000000u | (static_cast<uint32_t>(rng() % 0x2000) & 0x03FFFFFFu));  // bl near
        put(0xAA0003E0u | (rng() % 31));                     // mov
        put(0xA8C17BFDu);                                    // ldp x29,x30,[sp],#16
        put(0xD65F03C0u);                                    // ret
        put(0xD503201Fu);                                    // nop
      }
      while (i < a64.size()) a64[i++] = 0;
    }
    check(spectra8::detail::detect_exec_arch(a64.data(), a64.size()) == ExecArch::Arm64,
          "detect_exec: headerless AArch64 instruction stream");
    check(spectra8::detail::detect_filter(a64.data(), a64.size()) == Filter::BcjArm64,
          "detect: AArch64 code -> bcj_arm64");
    // Negatives: random bytes, text, a u64 column of 232 (bytes E8 00 00 ..).
    std::vector<u8> rnd(1 << 16);
    for (auto& x : rnd) x = static_cast<u8>(rng());
    check(spectra8::detail::detect_exec_arch(rnd.data(), rnd.size()) == ExecArch::None,
          "detect_exec: random -> none");
    std::vector<u8> txt(1 << 16);
    for (size_t i = 0; i < txt.size(); ++i) txt[i] = static_cast<u8>("the quick brown fox jumps over the lazy dog\n"[i % 44]);
    check(spectra8::detail::detect_exec_arch(txt.data(), txt.size()) == ExecArch::None,
          "detect_exec: text -> none");
    std::vector<u8> col(1 << 16, 0);
    for (size_t i = 0; i < col.size(); i += 8) col[i] = 0xE8;
    check(spectra8::detail::detect_exec_arch(col.data(), col.size()) == ExecArch::None,
          "detect_exec: u64 column of 232 -> none");
  }

  // Detector: the new filters must actually be picked on their target
  // workloads, and random data must still yield None.
  {
    auto d = make_offsets_u64(1 << 16);
    check(spectra8::detail::detect_filter(d.data(), d.size()) ==
              Filter::DeltaShuffle64,
          "detect: monotonic u64 offsets -> delta_shuffle64");
  }
  {
    // Two interleaved noisy u64 ramps far apart (x/y pairs): lag-1
    // residuals alternate between +big and -big, lag-2 residuals are small.
    // (Interleaved u32 pairs would be one u64 ramp, which delta64 covers.)
    const size_t n = 1 << 16;
    std::vector<u8> d(n);
    uint64_t a = 1000000ull, b = 3000000000000ull;
    for (size_t i = 0; i + 16 <= n; i += 16) {
      a += 30 + (rng() % 20);
      b += 30 + (rng() % 20);
      std::memcpy(&d[i], &a, 8);
      std::memcpy(&d[i + 8], &b, 8);
    }
    check(spectra8::detail::detect_filter(d.data(), d.size()) ==
              Filter::Lag2Shuffle64,
          "detect: interleaved u64 ramps -> lag2_shuffle64");
  }
  {
    auto d = make_oscillating(1 << 16, 4, 200000);
    check(spectra8::detail::detect_filter(d.data(), d.size()) ==
              Filter::ZigzagShuffle32,
          "detect: oscillating i32 -> zigzag_shuffle32");
  }
  {
    auto d = make_oscillating(1 << 16, 8, 200000);
    check(spectra8::detail::detect_filter(d.data(), d.size()) ==
              Filter::ZigzagShuffle64,
          "detect: oscillating i64 -> zigzag_shuffle64");
  }
  {
    // Quadratic ramp: v = 50 + 2*i + (i*(i-1))/2. Second difference is exactly 1 (constant positive).
    std::vector<u8> d(1 << 16);
    for (size_t i = 0; i + 4 <= d.size(); i += 4) {
      uint32_t idx = static_cast<uint32_t>(i / 4);
      uint32_t v = 50u + 2u * idx + (idx * (idx - 1u)) / 2u;
      std::memcpy(d.data() + i, &v, 4);
    }
    auto f_actual = spectra8::detail::detect_filter(d.data(), d.size());
    check(f_actual == Filter::Delta2Shuffle32 || f_actual == Filter::ZigzagDelta2Shuffle32,
          "detect: quadratic ramp -> delta2_shuffle32");
  }
  {
    std::vector<u8> d(1 << 16);
    for (auto& x : d) x = static_cast<u8>(rng());
    check(spectra8::detail::detect_filter(d.data(), d.size()) ==
              Filter::None,
          "detect: random still yields None");
  }

  // bitshuffle64 and delta32+bitshuffle32: roundtrip, and the detector picks
  // them on a binomial float64 column and a stepped i32 counter.
  {
    const Filter nfs[] = {Filter::BitShuffle64, Filter::DeltaBitShuffle32};
    const char* nname[] = {"bitsh64", "d32+bit"};
    for (int fi = 0; fi < 2; ++fi) {
      for (size_t n : {size_t{0}, size_t{7}, size_t{8}, size_t{15}, size_t{64},
                       size_t{100003}}) {
        std::vector<u8> d(n);
        for (auto& x : d) x = static_cast<u8>(rng());
        roundtrip(nfs[fi], d, std::string(nname[fi]) + " n=" + std::to_string(n));
      }
    }
    std::vector<u8> col(1 << 16);
    {
      std::mt19937_64 r(0xABCDEF ^ 7);
      std::binomial_distribution<long long> dist(4096, 0.5);
      for (size_t i = 0; i + 8 <= col.size(); i += 8) {
        const double v = static_cast<double>(dist(r)) * (100.0 / 4096.0);
        std::memcpy(col.data() + i, &v, 8);
      }
    }
    check(spectra8::detail::detect_filter(col.data(), col.size()) ==
              Filter::BitShuffle64,
          "detect: binomial f64 -> bitshuffle64");
    std::vector<u8> ctr(1 << 16);
    uint32_t c = 0;
    std::mt19937_64 step(1);
    for (size_t i = 0; i + 4 <= ctr.size(); i += 4) {
      c += 1 + static_cast<uint32_t>(step() % 3);
      std::memcpy(ctr.data() + i, &c, 4);
    }
    check(spectra8::detail::detect_filter(ctr.data(), ctr.size()) ==
              Filter::DeltaBitShuffle32,
          "detect: stepped i32 counter -> delta32+bitshuffle32");
  }

  // Full-sample detect: middle 16 KiB is noise (default sample → None) while
  // the rest is a monotonic u32 ramp (full sample → a delta/shuffle filter).
  {
    constexpr size_t n = 1u << 16;  // 64 KiB
    std::vector<u8> d(n);
    // Mid-sample window is [24 KiB, 40 KiB) for n=64K, sn=16K.
    for (size_t i = 0; i < n; i += 4) {
      std::uint32_t v;
      if (i >= (24u << 10) && i < (40u << 10))
        v = static_cast<std::uint32_t>(rng());
      else
        v = static_cast<std::uint32_t>(i / 4);
      std::memcpy(d.data() + i, &v, 4);
    }
    check(spectra8::detail::detect_filter(d.data(), d.size(), false) ==
              Filter::None,
          "detect: mid-sample noise -> None");
    check(spectra8::detail::detect_filter(d.data(), d.size(), true) !=
              Filter::None,
          "detect: full-sample sees ramp -> some filter");

    spectra8::CompressParams p;
    p.level = 5;
    p.filters = true;
    p.filter_sample_full = true;
    auto src = bytes(d);
    std::vector<std::byte> comp(spectra8::compress_bound(src.size()));
    auto r = spectra8::compress(src, comp, p);
    check(r.ok(), "detect: filter_sample_full compress ok");
    if (r.ok()) {
      comp.resize(r.value);
      std::vector<std::byte> out(d.size());
      auto dec = spectra8::decompress(comp, out);
      check(dec.ok() && dec.value == d.size() &&
                std::memcmp(out.data(), d.data(), d.size()) == 0,
            "detect: filter_sample_full roundtrip");
    }
  }

  // End-to-end: filters-on must roundtrip bit-exactly and clearly beat
  // filters-off on the target columns.
  auto run = [&](const std::vector<u8>& in, bool filt,
                 const std::string& tag) -> size_t {
    spectra8::CompressParams p;
    p.level = 9;
    p.filters = filt;
    auto src = bytes(in);
    std::vector<std::byte> comp(spectra8::compress_bound(src.size()));
    auto r = spectra8::compress(src, comp, p);
    check(r.ok(), "new filter e2e: compress ok (" + tag + ")");
    if (!r.ok()) return 0;
    comp.resize(r.value);
    std::vector<std::byte> out(in.size());
    auto d = spectra8::decompress(comp, out);
    check(d.ok() && d.value == in.size() &&
              std::memcmp(out.data(), in.data(), in.size()) == 0,
          "new filter e2e: roundtrip (" + tag + ")");
    return r.value;
  };
  {
    auto d = make_offsets_u64(1 << 20);
    const size_t off = run(d, false, "u64 offsets off");
    const size_t on = run(d, true, "u64 offsets on");
    // Raw LZ already does well on monotonic offsets (distance-8 matches on
    // the shared high bytes), so the filter win is ~25%, not multiples.
    check(on * 5 < off * 4, "new filter e2e: u64 offsets clearly smaller");
  }
  {
    // Amplitude large enough that dict32 bails (>64K distinct values), so any
    // win is attributable to zigzag_shuffle32. Since the unfiltered Beam
    // block codes its literals with the Crystal lattice coder (stride 4,
    // per-lane contexts), the raw encode already captures what the
    // transform exposes and the two come out equal; the filter must still
    // never make the block larger.
    auto d = make_oscillating(1 << 20, 4, 200000);
    const size_t off = run(d, false, "i32 oscillating off");
    const size_t on = run(d, true, "i32 oscillating on");
    check(on <= off, "new filter e2e: oscillating i32 not larger");
  }
}

void test_byte_rle_codec() {
  auto roundtrip = [&](const std::vector<spectra8::detail::u8>& in,
                       const std::string& tag) {
    std::vector<spectra8::detail::u8> enc;
    spectra8::detail::byte_rle_encode(in.data(), in.size(), enc);
    std::vector<spectra8::detail::u8> back(in.size());
    auto st = spectra8::detail::byte_rle_decode(enc.data(), enc.size(),
                                                 back.data(), back.size());
    check(st == spectra8::Status::Ok, "byte rle: decode ok (" + tag + ")");
    check(back == in, "byte rle: roundtrip (" + tag + ")");
  };

  roundtrip({}, "empty");
  roundtrip({1, 2, 3}, "short");
  {
    std::vector<spectra8::detail::u8> z(40000, 0);
    for (size_t i = 10000; i < 20000; ++i) z[i] = static_cast<spectra8::detail::u8>(i & 255);
    roundtrip(z, "sparse-zeros");
  }
  {
    // Shuffle-like lanes: long runs of identical bytes.
    std::vector<spectra8::detail::u8> lanes(128 * 1024);
    const size_t cnt = lanes.size() / 4;
    for (size_t e = 0; e < cnt; ++e) {
      lanes[e] = 0;
      lanes[cnt + e] = static_cast<spectra8::detail::u8>(e & 3);
      lanes[2 * cnt + e] = static_cast<spectra8::detail::u8>((e >> 8) & 255);
      lanes[3 * cnt + e] = static_cast<spectra8::detail::u8>(e & 255);
    }
    roundtrip(lanes, "shuffle-lanes");
  }
}

void test_filter_byte_rle_path() {
  std::mt19937_64 rng(31);
  auto run = [&](const std::vector<uint8_t>& in, bool filt, bool bwt) -> size_t {
    spectra8::CompressParams p;
    p.level = 9;
    p.filters = filt;
    p.codec = bwt ? spectra8::Codec::Prism : spectra8::Codec::Beam;
    p.block_log = 17;  // 128 KiB blocks
    auto src = bytes(in);
    std::vector<std::byte> comp(spectra8::compress_bound(src.size()));
    auto r = spectra8::compress(src, comp, p);
    check(r.ok(), "filter rle: compress ok");
    if (!r.ok()) return 0;
    comp.resize(r.value);
    std::vector<std::byte> out(in.size());
    auto d = spectra8::decompress(comp, out);
    check(d.ok() && d.value == in.size(), "filter rle: decompress size");
    check(in.empty() || std::memcmp(out.data(), in.data(), in.size()) == 0,
          "filter rle: roundtrip bytes");
    return r.value;
  };

  // int32 column: filter + byte-RLE should roundtrip (LZ path).
  {
    std::vector<uint8_t> d(128 * 1024);
    uint32_t v = 100000;
    for (size_t i = 0; i + 4 <= d.size(); i += 4) {
      v += static_cast<uint32_t>(static_cast<int>(rng() % 101) - 50);
      std::memcpy(&d[i], &v, 4);
    }
    (void)run(d, true, false);
  }

  // Same data through filter + byte-RLE + BWT.
  {
    std::vector<uint8_t> d(128 * 1024);
    uint32_t v = 50000;
    for (size_t i = 0; i + 4 <= d.size(); i += 4) {
      v += static_cast<uint32_t>(static_cast<int>(rng() % 51) - 25);
      std::memcpy(&d[i], &v, 4);
    }
    (void)run(d, true, true);
  }
}

void test_filter_fallback() {
  std::vector<uint8_t> d(128 * 1024);
  uint32_t v = 100000;
  for (size_t i = 0; i + 4 <= d.size(); i += 4) {
    v += 3;
    std::memcpy(&d[i], &v, 4);
  }
  auto src = bytes(d);
  spectra8::CompressParams p;
  p.level = 6;
  p.filters = true;
  p.filter_fallback = false;
  p.block_log = 17;
  std::vector<std::byte> comp(spectra8::compress_bound(src.size()));
  auto r = spectra8::compress(src, comp, p);
  check(r.ok(), "filter fallback: compress ok");
  if (!r.ok()) return;
  comp.resize(r.value);
  std::vector<std::byte> out(d.size());
  auto dr = spectra8::decompress(comp, out);
  check(dr.ok() && dr.value == d.size(), "filter fallback: decompress size");
  check(std::memcmp(out.data(), d.data(), d.size()) == 0,
        "filter fallback: roundtrip");
}

void test_filter_rle_dict_toggles() {
  std::vector<uint8_t> d(64 * 1024);
  uint32_t v = 42;
  for (size_t i = 0; i + 4 <= d.size(); i += 4) {
    v += 1;
    std::memcpy(&d[i], &v, 4);
  }
  auto src = bytes(d);
  auto roundtrip = [&](const spectra8::CompressParams& p,
                       const std::string& tag) {
    std::vector<std::byte> comp(spectra8::compress_bound(src.size()));
    auto r = spectra8::compress(src, comp, p);
    check(r.ok(), "filter toggles: compress ok (" + tag + ")");
    if (!r.ok()) return;
    comp.resize(r.value);
    std::vector<std::byte> out(d.size());
    auto dr = spectra8::decompress(comp, out);
    check(dr.ok() && dr.value == d.size(),
          "filter toggles: decompress size (" + tag + ")");
    check(std::memcmp(out.data(), d.data(), d.size()) == 0,
          "filter toggles: roundtrip (" + tag + ")");
  };

  spectra8::CompressParams base;
  base.level = 6;
  base.filters = true;
  base.block_log = 17;
  roundtrip(base, "defaults");
  base.filter_rle = false;
  roundtrip(base, "no-rle");
  base.filter_rle = true;
  base.filter_dict = false;
  roundtrip(base, "no-dict");
}

void test_intrinsic_byte_rle() {
  auto roundtrip = [&](const std::vector<uint8_t>& in, bool bwt,
                       const std::string& tag, bool expect_rle = false) {
    spectra8::CompressParams p;
    p.level = 9;
    p.filters = false;
    p.codec = bwt ? spectra8::Codec::Prism : spectra8::Codec::Beam;
    p.block_log = 17;
    auto src = bytes(in);
    std::vector<std::byte> comp(spectra8::compress_bound(src.size()));
    auto r = spectra8::compress(src, comp, p);
    check(r.ok(), "intrinsic rle: compress ok (" + tag + ")");
    if (!r.ok()) return;
    comp.resize(r.value);
    std::vector<std::byte> out(in.size());
    auto d = spectra8::decompress(comp, out);
    check(d.ok() && d.value == in.size(),
          "intrinsic rle: decompress size (" + tag + ")");
    check(in.empty() || std::memcmp(out.data(), in.data(), in.size()) == 0,
          "intrinsic rle: roundtrip (" + tag + ")");
    if (expect_rle) {
      // Either intrinsic byte-RLE (kFlagFilters) or Strategy B's run-heavy LZ
      // fallback — both are valid ways to milk long identical-byte lanes.
      const auto* raw = reinterpret_cast<const uint8_t*>(comp.data());
      check((raw[5] & 0x10) || r.value < in.size() / 8,
            "intrinsic rle: filters flag or strong LZ fallback (" + tag + ")");
      check(r.value < in.size() / 4,
            "intrinsic rle: strong ratio (" + tag + ")");
    }
  };

  // Plain LZ already handles long zero runs; intrinsic RLE is optional here.
  std::vector<uint8_t> zeros(128 * 1024, 0);
  zeros[0] = 1;
  roundtrip(zeros, false, "lz-zeros");
  roundtrip(zeros, true, "bwt-zeros");

  // Columnar lanes: long identical-byte runs without an external filter.
  std::vector<uint8_t> lanes(128 * 1024);
  const size_t cnt = lanes.size() / 4;
  for (size_t e = 0; e < cnt; ++e) {
    lanes[e] = 0;
    lanes[cnt + e] = static_cast<uint8_t>(e & 3);
    lanes[2 * cnt + e] = 0;
    lanes[3 * cnt + e] = 0;
  }
  roundtrip(lanes, true, "bwt-lanes", true);
}

void test_bwt_primitives() {
  using spectra8::detail::u8;
  using spectra8::detail::usize;

  // BWT forward+inverse must reproduce the input exactly.
  auto bwt_roundtrip = [&](const std::vector<u8>& in, const std::string& tag) {
    const usize n = in.size();
    if (n == 0) return;
    std::vector<u8> fwd(n), back(n);
    usize primary = spectra8::detail::bwt_forward(in.data(), n, fwd.data());
    // Sentinel BWT works over a length-(n+1) matrix, so primary is in [0, n].
    check(primary <= n, "bwt prim: primary in range (" + tag + ")");
    auto st = spectra8::detail::bwt_inverse(fwd.data(), n, primary, back.data());
    check(st == spectra8::Status::Ok, "bwt prim: inverse ok (" + tag + ")");
    check(std::memcmp(in.data(), back.data(), n) == 0,
          "bwt prim: roundtrip (" + tag + ")");
  };

  // MTF forward+inverse must be the identity.
  auto mtf_roundtrip = [&](const std::vector<u8>& in, const std::string& tag) {
    std::vector<u8> buf = in;
    spectra8::detail::mtf_forward(buf.data(), buf.size());
    spectra8::detail::mtf_inverse(buf.data(), buf.size());
    check(in.empty() || std::memcmp(in.data(), buf.data(), in.size()) == 0,
          "mtf prim: roundtrip (" + tag + ")");
  };

  // Classic textbook case.
  {
    std::vector<u8> banana = {'b', 'a', 'n', 'a', 'n', 'a'};
    bwt_roundtrip(banana, "banana");
    mtf_roundtrip(banana, "banana");
  }
  // Single byte and all-equal.
  {
    bwt_roundtrip({0x42}, "single");
    bwt_roundtrip(std::vector<u8>(1000, 0x7F), "all-equal");
    mtf_roundtrip(std::vector<u8>(1000, 0x7F), "all-equal");
  }
  // Random and highly repetitive (adversarial for the suffix array).
  {
    std::mt19937_64 rng(1234);
    std::vector<u8> rnd(50000);
    for (auto& b : rnd) b = static_cast<u8>(rng());
    bwt_roundtrip(rnd, "random");
    mtf_roundtrip(rnd, "random");

    std::vector<u8> rep(50000, 0);
    for (usize i = 0; i < rep.size(); ++i) rep[i] = static_cast<u8>((i % 7) == 0 ? 1 : 0);
    bwt_roundtrip(rep, "repetitive");

    std::vector<u8> runs(40000);
    for (usize i = 0; i < runs.size(); ++i) runs[i] = static_cast<u8>('A' + ((i / 500) % 3));
    bwt_roundtrip(runs, "runs");
  }
  // Full byte alphabet.
  {
    std::vector<u8> all(2560);
    for (usize i = 0; i < all.size(); ++i) all[i] = static_cast<u8>(i % 256);
    bwt_roundtrip(all, "alphabet");
    mtf_roundtrip(all, "alphabet");
  }
  // Randomized small-input sweep over tiny alphabets. SA-IS bugs tend to hide on
  // short strings with many equal/tied suffixes, so hammer those exhaustively.
  {
    std::mt19937_64 rng(98765);
    bool ok = true;
    for (int trial = 0; trial < 20000 && ok; ++trial) {
      const usize n = 1 + (rng() % 64);
      const int alpha = 2 + static_cast<int>(rng() % 5);
      std::vector<u8> s(n), fwd(n), back(n);
      for (auto& b : s) b = static_cast<u8>(rng() % alpha);
      const usize prim = spectra8::detail::bwt_forward(s.data(), n, fwd.data());
      const auto st = spectra8::detail::bwt_inverse(fwd.data(), n, prim, back.data());
      if (st != spectra8::Status::Ok || prim > n ||
          std::memcmp(s.data(), back.data(), n) != 0)
        ok = false;
    }
    check(ok, "bwt prim: randomized small-input roundtrip sweep");
  }
  // Inverse must reject an out-of-range primary index.
  {
    std::vector<u8> fwd = {1, 2, 3, 4};
    std::vector<u8> back(4);
    // primary must index a row of the (n+1)-row matrix, i.e. be <= n; n+1 is bad.
    auto st = spectra8::detail::bwt_inverse(fwd.data(), 4, 5, back.data());
    check(st != spectra8::Status::Ok, "bwt prim: bad primary rejected");
  }
  // Multi-chain parallel inverse (16-chain is the encoder default on large blocks).
  {
    std::mt19937_64 rng(4242);
    for (int chains : {8, 12, 16}) {
      for (usize n : {usize{300000}, usize{512} * 1024, usize{8 * 1024 * 1024 - 3}}) {
        std::vector<u8> in(n);
        for (auto& b : in) b = static_cast<u8>(rng());
        std::vector<u8> fwd(n), back(n);
        std::vector<usize> anchors(static_cast<usize>(chains), 0);
        const usize primary = spectra8::detail::bwt_forward(
            in.data(), n, fwd.data(), anchors.data(), chains);
        const auto st = spectra8::detail::bwt_inverse(
            fwd.data(), n, primary, back.data(), anchors.data(), chains);
        const std::string tag = std::to_string(chains) + "-chain n=" + std::to_string(n);
        check(st == spectra8::Status::Ok, "bwt prim: " + tag + " inverse ok");
        check(std::memcmp(in.data(), back.data(), n) == 0, "bwt prim: " + tag + " roundtrip");
      }
    }
  }
  // Compact LF-table path (m <= 64K => u16 link + u8 sym): exercise walk4/8/16
  // _compact directly. The default encoder never pairs <64 KiB blocks with
  // multi-chain anchors, so force it here for correctness coverage.
  {
    std::mt19937_64 rng(525);
    for (int chains : {4, 8, 16}) {
      for (usize n : {usize{4096}, usize{32768}, usize{65535}}) {
        std::vector<u8> in(n);
        for (auto& b : in) b = static_cast<u8>('A' + (rng() % 6));
        std::vector<u8> fwd(n), back(n);
        std::vector<usize> anchors(static_cast<usize>(chains), 0);
        const usize primary = spectra8::detail::bwt_forward(
            in.data(), n, fwd.data(), anchors.data(), chains);
        const auto st = spectra8::detail::bwt_inverse(
            fwd.data(), n, primary, back.data(), anchors.data(), chains);
        const std::string tag =
            "compact " + std::to_string(chains) + "-chain n=" + std::to_string(n);
        check(st == spectra8::Status::Ok, "bwt prim: " + tag + " inverse ok");
        check(std::memcmp(in.data(), back.data(), n) == 0,
              "bwt prim: " + tag + " roundtrip");
      }
    }
  }
}

void test_arith_codec() {
  auto roundtrip = [&](const std::vector<spectra8::detail::u8>& in,
                       const std::string& tag) {
    std::vector<spectra8::detail::u8> payload;
    spectra8::detail::arith::encode(payload, in.data(), in.size());
    std::vector<spectra8::detail::u8> back(in.size());
    auto st = spectra8::detail::arith::decode(payload.data(), payload.size(),
                                               back.data(), back.size());
    check(st == spectra8::Status::Ok, "arith: decode ok (" + tag + ")");
    check(back == in, "arith: roundtrip (" + tag + ")");
  };

  roundtrip({}, "empty");
  std::vector<spectra8::detail::u8> skew(8000);
  for (size_t i = 0; i < skew.size(); ++i)
    skew[i] = (i % 17 == 0) ? 0 : static_cast<spectra8::detail::u8>((i % 40) + 4);
  roundtrip(skew, "skew");

  std::mt19937_64 rng(91);
  for (int t = 0; t < 50; ++t) {
    std::vector<spectra8::detail::u8> in(static_cast<size_t>(rng() % 12000));
    for (auto& b : in) b = static_cast<spectra8::detail::u8>(rng() % 64);
    roundtrip(in, "rand");
  }
}

void test_arith_adaptive_codec() {
  auto roundtrip = [&](const std::vector<spectra8::detail::u8>& in,
                       const std::string& tag) {
    std::vector<spectra8::detail::u8> payload;
    spectra8::detail::arith::encode_adaptive(payload, in.data(), in.size());
    std::vector<spectra8::detail::u8> back(in.size());
    auto st = spectra8::detail::arith::decode_adaptive(
        payload.data(), payload.size(), back.data(), back.size());
    check(st == spectra8::Status::Ok, "arith ad: decode ok (" + tag + ")");
    check(back == in, "arith ad: roundtrip (" + tag + ")");
  };

  roundtrip({}, "empty");
  roundtrip({7}, "one");
  std::vector<spectra8::detail::u8> skew(8000);
  for (size_t i = 0; i < skew.size(); ++i)
    skew[i] = (i % 17 == 0) ? 0 : static_cast<spectra8::detail::u8>((i % 40) + 4);
  roundtrip(skew, "skew");

  // Full-alphabet stream long enough to force model rescaling.
  std::vector<spectra8::detail::u8> wide(200000);
  std::mt19937_64 wr(5);
  for (auto& b : wide) b = static_cast<spectra8::detail::u8>(wr() & 0xFF);
  roundtrip(wide, "wide-rescale");

  std::mt19937_64 rng(123);
  for (int t = 0; t < 80; ++t) {
    std::vector<spectra8::detail::u8> in(static_cast<size_t>(rng() % 12000));
    for (auto& b : in) b = static_cast<spectra8::detail::u8>(rng() % 64);
    roundtrip(in, "rand");
  }
}

void test_ans_codec() {
  using spectra8::detail::u8;
  using spectra8::detail::usize;

  // Roundtrip via one-shot decode and via chunked StreamDecoder pulls with
  // chunk sizes that are not multiples of the interleave width, so the
  // align/main/tail phases of the 2- and 4-state decoders all execute.
  // Sizes below 4096 use the dual-state payload (mode 1), larger ones the
  // 4-state payload (mode 2).
  auto roundtrip = [&](const std::vector<u8>& in, const std::string& tag) {
    std::vector<u8> payload;
    spectra8::detail::ans::encode(payload, in.data(), in.size());
    std::vector<u8> back(in.size());
    auto st = spectra8::detail::ans::decode(payload.data(), payload.size(),
                                             back.data(), back.size());
    check(st == spectra8::Status::Ok, "ans: decode ok (" + tag + ")");
    check(back == in, "ans: roundtrip (" + tag + ")");

    for (usize chunk : {usize{1}, usize{3}, usize{7}, usize{1001}}) {
      spectra8::detail::ans::StreamDecoder sd;
      st = sd.init(payload.data(), payload.size(), in.size());
      check(st == spectra8::Status::Ok, "ans: stream init (" + tag + ")");
      std::vector<u8> sb(in.size());
      bool ok = true;
      for (usize off = 0; off < in.size(); off += chunk) {
        const usize n = std::min(chunk, in.size() - off);
        if (sd.decode(sb.data() + off, n) != spectra8::Status::Ok) {
          ok = false;
          break;
        }
      }
      check(ok && sb == in,
            "ans: chunked roundtrip (" + tag + " chunk=" +
                std::to_string(chunk) + ")");
    }
  };

  std::vector<u8> skew(20000);
  for (size_t i = 0; i < skew.size(); ++i)
    skew[i] = (i % 17 == 0) ? 0 : static_cast<u8>((i % 40) + 4);
  roundtrip(skew, "skew-4way");
  roundtrip(std::vector<u8>(skew.begin(), skew.begin() + 3000), "skew-2way");
  roundtrip(std::vector<u8>(skew.begin(), skew.begin() + 4096), "boundary");
  roundtrip({1, 2}, "two");
  roundtrip(std::vector<u8>(5000, 42), "constant");

  std::mt19937_64 rng(77);
  for (int t = 0; t < 40; ++t) {
    std::vector<u8> in(2 + static_cast<size_t>(rng() % 20000));
    for (auto& b : in) b = static_cast<u8>(rng() % 64);
    roundtrip(in, "rand");
  }
}

void test_ans_o1_codec() {
  using spectra8::detail::u8;

  auto roundtrip = [&](const std::vector<u8>& in, const std::string& tag) {
    std::vector<u8> payload;
    spectra8::detail::ans::encode_o1(payload, in.data(), in.size());
    std::vector<u8> back(in.size());
    auto st = spectra8::detail::ans::decode_o1(payload.data(), payload.size(),
                                                back.data(), back.size());
    check(st == spectra8::Status::Ok, "ans o1: decode ok (" + tag + ")");
    check(back == in, "ans o1: roundtrip (" + tag + ")");
  };

  // MTF-shaped stream: strong prev-symbol structure; hits the fine low-rank
  // AnsO1 classes ({0..3},{4..7},…) plus the high-rank tail bucket.
  std::vector<u8> mtf(30000);
  std::mt19937_64 rng(99);
  for (size_t i = 0; i < mtf.size(); ++i) {
    const u8 prev = i ? mtf[i - 1] : 0;
    const int roll = static_cast<int>(rng() % 100);
    if (prev == 0)
      mtf[i] = roll < 60 ? 0 : static_cast<u8>(1 + rng() % 30);
    else
      mtf[i] = roll < 35 ? 0 : static_cast<u8>(rng() % (roll < 90 ? 8 : 200));
  }
  roundtrip(mtf, "mtf");
  roundtrip(std::vector<u8>(mtf.begin(), mtf.begin() + 1), "one");
  roundtrip(std::vector<u8>(mtf.begin(), mtf.begin() + 2), "two");
  roundtrip(std::vector<u8>(mtf.begin(), mtf.begin() + 1023), "odd");
  roundtrip(std::vector<u8>(7000, 42), "constant");
  roundtrip({}, "empty");

  for (int t = 0; t < 40; ++t) {
    std::vector<u8> in(1 + static_cast<size_t>(rng() % 20000));
    for (auto& b : in) b = static_cast<u8>(rng() % (t % 2 ? 256 : 5));
    roundtrip(in, "rand");
  }
}

void test_arith_adaptive_o1_codec() {
  using spectra8::detail::u8;

  auto roundtrip = [&](const std::vector<u8>& in, const std::string& tag) {
    std::vector<u8> payload;
    spectra8::detail::arith::encode_adaptive_o1(payload, in.data(), in.size());
    std::vector<u8> back(in.size());
    auto st = spectra8::detail::arith::decode_adaptive_o1(
        payload.data(), payload.size(), back.data(), back.size());
    check(st == spectra8::Status::Ok, "arith o1: decode ok (" + tag + ")");
    check(back == in, "arith o1: roundtrip (" + tag + ")");
  };

  std::mt19937_64 rng(123);
  std::vector<u8> mtf(30000);
  for (size_t i = 0; i < mtf.size(); ++i) {
    const u8 prev = i ? mtf[i - 1] : 0;
    const int roll = static_cast<int>(rng() % 100);
    if (prev == 0)
      mtf[i] = roll < 60 ? 0 : static_cast<u8>(1 + rng() % 30);
    else
      mtf[i] = roll < 35 ? 0 : static_cast<u8>(rng() % (roll < 90 ? 8 : 200));
  }
  roundtrip(mtf, "mtf");
  roundtrip(std::vector<u8>(mtf.begin(), mtf.begin() + 1), "one");
  roundtrip(std::vector<u8>(7000, 42), "constant");
  for (int t = 0; t < 30; ++t) {
    std::vector<u8> in(1 + static_cast<size_t>(rng() % 20000));
    for (auto& b : in) b = static_cast<u8>(rng() % (t % 2 ? 256 : 5));
    roundtrip(in, "rand");
  }
}

void test_bwt_mtf_bzip2_rle() {
  auto roundtrip = [&](const std::vector<spectra8::detail::u8>& in,
                       const std::string& tag) {
    std::vector<spectra8::detail::u8> enc, back;
    spectra8::detail::bwt_mtf_bzip2_encode(in.data(), in.size(), enc);
    spectra8::detail::bwt_mtf_bzip2_decode(enc.data(), enc.size(), back);
    check(back == in, "bwt mtf bzip2 rle: roundtrip (" + tag + ")");
  };

  roundtrip({}, "empty");
  roundtrip({0}, "one-zero");
  roundtrip({0, 0, 0, 0, 0, 3}, "run-then-lit");
  roundtrip({1, 0, 0, 2, 0, 0, 0, 4}, "mixed");

  std::mt19937_64 rng(77);
  for (int t = 0; t < 200; ++t) {
    std::vector<spectra8::detail::u8> in(static_cast<size_t>(rng() % 4000));
    for (auto& b : in) b = static_cast<spectra8::detail::u8>(rng() % 32);
    roundtrip(in, "rand");
  }

  // The fused inverse (expand + MTF-inverse in one pass) must equal the
  // two-step path (decode rank stream, then mtf_inverse).
  auto fused_equiv = [&](const std::vector<spectra8::detail::u8>& ranks,
                         const std::string& tag) {
    std::vector<spectra8::detail::u8> enc;
    spectra8::detail::bwt_mtf_bzip2_encode(ranks.data(), ranks.size(), enc);

    std::vector<spectra8::detail::u8> two(ranks.size());
    std::vector<spectra8::detail::u8> ranks_back;
    spectra8::detail::bwt_mtf_bzip2_decode(enc.data(), enc.size(), ranks_back,
                                            ranks.size());
    if (!ranks_back.empty()) {
      std::memcpy(two.data(), ranks_back.data(), ranks_back.size());
      spectra8::detail::mtf_inverse(two.data(), two.size());
    }

    std::vector<spectra8::detail::u8> fused(ranks.size());
    spectra8::detail::bwt_mtf_bzip2_inverse(enc.data(), enc.size(),
                                             fused.data(), fused.size());
    check(fused == two, "bwt mtf bzip2 fused inverse (" + tag + ")");
  };
  fused_equiv({}, "empty");
  fused_equiv({0, 0, 0, 1, 0, 5, 0, 0, 2}, "mixed");
  for (int t = 0; t < 100; ++t) {
    std::vector<spectra8::detail::u8> in(static_cast<size_t>(rng() % 4000));
    for (auto& b : in) b = static_cast<spectra8::detail::u8>(rng() % 24);
    fused_equiv(in, "rand");
  }
}

void test_unified_codecs() {
  check(spectra8::codec_available(spectra8::Codec::Beam), "codec: lz available");
  check(spectra8::codec_available(spectra8::Codec::Prism),
        "codec: bwt available");
  check(spectra8::codec_available(spectra8::Codec::Mosaic),
        "codec: mosaic available");
  check(spectra8::codec_available(spectra8::Codec::Oracle),
        "codec: oracle available");
  check(spectra8::codec_available(spectra8::Codec::Crystal),
        "codec: crystal available");
  check(spectra8::codec_available(spectra8::Codec::Flash),
        "codec: flash available");
  check(spectra8::codec_available(spectra8::Codec::Laser),
        "codec: laser available");
  check(spectra8::codec_available(spectra8::Codec::Grating),
        "codec: grating available");

  std::vector<uint8_t> raw(80000);
  for (size_t i = 0; i < raw.size(); ++i)
    raw[i] = static_cast<uint8_t>((i * 31) ^ (i >> 2));
  auto src = bytes(raw);

  auto roundtrip = [&](spectra8::Codec codec, const char* tag) {
    if (!spectra8::codec_available(codec)) return;
    spectra8::CompressParams p;
    p.codec = codec;
    p.level = 5;
    const size_t bound = spectra8::compress_bound(src.size(), codec);
    check(bound > 0, std::string("codec: bound > 0 (") + tag + ")");
    std::vector<std::byte> comp(bound);
    auto r = spectra8::compress(src, comp, p);
    check(r.ok(), std::string("codec: compress ok (") + tag + ")");
    if (!r.ok()) return;
    comp.resize(r.value);
    auto cs = spectra8::get_frame_content_size(comp);
    check(cs && *cs == raw.size(),
          std::string("codec: content size (") + tag + ")");
    std::vector<std::byte> out(raw.size());
    auto d = spectra8::decompress(comp, out);
    check(d.ok() && d.value == raw.size() &&
              std::memcmp(out.data(), raw.data(), raw.size()) == 0,
          std::string("codec: roundtrip (") + tag + ")");
  };

  roundtrip(spectra8::Codec::Beam, "lz");
  roundtrip(spectra8::Codec::Prism, "bwt");
  roundtrip(spectra8::Codec::Mosaic, "mosaic");
  roundtrip(spectra8::Codec::Oracle, "oracle");
  roundtrip(spectra8::Codec::Crystal, "crystal");
  roundtrip(spectra8::Codec::Flash, "flash");
  roundtrip(spectra8::Codec::Laser, "laser");
  roundtrip(spectra8::Codec::Grating, "grating");

  // Concurrent Laser encodes (the match finder used to read src[n] after a
  // full-length match). 8 threads × long-match pages is the tryzip oracle case.
  {
    std::vector<uint8_t> rep(1 << 18);
    for (size_t i = 0; i < rep.size(); ++i)
      rep[i] = static_cast<uint8_t>((i % 17) + 3 * (i >> 8));
    std::vector<uint8_t> run(1 << 18, 0xA5);
    const std::vector<uint8_t>* pages[] = {&raw, &rep, &run};
    std::atomic<int> fails{0};
    auto laser_job = [&](int tid) {
      for (int it = 0; it < 8; ++it) {
        const auto& page = *pages[static_cast<size_t>((tid + it) % 3)];
        auto in = bytes(page);
        spectra8::CompressParams p;
        p.codec = spectra8::Codec::Laser;
        p.level = 5;
        p.threads = 1;
        p.filters = (it & 1) != 0;
        if (p.filters) {
          p.filter_mask = 1ull << (1u + static_cast<unsigned>((tid + it) % 18));
          p.filter_fallback = false;
        }
        std::vector<std::byte> comp(spectra8::compress_bound(page.size(), p.codec));
        auto r = spectra8::compress(in, comp, p);
        if (!r.ok()) {
          fails.fetch_add(1);
          continue;
        }
        comp.resize(r.value);
        std::vector<std::byte> out(page.size());
        auto d = spectra8::decompress(comp, out);
        if (!d.ok() || d.value != page.size() ||
            std::memcmp(out.data(), page.data(), page.size()) != 0)
          fails.fetch_add(1);
      }
    };
    std::vector<std::thread> pool;
    for (int t = 0; t < 8; ++t) pool.emplace_back(laser_job, t);
    for (auto& th : pool) th.join();
    check(fails.load() == 0, "codec: concurrent laser compress");
  }
  roundtrip(spectra8::Codec::Zstd, "zstd");

  // Peaked integer columns: width 2/4 plus an escape tail, the case Grating
  // codes as whole scalars rather than bytes.
  auto grating_ints = [&](const std::vector<uint8_t>& col, const char* tag) {
    auto csrc = bytes(col);
    spectra8::CompressParams p;
    p.codec = spectra8::Codec::Grating;
    p.level = 1;
    const size_t bound = spectra8::compress_bound(csrc.size(), p.codec);
    std::vector<std::byte> comp(bound);
    auto r = spectra8::compress(csrc, comp, p);
    check(r.ok() && r.value > 0 && r.value < col.size(),
          std::string("grating: compresses (") + tag + ")");
    if (!r.ok()) return;
    comp.resize(r.value);
    std::vector<std::byte> out(col.size());
    auto d = spectra8::decompress(comp, out);
    check(d.ok() && d.value == col.size() &&
              std::memcmp(out.data(), col.data(), col.size()) == 0,
          std::string("grating: integer roundtrip (") + tag + ")");
  };
  {
    std::vector<uint8_t> i16(64 * 1024);
    std::mt19937 rng(1);
    std::geometric_distribution<int> mag(0.15);
    for (size_t i = 0; i + 2 <= i16.size(); i += 2) {
      int m = mag(rng);
      std::int16_t v = static_cast<std::int16_t>((rng() & 1) ? -m : m);
      std::memcpy(i16.data() + i, &v, 2);
    }
    grating_ints(i16, "laplace i16");
    std::vector<uint8_t> i32(64 * 1024);
    for (size_t i = 0; i + 4 <= i32.size(); i += 4) {
      std::int32_t v = (rng() % 100) < 92 ? static_cast<std::int32_t>(rng() % 32)
                                          : static_cast<std::int32_t>(rng() % 200000);
      std::memcpy(i32.data() + i, &v, 4);
    }
    grating_ints(i32, "tail i32");
  }

  // Flash L1 (raw interleaved literals) and L5 (optional ratio header) both
  // roundtrip through the same format-driven decoder.
  auto flash_level = [&](int level, const char* tag) {
    spectra8::CompressParams p;
    p.codec = spectra8::Codec::Flash;
    p.level = level;
    const size_t bound = spectra8::compress_bound(src.size(), spectra8::Codec::Flash);
    std::vector<std::byte> comp(bound);
    auto r = spectra8::compress(src, comp, p);
    check(r.ok(), std::string("flash L") + std::to_string(level) + " compress (" +
                      tag + ")");
    if (!r.ok()) return;
    comp.resize(r.value);
    std::vector<std::byte> out(raw.size());
    auto d = spectra8::decompress(comp, out);
    check(d.ok() && d.value == raw.size() &&
              std::memcmp(out.data(), raw.data(), raw.size()) == 0,
          std::string("flash L") + std::to_string(level) + " roundtrip (" + tag + ")");
  };
  flash_level(1, "fast");
  flash_level(5, "ratio");

  // Direct payload: L1 must not emit the 0x1F header; L5 Huffman residuals
  // must still decode with the same decode_flash_block.
  {
    using spectra8::detail::u8;
    using spectra8::detail::usize;
    std::vector<u8> blk(80000);
    for (size_t i = 0; i < blk.size(); ++i)
      blk[i] = static_cast<u8>((i * 31) ^ (i >> 2));
    std::vector<u8> payload(blk.size() + 80);
    std::vector<u8> out(blk.size());

    const usize n1 = spectra8::detail::encode_flash_block(
        blk.data(), blk.size(), payload.data(), payload.size(), 1);
    check(n1 > 0 && n1 < blk.size(), "flash L1 payload encodes");
    check(!(n1 >= 4 && payload[0] == 0x1F && payload[1] == 0 && payload[2] == 0),
          "flash L1 payload has no ratio header");
    check(spectra8::detail::decode_flash_block(payload.data(), n1, out.data(),
                                                blk.size()) == spectra8::Status::Ok &&
              std::memcmp(out.data(), blk.data(), blk.size()) == 0,
          "flash L1 payload decodes");

    // Unique 4-byte groups with a constant first byte: no LZ matches, but
    // Huffman on the literal bank pays, so L5 emits the 0x1F header.
    std::vector<u8> residual(4 * 12000);
    for (size_t g = 0; g < 12000; ++g) {
      residual[g * 4 + 0] = 7;
      residual[g * 4 + 1] = static_cast<u8>(g);
      residual[g * 4 + 2] = static_cast<u8>(g >> 8);
      residual[g * 4 + 3] = static_cast<u8>(g >> 16);
    }
    payload.assign(residual.size() + 80, 0);
    out.assign(residual.size(), 0);
    const usize n5 = spectra8::detail::encode_flash_block(
        residual.data(), residual.size(), payload.data(), payload.size(), 5);
    check(n5 > 0 && n5 < residual.size(), "flash L5 residual encodes");
    const bool l5_hdr =
        n5 >= 4 && payload[0] == 0x1F && payload[1] == 0 && payload[2] == 0;
    check(l5_hdr && (payload[3] & 0x01) != 0, "flash L5 residual uses Huffman bank");
    check(spectra8::detail::decode_flash_block(payload.data(), n5, out.data(),
                                                residual.size()) ==
                  spectra8::Status::Ok &&
              std::memcmp(out.data(), residual.data(), residual.size()) == 0,
          "flash L5 Huffman payload decodes");
  }

  // Test level 0 (Flash mode default)
  {
    spectra8::CompressParams p;
    p.level = 0;
    const size_t bound = spectra8::compress_bound(src.size());
    std::vector<std::byte> comp(bound);
    auto r = spectra8::compress(src, comp, p);
    check(r.ok(), "level 0: compress ok");
    if (r.ok()) {
      comp.resize(r.value);
      std::vector<std::byte> out(raw.size());
      auto d = spectra8::decompress(comp, out);
      check(d.ok() && d.value == raw.size() &&
            std::memcmp(out.data(), raw.data(), raw.size()) == 0,
            "level 0: roundtrip ok");
    }
  }

  // Regression: Crystal's BWT cost estimate caches the block transform on
  // the encoding thread; when the lattice path wins the entry used to stay
  // valid for (pointer, length), and a later Mosaic/Oracle encode of
  // *different* data in the same buffer was served the stale transform and
  // wrote a payload for the wrong block. Same buffer, same size, new bytes.
  {
    const std::size_t n = 1 << 18;
    std::vector<std::byte> buf(n);
    std::mt19937 rng(77);
    // Numeric i16 records: the estimate runs, the lattice wins.
    for (std::size_t i = 0; i < n; i += 2) {
      const auto v = static_cast<std::uint16_t>(1000 + (rng() % 64));
      buf[i] = std::byte(v & 0xFF);
      buf[i + 1] = std::byte(v >> 8);
    }
    std::vector<std::byte> comp(spectra8::compress_bound(n));
    spectra8::CompressParams p;
    p.codec = spectra8::Codec::Crystal;
    p.level = 5;
    p.filters = false;
    check(spectra8::compress(buf, comp, p).ok(), "stale-bwt: crystal encode");
    // Same buffer, different content: text-like bytes.
    for (std::size_t i = 0; i < n; ++i)
      buf[i] = std::byte("the quick brown fox jumps over the lazy dog "[(i * 7 + (i / 97)) % 44]);
    for (auto codec : {spectra8::Codec::Mosaic, spectra8::Codec::Oracle}) {
      p.codec = codec;
      auto r = spectra8::compress(buf, comp, p);
      check(r.ok(), "stale-bwt: encode");
      std::vector<std::byte> out(n);
      auto d = spectra8::decompress(
          std::span<const std::byte>(comp.data(), r.value), out);
      check(d.ok() && d.value == n && out == buf,
            "stale-bwt: roundtrip after buffer reuse");
    }
  }
  roundtrip(spectra8::Codec::Bzip3, "bzip3");
  roundtrip(spectra8::Codec::Lzma, "lzma");
  roundtrip(spectra8::Codec::Auto, "auto");

  auto mosaic_rt = [&](const std::vector<uint8_t>& raw, const char* tag, int level) {
    auto s = bytes(raw);
    spectra8::CompressParams p;
    p.codec = spectra8::Codec::Mosaic;
    p.level = level;
    std::vector<std::byte> comp(spectra8::compress_bound(s.size(), p.codec));
    auto r = spectra8::compress(s, comp, p);
    check(r.ok(), std::string("mosaic: compress (") + tag + ")");
    if (!r.ok()) return;
    std::vector<std::byte> out(raw.size() ? raw.size() : 1);
    auto d = spectra8::decompress(
        std::span<const std::byte>(comp.data(), r.value),
        std::span<std::byte>(out.data(), raw.size()));
    check(d.ok() && d.value == raw.size() &&
              (raw.empty() || std::memcmp(out.data(), raw.data(), raw.size()) == 0),
          std::string("mosaic: roundtrip (") + tag + ")");
  };
  {
    std::vector<uint8_t> z(4096, 0);
    mosaic_rt(z, "zeros", 5);
    std::vector<uint8_t> rep(20000);
    for (size_t i = 0; i < rep.size(); ++i) rep[i] = static_cast<uint8_t>(i % 17);
    mosaic_rt(rep, "mod17", 5);
    mosaic_rt(rep, "mod17-L3", 3);
    mosaic_rt(rep, "mod17-L9", 9);
    std::vector<uint8_t> txt(50000);
    for (size_t i = 0; i < txt.size(); ++i)
      txt[i] = static_cast<uint8_t>("the quick brown fox "[i % 20]);
    mosaic_rt(txt, "text", 5);
    std::vector<uint8_t> lat(65536);
    for (size_t i = 0; i < lat.size(); ++i)
      lat[i] = static_cast<uint8_t>(((i / 8) * 13) ^ (i & 7));
    mosaic_rt(lat, "lattice", 5);
    mosaic_rt(lat, "lattice-L3", 3);
    std::vector<uint8_t> empty;
    mosaic_rt(empty, "empty", 5);
    spectra8::CompressParams pf;
    pf.codec = spectra8::Codec::Mosaic;
    pf.level = 5;
    pf.filters = true;
    std::vector<std::byte> src = bytes(txt);
    std::vector<std::byte> comp(spectra8::compress_bound(src.size(), pf.codec));
    auto r = spectra8::compress(src, comp, pf);
    check(r.ok(), "mosaic: filter compress");
    if (r.ok()) {
      std::vector<std::byte> out(txt.size());
      auto d = spectra8::decompress(
          std::span<const std::byte>(comp.data(), r.value), out);
      check(d.ok() && d.value == txt.size() &&
                std::memcmp(out.data(), txt.data(), txt.size()) == 0,
            "mosaic: filter roundtrip");
    }
  }

  auto oracle_rt = [&](const std::vector<uint8_t>& raw, const char* tag, int level) {
    auto s = bytes(raw);
    spectra8::CompressParams p;
    p.codec = spectra8::Codec::Oracle;
    p.level = level;
    std::vector<std::byte> comp(spectra8::compress_bound(s.size(), p.codec));
    auto r = spectra8::compress(s, comp, p);
    check(r.ok(), std::string("oracle: compress (") + tag + ")");
    if (!r.ok()) return;
    std::vector<std::byte> out(raw.size() ? raw.size() : 1);
    auto d = spectra8::decompress(
        std::span<const std::byte>(comp.data(), r.value),
        std::span<std::byte>(out.data(), raw.size()));
    check(d.ok() && d.value == raw.size() &&
              (raw.empty() || std::memcmp(out.data(), raw.data(), raw.size()) == 0),
          std::string("oracle: roundtrip (") + tag + ")");
  };
  {
    std::vector<uint8_t> z(4096, 0);
    oracle_rt(z, "zeros", 5);
    std::vector<uint8_t> rep(20000);
    for (size_t i = 0; i < rep.size(); ++i) rep[i] = static_cast<uint8_t>(i % 17);
    oracle_rt(rep, "mod17", 5);
    oracle_rt(rep, "mod17-L3", 3);
    oracle_rt(rep, "mod17-L9", 9);
    std::vector<uint8_t> txt(50000);
    for (size_t i = 0; i < txt.size(); ++i)
      txt[i] = static_cast<uint8_t>("the quick brown fox "[i % 20]);
    oracle_rt(txt, "text", 5);
    std::vector<uint8_t> lat(65536);
    for (size_t i = 0; i < lat.size(); ++i)
      lat[i] = static_cast<uint8_t>(((i / 8) * 13) ^ (i & 7));
    oracle_rt(lat, "lattice", 5);
    std::vector<uint8_t> empty;
    oracle_rt(empty, "empty", 5);
    spectra8::CompressParams pf;
    pf.codec = spectra8::Codec::Oracle;
    pf.level = 5;
    pf.filters = true;
    std::vector<std::byte> src = bytes(txt);
    std::vector<std::byte> comp(spectra8::compress_bound(src.size(), pf.codec));
    auto r = spectra8::compress(src, comp, pf);
    check(r.ok(), "oracle: filter compress");
    if (r.ok()) {
      std::vector<std::byte> out(txt.size());
      auto d = spectra8::decompress(
          std::span<const std::byte>(comp.data(), r.value), out);
      check(d.ok() && d.value == txt.size() &&
                std::memcmp(out.data(), txt.data(), txt.size()) == 0,
            "oracle: filter roundtrip");
    }
  }

  // Crystal: both payload kinds (static lattice coder L1-9, context mixing
  // L10-11), both context sets (stride 1 and record lattices), multi-stream
  // splits, tiny/empty inputs, filters, and a multi-block frame; plus the
  // block must never be larger than stored.
  auto crystal_rt = [&](const std::vector<uint8_t>& raw, const char* tag,
                       int level, bool filters = false) {
    auto s = bytes(raw);
    spectra8::CompressParams p;
    p.codec = spectra8::Codec::Crystal;
    p.level = level;
    p.filters = filters;
    std::vector<std::byte> comp(spectra8::compress_bound(s.size(), p.codec));
    auto r = spectra8::compress(s, comp, p);
    check(r.ok(), std::string("crystal: compress (") + tag + ")");
    if (!r.ok()) return;
    std::vector<std::byte> out(raw.size() ? raw.size() : 1);
    auto d = spectra8::decompress(
        std::span<const std::byte>(comp.data(), r.value),
        std::span<std::byte>(out.data(), raw.size()));
    check(d.ok() && d.value == raw.size() &&
              (raw.empty() || std::memcmp(out.data(), raw.data(), raw.size()) == 0),
          std::string("crystal: roundtrip (") + tag + ")");
  };
  {
    std::vector<uint8_t> empty;
    crystal_rt(empty, "empty", 5);
    std::vector<uint8_t> one(1, 42);
    crystal_rt(one, "one byte", 5);
    std::vector<uint8_t> tiny(63);
    for (size_t i = 0; i < tiny.size(); ++i) tiny[i] = static_cast<uint8_t>(i * 7);
    crystal_rt(tiny, "tiny (stored)", 5);
    std::vector<uint8_t> z(4096, 0);
    crystal_rt(z, "zeros", 5);
    std::vector<uint8_t> rep(20000);
    for (size_t i = 0; i < rep.size(); ++i) rep[i] = static_cast<uint8_t>(i % 17);
    for (int level : {1, 3, 4, 6, 7, 9, 10, 11}) {
      char tag[32];
      std::snprintf(tag, sizeof(tag), "mod17-L%d", level);
      crystal_rt(rep, tag, level);
    }
    std::vector<uint8_t> txt(50000);
    for (size_t i = 0; i < txt.size(); ++i)
      txt[i] = static_cast<uint8_t>("the quick brown fox "[i % 20]);
    crystal_rt(txt, "text", 5);
    crystal_rt(txt, "text-L1", 1);
    crystal_rt(txt, "text-L9", 9);
    // Record columns: pseudo-random i32 (stride 4), i16 (stride 2), i64.
    std::vector<uint8_t> rec(131072);
    {
      uint32_t x = 12345;
      for (size_t i = 0; i < rec.size(); i += 4) {
        x = x * 1664525u + 1013904223u;
        const uint32_t v = 1000 + (x >> 24) + ((x >> 16) & 63);
        rec[i] = static_cast<uint8_t>(v);
        rec[i + 1] = static_cast<uint8_t>(v >> 8);
        rec[i + 2] = 0;
        rec[i + 3] = 0;
      }
    }
    crystal_rt(rec, "i32-records", 1);
    crystal_rt(rec, "i32-records-L5", 5);
    crystal_rt(rec, "i32-records-L9", 9);
    crystal_rt(rec, "i32-records-L11", 11);
    crystal_rt(rec, "i32-records+filters", 5, true);
    // Experimental cmix-style LSTM tier (SPECTRA8_CRYSTAL_LSTM): both the
    // byte-flag (records) and bitwise (text) ensembles, small model so the
    // test stays cheap; the payload must decode without the env var set.
    {
      setenv("SPECTRA8_CRYSTAL_LSTM", "16,4,0.002,1", 1);
      std::vector<uint8_t> rec_small(rec.begin(), rec.begin() + 32768);
      std::vector<uint8_t> txt_small(txt.begin(), txt.begin() + 20000);
      spectra8::CompressParams p;
      p.codec = spectra8::Codec::Crystal;
      p.level = 11;
      std::vector<std::byte> comp_rec(spectra8::compress_bound(rec_small.size(), p.codec));
      std::vector<std::byte> comp_txt(spectra8::compress_bound(txt_small.size(), p.codec));
      auto r1 = spectra8::compress(bytes(rec_small), comp_rec, p);
      auto r2 = spectra8::compress(bytes(txt_small), comp_txt, p);
      unsetenv("SPECTRA8_CRYSTAL_LSTM");
      check(r1.ok() && r2.ok(), "crystal: LSTM tier compress");
      if (r1.ok() && r2.ok()) {
        std::vector<std::byte> out(rec_small.size());
        auto d1 = spectra8::decompress(
            std::span<const std::byte>(comp_rec.data(), r1.value), out);
        check(d1.ok() && d1.value == rec_small.size() &&
                  std::memcmp(out.data(), rec_small.data(), rec_small.size()) == 0,
              "crystal: LSTM tier roundtrip (records, flag mode)");
        out.assign(txt_small.size(), std::byte{});
        auto d2 = spectra8::decompress(
            std::span<const std::byte>(comp_txt.data(), r2.value), out);
        check(d2.ok() && d2.value == txt_small.size() &&
                  std::memcmp(out.data(), txt_small.data(), txt_small.size()) == 0,
              "crystal: LSTM tier roundtrip (text, bitwise mode)");
      }
    }
    // 1 MiB of correlated f64-like records (stride 8): a full-size block
    // for the static coder (8 streams, lattice/trend features, base byte).
    std::vector<uint8_t> big(1 << 20);
    {
      uint32_t x = 777;
      int32_t walk = 0;
      for (size_t i = 0; i < big.size(); i += 8) {
        x = x * 1664525u + 1013904223u;
        walk += static_cast<int32_t>(x >> 28) - 7;
        const uint64_t v = static_cast<uint64_t>(static_cast<int64_t>(walk)) << 20;
        for (int b = 0; b < 8; ++b) big[i + b] = static_cast<uint8_t>(v >> (8 * b));
      }
    }
    crystal_rt(big, "f64-walk-1MiB-L1", 1);
    crystal_rt(big, "f64-walk-1MiB-L5", 5);
    {
      // Static payload with flipped bytes and no checksums: decoding must
      // fail or produce garbage, never crash or read out of bounds.
      auto s = bytes(rec);
      spectra8::CompressParams p;
      p.codec = spectra8::Codec::Crystal;
      p.level = 5;
      std::vector<std::byte> comp(spectra8::compress_bound(s.size(), p.codec));
      auto r = spectra8::compress(s, comp, p);
      check(r.ok(), "crystal: static compress for corruption test");
      if (r.ok()) {
        comp.resize(r.value);
        std::vector<std::byte> out(rec.size());
        uint32_t seed = 99;
        for (int it = 0; it < 32; ++it) {
          std::vector<std::byte> bad = comp;
          seed = seed * 1664525u + 1013904223u;
          const size_t at = 16 + (seed >> 8) % (bad.size() - 16);
          bad[at] ^= std::byte{static_cast<uint8_t>(1u << (seed & 7))};
          (void)spectra8::decompress(bad, out);
        }
        check(true, "crystal: corrupt static payload survives");
      }
    }
    std::vector<uint8_t> rnd(70000);
    {
      uint64_t x = 88172645463325252ull;
      for (auto& b : rnd) {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        b = static_cast<uint8_t>(x);
      }
    }
    crystal_rt(rnd, "random (stored)", 5);
    {
      // Incompressible input must not expand beyond the stored bound.
      auto s = bytes(rnd);
      spectra8::CompressParams p;
      p.codec = spectra8::Codec::Crystal;
      p.level = 5;
      std::vector<std::byte> comp(spectra8::compress_bound(s.size(), p.codec));
      auto r = spectra8::compress(s, comp, p);
      check(r.ok() && r.value <= rnd.size() + 64, "crystal: random not expanded");
    }
    // Multi-block frame (block_log 14 => 16 KiB blocks), threads > 1.
    {
      auto s = bytes(rec);
      spectra8::CompressParams p;
      p.codec = spectra8::Codec::Crystal;
      p.level = 5;
      p.block_log = 14;
      p.threads = 4;
      p.block_checksums = true;
      p.content_checksum = true;
      std::vector<std::byte> comp(spectra8::compress_bound(s.size(), p.codec));
      auto r = spectra8::compress(s, comp, p);
      check(r.ok(), "crystal: multi-block compress");
      if (r.ok()) {
        std::vector<std::byte> out(rec.size());
        spectra8::DecompressParams dp;
        dp.threads = 4;
        auto d = spectra8::decompress(
            std::span<const std::byte>(comp.data(), r.value), out, dp);
        check(d.ok() && d.value == rec.size() &&
                  std::memcmp(out.data(), rec.data(), rec.size()) == 0,
              "crystal: multi-block roundtrip");
        // Corrupting the payload must fail cleanly (checksums on).
        std::vector<std::byte> bad(comp.begin(), comp.begin() + r.value);
        bad[bad.size() / 2] ^= std::byte{0x5A};
        auto bd = spectra8::decompress(bad, out);
        check(!bd.ok(), "crystal: corrupt payload detected");
      }
    }
    // Filters on text (byte-RLE / transform path) roundtrip.
    crystal_rt(txt, "text+filters", 5, true);
  }

  // ---------------------------------------------------------------------------
  // Crystal Static Probability Profiles (zero per-block header tax)
  // ---------------------------------------------------------------------------
  {
    // 1. FloatKinematics: 32-bit IEEE-754 floats
    {
      std::vector<float> floats(8192);
      float val = 45.0f;
      for (size_t i = 0; i < floats.size(); ++i) {
        val += ((i % 11) - 5) * 0.1f;
        if (val < 1.0f) val = 1.0f;
        floats[i] = val;
      }
      auto raw = std::span<const std::byte>(
          reinterpret_cast<const std::byte*>(floats.data()), floats.size() * sizeof(float));

      spectra8::CompressParams p;
      p.profile = spectra8::Profile::FloatKinematics;
      std::vector<std::byte> comp(spectra8::compress_bound(raw.size()));
      auto r = spectra8::compress(raw, comp, p);
      check(r.ok(), "crystal_profile: FloatKinematics compress");
      if (r.ok()) {
        std::vector<std::byte> out(raw.size());
        auto d = spectra8::decompress(
            std::span<const std::byte>(comp.data(), r.value), out);
        check(d.ok() && d.value == raw.size() &&
                  std::memcmp(out.data(), raw.data(), raw.size()) == 0,
              "crystal_profile: FloatKinematics roundtrip");
        check(r.value < raw.size(), "crystal_profile: FloatKinematics compressed");
      }
    }

    // 2. PoissonCounts: 32-bit discrete integer multiplicities
    {
      std::vector<uint32_t> counts(8192);
      for (size_t i = 0; i < counts.size(); ++i) {
        counts[i] = static_cast<uint32_t>((i * 7) % 19);
      }
      auto raw = std::span<const std::byte>(
          reinterpret_cast<const std::byte*>(counts.data()), counts.size() * sizeof(uint32_t));

      spectra8::CompressParams p;
      p.profile = spectra8::Profile::PoissonCounts;
      std::vector<std::byte> comp(spectra8::compress_bound(raw.size()));
      auto r = spectra8::compress(raw, comp, p);
      check(r.ok(), "crystal_profile: PoissonCounts compress");
      if (r.ok()) {
        std::vector<std::byte> out(raw.size());
        auto d = spectra8::decompress(
            std::span<const std::byte>(comp.data(), r.value), out);
        check(d.ok() && d.value == raw.size() &&
                  std::memcmp(out.data(), raw.data(), raw.size()) == 0,
              "crystal_profile: PoissonCounts roundtrip");
        check(r.value * 2 < raw.size(), "crystal_profile: PoissonCounts >2x ratio");
      }
    }

    // 3. IntMonotonic: 32-bit monotonically increasing event IDs
    {
      std::vector<uint32_t> event_ids(8192);
      uint32_t id = 100000;
      for (size_t i = 0; i < event_ids.size(); ++i) {
        if (i % 3 != 0) ++id;
        event_ids[i] = id;
      }
      auto raw = std::span<const std::byte>(
          reinterpret_cast<const std::byte*>(event_ids.data()), event_ids.size() * sizeof(uint32_t));

      spectra8::CompressParams p;
      p.profile = spectra8::Profile::IntMonotonic;
      std::vector<std::byte> comp(spectra8::compress_bound(raw.size()));
      auto r = spectra8::compress(raw, comp, p);
      check(r.ok(), "crystal_profile: IntMonotonic compress");
      if (r.ok()) {
        std::vector<std::byte> out(raw.size());
        auto d = spectra8::decompress(
            std::span<const std::byte>(comp.data(), r.value), out);
        check(d.ok() && d.value == raw.size() &&
                  std::memcmp(out.data(), raw.data(), raw.size()) == 0,
              "crystal_profile: IntMonotonic roundtrip");
        check(r.value * 3 < raw.size(), "crystal_profile: IntMonotonic >3x ratio");
      }
    }

    // 4. Text profile
    {
      std::string text_str;
      for (int i = 0; i < 500; ++i) {
        text_str += "{\"event\": " + std::to_string(i) + ", \"pt\": 42.5, \"name\": \"electron\"}\n";
      }
      auto raw = std::span<const std::byte>(
          reinterpret_cast<const std::byte*>(text_str.data()), text_str.size());

      spectra8::CompressParams p;
      p.profile = spectra8::Profile::Text;
      std::vector<std::byte> comp(spectra8::compress_bound(raw.size()));
      auto r = spectra8::compress(raw, comp, p);
      check(r.ok(), "crystal_profile: Text compress");
      if (r.ok()) {
        std::vector<std::byte> out(raw.size());
        auto d = spectra8::decompress(
            std::span<const std::byte>(comp.data(), r.value), out);
        check(d.ok() && d.value == raw.size() &&
                  std::memcmp(out.data(), raw.data(), raw.size()) == 0,
              "crystal_profile: Text roundtrip");
      }
    }

    // 5. Robustness: random data with profiles
    {
      std::vector<uint8_t> rnd(4096);
      uint32_t x = 987654321;
      for (auto& b : rnd) {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        b = static_cast<uint8_t>(x);
      }
      auto raw = std::span<const std::byte>(
          reinterpret_cast<const std::byte*>(rnd.data()), rnd.size());
      for (auto prof : {spectra8::Profile::FloatKinematics, spectra8::Profile::PoissonCounts,
                        spectra8::Profile::IntMonotonic, spectra8::Profile::Text}) {
        spectra8::CompressParams p;
        p.profile = prof;
        std::vector<std::byte> comp(spectra8::compress_bound(raw.size()));
        auto r = spectra8::compress(raw, comp, p);
        check(r.ok(), "crystal_profile: robustness compress");
        if (r.ok()) {
          std::vector<std::byte> out(raw.size());
          auto d = spectra8::decompress(
              std::span<const std::byte>(comp.data(), r.value), out);
          check(d.ok() && d.value == raw.size() &&
                    std::memcmp(out.data(), raw.data(), raw.size()) == 0,
                "crystal_profile: robustness roundtrip");
        }
      }
    }

    // 6. Custom Profile Training and Registration
    {
      std::vector<uint32_t> sample(4096);
      for (size_t i = 0; i < sample.size(); ++i) {
        sample[i] = static_cast<uint32_t>(i % 37);
      }
      auto sample_bytes = std::span<const std::byte>(
          reinterpret_cast<const std::byte*>(sample.data()), sample.size() * sizeof(uint32_t));

      bool reg_ok = spectra8::train_and_register_profile(5, sample_bytes, 4);
      check(reg_ok, "crystal_profile: train_and_register_profile");

      std::vector<uint32_t> test_data(8192);
      for (size_t i = 0; i < test_data.size(); ++i) {
        test_data[i] = static_cast<uint32_t>((i + 5) % 37);
      }
      auto raw = std::span<const std::byte>(
          reinterpret_cast<const std::byte*>(test_data.data()), test_data.size() * sizeof(uint32_t));

      spectra8::CompressParams p;
      p.profile = spectra8::Profile::Custom;
      p.profile_id = 5;
      std::vector<std::byte> comp(spectra8::compress_bound(raw.size()));
      auto r = spectra8::compress(raw, comp, p);
      check(r.ok(), "crystal_profile: custom profile compress");
      if (r.ok()) {
        std::vector<std::byte> out(raw.size());
        auto d = spectra8::decompress(
            std::span<const std::byte>(comp.data(), r.value), out);
        check(d.ok() && d.value == raw.size() &&
                  std::memcmp(out.data(), raw.data(), raw.size()) == 0,
              "crystal_profile: custom profile roundtrip");
      }

      // Test train_profile_model, register_profile_model, export_profile_model
      auto trained_model = spectra8::train_profile_model(sample_bytes, 4);
      check(!trained_model.empty(), "crystal_profile: train_profile_model non-empty");

      auto model_span = std::span<const std::byte>(
          reinterpret_cast<const std::byte*>(trained_model.data()), trained_model.size());
      bool reg_model_ok = spectra8::register_profile_model(6, model_span);
      check(reg_model_ok, "crystal_profile: register_profile_model");

      p.profile_id = 6;
      r = spectra8::compress(raw, comp, p);
      check(r.ok(), "crystal_profile: register_profile_model compress");
      if (r.ok()) {
        std::vector<std::byte> out(raw.size());
        auto d = spectra8::decompress(
            std::span<const std::byte>(comp.data(), r.value), out);
        check(d.ok() && d.value == raw.size() &&
                  std::memcmp(out.data(), raw.data(), raw.size()) == 0,
              "crystal_profile: register_profile_model roundtrip");
      }

      // Test extended profile ID >= 255 (e.g. 300)
      bool reg_ext_ok = spectra8::register_profile_model(300, model_span);
      check(reg_ext_ok, "crystal_profile: register_profile_model extended id (300)");
      p.profile_id = 300;
      r = spectra8::compress(raw, comp, p);
      check(r.ok(), "crystal_profile: extended profile id compress");
      if (r.ok()) {
        std::vector<std::byte> out(raw.size());
        auto d = spectra8::decompress(
            std::span<const std::byte>(comp.data(), r.value), out);
        check(d.ok() && d.value == raw.size() &&
                  std::memcmp(out.data(), raw.data(), raw.size()) == 0,
              "crystal_profile: extended profile id roundtrip");
      }

      // Test export_profile_model
      auto exported = spectra8::export_profile_model(6);
      check(exported == trained_model, "crystal_profile: export_profile_model matches trained");

      // Test C API
      size_t c_model_size = 0;
      int c_train_query = spectra8_train_profile_model(sample_bytes.data(), sample_bytes.size(), 4, nullptr, &c_model_size);
      check(c_train_query == 1 && c_model_size == trained_model.size(), "crystal_profile: C train query size");
      std::vector<uint8_t> c_model_buf(c_model_size);
      int c_train_ok = spectra8_train_profile_model(sample_bytes.data(), sample_bytes.size(), 4, c_model_buf.data(), &c_model_size);
      check(c_train_ok == 1 && c_model_buf == trained_model, "crystal_profile: C train model");
      int c_reg_ok = spectra8_register_profile_model(7, c_model_buf.data(), c_model_buf.size());
      check(c_reg_ok == 1, "crystal_profile: C register model");
      size_t c_export_size = 0;
      int c_exp_query = spectra8_export_profile_model(7, nullptr, &c_export_size);
      check(c_exp_query == 1 && c_export_size == c_model_size, "crystal_profile: C export query size");
    }

    // 6b. Per-page lane modes on profile blocks: a page whose lanes are
    // constant, or leave the profile's support, stays on the profile
    // (constant / raw lanes) instead of escaping to a per-page model.
    {
      std::mt19937 rng(11);
      std::vector<uint32_t> sample(16384);
      for (auto& v : sample) v = static_cast<uint32_t>(std::poisson_distribution<int>(40)(rng));
      auto sample_bytes = std::span<const std::byte>(
          reinterpret_cast<const std::byte*>(sample.data()), sample.size() * sizeof(uint32_t));
      check(spectra8::train_and_register_profile(9, sample_bytes, 4), "crystal_profile: lane modes train");

      spectra8::CompressParams p;
      p.profile = spectra8::Profile::Custom;
      p.profile_id = 9;
      auto roundtrip = [&](const std::vector<uint32_t>& data, const char* what) -> size_t {
        auto raw = std::span<const std::byte>(
            reinterpret_cast<const std::byte*>(data.data()), data.size() * sizeof(uint32_t));
        std::vector<std::byte> comp(spectra8::compress_bound(raw.size()));
        auto r = spectra8::compress(raw, comp, p);
        check(r.ok(), what);
        if (!r.ok()) return 0;
        std::vector<std::byte> out(raw.size());
        auto d = spectra8::decompress(std::span<const std::byte>(comp.data(), r.value), out);
        check(d.ok() && d.value == raw.size() && std::memcmp(out.data(), raw.data(), raw.size()) == 0, what);
        return r.value;
      };
      // Constant high lanes (as in the sample): coded well below the raw size.
      std::vector<uint32_t> like(12000);
      for (auto& v : like) v = static_cast<uint32_t>(std::poisson_distribution<int>(40)(rng));
      size_t s1 = roundtrip(like, "crystal_profile: lane modes, constant lanes");
      check(s1 * 4 < like.size() * sizeof(uint32_t), "crystal_profile: lane modes, constant lanes ratio");
      // Lane 2 outside the support (noise): that lane goes raw, the others
      // keep the profile; the block costs about the raw lane plus the rest.
      std::vector<uint32_t> noisy(12000);
      for (auto& v : noisy)
        v = static_cast<uint32_t>(std::poisson_distribution<int>(40)(rng)) | ((rng() & 0xFFu) << 16);
      size_t s2 = roundtrip(noisy, "crystal_profile: lane modes, raw lane");
      check(s2 > noisy.size() && s2 < noisy.size() * 2, "crystal_profile: lane modes, raw lane size");
      // Every lane constant: no coded lane, a few bytes.
      std::vector<uint32_t> flat(12000, 0x17u);
      size_t s3 = roundtrip(flat, "crystal_profile: lane modes, all constant");
      check(s3 > 0 && s3 < 64, "crystal_profile: lane modes, all constant size");
    }

    // 7. Thread-local Profile API
    {
      check(spectra8::get_thread_profile() == spectra8::Profile::None,
            "crystal_profile: thread profile initial None");
      spectra8::set_thread_profile(spectra8::Profile::IntMonotonic);
      check(spectra8::get_thread_profile() == spectra8::Profile::IntMonotonic,
            "crystal_profile: thread profile set IntMonotonic");

      std::vector<uint32_t> event_ids(4096);
      for (size_t i = 0; i < event_ids.size(); ++i) event_ids[i] = static_cast<uint32_t>(500 + i);
      auto raw = std::span<const std::byte>(
          reinterpret_cast<const std::byte*>(event_ids.data()), event_ids.size() * sizeof(uint32_t));

      spectra8::CompressParams p; // default params, profile resolved from thread-local
      std::vector<std::byte> comp(spectra8::compress_bound(raw.size()));
      auto r = spectra8::compress(raw, comp, p);
      check(r.ok(), "crystal_profile: thread-local profile compress");
      if (r.ok()) {
        std::vector<std::byte> out(raw.size());
        auto d = spectra8::decompress(
            std::span<const std::byte>(comp.data(), r.value), out);
        check(d.ok() && d.value == raw.size() &&
                  std::memcmp(out.data(), raw.data(), raw.size()) == 0,
              "crystal_profile: thread-local profile roundtrip");
      }
      spectra8::set_thread_profile(spectra8::Profile::None);
    }

    // 8. C API integration
    {
      spectra8_set_thread_profile(SPECTRA8_PROFILE_FLOAT_KINEMATICS);
      check(spectra8_get_thread_profile() == SPECTRA8_PROFILE_FLOAT_KINEMATICS,
            "crystal_profile: C thread profile set");

      std::vector<float> floats(4096, 123.456f);
      size_t written = 0;
      std::vector<uint8_t> comp(spectra8_compress_bound(floats.size() * sizeof(float)));
      spectra8_compress_options opts{};
      opts.profile = SPECTRA8_PROFILE_FLOAT_KINEMATICS;

      spectra8_status st = spectra8_compress_with_options(
          floats.data(), floats.size() * sizeof(float),
          comp.data(), comp.size(), &opts, &written);
      check(st == SPECTRA8_OK, "crystal_profile: C compress_with_options");

      std::vector<float> out(floats.size(), 0.0f);
      size_t decomp_written = 0;
      st = spectra8_decompress(comp.data(), written, out.data(), out.size() * sizeof(float), &decomp_written);
      check(st == SPECTRA8_OK && decomp_written == floats.size() * sizeof(float) &&
                std::memcmp(out.data(), floats.data(), floats.size() * sizeof(float)) == 0,
            "crystal_profile: C decompress roundtrip");

      spectra8_set_thread_profile(SPECTRA8_PROFILE_NONE);
    }
    std::fprintf(stderr, "  [crystal_profile] all pre-trained static profiles & API checks passed\n");
  }

  // Auto: with an unreachable gain bar, keep the fastest available decoder;
  // with gain=0, take the absolute smallest among successful trials.
  {
    check(spectra8::codec_available(spectra8::Codec::Auto),
          "codec: auto available");
    // Highly compressible: slower codecs usually win size by a wide margin.
    std::vector<uint8_t> zeros(1 << 16, 0);
    for (size_t i = 0; i < zeros.size(); ++i)
      zeros[i] = static_cast<uint8_t>((i / 64) & 0xff);
    auto zsrc = bytes(zeros);

    auto compress_auto = [&](int gain) {
      spectra8::CompressParams p;
      p.codec = spectra8::Codec::Auto;
      p.level = 5;
      p.auto_min_gain_pct = gain;
      std::vector<std::byte> comp(
          spectra8::compress_bound(zsrc.size(), spectra8::Codec::Auto));
      auto r = spectra8::compress(zsrc, comp, p);
      check(r.ok(), "codec: auto gain compress ok");
      if (!r.ok()) return std::vector<std::byte>{};
      comp.resize(r.value);
      return comp;
    };

    auto high = compress_auto(90);
    auto low = compress_auto(0);
    if (!high.empty() && !low.empty()) {
      check(low.size() <= high.size(),
            "codec: auto gain=0 not larger than gain=90");
      std::vector<std::byte> out(zeros.size());
      auto d = spectra8::decompress(high, out);
      check(d.ok() && d.value == zeros.size() &&
                std::memcmp(out.data(), zeros.data(), zeros.size()) == 0,
            "codec: auto gain=90 roundtrip");
      d = spectra8::decompress(low, out);
      check(d.ok() && d.value == zeros.size() &&
                std::memcmp(out.data(), zeros.data(), zeros.size()) == 0,
            "codec: auto gain=0 roundtrip");
    }

    // auto_max_level: roundtrip + C ABI; usually not larger than level=5 @ g0.
    {
      spectra8::CompressParams p;
      p.codec = spectra8::Codec::Auto;
      p.level = 5;
      p.auto_min_gain_pct = 0;
      p.auto_max_level = true;
      std::vector<std::byte> comp(
          spectra8::compress_bound(zsrc.size(), spectra8::Codec::Auto));
      auto r = spectra8::compress(zsrc, comp, p);
      check(r.ok(), "codec: auto_max_level compress ok");
      if (r.ok()) {
        comp.resize(r.value);
        if (!low.empty())
          check(comp.size() <= low.size(),
                "codec: auto_max_level not larger than L5 g0");
        std::vector<std::byte> out(zeros.size());
        auto d = spectra8::decompress(comp, out);
        check(d.ok() && d.value == zeros.size() &&
                  std::memcmp(out.data(), zeros.data(), zeros.size()) == 0,
              "codec: auto_max_level roundtrip");
      }

      spectra8_compress_options opt{};
      opt.codec = SPECTRA8_CODEC_AUTO;
      opt.level = 5;
      opt.auto_min_gain_pct = 0;
      opt.auto_max_level = 1;
      opt.filter_fallback = 1;
      opt.filter_rle = 1;
      opt.filter_dict = 1;
      std::vector<std::byte> ccomp(
          spectra8_compress_bound_codec(zsrc.size(), SPECTRA8_CODEC_AUTO));
      size_t written = 0;
      const auto st = spectra8_compress_with_options(
          zsrc.data(), zsrc.size(), ccomp.data(), ccomp.size(), &opt, &written);
      check(st == SPECTRA8_OK && written > 0, "codec: auto_max_level C ABI");
      if (st == SPECTRA8_OK) {
        std::vector<std::byte> out(zeros.size());
        size_t dout = 0;
        const auto dst = spectra8_decompress(ccomp.data(), written, out.data(),
                                              out.size(), &dout);
        check(dst == SPECTRA8_OK && dout == zeros.size() &&
                  std::memcmp(out.data(), zeros.data(), zeros.size()) == 0,
              "codec: auto_max_level C ABI roundtrip");
      }
    }

    spectra8::reset_compress_stats();
    check(spectra8::get_compress_stats().auto_selections == 0,
          "codec: auto stats reset");
    for (int i = 0; i < 3; ++i) (void)compress_auto(1);
    const auto ast = spectra8::get_compress_stats();
    check(ast.auto_selections == 3, "codec: auto stats selections");
    std::uint64_t hits = 0;
    for (std::size_t i = 0; i < 9; ++i) hits += ast.auto_codec_hits[i];
    check(hits == 3, "codec: auto stats hits sum");
    const auto arep = spectra8::format_auto_codec_stats(ast);
    check(arep.find("auto codec selection") != std::string::npos,
          "codec: auto stats format");
    check(arep.find("%") != std::string::npos, "codec: auto stats percent");
    spectra8::reset_compress_stats();

    // auto_codecs: the race only visits codecs in the mask. With gain 0 (take
    // the smallest) over a few hundred noisy pages the winner mix is spread
    // out; under kAutoCodecsRoot every hit must land on zstd/beam/crystal.
    {
      std::mt19937 rng(0x5eed);
      std::vector<std::vector<std::byte>> pages;
      for (int k = 0; k < 6; ++k) {
        std::vector<uint8_t> pg(48 * 1024);
        for (size_t i = 0; i + 4 <= pg.size(); i += 4) {
          const uint32_t v = 100000u + (rng() % (1u << (6 + 2 * k)));
          std::memcpy(&pg[i], &v, 4);
        }
        pages.push_back(bytes(pg));
      }
      auto race = [&](spectra8::CodecMask mask, bool filters) {
        spectra8::reset_compress_stats();
        for (const auto& pg : pages) {
          spectra8::CompressParams p;
          p.codec = spectra8::Codec::Auto;
          p.level = 3;
          p.auto_min_gain_pct = 0;
          p.auto_codecs = mask;
          p.filters = filters;
          std::vector<std::byte> comp(
              spectra8::compress_bound(pg.size(), spectra8::Codec::Auto));
          auto r = spectra8::compress(pg, comp, p);
          check(r.ok(), "codec: auto_codecs compress ok");
          if (!r.ok()) continue;
          std::vector<std::byte> out(pg.size());
          auto d = spectra8::decompress(std::span<const std::byte>(comp.data(), r.value), out);
          check(d.ok() && d.value == pg.size() &&
                    std::memcmp(out.data(), pg.data(), pg.size()) == 0,
                "codec: auto_codecs roundtrip");
        }
        return spectra8::get_compress_stats();
      };
      const auto root = race(spectra8::kAutoCodecsRoot, true);
      std::uint64_t outside = 0, inside = 0;
      for (std::size_t i = 0; i < 9; ++i) {
        const auto bit = spectra8::CodecMask{1} << i;
        if (spectra8::kAutoCodecsRoot & bit) inside += root.auto_codec_hits[i];
        else outside += root.auto_codec_hits[i];
      }
      check(inside == pages.size() && outside == 0,
            "codec: auto_codecs root mask only hits zstd/beam/crystal");

      // A single-codec mask degenerates to that codec.
      const auto only_beam = race(spectra8::codec_bit(spectra8::Codec::Beam), false);
      check(only_beam.auto_codec_hits[static_cast<int>(spectra8::Codec::Beam)] == pages.size(),
            "codec: auto_codecs beam-only mask");

      // A mask with no available candidate is rejected.
      {
        spectra8::CompressParams p;
        p.codec = spectra8::Codec::Auto;
        p.auto_codecs = spectra8::codec_bit(spectra8::Codec::Oracle);
        std::vector<std::byte> comp(spectra8::compress_bound(pages[0].size()));
        auto r = spectra8::compress(pages[0], comp, p);
        check(!r.ok() && r.status == spectra8::Status::InvalidArgument,
              "codec: auto_codecs empty mask -> InvalidArgument");
      }

      // C ABI: SPECTRA8_AUTO_CODECS_ROOT behaves like the C++ constant.
      {
        spectra8::reset_compress_stats();
        spectra8_compress_options opt{};
        opt.codec = SPECTRA8_CODEC_AUTO;
        opt.level = 3;
        opt.auto_codecs = SPECTRA8_AUTO_CODECS_ROOT;
        opt.filter_fallback = 1;
        opt.filter_rle = 1;
        opt.filter_dict = 1;
        check(SPECTRA8_AUTO_CODECS_ROOT == spectra8::kAutoCodecsRoot &&
                  SPECTRA8_AUTO_CODECS_ALL == spectra8::kAutoCodecsAll &&
                  SPECTRA8_FILTERS_ALL == spectra8::kFiltersAll &&
                  SPECTRA8_FILTERS_32 == spectra8::kFilters32 &&
                  SPECTRA8_FILTERS_64 == spectra8::kFilters64 &&
                  SPECTRA8_FILTERS_16 == spectra8::kFilters16,
              "codec: C mask constants match C++");
        for (const auto& pg : pages) {
          std::vector<std::byte> ccomp(
              spectra8_compress_bound_codec(pg.size(), SPECTRA8_CODEC_AUTO));
          size_t written = 0;
          const auto st = spectra8_compress_with_options(
              pg.data(), pg.size(), ccomp.data(), ccomp.size(), &opt, &written);
          check(st == SPECTRA8_OK && written > 0, "codec: auto_codecs C ABI");
        }
        const auto cst = spectra8::get_compress_stats();
        std::uint64_t c_outside = 0;
        for (std::size_t i = 0; i < 9; ++i)
          if (!(spectra8::kAutoCodecsRoot & (spectra8::CodecMask{1} << i)))
            c_outside += cst.auto_codec_hits[i];
        check(c_outside == 0, "codec: auto_codecs C ABI root mask honoured");
      }
      spectra8::reset_compress_stats();
    }

    // filter_mask: detection may only nominate transforms in the mask. A
    // 4-byte column that detects as a 32-bit transform must pick a 64-bit
    // one (or none) under kFilters64, and exactly the forced id under a
    // single-bit mask.
    {
      std::vector<uint8_t> col(64 * 1024);
      std::mt19937 rng(0xf11);
      for (size_t i = 0; i + 4 <= col.size(); i += 4) {
        const uint32_t v = 5000000u + (rng() % 300u);
        std::memcpy(&col[i], &v, 4);
      }
      using spectra8::detail::Filter;
      const Filter any = spectra8::detail::detect_filter(col.data(), col.size());
      check(any != Filter::None, "filter_mask: baseline detects a transform");
      const Filter w64 = spectra8::detail::detect_filter(col.data(), col.size(), false,
                                                          spectra8::kFilters64);
      check(w64 == Filter::None || w64 == Filter::Shuffle64 || w64 == Filter::Delta64 ||
                w64 == Filter::ZigzagDelta64 || w64 == Filter::DeltaShuffle64 ||
                w64 == Filter::ZigzagShuffle64 || w64 == Filter::BitShuffle64 ||
                w64 == Filter::XorShuffle64 || w64 == Filter::Delta2Shuffle64,
            "filter_mask: 64-bit mask yields a 64-bit transform or none");
      const Filter forced = spectra8::detail::detect_filter(
          col.data(), col.size(), false, spectra8::filter_bit(spectra8::FilterId::Shuffle32));
      check(forced == Filter::Shuffle32 || forced == Filter::None,
            "filter_mask: single-bit mask forces that id");
      check(spectra8::detail::detect_filter(col.data(), col.size(), false,
                                             spectra8::filter_bit(spectra8::FilterId::None)) ==
                Filter::None,
            "filter_mask: none-only mask disables detection");
      check(spectra8::detail::detect_dict_width(col.data(), col.size()) == 4,
            "filter_mask: dict detects w4");
      check(spectra8::detail::detect_dict_width(col.data(), col.size(), false,
                                                 spectra8::kFilters64) == 0,
            "filter_mask: dict32 blocked by 64-bit mask");

      // End to end through every Spectra8 flavour + an external: the frame must
      // roundtrip and, under the shuffle32-only mask, use no other filter.
      auto csrc = bytes(col);
      for (spectra8::Codec c : {spectra8::Codec::Beam, spectra8::Codec::Prism,
                                 spectra8::Codec::Crystal, spectra8::Codec::Flash,
                                 spectra8::Codec::Zstd}) {
        if (!spectra8::codec_available(c)) continue;
        spectra8::CompressParams p;
        p.codec = c;
        p.level = 3;
        p.filters = true;
        p.filter_mask = spectra8::filter_bit(spectra8::FilterId::Shuffle32);
        std::vector<std::byte> comp(spectra8::compress_bound(csrc.size(), c));
        auto r = spectra8::compress(csrc, comp, p);
        check(r.ok(), "filter_mask: compress ok");
        if (!r.ok()) continue;
        std::vector<std::byte> out(csrc.size());
        auto d = spectra8::decompress(std::span<const std::byte>(comp.data(), r.value), out);
        check(d.ok() && d.value == csrc.size() &&
                  std::memcmp(out.data(), csrc.data(), csrc.size()) == 0,
              "filter_mask: roundtrip");
      }
    }

    // Regression: ~1 MiB float32 pages (Jet_pt-like). A buggy bzip3 path once
    // returned a ~29-byte frame that "decompressed" to 0 bytes; Auto g0%
    // selected it as the size winner. Must roundtrip with filters on/off.
    {
      const size_t n = 1u << 20;
      std::vector<uint8_t> jet(n);
      for (size_t i = 0; i < n; i += 4) {
        float v = 30.f + float((i / 4) % 50) + float((i / 4) % 97) * 0.01f;
        std::memcpy(jet.data() + i, &v, sizeof(v));
      }
      auto jsrc = bytes(jet);
      for (bool filt : {false, true}) {
        for (int gain : {0, 1}) {
          spectra8::CompressParams p;
          p.codec = spectra8::Codec::Auto;
          p.level = 5;
          p.auto_min_gain_pct = gain;
          p.filters = filt;
          std::vector<std::byte> comp(
              spectra8::compress_bound(jsrc.size(), spectra8::Codec::Auto));
          auto r = spectra8::compress(jsrc, comp, p);
          check(r.ok(), "codec: auto jet_pt compress");
          if (!r.ok()) continue;
          comp.resize(r.value);
          // Must not look like a "win" of a few dozen bytes on 1 MiB input.
          check(r.value > 256, "codec: auto jet_pt not tiny corrupt frame");
          auto cs = spectra8::get_frame_content_size(comp);
          check(cs && *cs == jet.size(), "codec: auto jet_pt content size");
          std::vector<std::byte> out(jet.size());
          auto d = spectra8::decompress(comp, out);
          check(d.ok() && d.value == jet.size() &&
                    std::memcmp(out.data(), jet.data(), jet.size()) == 0,
                "codec: auto jet_pt roundtrip");
        }
      }
    }
  }

  // Tiny / high-entropy payloads (RNTuple-sized pages) must not fail bzip3.
  if (spectra8::codec_available(spectra8::Codec::Bzip3)) {
    auto roundtrip_small = [&](size_t n, const char* tag) {
      std::vector<uint8_t> raw(n);
      for (size_t i = 0; i < raw.size(); ++i)
        raw[i] = static_cast<uint8_t>((i * 17) ^ (i >> 1) ^ 0xA5);
      auto src = bytes(raw);
      spectra8::CompressParams p;
      p.codec = spectra8::Codec::Bzip3;
      p.level = 5;
      std::vector<std::byte> comp(spectra8::compress_bound(src.size(), p.codec));
      auto r = spectra8::compress(src, comp, p);
      check(r.ok(), std::string("codec: bzip3 small compress ok (") + tag + ")");
      if (!r.ok()) return;
      comp.resize(r.value);
      std::vector<std::byte> out(raw.size());
      auto d = spectra8::decompress(comp, out);
      check(d.ok() && d.value == raw.size() &&
                std::memcmp(out.data(), raw.data(), raw.size()) == 0,
            std::string("codec: bzip3 small roundtrip (") + tag + ")");
    };
    roundtrip_small(256, "256B");
    roundtrip_small(4096, "4KiB");

    std::vector<uint8_t> raw(512);
    for (size_t i = 0; i < raw.size(); ++i)
      raw[i] = static_cast<uint8_t>(i * 3);
    auto src = bytes(raw);
    spectra8::CompressParams p;
    p.codec = spectra8::Codec::Bzip3;
    p.level = 5;
    p.filters = true;
    p.filter_fallback = true;
    std::vector<std::byte> comp(spectra8::compress_bound(src.size(), p.codec));
    auto r = spectra8::compress(src, comp, p);
    check(r.ok(), "codec: bzip3+filters small compress ok");
    if (r.ok()) {
      comp.resize(r.value);
      std::vector<std::byte> out(raw.size());
      auto d = spectra8::decompress(comp, out);
      check(d.ok() && d.value == raw.size(), "codec: bzip3+filters small decompress size");
      check(std::memcmp(out.data(), raw.data(), raw.size()) == 0,
            "codec: bzip3+filters small roundtrip");
    }

    // ROOT passes a zip buffer no larger than the uncompressed page. When the
    // L4EF filtered frame would exceed that slot, fall back to raw bzip3/identity.
    {
      std::vector<std::uint32_t> cols(96);
      for (size_t i = 0; i < cols.size(); ++i) cols[i] = static_cast<std::uint32_t>(i);
      std::vector<std::byte> src(cols.size() * sizeof(std::uint32_t));
      std::memcpy(src.data(), cols.data(), src.size());
      spectra8::CompressParams tight;
      tight.codec = spectra8::Codec::Bzip3;
      tight.level = 5;
      tight.filters = true;
      tight.filter_fallback = true;
      std::vector<std::byte> comp(src.size());
      auto r = spectra8::compress(src, comp, tight);
      check(r.ok(), "codec: bzip3 tight dst with filters ok");
      check(r.value <= src.size(), "codec: bzip3 tight dst fits slot");
      if (r.ok()) {
        comp.resize(r.value);
        std::vector<std::byte> out(src.size());
        auto d = spectra8::decompress(comp, out);
        check(d.ok() && d.value == src.size(), "codec: bzip3 tight dst decompress size");
        check(std::memcmp(out.data(), src.data(), src.size()) == 0,
              "codec: bzip3 tight dst roundtrip");
      }
    }
  }

  // Filters with external codecs use the L4EF wrapper and roundtrip.
  if (spectra8::codec_available(spectra8::Codec::Zstd)) {
    std::vector<uint8_t> col(64 * 1024);
    uint32_t v = 1000;
    for (size_t i = 0; i + 4 <= col.size(); i += 4) {
      v += 3;
      std::memcpy(&col[i], &v, 4);
    }
    auto src = bytes(col);
    spectra8::CompressParams p;
    p.codec = spectra8::Codec::Zstd;
    p.level = 5;
    p.filters = true;
    p.filter_fallback = false;
    std::vector<std::byte> comp(spectra8::compress_bound(src.size(), p.codec));
    auto r = spectra8::compress(src, comp, p);
    check(r.ok(), "codec: zstd+filters compress ok");
    if (r.ok()) {
      comp.resize(r.value);
      check(spectra8::detail::looks_like_l4ef(comp), "codec: zstd+filters uses L4EF");
      std::vector<std::byte> out(col.size());
      auto d = spectra8::decompress(comp, out);
      check(d.ok() && d.value == col.size(), "codec: zstd+filters decompress size");
      check(std::memcmp(out.data(), col.data(), col.size()) == 0,
            "codec: zstd+filters roundtrip");
    }
  }

  if (spectra8::codec_available(spectra8::Codec::Beam)) {
    spectra8::reset_compress_stats();
    auto make_ramp_page = [](size_t nwords) {
      std::vector<std::byte> page(nwords * sizeof(std::uint32_t));
      for (size_t i = 0; i < nwords; ++i) {
        const std::uint32_t v = static_cast<std::uint32_t>(i);
        std::memcpy(page.data() + i * sizeof(v), &v, sizeof(v));
      }
      return page;
    };
    spectra8::CompressParams p;
    p.codec = spectra8::Codec::Beam;
    p.level = 5;
    p.filters = true;
    for (int i = 0; i < 10; ++i) {
      const auto a = make_ramp_page(32);
      std::vector<std::byte> comp(spectra8::compress_bound(a.size()));
      (void)spectra8::compress(a, comp, p);
    }
    for (int i = 0; i < 5; ++i) {
      const auto b = make_ramp_page(128);
      std::vector<std::byte> comp(spectra8::compress_bound(b.size()));
      (void)spectra8::compress(b, comp, p);
    }
    const auto stats = spectra8::get_compress_stats();
    check(stats.num_pages == 15, "stats: page count");
    check(stats.uncompressed_bytes == 10ull * 128 + 5ull * 512, "stats: uncompressed total");
    check(stats.compressed_bytes > 0, "stats: compressed total");
    check(stats.wire_spectra8_pages == 15, "stats: Spectra8 wire pages");
    std::uint64_t detected = 0;
    for (std::size_t i = 0; i < static_cast<std::size_t>(spectra8::StatsFilterKind::Count); ++i)
      detected += stats.detected_filter_hits[i];
    check(detected > 0, "stats: detected filter histogram");
    check(stats.plain_pages + stats.no_transform_pages + stats.raw_won_pages +
              stats.filtered_won_pages + stats.stored_fallback_pages ==
          stats.num_pages,
          "stats: page outcomes sum");
    const auto report = spectra8::format_compress_stats(stats);
    check(report.find("filters detected (autodetect") != std::string::npos,
          "stats: detected filter section");
    check(report.find("page outcomes") != std::string::npos, "stats: page outcomes section");
    check(report.find("wire format:") != std::string::npos, "stats: wire format section");
    check(report.find("avg page size:") != std::string::npos, "stats: formatted report");
    spectra8::reset_compress_stats();
    check(spectra8::get_compress_stats().num_pages == 0, "stats: reset");
  }

  // decompress_block is Spectra8-only.
  if (spectra8::codec_available(spectra8::Codec::Zstd)) {
    spectra8::CompressParams p;
    p.codec = spectra8::Codec::Zstd;
    std::vector<std::byte> comp(
        spectra8::compress_bound(src.size(), spectra8::Codec::Zstd));
    auto r = spectra8::compress(src, comp, p);
    check(r.ok(), "codec: zstd for block-reject setup");
    if (r.ok()) {
      comp.resize(r.value);
      std::vector<std::byte> out(raw.size());
      auto db = spectra8::decompress_block(comp, 0, out);
      check(db.status == spectra8::Status::InvalidArgument,
            "codec: decompress_block rejects zstd");
    }
  }
}

void test_bwt_mode() {
  std::mt19937_64 rng(2024);

  // Compress with the BWT block mode, verify single- and multi-threaded
  // roundtrips, and return the compressed size.
  auto run = [&](const std::vector<uint8_t>& in, int level) -> size_t {
    spectra8::CompressParams p;
    p.level = level;
    p.codec = spectra8::Codec::Prism;
    auto src = bytes(in);
    std::vector<std::byte> comp(spectra8::compress_bound(src.size()));
    auto r = spectra8::compress(src, comp, p);
    check(r.ok(), "bwt: compress ok");
    if (!r.ok()) return 0;
    check(r.value <= spectra8::compress_bound(src.size()),
          "bwt: within compress_bound");
    comp.resize(r.value);
    std::vector<std::byte> out(in.size());
    auto d = spectra8::decompress(comp, out);
    check(d.ok() && d.value == in.size(), "bwt: decompress size");
    check(in.empty() || std::memcmp(out.data(), in.data(), in.size()) == 0,
          "bwt: roundtrip bytes");
    // Multithreaded decode path must produce identical bytes.
    std::vector<std::byte> out2(in.size());
    spectra8::DecompressParams dp{0};
    auto d2 = spectra8::decompress(comp, out2, dp);
    check(d2.ok() && d2.value == in.size() &&
              (in.empty() ||
               std::memcmp(out2.data(), in.data(), in.size()) == 0),
          "bwt: MT roundtrip bytes");
    return r.value;
  };

  // Edge sizes: empty, tiny (below the BWT min-block fallback), boundaries, and
  // multi-block. Use low-entropy content so BWT engages on the larger sizes.
  for (size_t sz : {0u, 1u, 2u, 63u, 64u, 65u, 1000u, 70000u, 300000u}) {
    std::vector<uint8_t> d(sz);
    for (size_t i = 0; i < sz; ++i) d[i] = static_cast<uint8_t>(rng() % 5);
    (void)run(d, 9);
  }

  // All-equal block must roundtrip (memset fast path, not BWT).
  {
    std::vector<uint8_t> d(200000, 0x5A);
    (void)run(d, 9);
  }

  // Pure-random data must still roundtrip (stored fast path / no-bloat).
  {
    std::vector<uint8_t> d(400000);
    for (auto& b : d) b = static_cast<uint8_t>(rng());
    size_t on = run(d, 9);
    check(on <= d.size() + spectra8::compress_bound(0),
          "bwt: random not bloated");
  }

  // 128 KiB columnar-ish block: BWT roundtrip (exercises small-block + mode 2).
  {
    std::vector<uint8_t> col(128 * 1024);
    for (size_t i = 0; i < col.size(); ++i)
      col[i] = static_cast<uint8_t>((i / 4) & 255);
    run(col, 9);
  }

  // Repetitive text: BWT must roundtrip and should compress meaningfully.
  {
    std::vector<uint8_t> d;
    const char* w[] = {"the ", "quick ", "brown ", "fox ", "jumps ", "over ",
                       "lazy ", "dog "};
    for (int i = 0; i < 60000; ++i) {
      const char* s = w[rng() % 8];
      d.insert(d.end(), reinterpret_cast<const uint8_t*>(s),
               reinterpret_cast<const uint8_t*>(s) + std::strlen(s));
    }
    size_t on = run(d, 9);
    check(on < d.size() / 2, "bwt: text compresses < 0.5x");
  }

  // Highly repetitive, adversarial-for-SA input (long identical runs).
  {
    std::vector<uint8_t> d(300000, 0);
    for (size_t i = 0; i < d.size(); ++i)
      d[i] = static_cast<uint8_t>((i / 1000) & 1 ? 'A' : 'B');
    (void)run(d, 9);
  }

  // Determinism: single- vs multi-threaded compression must be byte-identical.
  {
    std::vector<uint8_t> in(500000);
    size_t i = 0;
    while (i < in.size()) {
      const char* s = "lorem ipsum dolor sit amet ";
      for (size_t k = 0; s[k] && i < in.size(); ++k)
        in[i++] = static_cast<uint8_t>(s[k]);
    }
    auto src = bytes(in);
    spectra8::CompressParams p1;
    p1.level = 9;
    p1.codec = spectra8::Codec::Prism;
    p1.threads = 1;
    spectra8::CompressParams pN = p1;
    pN.threads = 0;
    std::vector<std::byte> c1(spectra8::compress_bound(in.size()));
    std::vector<std::byte> cN(spectra8::compress_bound(in.size()));
    auto r1 = spectra8::compress(src, c1, p1);
    auto rN = spectra8::compress(src, cN, pN);
    check(r1.ok() && rN.ok() && r1.value == rN.value &&
              std::memcmp(c1.data(), cN.data(), r1.value) == 0,
          "bwt: 1T and NT produce identical bytes");
  }

  // Multiple BWT blocks + multi-segment symbol coding: force a small block size
  // so a larger input splits into several BWT blocks, each big enough (> 64 KiB
  // segment size) to exercise the per-segment table path and RLE0 expansion.
  {
    std::vector<uint8_t> in(900000);
    size_t i = 0;
    while (i < in.size()) {
      const char* s = "the quick brown fox jumps over the lazy dog ";
      for (size_t k = 0; s[k] && i < in.size(); ++k)
        in[i++] = static_cast<uint8_t>(s[k]);
    }
    auto run_blk = [&](int block_log, int threads) {
      spectra8::CompressParams p;
      p.level = 9;
      p.codec = spectra8::Codec::Prism;
      p.block_log = block_log;
      p.threads = threads;
      auto src = bytes(in);
      std::vector<std::byte> comp(spectra8::compress_bound(in.size()));
      auto r = spectra8::compress(src, comp, p);
      check(r.ok(), "bwt: multiblock compress ok");
      if (!r.ok()) return;
      comp.resize(r.value);
      std::vector<std::byte> out(in.size());
      spectra8::DecompressParams dp{threads};
      auto d = spectra8::decompress(comp, out, dp);
      check(d.ok() && d.value == in.size() &&
                std::memcmp(out.data(), in.data(), in.size()) == 0,
            "bwt: multiblock roundtrip (blk=" + std::to_string(block_log) + ")");
    };
    run_blk(17, 1);  // 128 KiB blocks -> several blocks, each multi-segment
    run_blk(17, 0);  // same, multithreaded
    run_blk(20, 0);  // 1 MiB blocks
  }
}

void test_bwt_wide_block() {
  // Input > 16 MiB in a single BWT block exercises the wide LF table path
  // (u32 link + u8 sym) and 16-chain parallel inverse.
  const size_t sz = 18 * 1024 * 1024;  // 18 MiB
  std::vector<uint8_t> in(sz);
  const char* phrase = "LHC Run-4 high-luminosity computing with Prism BWT context mixer wide block! ";
  const size_t plen = std::strlen(phrase);
  for (size_t i = 0; i < sz; ++i) in[i] = static_cast<uint8_t>(phrase[i % plen]);

  spectra8::CompressParams p;
  p.codec = spectra8::Codec::Prism;
  p.level = 9;
  p.block_log = 25;  // 32 MiB block size -> all 18 MiB fit in a single block
  auto src = bytes(in);
  std::vector<std::byte> comp(spectra8::compress_bound(sz));
  auto cr = spectra8::compress(src, comp, p);
  check(cr.ok(), "bwt wide: compress 18 MiB ok");
  if (!cr.ok()) return;
  comp.resize(cr.value);

  std::vector<std::byte> restored(sz);
  auto dr = spectra8::decompress(comp, restored);
  check(dr.ok(), "bwt wide: decompress 18 MiB ok");
  check(dr.value == sz, "bwt wide: length match");
  check(std::memcmp(restored.data(), in.data(), sz) == 0,
        "bwt wide: bit-exact roundtrip (single >16MB BWT block)");
}

void test_prism_verdict_bin_i32() {
  // Regression test: LZP pre-filter in Prism level 5 on dict32-filtered bin-i32
  // previously emitted an escape byte 255 for match marker 0xF2 even when val < 0
  // (hash table miss), desynchronizing the decompressor.
  const size_t bytes = 128 * 1024;
  std::vector<std::byte> d(bytes);
  std::mt19937_64 rng(0xabcdef ^ 7);
  std::binomial_distribution<long long> dist(50000, 0.50);
  size_t i = 0;
  while (i + 4 <= bytes) {
    int32_t v = static_cast<int32_t>(dist(rng));
    std::memcpy(&d[i], &v, 4);
    i += 4;
  }
  const auto* in = reinterpret_cast<const spectra8::detail::u8*>(d.data());
  std::vector<spectra8::detail::u8> filt;
  auto framed = spectra8::detail::detect_framed_filter(in, bytes, false, ~spectra8::detail::u32{0}, true);
  bool ok = spectra8::detail::framed_filter_fwd(framed, in, bytes, filt);
  check(ok, "dict32 fwd ok");

  spectra8::CompressParams p;
  p.codec = spectra8::Codec::Prism;
  p.level = 5;
  p.threads = 1;
  std::vector<std::byte> comp(spectra8::compress_bound(filt.size()));
  auto cr = spectra8::compress(
      std::span<const std::byte>(reinterpret_cast<const std::byte*>(filt.data()), filt.size()),
      comp, p);
  check(cr.ok(), "prism l5 compress ok");
  if (!cr.ok()) return;
  comp.resize(cr.value);

  std::vector<std::byte> dec(filt.size());
  auto dr = spectra8::decompress(comp, dec);
  check(dr.ok(), "prism l5 decompress ok");
  if (dr.ok()) {
    check(dr.value == filt.size(), "prism l5 length match");
    check(std::memcmp(dec.data(), filt.data(), filt.size()) == 0, "prism l5 memcmp match");
  }
}

void test_bwt_ultra() {
  using spectra8::detail::u8;
  using spectra8::detail::usize;
  using spectra8::detail::bwt_ultra_encode;
  using spectra8::detail::bwt_ultra_decode;

  // 1. Direct unit test: encode + decode on structured text
  {
    std::string text;
    for (int i = 0; i < 200; ++i) {
      text += "Wikipedia: The Burrows-Wheeler transform groups characters with common contexts. ";
      text += "Context mixing predicts next bits using multi-order probability tables and SSE. ";
      text += "Prism ultra achieves state-of-the-art compression ratios with fast parallel decode.\n";
    }
    const auto* src = reinterpret_cast<const u8*>(text.data());
    const usize n = text.size();
    std::vector<u8> enc;
    check(bwt_ultra_encode(src, n, enc), "bwt_ultra_encode ok");
    check(enc.size() < n, "bwt_ultra_encode compressed");

    std::vector<u8> dec(n, 0);
    auto st = bwt_ultra_decode(enc.data(), enc.size(), dec.data(), n);
    check(st == spectra8::Status::Ok, "bwt_ultra_decode ok");
    check(std::memcmp(src, dec.data(), n) == 0, "bwt_ultra roundtrip bit-exact");

    // Corruption handling
    std::vector<u8> bad = enc;
    bad[10] ^= 0xFF;  // corrupt present mask / counts
    st = bwt_ultra_decode(bad.data(), bad.size(), dec.data(), n);
    check(st == spectra8::Status::CorruptInput, "bwt_ultra corrupt bit returns CorruptInput");

    // Truncated input
    st = bwt_ultra_decode(enc.data(), enc.size() - 8, dec.data(), n);
    check(st == spectra8::Status::CorruptInput, "bwt_ultra truncated returns CorruptInput");
  }

  // 2. Full pipeline test: Codec::Prism level 11 (Ultra)
  {
    std::string text;
    for (int i = 0; i < 100; ++i) {
      text += "Spectra8 compression pipeline test with Prism Ultra mode enabled. ";
      text += "Text input auto-detects structured features and applies context mixing.\n";
    }
    std::vector<std::byte> csrc(text.size());
    std::memcpy(csrc.data(), text.data(), text.size());

    spectra8::CompressParams p;
    p.codec = spectra8::Codec::Prism;
    p.level = 11;
    p.threads = 1;

    std::vector<std::byte> comp(spectra8::compress_bound(csrc.size(), p.codec));
    auto cr = spectra8::compress(csrc, comp, p);
    check(cr.ok(), "prism ultra level 11 compress ok");
    if (cr.ok()) {
      comp.resize(cr.value);
      std::vector<std::byte> decomp(csrc.size());
      auto dr = spectra8::decompress(comp, decomp);
      check(dr.ok(), "prism ultra level 11 decompress ok");
      check(dr.value == csrc.size(), "prism ultra size match");
      check(std::memcmp(csrc.data(), decomp.data(), csrc.size()) == 0,
            "prism ultra bit-for-bit roundtrip");
    }
  }
  std::printf("  [bwt_ultra] all unit & pipeline checks passed\n");
}

void test_digit_norm_filter() {
  using spectra8::detail::u8;
  using spectra8::detail::usize;
  using spectra8::detail::digit_norm_detail::digit_norm_fwd;
  using spectra8::detail::digit_norm_detail::digit_norm_inv;
  using spectra8::detail::digit_norm_detail::detect_digit_norm;

  // 1. Basic text roundtrip with digits
  {
    const std::string text =
        "The LHC collision took place on 2026-10-11 at 13.6 TeV with 2808 bunches and run #987654321.";
    const auto* src = reinterpret_cast<const u8*>(text.data());
    const usize n = text.size();
    std::vector<u8> fwd;
    check(digit_norm_fwd(src, n, fwd), "digit_norm_fwd basic text");
    std::vector<u8> inv(n, 0);
    auto st = digit_norm_inv(fwd.data(), fwd.size(), inv.data(), n);
    check(st == spectra8::Status::Ok, "digit_norm_inv basic text ok");
    check(std::memcmp(src, inv.data(), n) == 0, "digit_norm basic text matches");
  }

  // 2. Edge cases
  {
    const std::vector<std::string> cases = {
        "7",
        "42",
        "123",
        "12345678901234567890",
        "a1b2c3d4e5f6g7h8i9j0",
        "9999999999",
        "=== 000 === 111 === 222 ===",
        "pi = 3.1415926535897932384626433832795028841971"
    };
    for (const auto& s : cases) {
      const auto* src = reinterpret_cast<const u8*>(s.data());
      const usize n = s.size();
      std::vector<u8> fwd;
      check(digit_norm_fwd(src, n, fwd), "digit_norm_fwd edge case");
      std::vector<u8> inv(n, 0);
      auto st = digit_norm_inv(fwd.data(), fwd.size(), inv.data(), n);
      check(st == spectra8::Status::Ok, "digit_norm_inv edge case ok");
      check(std::memcmp(src, inv.data(), n) == 0, "digit_norm edge case matches");
    }

    // No digits => returns false
    const std::string no_digits = "Hello, world! Just plain text with no numbers.";
    std::vector<u8> fwd_none;
    check(!digit_norm_fwd(reinterpret_cast<const u8*>(no_digits.data()), no_digits.size(), fwd_none),
          "digit_norm_fwd returns false when no digits");
  }

  // 3. Corruption rejection
  {
    const std::string text = "Test run 123456 with parameters 987654 and 42.";
    const auto* src = reinterpret_cast<const u8*>(text.data());
    const usize n = text.size();
    std::vector<u8> fwd;
    check(digit_norm_fwd(src, n, fwd), "digit_norm_fwd for corrupt test");
    std::vector<u8> inv(n, 0);

    // Truncated input
    for (size_t cut = 1; cut < fwd.size(); ++cut) {
      auto st = digit_norm_inv(fwd.data(), fwd.size() - cut, inv.data(), n);
      check(st == spectra8::Status::CorruptInput, "digit_norm_inv truncated returns corrupt");
    }

    // Corrupted side stream byte (> 109)
    std::vector<u8> bad = fwd;
    bad.back() = 250;
    auto st = digit_norm_inv(bad.data(), bad.size(), inv.data(), n);
    check(st == spectra8::Status::CorruptInput, "digit_norm_inv invalid side byte returns corrupt");

    // Length mismatch (dst length != header n)
    st = digit_norm_inv(fwd.data(), fwd.size(), inv.data(), n - 1);
    check(st == spectra8::Status::CorruptInput, "digit_norm_inv length mismatch returns corrupt");
  }

  // 4. Detector checks
  {
    // Text with numbers >= 1.5%
    std::string xml;
    for (int i = 0; i < 50; ++i) {
      xml += "<record id=\"" + std::to_string(10000 + i) + "\" val=\"" + std::to_string(i * 17) + "\"/>\n";
    }
    check(detect_digit_norm(reinterpret_cast<const u8*>(xml.data()), xml.size()),
          "detect_digit_norm matches XML with numbers");

    // Text with no numbers
    std::string prose = "This is a simple sentence that repeats without containing any numerals at all. ";
    while (prose.size() < 1024) prose += prose;
    check(!detect_digit_norm(reinterpret_cast<const u8*>(prose.data()), prose.size()),
          "detect_digit_norm rejects prose without numbers");

    // Random binary
    std::vector<u8> rnd(2048);
    std::mt19937 g(42);
    for (auto& b : rnd) b = static_cast<u8>(g());
    check(!detect_digit_norm(rnd.data(), rnd.size()), "detect_digit_norm rejects random binary");
  }

  // 5. Full pipeline compression / decompression across codecs
  {
    std::string corpus;
    corpus.reserve(64 * 1024);
    for (int i = 0; i < 1500; ++i) {
      corpus += "<event id=\"" + std::to_string(100000 + i) + "\" energy=\"" +
                std::to_string(i * 3.1415) + "\" tracks=\"" + std::to_string(i % 120) + "\"/>\n";
    }
    auto csrc = std::span<const std::byte>(
        reinterpret_cast<const std::byte*>(corpus.data()), corpus.size());

    for (spectra8::Codec c : {spectra8::Codec::Beam, spectra8::Codec::Prism,
                               spectra8::Codec::Crystal, spectra8::Codec::Flash,
                               spectra8::Codec::Zstd, spectra8::Codec::Bzip3,
                               spectra8::Codec::Lzma}) {
      if (!spectra8::codec_available(c)) continue;
      spectra8::CompressParams p;
      p.codec = c;
      p.level = 3;
      p.filters = true;
      p.filter_mask = spectra8::filter_bit(spectra8::FilterId::DigitNorm);
      std::vector<std::byte> comp(spectra8::compress_bound(csrc.size(), c));
      auto cr = spectra8::compress(csrc, comp, p);
      check(cr.ok(), "digit_norm pipeline compress ok");
      if (!cr.ok()) continue;

      std::vector<std::byte> dec(csrc.size());
      auto dr = spectra8::decompress(std::span<const std::byte>(comp.data(), cr.value), dec);
      if (!dr.ok()) {
        std::printf("codec %d failed decompress: status %d (%s)\n",
                    (int)c, (int)dr.status, spectra8::status_message(dr.status).data());
      }
      check(dr.ok(), "digit_norm pipeline decompress ok");
      check(dr.value == csrc.size(), "digit_norm pipeline size match");
      check(std::memcmp(dec.data(), csrc.data(), csrc.size()) == 0,
            "digit_norm pipeline bit-for-bit roundtrip");
    }
  }
  std::printf("  [digit_norm] all unit & pipeline checks passed\n");
}

void test_word_norm_filter() {
  using spectra8::detail::u8;
  using spectra8::detail::usize;
  using spectra8::detail::word_norm_detail::word_norm_fwd;
  using spectra8::detail::word_norm_detail::word_norm_inv;
  using spectra8::detail::word_norm_detail::detect_word_norm;

  // 1. Basic text roundtrip with word repetitions and mixed case
  {
    std::string text;
    for (int i = 0; i < 40; ++i) {
      text += "The large hadron collider at CERN accelerates particles to high energy. ";
      text += "THE COLLISION PRODUCES PARTICLES, and the detector measures their tracks. ";
      text += "Specialized experiments measure collision events with high precision and energy.\n";
    }
    const auto* src = reinterpret_cast<const u8*>(text.data());
    const usize n = text.size();
    std::vector<u8> fwd;
    check(word_norm_fwd(src, n, fwd), "word_norm_fwd basic text");
    check(fwd.size() < n, "word_norm_fwd output smaller than input");
    std::vector<u8> inv(n, 0);
    auto st = word_norm_inv(fwd.data(), fwd.size(), inv.data(), n);
    check(st == spectra8::Status::Ok, "word_norm_inv basic text ok");
    check(std::memcmp(src, inv.data(), n) == 0, "word_norm basic text matches");
  }

  // 2. High-byte and UTF-8 characters escaping
  {
    std::string text;
    for (int i = 0; i < 30; ++i) {
      text += "Le grand collisionneur de hadrons du CERN génère des collisions d'énergie élevée.\n";
      text += "Physik und Technologie der Teilchenbeschleuniger mit Präzisionsmessung © 2026.\n";
    }
    const auto* src = reinterpret_cast<const u8*>(text.data());
    const usize n = text.size();
    std::vector<u8> fwd;
    check(word_norm_fwd(src, n, fwd), "word_norm_fwd utf-8 text");
    std::vector<u8> inv(n, 0);
    auto st = word_norm_inv(fwd.data(), fwd.size(), inv.data(), n);
    check(st == spectra8::Status::Ok, "word_norm_inv utf-8 ok");
    check(std::memcmp(src, inv.data(), n) == 0, "word_norm utf-8 text matches");
  }

  // 3. Corruption rejection
  {
    std::string text;
    for (int i = 0; i < 20; ++i) {
      text += "The quick brown fox jumps over the lazy dog. Repeat the lazy dog sentence.\n";
    }
    const auto* src = reinterpret_cast<const u8*>(text.data());
    const usize n = text.size();
    std::vector<u8> fwd;
    check(word_norm_fwd(src, n, fwd), "word_norm_fwd for corrupt test");
    std::vector<u8> inv(n, 0);

    // Truncated input
    for (size_t cut = 1; cut < std::min(fwd.size(), size_t{30}); ++cut) {
      auto st = word_norm_inv(fwd.data(), fwd.size() - cut, inv.data(), n);
      check(st == spectra8::Status::CorruptInput, "word_norm_inv truncated returns corrupt");
    }

    // Corrupted magic
    std::vector<u8> bad = fwd;
    bad[0] = 0x00;
    auto st = word_norm_inv(bad.data(), bad.size(), inv.data(), n);
    check(st == spectra8::Status::CorruptInput, "word_norm_inv bad magic returns corrupt");

    // Length mismatch
    st = word_norm_inv(fwd.data(), fwd.size(), inv.data(), n - 1);
    check(st == spectra8::Status::CorruptInput, "word_norm_inv length mismatch returns corrupt");
  }

  // 4. Detector checks
  {
    std::string text = "This is normal English prose with common words and sentences repeating frequently. ";
    while (text.size() < 1024) text += text;
    check(detect_word_norm(reinterpret_cast<const u8*>(text.data()), text.size()),
          "detect_word_norm matches repetitive text");

    std::vector<u8> rnd(2048);
    std::mt19937 g(12345);
    for (auto& b : rnd) b = static_cast<u8>(g());
    check(!detect_word_norm(rnd.data(), rnd.size()), "detect_word_norm rejects random binary");
  }

  // 5. XML/HTML markup tags, entities, URLs, and C++ scope qualifiers
  {
    std::string text;
    for (int i = 0; i < 40; ++i) {
      text += "<page>\n";
      text += "  <title>Physics &quot;LHC&quot; Experiment &amp; Analysis</title>\n";
      text += "  <id>1042</id>\n";
      text += "  <text>Consult http://cern.ch and https://atlas.ch for std::vector measurements.</text>\n";
      text += "</page>\n";
    }
    const auto* src = reinterpret_cast<const u8*>(text.data());
    const usize n = text.size();
    std::vector<u8> fwd;
    check(word_norm_fwd(src, n, fwd), "word_norm_fwd markup and entities");
    check(fwd.size() < n, "word_norm_fwd markup output smaller than input");
    std::vector<u8> inv(n, 0);
    auto st = word_norm_inv(fwd.data(), fwd.size(), inv.data(), n);
    check(st == spectra8::Status::Ok, "word_norm_inv markup and entities ok");
    check(std::memcmp(src, inv.data(), n) == 0, "word_norm markup and entities roundtrip exact");
  }

  // 5. Full pipeline compression / decompression
  {
    std::string corpus;
    corpus.reserve(64 * 1024);
    for (int i = 0; i < 200; ++i) {
      corpus += "The CERN Large Hadron Collider (LHC) is the world's largest particle accelerator. ";
      corpus += "Physicists use detectors to observe collision products and search for new physics phenomena. ";
      corpus += "THE COLLIDER OPERATES AT HIGH LUMINOSITY to produce rare events with high statistical significance.\n";
    }
    auto csrc = std::span<const std::byte>(
        reinterpret_cast<const std::byte*>(corpus.data()), corpus.size());

    for (spectra8::Codec c : {spectra8::Codec::Beam, spectra8::Codec::Prism,
                               spectra8::Codec::Crystal, spectra8::Codec::Flash,
                               spectra8::Codec::Zstd, spectra8::Codec::Bzip3,
                               spectra8::Codec::Lzma}) {
      if (!spectra8::codec_available(c)) continue;
      spectra8::CompressParams p;
      p.codec = c;
      p.level = 3;
      p.filters = true;
      p.filter_mask = spectra8::filter_bit(spectra8::FilterId::WordNorm);
      std::vector<std::byte> comp(spectra8::compress_bound(csrc.size(), c));
      auto cr = spectra8::compress(csrc, comp, p);
      check(cr.ok(), "word_norm pipeline compress ok");
      if (!cr.ok()) continue;

      std::vector<std::byte> decomp(corpus.size());
      auto dr = spectra8::decompress(std::span<const std::byte>(comp.data(), cr.value), decomp);
      check(dr.ok(), "word_norm pipeline decompress ok");
      check(dr.value == csrc.size(), "word_norm pipeline size match");
      check(std::memcmp(csrc.data(), decomp.data(), csrc.size()) == 0,
            "word_norm pipeline bit-for-bit roundtrip");
    }
  }
  std::printf("  [word_norm] all unit & pipeline checks passed\n");
}

void test_base64_filter() {
  using spectra8::detail::u8;
  using spectra8::detail::usize;
  using spectra8::detail::base64_detail::base64_fwd;
  using spectra8::detail::base64_detail::base64_inv;
  using spectra8::detail::base64_detail::detect_base64;

  // 1. PEM-style wrapped base64 with header & footer literals
  {
    std::string pem =
        "-----BEGIN CERTIFICATE-----\n"
        "MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAzq2e5e4n5e6q7r8s9t0u1v2w\n"
        "3x4y5z6a7b8c9d0e1f2g3h4i5j6k7l8m9n0o1p2q3r4s5t6u7v8w9x0y1z2a3b4c5d6e\n"
        "7f8g9h0i1j2k3l4m5n6o7p8q9r0s1t2u3v4w5x6y7z8a9b0c1d2e3f4g5h6i7j8k9l0m\n"
        "1n2o3p4q5r6s7t8u9v0w1x2y3z4a5b6c7d8e9f0g1h2i3j4k5l6m7n8o9p0q1r2s3t4u\n"
        "5v6w7x8y9z0=\n"
        "-----END CERTIFICATE-----\n";
    while (pem.size() < 1024) pem += pem;

    const auto* src = reinterpret_cast<const u8*>(pem.data());
    const usize n = pem.size();
    check(detect_base64(src, n), "detect_base64 matches PEM certificate");
    std::vector<u8> fwd;
    check(base64_fwd(src, n, fwd), "base64_fwd PEM certificate");
    check(fwd.size() < n, "base64_fwd output smaller than input");
    std::vector<u8> inv(n, 0);
    auto st = base64_inv(fwd.data(), fwd.size(), inv.data(), n);
    check(st == spectra8::Status::Ok, "base64_inv PEM ok");
    check(std::memcmp(src, inv.data(), n) == 0, "base64 PEM text matches bit-for-bit");
  }

  // 2. JSON payload with continuous (unwrapped) Base64
  {
    std::vector<u8> raw(148);
    for (size_t i = 0; i < raw.size(); ++i) raw[i] = static_cast<u8>((i * 17 + 3) & 0xFF);
    std::string b64(raw.size() * 2 + 16, '\0');
    usize b64_sz = spectra8::detail::base64_detail::encode_b64_chunk(
        raw.data(), raw.size(), false, 2, 0, false, 0,
        reinterpret_cast<u8*>(b64.data()), b64.size());
    b64.resize(b64_sz);
    std::string json = "{\"status\": 200, \"payload\": \"" + b64 + "\", \"timestamp\": 1700000000}\n";
    while (json.size() < 2048) json += json;

    const auto* src = reinterpret_cast<const u8*>(json.data());
    const usize n = json.size();
    check(detect_base64(src, n), "detect_base64 matches JSON with base64 payload");
    std::vector<u8> fwd;
    check(base64_fwd(src, n, fwd), "base64_fwd JSON payload");
    std::vector<u8> inv(n, 0);
    auto st = base64_inv(fwd.data(), fwd.size(), inv.data(), n);
    check(st == spectra8::Status::Ok, "base64_inv JSON ok");
    check(std::memcmp(src, inv.data(), n) == 0, "base64 JSON matches bit-for-bit");
  }

  // 3. Corruption rejection
  {
    std::string pem =
        "-----BEGIN CERTIFICATE-----\n"
        "MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAzq2e5e4n5e6q7r8s9t0u1v2w\n"
        "3x4y5z6a7b8c9d0e1f2g3h4i5j6k7l8m9n0o1p2q3r4s5t6u7v8w9x0y1z2a3b4c5d6e\n"
        "7f8g9h0i1j2k3l4m5n6o7p8q9r0s1t2u3v4w5x6y7z8a9b0c1d2e3f4g5h6i7j8k9l0m\n"
        "1n2o3p4q5r6s7t8u9v0w1x2y3z4a5b6c7d8e9f0g1h2i3j4k5l6m7n8o9p0q1r2s3t4u\n"
        "5v6w7x8y9z0=\n"
        "-----END CERTIFICATE-----\n";
    while (pem.size() < 1024) pem += pem;
    const auto* src = reinterpret_cast<const u8*>(pem.data());
    const usize n = pem.size();
    std::vector<u8> fwd;
    check(base64_fwd(src, n, fwd), "base64_fwd for corrupt test");
    std::vector<u8> inv(n, 0);

    // Truncated input
    for (size_t cut = 1; cut < std::min(fwd.size(), size_t{20}); ++cut) {
      auto st = base64_inv(fwd.data(), fwd.size() - cut, inv.data(), n);
      check(st == spectra8::Status::CorruptInput, "base64_inv truncated returns corrupt");
    }

    // Length mismatch
    auto st = base64_inv(fwd.data(), fwd.size(), inv.data(), n - 1);
    check(st == spectra8::Status::CorruptInput, "base64_inv length mismatch returns corrupt");
  }

  // 4. Full pipeline compression / decompression
  {
    std::string pem =
        "-----BEGIN PUBLIC KEY-----\n"
        "MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAy5t9d0r1s2t3u4v5w6x7y8z9\n"
        "0a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0u1v2w3x4y5z6a7b8c9d0e1f2g3h\n"
        "4i5j6k7l8m9n0o1p2q3r4s5t6u7v8w9x0y1z2a3b4c5d6e7f8g9h0i1j2k3l4m5n6o7p\n"
        "8q9r0s1t2u3v4w5x6y7z8a9b0c1d2e3f4g5h6i7j8k9l0m1n2o3p4q5r6s7t8u9v0w1x\n"
        "2y3z4a5b6c7d8e9f0g1h2i3j4k5l6m7n8o9p0q1r2s3t4u5v6w7x8y9z0ABCDEF==\n"
        "-----END PUBLIC KEY-----\n";
    while (pem.size() < 16384) pem += pem;
    auto csrc = std::span<const std::byte>(
        reinterpret_cast<const std::byte*>(pem.data()), pem.size());

    for (spectra8::Codec c : {spectra8::Codec::Beam, spectra8::Codec::Prism,
                               spectra8::Codec::Crystal, spectra8::Codec::Flash,
                               spectra8::Codec::Zstd, spectra8::Codec::Bzip3,
                               spectra8::Codec::Lzma}) {
      if (!spectra8::codec_available(c)) continue;
      spectra8::CompressParams p;
      p.codec = c;
      p.level = 3;
      p.filters = true;
      p.filter_mask = spectra8::filter_bit(spectra8::FilterId::Base64);
      std::vector<std::byte> comp(spectra8::compress_bound(csrc.size(), c));
      auto cr = spectra8::compress(csrc, comp, p);
      check(cr.ok(), "base64 pipeline compress ok");
      if (!cr.ok()) continue;

      std::vector<std::byte> decomp(pem.size());
      auto dr = spectra8::decompress(std::span<const std::byte>(comp.data(), cr.value), decomp);
      check(dr.ok(), "base64 pipeline decompress ok");
      check(dr.value == csrc.size(), "base64 pipeline size match");
      check(std::memcmp(csrc.data(), decomp.data(), csrc.size()) == 0,
            "base64 pipeline bit-for-bit roundtrip");
    }
  }
  std::printf("  [base64] all unit & pipeline checks passed\n");
}

void test_crystal_match_bypass() {
  std::string repetitive;
  for (int i = 0; i < 300; ++i) {
    repetitive += "Repeated sentence with an exact long match section to engage match bypass in crystal. ";
    repetitive += "Second repetitive line demonstrating multi-order context mixing with bypass active.\n";
  }
  auto csrc = std::span<const std::byte>(
      reinterpret_cast<const std::byte*>(repetitive.data()), repetitive.size());

  spectra8::CompressParams p;
  p.codec = spectra8::Codec::Crystal;
  p.level = 10;
  std::vector<std::byte> comp(spectra8::compress_bound(csrc.size(), spectra8::Codec::Crystal));
  auto cr = spectra8::compress(csrc, comp, p);
  check(cr.ok(), "crystal L10 match bypass compress ok");
  if (cr.ok()) {
    std::vector<std::byte> decomp(repetitive.size());
    auto dr = spectra8::decompress(std::span<const std::byte>(comp.data(), cr.value), decomp);
    check(dr.ok(), "crystal L10 match bypass decompress ok");
    check(dr.value == csrc.size(), "crystal L10 decomp size match");
    check(std::memcmp(csrc.data(), decomp.data(), csrc.size()) == 0,
          "crystal L10 match bypass roundtrip bit-exact");
  }

  std::printf("  [crystal_match_bypass] level 10 bit-exact roundtrip passed\n");
}

void test_fused_transforms() {
  using spectra8::detail::Filter;
  using spectra8::detail::u8;
  std::mt19937_64 rng(42);

  const Filter filters[] = {
      Filter::DeltaShuffle32,       Filter::ZigzagDeltaShuffle32,
      Filter::DeltaShuffle64,       Filter::ZigzagShuffle32,
      Filter::ZigzagShuffle64,     Filter::XorShuffle32,
      Filter::XorShuffle64,         Filter::Delta2Shuffle32,
      Filter::ZigzagDelta2Shuffle32, Filter::Delta2Shuffle64};

  const size_t test_sizes[] = {0, 1, 3, 4, 7, 8, 15, 16, 63, 64, 127, 128, 511, 1024, 1025, 2048, 4099, 16384};

  for (Filter f : filters) {
    for (size_t n : test_sizes) {
      std::vector<u8> in(n);
      for (size_t i = 0; i < n; ++i) in[i] = static_cast<u8>(rng());
      std::vector<u8> fwd(n), inv(n);
      spectra8::detail::apply_filter(f, in.data(), n, fwd.data());
      spectra8::detail::invert_filter(f, fwd.data(), n, inv.data());
      check(inv == in, "fused transform roundtrip bit-exact");

      // Verify bit-for-bit equivalence against two-pass scratch transform
      std::vector<u8> reference(n);
      std::vector<u8> scratch(n);
      switch (f) {
        case Filter::DeltaShuffle32:
          spectra8::detail::delta_fwd(in.data(), n, scratch.data(), 4);
          spectra8::detail::shuffle_fwd(scratch.data(), n, reference.data(), 4);
          break;
        case Filter::ZigzagDeltaShuffle32:
          spectra8::detail::zigzag_delta_fwd(in.data(), n, scratch.data(), 4);
          spectra8::detail::shuffle_fwd(scratch.data(), n, reference.data(), 4);
          break;
        case Filter::DeltaShuffle64:
          spectra8::detail::delta_fwd(in.data(), n, scratch.data(), 8);
          spectra8::detail::shuffle_fwd(scratch.data(), n, reference.data(), 8);
          break;
        case Filter::ZigzagShuffle32:
          spectra8::detail::zigzag_fwd(in.data(), n, scratch.data(), 4);
          spectra8::detail::shuffle_fwd(scratch.data(), n, reference.data(), 4);
          break;
        case Filter::ZigzagShuffle64:
          spectra8::detail::zigzag_fwd(in.data(), n, scratch.data(), 8);
          spectra8::detail::shuffle_fwd(scratch.data(), n, reference.data(), 8);
          break;
        case Filter::XorShuffle32:
          spectra8::detail::xor_delta_fwd(in.data(), n, scratch.data(), 4);
          spectra8::detail::shuffle_fwd(scratch.data(), n, reference.data(), 4);
          break;
        case Filter::XorShuffle64:
          spectra8::detail::xor_delta_fwd(in.data(), n, scratch.data(), 8);
          spectra8::detail::shuffle_fwd(scratch.data(), n, reference.data(), 8);
          break;
        case Filter::Delta2Shuffle32:
          spectra8::detail::delta2_fwd(in.data(), n, scratch.data(), 4);
          spectra8::detail::shuffle_fwd(scratch.data(), n, reference.data(), 4);
          break;
        case Filter::ZigzagDelta2Shuffle32:
          spectra8::detail::zigzag_delta2_fwd(in.data(), n, scratch.data(), 4);
          spectra8::detail::shuffle_fwd(scratch.data(), n, reference.data(), 4);
          break;
        case Filter::Delta2Shuffle64:
          spectra8::detail::delta2_fwd(in.data(), n, scratch.data(), 8);
          spectra8::detail::shuffle_fwd(scratch.data(), n, reference.data(), 8);
          break;
        default: break;
      }
      check(fwd == reference, "fused transform matches 2-pass reference exactly");
    }
  }
  std::printf("  [fused_transforms] all fused multi-pass transforms bit-exact with 2-pass reference\n");
}

void test_accelerated_bit_unpack() {
  using spectra8::detail::u8;
  using spectra8::detail::u32;
  using spectra8::detail::u64;
  std::mt19937_64 rng(101);

  // 1. Test bw == 0 instant refill
  for (int w : {2, 4, 8}) {
    const size_t n_elem = 500;
    std::vector<u8> dst(n_elem * w, 0x55);
    const int64_t fmin = 1234567;
    spectra8::detail::pack_detail::unpack_inliers_span(
        nullptr, 0, dst.data(), 0, n_elem, 0, 0, fmin, w);
    for (size_t i = 0; i < n_elem; ++i) {
      int64_t v = 0;
      if (w == 2) v = *reinterpret_cast<const int16_t*>(dst.data() + i * 2);
      else if (w == 4) v = *reinterpret_cast<const int32_t*>(dst.data() + i * 4);
      else v = *reinterpret_cast<const int64_t*>(dst.data() + i * 8);
      check(v == (w == 2 ? static_cast<int16_t>(fmin) : (w == 4 ? static_cast<int32_t>(fmin) : fmin)),
            "bw=0 unpack_inliers_span match");
    }
  }

  // 2. Test various bit widths through BitWriter and unpack_inliers_span
  const int test_bws[] = {1, 2, 5, 8, 11, 16, 21, 24, 31, 32, 48, 60};
  for (int bw : test_bws) {
    const size_t n_elem = 256;
    const u64 mask = bw >= 64 ? ~u64{0} : ((u64{1} << bw) - 1);
    std::vector<u64> residuals(n_elem);
    std::vector<u8> packed;
    spectra8::detail::pack_detail::BitWriter writer(packed);
    for (size_t i = 0; i < n_elem; ++i) {
      residuals[i] = rng() & mask;
      writer.put(residuals[i], bw);
    }
    writer.flush();

    for (int w : {2, 4, 8}) {
      if (bw >= w * 8) continue;
      std::vector<u8> dst(n_elem * w, 0);
      const int64_t fmin = 100;
      spectra8::detail::pack_detail::unpack_inliers_span(
          packed.data(), packed.size(), dst.data(), 0, n_elem, 0, bw, fmin, w);
      bool all_ok = true;
      for (size_t i = 0; i < n_elem; ++i) {
        u64 got = 0;
        if (w == 2) got = *reinterpret_cast<const uint16_t*>(dst.data() + i * 2) - static_cast<uint16_t>(fmin);
        else if (w == 4) got = *reinterpret_cast<const uint32_t*>(dst.data() + i * 4) - static_cast<uint32_t>(fmin);
        else got = *reinterpret_cast<const uint64_t*>(dst.data() + i * 8) - static_cast<uint64_t>(fmin);
        if (got != residuals[i]) {
          all_ok = false;
          break;
        }
      }
      check(all_ok, "unpack_inliers_span matches writer for bw=" + std::to_string(bw) + " w=" + std::to_string(w));
    }
  }
  std::printf("  [accelerated_bit_unpack] bit reader accumulator refill & span unpack checks passed\n");
}

void test_bitround_filter() {
  using spectra8::detail::Filter;
  using spectra8::detail::u8;
  std::mt19937_64 rng(303);

  // 1. Test BitRound32 mantissa noise zeroing and shuffle
  {
    const size_t n_floats = 4096;
    const size_t n_bytes = n_floats * 4;
    std::vector<float> orig(n_floats);
    std::uniform_real_distribution<float> dist(1.0f, 100.0f);
    for (size_t i = 0; i < n_floats; ++i) orig[i] = dist(rng);

    const u8* in_bytes = reinterpret_cast<const u8*>(orig.data());
    std::vector<u8> fwd(n_bytes), inv(n_bytes);
    spectra8::detail::apply_filter(Filter::BitRound32, in_bytes, n_bytes, fwd.data());
    spectra8::detail::invert_filter(Filter::BitRound32, fwd.data(), n_bytes, inv.data());

    const float* rounded = reinterpret_cast<const float*>(inv.data());
    for (size_t i = 0; i < n_floats; ++i) {
      const float diff = std::abs(orig[i] - rounded[i]);
      const float rel_err = diff / orig[i];
      check(rel_err < 0.002f, "bitround32 precision within 10-bit mantissa bound");

      uint32_t u;
      std::memcpy(&u, &rounded[i], 4);
      check((u & 0x1FFF) == 0, "bitround32 low 13 mantissa bits zeroed");
    }

    size_t zero_bytes = 0;
    for (size_t i = 0; i < n_floats; ++i) {
      if (fwd[i] == 0) ++zero_bytes;
    }
    check(zero_bytes == n_floats, "bitround32 lowest byte plane is 100% zeroed");
  }

  // 2. Test BitRound64 mantissa noise zeroing and shuffle
  {
    const size_t n_doubles = 2048;
    const size_t n_bytes = n_doubles * 8;
    std::vector<double> orig(n_doubles);
    std::uniform_real_distribution<double> dist(0.1, 1000.0);
    for (size_t i = 0; i < n_doubles; ++i) orig[i] = dist(rng);

    const u8* in_bytes = reinterpret_cast<const u8*>(orig.data());
    std::vector<u8> fwd(n_bytes), inv(n_bytes);
    spectra8::detail::apply_filter(Filter::BitRound64, in_bytes, n_bytes, fwd.data());
    spectra8::detail::invert_filter(Filter::BitRound64, fwd.data(), n_bytes, inv.data());

    const double* rounded = reinterpret_cast<const double*>(inv.data());
    for (size_t i = 0; i < n_doubles; ++i) {
      const double diff = std::abs(orig[i] - rounded[i]);
      const double rel_err = diff / orig[i];
      check(rel_err < 0.000003, "bitround64 precision within 20-bit mantissa bound");

      uint64_t u;
      std::memcpy(&u, &rounded[i], 8);
      check((u & 0xFFFFFFFFULL) == 0, "bitround64 low 32 mantissa bits zeroed");
    }
  }

  // 3. Pipeline compression with BitRound32 on float column
  {
    const size_t n_floats = 16384;
    std::vector<float> col(n_floats);
    std::uniform_real_distribution<float> dist(0.5f, 50.0f);
    for (size_t i = 0; i < n_floats; ++i) col[i] = dist(rng);

    auto src_span = std::span<const std::byte>(
        reinterpret_cast<const std::byte*>(col.data()), col.size() * sizeof(float));

    spectra8::CompressParams p;
    p.codec = spectra8::Codec::Beam;
    p.level = 4;
    p.filters = true;
    p.filter_mask = spectra8::filter_bit(spectra8::FilterId::BitRound32);

    std::vector<std::byte> comp(spectra8::compress_bound(src_span.size(), p.codec));
    auto cr = spectra8::compress(src_span, comp, p);
    check(cr.ok(), "bitround32 compress ok");
    if (cr.ok()) {
      std::vector<std::byte> decomp(src_span.size());
      auto dr = spectra8::decompress(
          std::span<const std::byte>(comp.data(), cr.value), decomp);
      check(dr.ok(), "bitround32 decompress ok");
      check(dr.value == src_span.size(), "bitround32 size match");

      const float* dfloats = reinterpret_cast<const float*>(decomp.data());
      for (size_t i = 0; i < n_floats; ++i) {
        const float diff = std::abs(col[i] - dfloats[i]);
        check(diff / col[i] < 0.002f, "pipeline bitround32 precision maintained");
      }
    }
  }
  std::printf("  [bitround] BitRound32 and BitRound64 mantissa noise trimming verified\n");
}

void test_context_oracle() {
  // Test context oracle on structured data (record stride 4) and text
  std::mt19937_64 rng(42);
  const size_t n = 65536;
  std::vector<uint8_t> d(n);

  // Generate structured record data: 4-byte records with correlated fields
  for (size_t i = 0; i < n; i += 4) {
    uint8_t base = static_cast<uint8_t>((i / 4) & 0xFF);
    d[i + 0] = base;
    d[i + 1] = static_cast<uint8_t>(base + (rng() % 3));
    d[i + 2] = static_cast<uint8_t>(rng() % 5);
    d[i + 3] = static_cast<uint8_t>(base ^ 0xAA);
  }

  // 1. C++ API with context_oracle = true
  {
    spectra8::CompressParams p;
    p.codec = spectra8::Codec::Beam;
    p.level = spectra8::kUltraLevel;
    p.context_oracle = true;

    auto src = bytes(d);
    std::vector<std::byte> comp(spectra8::compress_bound(src.size()));
    auto r = spectra8::compress(src, comp, p);
    check(r.ok(), "context_oracle: compress ok");
    comp.resize(r.value);

    std::vector<std::byte> out(n);
    auto dec = spectra8::decompress(comp, out);
    check(dec.ok() && dec.value == n, "context_oracle: decompress size");
    check(std::memcmp(out.data(), d.data(), n) == 0, "context_oracle: roundtrip match");
  }

  // 2. Near-optimal level with context_oracle = true
  {
    spectra8::CompressParams p;
    p.codec = spectra8::Codec::Beam;
    p.level = spectra8::kNearOptLevel;
    p.context_oracle = true;

    auto src = bytes(d);
    std::vector<std::byte> comp(spectra8::compress_bound(src.size()));
    auto r = spectra8::compress(src, comp, p);
    check(r.ok(), "context_oracle near-opt: compress ok");
    comp.resize(r.value);

    std::vector<std::byte> out(n);
    auto dec = spectra8::decompress(comp, out);
    check(dec.ok() && dec.value == n, "context_oracle near-opt: decompress size");
    check(std::memcmp(out.data(), d.data(), n) == 0, "context_oracle near-opt: roundtrip match");
  }

  // 3. C API with context_oracle = 1
  {
    spectra8_compress_options opts{};
    opts.codec = SPECTRA8_CODEC_BEAM;
    opts.level = SPECTRA8_MAX_LEVEL;
    opts.context_oracle = 1;

    std::vector<uint8_t> comp(spectra8_compress_bound(n));
    size_t written = 0;
    spectra8_status cs = spectra8_compress_with_options(
        d.data(), n, comp.data(), comp.size(), &opts, &written);
    check(cs == SPECTRA8_OK && written > 0, "context_oracle C API: compress ok");

    std::vector<uint8_t> out(n);
    size_t dec_written = 0;
    spectra8_status ds = spectra8_decompress(
        comp.data(), written, out.data(), out.size(), &dec_written);
    check(ds == SPECTRA8_OK && dec_written == n, "context_oracle C API: decompress size");
    check(std::memcmp(out.data(), d.data(), n) == 0, "context_oracle C API: roundtrip match");
  }

  // 4. Environment variable SPECTRA8_CONTEXT_ORACLE=1
  {
    setenv("SPECTRA8_CONTEXT_ORACLE", "1", 1);
    spectra8::CompressParams p;
    p.codec = spectra8::Codec::Beam;
    p.level = spectra8::kUltraLevel;

    auto src = bytes(d);
    std::vector<std::byte> comp(spectra8::compress_bound(src.size()));
    auto r = spectra8::compress(src, comp, p);
    check(r.ok(), "context_oracle env var: compress ok");
    comp.resize(r.value);

    std::vector<std::byte> out(n);
    auto dec = spectra8::decompress(comp, out);
    check(dec.ok() && dec.value == n, "context_oracle env var: decompress size");
    check(std::memcmp(out.data(), d.data(), n) == 0, "context_oracle env var: roundtrip match");
    unsetenv("SPECTRA8_CONTEXT_ORACLE");
  }
}

}  // namespace

int main() {
  test_compress_bound();
  test_buffer_too_small();
  test_bad_inputs();
  test_corruption_detection();
  test_determinism();
  test_entropy_modes();
  test_entropy_golden();
  test_all_single_byte_values();
  test_large_input();
  test_c_abi();
  test_version();
  test_threading_determinism();
  test_block_sizes();
  test_seek_table_and_random_access();
  test_block_checksum_detection();
  test_incompressible_fast_path();
  test_ultra_level();
  test_context_oracle();
  test_filters();
  test_dict_filter();
  test_pack_filters();
  test_transform_kernels();
  test_new_composed_filters();
  test_byte_rle_codec();
  test_filter_byte_rle_path();
  test_filter_fallback();
  test_filter_rle_dict_toggles();
  test_intrinsic_byte_rle();
  test_bwt_primitives();
  test_arith_codec();
  test_arith_adaptive_codec();
  test_ans_codec();
  test_ans_o1_codec();
  test_arith_adaptive_o1_codec();
  test_bwt_mtf_bzip2_rle();
  test_bwt_mode();
  test_bwt_wide_block();
  test_prism_verdict_bin_i32();
  test_bwt_ultra();
  test_digit_norm_filter();
  test_word_norm_filter();
  test_base64_filter();
  test_crystal_match_bypass();
  test_fused_transforms();
  test_accelerated_bit_unpack();
  test_bitround_filter();
  test_unified_codecs();
  test_gpu_path();
  test_page_pump();

  if (g_failures == 0) {
    std::printf("All %d API checks passed.\n", g_checks);
    return 0;
  }
  std::fprintf(stderr, "%d/%d API checks FAILED.\n", g_failures, g_checks);
  return 1;
}
