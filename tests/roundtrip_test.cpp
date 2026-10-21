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

// Roundtrip and fuzz tests for spectra8. Returns non-zero on any failure.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "spectra8/spectra8.hpp"
#include "prism/lzp.hpp"
#include "prism/bwt_cm.hpp"

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

std::vector<std::byte> to_bytes(const std::vector<uint8_t>& v) {
  std::vector<std::byte> out(v.size());
  std::memcpy(out.data(), v.data(), v.size());
  return out;
}

bool roundtrip(const std::vector<uint8_t>& data, int level,
               const std::string& name) {
  auto src = to_bytes(data);
  std::vector<std::byte> comp(spectra8::compress_bound(src.size()));
  auto cr = spectra8::compress(src, comp, level);
  if (!cr) {
    std::fprintf(stderr, "FAIL: %s L%d compress: %s\n", name.c_str(), level,
                 std::string(spectra8::status_message(cr.status)).c_str());
    return false;
  }
  comp.resize(cr.value);

  auto cs = spectra8::get_frame_content_size(comp);
  if (!cs || *cs != src.size()) {
    std::fprintf(stderr, "FAIL: %s L%d content size\n", name.c_str(), level);
    return false;
  }

  std::vector<std::byte> dec(src.size());
  auto dr = spectra8::decompress(comp, dec);
  if (!dr || dr.value != src.size()) {
    std::fprintf(stderr, "FAIL: %s L%d decompress: %s\n", name.c_str(), level,
                 std::string(spectra8::status_message(dr.status)).c_str());
    return false;
  }
  if (std::memcmp(src.data(), dec.data(), src.size()) != 0) {
    std::fprintf(stderr, "FAIL: %s L%d data mismatch\n", name.c_str(), level);
    return false;
  }
  return true;
}

void test_corpus(const std::vector<uint8_t>& data, const std::string& name) {
  for (int lvl = spectra8::kMinLevel; lvl <= spectra8::kMaxLevel; ++lvl) {
    check(roundtrip(data, lvl, name),
          name + " level " + std::to_string(lvl));
  }
}

// Roundtrip with an explicit block size (exercises the binary-tree match
// finder, which only runs on blocks > 1 MiB at levels >= 7).
bool roundtrip_codec(const std::vector<uint8_t>& data, spectra8::Codec codec,
                     int level, const std::string& name) {
  auto src = to_bytes(data);
  spectra8::CompressParams cp;
  cp.codec = codec;
  cp.level = level;
  std::vector<std::byte> comp(spectra8::compress_bound(src.size(), codec));
  auto cr = spectra8::compress(src, comp, cp);
  if (!cr) {
    std::fprintf(stderr, "FAIL: %s L%d compress: %s\n", name.c_str(), level,
                 std::string(spectra8::status_message(cr.status)).c_str());
    return false;
  }
  comp.resize(cr.value);
  std::vector<std::byte> dec(src.size());
  auto dr = spectra8::decompress(comp, dec);
  if (!dr || dr.value != src.size()) {
    std::fprintf(stderr, "FAIL: %s L%d decompress: %s\n", name.c_str(), level,
                 std::string(spectra8::status_message(dr.status)).c_str());
    return false;
  }
  return std::memcmp(src.data(), dec.data(), src.size()) == 0;
}

bool roundtrip_blocklog(const std::vector<uint8_t>& data, int level,
                        int block_log, const std::string& name) {
  auto src = to_bytes(data);
  spectra8::CompressParams cp;
  cp.level = level;
  cp.block_log = block_log;
  std::vector<std::byte> comp(spectra8::compress_bound(src.size()));
  auto cr = spectra8::compress(src, comp, cp);
  if (!cr) return false;
  comp.resize(cr.value);
  std::vector<std::byte> dec(src.size());
  auto dr = spectra8::decompress(comp, dec);
  if (!dr || dr.value != src.size()) return false;
  return std::memcmp(src.data(), dec.data(), src.size()) == 0;
}

}  // namespace

int main() {
  std::mt19937_64 rng(0xC0FFEE);

  // Empty input.
  test_corpus({}, "empty");

  // Single byte / small.
  test_corpus({42}, "one-byte");
  test_corpus({1, 2, 3}, "tiny");

  // All equal (memset path).
  {
    std::vector<uint8_t> d(100000, 0x5A);
    test_corpus(d, "all-equal");
  }

  // Highly repetitive (LZ + repeat offsets).
  {
    std::vector<uint8_t> d;
    const char* pat = "the quick brown fox jumps over the lazy dog. ";
    for (int i = 0; i < 5000; ++i)
      d.insert(d.end(), pat, pat + std::strlen(pat));
    test_corpus(d, "repetitive-text");
  }

  // Text-like with structure.
  {
    std::vector<uint8_t> d;
    for (int i = 0; i < 200000; ++i)
      d.push_back(static_cast<uint8_t>('a' + (i * 7 + (i >> 3)) % 26));
    test_corpus(d, "structured");
  }

  // Incompressible random.
  {
    std::vector<uint8_t> d(150000);
    for (auto& b : d) b = static_cast<uint8_t>(rng());
    test_corpus(d, "random");
  }

  // Mixed: random blocks + repeated blocks, crossing block boundaries.
  {
    std::vector<uint8_t> d;
    for (int blk = 0; blk < 8; ++blk) {
      if (blk % 2 == 0) {
        for (int i = 0; i < 70000; ++i) d.push_back(static_cast<uint8_t>(i & 0xFF));
      } else {
        for (int i = 0; i < 70000; ++i) d.push_back(static_cast<uint8_t>(rng()));
      }
    }
    test_corpus(d, "mixed-blocks");
  }

  // Regression: one block with no byte-RLE gain followed by one block of long
  // byte runs (byte-RLE wins). The RLE block sets the frame-global filter flag,
  // so the plain block must still carry a filter byte for the decoder.
  {
    std::vector<uint8_t> d;
    for (int i = 0; i < 262144; ++i)
      d.push_back(static_cast<uint8_t>('a' + (i * 7 + (i >> 3)) % 26));
    for (int r = 0; r < 40; ++r)
      d.insert(d.end(), 700, static_cast<uint8_t>(r));
    test_corpus(d, "mixed-rle-flag");
  }

  // Sizes around the block boundary (256 KiB).
  for (size_t sz : {262143u, 262144u, 262145u, 524288u, 524290u}) {
    std::vector<uint8_t> d(sz);
    for (size_t i = 0; i < sz; ++i)
      d[i] = static_cast<uint8_t>((i * 131 + (i >> 5)) ^ (i >> 11));
    test_corpus(d, "boundary-" + std::to_string(sz));
  }

  // Random-length fuzz roundtrips.
  for (int t = 0; t < 200; ++t) {
    size_t len = rng() % 40000;
    std::vector<uint8_t> d(len);
    int mode = static_cast<int>(rng() % 3);
    for (size_t i = 0; i < len; ++i) {
      if (mode == 0) d[i] = static_cast<uint8_t>(rng());
      else if (mode == 1) d[i] = static_cast<uint8_t>(rng() % 4);
      else d[i] = static_cast<uint8_t>('A' + (i % 20));
    }
    check(roundtrip(d, 1 + static_cast<int>(rng() % 9), "fuzz"),
          "fuzz roundtrip t=" + std::to_string(t));
  }

  // Big-block roundtrips (binary-tree match finder at levels >= 7 on blocks
  // > 1 MiB): text-like, repetitive, and near-random thirds in one 2 MiB
  // block, at a greedy BT level and an optimal-parser BT level. Kept small:
  // the optimal levels are slow by design.
  {
    std::vector<uint8_t> d(3u << 19);
    for (size_t i = 0; i < d.size(); ++i)
      d[i] = static_cast<uint8_t>('a' + (i * 7 + (i >> 3)) % 26);
    for (size_t i = 1u << 19; i < (1u << 19) + 150000; ++i) d[i] = d[i - 65537];
    for (size_t i = 2u << 19; i < d.size(); i += 3)
      d[i] = static_cast<uint8_t>(rng());
    check(roundtrip_blocklog(d, 9, 21, "bt-big-block"), "bt-big-block L9");
    check(roundtrip_blocklog(d, 11, 21, "bt-big-block"), "bt-big-block L11");
    // Levels 3-6 with blocks over 256 KiB take the row-based tag finder path.
    check(roundtrip_blocklog(d, 3, 21, "row-big-block"), "row-big-block L3");
    check(roundtrip_blocklog(d, 5, 21, "row-big-block"), "row-big-block L5");
    check(roundtrip_blocklog(d, 6, 20, "row-big-block"), "row-big-block L6");
  }

  // Decoder robustness: corrupted / truncated input must not crash and must
  // report an error rather than succeed.
  {
    std::vector<uint8_t> d(20000);
    for (auto& b : d) b = static_cast<uint8_t>(rng());
    auto src = to_bytes(d);
    std::vector<std::byte> comp(spectra8::compress_bound(src.size()));
    auto cr = spectra8::compress(src, comp, 6);
    comp.resize(cr.value);

    for (int t = 0; t < 300; ++t) {
      std::vector<std::byte> bad = comp;
      // Truncate and/or flip bytes.
      if (rng() & 1 && bad.size() > 8)
        bad.resize(bad.size() - (rng() % bad.size()));
      for (int f = 0; f < 4 && !bad.empty(); ++f) {
        size_t idx = rng() % bad.size();
        bad[idx] = static_cast<std::byte>(rng());
      }
      std::vector<std::byte> dec(src.size() + 16);
      auto dr = spectra8::decompress(bad, dec);
      // Either it errors out, or (rarely) it happens to decode to the right
      // size; either way it must not crash and must not over-write.
      (void)dr;
    }
    check(true, "decoder fuzz did not crash");
  }

  // LZP roundtrip test: repetitive text patterns and match-marker escapes
  {
    std::string sample = "The quick brown fox jumps over the lazy dog. "
                         "Compression suites evaluate algorithms on diverse datasets. "
                         "The quick brown fox jumps over the lazy dog. "
                         "Compression suites evaluate algorithms on diverse datasets. ";
    while (sample.size() < 100000) sample += sample;
    std::vector<uint8_t> d(sample.begin(), sample.end());
    // Also inject some 0xF2 markers at the beginning (val < 0) and in the middle (val >= 0)
    for (size_t i = 0; i < 16; ++i) d[i] = 0xF2;
    for (size_t i = 500; i < 600; ++i) d[i] = 0xF2;
    std::vector<uint8_t> comp(d.size() + 1024);
    std::size_t out_len = 0;
    bool ok = spectra8::detail::lzp_compress(d.data(), d.size(), comp.data(), comp.size(), out_len);
    check(ok, "lzp_compress repetitive text");
    check(out_len < d.size(), "lzp_compress size reduction");

    std::vector<uint8_t> dec(d.size());
    auto st = spectra8::detail::lzp_decompress(comp.data(), out_len, dec.data(), dec.size());
    check(st == spectra8::Status::Ok, "lzp_decompress status");
    check(dec == d, "lzp bit-exact roundtrip");
  }

  // BWT-CM roundtrip test
  {
    std::string sample = "To be, or not to be, that is the question: "
                         "Whether 'tis nobler in the mind to suffer "
                         "The slings and arrows of outrageous fortune, "
                         "Or to take arms against a sea of troubles";
    while (sample.size() < 50000) sample += sample;
    std::vector<uint8_t> d(sample.begin(), sample.end());
    std::vector<uint8_t> comp;
    bool ok = spectra8::detail::bwt_cm_encode(d.data(), d.size(), comp);
    check(ok, "bwt_cm_encode");
    check(comp.size() < d.size(), "bwt_cm_encode compression");

    std::vector<uint8_t> dec(d.size());
    auto st = spectra8::detail::bwt_cm_decode(comp.data(), comp.size(), dec.data(), dec.size());
    check(st == spectra8::Status::Ok, "bwt_cm_decode status");
    check(dec == d, "bwt_cm bit-exact roundtrip");
  }

  // Flash L1 (raw interleaved literals) and L5 (optional ratio header).
  {
    std::vector<uint8_t> d;
    const char* pat = "the quick brown fox jumps over the lazy dog. ";
    for (int i = 0; i < 5000; ++i)
      d.insert(d.end(), pat, pat + std::strlen(pat));
    check(roundtrip_codec(d, spectra8::Codec::Flash, 1, "flash-rep"),
          "flash L1 repetitive");
    check(roundtrip_codec(d, spectra8::Codec::Flash, 5, "flash-rep"),
          "flash L5 repetitive");

    std::vector<uint8_t> residual(4 * 12000);
    for (size_t g = 0; g < 12000; ++g) {
      residual[g * 4 + 0] = 7;
      residual[g * 4 + 1] = static_cast<uint8_t>(g);
      residual[g * 4 + 2] = static_cast<uint8_t>(g >> 8);
      residual[g * 4 + 3] = static_cast<uint8_t>(g >> 16);
    }
    check(roundtrip_codec(residual, spectra8::Codec::Flash, 1, "flash-res"),
          "flash L1 residual");
    check(roundtrip_codec(residual, spectra8::Codec::Flash, 5, "flash-res"),
          "flash L5 residual");
  }

  // Laser roundtrip tests across patterns and levels (1, 3, 6, 9)
  {
    auto test_laser_pattern = [&](const std::vector<uint8_t>& data, const std::string& name) {
      for (int lvl : {1, 3, 6, 9}) {
        auto src = to_bytes(data);
        spectra8::CompressParams cp;
        cp.codec = spectra8::Codec::Laser;
        cp.level = lvl;
        std::vector<std::byte> comp(spectra8::compress_bound(src.size(), cp.codec));
        auto cr = spectra8::compress(src, comp, cp);
        check(cr.ok(), "Laser " + name + " L" + std::to_string(lvl) + " compress");
        if (!cr.ok()) continue;
        comp.resize(cr.value);
        auto cs = spectra8::get_frame_content_size(comp);
        check(cs && *cs == src.size(), "Laser " + name + " L" + std::to_string(lvl) + " content size");
        std::vector<std::byte> dec(src.size());
        auto dr = spectra8::decompress(comp, dec);
        if (!dr.ok() || dr.value != src.size()) {
          std::fprintf(stderr, "Laser %s L%d decompress failed: status=%s val=%zu src_sz=%zu\n",
                       name.c_str(), lvl, std::string(spectra8::status_message(dr.status)).c_str(),
                       dr.value, src.size());
        }
        check(dr.ok() && dr.value == src.size(), "Laser " + name + " L" + std::to_string(lvl) + " decompress");
        check(std::memcmp(src.data(), dec.data(), src.size()) == 0,
              "Laser " + name + " L" + std::to_string(lvl) + " bit-exact");
      }
    };

    // 1. Zeros & edge sizes
    test_laser_pattern({}, "empty");
    test_laser_pattern({0}, "single-zero");
    test_laser_pattern(std::vector<uint8_t>(65536, 0), "zeros-64k");

    // 2. Repetitive text & patterns
    {
      std::string rep = "Laser compressor LZMA2-class 5-stream decoupled range coder. ";
      std::vector<uint8_t> d;
      for (int i = 0; i < 2000; ++i)
        d.insert(d.end(), rep.begin(), rep.end());
      test_laser_pattern(d, "repetitive");
    }

    // 3. Incompressible random
    {
      std::vector<uint8_t> rnd(32768);
      for (auto& b : rnd) b = static_cast<uint8_t>(rng());
      test_laser_pattern(rnd, "random");
    }

    // 4. Binary machine-code like pattern (calls E8 rel32, jumps E9 rel32, movs 48 89)
    {
      std::vector<uint8_t> bin(80000);
      for (size_t i = 0; i + 8 < bin.size(); i += 8) {
        bin[i] = 0xE8;  // call
        uint32_t rel = static_cast<uint32_t>((i * 13) & 0x1FFFF);
        std::memcpy(&bin[i + 1], &rel, 4);
        bin[i + 5] = 0x48;
        bin[i + 6] = 0x89;
        bin[i + 7] = static_cast<uint8_t>(i & 0x7);
      }
      test_laser_pattern(bin, "binary-x86");
    }

    // 5. Realistic code/text pattern
    {
      std::string code =
          "#include <iostream>\n"
          "int main(int argc, char** argv) {\n"
          "  for (int i = 0; i < argc; ++i) {\n"
          "    std::cout << \"arg: \" << argv[i] << std::endl;\n"
          "  }\n"
          "  return 0;\n"
          "}\n";
      std::vector<uint8_t> d;
      while (d.size() < 70000)
        d.insert(d.end(), code.begin(), code.end());
      test_laser_pattern(d, "realistic-code");
    }
  }

  if (g_failures == 0) {
    std::printf("All %d checks passed.\n", g_checks);
    return 0;
  }
  std::fprintf(stderr, "%d/%d checks FAILED.\n", g_failures, g_checks);
  return 1;
}
