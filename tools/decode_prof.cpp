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

// Encode/decode throughput harness for the BWT path. Builds a few 128 KiB
// columnar datasets, compresses each with BWT L9, then times decode (and
// encode) so we can iterate on entropy-coder speed without the full verdict.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "spectra8/spectra8.hpp"

namespace {

using Clock = std::chrono::steady_clock;

struct Set {
  std::vector<std::byte> src;
  std::vector<std::byte> comp;
};

std::vector<std::byte> make_i32(size_t n, uint64_t seed) {
  std::vector<std::byte> v(n);
  std::mt19937_64 rng(seed);
  std::binomial_distribution<int> bd(50000, 0.5);
  auto* p = reinterpret_cast<int32_t*>(v.data());
  for (size_t i = 0; i < n / 4; ++i) p[i] = bd(rng);
  return v;
}
std::vector<std::byte> make_i16(size_t n, uint64_t seed) {
  std::vector<std::byte> v(n);
  std::mt19937_64 rng(seed);
  std::binomial_distribution<int> bd(4000, 0.30);
  auto* p = reinterpret_cast<int16_t*>(v.data());
  for (size_t i = 0; i < n / 2; ++i) p[i] = static_cast<int16_t>(bd(rng));
  return v;
}
std::vector<std::byte> make_f64(size_t n, uint64_t seed) {
  std::vector<std::byte> v(n);
  std::mt19937_64 rng(seed);
  std::binomial_distribution<int> bd(4096, 0.50);
  auto* p = reinterpret_cast<double*>(v.data());
  for (size_t i = 0; i < n / 8; ++i) p[i] = 100.0 * bd(rng) / 4096.0;
  return v;
}

}  // namespace

int main(int argc, char** argv) {
  const size_t n = 128 * 1024;
  const int seconds = (argc > 1) ? std::atoi(argv[1]) : 6;

  std::vector<Set> sets;
  for (auto fn : {make_i32, make_i16, make_f64}) {
    Set s;
    s.src = fn(n, 999);
    sets.push_back(std::move(s));
  }

  spectra8::CompressParams cp;
  cp.level = 9;
  cp.codec = spectra8::Codec::Bwt;
  cp.threads = 1;
  spectra8::DecompressParams dp{1};

  double sum_ratio = 0;
  for (auto& s : sets) {
    s.comp.resize(spectra8::compress_bound(n));
    auto r = spectra8::compress(s.src, s.comp, cp);
    if (!r) { std::fprintf(stderr, "compress failed\n"); return 1; }
    s.comp.resize(r.value);
    sum_ratio += double(n) / double(r.value);
    std::vector<std::byte> out(n);
    auto d = spectra8::decompress(
        std::span<const std::byte>(s.comp.data(), s.comp.size()), out, dp);
    if (!d || d.value != n || std::memcmp(out.data(), s.src.data(), n) != 0) {
      std::fprintf(stderr, "decode mismatch\n");
      return 1;
    }
  }
  std::printf("avg ratio %.4f over %zu sets\n", sum_ratio / sets.size(),
              sets.size());

  std::vector<std::byte> out(n);

  // Decode throughput.
  {
    auto end = Clock::now() + std::chrono::seconds(seconds);
    uint64_t bytes = 0;
    volatile uint8_t sink = 0;
    while (Clock::now() < end) {
      for (auto& s : sets) {
        (void)spectra8::decompress(
            std::span<const std::byte>(s.comp.data(), s.comp.size()), out, dp);
        sink ^= static_cast<uint8_t>(out[0]);
        bytes += n;
      }
    }
    // Re-measure precise elapsed for the loop just run.
    auto t0 = Clock::now();
    uint64_t b2 = 0;
    for (int rep = 0; rep < 50; ++rep)
      for (auto& s : sets) {
        (void)spectra8::decompress(
            std::span<const std::byte>(s.comp.data(), s.comp.size()), out, dp);
        sink ^= static_cast<uint8_t>(out[0]);
        b2 += n;
      }
    double secs =
        std::chrono::duration<double>(Clock::now() - t0).count();
    std::printf("decode %.1f MB/s  (sink=%u)\n",
                double(b2) / 1e6 / secs, unsigned(sink));
  }

  // Encode throughput.
  {
    std::vector<std::byte> comp(spectra8::compress_bound(n));
    auto t0 = Clock::now();
    uint64_t b2 = 0;
    for (int rep = 0; rep < 20; ++rep)
      for (auto& s : sets) {
        (void)spectra8::compress(s.src, comp, cp);
        b2 += n;
      }
    double secs = std::chrono::duration<double>(Clock::now() - t0).count();
    std::printf("encode %.1f MB/s\n", double(b2) / 1e6 / secs);
  }
  return 0;
}
