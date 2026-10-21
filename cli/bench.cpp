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

// spectra8 benchmark (invoked as `spectra8 bench`).
//
// Runs for a fixed wall-clock budget, measuring compression ratio and
// compress/decompress throughput across several data profiles and a few
// representative levels, then prints a formatted report.
#include "bench.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include "spectra8/spectra8.hpp"
#include "spectra8/page_service.hpp"
#include "prism/bwt.hpp"
#include "prism/bwt_decode.hpp"
#include "util/byte_rle.hpp"
#include "util/pack_filter.hpp"  // internal: dict / for / alp external staging
#include "util/transform.hpp"    // internal: report the auto-detected filter

#ifdef SPECTRA8_HAVE_ZSTD
#include <zstd.h>
#endif

#ifdef SPECTRA8_HAVE_LZMA
#include <lzma.h>
#endif

#ifdef SPECTRA8_HAVE_BZIP3
#include <libbz3.h>
#endif

namespace spectra8::detail {
struct FlashStats {
  usize in_len = 0;
  usize out_len = 0;
  usize lit_bytes = 0;
  usize huff_lit_bytes = 0;
  usize token_bytes = 0;
  usize offset_bytes = 0;
  usize ext_bytes = 0;
  usize num_matches = 0;
  usize num_short_offsets = 0;
  usize num_wide_offsets = 0;
  usize num_rep0 = 0;
  usize num_rep1 = 0;
};
FlashStats get_last_flash_stats() noexcept;
}

namespace spectra8_cli {
namespace {

#ifdef SPECTRA8_HAVE_BZIP3
// bzip3 has no -0..-9 levels; block size (65 KiB .. 511 MiB) controls ratio/speed.
// Verdict picks an intermediate vs a largest practical block for the dataset size.
constexpr uint32_t kBzip3BlockMin = 65u * 1024u;
constexpr uint32_t kBzip3Block768KiB = 768u * 1024u;
constexpr uint32_t kBzip3Block8MiB = 8u * 1024u * 1024u;
constexpr uint32_t kBzip3Block64MiB = 64u * 1024u * 1024u;

uint32_t bzip3_verdict_block_intermediate(std::size_t bytes) {
  uint32_t blk;
  if (bytes > kBzip3Block8MiB)
    blk = kBzip3Block8MiB;
  else if (bytes > kBzip3Block768KiB)
    blk = kBzip3Block768KiB;
  else
    blk = kBzip3BlockMin;

  if (blk == bytes) {
    if (blk > kBzip3BlockMin)
      blk /= 2;
    else
      blk = kBzip3Block768KiB;
  }
  return blk;
}

uint32_t bzip3_verdict_block_highest(std::size_t bytes) {
  // Do not pass block_size == in_size: bz3_compress accepts it but
  // bz3_decompress fails.
  uint32_t blk = (bytes >= kBzip3Block8MiB) ? kBzip3Block64MiB : kBzip3Block8MiB;
  if (blk == bytes) {
    blk = (blk == kBzip3Block64MiB) ? (32u * 1024u * 1024u) : kBzip3Block64MiB;
  }
  return blk;
}

void bzip3_verdict_label(char* buf, std::size_t bufsz, uint32_t blk,
                         const char* flt_sfx) {
  const char* sfx = flt_sfx;
  if (blk >= 1024u * 1024u)
    std::snprintf(buf, bufsz, "bzip3 %uM%s", blk / (1024u * 1024u), sfx);
  else if (blk >= 1024u)
    std::snprintf(buf, bufsz, "bzip3 %uk%s", blk / 1024u, sfx);
  else
    std::snprintf(buf, bufsz, "bzip3 %u%s", blk, sfx);
}
#endif  // SPECTRA8_HAVE_BZIP3

using clock_t_ = std::chrono::steady_clock;
double now_s() {
  return std::chrono::duration<double>(clock_t_::now().time_since_epoch()).count();
}

struct Dataset {
  std::string name;
  std::vector<std::byte> data;
  bool avg = true;  // included in the averaged summary rows
};

void fill_text(std::vector<std::byte>& d, std::mt19937_64& rng) {
  static const char* words[] = {
      "the ", "quick ", "brown ", "fox ", "jumps ", "over ", "lazy ", "dog ",
      "compression ", "entropy ", "codec ", "stream ", "buffer ", "offset ",
      "literal ", "match ", "window ", "frame ", "block ", "ratio "};
  size_t i = 0;
  while (i < d.size()) {
    const char* w = words[rng() % 20];
    size_t l = std::strlen(w);
    for (size_t k = 0; k < l && i < d.size(); ++k) d[i++] = static_cast<std::byte>(w[k]);
    if (rng() % 12 == 0 && i < d.size()) d[i++] = static_cast<std::byte>('\n');
  }
}

void fill_source(std::vector<std::byte>& d, std::mt19937_64& rng) {
  static const char* snippet =
      "for (size_t i = 0; i < n; ++i) {\n"
      "    auto v = compute(data[i], offset);\n"
      "    if (v > threshold) { results.push_back(v); }\n"
      "}\n";
  size_t sl = std::strlen(snippet);
  size_t i = 0;
  while (i < d.size()) {
    for (size_t k = 0; k < sl && i < d.size(); ++k) d[i++] = static_cast<std::byte>(snippet[k]);
    if (rng() % 5 == 0 && i < d.size()) d[i++] = static_cast<std::byte>(' ');
  }
}

void fill_json(std::vector<std::byte>& d, std::mt19937_64& rng) {
  size_t i = 0;
  char buf[128];
  while (i < d.size()) {
    int n = std::snprintf(buf, sizeof(buf),
                          "{\"id\":%u,\"name\":\"item\",\"value\":%u,\"ok\":%s},\n",
                          static_cast<unsigned>(rng() % 100000),
                          static_cast<unsigned>(rng() % 1000),
                          (rng() & 1) ? "true" : "false");
    for (int k = 0; k < n && i < d.size(); ++k) d[i++] = static_cast<std::byte>(buf[k]);
  }
}

void fill_binary(std::vector<std::byte>& d, std::mt19937_64& rng) {
  // Records of a monotonically increasing counter + small noise: the high bytes
  // are very low entropy, the low bytes are noisy.
  uint32_t ctr = 0;
  size_t i = 0;
  while (i + 8 <= d.size()) {
    ctr += 1 + static_cast<uint32_t>(rng() % 3);
    uint32_t a = ctr;
    uint32_t b = static_cast<uint32_t>(rng() & 0xFFFF);
    std::memcpy(&d[i], &a, 4);
    std::memcpy(&d[i + 4], &b, 4);
    i += 8;
  }
}

void fill_random(std::vector<std::byte>& d, std::mt19937_64& rng) {
  for (auto& b : d) b = static_cast<std::byte>(rng());
}

// Physics-style columnar data: arrays of a single numeric type whose values
// follow a binomial distribution (counts, occupancies, etc.). The byte layout
// is what stresses the codec: the high-order bytes of each element are nearly
// constant while the low-order bytes carry the statistical noise.
template <class T>
void fill_binomial_int(std::vector<std::byte>& d, std::mt19937_64& rng,
                       long long trials, double prob) {
  std::binomial_distribution<long long> dist(trials, prob);
  size_t i = 0;
  while (i + sizeof(T) <= d.size()) {
    T v = static_cast<T>(dist(rng));
    std::memcpy(&d[i], &v, sizeof(T));
    i += sizeof(T);
  }
}

// Same idea but stored as floating point: a binomial count normalized to a
// physical-looking measurement in [0, scale]. This fills the mantissa with
// genuine noise (low bytes high-entropy) while the exponent bytes stay
// structured, which is representative of real float/double physics columns.
template <class T>
void fill_binomial_float(std::vector<std::byte>& d, std::mt19937_64& rng,
                         long long trials, double prob, double scale) {
  std::binomial_distribution<long long> dist(trials, prob);
  const double inv = scale / static_cast<double>(trials);
  size_t i = 0;
  while (i + sizeof(T) <= d.size()) {
    T v = static_cast<T>(static_cast<double>(dist(rng)) * inv);
    std::memcpy(&d[i], &v, sizeof(T));
    i += sizeof(T);
  }
}

// Gaussian variants of the physics columns. Parameters are chosen to match
// the binomial(50000,0.5) profile's moments (mean 25000, sigma ~112) so the
// byte-level structure is comparable: near-constant high bytes, noisy low
// bytes. Values are rounded to the nearest integer so the cardinality stays
// realistic for count-like columns.
// Note: std::normal_distribution's output sequence is implementation-defined
// (like the mt19937 note elsewhere), so exact bytes differ across platforms;
// seeds stay deterministic per (base seed, dataset name).
template <class T>
void fill_gaussian_int(std::vector<std::byte>& d, std::mt19937_64& rng,
                       double mean, double sigma) {
  std::normal_distribution<double> dist(mean, sigma);
  size_t i = 0;
  while (i + sizeof(T) <= d.size()) {
    T v = static_cast<T>(std::llround(dist(rng)));
    std::memcpy(&d[i], &v, sizeof(T));
    i += sizeof(T);
  }
}

// Gaussian float columns are kept continuous (no quantization to a count
// grid): a calibrated analog measurement (energy, time, position) is a real
// number with full mantissa noise, which is just as physics-representative as
// the quantized binomial profile. The unbounded cardinality is deliberate: it
// defeats the dictionary-remap filters and exercises the shuffle/delta
// transforms instead, giving a useful contrast to the low-cardinality
// binomial float columns.
template <class T>
void fill_gaussian_float(std::vector<std::byte>& d, std::mt19937_64& rng,
                         double mean, double sigma) {
  std::normal_distribution<double> dist(mean, sigma);
  size_t i = 0;
  while (i + sizeof(T) <= d.size()) {
    T v = static_cast<T>(dist(rng));
    std::memcpy(&d[i], &v, sizeof(T));
    i += sizeof(T);
  }
}

// Poisson variants: raw event/hit counts (lambda 25000 gives sigma ~158,
// slightly wider than the matched binomial's ~112, so these columns should
// compress marginally worse). Output sequence is implementation-defined, same
// caveat as the gaussian fills.
template <class T>
void fill_poisson_int(std::vector<std::byte>& d, std::mt19937_64& rng,
                      long long lambda) {
  std::poisson_distribution<long long> dist(static_cast<double>(lambda));
  size_t i = 0;
  while (i + sizeof(T) <= d.size()) {
    T v = static_cast<T>(dist(rng));
    std::memcpy(&d[i], &v, sizeof(T));
    i += sizeof(T);
  }
}

// Poisson float columns mirror the binomial float profile: a count quantized
// on the lambda grid, scaled to a physical-looking [0, ~100] range (lambda
// 2048 * 100/4096 has mean 50, like the binomial(4096,0.5) profile).
template <class T>
void fill_poisson_float(std::vector<std::byte>& d, std::mt19937_64& rng,
                        long long lambda, double scale) {
  std::poisson_distribution<long long> dist(static_cast<double>(lambda));
  size_t i = 0;
  while (i + sizeof(T) <= d.size()) {
    T v = static_cast<T>(static_cast<double>(dist(rng)) * scale);
    std::memcpy(&d[i], &v, sizeof(T));
    i += sizeof(T);
  }
}

// Continuous float waveform / analog pulse: calorimeter shower pulse, drift
// times, or analog sensor telemetry with correlated adjacent samples.
// Adjacent IEEE-754 floats share sign, exponent, and high mantissa bits,
// directly testing FilterId::XorShuffle32.
inline void fill_wave_f32(std::vector<std::byte>& d, std::mt19937_64& rng) {
  std::uniform_real_distribution<float> noise(-0.005f, 0.005f);
  double t = 0.0;
  size_t i = 0;
  while (i + 4 <= d.size()) {
    float v = static_cast<float>(125.0 + 50.0 * std::sin(t * 0.015) + 20.0 * std::cos(t * 0.04)) + noise(rng);
    std::memcpy(&d[i], &v, 4);
    i += 4;
    t += 1.0;
  }
  for (; i < d.size(); ++i) d[i] = static_cast<std::byte>(rng());
}

// Double-precision continuous waveform: particle trajectory coordinates or
// sensor timeseries with smooth velocity. Targets FilterId::XorShuffle64.
inline void fill_wave_f64(std::vector<std::byte>& d, std::mt19937_64& rng) {
  std::uniform_real_distribution<double> noise(-0.001, 0.001);
  double t = 0.0;
  size_t i = 0;
  while (i + 8 <= d.size()) {
    double v = 500.0 + 150.0 * std::sin(t * 0.008) + 45.0 * std::cos(t * 0.02) + noise(rng);
    std::memcpy(&d[i], &v, 8);
    i += 8;
    t += 1.0;
  }
  for (; i < d.size(); ++i) d[i] = static_cast<std::byte>(rng());
}

// Linear ramp with steady cadence and small signed jitter: e.g. event counter,
// steady DAQ clock sequence, or detector tracker channel. Second-order delta
// (Delta2Shuffle32 / ZigzagDelta2Shuffle32 / Crystal kTrend) zeroes out the slope.
inline void fill_ramp_i32(std::vector<std::byte>& d, std::mt19937_64& rng) {
  std::uniform_int_distribution<int32_t> jitter(-2, 2);
  int32_t val = 100000;
  size_t i = 0;
  while (i + 4 <= d.size()) {
    int32_t v = val + jitter(rng);
    std::memcpy(&d[i], &v, 4);
    i += 4;
    val += 25;
  }
  for (; i < d.size(); ++i) d[i] = static_cast<std::byte>(rng());
}

// 64-bit monotonic timestamps with steady rate and clock jitter: e.g. DAQ
// event timestamps or storage logs. Targets FilterId::Delta2Shuffle64.
inline void fill_ts_u64(std::vector<std::byte>& d, std::mt19937_64& rng) {
  std::uniform_int_distribution<int64_t> jitter(-3, 3);
  uint64_t ts = 1700000000000ULL;
  size_t i = 0;
  while (i + 8 <= d.size()) {
    uint64_t v = static_cast<uint64_t>(static_cast<int64_t>(ts) + jitter(rng));
    std::memcpy(&d[i], &v, 8);
    i += 8;
    ts += 50;
  }
  for (; i < d.size(); ++i) d[i] = static_cast<std::byte>(rng());
}

// Decimal-scaled float measurements: values rounded to 2 decimal places
// (e.g. currency, temperature, or calibration factors). Targets FilterId::Alp64.
inline void fill_dec_f64(std::vector<std::byte>& d, std::mt19937_64& rng) {
  std::uniform_int_distribution<int32_t> cents(100, 25000);
  size_t i = 0;
  while (i + 8 <= d.size()) {
    double v = static_cast<double>(cents(rng)) * 0.01;
    std::memcpy(&d[i], &v, 8);
    i += 8;
  }
  for (; i < d.size(); ++i) d[i] = static_cast<std::byte>(rng());
}

// Compiled x86_64 machine code with relative CALL (0xE8) and JMP (0xE9)
// instructions to common subroutine targets. Targets FilterId::BcjX86.
inline void fill_code_x86(std::vector<std::byte>& d, std::mt19937_64& rng) {
  const uint32_t targets[16] = {
      0x00401000, 0x00401250, 0x00401800, 0x00402100,
      0x00403500, 0x00404000, 0x00405200, 0x00406400,
      0x00408000, 0x00409100, 0x0040A000, 0x0040B500,
      0x0040C000, 0x0040D300, 0x0040E000, 0x0040F200
  };
  static const uint8_t kOpcodes[] = {
      0x55,                                     // push rbp
      0x48, 0x89, 0xE5,                         // mov rbp, rsp
      0x48, 0x83, 0xEC, 0x20,                   // sub rsp, 32
      0x48, 0x8B, 0x05, 0x10, 0x00, 0x00, 0x00, // mov rax, [rip+16]
      0x85, 0xC0,                               // test eax, eax
      0x74, 0x0A,                               // jz +10
      0x89, 0xC7,                               // mov edi, eax
      0x48, 0x89, 0xF8,                         // mov rax, rdi
      0x31, 0xC0,                               // xor eax, eax
      0xC9,                                     // leave
      0xC3,                                     // ret
      0x90, 0x90, 0x90, 0x90                    // nop
  };
  size_t i = 0;
  while (i + 16 <= d.size()) {
    if ((rng() % 5) == 0 && i + 5 <= d.size()) {
      const uint8_t op = (rng() % 4 == 0) ? 0xE9 : 0xE8;
      d[i] = static_cast<std::byte>(op);
      const uint32_t target = targets[rng() % 16];
      const uint32_t pc = static_cast<uint32_t>(i + 5);
      const uint32_t rel = target - pc;
      std::memcpy(&d[i + 1], &rel, 4);
      i += 5;
    } else {
      const size_t op_idx = rng() % (sizeof(kOpcodes) - 4);
      const size_t op_len = 1 + (rng() % 4);
      for (size_t k = 0; k < op_len && i < d.size(); ++k) {
        d[i++] = static_cast<std::byte>(kOpcodes[(op_idx + k) % sizeof(kOpcodes)]);
      }
    }
  }
  while (i < d.size()) d[i++] = static_cast<std::byte>(0x90);
}

// Attempts to load up to `max_bytes` from enwik8.
// Searches ENWIK8_PATH, SPECTRA8_ENWIK8, ./enwik8, ~/.cache/spectra8/enwik8, /tmp/enwik8.
// If missing, unzips an existing cache zip or attempts download from Matt Mahoney's site.
inline std::vector<std::byte> load_enwik8(size_t max_bytes) {
  std::vector<std::string> candidates;
  if (const char* p = std::getenv("ENWIK8_PATH")) candidates.push_back(p);
  if (const char* p = std::getenv("SPECTRA8_ENWIK8")) candidates.push_back(p);
  candidates.push_back("./enwik8");
  if (const char* home = std::getenv("HOME")) {
    candidates.push_back(std::string(home) + "/.cache/spectra8/enwik8");
  }
  candidates.push_back("/tmp/enwik8");
  candidates.push_back("/var/tmp/enwik8");

  auto try_read = [&](const std::string& path) -> std::vector<std::byte> {
    std::error_code ec;
    if (!std::filesystem::exists(path, ec) || std::filesystem::is_directory(path, ec)) {
      return {};
    }
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return {};
    std::fseek(f, 0, SEEK_END);
    long sz = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (sz <= 0) {
      std::fclose(f);
      return {};
    }
    size_t to_read = (max_bytes == 0) ? static_cast<size_t>(sz)
                                       : std::min(max_bytes, static_cast<size_t>(sz));
    std::vector<std::byte> buf(to_read);
    size_t nr = std::fread(buf.data(), 1, to_read, f);
    std::fclose(f);
    buf.resize(nr);
    return buf;
  };

  for (const auto& c : candidates) {
    auto data = try_read(c);
    if (!data.empty()) return data;
  }

  // Not found: check if .zip is available to unzip, or try downloading.
  std::string cache_dir = "/tmp";
  if (const char* home = std::getenv("HOME")) {
    cache_dir = std::string(home) + "/.cache/spectra8";
  }
  std::string cache_enwik8 = cache_dir + "/enwik8";
  std::string cache_zip = cache_dir + "/enwik8.zip";

  std::error_code ec;
  if (std::filesystem::exists(cache_zip, ec)) {
    std::string cmd = "unzip -q -o \"" + cache_zip + "\" -d \"" + cache_dir + "\" 2>/dev/null";
    (void)std::system(cmd.c_str());
    auto data = try_read(cache_enwik8);
    if (!data.empty()) return data;
  }

  // Attempt download via curl
  std::printf(" enwik8 not found locally; downloading from https://mattmahoney.net/dc/enwik8.zip ...\n");
  std::string dl_cmd = "mkdir -p \"" + cache_dir + "\" && curl -sSL --connect-timeout 15 -o \"" +
                       cache_zip + "\" https://mattmahoney.net/dc/enwik8.zip && unzip -q -o \"" +
                       cache_zip + "\" -d \"" + cache_dir + "\" 2>/dev/null";
  int rc = std::system(dl_cmd.c_str());
  if (rc == 0) {
    auto data = try_read(cache_enwik8);
    if (!data.empty()) return data;
  }

  std::fprintf(stderr,
               "spectra8 bench: warning: could not load or download enwik8 dataset.\n"
               "                Falling back to synthetic text. Set ENWIK8_PATH or download\n"
               "                https://mattmahoney.net/dc/enwik8.zip to ~/.cache/spectra8/enwik8.\n");
  return {};
}

// Peaked integer columns: a dense core plus a long tail of rare values, stored
// as little-endian scalars. This is the spectrum Grating codes as one symbol
// per integer, which byte-plane columns (the bin/gau/poi sets) do not have.
void fill_laplace_int(std::vector<std::byte>& d, std::mt19937_64& rng, int width,
                      double p) {
  std::geometric_distribution<int> mag(p);
  size_t i = 0;
  while (i + static_cast<size_t>(width) <= d.size()) {
    int m = mag(rng);
    if (m > 1000000) m = 1000000;
    std::int64_t v = (rng() & 1) ? -static_cast<std::int64_t>(m) : m;
    std::memcpy(&d[i], &v, static_cast<size_t>(width));
    i += static_cast<size_t>(width);
  }
}

void fill_tail_i32(std::vector<std::byte>& d, std::mt19937_64& rng) {
  std::uniform_int_distribution<int> coin(0, 99);
  std::uniform_int_distribution<int> core(0, 31);
  std::uniform_int_distribution<int> tail(0, 200000);
  size_t i = 0;
  while (i + 4 <= d.size()) {
    std::int32_t v = coin(rng) < 92 ? core(rng) : tail(rng);
    std::memcpy(&d[i], &v, 4);
    i += 4;
  }
}

void fill_zipf_i32(std::vector<std::byte>& d, std::mt19937_64& rng) {
  std::uniform_real_distribution<double> u(1e-9, 1.0);
  size_t i = 0;
  while (i + 4 <= d.size()) {
    const double x = std::pow(u(rng), -0.65);
    std::int32_t v = static_cast<std::int32_t>(std::min(200000.0, x));
    std::memcpy(&d[i], &v, 4);
    i += 4;
  }
}

void fill_cat_i16(std::vector<std::byte>& d, std::mt19937_64& rng) {
  std::uniform_int_distribution<int> coin(0, 99);
  std::uniform_int_distribution<int> core(0, 23);
  size_t i = 0;
  std::uint16_t uniq = 1000;
  while (i + 2 <= d.size()) {
    std::uint16_t v = coin(rng) < 96 ? static_cast<std::uint16_t>(core(rng)) : uniq++;
    if (uniq == 0) uniq = 1000;
    std::memcpy(&d[i], &v, 2);
    i += 2;
  }
}

void fill_ids_i64(std::vector<std::byte>& d, std::mt19937_64& rng) {
  std::uniform_int_distribution<int> coin(0, 99);
  std::uniform_int_distribution<int> core(0, 63);
  size_t i = 0;
  std::uint64_t uniq = 1ull << 40;
  while (i + 8 <= d.size()) {
    std::uint64_t v;
    if (coin(rng) < 97)
      v = static_cast<std::uint64_t>(core(rng) * core(rng));
    else
      v = uniq++;
    std::memcpy(&d[i], &v, 8);
    i += 8;
  }
}

std::vector<Dataset> make_rawint_datasets(size_t bytes, std::uint64_t seed) {
  std::vector<Dataset> sets;
  auto add = [&](const char* name, auto fn) {
    std::mt19937_64 rng(seed ^ static_cast<std::uint64_t>(std::strlen(name)));
    Dataset ds;
    ds.name = name;
    ds.data.resize(bytes);
    fn(ds.data, rng);
    sets.push_back(std::move(ds));
  };
  add("lap-i16", [](std::vector<std::byte>& d, std::mt19937_64& r) {
    fill_laplace_int(d, r, 2, 0.12);
  });
  add("lap-i32", [](std::vector<std::byte>& d, std::mt19937_64& r) {
    fill_laplace_int(d, r, 4, 0.04);
  });
  add("tail-i32", fill_tail_i32);
  add("zipf-i32", fill_zipf_i32);
  add("cat-i16", fill_cat_i16);
  add("ids-i64", fill_ids_i64);
  return sets;
}

void fill_arm64_text(std::vector<std::byte>& d, std::mt19937_64& rng) {
  const std::uint32_t targets[8] = {0x40, 0x180, 0x400, 0x900,
                                    0x1200, 0x2000, 0x3000, 0x4000};
  size_t i = 0;
  auto put = [&](std::uint32_t w) {
    if (i + 4 > d.size()) return;
    std::memcpy(&d[i], &w, 4);
    i += 4;
  };
  while (i + 16 <= d.size()) {
    const int k = static_cast<int>(rng() % 6);
    if (k == 0) {
      put(0xA9BF7BFD);  // stp x29, x30, [sp, #-16]!
      put(0x910003FD);  // mov x29, sp
    } else if (k == 1) {
      put(0xD65F03C0);  // ret
    } else if (k == 2 || k == 3) {
      const std::uint32_t pc = static_cast<std::uint32_t>(i);
      const std::uint32_t target = targets[rng() % 8];
      std::int32_t rel = static_cast<std::int32_t>(target - pc) >> 2;
      rel &= 0x03FFFFFF;
      const std::uint32_t op = (k == 2) ? 0x94000000u : 0x14000000u;  // bl / b
      put(op | static_cast<std::uint32_t>(rel));
    } else if (k == 4) {
      put(0x90000000u | static_cast<std::uint32_t>(rng() & 0x7FFFF));  // adrp-like
      put(0x91000000u | static_cast<std::uint32_t>(rng() & 0x3FFFFF));
    } else {
      put(0xD2800000u | static_cast<std::uint32_t>((rng() & 0xFFFF) << 5));
    }
  }
  while (i < d.size()) d[i++] = static_cast<std::byte>(0);
}

void fill_rodata(std::vector<std::byte>& d, std::mt19937_64& rng) {
  static const char* words[] = {"compress", "symbol", "column", "decode",
                                "page",     "frame",  "offset", "value"};
  size_t i = 0;
  while (i + 16 <= d.size()) {
    if ((rng() % 3) == 0) {
      const char* w = words[rng() % 8];
      const size_t n = std::strlen(w);
      for (size_t k = 0; k < n && i < d.size(); ++k)
        d[i++] = static_cast<std::byte>(w[k]);
      if (i < d.size()) d[i++] = static_cast<std::byte>(0);
    } else {
      std::uint64_t ptr = 0x100000000ull + (rng() % 0x100000);
      std::memcpy(&d[i], &ptr, 8);
      i += 8;
    }
  }
  while (i < d.size()) d[i++] = static_cast<std::byte>(0);
}

bool is_shared_lib(const std::filesystem::path& p) {
  const std::string name = p.filename().string();
  if (name.size() > 6 && name.compare(name.size() - 6, 6, ".dylib") == 0) return true;
  if (name.size() > 3 && name.compare(name.size() - 3, 3, ".so") == 0) return true;
  return name.find(".so.") != std::string::npos;
}

std::vector<Dataset> make_binary_datasets(size_t bytes) {
  std::vector<std::filesystem::path> roots;
  if (const char* env = std::getenv("SPECTRA8_BINARIES_DIR")) roots.emplace_back(env);
  roots.emplace_back("/opt/homebrew/lib");
  roots.emplace_back("/usr/local/lib64");
  roots.emplace_back("/usr/local/lib");
  roots.emplace_back("/usr/lib64");
  roots.emplace_back("/usr/lib/x86_64-linux-gnu");
  roots.emplace_back("/usr/lib/aarch64-linux-gnu");
  roots.emplace_back("/usr/lib");

  struct Cand {
    std::filesystem::path path;
    std::uintmax_t size;
  };
  std::vector<Cand> cands;
  std::unordered_set<std::string> seen;
  for (const auto& root : roots) {
    std::error_code ec;
    if (!std::filesystem::is_directory(root, ec)) continue;
    const auto opt = std::filesystem::directory_options::skip_permission_denied;
    for (std::filesystem::directory_iterator it(root, opt, ec), end; !ec && it != end;
         it.increment(ec)) {
      if (!it->is_regular_file(ec)) continue;
      if (!is_shared_lib(it->path())) continue;
      const auto sz = it->file_size(ec);
      if (ec || sz < bytes || sz < 65536) continue;
      std::error_code cec;
      const auto canon = std::filesystem::canonical(it->path(), cec);
      const std::string key = cec ? it->path().string() : canon.string();
      if (!seen.insert(key).second) continue;
      cands.push_back(Cand{cec ? it->path() : canon, sz});
    }
  }
  std::sort(cands.begin(), cands.end(),
            [](const Cand& a, const Cand& b) { return a.size > b.size; });

  std::vector<Dataset> sets;
  for (const Cand& c : cands) {
    if (sets.size() >= 6) break;
    std::ifstream in(c.path, std::ios::binary);
    if (!in) continue;
    const std::uintmax_t off = (c.size - bytes) / 2;
    in.seekg(static_cast<std::streamoff>(off));
    Dataset ds;
    ds.name = c.path.filename().string();
    if (ds.name.size() > 32) ds.name.resize(32);
    ds.data.resize(bytes);
    in.read(reinterpret_cast<char*>(ds.data.data()), static_cast<std::streamsize>(bytes));
    if (static_cast<size_t>(in.gcount()) != bytes) continue;
    sets.push_back(std::move(ds));
  }
  if (sets.size() >= 2) return sets;

  std::fprintf(stderr,
               "spectra8 bench: no shared library >= %zu bytes under "
               "SPECTRA8_BINARIES_DIR, /opt/homebrew/lib, /usr/local/lib{64}, /usr/lib{64};\n"
               "                using synthetic arm64 / x86-64 / rodata pages\n",
               bytes);
  sets.clear();
  auto add = [&](const char* name, auto fn) {
    std::mt19937_64 rng(0xB171A71Eull ^ static_cast<std::uint64_t>(std::strlen(name)));
    Dataset ds;
    ds.name = name;
    ds.data.resize(bytes);
    fn(ds.data, rng);
    sets.push_back(std::move(ds));
  };
  add("arm64-text", fill_arm64_text);
  add("x86-text", fill_code_x86);
  add("rodata", fill_rodata);
  return sets;
}

std::vector<Dataset> make_datasets(size_t bytes, BenchDist dist,
                                   std::uint64_t seed) {
  std::vector<Dataset> sets;
  if (dist == BenchDist::RawInt) return make_rawint_datasets(bytes, seed);
  if (dist == BenchDist::Binaries) return make_binary_datasets(bytes);
  if (dist == BenchDist::Text) {
    Dataset ds;
    ds.name = "enwik8";
    ds.data = load_enwik8(bytes);
    if (ds.data.empty()) {
      const size_t sz = bytes > 0 ? bytes : (size_t{8} << 20);
      ds.data.resize(sz);
      std::mt19937_64 rng(seed ^ 0x54455854ULL);
      fill_text(ds.data, rng);
    } else if (bytes > 0 && ds.data.size() < bytes) {
      const size_t orig_sz = ds.data.size();
      ds.data.resize(bytes);
      for (size_t i = orig_sz; i < bytes; ++i) ds.data[i] = ds.data[i % orig_sz];
    }
    ds.avg = true;
    sets.push_back(std::move(ds));
    return sets;
  }
  using Fill = std::function<void(std::vector<std::byte>&, std::mt19937_64&)>;
  auto add = [&](const char* name, Fill fn) {
    std::mt19937_64 rng(seed ^ static_cast<std::uint64_t>(std::strlen(name)));
    Dataset ds;
    ds.name = name;
    ds.data.resize(bytes);
    fn(ds.data, rng);
    sets.push_back(std::move(ds));
  };
  add("text", fill_text);
  add("source", fill_source);
  add("json", fill_json);
  add("binary", fill_binary);
  add("random", fill_random);
  // Physics columnar profiles: numeric arrays whose values follow the
  // selected statistical distribution(s). Parameters are kept modest so
  // generation is cheap; the value spread is what matters, and (realistically
  // for counts in a wide type) the high-order bytes stay near-constant while
  // the low-order bytes carry the noise. The gaussian/poisson families are
  // parameterized to be comparable to the binomial ones (see the fill_*
  // comments above).
  const bool bin = dist == BenchDist::Binomial || dist == BenchDist::Classic16 || dist == BenchDist::All;
  const bool gau = dist == BenchDist::Gaussian || dist == BenchDist::Classic16 || dist == BenchDist::All;
  const bool poi = dist == BenchDist::Poisson || dist == BenchDist::Classic16 || dist == BenchDist::All;
  const bool ext = dist == BenchDist::All;
  if (bin) {
    add("bin-i16", [](std::vector<std::byte>& d, std::mt19937_64& r) {
      fill_binomial_int<std::int16_t>(d, r, 4000, 0.30);
    });
    add("bin-i32", [](std::vector<std::byte>& d, std::mt19937_64& r) {
      fill_binomial_int<std::int32_t>(d, r, 50000, 0.50);
    });
    add("bin-i64", [](std::vector<std::byte>& d, std::mt19937_64& r) {
      fill_binomial_int<std::int64_t>(d, r, 50000, 0.50);
    });
    add("bin-f32", [](std::vector<std::byte>& d, std::mt19937_64& r) {
      fill_binomial_float<float>(d, r, 4096, 0.50, 100.0);
    });
    add("bin-f64", [](std::vector<std::byte>& d, std::mt19937_64& r) {
      fill_binomial_float<double>(d, r, 4096, 0.50, 100.0);
    });
  }
  if (gau) {
    // mean 25000, sigma ~112: the moments of binomial(50000, 0.5).
    add("gau-i16", [](std::vector<std::byte>& d, std::mt19937_64& r) {
      fill_gaussian_int<std::int16_t>(d, r, 25000.0, 111.8);
    });
    add("gau-i32", [](std::vector<std::byte>& d, std::mt19937_64& r) {
      fill_gaussian_int<std::int32_t>(d, r, 25000.0, 111.8);
    });
    add("gau-i64", [](std::vector<std::byte>& d, std::mt19937_64& r) {
      fill_gaussian_int<std::int64_t>(d, r, 25000.0, 111.8);
    });
    // mean 50, sigma ~0.78: binomial(4096, 0.5) scaled to [0, 100].
    add("gau-f32", [](std::vector<std::byte>& d, std::mt19937_64& r) {
      fill_gaussian_float<float>(d, r, 50.0, 0.78);
    });
    add("gau-f64", [](std::vector<std::byte>& d, std::mt19937_64& r) {
      fill_gaussian_float<double>(d, r, 50.0, 0.78);
    });
  }
  if (poi) {
    // lambda 25000 (sigma ~158, a bit wider than the matched binomial).
    add("poi-i16", [](std::vector<std::byte>& d, std::mt19937_64& r) {
      fill_poisson_int<std::int16_t>(d, r, 25000);
    });
    add("poi-i32", [](std::vector<std::byte>& d, std::mt19937_64& r) {
      fill_poisson_int<std::int32_t>(d, r, 25000);
    });
    add("poi-i64", [](std::vector<std::byte>& d, std::mt19937_64& r) {
      fill_poisson_int<std::int64_t>(d, r, 25000);
    });
    // lambda 2048 scaled by 100/4096: mean 50, matching the binomial floats.
    add("poi-f32", [](std::vector<std::byte>& d, std::mt19937_64& r) {
      fill_poisson_float<float>(d, r, 2048, 100.0 / 4096.0);
    });
    add("poi-f64", [](std::vector<std::byte>& d, std::mt19937_64& r) {
      fill_poisson_float<double>(d, r, 2048, 100.0 / 4096.0);
    });
  }
  if (ext) {
    add("wave-f32", fill_wave_f32);
    add("wave-f64", fill_wave_f64);
    add("ramp-i32", fill_ramp_i32);
    add("ts-u64", fill_ts_u64);
    add("dec-f64", fill_dec_f64);
    add("code-x86", fill_code_x86);
  }
  // These profiles are excluded from the averaged summary rows: the text-like
  // ones are window/parser-bound and skew the mean for buffer/columnar
  // workloads, and `random` is incompressible (ratio ~1.0), which taints the
  // ratio average. They are still shown in the per-dataset tables.
  for (Dataset& ds : sets)
    if (ds.name == "text" || ds.name == "source" || ds.name == "json" ||
        ds.name == "random")
      ds.avg = false;
  return sets;
}

// Repeatedly run `fn` until at least `min_s` seconds elapse (or the deadline is
// hit), returning the achieved throughput in MB/s.
template <class F>
double throughput(F&& fn, size_t bytes, double min_s, double deadline) {
  if (now_s() < deadline) fn();  // warmup (skipped once the budget is spent)
  size_t iters = 0;
  double t0 = now_s();
  double elapsed = 0;
  do {
    fn();
    ++iters;
    elapsed = now_s() - t0;
  } while (elapsed < min_s && now_s() < deadline);
  if (elapsed <= 0) elapsed = 1e-9;
  return (static_cast<double>(bytes) * iters) / 1e6 / elapsed;
}

// Aggregate equal-sized per-dataset throughputs as total_bytes / total_time
// (harmonic mean of MB/s). An arithmetic mean of rates is misleading when
// Codec::Auto (or filters) pick different codecs per dataset: 50% @ 3 GB/s +
// 50% @ 20 MB/s is ~40 MB/s sequentially, not ~1.5 GB/s. Used by --verdict
// and the plain-bench / --compare summary rows.
inline void add_inv_rate(double& inv, double mbs) {
  if (mbs > 0) inv += 1.0 / mbs;
}
inline double harm_rate(double inv, int n) {
  return (n > 0 && inv > 0) ? static_cast<double>(n) / inv : 0;
}

struct VerdictRateAgg {
  double inv_c = 0;
  double inv_d = 0;
  double orig = 0;
  double comp = 0;
  int n = 0;

  void add(double comp_mbs, double dec_mbs, size_t orig_n, size_t comp_n) {
    add_inv_rate(inv_c, comp_mbs);
    add_inv_rate(inv_d, dec_mbs);
    orig += static_cast<double>(orig_n);
    comp += static_cast<double>(comp_n);
    ++n;
  }

  double ratio() const { return comp > 0 ? orig / comp : 0; }
  double comp_mbs() const { return harm_rate(inv_c, n); }
  double dec_mbs() const { return harm_rate(inv_d, n); }
};

void hr(char c = '-', int n = 78) {
  for (int i = 0; i < n; ++i) std::putchar(c);
  std::putchar('\n');
}

#ifdef SPECTRA8_HAVE_ZSTD
// Map an spectra8 level onto a comparable zstd level. spectra8 levels 1-9 trade
// speed for ratio much like zstd's 1-19 band, so we line up the shared 1-9
// numbers directly; spectra8's optimal parser levels (10 near-optimal, 11 ultra)
// are matched against high zstd levels rather than its slow 20-22 ultra tier,
// keeping the run
// bounded while still pitting "best practical ratio" against "best practical
// ratio".
int zstd_level_for(int spectra8_level) {
  switch (spectra8_level) {
    case 1: return 1;
    case 3: return 3;
    case 5: return 6;
    case 6: return 9;
    case 9: return 17;
    case spectra8::kNearOptLevel: return 18;  // near-optimal
    case spectra8::kUltraLevel: return 19;
    default: return spectra8_level;
  }
}
#endif  // SPECTRA8_HAVE_ZSTD

}  // namespace

const char* filter_name(spectra8::detail::Filter f) {
  using F = spectra8::detail::Filter;
  switch (f) {
    case F::None: return "raw";
    case F::Shuffle16: return "shuf16";
    case F::Shuffle32: return "shuf32";
    case F::Shuffle64: return "shuf64";
    case F::Delta16: return "delta16";
    case F::Delta32: return "delta32";
    case F::Delta64: return "delta64";
    case F::ZigzagDelta16: return "zz16";
    case F::ZigzagDelta32: return "zz32";
    case F::ZigzagDelta64: return "zz64";
    case F::DeltaShuffle32: return "d32+sh";
    case F::ZigzagDeltaShuffle32: return "zz32+sh";
    case F::BitShuffle32: return "bitsh32";
    case F::Dict16: return "dict16";
    case F::Dict32: return "dict32";
    case F::Dict64: return "dict64";
    case F::For16: return "for16";
    case F::For32: return "for32";
    case F::For64: return "for64";
    case F::Alp32: return "alp32";
    case F::Alp64: return "alp64";
    case F::DeltaShuffle64: return "dshuf64";
    case F::ZigzagShuffle32: return "zzshuf32";
    case F::ZigzagShuffle64: return "zzshuf64";
    case F::BitShuffle64: return "bitsh64";
    case F::DeltaBitShuffle32: return "d32+bit";
    case F::XorShuffle32: return "xorshuf32";
    case F::XorShuffle64: return "xorshuf64";
    case F::Delta2Shuffle32: return "d2shuf32";
    case F::ZigzagDelta2Shuffle32: return "zzd2shuf32";
    case F::Delta2Shuffle64: return "d2shuf64";
    case F::BcjX86: return "bcj_x86";
    case F::DigitNorm: return "digit_norm";
    case F::WordNorm: return "word_norm";
    case F::Base64: return "base64";
    case F::BitRound32: return "bitround32";
    case F::BitRound64: return "bitround64";
    case F::Lag2Shuffle32: return "lag2shuf32";
    case F::Lag2Shuffle64: return "lag2shuf64";
    case F::XorLag2Shuffle32: return "xorlag2shuf32";
    case F::BcjArm64: return "bcj_arm64";
    default: return "?";
  }
}

// Render a byte count as a compact human-readable size for the report header.
std::string human_size(std::size_t bytes) {
  char buf[32];
  if (bytes >= (size_t{1} << 20)) {
    if (bytes % (size_t{1} << 20) == 0)
      std::snprintf(buf, sizeof(buf), "%zu MiB", bytes >> 20);
    else
      std::snprintf(buf, sizeof(buf), "%.1f MiB", static_cast<double>(bytes) / (1024.0 * 1024.0));
  } else if (bytes >= 1024) {
    if (bytes % 1024 == 0)
      std::snprintf(buf, sizeof(buf), "%zu KiB", bytes >> 10);
    else
      std::snprintf(buf, sizeof(buf), "%.1f KiB", static_cast<double>(bytes) / 1024.0);
  } else {
    std::snprintf(buf, sizeof(buf), "%zu B", bytes);
  }
  return std::string(buf);
}

// Sweep parallel BWT inverse chain counts; recommend env SPECTRA8_BWT_ENCODE_CHAINS.
int run_bwt_chain_sweep(std::size_t bytes, double budget_s) {
  using spectra8::detail::u8;
  using spectra8::detail::u32;
  using spectra8::detail::usize;

  constexpr usize kParallelMin = 256 * 1024;
  if (bytes < kParallelMin) bytes = 1024 * 1024;
  if (budget_s <= 0) budget_s = 8.0;

  std::vector<std::byte> raw(bytes);
  std::mt19937_64 rng(991);
  fill_text(raw, rng);

  const usize n = bytes;
  std::vector<u8> src(n), bwt(n), dst(n), chk(n);
  std::memcpy(src.data(), raw.data(), n);

  const usize primary =
      spectra8::detail::bwt_forward(src.data(), n, bwt.data(), nullptr, 1);
  const usize m = n + 1;
  thread_local std::vector<u32> nav;
  nav.resize(m);
  spectra8::detail::bwt_decode::build_nav(bwt.data(), n, primary, nav.data());

  static const int kSweep[] = {1, 4, 8, 12, 16};
  const int ncfg = static_cast<int>(sizeof(kSweep) / sizeof(kSweep[0]));
  const double deadline = now_s() + budget_s;
  const double min_each =
      std::max(0.15, budget_s / static_cast<double>(ncfg * 4));

  double best_walk = 0.0;
  int best_chains = 1;

  hr('=');
  std::printf(" BWT parallel-chain sweep  (v%s)\n",
              std::string(spectra8::version()).c_str());
  std::printf("   block %s, text corpus, nav built once, anchors per config\n",
              human_size(bytes).c_str());
  hr('=');
  std::printf(" %-6s %12s %12s %10s\n", "chains", "walk MB/s", "full inv MB/s",
              "verify");
  hr('-');

  for (int ci = 0; ci < ncfg; ++ci) {
    const int chains = kSweep[ci];
    usize anchors[spectra8::detail::kBwtMaxChains] = {0};
    if (chains > 1)
      (void)spectra8::detail::bwt_forward(src.data(), n, bwt.data(), anchors,
                                          chains);

    auto walk_once = [&]() {
      if (chains == 16)
        spectra8::detail::bwt_decode::walk16(nav.data(), n, anchors, dst.data());
      else if (chains == 8)
        spectra8::detail::bwt_decode::walk8(nav.data(), n, anchors, dst.data());
      else if (chains == 4)
        spectra8::detail::bwt_decode::walk4(nav.data(), n, anchors, dst.data());
      else if (chains > 1)
        spectra8::detail::bwt_decode::walk_n(nav.data(), n, anchors, chains,
                                              dst.data());
      else
        spectra8::detail::bwt_decode::walk1(nav.data(), primary, n, dst.data());
    };

    const bool ok =
        [&] {
          walk_once();
          return std::memcmp(src.data(), dst.data(), n) == 0;
        }();
    if (!ok) {
      std::fprintf(stderr, "bwt-chains: verify failed for %d chains\n", chains);
      return 1;
    }

    const double walk_mbs = throughput(walk_once, bytes, min_each, deadline);
    const double full_mbs = throughput(
        [&] {
          (void)spectra8::detail::bwt_inverse(bwt.data(), n, primary, chk.data(),
                                             chains > 1 ? anchors : nullptr,
                                             chains);
        },
        bytes, min_each, deadline);

    std::printf(" %-6d %12.1f %12.1f %10s\n", chains, walk_mbs, full_mbs,
                "ok");
    if (chains > 1 && walk_mbs > best_walk) {
      best_walk = walk_mbs;
      best_chains = chains;
    }
  }

  hr('-');
  if (best_chains > 1) {
    std::printf(" recommended for this host: %d chains (walk %.1f MB/s)\n",
                best_chains, best_walk);
    std::printf("   export SPECTRA8_BWT_ENCODE_CHAINS=%d\n", best_chains);
    std::printf("   (or set kBwtEncodeChains in src/prism/bwt.hpp to recompile)\n");
    std::printf(" current encoder default: %d (env override via bwt_encode_chains)\n",
                spectra8::detail::bwt_encode_chains());
  } else {
    std::printf(" parallel chains did not beat single-chain walk on this run\n");
  }
  hr('=');
  return 0;
}

// Sweep BWT block size to isolate the inverse-BWT nav-table cache effect: same
// total corpus bytes, split into blocks of 2^bl, each inverted exactly as the
// decoder does (per-block nav build + LF walk at the encoder's chain count).
// Smaller blocks shrink the random-access nav footprint (and engage the compact
// u16+u8 LF table at <= 64 KiB), at a BWT-context (ratio) cost not measured here.
int run_bwt_blocksize_sweep(std::size_t bytes, double budget_s) {
  using spectra8::detail::u8;
  using spectra8::detail::u32;
  using spectra8::detail::usize;

  if (bytes < (usize{4} << 20)) bytes = usize{4} << 20;  // need several blocks
  if (budget_s <= 0) budget_s = 8.0;

  std::vector<std::byte> raw(bytes);
  std::mt19937_64 rng(991);
  fill_text(raw, rng);
  const usize n = bytes;
  std::vector<u8> src(n), dst(n);
  std::memcpy(src.data(), raw.data(), n);

  constexpr usize kBwtParallelMin = 256 * 1024;
  const int enc_chains = spectra8::detail::bwt_encode_chains();

  static const int kBl[] = {16, 17, 18, 19, 20, 21, 22};
  const int ncfg = static_cast<int>(sizeof(kBl) / sizeof(kBl[0]));
  const double deadline = now_s() + budget_s;
  const double min_each = std::max(0.15, budget_s / static_cast<double>(ncfg));

  hr('=');
  std::printf(" BWT block-size sweep  (v%s)\n",
              std::string(spectra8::version()).c_str());
  std::printf("   corpus %s, text; per-block nav rebuilt + inverted (as decode)\n",
              human_size(bytes).c_str());
  std::printf("   default encode chains=%d; <64 KiB blocks use compact u16+u8 nav\n",
              enc_chains);
  hr('=');
  std::printf(" %-10s %7s %7s %9s %12s %8s\n", "block", "chains", "nav B/row",
              "navKB", "inv MB/s", "verify");
  hr('-');

  // Per-block precomputed BWT state (built once, outside timing).
  struct Blk {
    usize off, len, primary;
    int chains;
    usize anchors[spectra8::detail::kBwtMaxChains];
  };

  for (int ci = 0; ci < ncfg; ++ci) {
    const usize blk = usize{1} << kBl[ci];
    std::vector<u8> bwt(n);
    std::vector<Blk> blks;
    blks.reserve(n / blk + 1);

    for (usize off = 0; off < n; off += blk) {
      const usize len = std::min(blk, n - off);
      Blk b{};
      b.off = off;
      b.len = len;
      const bool parallel = (len >= kBwtParallelMin) &&
                            (len + 1 <= spectra8::detail::kBwtMaxBlock);
      b.chains = parallel ? enc_chains : 1;
      for (int c = 0; c < spectra8::detail::kBwtMaxChains; ++c) b.anchors[c] = 0;
      b.primary = spectra8::detail::bwt_forward(
          src.data() + off, len, bwt.data() + off, parallel ? b.anchors : nullptr,
          b.chains);
      blks.push_back(b);
    }

    auto invert_all = [&]() {
      for (const Blk& b : blks)
        (void)spectra8::detail::bwt_inverse(
            bwt.data() + b.off, b.len, b.primary, dst.data() + b.off,
            b.chains > 1 ? b.anchors : nullptr, b.chains);
    };

    invert_all();
    const bool ok = std::memcmp(src.data(), dst.data(), n) == 0;
    if (!ok) {
      std::fprintf(stderr, "bwt-blocksize: verify failed for block 2^%d\n",
                   kBl[ci]);
      return 1;
    }

    const usize m_typ = std::min(blk, n) + 1;
    const int nav_brow = (m_typ <= (usize{1} << 16)) ? 3 : 4;
    const double nav_kb = static_cast<double>(m_typ) * nav_brow / 1024.0;
    const double inv_mbs = throughput(invert_all, bytes, min_each, deadline);

    std::printf(" %-10s %7d %7d %9.0f %12.1f %8s\n",
                human_size(blk).c_str(), blks.front().chains, nav_brow, nav_kb,
                inv_mbs, "ok");
  }
  hr('-');
  std::printf(" note: this measures decode speed only; smaller blocks cost BWT\n");
  std::printf("       ratio (less context). Use bench --verdict for the ratio side.\n");
  hr('=');
  return 0;
}

namespace {

// External filter (+ optional byte-RLE) staging for `bench --verdict` rows.
struct VerdictExtStage {
  spectra8::detail::Filter filter = spectra8::detail::Filter::None;
  int dict_w = 0;         // nonzero: staged form is the dictionary stream
  bool has_ext = false;   // some external candidate exists (filter or dict)
  bool use_ext = false;   // filtered and/or RLE path beats raw
  bool use_rle = false;
  std::size_t codec_len = 0;  // bytes fed to the downstream codec
  const spectra8::detail::u8* codec_src = nullptr;
};

void verdict_ext_prepare(const spectra8::detail::u8* in, std::size_t n,
                         bool no_filter, bool no_rle, bool no_dict,
                         bool full_sample,
                         std::vector<spectra8::detail::u8>& filt,
                         std::vector<spectra8::detail::u8>& rle,
                         VerdictExtStage& st) {
  st = VerdictExtStage{};
  st.codec_src = in;
  st.codec_len = n;
  if (no_filter) return;

  // A dictionary remap subsumes the byte transforms whenever it builds.
  // for/alp compete with the byte filter on adjacency cost. A raw-input
  // fallback compress still guards the final pick. `--no-dict` skips only
  // the dictionary; frame-of-reference and decimal integerization stay.
  const auto framed = spectra8::detail::detect_framed_filter(
      in, n, full_sample, ~spectra8::detail::u64{0}, !no_dict);
  if (spectra8::detail::filter_is_dict(framed)) {
    filt.clear();
    if (spectra8::detail::framed_filter_fwd(framed, in, n, filt)) {
      st.has_ext = true;
      st.filter = framed;
      st.dict_w = spectra8::detail::dict_filter_width(framed);
      st.codec_src = filt.data();
      st.codec_len = filt.size();
      return;  // dict streams are dense; byte-RLE never stacks on them
    }
  }

  st.filter = spectra8::detail::detect_filter(in, n, full_sample);
  std::vector<spectra8::detail::u8> pack;
  const bool pack_ok =
      (spectra8::detail::filter_is_for(framed) ||
       spectra8::detail::filter_is_alp(framed) ||
       spectra8::detail::filter_is_digit_norm(framed) ||
       spectra8::detail::filter_is_word_norm(framed) ||
       spectra8::detail::filter_is_base64(framed)) &&
      spectra8::detail::framed_filter_fwd(framed, in, n, pack);
  if (st.filter == spectra8::detail::Filter::None && !pack_ok) return;
  if (pack_ok) {
    bool use_pack = st.filter == spectra8::detail::Filter::None;
    if (!use_pack) {
      filt.resize(n);
      spectra8::detail::apply_filter(st.filter, in, n, filt.data());
      use_pack = spectra8::detail::framed_beats_byte(filt.data(), n, st.filter, pack);
    }
    if (use_pack) {
      filt.swap(pack);
      st.has_ext = true;
      st.filter = framed;
      st.dict_w = 0;
      st.codec_src = filt.data();
      st.codec_len = filt.size();
      return;
    }
  }
  if (st.filter == spectra8::detail::Filter::None) return;
  st.has_ext = true;

  filt.resize(n);
  spectra8::detail::apply_filter(st.filter, in, n, filt.data());
  st.codec_src = filt.data();
  st.codec_len = n;

  // RLE is only a *candidate* here: a smaller intermediate does not mean a
  // smaller compressed frame (LZ/zstd often match the same runs themselves).
  // Callers must compete filter vs filter+RLE on compressed size via
  // verdict_ext_prefer_smaller() before comparing to raw.
  if (!no_rle) {
    spectra8::detail::byte_rle_encode(filt.data(), n, rle);
    if (rle.size() < n) {
      st.use_rle = true;
      st.codec_src = rle.data();
      st.codec_len = rle.size();
    }
  }
}

// When staging selected filter+RLE (`st.use_rle`), also consider the
// filter-only compressed size. Keep whichever is smaller and retarget `st`.
// `filt` must still hold the non-RLE filtered block of length `n`.
void verdict_ext_prefer_smaller(VerdictExtStage& st, std::size_t n,
                                const std::vector<spectra8::detail::u8>& filt,
                                std::size_t cs_filter_rle,
                                std::size_t cs_filter_only,
                                std::size_t& cs_ext) {
  if (!st.use_rle) {
    cs_ext = cs_filter_rle;
    return;
  }
  if (cs_filter_only > 0 && cs_filter_only <= cs_filter_rle) {
    st.use_rle = false;
    st.codec_src = filt.data();
    st.codec_len = n;
    cs_ext = cs_filter_only;
  } else {
    cs_ext = cs_filter_rle;
  }
}

// Pick external filter path vs raw using size comparison, or trust detection
// when `no_filter_fallback` (skip the raw compress trial).
bool verdict_ext_use(const VerdictExtStage& st, bool no_filter,
                     bool no_filter_fallback, std::size_t cs_raw,
                     std::size_t cs_ext) {
  if (no_filter || !st.has_ext) return false;
  if (no_filter_fallback) return cs_ext > 0;
  return cs_ext < cs_raw;
}

// Recovery work only (byte-RLE expansion + filter inversion, no
// verification): this is legitimate decode-side work, so the timed decode
// lambdas call this variant to avoid timing a full-buffer memcmp.
bool verdict_ext_invert(const VerdictExtStage& st, std::size_t n,
                        const spectra8::detail::u8* codec_out,
                        std::vector<spectra8::detail::u8>& filt_scratch,
                        std::vector<spectra8::detail::u8>& orig_out) {
  if (spectra8::detail::filter_is_framed(st.filter))
    return spectra8::detail::framed_filter_inv(st.filter, codec_out, st.codec_len,
                                                orig_out.data(), n) ==
           spectra8::Status::Ok;
  if (st.dict_w != 0)
    return spectra8::detail::dict_filter_inv(codec_out, st.codec_len,
                                              orig_out.data(), n,
                                              st.dict_w) == spectra8::Status::Ok;
  const spectra8::detail::u8* filtered = codec_out;
  if (st.use_rle) {
    filt_scratch.resize(n);
    if (spectra8::detail::byte_rle_decode(codec_out, st.codec_len,
                                           filt_scratch.data(), n) !=
        spectra8::Status::Ok)
      return false;
    filtered = filt_scratch.data();
  }
  spectra8::detail::invert_filter(st.filter, filtered, n, orig_out.data());
  return true;
}

// Recovery + byte-exact verification, used by the one-shot correctness checks
// before timing.
bool verdict_ext_recover(const VerdictExtStage& st, std::size_t n,
                         const spectra8::detail::u8* codec_out,
                         std::vector<spectra8::detail::u8>& filt_scratch,
                         std::vector<spectra8::detail::u8>& orig_out,
                         const spectra8::detail::u8* expected) {
  if (!st.use_ext)
    return std::memcmp(codec_out, expected, n) == 0;
  if (!verdict_ext_invert(st, n, codec_out, filt_scratch, orig_out))
    return false;
  return std::memcmp(orig_out.data(), expected, n) == 0;
}

// Rebuild a stage that was already chosen (entropy detect or brute winner)
// so timed loops re-apply that filter and do not search again.
void verdict_ext_reapply(const spectra8::detail::u8* in, std::size_t n,
                         const VerdictExtStage& chosen,
                         std::vector<spectra8::detail::u8>& filt,
                         std::vector<spectra8::detail::u8>& rle,
                         VerdictExtStage& st) {
  st = VerdictExtStage{};
  st.filter = chosen.filter;
  st.dict_w = chosen.dict_w;
  st.has_ext = chosen.has_ext;
  st.use_ext = chosen.use_ext;
  st.use_rle = chosen.use_rle;
  if (!chosen.use_ext) {
    st.codec_src = in;
    st.codec_len = n;
    return;
  }
  if (spectra8::detail::filter_is_framed(chosen.filter)) {
    if (!spectra8::detail::framed_filter_fwd(chosen.filter, in, n, filt)) {
      st.use_ext = false;
      st.codec_src = in;
      st.codec_len = n;
      return;
    }
    st.codec_src = filt.data();
    st.codec_len = filt.size();
    return;
  }
  filt.resize(n);
  if (chosen.filter != spectra8::detail::Filter::None)
    spectra8::detail::apply_filter(chosen.filter, in, n, filt.data());
  else
    std::memcpy(filt.data(), in, n);
  if (chosen.use_rle) {
    spectra8::detail::byte_rle_encode(filt.data(), n, rle);
    st.codec_src = rle.data();
    st.codec_len = rle.size();
  } else {
    st.codec_src = filt.data();
    st.codec_len = n;
  }
}

// Compress every filter and keep the smallest frame. `compress` returns the
// compressed size, or 0 to reject a candidate. The returned size is that
// winner; raw keep-smaller is still the caller's job. Speeds measured later
// must reapply `st` rather than search again.
template <class Compress>
std::size_t verdict_ext_brute(const spectra8::detail::u8* in, std::size_t n,
                              bool no_rle, bool no_dict,
                              std::vector<spectra8::detail::u8>& filt,
                              std::vector<spectra8::detail::u8>& rle,
                              VerdictExtStage& st, Compress&& compress) {
  st = VerdictExtStage{};
  st.codec_src = in;
  st.codec_len = n;
  struct Best {
    spectra8::detail::Filter filter = spectra8::detail::Filter::None;
    bool use_rle = false;
    std::size_t cs = static_cast<std::size_t>(-1);
    std::vector<spectra8::detail::u8> plain;
    std::vector<spectra8::detail::u8> rle_bytes;
  } best;
  std::vector<spectra8::detail::u8> scratch, rle_scratch;
  const int last = static_cast<int>(spectra8::detail::Filter::kMaxValue);
  for (int id = 1; id <= last; ++id) {
    const auto f = static_cast<spectra8::detail::Filter>(id);
    if (no_dict && spectra8::detail::filter_is_dict(f)) continue;
    const spectra8::detail::u8* plain = nullptr;
    std::size_t plain_n = 0;
    if (spectra8::detail::filter_is_framed(f)) {
      scratch.clear();
      if (!spectra8::detail::framed_filter_fwd(f, in, n, scratch)) continue;
      plain = scratch.data();
      plain_n = scratch.size();
    } else {
      scratch.resize(n);
      spectra8::detail::apply_filter(f, in, n, scratch.data());
      plain = scratch.data();
      plain_n = n;
    }
    auto take = [&](const spectra8::detail::u8* src, std::size_t len, bool rle_flag) {
      const std::size_t cs = compress(src, len);
      if (cs == 0 || cs >= best.cs) return;
      best.cs = cs;
      best.filter = f;
      best.use_rle = rle_flag;
      best.plain.assign(plain, plain + plain_n);
      if (rle_flag)
        best.rle_bytes.assign(src, src + len);
      else
        best.rle_bytes.clear();
    };
    take(plain, plain_n, false);
    if (!no_rle && !spectra8::detail::filter_is_framed(f)) {
      spectra8::detail::byte_rle_encode(scratch.data(), n, rle_scratch);
      if (rle_scratch.size() < n) take(rle_scratch.data(), rle_scratch.size(), true);
    }
  }
  if (best.cs == static_cast<std::size_t>(-1)) return 0;
  filt.swap(best.plain);
  rle.swap(best.rle_bytes);
  st.filter = best.filter;
  st.use_rle = best.use_rle;
  st.has_ext = true;
  st.dict_w = spectra8::detail::filter_is_dict(best.filter)
                  ? spectra8::detail::dict_filter_width(best.filter)
                  : 0;
  if (best.use_rle) {
    st.codec_src = rle.data();
    st.codec_len = rle.size();
  } else {
    st.codec_src = filt.data();
    st.codec_len = filt.size();
  }
  return best.cs;
}

// Paired rows: " %-24s  %17s  %17s  %17s" = 82 columns. The three numeric
// fields share one width so "brute / det" and the values share a right edge.
constexpr int kVerdictPairWidth = 1 + 24 + 2 + 17 + 2 + 17 + 2 + 17;

void print_verdict_header(bool paired) {
  if (!paired) {
    std::printf(" %-24s %9s %12s %12s\n", "config", "ratio", "comp MB/s",
                "dec MB/s");
    std::fflush(stdout);
    return;
  }
  std::printf(" %-24s  %17s  %17s  %17s\n", "config", "ratio", "comp MB/s",
              "dec MB/s");
  std::printf(" %-24s  %17s  %17s  %17s\n", "", "brute / det", "brute / det",
              "brute / det");
  std::fflush(stdout);
}

void print_verdict_pair(const char* label, const VerdictRateAgg& brute,
                        const VerdictRateAgg& det) {
  char ratio[32], comp[40], decs[40];
  std::snprintf(ratio, sizeof(ratio), "%.3f / %.3f", brute.ratio(), det.ratio());
  std::snprintf(comp, sizeof(comp), "%.1f / %.1f", brute.comp_mbs(),
                det.comp_mbs());
  std::snprintf(decs, sizeof(decs), "%.1f / %.1f", brute.dec_mbs(), det.dec_mbs());
  std::printf(" %-24s  %17s  %17s  %17s\n", label, ratio, comp, decs);
  std::fflush(stdout);
}

void print_verdict_na(const char* label, bool paired) {
  if (!paired) {
    std::printf(" %-24s %9s %12s %12s\n", label, "n/a", "n/a", "n/a");
  } else {
    std::printf(" %-24s  %17s  %17s  %17s\n", label, "n/a", "n/a", "n/a");
  }
  std::fflush(stdout);
}

}  // namespace

// Compact head-to-head for `bench --verdict`: over the summary datasets only
// (the binary/columnar profiles; 6 for a single `--dist`, 16 for `--dist all`).
// With `brute` (and filters on), each numeric column is the brute-force winner
// over the entropy-detected filter. Brute applies every filter, compresses it,
// and keeps the smallest (still keep-smaller versus raw). Speeds time that
// winner; the search is outside the timer. Without `brute`, each column is the
// entropy-detected filter only. Built-in spectra8 per-block filtering is disabled
// so compress speed is comparable to zstd/BWT/lzma. With `no_filter`, raw
// input is compressed as-is and the columns are single values.
// Byte-RLE on the filtered bytes is off by default; pass `no_rle=false` via
// `--rle` to also try filter+RLE and keep the better of filter / filter+RLE /
// raw by compressed size. With `no_filter_fallback`, only the detected filter
// path is compressed (no raw trial).
// With `full`, every level of every implementation is scanned (spectra8 Beam, BWT,
// Mosaic/Oracle/Crystal L1..L11, zstd L1..max, lzma P0..P9+P9e) instead of the compact
// selection;
// bzip3 has no levels, so its two block-size rows are unchanged.
// With `no_dict`, the dictionary-remap filters (dict32/64) are excluded from
// the external staging (byte transforms only), for isolating their effect.
// With `full_sample`, detect scores the whole block (else 16 KiB mid-sample).
// With `auto_max_level`, Auto rows race each codec at max effort (else L5).
int run_verdict(double budget, std::size_t bytes, bool no_filter, bool no_rle,
                bool no_filter_fallback, bool full, bool no_dict,
                bool full_sample, bool auto_max_level, bool brute, BenchDist dist,
                std::uint64_t seed, bool complete = false,
                spectra8::CodecMask only_rows = ~spectra8::CodecMask{0},
                bool stats = false) {
  if (bytes == 0) {
    if (complete && dist == BenchDist::Text) {
      bytes = 0;  // 0 indicates complete file to make_datasets
    } else if (dist == BenchDist::Text) {
      bytes = size_t{8} << 20;  // default 8 MiB for text
    } else if (dist == BenchDist::Binaries) {
      bytes = size_t{1} << 20;  // 1 MiB slice of a shared library
    } else {
      bytes = size_t{128} << 10;
    }
  }

  auto datasets = make_datasets(bytes, dist, seed);
  if (bytes == 0 && !datasets.empty()) {
    bytes = datasets[0].data.size();
  }
  int navg = 0;
  for (const Dataset& ds : datasets)
    if (ds.avg) ++navg;
  if (navg == 0) navg = 1;

  const size_t max_ext_len = bytes * 2 + 65536;
  std::vector<std::byte> comp(spectra8::compress_bound(max_ext_len));
  std::vector<std::byte> dec(bytes);
  // Scratch for the externally-applied byte filter used by the zstd and BWT
  // configs (which have no built-in filter stage): `filt` holds the transformed
  // input, `unfilt` the inverse-transformed output for the roundtrip check.
  std::vector<spectra8::detail::u8> filt(max_ext_len);
  std::vector<spectra8::detail::u8> rle;
  std::vector<spectra8::detail::u8> unfilt(bytes);
  std::vector<spectra8::detail::u8> codec_dec(max_ext_len);

  // Full scan: spectra8 Beam, BWT, Mosaic, Oracle and Crystal walk every level; the external
  // codecs walk their own full level/preset range.
  const int spectra8_configs =
      full ? (5 * spectra8::kMaxLevel + (dist == BenchDist::Text ? 1 : 0) + 1)
           : (16 + (dist == BenchDist::Text ? 1 : 0) + 1);
#ifdef SPECTRA8_HAVE_ZSTD
  const int zstd_configs = full ? ZSTD_maxCLevel() : 3;  // else: max + L5 + L1
#else
  const int zstd_configs = 0;
#endif
#ifdef SPECTRA8_HAVE_LZMA
  const int lzma_configs = full ? 11 : 2;  // P0..P9 + P9e, else P9e + P6
#else
  const int lzma_configs = 0;
#endif
#ifdef SPECTRA8_HAVE_BZIP3
  const int bzip3_configs = 2;  // intermediate + highest block size
#else
  const int bzip3_configs = 0;
#endif
  const int auto_configs = 6;  // Codec::Auto g5%..g0% (L5 or max effort)
  // Two timed loops (compress + decompress) per dataset per config.
  // `--brute` measures each row twice: brute-force winner and the
  // entropy-detected filter. The search itself is outside those timers.
  if (brute && no_filter) brute = false;
  const bool paired = brute && !no_filter;
  const int measure_passes = paired ? 2 : 1;
  const int total_loops =
      (spectra8_configs + zstd_configs + lzma_configs + bzip3_configs +
       auto_configs) *
      navg * 2 * measure_passes;
  double min_each = budget / (total_loops > 0 ? total_loops : 1);
  if (min_each < 0.05) min_each = 0.05;
  if (min_each > 1.5) min_each = 1.5;
  const double deadline = now_s() + budget;

  hr('=');
  std::printf(" spectra8 verdict  (v%s)\n",
              std::string(spectra8::version()).c_str());
  const char* dist_desc =
      dist == BenchDist::Binaries
          ? "shared-library slices (arm64/x86 text and rodata if none qualify)"
          : dist == BenchDist::RawInt
                ? "peaked integer columns (lap/tail/zipf/cat i16/i32, ids i64)"
                : dist == BenchDist::All
          ? "binary + {bin,gau,poi}-{i16,i32,i64,f32,f64} + {wave,ramp,ts,dec,code}"
          : dist == BenchDist::Classic16
                ? "binary + {bin,gau,poi}-{i16,i32,i64,f32,f64} (classic 16)"
                : dist == BenchDist::Gaussian
                      ? "binary + gau-i16/i32/i64/f32/f64"
                      : dist == BenchDist::Poisson
                            ? "binary + poi-i16/i32/i64/f32/f64"
                            : dist == BenchDist::Text
                                  ? (complete ? "enwik8 (complete Wikipedia text dump)"
                                              : "enwik8 (Wikipedia text dump)")
                                  : "binary + bin-i16/i32/i64/f32/f64";
  std::printf("   avg over %d dataset%s (%s), "
              "%s each, single thread%s%s%s; seed 0x%llx\n",
              navg, navg == 1 ? "" : "s", dist_desc, human_size(bytes).c_str(),
              full ? "; full level scan" : "",
              auto_max_level ? "; Auto at max effort" : "",
              (!no_filter && full_sample) ? "; filter detect=full block"
                                          : "",
              static_cast<unsigned long long>(seed));
  hr('=');

  if (!no_filter) {
    // Show which transform detect_filter picks for each averaged dataset (all
    // configs apply it externally for a fair compress-speed comparison).
    // With --dist all this is 16 entries, so wrap at the table width.
    std::printf(" entropy-detected filter per dataset:\n ");
    int col = 1;
    for (const Dataset& ds : datasets) {
      if (!ds.avg) continue;
      const auto* u8d =
          reinterpret_cast<const spectra8::detail::u8*>(ds.data.data());
      // Mirror verdict_ext_prepare: dictionary first, then for/alp when they
      // beat the byte filter, otherwise the byte transform.
      VerdictExtStage shown;
      std::vector<spectra8::detail::u8> tmp, tmp_rle;
      verdict_ext_prepare(u8d, bytes, false, true, no_dict, full_sample, tmp,
                          tmp_rle, shown);
      const char* fname =
          shown.has_ext ? filter_name(shown.filter) : "raw";
      char item[48];
      int len = std::snprintf(item, sizeof(item), " %s=%s", ds.name.c_str(),
                              fname);
      if (col + len > (paired ? kVerdictPairWidth : 78)) {
        std::printf("\n ");
        col = 1;
      }
      std::printf("%s", item);
      col += len;
    }
    std::printf("\n");
  }
  const int rule = paired ? kVerdictPairWidth : 78;
  hr('-', rule);
  print_verdict_header(paired);
  hr('-', rule);

  // One spectra8 config (single thread) averaged over the summary datasets.
  auto run_spectra8 = [&](const char* label,
                      spectra8::CompressParams base) -> bool {
    base.threads = 1;
    VerdictRateAgg agg;
    for (const Dataset& ds : datasets) {
      if (!ds.avg) continue;
      auto cr = spectra8::compress(ds.data, comp, base);
      if (!cr) {
        std::fprintf(stderr, "verdict: %s compress failed (%s)\n", label,
                     ds.name.c_str());
        return false;
      }
      const size_t cs = cr.value;
      std::span<const std::byte> cspan(comp.data(), cs);
      spectra8::DecompressParams d1{1};
      auto vr = spectra8::decompress(cspan, dec, d1);
      if (!vr || vr.value != bytes ||
          std::memcmp(dec.data(), ds.data.data(), bytes) != 0) {
        std::fprintf(stderr, "verdict: %s verify failed (%s)\n", label,
                     ds.name.c_str());
        return false;
      }
      double tc = throughput(
          [&] { (void)spectra8::compress(ds.data, comp, base); }, bytes,
          min_each, deadline);
      double td = throughput(
          [&] { (void)spectra8::decompress(cspan, dec, d1); }, bytes, min_each,
          deadline);
      agg.add(tc, td, bytes, cs);
    }
    std::printf(" %-24s %9.3f %12.1f %12.1f\n", label, agg.ratio(),
                agg.comp_mbs(), agg.dec_mbs());
    std::fflush(stdout);
    return true;
  };

  // Same as run_spectra8 but with filter (+ optional byte-RLE) applied externally.
  // `brute` compresses every filter and keeps the smallest; otherwise the
  // entropy detector nominates one filter.
  auto measure_spectra8_ext = [&](const char* label, spectra8::CompressParams base,
                              bool brute, VerdictRateAgg& agg) -> bool {
    base.threads = 1;
    spectra8::DecompressParams d1{1};
    agg = VerdictRateAgg{};
    for (const Dataset& ds : datasets) {
      if (!ds.avg) continue;
      const auto* in =
          reinterpret_cast<const spectra8::detail::u8*>(ds.data.data());
      VerdictExtStage st;
      size_t cs_raw = 0;
      size_t cs_ext = 0;
      if (brute) {
        cs_ext = verdict_ext_brute(
            in, bytes, no_rle, no_dict, filt, rle, st,
            [&](const spectra8::detail::u8* p, std::size_t len) -> std::size_t {
              std::span<const std::byte> work(
                  reinterpret_cast<const std::byte*>(p), len);
              auto fr = spectra8::compress(work, comp, base);
              return fr ? fr.value : 0;
            });
      } else {
        verdict_ext_prepare(in, bytes, false, no_rle, no_dict, full_sample, filt,
                            rle, st);
      }
      if (!brute && st.has_ext) {
        std::span<const std::byte> work_span(
            reinterpret_cast<const std::byte*>(st.codec_src), st.codec_len);
        auto fr = spectra8::compress(work_span, comp, base);
        if (!fr) {
          std::fprintf(stderr, "verdict: %s filter compress failed (%s)\n",
                       label, ds.name.c_str());
          return false;
        }
        cs_ext = fr.value;
        if (st.use_rle) {
          std::span<const std::byte> filt_span(
              reinterpret_cast<const std::byte*>(filt.data()), bytes);
          auto ff = spectra8::compress(filt_span, comp, base);
          if (!ff) {
            std::fprintf(stderr, "verdict: %s filter-only compress failed (%s)\n",
                         label, ds.name.c_str());
            return false;
          }
          verdict_ext_prefer_smaller(st, bytes, filt, cs_ext, ff.value, cs_ext);
        }
      }
      if (!no_filter_fallback || !st.has_ext) {
        auto rr = spectra8::compress(ds.data, comp, base);
        if (!rr) {
          std::fprintf(stderr, "verdict: %s compress failed (%s)\n", label,
                       ds.name.c_str());
          return false;
        }
        cs_raw = rr.value;
      }
      st.use_ext =
          verdict_ext_use(st, false, no_filter_fallback, cs_raw, cs_ext);
      const size_t cs = st.use_ext ? cs_ext : cs_raw;
      if (st.use_ext) {
        std::span<const std::byte> work_span(
            reinterpret_cast<const std::byte*>(st.codec_src), st.codec_len);
        (void)spectra8::compress(work_span, comp, base);
      } else {
        (void)spectra8::compress(ds.data, comp, base);
      }
      std::span<const std::byte> cspan(comp.data(), cs);

      bool ok = false;
      if (st.use_ext) {
        std::span<std::byte> work_out(
            reinterpret_cast<std::byte*>(codec_dec.data()), st.codec_len);
        auto vr = spectra8::decompress(cspan, work_out, d1);
        ok = vr && vr.value == st.codec_len &&
             verdict_ext_recover(st, bytes, codec_dec.data(), filt, unfilt, in);
      } else {
        auto vr = spectra8::decompress(cspan, dec, d1);
        ok = vr && vr.value == bytes &&
             std::memcmp(dec.data(), ds.data.data(), bytes) == 0;
      }
      if (!ok) {
        std::fprintf(stderr, "verdict: %s verify failed (%s)\n", label,
                     ds.name.c_str());
        return false;
      }

      const VerdictExtStage chosen = st;
      double tc =
          st.use_ext
              ? throughput(
                    [&] {
                      VerdictExtStage tmp;
                      verdict_ext_reapply(in, bytes, chosen, filt, rle, tmp);
                      std::span<const std::byte> work_span(
                          reinterpret_cast<const std::byte*>(tmp.codec_src),
                          tmp.codec_len);
                      (void)spectra8::compress(work_span, comp, base);
                    },
                    bytes, min_each, deadline)
              : throughput(
                    [&] { (void)spectra8::compress(ds.data, comp, base); },
                    bytes, min_each, deadline);
      double td =
          st.use_ext
              ? throughput(
                    [&] {
                      std::span<std::byte> work_out(
                          reinterpret_cast<std::byte*>(codec_dec.data()),
                          st.codec_len);
                      (void)spectra8::decompress(cspan, work_out, d1);
                      (void)verdict_ext_invert(st, bytes, codec_dec.data(),
                                               filt, unfilt);
                    },
                    bytes, min_each, deadline)
              : throughput(
                    [&] { (void)spectra8::decompress(cspan, dec, d1); }, bytes,
                    min_each, deadline);
      agg.add(tc, td, bytes, cs);
      if (stats) {
        std::printf("  %-10s %-16s: %6zu -> %6zu (%5.3fx)  comp %6.1f MB/s  dec %6.1f MB/s\n",
                    ds.name.c_str(), label, bytes, cs, static_cast<double>(bytes) / cs, tc, td);
        if (base.codec == spectra8::Codec::Flash) {
          auto fs = spectra8::detail::get_last_flash_stats();
          std::printf("    [Flash breakdown: lits=%zu (huff_est=%zu, win=%+zd), tokens=%zu, offsets=%zu (short=%zu, wide=%zu, rep0=%zu, rep1=%zu), ext=%zu, matches=%zu]\n",
                      fs.lit_bytes, fs.huff_lit_bytes, static_cast<ssize_t>(fs.lit_bytes) - static_cast<ssize_t>(fs.huff_lit_bytes),
                      fs.token_bytes, fs.offset_bytes, fs.num_short_offsets, fs.num_wide_offsets,
                      fs.num_rep0, fs.num_rep1, fs.ext_bytes, fs.num_matches);
        }
      }
    }
    return true;
  };

  const char* flt_sfx =
      no_filter ? "" : (no_rle ? " +flt" : " +flt+rle");
  auto run_spectra8_row = [&](const char* stem,
                          spectra8::CompressParams base) -> bool {
    char label[32];
    std::snprintf(label, sizeof(label), "%s%s", stem, flt_sfx);
    if (no_filter) return run_spectra8(label, base);
    VerdictRateAgg det;
    if (!measure_spectra8_ext(label, base, false, det)) return false;
    if (!paired) {
      std::printf(" %-24s %9.3f %12.1f %12.1f\n", label, det.ratio(),
                  det.comp_mbs(), det.dec_mbs());
      std::fflush(stdout);
      return true;
    }
    VerdictRateAgg brute_agg;
    if (!measure_spectra8_ext(label, base, true, brute_agg)) return false;
    print_verdict_pair(label, brute_agg, det);
    return true;
  };

  // Flash: L1 is the original interleaved-literal SIMD path (~4 GB/s
  // decode); L5 may emit the optional Huffman / compact / rep header.
  auto run_flash_level = [&](int level) -> bool {
    if (!(only_rows & spectra8::codec_bit(spectra8::Codec::Flash))) return true;
    spectra8::CompressParams p;
    p.level = level;
    p.codec = spectra8::Codec::Flash;
    p.filters = false;
    char stem[24];
    std::snprintf(stem, sizeof(stem), "spectra8 Flash L%d", level);
    return run_spectra8_row(stem, p);
  };

  for (int l : {1, 5})
    if (!run_flash_level(l)) return 1;

  // One spectra8 Beam row at a given level (filter applied externally unless
  // --no-filter; level 11 keeps its "ultra" label). Level 0 is Flash, not Beam.
  auto run_lz_level = [&](int level) -> bool {
    if (level <= 0) return true;
    spectra8::CompressParams p;
    p.level = level;
    p.codec = spectra8::Codec::Beam;
    if (!(only_rows & spectra8::codec_bit(p.codec))) return true;
    p.filters = false;
    char stem[24];
    if (level == spectra8::kUltraLevel)
      std::snprintf(stem, sizeof(stem), "spectra8 Beam ultra");
    else
      std::snprintf(stem, sizeof(stem), "spectra8 Beam L%d", level);
    return run_spectra8_row(stem, p);
  };

  if (full) {
    for (int l = 1; l <= spectra8::kMaxLevel; ++l)
      if (!run_lz_level(l)) return 1;
  } else {
    for (int l : {1, 5, 9, spectra8::kUltraLevel})
      if (!run_lz_level(l)) return 1;
  }

#ifdef SPECTRA8_HAVE_ZSTD
  if (only_rows & spectra8::codec_bit(spectra8::Codec::Zstd)) {
    const size_t zbound = ZSTD_compressBound(max_ext_len);
    std::vector<std::byte> zc(zbound);    // compress scratch
    std::vector<std::byte> zsrc(zbound);  // stable decode src
    // One zstd config at a given level, with the same external best-filter
    // (keep-smaller) logic as the spectra8 Prism config.
    auto measure_zstd = [&](int zlevel, bool brute, VerdictRateAgg& agg) -> bool {
      agg = VerdictRateAgg{};
      for (const Dataset& ds : datasets) {
        if (!ds.avg) continue;
        const auto* in =
            reinterpret_cast<const spectra8::detail::u8*>(ds.data.data());
        VerdictExtStage st;
        size_t cs_raw = 0;
        size_t cs_ext = 0;
        if (brute) {
          cs_ext = verdict_ext_brute(
              in, bytes, no_rle, no_dict, filt, rle, st,
              [&](const spectra8::detail::u8* p, std::size_t len) -> std::size_t {
                const size_t cs =
                    ZSTD_compress(zc.data(), zc.size(), p, len, zlevel);
                return ZSTD_isError(cs) ? 0 : cs;
              });
        } else {
          verdict_ext_prepare(in, bytes, no_filter, no_rle, no_dict, full_sample,
                              filt, rle, st);
        }
        if (!brute && !no_filter && st.has_ext) {
          cs_ext = ZSTD_compress(zc.data(), zc.size(), st.codec_src,
                                 st.codec_len, zlevel);
          if (ZSTD_isError(cs_ext)) {
            std::fprintf(stderr, "verdict: zstd filter compress failed (%s): %s\n",
                         ds.name.c_str(), ZSTD_getErrorName(cs_ext));
            return false;
          }
          if (st.use_rle) {
            const size_t cs_filt =
                ZSTD_compress(zc.data(), zc.size(), filt.data(), bytes, zlevel);
            if (ZSTD_isError(cs_filt)) {
              std::fprintf(stderr,
                           "verdict: zstd filter-only compress failed (%s): %s\n",
                           ds.name.c_str(), ZSTD_getErrorName(cs_filt));
              return false;
            }
            verdict_ext_prefer_smaller(st, bytes, filt, cs_ext, cs_filt, cs_ext);
          }
        }
        if (no_filter || !no_filter_fallback || !st.has_ext) {
          cs_raw = ZSTD_compress(zc.data(), zc.size(), ds.data.data(), bytes,
                                 zlevel);
          if (ZSTD_isError(cs_raw)) {
            std::fprintf(stderr, "verdict: zstd compress failed (%s): %s\n",
                         ds.name.c_str(), ZSTD_getErrorName(cs_raw));
            return false;
          }
        }
        st.use_ext = verdict_ext_use(st, no_filter, no_filter_fallback, cs_raw,
                                     cs_ext);
        const size_t zcs = st.use_ext ? cs_ext : cs_raw;
        const void* csrc =
            st.use_ext ? static_cast<const void*>(st.codec_src) : ds.data.data();
        const size_t clen = st.use_ext ? st.codec_len : bytes;
        ZSTD_compress(zsrc.data(), zsrc.size(), csrc, clen, zlevel);

        size_t zdr = ZSTD_decompress(codec_dec.data(), codec_dec.size(),
                                     zsrc.data(), zcs);
        const size_t expect_dec = st.use_ext ? st.codec_len : bytes;
        bool ok = !ZSTD_isError(zdr) && zdr == expect_dec;
        if (ok)
          ok = verdict_ext_recover(st, bytes, codec_dec.data(), filt, unfilt,
                                   in);
        if (!ok) {
          std::fprintf(stderr, "verdict: zstd verify failed (%s)\n",
                       ds.name.c_str());
          return false;
        }

        const VerdictExtStage chosen = st;
        double tc =
            st.use_ext
                ? throughput(
                      [&] {
                        VerdictExtStage tmp;
                        verdict_ext_reapply(in, bytes, chosen, filt, rle, tmp);
                        (void)ZSTD_compress(zc.data(), zc.size(), tmp.codec_src,
                                            tmp.codec_len, zlevel);
                      },
                      bytes, min_each, deadline)
                : throughput(
                      [&] {
                        (void)ZSTD_compress(zc.data(), zc.size(),
                                            ds.data.data(), bytes, zlevel);
                      },
                      bytes, min_each, deadline);
        double td =
            st.use_ext
                ? throughput(
                      [&] {
                        (void)ZSTD_decompress(codec_dec.data(),
                                              codec_dec.size(), zsrc.data(),
                                              zcs);
                        (void)verdict_ext_invert(st, bytes, codec_dec.data(),
                                                 filt, unfilt);
                      },
                      bytes, min_each, deadline)
                : throughput(
                      [&] {
                        (void)ZSTD_decompress(codec_dec.data(),
                                              codec_dec.size(), zsrc.data(),
                                              zcs);
                      },
                      bytes, min_each, deadline);
        agg.add(tc, td, bytes, zcs);
        if (stats) {
          char zlabel[32];
          std::snprintf(zlabel, sizeof(zlabel), "zstd L%d", zlevel);
          std::printf("  %-10s %-16s: %6zu -> %6zu (%5.3fx)  comp %6.1f MB/s  dec %6.1f MB/s\n",
                      ds.name.c_str(), zlabel, bytes, zcs, static_cast<double>(bytes) / zcs, tc, td);
        }
      }
      return true;
    };
    auto run_zstd = [&](int zlevel) -> bool {
      char zlabel[32];
      std::snprintf(zlabel, sizeof(zlabel), "zstd     L%d%s", zlevel, flt_sfx);
      if (!paired) {
        VerdictRateAgg agg;
        if (!measure_zstd(zlevel, false, agg)) return false;
        std::printf(" %-24s %9.3f %12.1f %12.1f\n", zlabel, agg.ratio(),
                    agg.comp_mbs(), agg.dec_mbs());
        std::fflush(stdout);
        return true;
      }
      VerdictRateAgg det, brute_agg;
      if (!measure_zstd(zlevel, false, det)) return false;
      if (!measure_zstd(zlevel, true, brute_agg)) return false;
      print_verdict_pair(zlabel, brute_agg, det);
      return true;
    };
    if (full) {
      for (int l = 1; l <= ZSTD_maxCLevel(); ++l)
        if (!run_zstd(l)) return 1;
    } else {
      if (!run_zstd(ZSTD_maxCLevel())) return 1;  // highest level (22)
      if (!run_zstd(5)) return 1;
      if (!run_zstd(1)) return 1;
    }
  }
#else
  if (only_rows & spectra8::codec_bit(spectra8::Codec::Zstd)) {
    char zlabel[32];
    std::snprintf(zlabel, sizeof(zlabel), "zstd     Lmax%s", flt_sfx);
    print_verdict_na(zlabel, paired);
    std::snprintf(zlabel, sizeof(zlabel), "zstd     L5%s", flt_sfx);
    print_verdict_na(zlabel, paired);
    std::snprintf(zlabel, sizeof(zlabel), "zstd     L1%s", flt_sfx);
    print_verdict_na(zlabel, paired);
  }
#endif

  auto run_bwt_level = [&](int level, int block_log = 0,
                           const char* label_stem = nullptr) -> bool {
    if (!(only_rows & spectra8::codec_bit(spectra8::Codec::Prism))) return true;
    spectra8::CompressParams p;
    p.level = level;
    p.codec = spectra8::Codec::Prism;
    if (block_log > 0) p.block_log = block_log;
    char stem[32];
    if (label_stem) {
      std::snprintf(stem, sizeof(stem), "%s", label_stem);
    } else {
      std::snprintf(stem, sizeof(stem), "spectra8 Prism L%d", level);
    }
    return run_spectra8_row(stem, p);
  };

  if (full) {
    for (int l = spectra8::kMinLevel; l <= spectra8::kMaxLevel; ++l)
      if (!run_bwt_level(l)) return 1;
    if (dist == BenchDist::Text) {
      if (!run_bwt_level(11, 27, "spectra8 Prism L11 128M")) return 1;
    }
  } else {
    for (int l : {5, 9})
      if (!run_bwt_level(l)) return 1;
    if (dist == BenchDist::Text) {
      if (!run_bwt_level(11, 27, "spectra8 Prism L11 128M")) return 1;
    }
  }

  auto run_mosaic_level = [&](int level) -> bool {
    if (!(only_rows & spectra8::codec_bit(spectra8::Codec::Mosaic))) return true;
    spectra8::CompressParams p;
    p.level = level;
    p.codec = spectra8::Codec::Mosaic;
    p.filters = false;
    char stem[24];
    std::snprintf(stem, sizeof(stem), "spectra8 Mosaic L%d", level);
    return run_spectra8_row(stem, p);
  };

  if (full) {
    for (int l = spectra8::kMinLevel; l <= spectra8::kMaxLevel; ++l)
      if (!run_mosaic_level(l)) return 1;
  } else {
    for (int l : {5, 9})
      if (!run_mosaic_level(l)) return 1;
  }

  auto run_oracle_level = [&](int level) -> bool {
    if (!(only_rows & spectra8::codec_bit(spectra8::Codec::Oracle))) return true;
    spectra8::CompressParams p;
    p.level = level;
    p.codec = spectra8::Codec::Oracle;
    p.filters = false;
    char stem[24];
    std::snprintf(stem, sizeof(stem), "spectra8 Oracle L%d", level);
    return run_spectra8_row(stem, p);
  };

  if (full) {
    for (int l = spectra8::kMinLevel; l <= spectra8::kMaxLevel; ++l)
      if (!run_oracle_level(l)) return 1;
  } else {
    for (int l : {5, 9})
      if (!run_oracle_level(l)) return 1;
  }

  // Crystal: record-lattice models. L1-9 is the static lattice coder (slow
  // encoder search, table-driven rANS decode); L10-11 is context mixing.
  auto run_crystal_level = [&](int level) -> bool {
    if (!(only_rows & spectra8::codec_bit(spectra8::Codec::Crystal))) return true;
    spectra8::CompressParams p;
    p.level = level;
    p.codec = spectra8::Codec::Crystal;
    p.filters = false;
    char stem[24];
    std::snprintf(stem, sizeof(stem), "spectra8 Crystal L%d", level);
    return run_spectra8_row(stem, p);
  };

  if (full) {
    for (int l = spectra8::kMinLevel; l <= spectra8::kMaxLevel; ++l)
      if (!run_crystal_level(l)) return 1;
  } else {
    for (int l : {1, 5, 9, 11})
      if (!run_crystal_level(l)) return 1;
  }

  // Grating has one model. Level does not change the bitstream; 4 states below
  // 4096 symbols, 8 states from there up.
  auto run_grating = [&]() -> bool {
    if (!(only_rows & spectra8::codec_bit(spectra8::Codec::Grating))) return true;
    spectra8::CompressParams p;
    p.level = 5;
    p.codec = spectra8::Codec::Grating;
    p.filters = false;
    return run_spectra8_row("spectra8 Grating", p);
  };
  if (!run_grating()) return 1;

  // Laser: decoupled 5-stream range coder with optimal parser for binaries.
  auto run_laser_level = [&](int level) -> bool {
    if (!(only_rows & spectra8::codec_bit(spectra8::Codec::Laser))) return true;
    spectra8::CompressParams p;
    p.level = level;
    p.codec = spectra8::Codec::Laser;
    p.filters = false;
    char stem[24];
    std::snprintf(stem, sizeof(stem), "spectra8 Laser L%d", level);
    return run_spectra8_row(stem, p);
  };

  if (full) {
    for (int l = spectra8::kMinLevel; l <= spectra8::kMaxLevel; ++l)
      if (!run_laser_level(l)) return 1;
  } else {
    for (int l : {1, 5, 9})
      if (!run_laser_level(l)) return 1;
  }

#ifdef SPECTRA8_HAVE_LZMA
  if (only_rows & spectra8::codec_bit(spectra8::Codec::Lzma)) {
    // .xz buffer bound for a single-shot stream; same for every dataset.
    const size_t lbound = lzma_stream_buffer_bound(max_ext_len);
    std::vector<std::byte> lc(lbound);    // compress scratch
    std::vector<std::byte> lsrc(lbound);  // stable decode src
    // One LZMA config at a given preset, with the same external best-filter
    // (keep-smaller) logic as the zstd / BWT rows. No integrity check is added
    // so the size comparison is on the raw compressed payload.
    auto lzma_pack = [&](uint32_t preset, const spectra8::detail::u8* src,
                         std::size_t src_len, std::byte* out) -> size_t {
      size_t out_pos = 0;
      lzma_ret rc = lzma_easy_buffer_encode(
          preset, LZMA_CHECK_NONE, nullptr,
          reinterpret_cast<const uint8_t*>(src), src_len,
          reinterpret_cast<uint8_t*>(out), &out_pos, lbound);
      return rc == LZMA_OK ? out_pos : 0;
    };
    auto measure_lzma = [&](uint32_t preset, bool brute,
                             VerdictRateAgg& agg) -> bool {
      agg = VerdictRateAgg{};
      for (const Dataset& ds : datasets) {
        if (!ds.avg) continue;
        const auto* in =
            reinterpret_cast<const spectra8::detail::u8*>(ds.data.data());
        VerdictExtStage st;
        size_t cs_raw = 0;
        size_t cs_ext = 0;
        if (brute) {
          cs_ext = verdict_ext_brute(
              in, bytes, no_rle, no_dict, filt, rle, st,
              [&](const spectra8::detail::u8* p, std::size_t len) -> std::size_t {
                return lzma_pack(preset, p, len, lc.data());
              });
        } else {
          verdict_ext_prepare(in, bytes, no_filter, no_rle, no_dict, full_sample,
                              filt, rle, st);
        }
        if (!brute && !no_filter && st.has_ext) {
          cs_ext = lzma_pack(preset, st.codec_src, st.codec_len, lc.data());
          if (cs_ext == 0) {
            std::fprintf(stderr, "verdict: lzma filter compress failed (%s)\n",
                         ds.name.c_str());
            return false;
          }
          if (st.use_rle) {
            const size_t cs_filt = lzma_pack(preset, filt.data(), bytes, lc.data());
            if (cs_filt == 0) {
              std::fprintf(stderr,
                           "verdict: lzma filter-only compress failed (%s)\n",
                           ds.name.c_str());
              return false;
            }
            verdict_ext_prefer_smaller(st, bytes, filt, cs_ext, cs_filt, cs_ext);
          }
        }
        if (no_filter || !no_filter_fallback || !st.has_ext) {
          cs_raw = lzma_pack(preset, in, bytes, lc.data());
          if (cs_raw == 0) {
            std::fprintf(stderr, "verdict: lzma compress failed (%s)\n",
                         ds.name.c_str());
            return false;
          }
        }
        st.use_ext = verdict_ext_use(st, no_filter, no_filter_fallback, cs_raw,
                                     cs_ext);
        const size_t lcs = st.use_ext ? cs_ext : cs_raw;
        const spectra8::detail::u8* csrc =
            st.use_ext ? st.codec_src : in;
        const std::size_t clen = st.use_ext ? st.codec_len : bytes;
        if (lzma_pack(preset, csrc, clen, lsrc.data()) != lcs) {
          std::fprintf(stderr, "verdict: lzma re-pack mismatch (%s)\n",
                       ds.name.c_str());
          return false;
        }

        uint64_t memlimit = UINT64_MAX;
        size_t in_pos = 0, out_pos = 0;
        const std::size_t expect_dec = st.use_ext ? st.codec_len : bytes;
        lzma_ret dr = lzma_stream_buffer_decode(
            &memlimit, 0, nullptr, reinterpret_cast<const uint8_t*>(lsrc.data()),
            &in_pos, lcs, reinterpret_cast<uint8_t*>(codec_dec.data()), &out_pos,
            codec_dec.size());
        bool ok = dr == LZMA_OK && out_pos == expect_dec;
        if (ok)
          ok = verdict_ext_recover(st, bytes, codec_dec.data(), filt, unfilt,
                                   in);
        if (!ok) {
          std::fprintf(stderr, "verdict: lzma verify failed (%s)\n",
                       ds.name.c_str());
          return false;
        }

        const VerdictExtStage chosen = st;
        double tc =
            st.use_ext
                ? throughput(
                      [&] {
                        VerdictExtStage tmp;
                        verdict_ext_reapply(in, bytes, chosen, filt, rle, tmp);
                        (void)lzma_pack(preset, tmp.codec_src, tmp.codec_len,
                                        lc.data());
                      },
                      bytes, min_each, deadline)
                : throughput(
                      [&] { (void)lzma_pack(preset, in, bytes, lc.data()); },
                      bytes, min_each, deadline);
        double td =
            st.use_ext
                ? throughput(
                      [&] {
                        uint64_t ml = UINT64_MAX;
                        size_t ip = 0, op = 0;
                        [[maybe_unused]] const lzma_ret dr =
                            lzma_stream_buffer_decode(
                                &ml, 0, nullptr,
                                reinterpret_cast<const uint8_t*>(lsrc.data()),
                                &ip, lcs,
                                reinterpret_cast<uint8_t*>(codec_dec.data()),
                                &op, codec_dec.size());
                        (void)verdict_ext_invert(st, bytes, codec_dec.data(),
                                                 filt, unfilt);
                      },
                      bytes, min_each, deadline)
                : throughput(
                      [&] {
                        uint64_t ml = UINT64_MAX;
                        size_t ip = 0, op = 0;
                        [[maybe_unused]] const lzma_ret dr =
                            lzma_stream_buffer_decode(
                                &ml, 0, nullptr,
                                reinterpret_cast<const uint8_t*>(lsrc.data()),
                                &ip, lcs,
                                reinterpret_cast<uint8_t*>(codec_dec.data()),
                                &op, codec_dec.size());
                      },
                      bytes, min_each, deadline);
        agg.add(tc, td, bytes, lcs);
      }
      return true;
    };
    auto run_lzma = [&](uint32_t preset, const char* stem) -> bool {
      char tag[32];
      std::snprintf(tag, sizeof(tag), "%s%s", stem, flt_sfx);
      if (!paired) {
        VerdictRateAgg agg;
        if (!measure_lzma(preset, false, agg)) return false;
        std::printf(" %-24s %9.3f %12.1f %12.1f\n", tag, agg.ratio(),
                    agg.comp_mbs(), agg.dec_mbs());
        std::fflush(stdout);
        return true;
      }
      VerdictRateAgg det, brute_agg;
      if (!measure_lzma(preset, false, det)) return false;
      if (!measure_lzma(preset, true, brute_agg)) return false;
      print_verdict_pair(tag, brute_agg, det);
      return true;
    };
    if (full) {
      for (uint32_t p = 0; p <= 9; ++p) {
        char stem[24];
        std::snprintf(stem, sizeof(stem), "lzma     P%u", p);
        if (!run_lzma(p, stem)) return 1;
      }
      if (!run_lzma(9 | LZMA_PRESET_EXTREME, "lzma     P9e")) return 1;
    } else {
      if (!run_lzma(9 | LZMA_PRESET_EXTREME, "lzma     P9e")) return 1;
      if (!run_lzma(6, "lzma     P6")) return 1;
    }
  }
#else
  if (only_rows & spectra8::codec_bit(spectra8::Codec::Lzma)) {
    char llabel[32];
    std::snprintf(llabel, sizeof(llabel), "lzma     P9e%s", flt_sfx);
    print_verdict_na(llabel, paired);
    std::snprintf(llabel, sizeof(llabel), "lzma     P6%s", flt_sfx);
    print_verdict_na(llabel, paired);
  }
#endif

#ifdef SPECTRA8_HAVE_BZIP3
  if (only_rows & spectra8::codec_bit(spectra8::Codec::Bzip3)) {
    const size_t bbound = bz3_bound(max_ext_len);
    std::vector<std::byte> bc(bbound);
    std::vector<std::byte> bsrc(bbound);
    auto bzip3_pack = [&](uint32_t block_size, const spectra8::detail::u8* src,
                          std::size_t src_len, uint8_t* out,
                          size_t& out_sz) -> bool {
      out_sz = bbound;
      const int rc = bz3_compress(block_size,
                                  reinterpret_cast<const uint8_t*>(src), out,
                                  src_len, &out_sz);
      return rc == BZ3_OK && out_sz > 0;
    };
    auto measure_bzip3 = [&](uint32_t block_size, bool brute,
                              VerdictRateAgg& agg) -> bool {
      agg = VerdictRateAgg{};
      for (const Dataset& ds : datasets) {
        if (!ds.avg) continue;
        const auto* in =
            reinterpret_cast<const spectra8::detail::u8*>(ds.data.data());
        VerdictExtStage st;
        size_t cs_raw = 0;
        size_t cs_ext = 0;
        if (brute) {
          // bz3_decompress rejects frames whose block exceeds block_size, so
          // a candidate that compresses but will not round-trip is skipped.
          cs_ext = verdict_ext_brute(
              in, bytes, no_rle, no_dict, filt, rle, st,
              [&](const spectra8::detail::u8* p, std::size_t len) -> std::size_t {
                size_t sz = 0;
                if (!bzip3_pack(block_size, p, len,
                                reinterpret_cast<uint8_t*>(bc.data()), sz))
                  return 0;
                size_t probe = len;
                if (bz3_decompress(reinterpret_cast<const uint8_t*>(bc.data()),
                                   reinterpret_cast<uint8_t*>(codec_dec.data()),
                                   sz, &probe) != BZ3_OK)
                  return 0;
                return sz;
              });
        } else {
          verdict_ext_prepare(in, bytes, no_filter, no_rle, no_dict, full_sample,
                              filt, rle, st);
        }
        if (!brute && !no_filter && st.has_ext) {
          if (!bzip3_pack(block_size, st.codec_src, st.codec_len,
                          reinterpret_cast<uint8_t*>(bc.data()), cs_ext)) {
            std::fprintf(stderr,
                         "verdict: bzip3 filter compress failed (%s)\n",
                         ds.name.c_str());
            return false;
          }
          // libbzip3 high-level API limitation: bz3_compress emits blocks
          // larger than block_size when a block is incompressible, but
          // bz3_decompress rejects those frames (BZ3_ERR_MALFORMED_HEADER).
          // High-entropy filtered streams (e.g. bitshuffled continuous
          // gaussian floats) hit this at small block sizes, so probe the
          // roundtrip once and fall back to the raw input for this dataset.
          size_t probe_sz = st.codec_len;
          if (bz3_decompress(reinterpret_cast<const uint8_t*>(bc.data()),
                             reinterpret_cast<uint8_t*>(codec_dec.data()),
                             cs_ext, &probe_sz) != BZ3_OK) {
            st.has_ext = false;
            cs_ext = 0;
          } else if (st.use_rle) {
            size_t cs_filt = 0;
            if (bzip3_pack(block_size, filt.data(), bytes,
                           reinterpret_cast<uint8_t*>(bc.data()), cs_filt)) {
              size_t filt_probe = bytes;
              if (bz3_decompress(reinterpret_cast<const uint8_t*>(bc.data()),
                                 reinterpret_cast<uint8_t*>(codec_dec.data()),
                                 cs_filt, &filt_probe) == BZ3_OK) {
                verdict_ext_prefer_smaller(st, bytes, filt, cs_ext, cs_filt,
                                           cs_ext);
              }
            }
          }
        }
        if (no_filter || !no_filter_fallback || !st.has_ext) {
          if (!bzip3_pack(block_size, in, bytes,
                          reinterpret_cast<uint8_t*>(bc.data()), cs_raw)) {
            std::fprintf(stderr, "verdict: bzip3 compress failed (%s)\n",
                         ds.name.c_str());
            return false;
          }
        }
        st.use_ext = verdict_ext_use(st, no_filter, no_filter_fallback, cs_raw,
                                     cs_ext);
        const size_t bcs = st.use_ext ? cs_ext : cs_raw;
        const spectra8::detail::u8* csrc = st.use_ext ? st.codec_src : in;
        const std::size_t clen = st.use_ext ? st.codec_len : bytes;
        size_t repack = bcs;
        if (!bzip3_pack(block_size, csrc, clen,
                        reinterpret_cast<uint8_t*>(bsrc.data()), repack) ||
            repack != bcs) {
          std::fprintf(stderr, "verdict: bzip3 re-pack mismatch (%s)\n",
                       ds.name.c_str());
          return false;
        }

        size_t out_sz = st.use_ext ? st.codec_len : bytes;
        const int dr = bz3_decompress(
            reinterpret_cast<const uint8_t*>(bsrc.data()),
            reinterpret_cast<uint8_t*>(codec_dec.data()), bcs, &out_sz);
        bool ok = dr == BZ3_OK && out_sz == (st.use_ext ? st.codec_len : bytes);
        if (ok)
          ok = verdict_ext_recover(st, bytes, codec_dec.data(), filt, unfilt,
                                   in);
        if (!ok) {
          std::fprintf(stderr, "verdict: bzip3 verify failed (%s)\n",
                       ds.name.c_str());
          return false;
        }

        const VerdictExtStage chosen = st;
        double tc =
            st.use_ext
                ? throughput(
                      [&] {
                        VerdictExtStage tmp;
                        verdict_ext_reapply(in, bytes, chosen, filt, rle, tmp);
                        size_t os = bbound;
                        (void)bzip3_pack(block_size, tmp.codec_src, tmp.codec_len,
                                         reinterpret_cast<uint8_t*>(bc.data()),
                                         os);
                      },
                      bytes, min_each, deadline)
                : throughput(
                      [&] {
                        size_t os = bbound;
                        (void)bzip3_pack(block_size, in, bytes,
                                         reinterpret_cast<uint8_t*>(bc.data()),
                                         os);
                      },
                      bytes, min_each, deadline);
        double td =
            st.use_ext
                ? throughput(
                      [&] {
                        size_t os = st.codec_len;
                        (void)bz3_decompress(
                            reinterpret_cast<const uint8_t*>(bsrc.data()),
                            reinterpret_cast<uint8_t*>(codec_dec.data()), bcs,
                            &os);
                        (void)verdict_ext_invert(st, bytes, codec_dec.data(),
                                                 filt, unfilt);
                      },
                      bytes, min_each, deadline)
                : throughput(
                      [&] {
                        size_t os = bytes;
                        (void)bz3_decompress(
                            reinterpret_cast<const uint8_t*>(bsrc.data()),
                            reinterpret_cast<uint8_t*>(codec_dec.data()), bcs,
                            &os);
                      },
                      bytes, min_each, deadline);
        agg.add(tc, td, bytes, bcs);
      }
      return true;
    };
    auto run_bzip3 = [&](uint32_t block_size) -> bool {
      char blabel[32];
      bzip3_verdict_label(blabel, sizeof(blabel), block_size, flt_sfx);
      if (!paired) {
        VerdictRateAgg agg;
        if (!measure_bzip3(block_size, false, agg)) return false;
        std::printf(" %-24s %9.3f %12.1f %12.1f\n", blabel, agg.ratio(),
                    agg.comp_mbs(), agg.dec_mbs());
        std::fflush(stdout);
        return true;
      }
      VerdictRateAgg det, brute_agg;
      if (!measure_bzip3(block_size, false, det)) return false;
      if (!measure_bzip3(block_size, true, brute_agg)) return false;
      print_verdict_pair(blabel, brute_agg, det);
      return true;
    };
    if (!run_bzip3(bzip3_verdict_block_intermediate(bytes))) return 1;
    if (!run_bzip3(bzip3_verdict_block_highest(bytes))) return 1;
  }
#else
  if (only_rows & spectra8::codec_bit(spectra8::Codec::Bzip3)) {
    char blabel[32];
    std::snprintf(blabel, sizeof(blabel), "bzip3  mid%s", flt_sfx);
    print_verdict_na(blabel, paired);
    std::snprintf(blabel, sizeof(blabel), "bzip3  max%s", flt_sfx);
    print_verdict_na(blabel, paired);
  }
#endif

  // Final rows: Codec::Auto across gain thresholds (g5%..g0%), at L5 or each
  // codec's max effort (--auto-max-level). Higher gain keeps faster decoders;
  // g0% always takes the smallest (max ratio).
  auto run_auto_row = [&](int gain_pct, const char* stem) -> bool {
    spectra8::CompressParams p;
    p.codec = spectra8::Codec::Auto;
    p.level = 5;
    p.auto_max_level = auto_max_level;
    p.auto_min_gain_pct = gain_pct;
    p.filters = false;
    if (!run_spectra8_row(stem, p)) return false;

    // Winner distribution: one Auto compress per avg dataset on the same
    // kept payload as the row (external filter staging). Probe compresses are
    // discarded so throughput / keep-smaller trials do not skew the %.
    struct KeptPayload {
      std::vector<std::byte> data;
    };
    std::vector<KeptPayload> kept;
    kept.reserve(static_cast<size_t>(navg));
    for (const Dataset& ds : datasets) {
      if (!ds.avg) continue;
      KeptPayload k;
      if (no_filter) {
        k.data = ds.data;
        kept.push_back(std::move(k));
        continue;
      }
      const auto* in =
          reinterpret_cast<const spectra8::detail::u8*>(ds.data.data());
      VerdictExtStage st;
      verdict_ext_prepare(in, bytes, false, no_rle, no_dict, full_sample, filt,
                          rle, st);
      size_t cs_raw = 0;
      size_t cs_ext = 0;
      if (st.has_ext) {
        std::span<const std::byte> work_span(
            reinterpret_cast<const std::byte*>(st.codec_src), st.codec_len);
        auto fr = spectra8::compress(work_span, comp, p);
        if (!fr) {
          std::fprintf(stderr,
                       "verdict: auto stats filter compress failed (%s)\n",
                       ds.name.c_str());
          return false;
        }
        cs_ext = fr.value;
        if (st.use_rle) {
          std::span<const std::byte> filt_span(
              reinterpret_cast<const std::byte*>(filt.data()), bytes);
          auto ff = spectra8::compress(filt_span, comp, p);
          if (!ff) {
            std::fprintf(stderr,
                         "verdict: auto stats filter-only compress failed "
                         "(%s)\n",
                         ds.name.c_str());
            return false;
          }
          verdict_ext_prefer_smaller(st, bytes, filt, cs_ext, ff.value, cs_ext);
        }
      }
      if (!no_filter_fallback || !st.has_ext) {
        auto rr = spectra8::compress(ds.data, comp, p);
        if (!rr) {
          std::fprintf(stderr, "verdict: auto stats compress failed (%s)\n",
                       ds.name.c_str());
          return false;
        }
        cs_raw = rr.value;
      }
      st.use_ext =
          verdict_ext_use(st, false, no_filter_fallback, cs_raw, cs_ext);
      if (st.use_ext) {
        k.data.assign(reinterpret_cast<const std::byte*>(st.codec_src),
                      reinterpret_cast<const std::byte*>(st.codec_src) +
                          st.codec_len);
      } else {
        k.data = ds.data;
      }
      kept.push_back(std::move(k));
    }
    spectra8::reset_compress_stats();
    for (const KeptPayload& k : kept) {
      auto r = spectra8::compress(k.data, comp, p);
      if (!r) {
        std::fprintf(stderr, "verdict: auto stats final compress failed\n");
        return false;
      }
    }
    const auto report =
        spectra8::format_auto_codec_stats(spectra8::get_compress_stats());
    if (!report.empty()) std::printf("%s", report.c_str());
    return true;
  };
  if (only_rows & spectra8::codec_bit(spectra8::Codec::Auto)) {
    for (int gain : {5, 4, 3, 2, 1, 0}) {
      char stem[24];
      if (auto_max_level)
        std::snprintf(stem, sizeof(stem), "auto max g%d%%", gain);
      else
        std::snprintf(stem, sizeof(stem), "auto L5 g%d%% ", gain);
      if (!run_auto_row(gain, stem)) return 1;
    }
  }

  hr('-', rule);
#ifndef SPECTRA8_HAVE_ZSTD
  std::printf(" note: built without zstd; the zstd rows are unavailable "
              "(reconfigure with libzstd).\n");
#endif
#ifndef SPECTRA8_HAVE_LZMA
  std::printf(" note: built without liblzma; the lzma rows are unavailable "
              "(reconfigure with liblzma/xz).\n");
#endif
#ifndef SPECTRA8_HAVE_BZIP3
  std::printf(" note: built without libbzip3; the bzip3 rows are unavailable "
              "(reconfigure with libbzip3, e.g. `brew install bzip3`).\n");
#endif
  if (auto_max_level) {
    std::printf(" note: auto max g5%%..g0%% race available codecs at each "
                "flavour's highest effort (Beam/Prism/Mosaic/Crystal 11, Zstd max, "
                "Lzma 9, Bzip3 block_log 28). A slower decoder needs a size "
                "win of gain%% per doubling of decode time (zstd < beam < mosaic "
                "< prism < lzma < bzip3 < crystal CM); g0%% always takes the "
                "smallest (max ratio). Winner "
                "mixes under each row are one selection per avg dataset.\n");
  } else {
    std::printf(" note: auto L5 g5%%..g0%% race available codecs at level 5. "
                "A slower decoder needs a size win of gain%% per doubling of "
                "decode time (zstd < beam < crystal < mosaic < prism < lzma < "
                "bzip3; e.g. bzip3 over crystal, ~19x slower, needs ~4.2x gain); "
                "g0%% always takes the smallest (max ratio). Winner mixes "
                "under each row are "
                "one selection per avg dataset. Use --auto-max-level for "
                "each codec's max effort instead.\n");
  }
  std::printf(" note: single-thread; ratio = total original / total compressed; "
              "comp/dec MB/s = sequential throughput across avg datasets "
              "(harmonic mean of per-dataset rates, not arithmetic mean -- "
              "a 50/50 fast+slow Auto mix reports near the slow codec).\n");
  if (no_filter) {
    std::printf(" note: No external filter applied; each codec compresses the "
                "raw\n");
    std::printf("       dataset as-is. spectra8 built-in per-block filters are off.\n");
  } else if (paired) {
    std::printf(" note: each numeric column is brute-force filter / "
                "entropy-detected filter.\n");
    std::printf("       Brute compresses every candidate and keeps the smallest");
    if (no_filter_fallback)
      std::printf("\n       (no raw trial)");
    else
      std::printf("\n       (still keep-smaller versus raw)");
    std::printf(". Speeds time that winner; detection and the\n");
    std::printf("       search are not included. spectra8 built-in per-block filters are off.\n");
    if (!no_rle)
      std::printf("       Byte-RLE is also tried on size-preserving filters.\n");
    std::printf("       The Auto mix under each row is the entropy-detected payload.\n");
  } else if (no_filter_fallback) {
    std::printf(" note: All rows use the same external filter (detect once, "
                "encode only the\n");
    std::printf("       detected path; no raw compress trial). spectra8 built-in "
                "per-block filters are off.\n");
  } else if (no_rle) {
    std::printf(" note: All rows use the same external filter (detect once, "
                "compress raw and\n");
    std::printf("       filtered, keep smaller); spectra8 built-in per-block filters "
                "are off for a fair speed comparison.\n");
  } else {
    std::printf(" note: All rows use external filter + byte-RLE when smaller "
                "(detect once,\n");
    std::printf("       compress raw and transformed, keep smaller); spectra8 "
                "built-in per-block filters are off.\n");
  }
  if (!no_filter) {
    std::printf(" note: the entropy detector scores %s"
                " (`--filter-full-sample` to override).\n",
                full_sample ? "the whole block"
                            : "a 16 KiB mid-sample by default");
    if (paired)
      std::printf("       brute force scores every filter by compressed size.\n");
  }
  return 0;
}

// `bench --dir DIR`: the verdict table over the regular files of a directory
// (e.g. RNTuple pages or TTree baskets dumped to disk), one row per config,
// totals over all files. Ratio is printed both ways: out/in as storage
// tools report it, and in/out as the verdict does. Throughput is total bytes
// over total time (single thread), so large files weigh more, as on disk.
int run_dir_bench(const std::string& dir, bool no_filter, int auto_gain_pct,
                  spectra8::CodecMask auto_codecs, spectra8::CodecMask only_rows,
                  int level, unsigned parallel, bool stats) {
  namespace fs = std::filesystem;
  struct File {
    std::string name;
    std::vector<std::byte> data;
  };
  std::vector<File> files;
  std::error_code ec;
  for (const auto& e : fs::directory_iterator(dir, ec)) {
    if (!e.is_regular_file()) continue;
    std::ifstream in(e.path(), std::ios::binary | std::ios::ate);
    if (!in) continue;
    const auto len = static_cast<std::size_t>(in.tellg());
    if (len == 0) continue;
    File f;
    f.name = e.path().filename().string();
    f.data.resize(len);
    in.seekg(0);
    in.read(reinterpret_cast<char*>(f.data.data()), static_cast<std::streamsize>(len));
    files.push_back(std::move(f));
  }
  if (ec || files.empty()) {
    std::fprintf(stderr, "spectra8 bench --dir: no readable files in '%s'\n", dir.c_str());
    return 1;
  }
  std::sort(files.begin(), files.end(),
            [](const File& a, const File& b) { return a.name < b.name; });
  std::size_t total_in = 0, largest = 0;
  for (const File& f : files) {
    total_in += f.data.size();
    largest = std::max(largest, f.data.size());
  }

  struct Config {
    std::string label;
    spectra8::Codec codec;
    int level;
    spectra8::CodecMask mask = spectra8::kAutoCodecsAll;
    bool only_if_restricted = false;
  };
  const bool restricted = auto_codecs != spectra8::kAutoCodecsAll;
  const char* auto_sel_label = restricted && auto_codecs == spectra8::kAutoCodecsRoot
                                   ? "auto root" : "auto sel";
  std::vector<Config> configs;
  if (level <= 0) {
    configs = {
        {"spectra8 Flash L1", spectra8::Codec::Flash, 1},
        {"spectra8 Flash L5", spectra8::Codec::Flash, 5},
        {"zstd     L5", spectra8::Codec::Zstd, 5},
        {"zstd     L19", spectra8::Codec::Zstd, 19},
        {"spectra8 Beam L1", spectra8::Codec::Beam, 1},
        {"spectra8 Beam L5", spectra8::Codec::Beam, 5},
        {"spectra8 Prism L5", spectra8::Codec::Prism, 5},
        {"spectra8 Mosaic L5", spectra8::Codec::Mosaic, 5},
        {"spectra8 Oracle L5", spectra8::Codec::Oracle, 5},
        {"spectra8 Crystal L1", spectra8::Codec::Crystal, 1},
        {"spectra8 Crystal L5", spectra8::Codec::Crystal, 5},
        {"spectra8 Crystal L9", spectra8::Codec::Crystal, 9},
        {"spectra8 Grating", spectra8::Codec::Grating, 5},
        {"spectra8 Laser L5", spectra8::Codec::Laser, 5},
        {"spectra8 Laser L9", spectra8::Codec::Laser, 9},
        {"lzma     P6", spectra8::Codec::Lzma, 6},
        {"bzip3", spectra8::Codec::Bzip3, 5},
        {"auto L5", spectra8::Codec::Auto, 5, spectra8::kAutoCodecsAll},
    };
    configs.push_back({std::string(auto_sel_label) + " L5", spectra8::Codec::Auto, 5,
                       auto_codecs, true});
  } else {
    // `--level N`: one row per codec, every codec at N (zstd takes N as its
    // own level, lzma clamps to preset 9, bzip3 ignores it).
    const struct { const char* stem; spectra8::Codec codec; } kRows[] = {
        {"spectra8 Flash", spectra8::Codec::Flash},
        {"zstd    ", spectra8::Codec::Zstd},   {"spectra8 Beam", spectra8::Codec::Beam},
        {"spectra8 Prism", spectra8::Codec::Prism}, {"spectra8 Mosaic", spectra8::Codec::Mosaic},
        {"spectra8 Oracle", spectra8::Codec::Oracle}, {"spectra8 Crystal", spectra8::Codec::Crystal},
        {"spectra8 Grating", spectra8::Codec::Grating},
        {"spectra8 Laser", spectra8::Codec::Laser},
        {"lzma    ", spectra8::Codec::Lzma},   {"bzip3", spectra8::Codec::Bzip3},
        {"auto", spectra8::Codec::Auto},       {auto_sel_label, spectra8::Codec::Auto},
    };
    for (const auto& r : kRows) {
      const bool sel = r.codec == spectra8::Codec::Auto && r.stem == auto_sel_label;
      std::string label = r.stem;
      if (r.codec == spectra8::Codec::Lzma)
        label += " P" + std::to_string(std::min(level, 9));
      else if (r.codec == spectra8::Codec::Grating)
        label = "spectra8 Grating";
      else if (r.codec != spectra8::Codec::Bzip3)
        label += " L" + std::to_string(level);
      configs.push_back({label, r.codec, level,
                         sel ? auto_codecs : spectra8::kAutoCodecsAll, sel});
    }
  }

  std::printf("==============================================================================\n");
  std::printf(" spectra8 dir bench  (v%s)\n", std::string(spectra8::version()).c_str());
  if (parallel == 0) parallel = std::max(1u, std::thread::hardware_concurrency());
  std::printf("   %zu files, %.1f MiB total, largest %.1f KiB; %s; filters %s\n",
              files.size(), static_cast<double>(total_in) / 1048576.0,
              static_cast<double>(largest) / 1024.0,
              parallel <= 1 ? "single thread"
                            : (std::to_string(parallel) + " files in parallel").c_str(),
              no_filter ? "off" : "on");
  std::printf("   dir: %s\n", dir.c_str());
  std::printf("==============================================================================\n");
  std::printf(" %-24s %8s %8s %8s %11s %11s\n", "config", "out/in", "in/out", "out MiB",
              "comp MB/s", "dec MB/s");
  std::printf("------------------------------------------------------------------------------\n");
  // Two phases per config -- compress every file, then decompress every
  // file -- each run by `parallel` workers pulling files from a shared
  // counter and timed by wall clock, so the MB/s columns are the aggregate
  // throughput of the batch (per-core when parallel == 1).
  std::vector<std::vector<std::byte>> frames(files.size());
  auto run_phase = [&](const std::function<bool(std::size_t)>& work) -> double {
    std::atomic<std::size_t> next{0};
    std::atomic<bool> ok{true};
    auto worker = [&] {
      for (;;) {
        const std::size_t i = next.fetch_add(1);
        if (i >= files.size() || !ok.load()) return;
        if (!work(i)) ok.store(false);
      }
    };
    const double t0 = now_s();
    if (parallel <= 1) {
      worker();
    } else {
      std::vector<std::thread> pool;
      for (unsigned t = 0; t < parallel; ++t) pool.emplace_back(worker);
      for (auto& t : pool) t.join();
    }
    const double t1 = now_s();
    return ok.load() ? t1 - t0 : -1.0;
  };
  for (const Config& c : configs) {
    if (!(only_rows & spectra8::codec_bit(c.codec))) continue;
    if (!spectra8::codec_available(c.codec)) continue;
    if (c.only_if_restricted && !restricted) continue;
    std::atomic<std::size_t> total_out{0};
    std::atomic<int> fail_idx{-1};
    auto fail = [&](std::size_t i, const char* reason) {
      int none = -1;
      if (fail_idx.compare_exchange_strong(none, static_cast<int>(i))) {
        std::fprintf(stderr, "BENCH FAIL on file %zu (%s): %s\n", i, files[i].name.c_str(), reason);
      }
      return false;
    };
    spectra8::reset_compress_stats();
    const double t_comp = run_phase([&](std::size_t i) {
      const File& f = files[i];
      spectra8::CompressParams p;
      p.codec = c.codec;
      p.level = c.level;
      p.filters = !no_filter;
      p.auto_min_gain_pct = auto_gain_pct;
      p.auto_codecs = c.mask;
      auto& comp = frames[i];
      comp.resize(spectra8::compress_bound(f.data.size(), c.codec));
      auto r = spectra8::compress(f.data, comp, p);
      if (!r.ok()) return fail(i, "compress returned error");
      comp.resize(r.value);
      total_out.fetch_add(r.value);
      return true;
    });
    const auto cstats = spectra8::get_compress_stats();
    const double t_dec = t_comp < 0 ? -1.0 : run_phase([&](std::size_t i) {
      const File& f = files[i];
      std::vector<std::byte> dec(f.data.size());
      auto d = spectra8::decompress(frames[i], dec);
      if (!d.ok()) return fail(i, "decompress returned error");
      if (d.value != f.data.size()) return fail(i, "decompress size mismatch");
      if (std::memcmp(dec.data(), f.data.data(), f.data.size()) != 0)
        return fail(i, "decompressed content mismatch");
      return true;
    });
    const bool failed = t_comp < 0 || t_dec < 0;
    const std::string fail_name = failed && fail_idx.load() >= 0
                                      ? files[static_cast<std::size_t>(fail_idx.load())].name
                                      : std::string("?");
    std::string label = c.label;
    if (!no_filter) label += " +flt";
    if (c.codec == spectra8::Codec::Auto) {
      char g[16];
      std::snprintf(g, sizeof(g), " g%d%%", auto_gain_pct);
      label.insert(label.find(" +flt") == std::string::npos ? label.size() : label.find(" +flt"), g);
    }
    if (failed) {
      std::printf(" %-24s roundtrip FAILED on %s\n", label.c_str(), fail_name.c_str());
      continue;
    }
    const std::size_t out = total_out.load();
    std::printf(" %-24s %8.4f %8.3f %8.2f %11.2f %11.1f\n", label.c_str(),
                static_cast<double>(out) / static_cast<double>(total_in),
                static_cast<double>(total_in) / static_cast<double>(std::max<std::size_t>(out, 1)),
                static_cast<double>(out) / 1048576.0,
                static_cast<double>(total_in) / 1e6 / std::max(t_comp, 1e-9),
                static_cast<double>(total_in) / 1e6 / std::max(t_dec, 1e-9));
    // Per-row breakdown from the library's compress statistics: the Auto
    // codec mix is always shown for Auto rows; --stats adds the filter /
    // wire-format report (one line per non-zero bucket) for every row.
    const std::string report = stats ? spectra8::format_compress_stats(cstats)
                                     : c.codec == spectra8::Codec::Auto
                                           ? spectra8::format_auto_codec_stats(cstats)
                                           : std::string();
    if (!report.empty()) std::printf("%s", report.c_str());
  }
  std::printf("------------------------------------------------------------------------------\n");
  std::printf(" note: every file is compressed as one frame with the library defaults of the\n"
              "       config (built-in per-block filters when on; external codecs get the\n"
              "       L4EF filter wrapper). Roundtrip verified per file. --no-filter turns\n"
              "       the filter stage off for all rows; --auto-gain N sets the auto bar.\n");
  if (parallel > 1)
    std::printf(" note: --parallel %u: files are compressed / decompressed %u at a time, each\n"
                "       call single-threaded; comp/dec MB/s = total bytes over the wall time of\n"
                "       the batch (aggregate throughput of a %u-reader process).\n",
                parallel, parallel, parallel);
  return 0;
}

int run_h2d_bench(double budget_s, std::size_t bytes, spectra8::Codec codec) {
  if (budget_s <= 0) budget_s = 4.0;
  if (bytes == 0) bytes = 1u << 20;
  if (!spectra8::codec_available(codec)) {
    std::fprintf(stderr, "spectra8 bench --h2d: codec not available\n");
    return 1;
  }

  std::mt19937_64 rng(0x12d2);
  std::vector<std::byte> src(bytes);
  for (std::size_t i = 0; i < bytes; ++i) {
    if ((i & 15) == 0)
      src[i] = static_cast<std::byte>(rng() & 0xFF);
    else
      src[i] = static_cast<std::byte>(static_cast<unsigned>(src[i - 1]) + 1u);
  }

  spectra8::CompressParams cp;
  cp.codec = codec;
  cp.level = 5;
  cp.seek_table = !bench_codec_external(codec);
  cp.block_log = 16;  // 64 KiB pages
  std::vector<std::byte> frame(spectra8::compress_bound(bytes, codec));
  auto cr = spectra8::compress(src, frame, cp);
  if (!cr.ok()) {
    std::fprintf(stderr, "spectra8 bench --h2d: compress failed (%s)\n",
                 std::string(spectra8::status_message(cr.status)).c_str());
    return 1;
  }
  frame.resize(cr.value);

  spectra8::PagePump pump(spectra8::GpuBackend::Default);
  if (!pump.ok()) {
    std::fprintf(stderr, "spectra8 bench --h2d: transport failed\n");
    return 1;
  }
  if (auto s = pump.bind(frame); s != spectra8::Status::Ok) {
    std::fprintf(stderr, "spectra8 bench --h2d: bind failed (%s)\n",
                 std::string(spectra8::status_message(s)).c_str());
    return 1;
  }
  const std::size_t n = pump.total_bytes();
  void* dev = pump.transport().alloc_device(n);
  if (!dev) {
    std::fprintf(stderr, "spectra8 bench --h2d: device alloc failed\n");
    return 1;
  }

  std::vector<std::byte> host(n);
  auto warm = spectra8::decompress(frame, host);
  if (!warm.ok() || warm.value != n) {
    pump.transport().free_device(dev);
    std::fprintf(stderr, "spectra8 bench --h2d: warmup decompress failed\n");
    return 1;
  }

  auto time_loop = [&](auto&& fn) -> double {
    int iters = 0;
    const double t0 = now_s();
    do {
      fn();
      ++iters;
    } while (now_s() - t0 < budget_s / 3.0);
    return static_cast<double>(n) * static_cast<double>(iters) /
           std::max(now_s() - t0, 1e-9) / 1e6;
  };

  const double cpu_mbs = time_loop([&] {
    (void)spectra8::decompress(frame, host);
  });
  const double h2d_mbs = time_loop([&] {
    (void)pump.transport().h2d_async(dev, host.data(), n);
    (void)pump.transport().synchronize();
  });
  const double pump_mbs = time_loop([&] { (void)pump.push_all(dev); });

  std::vector<std::byte> back(n);
  const double consume_mbs = time_loop([&] {
    (void)pump.transport().d2h(back.data(), dev, n);
    volatile std::uint64_t acc = 0;
    for (std::size_t i = 0; i < n; i += 64)
      acc = acc + static_cast<unsigned>(back[i]);
    (void)acc;
  });

  (void)pump.transport().d2h(back.data(), dev, n);
  const bool match = std::memcmp(back.data(), src.data(), n) == 0;
  pump.transport().free_device(dev);

  std::printf("==============================================================================\n");
  std::printf(" spectra8 H2D page pump  (v%s)\n",
              std::string(spectra8::version()).c_str());
  std::printf("   backend %s  codec %s  %zu pages  %.2f MiB raw  %.1f KiB frame\n",
              spectra8::gpu_backend_name(pump.transport().backend()),
              bench_codec_name(codec), pump.page_count(),
              static_cast<double>(n) / 1048576.0,
              static_cast<double>(frame.size()) / 1024.0);
  std::printf("==============================================================================\n");
  std::printf(" %-22s %11s\n", "leg", "MB/s");
  std::printf("------------------------------------\n");
  std::printf(" %-22s %11.1f\n", "CPU decompress", cpu_mbs);
  std::printf(" %-22s %11.1f\n", "H2D (pre-decoded)", h2d_mbs);
  std::printf(" %-22s %11.1f\n", "pump (unzip+H2D)", pump_mbs);
  std::printf(" %-22s %11.1f\n", "dummy consume (D2H)", consume_mbs);
  std::printf("------------------------------------\n");
  std::printf(" roundtrip %s\n", match ? "ok" : "FAILED");
  if (pump_mbs > consume_mbs)
    std::printf(" note: pump is faster than the dummy consume; H2D is hidden.\n");
  else if (h2d_mbs + 1.0 < cpu_mbs)
    std::printf(" note: H2D is the slower leg; a GPU decoder would only win if it\n"
                "       beat CPU unzip enough to repay the PCIe uncompressed copy.\n");
  return match ? 0 : 1;
}

int print_codec_explain() {
  struct Row {
    const char* name;
    const char* words;
  };
  // Names as they appear in the benchmark tables; keywords only.
  static const Row rows[] = {
      {"spectra8 Flash L1", "Byte-aligned LZ77; raw interleaved literals; direct SIMD "
                        "vector copies (4-6 GB/s decode); NVMe line-rate ingestion"},
      {"spectra8 Flash L5", "Flash L1 plus optional Huffman literal bank / compact extras / "
                        "repeat-offset header when they pay (format-driven decode)"},
      {"spectra8 Beam", "LZ77; hash-chain lazy parser (L1 single probe, L5-9 chains); "
                    "7-slot repeat-offset cache; literal-len / match-len / offset-code "
                    "streams, Huffman or interleaved rANS; raw offset high bits, "
                    "tokenised low bits on record data; literals via Huffman, sub-delta "
                    "or Crystal static coder; 2-4 GB/s decode"},
      {"spectra8 Beam ultra", "Beam format; cost-based optimal parse (forward shortest path, "
                          "iterated token-histogram costs); L10-11; optional GPU match "
                          "finder; slow encode, Beam decode speed"},
      {"spectra8 Prism", "Burrows-Wheeler block sort (libsais suffix array); move-to-front; "
                     "zero-run coding; per-stream entropy pick (rANS, adaptive range "
                     "coder, order-1 at L7+, Huffman); multi-chain parallel inverse; "
                     "Beam fallback on run-heavy blocks; bzip2 lineage"},
      {"spectra8 Mosaic", "independent tiles (32 KiB-1 MiB); keep-smaller race of "
                      "lattice-ROLZ (per-context recency lists, stride copies), "
                      "mini-BWT + order-1 rANS or Crystal coder, Crystal BWT text "
                      "pipeline, Crystal lattice coder on raw tile"},
      {"spectra8 Oracle", "learned contexts; k-means of order-1 rows into 8/16 contexts; "
                      "per-cluster affine predictor (Q8) residual; race of learned "
                      "rANS, Beam, mini-BWT, Crystal coder; not in auto"},
      {"spectra8 Grating", "order-0 rANS over whole integers (1/2/4/8-byte); frequency "
                      "spectrum with an escape for the rare tail; 4-way decode, "
                      "8-way from 4096 symbols; no levels"},
      {"spectra8 Crystal", "static lattice coder; encoder-side modelling only: record "
                       "stride detect (1-16), per-lane context features (prev byte, "
                       "lane one/two records back, trend), MDL group merge, one "
                       "quantized table per group in arithmetic-coded header; decode "
                       "= lookup + rANS step, 16 interleaved streams (AVX2 gather body on x86-64); text blocks: BWT "
                       "-> MTF -> zero-run into same coder; L10-11 bitwise context "
                       "mixing"},
      {"spectra8 Laser", "LZMA2-class 5-stream decoupled range decoder; 12 LZMA states, "
                     "4 repeat offsets, matched literal coder; optimal/cluster parser "
                     "tuned for machine code; 2x+ LZMA2 decode throughput"},
      {"zstd", "external; LZ77 + FSE/Huffman; reference"},
      {"lzma", "external; LZ + range coder, optimal parse; reference"},
      {"bzip3", "external; BWT + LZP + order-2 context mixing; reference"},
      {"auto", "races the available flavours at the level; decode-speed-aware "
               "winner (--auto-gain)"},
  };
  constexpr int kName = 16;
  constexpr int kWidth = 78;
  std::printf("%-*s %s\n", kName, "codec", "algorithm (keywords)");
  std::printf("%-*s %s\n", kName, "----------------",
              "--------------------------------------------------------------");
  for (const Row& r : rows) {
    // Word-wrap the description at kWidth - kName - 1 columns, continuation
    // lines indented under the second column.
    const std::size_t avail = static_cast<std::size_t>(kWidth - kName - 1);
    std::string rest = r.words;
    bool first = true;
    while (!rest.empty()) {
      std::size_t cut = rest.size();
      if (cut > avail) {
        cut = rest.rfind(' ', avail);
        if (cut == std::string::npos || cut == 0) cut = avail;
      }
      std::printf("%-*s %s\n", kName, first ? r.name : "",
                  rest.substr(0, cut).c_str());
      first = false;
      rest.erase(0, cut);
      while (!rest.empty() && rest[0] == ' ') rest.erase(0, 1);
    }
  }
  return 0;
}

int run_benchmark(double budget, std::size_t bytes, bool gpu, bool compare,
                  bool no_ultra, bool filter, bool verdict,
                  bool verdict_no_filter, bool verdict_no_rle,
                  bool verdict_no_filter_fallback, bool bwt_chains,
                  bool verdict_full, bool verdict_no_dict,
                  bool filter_sample_full, bool verdict_auto_max_level,
                  bool verdict_brute, BenchDist dist, std::uint64_t seed,
                  spectra8::Codec codec, bool complete,
                  spectra8::CodecMask only_rows, bool stats) {
  if (budget <= 0) budget = 60.0;
  if (bytes == 0) {
    if (complete && dist == BenchDist::Text) {
      bytes = 0;
    } else if (dist == BenchDist::Text) {
      bytes = size_t{8} << 20;  // default 8 MiB for text
    } else if (dist == BenchDist::Binaries) {
      bytes = size_t{1} << 20;  // 1 MiB slice of a shared library
    } else {
      bytes = size_t{128} << 10;  // default 128 KiB
    }
  }

  if (bwt_chains) {
    if (int rc = run_bwt_chain_sweep(bytes, budget); rc != 0) return rc;
    std::printf("\n");
    return run_bwt_blocksize_sweep(bytes, budget);
  }

#ifndef SPECTRA8_HAVE_ZSTD
  const bool do_compare = false;
  if (compare) {
    std::fprintf(stderr,
                 "[compare] this build was compiled without zstd; the "
                 "--compare matrix is unavailable.\n"
                 "          reconfigure with zstd installed "
                 "(e.g. `brew install zstd`) to enable it.\n");
  }
#else
  const bool do_compare = compare;
#endif

  if (verdict)
    return run_verdict(budget, bytes, verdict_no_filter, verdict_no_rle,
                       verdict_no_filter_fallback, verdict_full,
                       verdict_no_dict, filter_sample_full,
                       verdict_auto_max_level, verdict_brute, dist, seed,
                       complete, only_rows, stats);

  // Split the wall-clock budget across the optional comparison sections so the
  // whole run still fits within roughly `budget` seconds.
  const bool do_gpu = gpu && spectra8::gpu_available();
  double main_frac = 1.0;
  if (gpu && do_compare) main_frac = 0.45;
  else if (gpu) main_frac = 0.65;
  else if (do_compare) main_frac = 0.55;
  if (filter) main_frac *= 0.7;  // reserve time for the filter comparison
  double cpu_budget = budget * main_frac;

  unsigned hc = std::thread::hardware_concurrency();
  int nthreads = (hc > 1) ? static_cast<int>(hc) : 1;

  // Levels 10 (near-optimal) and 11 (ultra) run the CPU optimal parser and are
  // far slower than 1-9, dominating the wall time, so `--no-ultra` drops both
  // for a much faster run. They are last in the list, so it is a shorter count.
  const int levels[] = {1,  3, 5, 6, 9, spectra8::kNearOptLevel,
                        spectra8::kUltraLevel};
  // --no-ultra drops both CPU optimal-parser levels (10 near-optimal, 11 ultra).
  const int num_levels = no_ultra ? 5 : 7;

  auto datasets = make_datasets(bytes, dist, seed);
  if (bytes == 0 && !datasets.empty()) {
    bytes = datasets[0].data.size();
  }

  // Per-level / per-thread-count parameter sets. Blocks are independent, so a
  // higher thread count yields identical bytes; only throughput changes.
  // External codecs reject threads != 1, so both columns stay single-thread.
  auto comp_params = [codec](int level, int threads) {
    spectra8::CompressParams p;
    p.codec = codec;
    p.level = level;
    p.threads = bench_codec_external(codec) ? 1 : threads;
    return p;
  };

  char comp_mt_lbl[24];
  char dec_mt_lbl[24];
  std::snprintf(comp_mt_lbl, sizeof(comp_mt_lbl), "comp %dT", nthreads);
  std::snprintf(dec_mt_lbl, sizeof(dec_mt_lbl), "dec %dT", nthreads);

  hr('=');
  std::printf(" spectra8 benchmark  (v%s)  codec %s\n",
              std::string(spectra8::version()).c_str(),
              bench_codec_name(codec));
  const std::string size_str = human_size(bytes);
  std::printf("   budget %.0f s   datasets %zu   size %s   "
              "levels %s   threads %d   seed 0x%llx\n",
              budget, datasets.size(), size_str.c_str(),
              no_ultra ? "{1,3,5,6,9}" : "{1,3,5,6,9,10=near-opt,11=ultra}",
              nthreads, static_cast<unsigned long long>(seed));
  std::printf("   throughput in MB/s; 1T = single thread, %dT = all cores\n",
              nthreads);
  if (bench_codec_external(codec)) {
    std::printf("   note: %s is single-thread; the all-core columns match 1T\n",
                bench_codec_name(codec));
  }
  hr('=');

  const double start = now_s();
  const double deadline = start + cpu_budget;
  // Each cell now performs four throughput measurements (comp 1T/NT, dec 1T/NT).
  const int meas_per_cell = 4;
  const int total_cells = static_cast<int>(datasets.size()) * num_levels;
  int done_cells = 0;

  // Per-level corpus aggregates for the summary row (harmonic rates + overall
  // ratio), matching --verdict. Per-dataset lines above stay as measured.
  struct LevelAgg {
    double inv_c1 = 0, inv_cN = 0, inv_d1 = 0, inv_dN = 0;
    double orig = 0, comp = 0;
    int n = 0;
  } level_agg[16];

  std::vector<std::byte> comp(spectra8::compress_bound(bytes));
  std::vector<std::byte> dec(bytes);

  for (const Dataset& ds : datasets) {
    std::printf("\n%-8s  (%s)\n", ds.name.c_str(), size_str.c_str());
    std::printf(" %-6s %8s %10s %10s %10s %10s\n", "level", "ratio", "comp 1T",
                comp_mt_lbl, "dec 1T", dec_mt_lbl);
    hr('-');

    for (int li = 0; li < num_levels; ++li) {
      int level = levels[li];

      double remaining = deadline - now_s();
      int remaining_cells = total_cells - done_cells;
      double per_cell = remaining / (remaining_cells > 0 ? remaining_cells : 1);
      double min_each = per_cell / meas_per_cell;
      if (min_each < 0.02) min_each = 0.02;
      if (min_each > 1.5) min_each = 1.5;

      auto p1 = comp_params(level, 1);
      auto pN = comp_params(level, 0);  // 0 => auto (all cores)
      spectra8::DecompressParams d1{1};
      spectra8::DecompressParams dN{0};

      auto cr = spectra8::compress(ds.data, comp, p1);
      if (!cr) {
        std::fprintf(stderr, "compress failed (%s L%d)\n", ds.name.c_str(), level);
        return 1;
      }
      size_t csize = cr.value;
      std::span<const std::byte> cspan(comp.data(), csize);
      auto dr = spectra8::decompress(cspan, dec, d1);
      if (!dr || dr.value != bytes ||
          std::memcmp(dec.data(), ds.data.data(), bytes) != 0) {
        std::fprintf(stderr, "verify failed (%s L%d)\n", ds.name.c_str(), level);
        return 1;
      }

      double c1 = throughput(
          [&] { (void)spectra8::compress(ds.data, comp, p1); }, bytes,
          min_each, deadline);
      double cN = throughput(
          [&] { (void)spectra8::compress(ds.data, comp, pN); }, bytes,
          min_each, deadline);
      double dd1 = throughput(
          [&] { (void)spectra8::decompress(cspan, dec, d1); }, bytes,
          min_each, deadline);
      double ddN = throughput(
          [&] { (void)spectra8::decompress(cspan, dec, dN); }, bytes,
          min_each, deadline);

      double ratio = static_cast<double>(bytes) / static_cast<double>(csize);
      std::printf(" %-6d %8.3f %10.1f %10.1f %10.1f %10.1f\n", level, ratio, c1,
                  cN, dd1, ddN);

      if (ds.avg) {
        LevelAgg& a = level_agg[level];
        add_inv_rate(a.inv_c1, c1);
        add_inv_rate(a.inv_cN, cN);
        add_inv_rate(a.inv_d1, dd1);
        add_inv_rate(a.inv_dN, ddN);
        a.orig += static_cast<double>(bytes);
        a.comp += static_cast<double>(csize);
        ++a.n;
      }
      ++done_cells;
    }
  }

  size_t n_avg = 0;
  for (const Dataset& ds : datasets)
    if (ds.avg) ++n_avg;

  std::printf("\n");
  hr('=');
  std::printf(" summary (sequential over %zu datasets, excl. "
              "text/source/json/random; speedup = %dT / 1T; "
              "MB/s = harmonic mean, ratio = total in/out)\n",
              n_avg, nthreads);
  hr('=');
  std::printf(" %-6s %8s %10s %10s %7s %10s %10s %7s\n", "level", "ratio",
              "comp 1T", comp_mt_lbl, "spd", "dec 1T", dec_mt_lbl, "spd");
  hr('-');
  double best_cspd = 0, best_dspd = 0;
  for (int li = 0; li < num_levels; ++li) {
    int level = levels[li];
    const LevelAgg& a = level_agg[level];
    double c1 = harm_rate(a.inv_c1, a.n), cN = harm_rate(a.inv_cN, a.n);
    double d1 = harm_rate(a.inv_d1, a.n), dN = harm_rate(a.inv_dN, a.n);
    double ratio = a.comp > 0 ? a.orig / a.comp : 0;
    double cspd = c1 > 0 ? cN / c1 : 0;
    double dspd = d1 > 0 ? dN / d1 : 0;
    if (cspd > best_cspd) best_cspd = cspd;
    if (dspd > best_dspd) best_dspd = dspd;
    std::printf(" %-6d %8.3f %10.1f %10.1f %6.2fx %10.1f %10.1f %6.2fx\n", level,
                ratio, c1, cN, cspd, d1, dN, dspd);
  }
  hr('-');
  std::printf(" peak speedup: comp %.2fx  decomp %.2fx  (on %d threads)\n",
              best_cspd, best_dspd, nthreads);

#ifdef SPECTRA8_HAVE_ZSTD
  if (do_compare) {
    std::vector<std::byte> zc(ZSTD_compressBound(bytes));      // compress scratch
    std::vector<std::byte> zsrc(ZSTD_compressBound(bytes));    // stable frame to decode

    // zstd multi-threaded compression via the advanced API. nbWorkers > 0
    // splits the input into jobs across worker threads (the analogue of
    // spectra8's per-block parallelism). If the linked zstd was built without
    // multithreading the parameter set errors and we fall back to one thread
    // (the all-core column then matches the single-thread one).
    ZSTD_CCtx* mtctx = ZSTD_createCCtx();
    size_t set_rc = ZSTD_CCtx_setParameter(mtctx, ZSTD_c_nbWorkers, nthreads);
    const bool zstd_mt = (nthreads > 1) && !ZSTD_isError(set_rc);

    struct CmpCell {
      const char* name;
      int level, zl;
      size_t lcs, zcs, zcsmt;       // compressed sizes (for corpus ratio)
      double rl, rz, rzmt;          // ratios: spectra8, zstd 1T, zstd MT
      double cl1, clN, cz1, czN;    // compress MB/s: spectra8 1T/NT, zstd 1T/NT
      double dl, dlN, dz;           // decode MB/s: spectra8 1T, spectra8 NT, zstd 1T
      bool avg;                     // included in the averaged summary rows
    };
    std::vector<CmpCell> cells;
    cells.reserve(datasets.size() * static_cast<size_t>(num_levels));

    const double cmp_budget = std::max(3.0, budget * (gpu ? 0.30 : 0.45));
    const double cmp_start = now_s();
    const double cmp_deadline = cmp_start + cmp_budget;
    const int cmp_meas = 7;  // cl1, clN, cz1, czN, dl, dlN, dz per cell
    const int cmp_total = static_cast<int>(datasets.size()) * num_levels;
    int cmp_done = 0;

    for (const Dataset& ds : datasets) {
      for (int li = 0; li < num_levels; ++li) {
        int level = levels[li];
        int zl = zstd_level_for(level);

        double remaining = cmp_deadline - now_s();
        int rem_cells = cmp_total - cmp_done;
        double per_cell = remaining / (rem_cells > 0 ? rem_cells : 1);
        double min_each = per_cell / cmp_meas;
        if (min_each < 0.02) min_each = 0.02;
        if (min_each > 1.0) min_each = 1.0;

        spectra8::CompressParams lp1;
        lp1.codec = codec;
        lp1.level = level;
        lp1.threads = 1;
        spectra8::CompressParams lpN;
        lpN.codec = codec;
        lpN.level = level;
        lpN.threads = bench_codec_external(codec) ? 1 : 0;  // 0 = all cores
        spectra8::DecompressParams ld{1};
        spectra8::DecompressParams ldN{0};  // auto = all cores

        // spectra8 frame (deterministic regardless of thread count, so the
        // 1T/NT compress loops leave an identical valid frame in `comp`).
        auto lr = spectra8::compress(ds.data, comp, lp1);
        if (!lr) {
          std::fprintf(stderr, "compare: spectra8 compress failed (%s L%d)\n",
                       ds.name.c_str(), level);
          return 1;
        }
        size_t lcs = lr.value;
        std::span<const std::byte> lspan(comp.data(), lcs);

        // zstd single-thread (simple API). Keep a stable copy in `zsrc` for the
        // decode benchmark, because the MT compress + the throughput loops
        // overwrite the `zc` scratch buffer.
        size_t zcs = ZSTD_compress(zc.data(), zc.size(), ds.data.data(), bytes, zl);
        if (ZSTD_isError(zcs)) {
          std::fprintf(stderr, "compare: zstd compress failed (%s zl%d): %s\n",
                       ds.name.c_str(), zl, ZSTD_getErrorName(zcs));
          return 1;
        }
        std::memcpy(zsrc.data(), zc.data(), zcs);
        size_t zdr = ZSTD_decompress(dec.data(), dec.size(), zsrc.data(), zcs);
        if (ZSTD_isError(zdr) || zdr != bytes ||
            std::memcmp(dec.data(), ds.data.data(), bytes) != 0) {
          std::fprintf(stderr, "compare: zstd verify failed (%s zl%d)\n",
                       ds.name.c_str(), zl);
          return 1;
        }

        // zstd multi-thread (advanced API): size for ratio + roundtrip check.
        ZSTD_CCtx_setParameter(mtctx, ZSTD_c_compressionLevel, zl);
        size_t zcsmt =
            ZSTD_compress2(mtctx, zc.data(), zc.size(), ds.data.data(), bytes);
        if (ZSTD_isError(zcsmt)) {
          std::fprintf(stderr, "compare: zstd MT compress failed (%s zl%d): %s\n",
                       ds.name.c_str(), zl, ZSTD_getErrorName(zcsmt));
          return 1;
        }
        {
          size_t v = ZSTD_decompress(dec.data(), dec.size(), zc.data(), zcsmt);
          if (ZSTD_isError(v) || v != bytes ||
              std::memcmp(dec.data(), ds.data.data(), bytes) != 0) {
            std::fprintf(stderr, "compare: zstd MT verify failed (%s zl%d)\n",
                         ds.name.c_str(), zl);
            return 1;
          }
        }

        double cl1 = throughput(
            [&] { (void)spectra8::compress(ds.data, comp, lp1); }, bytes,
            min_each, cmp_deadline);
        double clN = throughput(
            [&] { (void)spectra8::compress(ds.data, comp, lpN); }, bytes,
            min_each, cmp_deadline);
        double cz1 = throughput(
            [&] {
              (void)ZSTD_compress(zc.data(), zc.size(), ds.data.data(), bytes, zl);
            },
            bytes, min_each, cmp_deadline);
        double czN = throughput(
            [&] {
              (void)ZSTD_compress2(mtctx, zc.data(), zc.size(), ds.data.data(),
                                   bytes);
            },
            bytes, min_each, cmp_deadline);
        double dl = throughput(
            [&] { (void)spectra8::decompress(lspan, dec, ld); }, bytes,
            min_each, cmp_deadline);
        double dlN = throughput(
            [&] { (void)spectra8::decompress(lspan, dec, ldN); }, bytes,
            min_each, cmp_deadline);
        double dz = throughput(
            [&] {
              (void)ZSTD_decompress(dec.data(), dec.size(), zsrc.data(), zcs);
            },
            bytes, min_each, cmp_deadline);

        cells.push_back(CmpCell{
            ds.name.c_str(), level, zl, lcs, zcs, zcsmt,
            static_cast<double>(bytes) / static_cast<double>(lcs),
            static_cast<double>(bytes) / static_cast<double>(zcs),
            static_cast<double>(bytes) / static_cast<double>(zcsmt),
            cl1, clN, cz1, czN, dl, dlN, dz, ds.avg});
        ++cmp_done;
      }
    }

    auto cell_at = [&](int di, int li) -> const CmpCell& {
      return cells[static_cast<size_t>(di) * num_levels + li];
    };
    const int nds = static_cast<int>(datasets.size());
    // Summary excludes the text-like datasets (see Dataset::avg).

    // --- Table 1: single-thread head-to-head -------------------------------
    std::printf("\n");
    hr('=');
    std::printf(" spectra8 vs zstd %s  (level-matched, single-thread head-to-head)\n",
                ZSTD_versionString());
    std::printf("   ratio = in/out (higher is better); MB/s on one thread; "
                "L = spectra8, Z = zstd\n");
    std::printf("   level pairing: spectra8 {1,3,5,6,9,10,11} <-> zstd "
                "{1,3,6,9,17,18,19}\n");
    hr('=');
    std::printf(" %-8s %3s %3s %8s %8s %8s %8s %8s %8s\n", "dataset", "lvl",
                "zl", "ratio L", "ratio Z", "cMB/s L", "cMB/s Z", "dMB/s L",
                "dMB/s Z");
    hr('-');
    for (int di = 0; di < nds; ++di) {
      for (int li = 0; li < num_levels; ++li) {
        const CmpCell& c = cell_at(di, li);
        std::printf(" %-8s %3d %3d %8.3f %8.3f %8.1f %8.1f %8.1f %8.1f\n",
                    c.name, c.level, c.zl, c.rl, c.rz, c.cl1, c.cz1, c.dl, c.dz);
      }
      hr('-');
    }
    std::printf("\n");
    hr('=');
    std::printf(" single-thread summary (sequential over %zu datasets, excl. "
                "text/source/json/random; MB/s = harmonic mean; "
                "ratio = total in/out; adv = spectra8 ratio / zstd ratio)\n",
                n_avg);
    hr('=');
    std::printf(" %-5s %3s %8s %8s %7s %8s %8s %8s %8s\n", "lvl", "zl",
                "ratio L", "ratio Z", "adv", "cMB/s L", "cMB/s Z", "dMB/s L",
                "dMB/s Z");
    hr('-');
    for (int li = 0; li < num_levels; ++li) {
      double inv_cl1 = 0, inv_cz1 = 0, inv_dl = 0, inv_dz = 0;
      double orig = 0, comp_l = 0, comp_z = 0;
      int n = 0;
      for (int di = 0; di < nds; ++di) {
        const CmpCell& c = cell_at(di, li);
        if (!c.avg) continue;
        add_inv_rate(inv_cl1, c.cl1);
        add_inv_rate(inv_cz1, c.cz1);
        add_inv_rate(inv_dl, c.dl);
        add_inv_rate(inv_dz, c.dz);
        orig += static_cast<double>(bytes);
        comp_l += static_cast<double>(c.lcs);
        comp_z += static_cast<double>(c.zcs);
        ++n;
      }
      const double rl = comp_l > 0 ? orig / comp_l : 0;
      const double rz = comp_z > 0 ? orig / comp_z : 0;
      const double radv = rz > 0 ? rl / rz : 0;
      std::printf(" %-5d %3d %8.3f %8.3f %6.2fx %8.1f %8.1f %8.1f %8.1f\n",
                  levels[li], zstd_level_for(levels[li]), rl, rz, radv,
                  harm_rate(inv_cl1, n), harm_rate(inv_cz1, n),
                  harm_rate(inv_dl, n), harm_rate(inv_dz, n));
    }

    // --- Table 2: all-core compression -------------------------------------
    std::printf("\n");
    hr('=');
    std::printf(" all-core compression: spectra8 (block-parallel) vs zstd "
                "(--workers=%d)%s\n",
                nthreads, zstd_mt ? "" : "  [zstd MT unavailable]");
    std::printf("   cMB/s = all-core compress; spd = all-core / single-thread "
                "(from the table above)\n");
    std::printf("   ratio Z vs Zmt shows zstd's ratio cost from job splitting; "
                "spectra8 ratio is thread-independent\n");
    std::printf("   dL %dT = spectra8 all-core (block-parallel) decode; zstd "
                "has no multi-threaded single-frame decode\n",
                nthreads);
    std::printf("   note: random/incompressible frames stay single-thread on "
                "decode by design, so dL %dT ~ dL 1T there\n",
                nthreads);
    hr('=');
    std::printf(" %-8s %3s %3s %7s %7s %8s %8s %6s %9s %9s %6s\n", "dataset",
                "lvl", "zl", "ratio L", "rat Zmt", "cMB/s L", "cMB/s Z", "spd L",
                "dL 1T", dec_mt_lbl, "dspd L");
    hr('-');
    for (int di = 0; di < nds; ++di) {
      for (int li = 0; li < num_levels; ++li) {
        const CmpCell& c = cell_at(di, li);
        double sl = c.cl1 > 0 ? c.clN / c.cl1 : 0;
        double dsl = c.dl > 0 ? c.dlN / c.dl : 0;
        std::printf(" %-8s %3d %3d %7.3f %7.3f %8.1f %8.1f %5.2fx %9.1f %9.1f "
                    "%5.2fx\n",
                    c.name, c.level, c.zl, c.rl, c.rzmt, c.clN, c.czN, sl, c.dl,
                    c.dlN, dsl);
      }
      hr('-');
    }
    std::printf("\n");
    hr('=');
    std::printf(" all-core summary (sequential over %zu datasets, excl. "
                "text/source/json/random; MB/s = harmonic mean; "
                "ratio = total in/out)\n",
                n_avg);
    hr('=');
    std::printf(" %-5s %3s %7s %7s %8s %8s %6s %9s %9s %6s\n", "lvl", "zl",
                "ratio L", "rat Zmt", "cMB/s L", "cMB/s Z", "spd L", "dL 1T",
                dec_mt_lbl, "dspd L");
    hr('-');
    for (int li = 0; li < num_levels; ++li) {
      double inv_cl1 = 0, inv_clN = 0, inv_czN = 0, inv_dl = 0, inv_dlN = 0;
      double orig = 0, comp_l = 0, comp_zmt = 0;
      int n = 0;
      for (int di = 0; di < nds; ++di) {
        const CmpCell& c = cell_at(di, li);
        if (!c.avg) continue;
        add_inv_rate(inv_cl1, c.cl1);
        add_inv_rate(inv_clN, c.clN);
        add_inv_rate(inv_czN, c.czN);
        add_inv_rate(inv_dl, c.dl);
        add_inv_rate(inv_dlN, c.dlN);
        orig += static_cast<double>(bytes);
        comp_l += static_cast<double>(c.lcs);
        comp_zmt += static_cast<double>(c.zcsmt);
        ++n;
      }
      const double cl1 = harm_rate(inv_cl1, n);
      const double clN = harm_rate(inv_clN, n);
      const double dl = harm_rate(inv_dl, n);
      const double dlN = harm_rate(inv_dlN, n);
      const double sl = cl1 > 0 ? clN / cl1 : 0;
      const double dsl = dl > 0 ? dlN / dl : 0;
      std::printf(" %-5d %3d %7.3f %7.3f %8.1f %8.1f %4.2fx %9.1f %9.1f %4.2fx\n",
                  levels[li], zstd_level_for(levels[li]),
                  comp_l > 0 ? orig / comp_l : 0,
                  comp_zmt > 0 ? orig / comp_zmt : 0, clN, harm_rate(inv_czN, n),
                  sl, dl, dlN, dsl);
    }
    hr('-');
    std::printf(" note: zstd MT uses the advanced API (ZSTD_c_nbWorkers); its "
                "ratio can dip vs single-thread\n");
    std::printf("       due to job splitting. spectra8 compresses blocks in "
                "parallel with identical output bytes.\n");

    ZSTD_freeCCtx(mtctx);
  }
#endif  // SPECTRA8_HAVE_ZSTD

  if (gpu && !do_gpu) {
    std::printf("\n[gpu] requested but no Metal device is available; "
                "skipping GPU comparison.\n");
  } else if (do_gpu) {
    std::printf("\n");
    hr('=');
    std::printf(" GPU (Metal) match finder vs CPU   level 6   compress throughput"
                " + ratio\n");
    std::printf("   GPU offloads match finding; the CPU does the parse, entropy"
                " coding and framing.\n");
    if (codec != spectra8::Codec::Beam) {
      std::printf("   GPU comparison is Beam-only; --codec %s applies to the "
                  "tables above.\n",
                  bench_codec_name(codec));
    }
    hr('=');
    std::printf(" %-8s %10s %10s %10s %9s %9s\n", "dataset", "cpu 1T",
                comp_mt_lbl, "gpu", "ratio C", "ratio G");
    hr('-');

    const double gdeadline = now_s() + ((budget * 0.35 < 2.0) ? 2.0 : budget * 0.35);
    int gcells = static_cast<int>(datasets.size());
    int gdone = 0;
    for (const Dataset& ds : datasets) {
      double remaining = gdeadline - now_s();
      double per = remaining / ((gcells - gdone) > 0 ? (gcells - gdone) : 1);
      double min_each = per / 3.0;
      if (min_each < 0.05) min_each = 0.05;
      if (min_each > 1.0) min_each = 1.0;

      spectra8::CompressParams cpu1;
      cpu1.level = 6;
      cpu1.threads = 1;
      spectra8::CompressParams cpuN;
      cpuN.level = 6;
      cpuN.threads = 0;
      spectra8::CompressParams gp;
      gp.level = 6;
      gp.threads = 0;
      gp.gpu = true;

      auto rc = spectra8::compress(ds.data, comp, cpu1);
      size_t ccs = rc ? rc.value : bytes;
      auto rg = spectra8::compress(ds.data, comp, gp);
      size_t gcs = rg ? rg.value : bytes;
      // Confirm the GPU-produced frame decodes on the standard decoder.
      {
        std::span<const std::byte> gs(comp.data(), gcs);
        auto vr = spectra8::decompress(gs, dec);
        if (!vr || vr.value != bytes ||
            std::memcmp(dec.data(), ds.data.data(), bytes) != 0) {
          std::fprintf(stderr, "gpu verify failed (%s)\n", ds.name.c_str());
          return 1;
        }
      }

      double t_c1 = throughput(
          [&] { (void)spectra8::compress(ds.data, comp, cpu1); }, bytes,
          min_each, gdeadline);
      double t_cN = throughput(
          [&] { (void)spectra8::compress(ds.data, comp, cpuN); }, bytes,
          min_each, gdeadline);
      double t_g = throughput(
          [&] { (void)spectra8::compress(ds.data, comp, gp); }, bytes,
          min_each, gdeadline);

      std::printf(" %-8s %10.1f %10.1f %10.1f %9.3f %9.3f\n", ds.name.c_str(),
                  t_c1, t_cN, t_g,
                  static_cast<double>(bytes) / static_cast<double>(ccs),
                  static_cast<double>(bytes) / static_cast<double>(gcs));
      ++gdone;
    }
    hr('-');
    std::printf(" note: GPU uses 64 KiB blocks + a threadgroup hash (one block"
                " per threadgroup);\n");
    std::printf("       lower ratio than the CPU chain finder is expected. "
                "Decode is identical (CPU).\n");

    // Ultra: the GPU returns top-K candidates per position and the CPU optimal
    // parser decides. Ratio is deterministic, so one compress each suffices;
    // ultra is slow, so honor a tight deadline and skip the rest if it expires.
    std::printf("\n");
    hr('=');
    std::printf(" GPU top-K candidates vs CPU   level 11 (--ultra)   ratio +"
                " compress time\n");
    std::printf("   GPU finds many candidates/position; the CPU optimal parser"
                " decides. Ratio is the goal.\n");
    hr('=');
    std::printf(" %-8s %10s %10s %9s %9s\n", "dataset", "cpu s", "gpu s",
                "ratio C", "ratio G");
    hr('-');
    const double udeadline =
        now_s() + (budget * 0.20 < 3.0 ? 3.0 : budget * 0.20);
    for (const Dataset& ds : datasets) {
      if (now_s() >= udeadline) {
        std::printf(" (skipped remaining datasets: ultra time budget reached)\n");
        break;
      }
      spectra8::CompressParams cu;
      cu.level = spectra8::kUltraLevel;
      cu.threads = 0;
      spectra8::CompressParams gu;
      gu.level = spectra8::kUltraLevel;
      gu.threads = 0;
      gu.gpu = true;

      double t0 = now_s();
      auto ru = spectra8::compress(ds.data, comp, cu);
      double tc = now_s() - t0;
      size_t ucs = ru ? ru.value : bytes;

      t0 = now_s();
      auto rgu = spectra8::compress(ds.data, comp, gu);
      double tg = now_s() - t0;
      size_t gucs = rgu ? rgu.value : bytes;
      // Confirm the GPU-ultra frame decodes on the standard decoder.
      {
        std::span<const std::byte> gs(comp.data(), gucs);
        auto vr = spectra8::decompress(gs, dec);
        if (!vr || vr.value != bytes ||
            std::memcmp(dec.data(), ds.data.data(), bytes) != 0) {
          std::fprintf(stderr, "gpu-ultra verify failed (%s)\n", ds.name.c_str());
          return 1;
        }
      }

      std::printf(" %-8s %10.2f %10.2f %9.3f %9.3f\n", ds.name.c_str(), tc, tg,
                  static_cast<double>(bytes) / static_cast<double>(ucs),
                  static_cast<double>(bytes) / static_cast<double>(gucs));
    }
    hr('-');
    std::printf(" note: same decoder; GPU-ultra falls back to CPU-ultra if the"
                " top-K kernel is unavailable.\n");
  }

  if (filter) {
    std::printf("\n");
    hr('=');
    std::printf(" byte filters: --filter vs none   level 9   ratio + compress"
                " throughput\n");
    std::printf("   the encoder auto-detects a reversible byte transform "
                "(shuffle/delta/zigzag) per block\n");
    hr('=');
    std::printf(" %-8s %9s %9s %9s %7s %10s %10s\n", "dataset", "detected",
                "ratio off", "ratio on", "gain", "cMB/s off", "cMB/s on");
    hr('-');

    const double fdeadline = now_s() + std::max(3.0, budget * 0.25);
    const int fcells = static_cast<int>(datasets.size());
    int fdone = 0;
    for (const Dataset& ds : datasets) {
      double remaining = fdeadline - now_s();
      double per = remaining / ((fcells - fdone) > 0 ? (fcells - fdone) : 1);
      double min_each = per / 2.0;
      if (min_each < 0.02) min_each = 0.02;
      if (min_each > 1.0) min_each = 1.0;

      spectra8::CompressParams off;
      off.codec = codec;
      off.level = 9;
      off.threads = bench_codec_external(codec) ? 1 : 0;
      spectra8::CompressParams on = off;
      on.filters = true;
      on.filter_sample_full = filter_sample_full;

      auto roff = spectra8::compress(ds.data, comp, off);
      size_t cs_off = roff ? roff.value : bytes;
      auto ron = spectra8::compress(ds.data, comp, on);
      size_t cs_on = ron ? ron.value : bytes;
      // Confirm the filtered frame decodes on the standard decoder.
      {
        std::span<const std::byte> fs(comp.data(), cs_on);
        auto vr = spectra8::decompress(fs, dec);
        if (!vr || vr.value != bytes ||
            std::memcmp(dec.data(), ds.data.data(), bytes) != 0) {
          std::fprintf(stderr, "filter verify failed (%s)\n", ds.name.c_str());
          return 1;
        }
      }

      double t_off = throughput(
          [&] { (void)spectra8::compress(ds.data, comp, off); }, bytes,
          min_each, fdeadline);
      double t_on = throughput(
          [&] { (void)spectra8::compress(ds.data, comp, on); }, bytes, min_each,
          fdeadline);

      const auto* u8data =
          reinterpret_cast<const spectra8::detail::u8*>(ds.data.data());
      spectra8::detail::Filter det =
          spectra8::detail::detect_filter(u8data, bytes, filter_sample_full);
      const double r_off = static_cast<double>(bytes) / static_cast<double>(cs_off);
      const double r_on = static_cast<double>(bytes) / static_cast<double>(cs_on);
      const double gain = (r_off > 0) ? (r_on / r_off - 1.0) * 100.0 : 0.0;

      std::printf(" %-8s %9s %9.3f %9.3f %6.1f%% %10.1f %10.1f\n",
                  ds.name.c_str(), filter_name(det), r_off, r_on, gain, t_off,
                  t_on);
      ++fdone;
    }
    hr('-');
    std::printf(" note: each block also compresses unfiltered and keeps the "
                "smaller, so a filter never bloats a block;\n");
    std::printf("       tiny negative gains are the 1-byte/block filter tag. The"
                " 2nd compress is why cMB/s on is lower.\n");
    std::printf("       decode is the standard path (filter inverted "
                "transparently).\n");
  }

  std::printf(" total wall time: %.2f s\n", now_s() - start);
  return 0;
}

}  // namespace spectra8_cli
