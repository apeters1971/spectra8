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

// spectra8 command-line tool.
//
//   spectra8 [options] [input [output]]
//   spectra8 bench [seconds] [size] [options]
//   spectra8 tail [tail-options] [file ...]
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <sys/stat.h>

#include "bench.hpp"
#include "log.hpp"
#include "spectra8/spectra8.hpp"
#include "util/transform.hpp"  // detect_exec_arch, for the executable default

namespace {

struct Args {
  bool compress = true;
  bool to_stdout = false;
  bool force = false;
  bool quiet = false;
  int level = spectra8::kDefaultLevel;
  int threads = 1;       // <= 0 => auto
  int block_log = 0;     // 0 => default
  bool seek_table = false;
  bool block_crc = false;
  bool content_crc = false;
  bool filters = false;
  bool filter_fallback = true;
  bool filter_sample_full = false;
  bool gpu = false;
  bool context_oracle = false;
  spectra8::Codec codec = spectra8::Codec::Beam;
  int auto_min_gain_pct = 1;  // Codec::Auto only
  bool auto_max_level = false;
  spectra8::CodecMask auto_codecs = spectra8::kAutoCodecsAll;
  spectra8::FilterMask filter_mask = spectra8::kFiltersAll;
  bool logfile = false;
  bool codec_set = false;      // user passed --codec / a codec alias
  bool block_log_set = false;  // user passed --block-log
  bool level_set = false;      // user passed --level / -0..-9 / --ultra
  bool filters_set = false;    // user passed --filter
  std::string input;
  std::string output;
};

// A whole-file text input is worth the BWT path. Sample the start and, when
// the file is long enough, the middle, so a text header on binary data does
// not qualify. NUL bytes, other controls, and invalid UTF-8 reject the sample.
bool looks_like_text_sample(const unsigned char* p, std::size_t n) {
  if (n < 256) return false;
  std::size_t nul = 0, bad = 0, ws = 0, ok = 0;
  for (std::size_t i = 0; i < n;) {
    const unsigned char c = p[i];
    if (c == 0) {
      ++nul;
      ++i;
      continue;
    }
    if (c == '\t' || c == '\n' || c == '\r' || c == ' ') {
      ++ws;
      ++ok;
      ++i;
      continue;
    }
    if (c < 32 || c == 127) {
      ++bad;
      ++i;
      continue;
    }
    if (c < 127) {
      ++ok;
      ++i;
      continue;
    }
    int need = 0;
    if ((c & 0xE0) == 0xC0 && c >= 0xC2) need = 1;
    else if ((c & 0xF0) == 0xE0) need = 2;
    else if ((c & 0xF8) == 0xF0 && c <= 0xF4) need = 3;
    else {
      ++bad;
      ++i;
      continue;
    }
    if (i + static_cast<std::size_t>(need) >= n) {
      ++bad;
      break;
    }
    bool good = true;
    for (int k = 1; k <= need; ++k)
      if ((p[i + static_cast<std::size_t>(k)] & 0xC0) != 0x80) good = false;
    if (!good) {
      ++bad;
      ++i;
      continue;
    }
    ok += static_cast<std::size_t>(need) + 1;
    i += static_cast<std::size_t>(need) + 1;
  }
  if (nul * 1000 > n) return false;
  if (bad * 100 > n) return false;
  if (ws * 100 < n) return false;
  if (ok * 10 < n * 9) return false;
  return true;
}

bool looks_like_text_file(const std::vector<std::byte>& in) {
  constexpr std::size_t kMin = 4096;
  constexpr std::size_t kSample = 64 * 1024;
  if (in.size() < kMin) return false;
  const auto* p = reinterpret_cast<const unsigned char*>(in.data());
  const std::size_t n0 = std::min(in.size(), kSample);
  if (!looks_like_text_sample(p, n0)) return false;
  if (in.size() <= kSample * 2) return true;
  const std::size_t mid = in.size() / 2;
  const std::size_t n1 = std::min(kSample, in.size() - mid);
  return looks_like_text_sample(p + mid, n1);
}

// ELF / Mach-O / PE header, or an x86 / AArch64 instruction stream.
bool looks_like_executable_file(const std::vector<std::byte>& in) {
  return spectra8::detail::detect_exec_arch(
             reinterpret_cast<const unsigned char*>(in.data()), in.size()) !=
         spectra8::detail::ExecArch::None;
}

void usage(FILE* f) {
  std::fprintf(
      f,
      "spectra8 %s - fast LZ + entropy compressor\n"
      "\n"
      "Usage:\n"
      "  spectra8 [options] [input [output]]\n"
      "  spectra8 bench [seconds] [size] [options]\n"
      "  spectra8 tail [options] [file ...]\n"
      "\n"
      "Mode:\n"
      "  -z, --compress          compress (default)\n"
      "  -d, --decompress        decompress\n"
      "  -h, --help              show this help\n"
      "\n"
      "Level:\n"
      "  -<0-9>, --level=N       compression level 0-11 (default %d; 0 = Flash L1)\n"
      "                          L10: near-optimal CPU parser\n"
      "  --ultra                 level 11 (Beam): cost-based optimal parser\n"
      "  --flash                 alias for --codec flash (L1 = 4 GB/s SIMD decode;\n"
      "                          L5 = optional Huffman / compact extras / rep offset)\n"
      "\n"
      "I/O:\n"
      "  -c, --stdout            write to stdout\n"
      "  -f, --force             overwrite existing output\n"
      "  -q, --quiet             suppress the stats line\n"
      "  -T<n>, --threads=N      worker threads (0 = auto; default 1)\n"
      "  input / output          omit, '-', /dev/stdin, /dev/stdout for pipes\n"
      "                          (no files => stdin to stdout)\n"
      "\n"
      "Frame:\n"
      "  --block-log=N           block size as log2 bytes (14-28, 16 KiB .. 256 MiB)\n"
      "                          default: 18 at L1-2, 20 at L3-8, 21 at L9,\n"
      "                          22 at L10-11\n"
      "  --seek-table            emit a seek table (random access)\n"
      "  --block-crc             store a per-block checksum\n"
      "  --content-crc           emit + verify a whole-frame checksum\n"
      "                          (off by default; skip to keep decode memcpy-fast)\n"
      "  --logfile               one Spectra8 frame per input line, stored as one\n"
      "                          base64 line each (appendable; default suffix .l4l)\n"
      "\n"
      "Filters:\n"
      "  --filter                auto-detect + apply byte transforms\n"
      "                          (shuffle/delta/zigzag) for numeric/columnar data\n"
      "                          (Spectra8 Beam/Prism; L4EF for externals/Auto)\n"
      "  --filter-full-sample    score detect over the whole block\n"
      "                          (default: 16 KiB mid-sample)\n"
      "  --filter-mask=LIST      transforms detection may pick: comma list\n"
      "                          of names or ids (shuffle32, dict64, 15, ...),\n"
      "                          or `all` (default), `16`, `32`, `64` for\n"
      "                          one element width\n"
      "  --no-filter-fallback    encode only the detected filter path\n"
      "                          (skip the raw compress trial; faster)\n"
      "\n"
      "Codec:\n"
      "  --codec NAME            compression backend (default: beam;\n"
      "                          text files use prism, executables lzma,\n"
      "                          see below)\n"
      "                            beam      LZ77 (default; alias: lz)\n"
      "                            prism     BWT (alias: bwt)\n"
      "                            mosaic    tiled lattice / mini-BWT race\n"
      "                            oracle    learned-context demo (not in auto)\n"
      "                            crystal   static lattice / BWT for text; best\n"
      "                                      ratio, slow encode, fast decode;\n"
      "                                      L10-11: context mixing\n"
      "                            flash     ultra-fast byte-aligned LZ (L1: 4-6 GB/s\n"
      "                                      SIMD decode; L5: ratio tools when they\n"
      "                                      pay; NVMe line-rate ingestion)\n"
      "                            laser     LZMA2-class 5-stream decoupled range\n"
      "                                      coder for executables/binaries\n"
      "                            grating   order-0 integer rANS (no levels;\n"
      "                                      4/8-way decode)\n"
      "                            zstd      native frame (needs dep linked)\n"
      "                            bzip3     native frame (needs dep linked)\n"
      "                            lzma      native frame (needs dep linked)\n"
      "                            auto      race zstd/beam/mosaic/prism/laser/\n"
      "                                      lzma/bzip3/crystal at -N (or each max\n"
      "                                      with --auto-max-level); keep a\n"
      "                                      decode-speed-aware winner\n"
      "  --prism, --bwt, --mosaic, --oracle, --crystal, --flash, --laser,\n"
      "  --grating\n"
      "                          aliases for --codec NAME\n"
      "  Text files (UTF-8 or ASCII, whole-file compress) select Prism,\n"
      "  the BWT coder, at level 11 with 128 MiB blocks and word filter.\n"
      "  Executables (ELF / Mach-O / PE, or x86 / AArch64 code) select lzma\n"
      "  level 9 with the matching BCJ filter.\n"
      "  --codec, --level, or --block-log keeps the choice you passed.\n"
      "  --gpu                   Metal GPU match finder if available (Beam only)\n"
      "  --context-oracle        enable PAQ-style context cost modeling in Beam Ultra\n"
      "\n"
      "Auto (--codec auto):\n"
      "  --auto-gain=N           percent size gain a slower decoder needs per\n"
      "                          doubling of decode time (default 1;\n"
      "                          0 = always smallest)\n"
      "                          preference: zstd < beam < mosaic < prism <\n"
      "                                      lzma < bzip3 < crystal\n"
      "  --auto-max-level        race each codec at max effort, not -N\n"
      "                          (Beam/Prism/Mosaic/Crystal 11, Zstd max,\n"
      "                           Lzma 9, Bzip3 max block)\n"
      "  --auto-codecs=LIST      codecs to race: comma list of names, `all`\n"
      "                          (default) or `root` (= zstd,beam,crystal;\n"
      "                          the fast-decode set recommended for RNTuple)\n"
      "\n"
      "Bench:\n"
      "  Defaults: 60s, 128 KiB; size suffix k/m/g (128k, 4m; bare = MiB).\n"
      "  Default codec: crystal.\n"
      "\n"
      "  General:\n"
      "    --codec NAME            beam|prism|mosaic|oracle|crystal|grating|\n"
      "                            zstd|bzip3|lzma|auto; also --beam,\n"
      "                            --prism, --mosaic, --oracle, --crystal, --grating\n"
      "    --gpu                   Metal GPU comparison (Beam)\n"
      "    --compare               level-matched zstd matrix\n"
      "    --no-ultra              skip levels 10/11 (faster)\n"
      "    --filter                byte-filter on/off table\n"
      "    --bwt-chains            sweep BWT LF chains 4/8/12/16; print\n"
      "                            the best for this CPU\n"
      "    --h2d                   CPU unzip + pinned H2D page-pump timing\n"
      "    --explain               print codec / algorithm table and exit\n"
      "\n"
      "  Verdict:\n"
      "    --verdict               compact Beam L1/L5/L9/ultra vs zstd /\n"
      "                            Prism\n"
      "    --brute                 with --verdict: also compress every filter\n"
      "                            and print brute-force / entropy-detected\n"
      "    --full                  every level of every codec\n"
      "                            (Beam/Prism/Mosaic/Oracle/Crystal L1-L11,\n"
      "                             zstd L1-Lmax, lzma P0-P9+P9e)\n"
      "    --dir DIR               verdict table over files in DIR\n"
      "    --auto-gain N           with --dir: Auto decode-speed trade-off\n"
      "    --auto-codecs LIST      with --dir: add an Auto row restricted to\n"
      "                            LIST (names, `root` = zstd,beam,crystal)\n"
      "    --only LIST             with --dir: print only the rows of these\n"
      "                            codecs (names incl. auto/oracle, `all`,\n"
      "                            `root` = zstd,beam,crystal,auto)\n"
      "    --level N, -N           with --dir: one row per codec, all at\n"
      "                            level N (default: L1/L5/L9... matrix)\n"
      "    --parallel N            with --dir: process N files at a time\n"
      "                            (0 = all cores; default 1); speeds are\n"
      "                            aggregate wall-clock throughput\n"
      "    --stats                 with --dir: under every row print the\n"
      "                            filters detected / kept, wire format and\n"
      "                            Auto codec mix (Auto rows always show the mix)\n"
      "    --auto-max-level        race Auto at each codec's max (not L5)\n"
      "    --no-filter             skip the external byte-filter stage\n"
      "    --rle                   also try post-filter byte-RLE; keep best\n"
      "                            of filter / filter+RLE / raw by size\n"
      "    --no-rle                do not try post-filter RLE (default)\n"
      "    --no-filter-fallback    skip the raw compress trial (faster)\n"
      "    --no-dict               exclude dict32/64 (byte transforms only)\n"
      "    --filter-full-sample    score detect over the whole block\n"
      "\n"
      "  Synthetic data / text corpus:\n"
      "    --classic16             benchmark only the previous 16 classic datasets\n"
      "    --text, --enwik8        benchmark on the enwik8 Wikipedia text dataset\n"
      "                            (default: 8 MiB)\n"
      "    --binaries              shared-library slices (default: 1 MiB;\n"
      "                            SPECTRA8_BINARIES_DIR or homebrew/system lib)\n"
      "    --rawint                peaked integer columns (laplace, zipf,\n"
      "                            long tail) for the Grating coder\n"
      "                            (--no-filter keeps the scalars intact)\n"
      "    --complete              benchmark the complete dataset (all 100 MB of enwik8)\n"
      "    --dist D                binomial|gaussian|poisson|classic16|all|text|\n"
      "                            binaries|rawint\n"
      "                            (default: all with --verdict, else binomial)\n"
      "    --seed N                base RNG seed (decimal or 0xhex;\n"
      "                            default 0xABCDEF). Each named profile uses\n"
      "                            seed XOR strlen(name)\n"
      "\n"
      "Tail (logfile format):\n"
      "  spectra8 tail [options] [file ...]\n"
      "  Decodes a --logfile stream. Usual tail flags: -n/--lines=[+]N,\n"
      "  -c/--bytes=[+]N, -f/--follow[=descriptor|name], -F, -q, -v,\n"
      "  -s/--sleep-interval=S, --pid=PID, --retry. Default: last 10 lines.\n"
      "  No file, '-', or /dev/stdin reads a pipe.\n",
      std::string(spectra8::version()).c_str(), spectra8::kDefaultLevel);
}

bool write_file(const std::string& path, const std::byte* data, size_t n) {
  FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) return false;
  size_t put = std::fwrite(data, 1, n, f);
  std::fclose(f);
  return put == n;
}

// `--auto-codecs=LIST`: comma-separated codec names, or `all` / `root`
// (zstd,beam,crystal). Returns false on an unknown name.
bool parse_codec_mask(const char* list, spectra8::CodecMask& out);

// `--filter-mask=LIST`: comma-separated filter names / numeric ids, or
// `all` / `16` / `32` / `64` (element-width subsets). Returns false on an
// unknown name.
bool parse_filter_mask(const char* list, spectra8::FilterMask& out);

bool parse_codec_name(const char* name, spectra8::Codec& out) {
  if (std::strcmp(name, "beam") == 0 || std::strcmp(name, "lz") == 0) {
    out = spectra8::Codec::Beam;
    return true;
  }
  if (std::strcmp(name, "prism") == 0 || std::strcmp(name, "bwt") == 0) {
    out = spectra8::Codec::Prism;
    return true;
  }
  if (std::strcmp(name, "mosaic") == 0) {
    out = spectra8::Codec::Mosaic;
    return true;
  }
  if (std::strcmp(name, "oracle") == 0) {
    out = spectra8::Codec::Oracle;
    return true;
  }
  if (std::strcmp(name, "crystal") == 0) {
    out = spectra8::Codec::Crystal;
    return true;
  }
  if (std::strcmp(name, "flash") == 0 || std::strcmp(name, "lz4") == 0) {
    out = spectra8::Codec::Flash;
    return true;
  }
  if (std::strcmp(name, "laser") == 0) {
    out = spectra8::Codec::Laser;
    return true;
  }
  if (std::strcmp(name, "grating") == 0) {
    out = spectra8::Codec::Grating;
    return true;
  }
  if (std::strcmp(name, "zstd") == 0) {
    out = spectra8::Codec::Zstd;
    return true;
  }
  if (std::strcmp(name, "bzip3") == 0) {
    out = spectra8::Codec::Bzip3;
    return true;
  }
  if (std::strcmp(name, "lzma") == 0 || std::strcmp(name, "xz") == 0) {
    out = spectra8::Codec::Lzma;
    return true;
  }
  if (std::strcmp(name, "auto") == 0) {
    out = spectra8::Codec::Auto;
    return true;
  }
  return false;
}

bool parse_codec_mask(const char* list, spectra8::CodecMask& out) {
  std::string l(list);
  if (l == "all") { out = spectra8::kAutoCodecsAll; return true; }
  if (l == "root") { out = spectra8::kAutoCodecsRoot; return true; }
  spectra8::CodecMask m = 0;
  size_t pos = 0;
  while (pos <= l.size()) {
    size_t e = l.find(',', pos);
    if (e == std::string::npos) e = l.size();
    std::string tok = l.substr(pos, e - pos);
    if (!tok.empty()) {
      spectra8::Codec c;
      if (!parse_codec_name(tok.c_str(), c) || c == spectra8::Codec::Auto ||
          c == spectra8::Codec::Oracle) {
        std::fprintf(stderr, "spectra8: --auto-codecs: unknown or non-raceable codec '%s'\n",
                     tok.c_str());
        return false;
      }
      m |= spectra8::codec_bit(c);
    }
    pos = e + 1;
  }
  if (m == 0) return false;
  out = m;
  return true;
}

// `bench --dir --only LIST`: comma-separated codec names (any codec incl.
// `auto`, `oracle`), or `all` / `root` (zstd,beam,crystal,auto). The result is
// a row filter, so unlike --auto-codecs it may name Auto and Oracle.
bool parse_only_mask(const char* list, spectra8::CodecMask& out) {
  std::string l(list);
  if (l == "all") { out = ~spectra8::CodecMask{0}; return true; }
  if (l == "root") {
    out = spectra8::kAutoCodecsRoot | spectra8::codec_bit(spectra8::Codec::Auto);
    return true;
  }
  spectra8::CodecMask m = 0;
  size_t pos = 0;
  while (pos <= l.size()) {
    size_t e = l.find(',', pos);
    if (e == std::string::npos) e = l.size();
    std::string tok = l.substr(pos, e - pos);
    if (!tok.empty()) {
      spectra8::Codec c;
      if (!parse_codec_name(tok.c_str(), c)) {
        std::fprintf(stderr, "spectra8 bench: --only: unknown codec '%s'\n", tok.c_str());
        return false;
      }
      m |= spectra8::codec_bit(c);
    }
    pos = e + 1;
  }
  if (m == 0) return false;
  out = m;
  return true;
}

bool parse_filter_mask(const char* list, spectra8::FilterMask& out) {
  static const struct { const char* name; spectra8::FilterId id; } kNames[] = {
      {"none", spectra8::FilterId::None},
      {"shuffle16", spectra8::FilterId::Shuffle16},
      {"shuffle32", spectra8::FilterId::Shuffle32},
      {"shuffle64", spectra8::FilterId::Shuffle64},
      {"delta16", spectra8::FilterId::Delta16},
      {"delta32", spectra8::FilterId::Delta32},
      {"delta64", spectra8::FilterId::Delta64},
      {"zigzag_delta16", spectra8::FilterId::ZigzagDelta16},
      {"zigzag_delta32", spectra8::FilterId::ZigzagDelta32},
      {"zigzag_delta64", spectra8::FilterId::ZigzagDelta64},
      {"delta32+shuffle32", spectra8::FilterId::DeltaShuffle32},
      {"zigzag_delta32+shuffle32", spectra8::FilterId::ZigzagDeltaShuffle32},
      {"bitshuffle32", spectra8::FilterId::BitShuffle32},
      {"bitshuffle64", spectra8::FilterId::BitShuffle64},
      {"delta32+bitshuffle32", spectra8::FilterId::DeltaBitShuffle32},
      {"dict16", spectra8::FilterId::Dict16},
      {"dict32", spectra8::FilterId::Dict32},
      {"dict64", spectra8::FilterId::Dict64},
      {"for16", spectra8::FilterId::For16},
      {"for32", spectra8::FilterId::For32},
      {"for64", spectra8::FilterId::For64},
      {"alp32", spectra8::FilterId::Alp32},
      {"alp64", spectra8::FilterId::Alp64},
      {"delta64+shuffle64", spectra8::FilterId::DeltaShuffle64},
      {"zigzag32+shuffle32", spectra8::FilterId::ZigzagShuffle32},
      {"zigzag64+shuffle64", spectra8::FilterId::ZigzagShuffle64},
      {"xor_delta32+shuffle32", spectra8::FilterId::XorShuffle32},
      {"xorshuf32", spectra8::FilterId::XorShuffle32},
      {"xor_shuffle32", spectra8::FilterId::XorShuffle32},
      {"xor_delta64+shuffle64", spectra8::FilterId::XorShuffle64},
      {"xorshuf64", spectra8::FilterId::XorShuffle64},
      {"xor_shuffle64", spectra8::FilterId::XorShuffle64},
      {"delta2_32+shuffle32", spectra8::FilterId::Delta2Shuffle32},
      {"d2shuf32", spectra8::FilterId::Delta2Shuffle32},
      {"delta2_shuffle32", spectra8::FilterId::Delta2Shuffle32},
      {"zigzag_delta2_32+shuffle32", spectra8::FilterId::ZigzagDelta2Shuffle32},
      {"zigzag_delta2_shuffle32", spectra8::FilterId::ZigzagDelta2Shuffle32},
      {"zzd2shuf32", spectra8::FilterId::ZigzagDelta2Shuffle32},
      {"delta2_64+shuffle64", spectra8::FilterId::Delta2Shuffle64},
      {"d2shuf64", spectra8::FilterId::Delta2Shuffle64},
      {"delta2_shuffle64", spectra8::FilterId::Delta2Shuffle64},
      {"bcj_x86", spectra8::FilterId::BcjX86},
      {"bcj", spectra8::FilterId::BcjX86},
      {"bcj_arm64", spectra8::FilterId::BcjArm64},
      {"digit_norm", spectra8::FilterId::DigitNorm},
      {"digit", spectra8::FilterId::DigitNorm},
      {"text_digit", spectra8::FilterId::DigitNorm},
      {"word_norm", spectra8::FilterId::WordNorm},
      {"word", spectra8::FilterId::WordNorm},
      {"text_word", spectra8::FilterId::WordNorm},
      {"base64", spectra8::FilterId::Base64},
      {"b64", spectra8::FilterId::Base64},
      {"bitround32", spectra8::FilterId::BitRound32},
      {"bitround64", spectra8::FilterId::BitRound64},
      {"bitround", spectra8::FilterId::BitRound32},
      {"lag2shuf32", spectra8::FilterId::Lag2Shuffle32},
      {"lag2_32+shuffle32", spectra8::FilterId::Lag2Shuffle32},
      {"lag2shuf64", spectra8::FilterId::Lag2Shuffle64},
      {"lag2_64+shuffle64", spectra8::FilterId::Lag2Shuffle64},
      {"xorlag2shuf32", spectra8::FilterId::XorLag2Shuffle32},
      {"xor_lag2_32+shuffle32", spectra8::FilterId::XorLag2Shuffle32},
  };
  std::string l(list);
  if (l == "all") { out = spectra8::kFiltersAll; return true; }
  if (l == "16") { out = spectra8::kFilters16; return true; }
  if (l == "32") { out = spectra8::kFilters32; return true; }
  if (l == "64") { out = spectra8::kFilters64; return true; }
  spectra8::FilterMask m = 0;
  size_t pos = 0;
  while (pos <= l.size()) {
    size_t e = l.find(',', pos);
    if (e == std::string::npos) e = l.size();
    std::string tok = l.substr(pos, e - pos);
    if (!tok.empty()) {
      bool found = false;
      for (const auto& n : kNames)
        if (tok == n.name) { m |= spectra8::filter_bit(n.id); found = true; break; }
      if (!found) {
        char* end = nullptr;
        long id = std::strtol(tok.c_str(), &end, 10);
        if (end && *end == '\0' && id >= 0 && id <= 31) {
          m |= spectra8::FilterMask{1} << static_cast<unsigned>(id);
          found = true;
        }
      }
      if (!found) {
        std::fprintf(stderr, "spectra8: --filter-mask: unknown filter '%s'\n", tok.c_str());
        return false;
      }
    }
    pos = e + 1;
  }
  if (m == 0) return false;
  out = m;
  return true;
}

int parse_args(int argc, char** argv, Args& a) {
  std::vector<std::string> files;
  for (int i = 1; i < argc; ++i) {
    std::string s = argv[i];
    if (s.size() >= 2 && s[0] == '-' && s[1] == '-') {
      if (s == "--compress") a.compress = true;
      else if (s == "--decompress") a.compress = false;
      else if (s == "--stdout") a.to_stdout = true;
      else if (s == "--force") a.force = true;
      else if (s == "--quiet") a.quiet = true;
      else if (s == "--help") return 2;
      else if (s == "--ultra") {
        a.level = spectra8::kUltraLevel;
        a.level_set = true;
      } else if (s.rfind("--level=", 0) == 0) {
        a.level = std::atoi(s.c_str() + 8);
        a.level_set = true;
      } else if (s.rfind("--threads=", 0) == 0) a.threads = std::atoi(s.c_str() + 10);
      else if (s.rfind("--block-log=", 0) == 0) {
        a.block_log = std::atoi(s.c_str() + 12);
        a.block_log_set = true;
      }
      else if (s == "--seek-table") a.seek_table = true;
      else if (s == "--block-crc") a.block_crc = true;
      else if (s == "--content-crc") a.content_crc = true;
      else if (s == "--filter") {
        a.filters = true;
        a.filters_set = true;
      }
      else if (s == "--filter-full-sample") a.filter_sample_full = true;
      else if (s == "--no-filter-fallback") a.filter_fallback = false;
      else if (s == "--gpu") a.gpu = true;
      else if (s == "--context-oracle") a.context_oracle = true;
      else if (s == "--beam" || s == "--lz") {
        a.codec = spectra8::Codec::Beam;
        a.codec_set = true;
      } else if (s == "--bwt" || s == "--prism") {
        a.codec = spectra8::Codec::Prism;
        a.codec_set = true;
      } else if (s == "--mosaic") {
        a.codec = spectra8::Codec::Mosaic;
        a.codec_set = true;
      } else if (s == "--oracle") {
        a.codec = spectra8::Codec::Oracle;
        a.codec_set = true;
      } else if (s == "--crystal") {
        a.codec = spectra8::Codec::Crystal;
        a.codec_set = true;
      } else if (s == "--flash") {
        a.codec = spectra8::Codec::Flash;
        a.codec_set = true;
      } else if (s == "--laser") {
        a.codec = spectra8::Codec::Laser;
        a.codec_set = true;
      } else if (s == "--grating") {
        a.codec = spectra8::Codec::Grating;
        a.codec_set = true;
      } else if (s == "--codec" && i + 1 < argc) {
        if (!parse_codec_name(argv[++i], a.codec)) return -1;
        a.codec_set = true;
      } else if (s.rfind("--codec=", 0) == 0) {
        if (!parse_codec_name(s.c_str() + 8, a.codec)) return -1;
        a.codec_set = true;
      } else if (s == "--auto-gain" && i + 1 < argc) {
        a.auto_min_gain_pct = std::atoi(argv[++i]);
      } else if (s.rfind("--auto-gain=", 0) == 0) {
        a.auto_min_gain_pct = std::atoi(s.c_str() + 12);
      } else if (s == "--auto-max-level") {
        a.auto_max_level = true;
      } else if (s.rfind("--auto-codecs=", 0) == 0) {
        if (!parse_codec_mask(s.c_str() + 14, a.auto_codecs)) return -1;
      } else if (s == "--auto-codecs" && i + 1 < argc) {
        if (!parse_codec_mask(argv[++i], a.auto_codecs)) return -1;
      } else if (s.rfind("--filter-mask=", 0) == 0) {
        if (!parse_filter_mask(s.c_str() + 14, a.filter_mask)) return -1;
      } else if (s == "--filter-mask" && i + 1 < argc) {
        if (!parse_filter_mask(argv[++i], a.filter_mask)) return -1;
      } else if (s == "--logfile") {
        a.logfile = true;
      } else
        return -1;
    } else if (s.size() > 1 && s[0] == '-' &&
               std::strspn(s.c_str() + 1, "0123456789") == s.size() - 1) {
      a.level = std::atoi(s.c_str() + 1);
      a.level_set = true;
    } else if (s.size() >= 1 && s[0] == '-' && s.size() > 1) {
      for (size_t j = 1; j < s.size(); ++j) {
        char c = s[j];
        if (c == 'z') a.compress = true;
        else if (c == 'd') a.compress = false;
        else if (c == 'c') a.to_stdout = true;
        else if (c == 'f') a.force = true;
        else if (c == 'q') a.quiet = true;
        else if (c == 'h') return 2;
        else if (c == 'T') { a.threads = std::atoi(s.c_str() + j + 1); break; }
        else if (c >= '0' && c <= '9') {
          a.level = c - '0';
          a.level_set = true;
        }
        else return -1;
      }
    } else {
      files.push_back(s);
    }
  }
  if (files.empty()) {
    // Require -c, --stdout, or --logfile so a bare `spectra8` still
    // prints usage. Explicit '-' / /dev/stdin is handled below.
    if (!a.to_stdout && !a.logfile) return -1;
    a.input = "-";
    a.to_stdout = true;
    return 0;
  }
  a.input = files[0];
  if (files.size() > 1) a.output = files[1];
  if (spectra8_cli::is_stdin_path(a.input) && a.output.empty())
    a.to_stdout = true;
  if (spectra8_cli::is_stdout_path(a.output)) {
    a.to_stdout = true;
    a.output.clear();
  }
  return 0;
}

// Parse a dataset size like "128k", "4m", "0.5m", or a bare number (taken as
// MiB for backward compatibility). Returns 0 on failure / empty.
size_t parse_size(const char* s) {
  char* end = nullptr;
  double v = std::strtod(s, &end);
  if (end == s || v < 0) return 0;
  double mult = 1024.0 * 1024.0;  // bare number => MiB (legacy default)
  switch (*end) {
    case 'k': case 'K': mult = 1024.0; break;
    case 'm': case 'M': mult = 1024.0 * 1024.0; break;
    case 'g': case 'G': mult = 1024.0 * 1024.0 * 1024.0; break;
    case 'b': case 'B': mult = 1.0; break;
    default: break;  // no/unknown suffix => MiB
  }
  return static_cast<size_t>(v * mult);
}

bool read_all(FILE* f, std::vector<std::byte>& out) {
  out.clear();
  unsigned char buf[1 << 16];
  for (;;) {
    const size_t n = std::fread(buf, 1, sizeof buf, f);
    if (n) {
      const auto* p = reinterpret_cast<const std::byte*>(buf);
      out.insert(out.end(), p, p + n);
    }
    if (n < sizeof buf) return !std::ferror(f);
  }
}

bool read_path(const std::string& path, std::vector<std::byte>& out) {
  FILE* f = spectra8_cli::is_stdin_path(path) ? stdin
                                               : std::fopen(path.c_str(), "rb");
  if (!f) return false;
  const bool ok = read_all(f, out);
  if (f != stdin) std::fclose(f);
  return ok;
}

bool output_is_special(const std::string& path) {
  if (spectra8_cli::is_stdout_path(path)) return true;
  struct stat sb;
  if (::stat(path.c_str(), &sb) != 0) return false;
  return S_ISCHR(sb.st_mode) || S_ISFIFO(sb.st_mode);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc >= 2 && std::strcmp(argv[1], "tail") == 0) {
    spectra8_cli::TailOptions opt;
    const int tr = spectra8_cli::parse_tail_args(argc, argv, opt);
    if (tr == 2) {
      spectra8_cli::print_tail_usage(stdout);
      return 0;
    }
    if (tr < 0) {
      spectra8_cli::print_tail_usage(stderr);
      return 1;
    }
    return spectra8_cli::run_tail(opt);
  }

  // Subcommand: `spectra8 bench [seconds] [size] [--codec NAME] [--gpu]`.
  // e.g. `spectra8 bench` runs 60 s on 128 KiB Crystal datasets;
  // `spectra8 bench --codec beam` measures Beam; `spectra8 bench 20 4m`
  // uses 4 MiB datasets; `spectra8 bench --gpu` also compares the Metal GPU
  // match finder (Beam).
  if (argc >= 2 && std::strcmp(argv[1], "bench") == 0) {
    double seconds = 0.0;  // 0 => default
    size_t bytes = 0;      // 0 => default
    bool gpu = false;
    bool compare = false;
    bool no_ultra = false;
    bool filter = false;
    spectra8::Codec codec = spectra8::Codec::Crystal;
    bool verdict = false;
    bool verdict_full = false;
    bool verdict_no_filter = false;
    bool verdict_no_rle = true;
    bool verdict_rle = false;
    bool verdict_no_filter_fallback = false;
    bool verdict_no_dict = false;
    bool filter_sample_full = false;
    bool verdict_auto_max_level = false;
    bool verdict_brute = false;
    bool bwt_chains = false;
    std::string bench_dir;
    int bench_auto_gain = 1;
    spectra8::CodecMask bench_auto_codecs = spectra8::kAutoCodecsAll;
    spectra8::CodecMask bench_only = ~spectra8::CodecMask{0};
    int bench_level = 0;  // --dir: 0 => default multi-level matrix
    unsigned bench_parallel = 1;  // --dir: files in flight; 0 => all cores
    bool bench_stats = false;     // --dir: filter / wire / auto-mix report per row
    bool bench_complete = false;  // --complete: benchmark full dataset (100 MB for enwik8)
    bool h2d = false;
    spectra8_cli::BenchDist dist = spectra8_cli::BenchDist::Binomial;
    bool dist_set = false;
    std::uint64_t seed = spectra8_cli::kDefaultBenchSeed;
    auto parse_seed = [&](const char* s) -> bool {
      char* end = nullptr;
      errno = 0;
      const unsigned long long v = std::strtoull(s, &end, 0);
      if (errno != 0 || end == s || *end != '\0') return false;
      seed = static_cast<std::uint64_t>(v);
      return true;
    };
    auto parse_dist = [&](const char* s) -> bool {
      if (std::strcmp(s, "binomial") == 0)
        dist = spectra8_cli::BenchDist::Binomial;
      else if (std::strcmp(s, "gaussian") == 0)
        dist = spectra8_cli::BenchDist::Gaussian;
      else if (std::strcmp(s, "poisson") == 0)
        dist = spectra8_cli::BenchDist::Poisson;
      else if (std::strcmp(s, "classic") == 0 ||
               std::strcmp(s, "classic16") == 0 ||
               std::strcmp(s, "legacy") == 0 ||
               std::strcmp(s, "legacy16") == 0)
        dist = spectra8_cli::BenchDist::Classic16;
      else if (std::strcmp(s, "all") == 0)
        dist = spectra8_cli::BenchDist::All;
      else if (std::strcmp(s, "text") == 0 ||
               std::strcmp(s, "enwik8") == 0)
        dist = spectra8_cli::BenchDist::Text;
      else if (std::strcmp(s, "binaries") == 0)
        dist = spectra8_cli::BenchDist::Binaries;
      else if (std::strcmp(s, "rawint") == 0)
        dist = spectra8_cli::BenchDist::RawInt;
      else
        return false;
      dist_set = true;
      return true;
    };
    int pos = 0;
    for (int i = 2; i < argc; ++i) {
      if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
        usage(stdout);
        return 0;
      } else if (std::strcmp(argv[i], "--gpu") == 0) {
        gpu = true;
      } else if (std::strcmp(argv[i], "--compare") == 0) {
        compare = true;
      } else if (std::strcmp(argv[i], "--no-ultra") == 0) {
        no_ultra = true;
      } else if (std::strcmp(argv[i], "--filter") == 0) {
        filter = true;
      } else if (std::strcmp(argv[i], "--bwt") == 0) {
        std::fprintf(stderr,
                     "spectra8 bench: --bwt was removed; use --codec prism\n");
        return 1;
      } else if (std::strcmp(argv[i], "--codec") == 0) {
        if (i + 1 >= argc) {
          std::fprintf(stderr, "spectra8 bench: --codec requires a name\n");
          return 1;
        }
        if (!parse_codec_name(argv[++i], codec)) {
          std::fprintf(stderr,
                       "spectra8 bench: unknown codec '%s' (expected "
                       "beam|prism|mosaic|oracle|crystal|flash|laser|grating|zstd|bzip3|lzma|auto)\n",
                       argv[i]);
          return 1;
        }
      } else if (std::strncmp(argv[i], "--codec=", 8) == 0) {
        if (!parse_codec_name(argv[i] + 8, codec)) {
          std::fprintf(stderr,
                       "spectra8 bench: unknown codec '%s' (expected "
                       "beam|prism|mosaic|oracle|crystal|flash|laser|grating|zstd|bzip3|lzma|auto)\n",
                       argv[i] + 8);
          return 1;
        }
      } else if (std::strcmp(argv[i], "--beam") == 0) {
        codec = spectra8::Codec::Beam;
      } else if (std::strcmp(argv[i], "--prism") == 0) {
        codec = spectra8::Codec::Prism;
      } else if (std::strcmp(argv[i], "--mosaic") == 0) {
        codec = spectra8::Codec::Mosaic;
      } else if (std::strcmp(argv[i], "--oracle") == 0) {
        codec = spectra8::Codec::Oracle;
      } else if (std::strcmp(argv[i], "--crystal") == 0) {
        codec = spectra8::Codec::Crystal;
      } else if (std::strcmp(argv[i], "--flash") == 0) {
        codec = spectra8::Codec::Flash;
      } else if (std::strcmp(argv[i], "--laser") == 0) {
        codec = spectra8::Codec::Laser;
      } else if (std::strcmp(argv[i], "--grating") == 0) {
        codec = spectra8::Codec::Grating;
      } else if (std::strcmp(argv[i], "--explain") == 0) {
        return spectra8_cli::print_codec_explain();
      } else if (std::strcmp(argv[i], "--verdict") == 0) {
        verdict = true;
      } else if (std::strcmp(argv[i], "--classic16") == 0 ||
                 std::strcmp(argv[i], "--classic") == 0 ||
                 std::strcmp(argv[i], "--legacy16") == 0 ||
                 std::strcmp(argv[i], "--legacy") == 0) {
        dist = spectra8_cli::BenchDist::Classic16;
        dist_set = true;
      } else if (std::strcmp(argv[i], "--text") == 0 ||
                 std::strcmp(argv[i], "--enwik8") == 0) {
        dist = spectra8_cli::BenchDist::Text;
        dist_set = true;
      } else if (std::strcmp(argv[i], "--binaries") == 0) {
        dist = spectra8_cli::BenchDist::Binaries;
        dist_set = true;
      } else if (std::strcmp(argv[i], "--rawint") == 0) {
        dist = spectra8_cli::BenchDist::RawInt;
        dist_set = true;
      } else if (std::strcmp(argv[i], "--complete") == 0 ||
                 std::strcmp(argv[i], "--full-dataset") == 0 ||
                 std::strcmp(argv[i], "--complete-dataset") == 0) {
        bench_complete = true;
        dist = spectra8_cli::BenchDist::Text;
        dist_set = true;
      } else if (std::strcmp(argv[i], "--full") == 0) {
        verdict_full = true;
      } else if (std::strcmp(argv[i], "--no-filter") == 0) {
        verdict_no_filter = true;
      } else if (std::strcmp(argv[i], "--rle") == 0) {
        verdict_rle = true;
        verdict_no_rle = false;
      } else if (std::strcmp(argv[i], "--no-rle") == 0) {
        verdict_no_rle = true;
      } else if (std::strcmp(argv[i], "--no-filter-fallback") == 0) {
        verdict_no_filter_fallback = true;
      } else if (std::strcmp(argv[i], "--no-dict") == 0) {
        verdict_no_dict = true;
      } else if (std::strcmp(argv[i], "--filter-full-sample") == 0) {
        filter_sample_full = true;
      } else if (std::strcmp(argv[i], "--auto-max-level") == 0) {
        verdict_auto_max_level = true;
      } else if (std::strcmp(argv[i], "--brute") == 0) {
        verdict_brute = true;
      } else if (std::strcmp(argv[i], "--bwt-chains") == 0) {
        bwt_chains = true;
      } else if (std::strcmp(argv[i], "--h2d") == 0) {
        h2d = true;
      } else if (std::strcmp(argv[i], "--dir") == 0 && i + 1 < argc) {
        bench_dir = argv[++i];
      } else if (std::strncmp(argv[i], "--dir=", 6) == 0) {
        bench_dir = argv[i] + 6;
      } else if (std::strcmp(argv[i], "--auto-gain") == 0 && i + 1 < argc) {
        bench_auto_gain = std::atoi(argv[++i]);
      } else if (std::strncmp(argv[i], "--auto-gain=", 12) == 0) {
        bench_auto_gain = std::atoi(argv[i] + 12);
      } else if (std::strcmp(argv[i], "--auto-codecs") == 0 && i + 1 < argc) {
        if (!parse_codec_mask(argv[++i], bench_auto_codecs)) return 1;
      } else if (std::strncmp(argv[i], "--auto-codecs=", 14) == 0) {
        if (!parse_codec_mask(argv[i] + 14, bench_auto_codecs)) return 1;
      } else if (std::strcmp(argv[i], "--only") == 0 && i + 1 < argc) {
        if (!parse_only_mask(argv[++i], bench_only)) return 1;
      } else if (std::strncmp(argv[i], "--only=", 7) == 0) {
        if (!parse_only_mask(argv[i] + 7, bench_only)) return 1;
      } else if (std::strcmp(argv[i], "--stats") == 0) {
        bench_stats = true;
      } else if ((std::strcmp(argv[i], "--parallel") == 0 && i + 1 < argc) ||
                 std::strncmp(argv[i], "--parallel=", 11) == 0) {
        const char* v = argv[i][10] == '=' ? argv[i] + 11 : argv[++i];
        char* end = nullptr;
        const long n = std::strtol(v, &end, 10);
        if (end == v || *end != '\0' || n < 0 || n > 1024) {
          std::fprintf(stderr, "spectra8 bench: --parallel expects 0 (all cores) or 1..1024\n");
          return 1;
        }
        bench_parallel = static_cast<unsigned>(n);
      } else if ((std::strcmp(argv[i], "--level") == 0 && i + 1 < argc) ||
                 std::strncmp(argv[i], "--level=", 8) == 0 ||
                 (argv[i][0] == '-' && argv[i][1] >= '0' && argv[i][1] <= '9' &&
                  std::strspn(argv[i] + 1, "0123456789") == std::strlen(argv[i] + 1))) {
        if (std::strcmp(argv[i], "--level") == 0) bench_level = std::atoi(argv[++i]);
        else if (std::strncmp(argv[i], "--level=", 8) == 0) bench_level = std::atoi(argv[i] + 8);
        else bench_level = std::atoi(argv[i] + 1);
        if (bench_level < 0 || bench_level > spectra8::kUltraLevel) {
          std::fprintf(stderr, "spectra8 bench: --level must be 0..%d\n",
                       spectra8::kUltraLevel);
          return 1;
        }
      } else if (std::strcmp(argv[i], "--dist") == 0 && i + 1 < argc) {
        if (!parse_dist(argv[++i])) {
          std::fprintf(stderr,
                       "spectra8 bench: unknown --dist '%s' (expected "
                       "binomial|gaussian|poisson|classic16|all|text|binaries|rawint)\n",
                       argv[i]);
          return 1;
        }
      } else if (std::strncmp(argv[i], "--dist=", 7) == 0) {
        if (!parse_dist(argv[i] + 7)) {
          std::fprintf(stderr,
                       "spectra8 bench: unknown --dist '%s' (expected "
                       "binomial|gaussian|poisson|classic16|all|text|binaries|rawint)\n",
                       argv[i] + 7);
          return 1;
        }
      } else if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
        if (!parse_seed(argv[++i])) {
          std::fprintf(stderr,
                       "spectra8 bench: invalid --seed '%s' (expected an "
                       "integer, optional 0x prefix)\n",
                       argv[i]);
          return 1;
        }
      } else if (std::strncmp(argv[i], "--seed=", 7) == 0) {
        if (!parse_seed(argv[i] + 7)) {
          std::fprintf(stderr,
                       "spectra8 bench: invalid --seed '%s' (expected an "
                       "integer, optional 0x prefix)\n",
                       argv[i] + 7);
          return 1;
        }
      } else if (pos == 0) {
        seconds = std::atof(argv[i]);
        ++pos;
      } else if (pos == 1) {
        bytes = parse_size(argv[i]);
        ++pos;
      }
    }
    // Verdict measures all three distributions by default; the plain bench
    // keeps its historical binomial-only set unless --dist is given.
    if (bench_only != ~spectra8::CodecMask{0} && bench_dir.empty()) {
      verdict = true;
    }
    if (!dist_set && verdict) dist = spectra8_cli::BenchDist::All;
    if (!bench_dir.empty())
      return spectra8_cli::run_dir_bench(bench_dir, verdict_no_filter, bench_auto_gain,
                                          bench_auto_codecs, bench_only, bench_level,
                                          bench_parallel, bench_stats);
    if (h2d)
      return spectra8_cli::run_h2d_bench(seconds, bytes, codec);
    if (!spectra8::codec_available(codec)) {
      std::fprintf(stderr,
                   "spectra8 bench: codec '%s' is not available in this build "
                   "(rebuild with the matching dependency)\n",
                   spectra8_cli::bench_codec_name(codec));
      return 1;
    }
    return spectra8_cli::run_benchmark(
        seconds, bytes, gpu, compare, no_ultra, filter, verdict,
        verdict_no_filter, verdict_no_rle, verdict_no_filter_fallback,
        bwt_chains, verdict_full, verdict_no_dict, filter_sample_full,
        verdict_auto_max_level, verdict_brute, dist, seed, codec,
        bench_complete, bench_only, bench_stats);
  }

  Args a;
  int pr = parse_args(argc, argv, a);
  if (pr == 2) {
    usage(stdout);
    return 0;
  }
  if (pr < 0) {
    usage(stderr);
    return 1;
  }

  const bool external = a.codec == spectra8::Codec::Zstd ||
                        a.codec == spectra8::Codec::Bzip3 ||
                        a.codec == spectra8::Codec::Lzma;
  if (a.compress) {
    if (!spectra8::codec_available(a.codec)) {
      std::fprintf(stderr,
                   "spectra8: codec not available in this build (rebuild with "
                   "the matching dependency linked)\n");
      return 1;
    }
    if (a.codec == spectra8::Codec::Auto &&
        (a.gpu || a.seek_table || a.block_crc || a.content_crc)) {
      std::fprintf(stderr,
                   "spectra8: --gpu/--seek-table/--block-crc/--content-crc "
                   "are not supported with --codec auto\n");
      return 1;
    }
    // --filter is fine with the external codecs: the library wraps their
    // native payload in an L4EF frame (byte filter + codec), which `-d`
    // reads back like any other frame.
    if (external &&
        (a.gpu || a.seek_table || a.block_crc || a.content_crc ||
         a.threads != 1 ||
         ((a.codec == spectra8::Codec::Zstd ||
           a.codec == spectra8::Codec::Lzma) &&
          a.block_log != 0))) {
      std::fprintf(stderr,
                   "spectra8: --gpu/--seek-table/--block-crc/"
                   "--content-crc/--threads are Spectra8-only; for zstd/lzma, "
                   "--block-log is also rejected (bzip3 uses it as block size)\n");
      return 1;
    }
  }

  // Default output name when not writing to stdout.
  if (a.output.empty() && !a.to_stdout) {
    if (spectra8_cli::is_stdin_path(a.input)) {
      a.to_stdout = true;
    } else if (a.compress) {
      if (a.logfile)
        a.output = a.input + ".l4l";
      else if (a.codec == spectra8::Codec::Zstd)
        a.output = a.input + ".zst";
      else if (a.codec == spectra8::Codec::Bzip3)
        a.output = a.input + ".bz3";
      else if (a.codec == spectra8::Codec::Lzma)
        a.output = a.input + ".xz";
      else if (a.codec == spectra8::Codec::Auto)
        a.output = a.input + ".auto";
      else
        a.output = a.input + ".lhc4";
    } else {
      auto strip = [&](const char* ext) -> bool {
        const size_t n = std::strlen(ext);
        if (a.input.size() > n &&
            a.input.substr(a.input.size() - n) == ext) {
          a.output = a.input.substr(0, a.input.size() - n);
          return true;
        }
        return false;
      };
      if (!strip(".l4l") && !strip(".lhc4") && !strip(".zst") &&
          !strip(".bz3") && !strip(".xz") && !strip(".auto"))
        a.output = a.input + ".out";
    }
  }

  if (!a.to_stdout && !a.force && !output_is_special(a.output)) {
    struct stat sb;
    if (::stat(a.output.c_str(), &sb) == 0) {
      std::fprintf(stderr, "spectra8: '%s' already exists (use -f)\n",
                   a.output.c_str());
      return 1;
    }
  }

  if (a.logfile) {
    FILE* in = spectra8_cli::is_stdin_path(a.input)
                   ? stdin
                   : std::fopen(a.input.c_str(), "rb");
    if (!in) {
      std::fprintf(stderr, "spectra8: cannot read '%s'\n", a.input.c_str());
      return 1;
    }
    FILE* out = nullptr;
    bool own_out = false;
    if (a.to_stdout) {
      out = stdout;
    } else {
      out = std::fopen(a.output.c_str(), "wb");
      own_out = true;
    }
    if (!out) {
      if (in != stdin) std::fclose(in);
      std::fprintf(stderr, "spectra8: cannot write '%s'\n", a.output.c_str());
      return 1;
    }
    std::uint64_t lines = 0, ib = 0, ob = 0;
    int rc = 0;
    if (a.compress) {
      spectra8::CompressParams cp;
      cp.codec = a.codec;
      cp.level = a.level;
      cp.threads = a.threads;
      cp.block_log = a.block_log;
      cp.seek_table = a.seek_table;
      cp.block_checksums = a.block_crc;
      cp.content_checksum = a.content_crc;
      cp.filters = a.filters;
      cp.filter_fallback = a.filter_fallback;
      cp.filter_sample_full = a.filter_sample_full;
      cp.auto_min_gain_pct = a.auto_min_gain_pct;
      cp.auto_max_level = a.auto_max_level;
      cp.auto_codecs = a.auto_codecs;
      cp.filter_mask = a.filter_mask;
      cp.gpu = a.gpu;
      cp.context_oracle = a.context_oracle;
      rc = spectra8_cli::encode_logfile(in, out, cp, &lines, &ib, &ob);
    } else {
      rc = spectra8_cli::decode_logfile(in, out, &lines, &ib, &ob);
    }
    if (in != stdin) std::fclose(in);
    if (own_out) std::fclose(out);
    if (rc != 0) return rc;
    if (!a.quiet) {
      const double ratio = (a.compress ? (ob ? (double)ib / ob : 0.0)
                                       : (ib ? (double)ob / ib : 0.0));
      std::fprintf(stderr, "%-24s %10llu -> %10llu  (%.2fx, %llu lines)\n",
                   a.input.c_str(), (unsigned long long)ib,
                   (unsigned long long)ob, ratio, (unsigned long long)lines);
    }
    return 0;
  }

  std::vector<std::byte> input;
  if (!read_path(a.input, input)) {
    std::fprintf(stderr, "spectra8: cannot read '%s'\n", a.input.c_str());
    return 1;
  }

  std::vector<std::byte> output;
  size_t out_size = 0;

  if (a.compress && !a.logfile && (!a.codec_set || a.codec == spectra8::Codec::Prism) &&
      looks_like_text_file(input)) {
    a.codec = spectra8::Codec::Prism;
    if (!a.block_log_set) a.block_log = 27;  // 128 MiB
    if (!a.level_set) a.level = 11;
    if (!a.filters_set) a.filters = true;
    if (!a.quiet)
      std::fprintf(stderr,
                   "spectra8: text input, using Prism BWT level %d with 128 MiB blocks and word filter\n",
                   a.level);
  } else if (a.compress && !a.logfile && !a.codec_set &&
             spectra8::codec_available(spectra8::Codec::Lzma) &&
             looks_like_executable_file(input)) {
    // Machine code's redundancy is short-range and position-dependent; of
    // the linked codecs only lzma models it (README "Executables"). The
    // filter stage selects the matching BCJ.
    a.codec = spectra8::Codec::Lzma;
    if (!a.level_set) a.level = 9;
    if (!a.filters_set) a.filters = true;
    if (!a.quiet)
      std::fprintf(stderr, "spectra8: executable input, using lzma level %d with BCJ filter\n",
                   a.level);
  } else if (a.compress && !a.logfile && a.codec_set &&
             a.codec == spectra8::Codec::Crystal && !a.block_log_set &&
             looks_like_text_file(input)) {
    // Crystal's BWT text mode gains ~9% from 8 MiB blocks over the 1 MiB
    // default; the lattice path on binary data does not, so only on text.
    a.block_log = 23;
    if (!a.quiet)
      std::fprintf(stderr, "spectra8: text input, using Crystal with 8 MiB blocks\n");
  }
  // Bare `--flash` is the 4 GB/s SIMD path (L1). Pass `-5` / `--level=5` for
  // the optional Huffman / compact extras / rep-offset ratio tools.
  if (a.compress && a.codec_set && a.codec == spectra8::Codec::Flash &&
      !a.level_set) {
    a.level = 1;
  }

  if (a.compress) {
    const size_t bound = spectra8::compress_bound(input.size(), a.codec);
    output.resize(bound ? bound : spectra8::compress_bound(input.size()));
    spectra8::CompressParams cp;
    cp.codec = a.codec;
    cp.level = a.level;
    cp.threads = a.threads;
    cp.block_log = a.block_log;
    cp.seek_table = a.seek_table;
    cp.block_checksums = a.block_crc;
    cp.content_checksum = a.content_crc;
    cp.filters = a.filters;
    cp.filter_fallback = a.filter_fallback;
    cp.filter_sample_full = a.filter_sample_full;
    cp.auto_min_gain_pct = a.auto_min_gain_pct;
    cp.auto_max_level = a.auto_max_level;
    cp.auto_codecs = a.auto_codecs;
    cp.filter_mask = a.filter_mask;
    cp.gpu = a.gpu;
    cp.context_oracle = a.context_oracle;
    auto r = spectra8::compress(input, output, cp);
    if (!r) {
      std::fprintf(stderr, "spectra8: compression failed: %s\n",
                   std::string(spectra8::status_message(r.status)).c_str());
      return 1;
    }
    out_size = r.value;
  } else {
    auto cs = spectra8::get_frame_content_size(input);
    if (!cs) {
      std::fprintf(stderr,
                   "spectra8: '%s' has no recoverable content size "
                   "(not LHC4/zstd/bzip3/xz/L4EF, or size omitted)\n",
                   a.input.c_str());
      return 1;
    }
    output.resize(static_cast<size_t>(*cs));
    spectra8::DecompressParams dp;
    dp.threads = a.threads;
    auto r = spectra8::decompress(input, output, dp);
    if (!r) {
      std::fprintf(stderr, "spectra8: decompression failed: %s\n",
                   std::string(spectra8::status_message(r.status)).c_str());
      return 1;
    }
    out_size = r.value;
  }

  if (a.to_stdout) {
    std::fwrite(output.data(), 1, out_size, stdout);
  } else {
    if (!write_file(a.output, output.data(), out_size)) {
      std::fprintf(stderr, "spectra8: cannot write '%s'\n", a.output.c_str());
      return 1;
    }
  }

  if (!a.quiet) {
    size_t in = input.size();
    size_t out = out_size;
    double ratio = (a.compress ? (out ? (double)in / out : 0.0)
                               : (in ? (double)out / in : 0.0));
    std::fprintf(stderr, "%-24s %10zu -> %10zu  (%.2fx)\n", a.input.c_str(), in,
                 out, ratio);
  }
  return 0;
}
