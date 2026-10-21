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

// Benchmark subcommand for the spectra8 CLI.
#ifndef SPECTRA8_CLI_BENCH_HPP
#define SPECTRA8_CLI_BENCH_HPP

#include <cstddef>
#include <cstdint>
#include <string>

#include "spectra8/spectra8.hpp"

namespace spectra8_cli {

// Statistical distribution used for the synthetic numeric-column datasets
// (the i16/i32/i64/f32/f64 physics profiles). `All` generates every family
// (bin-*, gau-*, poi-* plus continuous waveforms, ramps, timestamps, decimals,
// and x86 code), which is the default for `bench --verdict`; `Classic16`
// generates the original 16 datasets; `Text` uses the enwik8 Wikipedia corpus;
// the plain bench path defaults to `Binomial` to keep its historical dataset set
// and runtime.
enum class BenchDist {
  Binomial,
  Gaussian,
  Poisson,
  Classic16,
  All,
  Text,
  Binaries,  // shared-library text/rodata slices (`bench --verdict --binaries`)
  RawInt     // peaked integer columns with a long tail (`--rawint`)
};

// Base seed for synthetic datasets. Each named profile uses
// `seed ^ strlen(name)` so columns stay independent. The historical default
// is 0xABCDEF; `--seed` changes it without otherwise altering generation.
inline constexpr std::uint64_t kDefaultBenchSeed = 0xABCDEF;

inline const char* bench_codec_name(spectra8::Codec c) noexcept {
  switch (c) {
    case spectra8::Codec::Beam:
      return "beam";
    case spectra8::Codec::Prism:
      return "prism";
    case spectra8::Codec::Zstd:
      return "zstd";
    case spectra8::Codec::Bzip3:
      return "bzip3";
    case spectra8::Codec::Lzma:
      return "lzma";
    case spectra8::Codec::Auto:
      return "auto";
    case spectra8::Codec::Mosaic:
      return "mosaic";
    case spectra8::Codec::Oracle:
      return "oracle";
    case spectra8::Codec::Crystal:
      return "crystal";
    case spectra8::Codec::Flash:
      return "flash";
    case spectra8::Codec::Laser:
      return "laser";
    case spectra8::Codec::Grating:
      return "grating";
  }
  return "?";
}

inline bool bench_codec_external(spectra8::Codec c) noexcept {
  return c == spectra8::Codec::Zstd || c == spectra8::Codec::Bzip3 ||
         c == spectra8::Codec::Lzma;
}

// Run the benchmark for `budget_s` seconds over datasets of `bytes` bytes each
// (0 selects the default of 128 KiB).
// `codec` is the compressor for the main level tables (default Crystal;
// `--codec NAME` / `--beam` / `--prism` / `--mosaic` / `--oracle` /
// `--crystal`). `--verdict` and `--bwt-chains` ignore this and keep their
// own codec sets.
// When `gpu` is true, also runs a CPU-vs-GPU (Metal) match-finder comparison
// (Beam only; independent of `codec`).
// When `compare` is true (and the CLI was built with zstd), also runs every
// dataset through libzstd at level-matched settings and prints a side-by-side
// comparison matrix of compression ratios and codec bandwidths.
// When `no_ultra` is true, levels 10 and 11 (the slow CPU optimal parser) are
// skipped, which cuts the wall time substantially.
// When `filter` is true, adds a table comparing each dataset with vs without
// the auto-detected byte filters (`--filter`), reporting the chosen transform.
// When `verdict` is true, skips all of the above and runs only a compact
// head-to-head over the 6 averaged datasets: spectra8 Beam L1/L5/L9/ultra, zstd, BWT,
// optional LZMA and bzip3 (ratio + single-thread throughput).
// When `verdict_no_filter` is true (only with `verdict`), skips the external
// byte-filter stage and compresses each dataset as raw input.
// When `verdict_no_rle` is true (the default, only with `verdict`, and not
// with `verdict_no_filter`), applies the external filter but skips byte-RLE.
// When `verdict_no_filter_fallback` is true (only with `verdict`, and not with
// `verdict_no_filter`), skips the raw compress trial and encodes only the
// detected external filter path (faster; all rows including zstd/lzma/bzip3).
// When `bwt_chains` is true, sweeps parallel LF-walk chain counts (4/8/12/16),
// prints throughput, and recommends SPECTRA8_BWT_ENCODE_CHAINS for this machine.
// When `verdict_full` is true (only with `verdict`), scans every level of
// every implementation (spectra8 Beam and BWT L1..L11, zstd L1..max, lzma
// P0..P9+P9e) instead of the compact per-codec selection.
// When `verdict_no_dict` is true (only with `verdict`), the dictionary-remap
// filters (dict32/64) are excluded from the external filter staging; only the
// size-preserving byte transforms compete. Useful to isolate their effect.
// When `filter_sample_full` is true, filter detect scores the whole block
// instead of the default 16 KiB mid-sample (applies to `--verdict` external
// staging and the `--filter` comparison table).
// When `verdict_auto_max_level` is true (only with `verdict`), the Auto rows
// race each codec at its highest effort instead of level 5.
// When `verdict_brute` is true (only with `verdict`, and not with
// `verdict_no_filter`), each numeric column is the brute-force filter over
// the entropy-detected one. Brute compresses every filter and keeps the
// smallest. Off by default; the search is much slower than detection.
// `dist` selects the statistical distribution of the synthetic numeric
// columns (`--dist`); the CLI defaults it to All for `--verdict` and to
// Binomial for the plain bench path.
// `seed` is the base RNG seed for those datasets (`--seed`, default
// kDefaultBenchSeed).
// Returns a process exit code (0 on success).
int run_benchmark(double budget_s, std::size_t bytes, bool gpu = false,
                  bool compare = false, bool no_ultra = false,
                  bool filter = false, bool verdict = false,
                  bool verdict_no_filter = false, bool verdict_no_rle = true,
                  bool verdict_no_filter_fallback = false,
                  bool bwt_chains = false, bool verdict_full = false,
                  bool verdict_no_dict = false,
                  bool filter_sample_full = false,
                  bool verdict_auto_max_level = false,
                  bool verdict_brute = false,
                  BenchDist dist = BenchDist::Binomial,
                  std::uint64_t seed = kDefaultBenchSeed,
                  spectra8::Codec codec = spectra8::Codec::Crystal,
                  bool complete = false,
                  spectra8::CodecMask only_rows = ~spectra8::CodecMask{0},
                  bool stats = false);

// `bench --explain`: prints a two-column table of every compressor name that
// appears in the benchmark tables with a keyword summary of its algorithm.
// Returns 0.
int print_codec_explain();

// `bench --dir DIR [--no-filter] [--auto-gain N]`: the verdict-style table
// over the files of a directory (real pages instead of synthetic datasets).
// Returns a process exit code (0 on success).
// `auto_codecs` restricts the Auto row (default all); when it differs from
// kAutoCodecsAll a second Auto row with the full set is printed for comparison.
// `only_rows` (`--only LIST`) is a codec_bit() mask over every Codec incl.
// Auto/Oracle: rows whose codec is not in it are skipped (~0u = all rows).
// `level` > 0 (`--level N` / `-N`) replaces the default multi-level matrix
// with one row per codec at that level.
// `parallel` (`--parallel N`) is the number of files processed concurrently,
// each call single-threaded; 0 = hardware concurrency, 1 = serial. Speeds are
// then wall-clock aggregate throughput of the batch.
// Auto rows always print the codec mix (which flavour won how many files);
// `stats` (`--stats`) prints the full CompressStats report -- filters detected
// / kept on the wire, wire format, Auto mix -- under every row.
int run_dir_bench(const std::string& dir, bool no_filter, int auto_gain_pct,
                  spectra8::CodecMask auto_codecs = spectra8::kAutoCodecsAll,
                  spectra8::CodecMask only_rows = ~spectra8::CodecMask{0},
                  int level = 0, unsigned parallel = 1, bool stats = false);

// `bench --h2d`: CPU unzip vs H2D vs the overlapped page pump (and a dummy
// device consume). Times the three legs in docs/cpu-h2d-page-service.md.
int run_h2d_bench(double budget_s, std::size_t bytes, spectra8::Codec codec);

}  // namespace spectra8_cli

#endif  // SPECTRA8_CLI_BENCH_HPP
