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

// Line-oriented Spectra8 logfile: one independent frame per text line.
#ifndef SPECTRA8_CLI_LOG_HPP
#define SPECTRA8_CLI_LOG_HPP

#include <cstdint>
#include <cstdio>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "spectra8/spectra8.hpp"

namespace spectra8_cli {

bool is_stdin_path(const std::string& path) noexcept;
bool is_stdout_path(const std::string& path) noexcept;

std::string base64_encode(std::span<const std::byte> src);
bool base64_decode(std::string_view src, std::vector<std::byte>& dst);

bool encode_log_line(std::string_view line, std::string& encoded,
                     const spectra8::CompressParams& cp);
bool decode_log_line(std::string_view encoded, std::string& line);

// Stream text lines <-> one base64 Spectra8 frame per line. Returns a process
// exit code. Optional counters are written when non-null.
int encode_logfile(FILE* in, FILE* out, const spectra8::CompressParams& cp,
                   std::uint64_t* lines = nullptr,
                   std::uint64_t* in_bytes = nullptr,
                   std::uint64_t* out_bytes = nullptr);
int decode_logfile(FILE* in, FILE* out, std::uint64_t* lines = nullptr,
                   std::uint64_t* in_bytes = nullptr,
                   std::uint64_t* out_bytes = nullptr);

struct TailOptions {
  bool from_start = false;  // +N (1-based start)
  bool by_bytes = false;
  std::int64_t count = 10;
  bool follow = false;
  bool follow_name = false;  // --follow=name / -F
  bool retry = false;
  bool quiet = false;
  bool verbose = false;
  double sleep_s = 1.0;
  int pid = 0;  // 0 = none
  std::vector<std::string> files;
};

// Parse `spectra8 tail ...` (argv[1] == "tail"). Returns 0, 2 for --help, -1
// on error.
int parse_tail_args(int argc, char** argv, TailOptions& opt);

int run_tail(const TailOptions& opt);

// Encoded-line window that `tail` would print before decode (no follow).
// Used by tests.
int tail_select_encoded(FILE* in, const TailOptions& opt,
                        std::vector<std::string>& lines);

void print_tail_usage(FILE* f);

}  // namespace spectra8_cli

#endif  // SPECTRA8_CLI_LOG_HPP
