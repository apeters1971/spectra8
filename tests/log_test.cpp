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

// Line-oriented logfile encode/decode/tail selection tests.
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "log.hpp"
#include "spectra8/spectra8.hpp"

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

}  // namespace

int main() {
  check(spectra8_cli::is_stdin_path("-"), "stdin -");
  check(spectra8_cli::is_stdin_path("/dev/stdin"), "stdin /dev/stdin");
  check(spectra8_cli::is_stdout_path("/dev/stdout"), "stdout /dev/stdout");
  check(!spectra8_cli::is_stdin_path("log.txt"), "not stdin");

  const char raw[] = "hello\x00world";
  std::vector<std::byte> bytes(sizeof raw - 1);
  std::memcpy(bytes.data(), raw, bytes.size());
  auto b64 = spectra8_cli::base64_encode(bytes);
  std::vector<std::byte> back;
  check(spectra8_cli::base64_decode(b64, back), "b64 decode");
  check(back == bytes, "b64 roundtrip");
  check(spectra8_cli::base64_decode("", back) && back.empty(), "b64 empty");

  spectra8::CompressParams cp;
  cp.level = 1;
  std::string enc, dec;
  check(spectra8_cli::encode_log_line("hello", enc, cp) && !enc.empty(),
        "encode hello");
  check(spectra8_cli::decode_log_line(enc, dec) && dec == "hello",
        "decode hello");
  check(spectra8_cli::encode_log_line("", enc, cp), "encode empty line");
  check(spectra8_cli::decode_log_line(enc, dec) && dec.empty(),
        "decode empty line");

  FILE* mid = std::tmpfile();
  FILE* out = std::tmpfile();
  check(mid && out, "tmpfile");
  if (!mid || !out) return 1;
  std::fputs("one\r\n\nthree\n", mid);
  std::rewind(mid);
  std::uint64_t lines = 0, ib = 0, ob = 0;
  check(spectra8_cli::encode_logfile(mid, out, cp, &lines, &ib, &ob) == 0,
        "encode_logfile");
  check(lines == 3, "three lines encoded");
  std::rewind(out);
  FILE* restored = std::tmpfile();
  check(restored, "tmpfile restore");
  if (!restored) return 1;
  check(spectra8_cli::decode_logfile(out, restored, &lines, &ib, &ob) == 0,
        "decode_logfile");
  std::rewind(restored);
  char buf[64] = {};
  check(std::fread(buf, 1, sizeof buf, restored) > 0, "read restored");
  check(std::strcmp(buf, "one\n\nthree\n") == 0, "crlf + empty roundtrip");

  FILE* encoded = std::tmpfile();
  check(encoded, "tmpfile encoded");
  if (!encoded) return 1;
  std::rewind(mid);
  check(spectra8_cli::encode_logfile(mid, encoded, cp) == 0, "re-encode");
  std::rewind(encoded);
  std::vector<std::string> selected;
  spectra8_cli::TailOptions last2;
  last2.count = 2;
  check(spectra8_cli::tail_select_encoded(encoded, last2, selected) == 0,
        "tail -n 2");
  check(selected.size() == 2, "tail -n 2 count");
  check(spectra8_cli::decode_log_line(selected[0], dec) && dec.empty(),
        "tail -n 2 first is empty line");
  check(spectra8_cli::decode_log_line(selected[1], dec) && dec == "three",
        "tail -n 2 last is three");

  std::rewind(encoded);
  spectra8_cli::TailOptions from2;
  from2.from_start = true;
  from2.count = 2;
  check(spectra8_cli::tail_select_encoded(encoded, from2, selected) == 0,
        "tail -n +2");
  check(selected.size() == 2, "tail -n +2 count");
  check(spectra8_cli::decode_log_line(selected[0], dec) && dec.empty(),
        "tail -n +2 starts at empty");
  check(spectra8_cli::decode_log_line(selected[1], dec) && dec == "three",
        "tail -n +2 ends at three");

  char* argv[] = {const_cast<char*>("spectra8"), const_cast<char*>("tail"),
                  const_cast<char*>("-n"),        const_cast<char*>("+4"),
                  const_cast<char*>("-q"),        const_cast<char*>("x.l4l")};
  spectra8_cli::TailOptions parsed;
  check(spectra8_cli::parse_tail_args(6, argv, parsed) == 0, "parse tail");
  check(parsed.from_start && parsed.count == 4, "parse +4");
  check(parsed.quiet && parsed.files.size() == 1 && parsed.files[0] == "x.l4l",
        "parse file");

  std::fclose(mid);
  std::fclose(out);
  std::fclose(restored);
  std::fclose(encoded);

  if (g_failures == 0) {
    std::printf("All %d logfile checks passed.\n", g_checks);
    return 0;
  }
  std::fprintf(stderr, "%d/%d logfile checks FAILED.\n", g_failures, g_checks);
  return 1;
}
