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

// Parity: decode_array vs ArrayStreamDecoder incremental decode.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "util/common.hpp"
#include "util/entropy.hpp"

using spectra8::Status;
using spectra8::detail::ArrayStreamDecoder;
using spectra8::detail::ByteReader;
using spectra8::detail::decode_array;
using spectra8::detail::encode_array;
using spectra8::detail::min_of;
using spectra8::detail::u8;
using spectra8::detail::usize;

namespace {

int g_failures = 0;

void check(bool cond, const char* msg) {
  if (!cond) {
    ++g_failures;
    std::fprintf(stderr, "FAIL: %s\n", msg);
  }
}

void test_one(const std::vector<uint8_t>& data, int level, usize chunk) {
  std::vector<u8> enc;
  encode_array(enc, data.data(), data.size(), level);

  ByteReader r0(enc.data(), enc.size());
  std::vector<u8> full;
  if (Status s = decode_array(r0, full); s != Status::Ok) {
    check(false, "decode_array failed");
    return;
  }

  ByteReader r1(enc.data(), enc.size());
  ArrayStreamDecoder sd;
  if (Status s = sd.begin(r1); s != Status::Ok) {
    check(false, "begin failed");
    return;
  }

  std::vector<u8> inc(data.size());
  usize off = 0;
  while (off < data.size()) {
    const usize n = min_of(chunk, data.size() - off);
    if (Status s = sd.decode(inc.data() + off, n); s != Status::Ok) {
      check(false, "incremental decode failed");
      return;
    }
    off += n;
  }
  check(sd.remaining() == 0, "remaining not zero");
  check(full.size() == data.size(), "size mismatch");
  check(std::memcmp(full.data(), inc.data(), data.size()) == 0,
        "data mismatch");
}

}  // namespace

int main() {
  std::mt19937 rng(0xBEEF);
  for (int level = 1; level <= 12; ++level) {
    for (usize len : {0u, 1u, 31u, 255u, 1023u, 1024u, 4096u, 20000u}) {
      std::vector<uint8_t> d(len);
      for (usize i = 0; i < len; ++i) d[i] = static_cast<uint8_t>(rng());
      for (usize chunk :
           {usize{1}, usize{3}, usize{17}, len ? len : usize{1}}) {
        test_one(d, level, chunk);
      }
    }
    // Runny data (RLE candidate).
    std::vector<uint8_t> run(5000, 0xAB);
    for (usize chunk : {1u, 7u, 500u}) test_one(run, level, chunk);
  }
  if (g_failures == 0)
    std::printf("stream_decoder_test: all checks passed\n");
  return g_failures ? 1 : 0;
}
