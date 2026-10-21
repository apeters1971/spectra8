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

// Reversible Base64 pre-filter (Filter::Base64 = 34):
// Automatically detects and decodes Base64 payloads (e.g. embedded in JSON,
// XML, PEM certificates, MIME streams, or raw Base64 files) into compact raw
// binary before compression.
//
// Base64 expands binary data by 33% and destroys byte alignment. By reversibly
// decoding Base64 runs to 8-bit binary, uncompressed size shrinks by 25%
// immediately and downstream LZ/BWT algorithms can exploit binary patterns
// (endianness, stride, zero-runs). The inverse filter re-encodes the binary
// back into the exact original Base64 characters, preserving line wrap (64/76/custom)
// and padding ('=' / '==').

#ifndef SPECTRA8_SRC_BASE64_FILTER_HPP
#define SPECTRA8_SRC_BASE64_FILTER_HPP

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "util/common.hpp"
#include "spectra8/spectra8.hpp"

namespace spectra8::detail::base64_detail {

inline constexpr u8 kBase64Magic = 0xB6;
inline constexpr u8 kBase64Ver = 1;

// Base64 lookup tables
inline constexpr char kStdChars[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
inline constexpr char kUrlChars[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";

inline int b64_val_std(u8 c) noexcept {
  if (c >= 'A' && c <= 'Z') return c - 'A';
  if (c >= 'a' && c <= 'z') return c - 'a' + 26;
  if (c >= '0' && c <= '9') return c - '0' + 52;
  if (c == '+') return 62;
  if (c == '/') return 63;
  return -1;
}

inline int b64_val_url(u8 c) noexcept {
  if (c >= 'A' && c <= 'Z') return c - 'A';
  if (c >= 'a' && c <= 'z') return c - 'a' + 26;
  if (c >= '0' && c <= '9') return c - '0' + 52;
  if (c == '-') return 62;
  if (c == '_') return 63;
  return -1;
}

inline bool is_std_b64_char(u8 c) noexcept {
  return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') ||
         c == '+' || c == '/';
}

inline bool is_candidate_b64_char(const u8* src, usize i, usize n) noexcept {
  const u8 c = src[i];
  if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') ||
      c == '+' || c == '/' || c == '_') {
    return true;
  }
  if (c == '-') {
    // Only allow '-' if it's isolated (not part of '--' or '---')
    if (i + 1 < n && src[i + 1] == '-') return false;
    if (i > 0 && src[i - 1] == '-') return false;
    return true;
  }
  return false;
}

// Sniffer: check if the block contains significant Base64 content (at least 64 consecutive
// Base64 characters and >= 20% of the sample).
inline bool detect_base64(const u8* src, usize n) noexcept {
  if (n < 96) return false;
  usize cur_run = 0;
  usize b64_total = 0;
  usize max_run = 0;

  for (usize i = 0; i < n; ++i) {
    if (is_candidate_b64_char(src, i, n)) {
      ++cur_run;
    } else if (src[i] == '\r' || src[i] == '\n' || src[i] == '=') {
      // Allowed within Base64 stream
    } else {
      if (cur_run >= 64) {
        b64_total += cur_run;
        if (cur_run > max_run) max_run = cur_run;
      }
      cur_run = 0;
    }
  }
  if (cur_run >= 64) {
    b64_total += cur_run;
    if (cur_run > max_run) max_run = cur_run;
  }

  return max_run >= 64 && (b64_total * 5 >= n);
}

// Encode binary bytes to Base64 directly into dst.
inline usize encode_b64_chunk(const u8* bin, usize bin_len, bool url_mode,
                              int pad_count, int newline_mode, bool trailing_nl,
                              u8 line_width, u8* dst, usize dst_cap) noexcept {
  const char* tab = url_mode ? kUrlChars : kStdChars;
  usize out_idx = 0;
  u32 col = 0;

  auto put_char = [&](char c) {
    if (out_idx < dst_cap) {
      dst[out_idx++] = static_cast<u8>(c);
    }
    if (newline_mode > 0 && line_width > 0) {
      if (++col == line_width) {
        col = 0;
        if (newline_mode == 2 && out_idx < dst_cap) dst[out_idx++] = '\r';
        if (out_idx < dst_cap) dst[out_idx++] = '\n';
      }
    }
  };

  usize i = 0;
  while (i + 3 <= bin_len) {
    const u32 b0 = bin[i];
    const u32 b1 = bin[i + 1];
    const u32 b2 = bin[i + 2];
    put_char(tab[b0 >> 2]);
    put_char(tab[((b0 & 3) << 4) | (b1 >> 4)]);
    put_char(tab[((b1 & 15) << 2) | (b2 >> 6)]);
    put_char(tab[b2 & 63]);
    i += 3;
  }

  if (i < bin_len) {
    const usize rem = bin_len - i;
    const u32 b0 = bin[i];
    put_char(tab[b0 >> 2]);
    if (rem == 1) {
      put_char(tab[(b0 & 3) << 4]);
      if (pad_count >= 2) {
        put_char('=');
        put_char('=');
      }
    } else {  // rem == 2
      const u32 b1 = bin[i + 1];
      put_char(tab[((b0 & 3) << 4) | (b1 >> 4)]);
      put_char(tab[(b1 & 15) << 2]);
      if (pad_count >= 1) {
        put_char('=');
      }
    }
  }

  if (trailing_nl && col > 0) {
    if (newline_mode == 2 && out_idx < dst_cap) dst[out_idx++] = '\r';
    if (out_idx < dst_cap) dst[out_idx++] = '\n';
  }

  return out_idx;
}

struct B64Span {
  usize start = 0;
  usize end = 0;
  bool is_b64 = false;
  bool url_mode = false;
  int pad_count = 0;
  int newline_mode = 0;  // 0: none, 1: \n, 2: \r\n
  bool trailing_nl = false;
  u8 line_width = 0;
  std::vector<u8> decoded;
};

// Forward transform
inline bool base64_fwd(const u8* src, usize n, std::vector<u8>& out) {
  if (n < 96) return false;

  std::vector<B64Span> chunks;
  usize i = 0;

  while (i < n) {
    // Find contiguous candidates of Base64 characters and newlines
    usize span_start = i;
    while (i < n && !is_candidate_b64_char(src, i, n)) ++i;
    if (i > span_start) {
      chunks.push_back({span_start, i, false, false, 0, 0, false, 0, {}});
    }
    if (i >= n) break;

    usize b64_start = i;
    usize count_chars = 0;
    usize count_dash_under = 0;
    usize count_plus_slash = 0;

    // Scan until non-base64, non-whitespace character
    while (i < n) {
      const u8 c = src[i];
      if (is_candidate_b64_char(src, i, n)) {
        ++count_chars;
        if (c == '-' || c == '_') ++count_dash_under;
        if (c == '+' || c == '/') ++count_plus_slash;
        ++i;
      } else if (c == '\r' || c == '\n') {
        ++i;
      } else if (c == '=') {
        // Padding
        ++i;
      } else {
        break;
      }
    }
    usize b64_end = i;

    if (count_chars < 64) {
      // Too short to be worth decoding
      chunks.push_back({b64_start, b64_end, false, false, 0, 0, false, 0, {}});
      continue;
    }

    // Analyze padding
    int pad_count = 0;
    usize p = b64_end;
    while (p > b64_start && (src[p - 1] == '\r' || src[p - 1] == '\n')) --p;
    while (p > b64_start && src[p - 1] == '=') {
      ++pad_count;
      --p;
    }
    if (pad_count > 2) pad_count = 2;

    const bool url_mode = (count_dash_under > count_plus_slash);
    auto decode_val = [&](u8 c) noexcept -> int {
      return url_mode ? b64_val_url(c) : b64_val_std(c);
    };

    // Analyze line wrapping: check first line break
    int newline_mode = 0;
    u8 line_width = 0;
    usize first_nl = b64_start;
    while (first_nl < b64_end && src[first_nl] != '\n' && src[first_nl] != '\r') ++first_nl;
    if (first_nl < b64_end) {
      const usize len0 = first_nl - b64_start;
      if (len0 >= 16 && len0 <= 255) {
        newline_mode = (src[first_nl] == '\r' && first_nl + 1 < b64_end &&
                        src[first_nl + 1] == '\n')
                           ? 2
                           : 1;
        line_width = static_cast<u8>(len0);
      }
    }

    // Decode binary
    std::vector<u8> bin;
    bin.reserve((count_chars * 3) / 4 + 4);

    u32 acc = 0;
    int acc_bits = 0;
    for (usize k = b64_start; k < p; ++k) {
      const int v = decode_val(src[k]);
      if (v >= 0) {
        acc = (acc << 6) | static_cast<u32>(v);
        acc_bits += 6;
        if (acc_bits >= 8) {
          acc_bits -= 8;
          bin.push_back(static_cast<u8>(acc >> acc_bits));
        }
      }
    }

    // Verify 100% lossless roundtrip before accepting
    const bool trailing_nl = (b64_end > b64_start && src[b64_end - 1] == '\n');
    std::vector<u8> re_enc(b64_end - b64_start + 64);
    const usize re_len = encode_b64_chunk(bin.data(), bin.size(), url_mode, pad_count,
                                          newline_mode, trailing_nl, line_width,
                                          re_enc.data(), re_enc.size());

    if (re_len == (b64_end - b64_start) &&
        std::memcmp(src + b64_start, re_enc.data(), re_len) == 0) {
      chunks.push_back({b64_start, b64_end, true, url_mode, pad_count, newline_mode,
                        trailing_nl, line_width, std::move(bin)});
    } else {
      chunks.push_back({b64_start, b64_end, false, false, 0, 0, false, 0, {}});
    }
  }

  // Check if any Base64 chunk was created
  usize b64_chunks = 0;
  for (const auto& c : chunks) {
    if (c.is_b64) ++b64_chunks;
  }
  if (b64_chunks == 0) return false;

  // Merge adjacent literal chunks
  std::vector<B64Span> merged;
  for (auto& c : chunks) {
    if (!merged.empty() && !merged.back().is_b64 && !c.is_b64) {
      merged.back().end = c.end;
    } else {
      merged.push_back(std::move(c));
    }
  }

  // Build framed stream
  out.clear();
  out.reserve(n);
  out.push_back(kBase64Magic);
  out.push_back(kBase64Ver);
  ByteWriter wr(out);
  wr.varint(n);
  wr.varint(merged.size());

  for (const auto& c : merged) {
    if (!c.is_b64) {
      wr.byte(0);  // Literal
      const usize len = c.end - c.start;
      wr.varint(len);
      wr.bytes(src + c.start, len);
    } else {
      wr.byte(1);  // Base64
      const u8 flags = static_cast<u8>((c.trailing_nl ? 32 : 0) | (c.newline_mode << 3) |
                                       (c.pad_count << 1) | (c.url_mode ? 1 : 0));
      wr.byte(flags);
      if (c.newline_mode > 0) {
        wr.byte(c.line_width);
      }
      wr.varint(c.decoded.size());
      wr.bytes(c.decoded.data(), c.decoded.size());
    }
  }

  return out.size() < n;
}

// Inverse transform
inline Status base64_inv(const u8* data, usize len, u8* dst, usize n) {
  if (!data || len < 4) return Status::CorruptInput;
  if (data[0] != kBase64Magic || data[1] != kBase64Ver) return Status::CorruptInput;

  ByteReader r(data + 2, len - 2);
  const u64 orig_len = r.varint();
  if (r.error || orig_len != n) return Status::CorruptInput;

  const u64 num_chunks = r.varint();
  if (r.error) return Status::CorruptInput;

  usize out_pos = 0;
  for (u64 ch = 0; ch < num_chunks; ++ch) {
    if (r.remaining() == 0) return Status::CorruptInput;
    const u8 tag = r.byte();
    if (tag == 0) {
      // Literal
      const u64 lit_len = r.varint();
      if (r.error || lit_len > r.remaining() || out_pos + lit_len > n) return Status::CorruptInput;
      const u8* lit_p = r.take(lit_len);
      if (!lit_p) return Status::CorruptInput;
      std::memcpy(dst + out_pos, lit_p, lit_len);
      out_pos += lit_len;
    } else if (tag == 1) {
      // Base64
      if (r.remaining() == 0) return Status::CorruptInput;
      const u8 flags = r.byte();
      const bool url_mode = (flags & 1) != 0;
      const int pad_count = (flags >> 1) & 3;
      const int newline_mode = (flags >> 3) & 3;
      const bool trailing_nl = (flags & 32) != 0;
      u8 line_width = 0;
      if (newline_mode > 0) {
        if (r.remaining() == 0) return Status::CorruptInput;
        line_width = r.byte();
      }
      const u64 dec_len = r.varint();
      if (r.error || dec_len > r.remaining()) return Status::CorruptInput;
      const u8* bin_p = r.take(dec_len);
      if (!bin_p) return Status::CorruptInput;

      // Re-encode into destination buffer
      const usize written = encode_b64_chunk(bin_p, dec_len, url_mode, pad_count,
                                             newline_mode, trailing_nl, line_width,
                                             dst + out_pos, n - out_pos);
      out_pos += written;
      if (out_pos > n) return Status::CorruptInput;
    } else {
      return Status::CorruptInput;
    }
  }

  if (out_pos != n || r.remaining() != 0) return Status::CorruptInput;
  return Status::Ok;
}

}  // namespace spectra8::detail::base64_detail

#endif  // SPECTRA8_SRC_BASE64_FILTER_HPP
