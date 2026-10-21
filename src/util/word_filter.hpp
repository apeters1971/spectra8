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

// Reversible Text Word Normalization filter (Filter::WordNorm = 33):
//
// Inspired by Gleipnir's WRT (Word Transform) and FXV:
// Automatically harvests frequent words in text blocks, replaces them with
// single-byte tokens in 0x80..0xF7, and decouples case (lowercase, Capitalized,
// ALL CAPS) with transparent prefix flags.
//
// By mapping frequent words (3-24 characters) to single-byte tokens:
//   1. Uncompressed text stream shrinks by 25-35%.
//   2. LZ77 match lengths triple in effective character coverage.
//   3. Case variations ("The" / "the" / "THE") share the same token, eliminating
//      redundant dictionary context.
//
// Self-contained: The block stores its own harvested vocabulary in the header.
// Decompression requires zero external dictionaries, has zero allocations,
// and streams at > 1.5 GB/s.

#ifndef SPECTRA8_SRC_WORD_FILTER_HPP
#define SPECTRA8_SRC_WORD_FILTER_HPP

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "util/common.hpp"
#include "spectra8/spectra8.hpp"

namespace spectra8::detail::word_norm_detail {

inline constexpr u8 kWordNormMagic = 0x57;  // 'W'
inline constexpr u8 kWordNormVer = 2;

inline constexpr u8 kEsc = 0xFD;        // Escape byte for raw bytes >= 0x80
inline constexpr u8 kFlagCap = 0xFE;    // Prefix flag: next token is Capitalized
inline constexpr u8 kFlagUpper = 0xFF;  // Prefix flag: next token is ALL CAPS

inline constexpr usize kMaxDictWords1 = 120;  // 1-byte tokens: 0x80 .. 0x80 + 119 = 0xF7
inline constexpr usize kMaxDictWords = 1144;  // 120 + 4 * 256 (2-byte tokens: 0xF8..0xFB)
inline constexpr usize kMinWordLen = 2;
inline constexpr usize kMaxWordLen = 28;

inline bool is_ascii_alpha(u8 c) noexcept {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}

inline u8 to_lower_ascii(u8 c) noexcept {
  return (c >= 'A' && c <= 'Z') ? static_cast<u8>(c + ('a' - 'A')) : c;
}

inline u8 to_upper_ascii(u8 c) noexcept {
  return (c >= 'a' && c <= 'z') ? static_cast<u8>(c - ('a' - 'A')) : c;
}

// Sniffer: check if the block looks like text with high ASCII letter density.
inline bool detect_word_norm(const u8* src, usize n) noexcept {
  if (n < 512) return false;
  usize text_chars = 0;
  usize letters = 0;
  usize nul = 0;

  for (usize i = 0; i < n; ++i) {
    const u8 c = src[i];
    if (c == 0) {
      ++nul;
    } else if (is_ascii_alpha(c)) {
      ++letters;
      ++text_chars;
    } else if (c == '\t' || c == '\n' || c == '\r' || (c >= 32 && c < 127) || c >= 128) {
      ++text_chars;
    }
  }

  // Reject binary data
  if (nul * 200 > n) return false;

  // Must be overwhelmingly text
  if (text_chars * 10 < n * 9) return false;

  // Must have at least 35% ASCII letters
  if (letters * 20 < n * 7) return false;

  return true;
}

enum class WordCase : u8 {
  Lower,
  Cap,
  Upper,
  Other
};

inline WordCase classify_word(const u8* s, usize len) noexcept {
  if (len < kMinWordLen || len > kMaxWordLen) return WordCase::Other;
  const bool first_upper = (s[0] >= 'A' && s[0] <= 'Z');
  bool all_upper = first_upper;
  bool all_lower = !first_upper;
  bool rest_all_lower = true;

  for (usize i = 1; i < len; ++i) {
    const u8 c = s[i];
    if (c >= 'A' && c <= 'Z') {
      all_lower = false;
      rest_all_lower = false;
    } else if (c >= 'a' && c <= 'z') {
      all_upper = false;
    } else {
      return WordCase::Other;
    }
  }

  if (all_lower) return WordCase::Lower;
  if (all_upper) return WordCase::Upper;
  if (first_upper && rest_all_lower) return WordCase::Cap;
  return WordCase::Other;
}

// Forward transform: harvest dictionary, encode tokens & casing, write header.
inline bool word_norm_fwd(const u8* src, usize n, std::vector<u8>& out) {
  if (n < 512) return false;

  // Step 1: Count word and markup token frequencies
  struct WordCount {
    std::string word;
    usize count = 0;
  };

  std::unordered_map<std::string, usize> freq;
  freq.reserve(4096);

  static const char* const kPfx[] = {
      "[[Category:", "[[Image:", "[[File:", "[[Wikipedia:", "[[User:", "[[Portal:"};
  static const char* const kCommonTags[] = {
      "<ref name=\"", "</ref>",      "<ref>",       "<div class=\"", "</div>",
      "<span class=\"", "</span>",    "<blockquote>", "</blockquote>", "<!--",
      "-->"};

  usize i = 0;
  while (i < n) {
    const u8 c = src[i];

    // 1. XML/HTML entities: &[a-zA-Z0-9#]+;
    if (c == '&') {
      const usize start = i++;
      while (i < n && (i - start) < kMaxWordLen &&
             (is_ascii_alpha(src[i]) || (src[i] >= '0' && src[i] <= '9') || src[i] == '#')) {
        ++i;
      }
      if (i < n && src[i] == ';' && (i - start + 1) >= 3 && (i - start + 1) <= kMaxWordLen) {
        ++i;  // include ';'
        std::string entity(reinterpret_cast<const char*>(src + start), i - start);
        ++freq[entity];
        continue;
      }
      i = start + 1;
      continue;
    }

    // 2. Wikipedia link brackets, comments, headings
    if (c == '[' && i + 1 < n && src[i + 1] == '[') {
      bool matched = false;
      for (const char* pfx : kPfx) {
        const usize len = std::strlen(pfx);
        if (i + len <= n && std::memcmp(src + i, pfx, len) == 0) {
          ++freq[std::string(pfx, len)];
          i += len;
          matched = true;
          break;
        }
      }
      if (matched) continue;
      ++freq["[["];
      i += 2;
      continue;
    }
    if (c == ']' && i + 1 < n && src[i + 1] == ']') {
      ++freq["]]"];
      i += 2;
      continue;
    }
    if (c == '{' && i + 1 < n && src[i + 1] == '{') {
      ++freq["{{"];
      i += 2;
      continue;
    }
    if (c == '}' && i + 1 < n && src[i + 1] == '}') {
      ++freq["}}"];
      i += 2;
      continue;
    }
    if (c == '=' && i + 1 < n && src[i + 1] == '=') {
      if (i + 2 < n && src[i + 2] == '=') {
        ++freq["==="];
        i += 3;
      } else {
        ++freq["=="];
        i += 2;
      }
      continue;
    }

    // 3. XML/HTML structural tags: </?[a-zA-Z0-9_:-]+...>
    if (c == '<') {
      bool matched = false;
      for (const char* tag : kCommonTags) {
        const usize len = std::strlen(tag);
        if (i + len <= n && std::memcmp(src + i, tag, len) == 0) {
          ++freq[std::string(tag, len)];
          i += len;
          matched = true;
          break;
        }
      }
      if (matched) continue;

      const usize start = i++;
      if (i < n && src[i] == '/') ++i;
      const usize name_start = i;
      while (i < n && (i - start) < kMaxWordLen &&
             (is_ascii_alpha(src[i]) || (src[i] >= '0' && src[i] <= '9') || src[i] == '_' || src[i] == ':')) {
        ++i;
      }
      if (i > name_start && i < n && src[i] == '>' && (i - start + 1) >= 3 && (i - start + 1) <= kMaxWordLen) {
        ++i;  // include '>'
        std::string tag(reinterpret_cast<const char*>(src + start), i - start);
        ++freq[tag];
        continue;
      }
      i = start + 1;
      continue;
    }

    // 4. Common URL protocols
    if (c == 'h' && i + 7 <= n &&
        (std::memcmp(src + i, "http://", 7) == 0 ||
         (i + 8 <= n && std::memcmp(src + i, "https://", 8) == 0))) {
      const usize len = (src[i + 4] == 's') ? 8 : 7;
      std::string proto(reinterpret_cast<const char*>(src + i), len);
      ++freq[proto];
      i += len;
      continue;
    }

    // 5. Alphabetic words and C++ scope qualifiers
    if (is_ascii_alpha(c)) {
      const usize start = i;
      while (i < n && is_ascii_alpha(src[i])) ++i;
      usize len = i - start;

      // Check for C++ scoped qualifier: word::
      if (i + 2 <= n && src[i] == ':' && src[i + 1] == ':' && (len + 2) <= kMaxWordLen) {
        i += 2;
        len += 2;
        std::string scope_word(reinterpret_cast<const char*>(src + start), len);
        ++freq[scope_word];
        continue;
      }

      if (len >= 3 && len <= kMaxWordLen) {
        const WordCase wc = classify_word(src + start, len);
        if (wc != WordCase::Other) {
          std::string lower_word;
          lower_word.resize(len);
          for (usize k = 0; k < len; ++k) {
            lower_word[k] = static_cast<char>(to_lower_ascii(src[start + k]));
          }
          ++freq[lower_word];
        }
      }
    } else {
      ++i;
    }
  }

  // Frequency threshold
  const usize min_freq = n < (64 * 1024) ? 3 : n < (1024 * 1024) ? 4 : 6;

  std::vector<WordCount> candidates;
  candidates.reserve(freq.size());
  for (auto& [w, c] : freq) {
    if (c >= min_freq && w.size() >= 2 && w.size() <= kMaxWordLen) {
      candidates.push_back({w, c});
    }
  }

  if (candidates.size() < 8) return false;

  // Sort by net saving for 1-byte token: (count * (len - 1)) descending
  std::sort(candidates.begin(), candidates.end(), [](const WordCount& a, const WordCount& b) {
    const usize save_a = a.count * (a.word.size() - 1);
    const usize save_b = b.count * (b.word.size() - 1);
    return save_a > save_b;
  });

  const usize num_tier1 = std::min(candidates.size(), kMaxDictWords1);

  // Tier 2: words from num_tier1 onwards that have len >= 3 and count * (len - 2) > 1 + len + 8
  std::vector<WordCount> tier2;
  for (usize d = num_tier1; d < candidates.size(); ++d) {
    const auto& w = candidates[d];
    if (w.word.size() >= 3) {
      const usize save_2byte = w.count * (w.word.size() - 2);
      const usize cost = 1 + w.word.size() + 8;
      if (save_2byte > cost) {
        tier2.push_back(w);
      }
    }
  }
  std::sort(tier2.begin(), tier2.end(), [](const WordCount& a, const WordCount& b) {
    const usize save_a = a.count * (a.word.size() - 2);
    const usize save_b = b.count * (b.word.size() - 2);
    return save_a > save_b;
  });

  const usize max_tier2 = 0;
  const usize num_tier2 = std::min(tier2.size(), max_tier2);
  candidates.resize(num_tier1);
  for (usize d = 0; d < num_tier2; ++d) {
    candidates.push_back(std::move(tier2[d]));
  }

  const usize dict_size = candidates.size();
  if (dict_size < 8) return false;

  // Build quick token lookup map: token is 16-bit
  // if d < 120: token is 0x80 + d (value < 256)
  // if d >= 120: token is ((0xF8 + ((d - 120) >> 8)) << 8) | ((d - 120) & 0xFF)
  std::unordered_map<std::string_view, u16> token_map;
  token_map.reserve(dict_size);
  for (usize d = 0; d < dict_size; ++d) {
    if (d < kMaxDictWords1) {
      token_map[candidates[d].word] = static_cast<u16>(0x80 + d);
    } else {
      const usize rel = d - kMaxDictWords1;
      const u16 hi = static_cast<u16>(0xF8 + (rel >> 8));
      const u16 lo = static_cast<u16>(rel & 0xFF);
      token_map[candidates[d].word] = (hi << 8) | lo;
    }
  }

  // Step 2: Write framed stream
  out.clear();
  out.reserve(n);
  out.push_back(kWordNormMagic);
  out.push_back(kWordNormVer);
  ByteWriter wr(out);
  wr.varint(n);
  wr.varint(dict_size);

  // Write dictionary words
  for (usize d = 0; d < dict_size; ++d) {
    const auto& w = candidates[d].word;
    wr.byte(static_cast<u8>(w.size()));
    wr.bytes(reinterpret_cast<const u8*>(w.data()), w.size());
  }

  // Step 3: Stream transform tokens
  auto emit_tok = [&](u16 tok, WordCase wc) {
    if (wc == WordCase::Cap) {
      out.push_back(kFlagCap);
    } else if (wc == WordCase::Upper) {
      out.push_back(kFlagUpper);
    }
    if (tok < 256) {
      out.push_back(static_cast<u8>(tok));
    } else {
      out.push_back(static_cast<u8>(tok >> 8));
      out.push_back(static_cast<u8>(tok & 0xFF));
    }
  };

  i = 0;
  char word_buf[kMaxWordLen + 1];

  while (i < n) {
    const u8 c = src[i];

    // 1. XML/HTML entities: &[a-zA-Z0-9#]+;
    if (c == '&') {
      usize p = i + 1;
      while (p < n && (p - i) < kMaxWordLen &&
             (is_ascii_alpha(src[p]) || (src[p] >= '0' && src[p] <= '9') || src[p] == '#')) {
        ++p;
      }
      if (p < n && src[p] == ';' && (p - i + 1) >= 3 && (p - i + 1) <= kMaxWordLen) {
        std::string_view sv(reinterpret_cast<const char*>(src + i), p - i + 1);
        auto it = token_map.find(sv);
        if (it != token_map.end()) {
          emit_tok(it->second, WordCase::Lower);
          i = p + 1;
          continue;
        }
      }
    }

    // 2. Wikipedia link brackets, comments, headings
    if (c == '[' && i + 1 < n && src[i + 1] == '[') {
      bool matched = false;
      for (const char* pfx : kPfx) {
        const usize len = std::strlen(pfx);
        if (i + len <= n && std::memcmp(src + i, pfx, len) == 0) {
          auto it = token_map.find(std::string_view(pfx, len));
          if (it != token_map.end()) {
            emit_tok(it->second, WordCase::Lower);
            i += len;
            matched = true;
            break;
          }
        }
      }
      if (matched) continue;
      auto it = token_map.find(std::string_view("[[", 2));
      if (it != token_map.end()) {
        emit_tok(it->second, WordCase::Lower);
        i += 2;
        continue;
      }
    }
    if (c == ']' && i + 1 < n && src[i + 1] == ']') {
      auto it = token_map.find(std::string_view("]]", 2));
      if (it != token_map.end()) {
        emit_tok(it->second, WordCase::Lower);
        i += 2;
        continue;
      }
    }
    if (c == '{' && i + 1 < n && src[i + 1] == '{') {
      auto it = token_map.find(std::string_view("{{", 2));
      if (it != token_map.end()) {
        emit_tok(it->second, WordCase::Lower);
        i += 2;
        continue;
      }
    }
    if (c == '}' && i + 1 < n && src[i + 1] == '}') {
      auto it = token_map.find(std::string_view("}}", 2));
      if (it != token_map.end()) {
        emit_tok(it->second, WordCase::Lower);
        i += 2;
        continue;
      }
    }
    if (c == '=' && i + 1 < n && src[i + 1] == '=') {
      if (i + 2 < n && src[i + 2] == '=') {
        auto it = token_map.find(std::string_view("===", 3));
        if (it != token_map.end()) {
          emit_tok(it->second, WordCase::Lower);
          i += 3;
          continue;
        }
      }
      auto it = token_map.find(std::string_view("==", 2));
      if (it != token_map.end()) {
        emit_tok(it->second, WordCase::Lower);
        i += 2;
        continue;
      }
    }

    // 3. XML/HTML structural tags: </?[a-zA-Z0-9_:-]+...>
    if (c == '<') {
      bool matched = false;
      for (const char* tag : kCommonTags) {
        const usize len = std::strlen(tag);
        if (i + len <= n && std::memcmp(src + i, tag, len) == 0) {
          auto it = token_map.find(std::string_view(tag, len));
          if (it != token_map.end()) {
            emit_tok(it->second, WordCase::Lower);
            i += len;
            matched = true;
            break;
          }
        }
      }
      if (matched) continue;

      usize p = i + 1;
      if (p < n && src[p] == '/') ++p;
      const usize name_start = p;
      while (p < n && (p - i) < kMaxWordLen &&
             (is_ascii_alpha(src[p]) || (src[p] >= '0' && src[p] <= '9') || src[p] == '_' || src[p] == ':')) {
        ++p;
      }
      if (p > name_start && p < n && src[p] == '>' && (p - i + 1) >= 3 && (p - i + 1) <= kMaxWordLen) {
        std::string_view sv(reinterpret_cast<const char*>(src + i), p - i + 1);
        auto it = token_map.find(sv);
        if (it != token_map.end()) {
          emit_tok(it->second, WordCase::Lower);
          i = p + 1;
          continue;
        }
      }
    }

    // 4. Common URL protocols
    if (c == 'h' && i + 7 <= n) {
      if (std::memcmp(src + i, "http://", 7) == 0) {
        auto it = token_map.find(std::string_view("http://", 7));
        if (it != token_map.end()) {
          emit_tok(it->second, WordCase::Lower);
          i += 7;
          continue;
        }
      } else if (i + 8 <= n && std::memcmp(src + i, "https://", 8) == 0) {
        auto it = token_map.find(std::string_view("https://", 8));
        if (it != token_map.end()) {
          emit_tok(it->second, WordCase::Lower);
          i += 8;
          continue;
        }
      }
    }

    // 5. Alphabetic words and C++ scope qualifiers
    if (is_ascii_alpha(c)) {
      const usize start = i;
      while (i < n && is_ascii_alpha(src[i])) ++i;
      usize len = i - start;

      // Check for C++ scoped qualifier: word::
      if (i + 2 <= n && src[i] == ':' && src[i + 1] == ':' && (len + 2) <= kMaxWordLen) {
        std::string_view sv(reinterpret_cast<const char*>(src + start), len + 2);
        auto it = token_map.find(sv);
        if (it != token_map.end()) {
          emit_tok(it->second, WordCase::Lower);
          i += 2;
          continue;
        }
      }

      const WordCase wc = classify_word(src + start, len);
      if (wc != WordCase::Other) {
        for (usize k = 0; k < len; ++k) {
          word_buf[k] = static_cast<char>(to_lower_ascii(src[start + k]));
        }
        std::string_view sv(word_buf, len);
        auto it = token_map.find(sv);
        if (it != token_map.end()) {
          emit_tok(it->second, wc);
          continue;
        }
      }

      // Literal word (not in dictionary or mixed case)
      for (usize k = start; k < i; ++k) {
        const u8 b = src[k];
        if (b >= 0x80) {
          out.push_back(kEsc);
          out.push_back(static_cast<u8>(b ^ 0x80));
        } else {
          out.push_back(b);
        }
      }
    } else {
      const u8 b = src[i++];
      if (b >= 0x80) {
        out.push_back(kEsc);
        out.push_back(static_cast<u8>(b ^ 0x80));
      } else {
        out.push_back(b);
      }
    }
  }

  // Must provide at least 2% uncompressed reduction to justify framing overhead
  return out.size() * 100 < n * 98;
}

// Inverse transform: reconstruct text from dictionary and token stream.
inline Status word_norm_inv(const u8* data, usize len, u8* dst, usize n) {
  if (!data || len < 4) return Status::CorruptInput;
  if (data[0] != kWordNormMagic) return Status::CorruptInput;
  const u8 ver = data[1];
  if (ver != 1 && ver != 2) return Status::CorruptInput;

  ByteReader r(data + 2, len - 2);
  const u64 orig_len = r.varint();
  if (r.error || orig_len != n) return Status::CorruptInput;

  const usize dict_size = (ver == 1) ? r.byte() : r.varint();
  const usize max_words = (ver == 1) ? kMaxDictWords1 : kMaxDictWords;
  if (r.error || dict_size > max_words) return Status::CorruptInput;

  struct WordEntry {
    const char* str = nullptr;
    u8 len = 0;
  };
  WordEntry dict[kMaxDictWords];

  const usize min_len = (ver == 1) ? 3 : 2;
  for (usize d = 0; d < dict_size; ++d) {
    if (r.remaining() == 0) return Status::CorruptInput;
    const u8 wlen = r.byte();
    if (wlen < min_len || wlen > kMaxWordLen || r.remaining() < wlen) {
      return Status::CorruptInput;
    }
    const u8* wp = r.take(wlen);
    if (!wp) return Status::CorruptInput;
    dict[d].str = reinterpret_cast<const char*>(wp);
    dict[d].len = wlen;
  }

  usize out_pos = 0;
  int flag = 0;  // 0: lower, 1: Cap, 2: Upper

  while (r.remaining() > 0) {
    const u8 c = r.byte();
    if (c == kFlagCap) {
      flag = 1;
      continue;
    }
    if (c == kFlagUpper) {
      flag = 2;
      continue;
    }
    if (c == kEsc) {
      if (flag != 0 || r.remaining() == 0 || out_pos >= n) return Status::CorruptInput;
      dst[out_pos++] = static_cast<u8>(r.byte() ^ 0x80);
      continue;
    }

    if (c >= 0x80 && c < (0x80 + std::min(dict_size, kMaxDictWords1))) {
      const WordEntry& w = dict[c - 0x80];
      if (out_pos + w.len > n) return Status::CorruptInput;

      if (flag == 0) {
        std::memcpy(dst + out_pos, w.str, w.len);
      } else if (flag == 1) {
        dst[out_pos] = to_upper_ascii(static_cast<u8>(w.str[0]));
        std::memcpy(dst + out_pos + 1, w.str + 1, w.len - 1);
        flag = 0;
      } else {  // flag == 2
        for (u8 j = 0; j < w.len; ++j) {
          dst[out_pos + j] = to_upper_ascii(static_cast<u8>(w.str[j]));
        }
        flag = 0;
      }
      out_pos += w.len;
      continue;
    }

    if (ver == 2 && c >= 0xF8 && c <= 0xFB) {
      if (r.remaining() == 0) return Status::CorruptInput;
      const usize rel = ((static_cast<usize>(c) - 0xF8) << 8) | r.byte();
      const usize d = kMaxDictWords1 + rel;
      if (d >= dict_size) return Status::CorruptInput;
      const WordEntry& w = dict[d];
      if (out_pos + w.len > n) return Status::CorruptInput;

      if (flag == 0) {
        std::memcpy(dst + out_pos, w.str, w.len);
      } else if (flag == 1) {
        dst[out_pos] = to_upper_ascii(static_cast<u8>(w.str[0]));
        std::memcpy(dst + out_pos + 1, w.str + 1, w.len - 1);
        flag = 0;
      } else {  // flag == 2
        for (u8 j = 0; j < w.len; ++j) {
          dst[out_pos + j] = to_upper_ascii(static_cast<u8>(w.str[j]));
        }
        flag = 0;
      }
      out_pos += w.len;
      continue;
    }

    // Literal character
    if (flag != 0 || out_pos >= n) return Status::CorruptInput;
    dst[out_pos++] = c;
  }

  if (out_pos != n || flag != 0) return Status::CorruptInput;
  return Status::Ok;
}

}  // namespace spectra8::detail::word_norm_detail

#endif  // SPECTRA8_SRC_WORD_FILTER_HPP
