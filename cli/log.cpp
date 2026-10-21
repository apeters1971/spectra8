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

// Line-oriented Spectra8 logfile I/O and `spectra8 tail`.
#include "log.hpp"

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <signal.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace spectra8_cli {
namespace {

constexpr char kB64[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

int b64_val(char c) {
  if (c >= 'A' && c <= 'Z') return c - 'A';
  if (c >= 'a' && c <= 'z') return c - 'a' + 26;
  if (c >= '0' && c <= '9') return c - '0' + 52;
  if (c == '+') return 62;
  if (c == '/') return 63;
  return -1;
}

bool read_line(FILE* f, std::string& line) {
  line.clear();
  int c = 0;
  bool any = false;
  while ((c = std::getc(f)) != EOF) {
    any = true;
    if (c == '\n') {
      if (!line.empty() && line.back() == '\r') line.pop_back();
      return true;
    }
    line.push_back(static_cast<char>(c));
  }
  if (any && !line.empty() && line.back() == '\r') line.pop_back();
  return any;
}

bool write_all(FILE* f, const void* p, std::size_t n) {
  return std::fwrite(p, 1, n, f) == n;
}

bool pid_alive(int pid) {
  if (pid <= 0) return true;
  if (::kill(pid, 0) == 0) return true;
  return errno != ESRCH;
}

void sleep_s(double s) {
  if (s < 0) s = 0;
  auto usec = static_cast<unsigned int>(s * 1e6);
  if (usec == 0) usec = 1;
  ::usleep(usec);
}

bool file_id(const std::string& path, dev_t* dev, ino_t* ino, off_t* sz) {
  struct stat st;
  if (::stat(path.c_str(), &st) != 0) return false;
  if (dev) *dev = st.st_dev;
  if (ino) *ino = st.st_ino;
  if (sz) *sz = st.st_size;
  return true;
}

bool seekable(FILE* f) {
  return f && ::fseeko(f, 0, SEEK_CUR) == 0;
}

off_t file_size(FILE* f) {
  if (!f) return -1;
  const off_t cur = ::ftello(f);
  if (cur < 0) return -1;
  if (::fseeko(f, 0, SEEK_END) != 0) return -1;
  const off_t end = ::ftello(f);
  ::fseeko(f, cur, SEEK_SET);
  return end;
}

int last_n_lines(FILE* f, std::int64_t n, std::vector<std::string>& lines) {
  lines.clear();
  if (n <= 0) return 0;
  if (!seekable(f)) {
    std::string line;
    while (read_line(f, line)) {
      lines.push_back(std::move(line));
      if (static_cast<std::int64_t>(lines.size()) > n)
        lines.erase(lines.begin());
    }
    return std::ferror(f) ? 1 : 0;
  }
  const off_t end = file_size(f);
  if (end < 0) return 1;
  if (end == 0) return 0;

  bool ends_nl = false;
  if (::fseeko(f, end - 1, SEEK_SET) != 0) return 1;
  const int last = std::getc(f);
  if (last == EOF) return 1;
  ends_nl = (last == '\n');
  const off_t work_end = ends_nl ? end - 1 : end;

  off_t start = 0;
  if (work_end > 0) {
    const off_t ch = 8192;
    std::vector<char> buf(static_cast<std::size_t>(ch));
    off_t pos = work_end;
    std::int64_t seen = 0;
    bool found = false;
    while (pos > 0 && !found) {
      const off_t step = pos > ch ? ch : pos;
      pos -= step;
      if (::fseeko(f, pos, SEEK_SET) != 0) return 1;
      if (std::fread(buf.data(), 1, static_cast<std::size_t>(step), f) !=
          static_cast<std::size_t>(step))
        return 1;
      for (off_t i = step - 1; i >= 0; --i) {
        if (buf[static_cast<std::size_t>(i)] != '\n') continue;
        ++seen;
        if (seen == n) {
          start = pos + i + 1;
          found = true;
          break;
        }
      }
    }
    if (!found) start = 0;
  }

  if (::fseeko(f, start, SEEK_SET) != 0) return 1;
  std::string line;
  while (read_line(f, line)) lines.push_back(std::move(line));
  return std::ferror(f) ? 1 : 0;
}

int from_line_n(FILE* f, std::int64_t n, std::vector<std::string>& lines) {
  lines.clear();
  if (n <= 0) n = 1;
  std::string line;
  std::int64_t i = 1;
  while (read_line(f, line)) {
    if (i >= n) lines.push_back(std::move(line));
    ++i;
  }
  return std::ferror(f) ? 1 : 0;
}

int last_n_bytes_lines(FILE* f, std::int64_t n, std::vector<std::string>& lines) {
  lines.clear();
  if (n <= 0) return 0;
  if (!seekable(f)) {
    std::string acc;
    char buf[4096];
    for (;;) {
      const std::size_t got = std::fread(buf, 1, sizeof buf, f);
      if (got) acc.append(buf, got);
      if (got < sizeof buf) break;
    }
    if (std::ferror(f)) return 1;
    if (static_cast<std::int64_t>(acc.size()) > n)
      acc.erase(0, acc.size() - static_cast<std::size_t>(n));
    std::size_t i = 0;
    if (!acc.empty() && acc[0] != '\n') {
      const auto nl = acc.find('\n');
      if (nl != std::string::npos) i = nl + 1;
    }
    std::string line;
    for (; i < acc.size(); ++i) {
      if (acc[i] == '\n') {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        lines.push_back(std::move(line));
        line.clear();
      } else {
        line.push_back(acc[i]);
      }
    }
    if (!line.empty()) {
      if (line.back() == '\r') line.pop_back();
      lines.push_back(std::move(line));
    }
    return 0;
  }
  const off_t end = file_size(f);
  if (end < 0) return 1;
  off_t start = end > n ? end - n : 0;
  if (::fseeko(f, start, SEEK_SET) != 0) return 1;
  if (start > 0) {
    std::string skip;
    if (!read_line(f, skip)) return std::ferror(f) ? 1 : 0;
  }
  std::string line;
  while (read_line(f, line)) lines.push_back(std::move(line));
  return std::ferror(f) ? 1 : 0;
}

int from_byte_n(FILE* f, std::int64_t n, std::vector<std::string>& lines) {
  lines.clear();
  if (n <= 0) n = 1;
  const off_t start = static_cast<off_t>(n - 1);
  if (seekable(f)) {
    const off_t end = file_size(f);
    if (end < 0) return 1;
    if (start >= end) return 0;
    if (::fseeko(f, start, SEEK_SET) != 0) return 1;
    if (start > 0) {
      std::string skip;
      if (!read_line(f, skip)) return std::ferror(f) ? 1 : 0;
    }
  } else {
    off_t skip = start;
    while (skip > 0) {
      const int c = std::getc(f);
      if (c == EOF) return std::ferror(f) ? 1 : 0;
      --skip;
    }
    if (start > 0) {
      std::string dump;
      if (!read_line(f, dump)) return std::ferror(f) ? 1 : 0;
    }
  }
  std::string line;
  while (read_line(f, line)) lines.push_back(std::move(line));
  return std::ferror(f) ? 1 : 0;
}

bool parse_count(const char* s, std::int64_t* n, bool* from_start) {
  if (!s || !*s) return false;
  *from_start = (*s == '+');
  if (*s == '+' || *s == '-') ++s;
  if (!*s) return false;
  char* end = nullptr;
  errno = 0;
  const long long v = std::strtoll(s, &end, 10);
  if (errno != 0 || end == s || *end != '\0' || v < 0) return false;
  *n = static_cast<std::int64_t>(v);
  return true;
}

int emit_decoded_lines(const std::vector<std::string>& encoded, FILE* out) {
  for (const auto& e : encoded) {
    if (e.empty()) continue;
    std::string line;
    if (!decode_log_line(e, line)) {
      std::fprintf(stderr, "spectra8 tail: bad frame line\n");
      return 1;
    }
    if (!write_all(out, line.data(), line.size()) || std::fputc('\n', out) == EOF)
      return 1;
  }
  if (std::fflush(out) != 0) return 1;
  return 0;
}

struct FollowFile {
  std::string path;
  FILE* f = nullptr;
  bool own = false;
  bool is_stdin = false;
  dev_t dev = 0;
  ino_t ino = 0;
};

void close_follow(FollowFile& ff) {
  if (ff.own && ff.f) std::fclose(ff.f);
  ff.f = nullptr;
  ff.own = false;
}

FILE* open_named(const std::string& path, FollowFile* ff) {
  if (is_stdin_path(path)) {
    if (ff) {
      ff->f = stdin;
      ff->own = false;
      ff->is_stdin = true;
    }
    return stdin;
  }
  FILE* f = std::fopen(path.c_str(), "rb");
  if (f && ff) {
    ff->f = f;
    ff->own = true;
    ff->is_stdin = false;
    file_id(path, &ff->dev, &ff->ino, nullptr);
  }
  return f;
}

int follow_new_lines(FollowFile& ff, FILE* out) {
  if (!ff.f) return 0;
  std::string line;
  while (read_line(ff.f, line)) {
    if (line.empty()) continue;
    std::string text;
    if (!decode_log_line(line, text)) {
      std::fprintf(stderr, "spectra8 tail: bad frame line\n");
      return 1;
    }
    if (!write_all(out, text.data(), text.size()) || std::fputc('\n', out) == EOF)
      return 1;
  }
  if (std::ferror(ff.f)) return 1;
  if (std::fflush(out) != 0) return 1;
  return 0;
}

bool print_header(const TailOptions& opt, const std::string& path, bool* first) {
  const bool many = opt.files.size() > 1;
  if (opt.quiet) return false;
  if (!opt.verbose && !many) return false;
  if (!*first) std::fputc('\n', stdout);
  *first = false;
  std::fprintf(stdout, "==> %s <==\n",
               is_stdin_path(path) ? "standard input" : path.c_str());
  return true;
}

}  // namespace

bool is_stdin_path(const std::string& path) noexcept {
  return path.empty() || path == "-" || path == "/dev/stdin" ||
         path == "/dev/fd/0";
}

bool is_stdout_path(const std::string& path) noexcept {
  return path == "-" || path == "/dev/stdout" || path == "/dev/fd/1";
}

std::string base64_encode(std::span<const std::byte> src) {
  std::string out;
  out.resize(((src.size() + 2) / 3) * 4);
  std::size_t o = 0;
  std::size_t i = 0;
  for (; i + 3 <= src.size(); i += 3) {
    const unsigned v = (static_cast<unsigned>(src[i]) << 16) |
                       (static_cast<unsigned>(src[i + 1]) << 8) |
                       static_cast<unsigned>(src[i + 2]);
    out[o++] = kB64[(v >> 18) & 63];
    out[o++] = kB64[(v >> 12) & 63];
    out[o++] = kB64[(v >> 6) & 63];
    out[o++] = kB64[v & 63];
  }
  if (i < src.size()) {
    unsigned v = static_cast<unsigned>(src[i]) << 16;
    if (i + 1 < src.size()) v |= static_cast<unsigned>(src[i + 1]) << 8;
    out[o++] = kB64[(v >> 18) & 63];
    out[o++] = kB64[(v >> 12) & 63];
    if (i + 1 < src.size()) {
      out[o++] = kB64[(v >> 6) & 63];
      out[o++] = '=';
    } else {
      out[o++] = '=';
      out[o++] = '=';
    }
  }
  out.resize(o);
  return out;
}

bool base64_decode(std::string_view src, std::vector<std::byte>& dst) {
  dst.clear();
  int val[4];
  int n = 0;
  auto flush = [&]() -> bool {
    if (n < 2) return n == 0;
    const unsigned v = (static_cast<unsigned>(val[0]) << 18) |
                       (static_cast<unsigned>(val[1]) << 12) |
                       (n > 2 ? static_cast<unsigned>(val[2]) << 6 : 0) |
                       (n > 3 ? static_cast<unsigned>(val[3]) : 0);
    dst.push_back(static_cast<std::byte>((v >> 16) & 0xFF));
    if (n > 2) dst.push_back(static_cast<std::byte>((v >> 8) & 0xFF));
    if (n > 3) dst.push_back(static_cast<std::byte>(v & 0xFF));
    n = 0;
    return true;
  };
  for (char c : src) {
    if (c == ' ' || c == '\t' || c == '\n' || c == '\r') continue;
    if (c == '=') {
      if (!flush()) return false;
      return true;
    }
    const int d = b64_val(c);
    if (d < 0) return false;
    val[n++] = d;
    if (n == 4 && !flush()) return false;
  }
  return flush();
}

bool encode_log_line(std::string_view line, std::string& encoded,
                     const spectra8::CompressParams& cp) {
  const auto* p = reinterpret_cast<const std::byte*>(line.data());
  std::span<const std::byte> src(p, line.size());
  std::vector<std::byte> frame(spectra8::compress_bound(src.size(), cp.codec));
  if (frame.empty())
    frame.resize(spectra8::compress_bound(src.size()));
  auto r = spectra8::compress(src, frame, cp);
  if (!r) return false;
  encoded = base64_encode({frame.data(), r.value});
  return true;
}

bool decode_log_line(std::string_view encoded, std::string& line) {
  std::vector<std::byte> frame;
  if (!base64_decode(encoded, frame)) return false;
  auto cs = spectra8::get_frame_content_size(frame);
  if (!cs) return false;
  std::vector<std::byte> out(static_cast<std::size_t>(*cs));
  auto r = spectra8::decompress(frame, out);
  if (!r || r.value != out.size()) return false;
  line.assign(reinterpret_cast<const char*>(out.data()), out.size());
  return true;
}

int encode_logfile(FILE* in, FILE* out, const spectra8::CompressParams& cp,
                   std::uint64_t* lines, std::uint64_t* in_bytes,
                   std::uint64_t* out_bytes) {
  if (!in || !out) return 1;
  std::uint64_t n = 0, ib = 0, ob = 0;
  std::string line, encoded;
  while (read_line(in, line)) {
    if (!encode_log_line(line, encoded, cp)) {
      std::fprintf(stderr, "spectra8: logfile encode failed\n");
      return 1;
    }
    if (!write_all(out, encoded.data(), encoded.size()) ||
        std::fputc('\n', out) == EOF) {
      std::fprintf(stderr, "spectra8: logfile write failed\n");
      return 1;
    }
    std::fflush(out);
    ++n;
    ib += line.size();
    ob += encoded.size() + 1;
  }
  if (std::ferror(in)) return 1;
  if (lines) *lines = n;
  if (in_bytes) *in_bytes = ib;
  if (out_bytes) *out_bytes = ob;
  return 0;
}

int decode_logfile(FILE* in, FILE* out, std::uint64_t* lines,
                   std::uint64_t* in_bytes, std::uint64_t* out_bytes) {
  if (!in || !out) return 1;
  std::uint64_t n = 0, ib = 0, ob = 0;
  std::string encoded, line;
  while (read_line(in, encoded)) {
    if (encoded.empty()) continue;
    if (!decode_log_line(encoded, line)) {
      std::fprintf(stderr, "spectra8: logfile decode failed (bad frame line)\n");
      return 1;
    }
    if (!write_all(out, line.data(), line.size()) || std::fputc('\n', out) == EOF) {
      std::fprintf(stderr, "spectra8: logfile write failed\n");
      return 1;
    }
    std::fflush(out);
    ++n;
    ib += encoded.size() + 1;
    ob += line.size() + 1;
  }
  if (std::ferror(in)) return 1;
  if (lines) *lines = n;
  if (in_bytes) *in_bytes = ib;
  if (out_bytes) *out_bytes = ob;
  return 0;
}

int tail_select_encoded(FILE* in, const TailOptions& opt,
                        std::vector<std::string>& lines) {
  if (!in) return 1;
  if (opt.from_start) {
    if (opt.by_bytes) return from_byte_n(in, opt.count, lines);
    return from_line_n(in, opt.count, lines);
  }
  if (opt.by_bytes) return last_n_bytes_lines(in, opt.count, lines);
  return last_n_lines(in, opt.count, lines);
}

void print_tail_usage(FILE* f) {
  std::fprintf(
      f,
      "Usage: spectra8 tail [options] [file ...]\n"
      "  Decode a --logfile stream (one Spectra8 frame per line).\n"
      "  No file, '-', or /dev/stdin reads a pipe / stdin.\n"
      "\n"
      "  -n, --lines=[+]N   last N lines (default 10); +N from line N\n"
      "  -c, --bytes=[+]N   last N bytes of the encoded file; +N from byte N\n"
      "  -f, --follow[=descriptor|name]\n"
      "                     output appended data as the file grows\n"
      "  -F                 same as --follow=name --retry\n"
      "  -q, --quiet        never print headers giving file names\n"
      "  -v, --verbose      always print headers giving file names\n"
      "  -s, --sleep-interval=S\n"
      "                     with -f, sleep S seconds between samples (default 1)\n"
      "  --pid=PID          with -f, exit when PID dies\n"
      "  --retry            keep trying to open a file that is missing\n"
      "  -h, --help         show this help\n");
}

int parse_tail_args(int argc, char** argv, TailOptions& opt) {
  for (int i = 2; i < argc; ++i) {
    const char* s = argv[i];
    if (s[0] != '-' || s[1] == '\0') {
      opt.files.emplace_back(s);
      continue;
    }
    if (s[1] == '-') {
      const char* longv = s + 2;
      if (std::strcmp(longv, "help") == 0) return 2;
      if (std::strcmp(longv, "quiet") == 0 || std::strcmp(longv, "silent") == 0)
        opt.quiet = true;
      else if (std::strcmp(longv, "verbose") == 0)
        opt.verbose = true;
      else if (std::strcmp(longv, "retry") == 0)
        opt.retry = true;
      else if (std::strcmp(longv, "follow") == 0) {
        opt.follow = true;
      } else if (std::strncmp(longv, "follow=", 7) == 0) {
        opt.follow = true;
        if (std::strcmp(longv + 7, "name") == 0)
          opt.follow_name = true;
        else if (std::strcmp(longv + 7, "descriptor") != 0)
          return -1;
      } else if (std::strncmp(longv, "lines=", 6) == 0) {
        if (!parse_count(longv + 6, &opt.count, &opt.from_start)) return -1;
        opt.by_bytes = false;
      } else if (std::strcmp(longv, "lines") == 0 && i + 1 < argc) {
        if (!parse_count(argv[++i], &opt.count, &opt.from_start)) return -1;
        opt.by_bytes = false;
      } else if (std::strncmp(longv, "bytes=", 6) == 0) {
        if (!parse_count(longv + 6, &opt.count, &opt.from_start)) return -1;
        opt.by_bytes = true;
      } else if (std::strcmp(longv, "bytes") == 0 && i + 1 < argc) {
        if (!parse_count(argv[++i], &opt.count, &opt.from_start)) return -1;
        opt.by_bytes = true;
      } else if (std::strncmp(longv, "sleep-interval=", 15) == 0) {
        opt.sleep_s = std::atof(longv + 15);
      } else if (std::strcmp(longv, "sleep-interval") == 0 && i + 1 < argc) {
        opt.sleep_s = std::atof(argv[++i]);
      } else if (std::strncmp(longv, "pid=", 4) == 0) {
        opt.pid = std::atoi(longv + 4);
      } else if (std::strcmp(longv, "pid") == 0 && i + 1 < argc) {
        opt.pid = std::atoi(argv[++i]);
      } else
        return -1;
      continue;
    }
    if (s[1] >= '0' && s[1] <= '9') {
      if (!parse_count(s + 1, &opt.count, &opt.from_start)) return -1;
      opt.by_bytes = false;
      continue;
    }
    for (std::size_t j = 1; s[j]; ++j) {
      const char c = s[j];
      if (c == 'h') return 2;
      if (c == 'q') opt.quiet = true;
      else if (c == 'v') opt.verbose = true;
      else if (c == 'f') opt.follow = true;
      else if (c == 'F') {
        opt.follow = true;
        opt.follow_name = true;
        opt.retry = true;
      } else if (c == 'n' || c == 'c') {
        opt.by_bytes = (c == 'c');
        const char* rest = s + j + 1;
        if (*rest) {
          if (!parse_count(rest, &opt.count, &opt.from_start)) return -1;
          break;
        }
        if (i + 1 >= argc) return -1;
        if (!parse_count(argv[++i], &opt.count, &opt.from_start)) return -1;
        break;
      } else if (c == 's') {
        const char* rest = s + j + 1;
        if (*rest) {
          opt.sleep_s = std::atof(rest);
          break;
        }
        if (i + 1 >= argc) return -1;
        opt.sleep_s = std::atof(argv[++i]);
        break;
      } else
        return -1;
    }
  }
  if (opt.files.empty()) opt.files.emplace_back("-");
  return 0;
}

int run_tail(const TailOptions& opt) {
  std::vector<FollowFile> files(opt.files.size());
  bool first_header = true;
  for (std::size_t i = 0; i < opt.files.size(); ++i) {
    files[i].path = opt.files[i];
    FILE* f = open_named(opt.files[i], &files[i]);
    while (!f && opt.retry && opt.follow) {
      if (!pid_alive(opt.pid)) return 0;
      sleep_s(opt.sleep_s);
      f = open_named(opt.files[i], &files[i]);
    }
    if (!f) {
      std::fprintf(stderr, "spectra8 tail: cannot open '%s'\n",
                   opt.files[i].c_str());
      return 1;
    }
    print_header(opt, opt.files[i], &first_header);
    std::vector<std::string> encoded;
    if (tail_select_encoded(f, opt, encoded) != 0) {
      std::fprintf(stderr, "spectra8 tail: read failed '%s'\n",
                   opt.files[i].c_str());
      return 1;
    }
    if (emit_decoded_lines(encoded, stdout) != 0) return 1;
  }

  if (!opt.follow) {
    for (auto& ff : files) close_follow(ff);
    return 0;
  }

  for (;;) {
    if (!pid_alive(opt.pid)) break;
    bool any_live = false;
    for (auto& ff : files) {
      if (ff.is_stdin) {
        if (std::feof(ff.f) && !opt.retry) continue;
        if (follow_new_lines(ff, stdout) != 0) return 1;
        if (!std::feof(ff.f) || opt.retry) any_live = true;
        continue;
      }
      if (ff.f) std::clearerr(ff.f);
      any_live = true;
      if (opt.follow_name || opt.retry) {
        dev_t dev = 0;
        ino_t ino = 0;
        off_t sz = 0;
        if (!file_id(ff.path, &dev, &ino, &sz)) {
          if (opt.retry) {
            close_follow(ff);
            continue;
          }
          std::fprintf(stderr, "spectra8 tail: '%s' vanished\n",
                       ff.path.c_str());
          return 1;
        }
        if (!ff.f || dev != ff.dev || ino != ff.ino) {
          close_follow(ff);
          if (!open_named(ff.path, &ff)) {
            if (opt.retry) continue;
            return 1;
          }
        } else if (ff.f) {
          const off_t cur = ::ftello(ff.f);
          if (cur >= 0 && sz < cur) ::fseeko(ff.f, 0, SEEK_SET);
        }
      }
      if (ff.f && follow_new_lines(ff, stdout) != 0) return 1;
    }
    if (!any_live) break;
    sleep_s(opt.sleep_s);
  }
  for (auto& ff : files) close_follow(ff);
  return 0;
}

}  // namespace spectra8_cli
