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

/* spectra8 - C ABI for the shared library.
 *
 * Plain C wrapper around the C++ implementation. Suitable for FFI / linking
 * against the shared library from other languages.
 */
#ifndef SPECTRA8_SPECTRA8_C_H
#define SPECTRA8_SPECTRA8_C_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum spectra8_status {
  SPECTRA8_OK = 0,
  SPECTRA8_BUFFER_TOO_SMALL = 1,
  SPECTRA8_BAD_MAGIC = 2,
  SPECTRA8_UNSUPPORTED_VERSION = 3,
  SPECTRA8_CORRUPT_INPUT = 4,
  SPECTRA8_CHECKSUM_MISMATCH = 5,
  SPECTRA8_INVALID_ARGUMENT = 6,
  SPECTRA8_INTERNAL_ERROR = 7
} spectra8_status;

typedef enum spectra8_codec {
  SPECTRA8_CODEC_BEAM = 0,
  SPECTRA8_CODEC_LZ = 0, /* legacy name for BEAM */
  SPECTRA8_CODEC_PRISM = 1,
  SPECTRA8_CODEC_BWT = 1, /* legacy name for PRISM */
  SPECTRA8_CODEC_ZSTD = 2,
  SPECTRA8_CODEC_BZIP3 = 3,
  SPECTRA8_CODEC_LZMA = 4,
  SPECTRA8_CODEC_AUTO = 5,
  SPECTRA8_CODEC_MOSAIC = 6,
  SPECTRA8_CODEC_ORACLE = 7,
  SPECTRA8_CODEC_CRYSTAL = 8,
  SPECTRA8_CODEC_FLASH = 9,
  SPECTRA8_CODEC_LASER = 10,
  SPECTRA8_CODEC_GRATING = 11
} spectra8_codec;

typedef enum spectra8_profile {
  SPECTRA8_PROFILE_NONE = 0,
  SPECTRA8_PROFILE_FLOAT_KINEMATICS = 1,
  SPECTRA8_PROFILE_POISSON_COUNTS = 2,
  SPECTRA8_PROFILE_INT_MONOTONIC = 3,
  SPECTRA8_PROFILE_TEXT = 4,
  SPECTRA8_PROFILE_CUSTOM = 5
} spectra8_profile;

#define SPECTRA8_MIN_LEVEL 0
#define SPECTRA8_MAX_LEVEL 11
#define SPECTRA8_DEFAULT_LEVEL 6

typedef struct spectra8_compress_options {
  spectra8_codec codec; /* default SPECTRA8_CODEC_BEAM */
  int level;             /* default SPECTRA8_DEFAULT_LEVEL */
  int block_log;         /* 0 => library default */
  /* Auto only: % size gain a slower decoder needs per doubling of decode
   * time (see CompressParams::auto_min_gain_pct). Zero-init = 0 (always
   * take the smallest). Set to 1 to match the C++ default. */
  int auto_min_gain_pct;
  int filters;           /* 0/1; L4EF wrapper for externals, Spectra8 blocks for
                          * Beam/Prism/Mosaic/Oracle/Crystal */
  int filter_fallback;   /* 0/1, default 1 */
  int filter_rle;        /* 0/1, default 1 */
  int filter_dict;       /* 0/1, default 1 */
  int filter_sample_full; /* 0/1; detect over whole block (default 16 KiB mid-sample) */
  /* Auto only: 0/1; race each codec at its max effort (Beam/Prism/Mosaic/Crystal 11, Zstd max,
   * Lzma 9, Bzip3 block_log 28). Zero-init = 0 (use `level` for all).
   * Appended for ABI stability with older zero-init clients. */
  int auto_max_level;
  /* Auto only: bitmask of codecs to race, bit = 1u << spectra8_codec.
   * 0 (zero-init) = all available (SPECTRA8_AUTO_CODECS_ALL).
   * SPECTRA8_AUTO_CODECS_ROOT = zstd | beam | crystal is the recommended
   * set for RNTuple pages (fast decoders only). */
  unsigned auto_codecs;
  /* With filters: bitmask of filter ids detection may nominate, bit = 1u <<
   * filter id (0 none, 1-3 shuffle16/32/64, 4-6 delta16/32/64, 7-9
   * zigzag_delta16/32/64, 10 delta32+shuffle32, 11 zigzag_delta32+shuffle32,
   * 12 bitshuffle32, 13/14 dict32/64, 15 delta64+shuffle64, 16/17
   * zigzag32/64+shuffle, 18 bitshuffle64, 19 delta32+bitshuffle32,
   * 20 dict16, 21/22/23 for16/32/64, 24/25 alp32/64,
   * 26/27 xorshuf32/64, 28/29 delta2/zzd2shuf32, 30 delta2shuf64, 31 bcj_x86,
   * 32 digit_norm, 33 word_norm, 34 base64, 35/36 bitround32/64,
   * 37/38 lag2shuf32/64, 39 xorlag2shuf32, 40 bcj_arm64).
   * 0 (zero-init) = all (SPECTRA8_FILTERS_ALL). */
  unsigned long long filter_mask;
  /* 0/1; PAQ-style context cost modeling / evaluation in Beam Ultra and deep search */
  int context_oracle;
  /* Static probability profile for Crystal block coder (0 = dynamic / none) */
  spectra8_profile profile;
  unsigned profile_id;       /* 0 defaults to (unsigned)profile */
} spectra8_compress_options;

#define SPECTRA8_CODEC_BIT(c) (1u << (unsigned)(c))
#define SPECTRA8_AUTO_CODECS_ALL                                             \
  (SPECTRA8_CODEC_BIT(SPECTRA8_CODEC_ZSTD) |                                \
   SPECTRA8_CODEC_BIT(SPECTRA8_CODEC_BEAM) |                                \
   SPECTRA8_CODEC_BIT(SPECTRA8_CODEC_CRYSTAL) |                             \
   SPECTRA8_CODEC_BIT(SPECTRA8_CODEC_MOSAIC) |                              \
   SPECTRA8_CODEC_BIT(SPECTRA8_CODEC_PRISM) |                               \
   SPECTRA8_CODEC_BIT(SPECTRA8_CODEC_LASER) |                               \
   SPECTRA8_CODEC_BIT(SPECTRA8_CODEC_LZMA) |                                \
   SPECTRA8_CODEC_BIT(SPECTRA8_CODEC_BZIP3))
#define SPECTRA8_AUTO_CODECS_ROOT                                            \
  (SPECTRA8_CODEC_BIT(SPECTRA8_CODEC_ZSTD) |                                \
   SPECTRA8_CODEC_BIT(SPECTRA8_CODEC_BEAM) |                                \
   SPECTRA8_CODEC_BIT(SPECTRA8_CODEC_CRYSTAL))
#define SPECTRA8_FILTER_BIT(id) (1ull << (unsigned)(id))
#define SPECTRA8_FILTERS_ALL 0xFFFFFFFFFFFFFFFFull
/* Element-width subsets (match spectra8::kFilters16/32/64). */
#define SPECTRA8_FILTERS_16                                                  \
  (SPECTRA8_FILTER_BIT(1) | SPECTRA8_FILTER_BIT(4) | SPECTRA8_FILTER_BIT(7) | \
   SPECTRA8_FILTER_BIT(20) | SPECTRA8_FILTER_BIT(21))
#define SPECTRA8_FILTERS_32                                                  \
  (SPECTRA8_FILTER_BIT(2) | SPECTRA8_FILTER_BIT(5) | SPECTRA8_FILTER_BIT(8) | \
   SPECTRA8_FILTER_BIT(10) | SPECTRA8_FILTER_BIT(11) | SPECTRA8_FILTER_BIT(12) | \
   SPECTRA8_FILTER_BIT(13) | SPECTRA8_FILTER_BIT(16) | SPECTRA8_FILTER_BIT(19) | \
   SPECTRA8_FILTER_BIT(22) | SPECTRA8_FILTER_BIT(24) |                      \
   SPECTRA8_FILTER_BIT(26) | SPECTRA8_FILTER_BIT(28) | SPECTRA8_FILTER_BIT(29) | \
   SPECTRA8_FILTER_BIT(37) | SPECTRA8_FILTER_BIT(39))
#define SPECTRA8_FILTERS_64                                                  \
  (SPECTRA8_FILTER_BIT(3) | SPECTRA8_FILTER_BIT(6) | SPECTRA8_FILTER_BIT(9) | \
   SPECTRA8_FILTER_BIT(14) | SPECTRA8_FILTER_BIT(15) | SPECTRA8_FILTER_BIT(17) | \
   SPECTRA8_FILTER_BIT(18) | SPECTRA8_FILTER_BIT(23) | SPECTRA8_FILTER_BIT(25) | \
   SPECTRA8_FILTER_BIT(27) | SPECTRA8_FILTER_BIT(30) | SPECTRA8_FILTER_BIT(38))

/* Upper bound on the compressed size for an input of `src_size` bytes. */
size_t spectra8_compress_bound(size_t src_size);

/* Tighter bound for a specific codec; 0 if that codec is unavailable. */
size_t spectra8_compress_bound_codec(size_t src_size, spectra8_codec codec);

/* 1 if the library can compress/decompress with `codec`, else 0. */
int spectra8_codec_available(spectra8_codec codec);

/* Compress `src_size` bytes from `src` into `dst` (capacity `dst_capacity`)
 * with Codec::Beam. On success returns SPECTRA8_OK and stores the written size
 * in `*written`. */
spectra8_status spectra8_compress(const void* src, size_t src_size,
                                    void* dst, size_t dst_capacity,
                                    int level, size_t* written);

/* Compress with an explicit codec (Beam/Prism/Mosaic/Oracle/Crystal/Zstd/Bzip3/Lzma/Auto). */
spectra8_status spectra8_compress_ex(const void* src, size_t src_size,
                                       void* dst, size_t dst_capacity,
                                       spectra8_codec codec, int level,
                                       size_t* written);

/* Compress with full options (codec, filters, auto_min_gain_pct, …).
 * Zero-initialize `options`; most defaults match C++ CompressParams, except
 * auto_min_gain_pct is 0 (see field comment). */
spectra8_status spectra8_compress_with_options(
    const void* src, size_t src_size, void* dst, size_t dst_capacity,
    const spectra8_compress_options* options, size_t* written);

/* Decompress a complete frame (Spectra8 / L4EF / zstd / bzip3 / xz, sniffed by
 * magic). On success returns SPECTRA8_OK and stores the written size in
 * `*written`. */
spectra8_status spectra8_decompress(const void* src, size_t src_size,
                                      void* dst, size_t dst_capacity,
                                      size_t* written);

/* Returns 1 and stores the recorded content size in `*size` if present /
 * recoverable, otherwise returns 0. */
int spectra8_frame_content_size(const void* src, size_t src_size,
                                 uint64_t* size);

/* Human-readable message for a status code. */
const char* spectra8_status_message(spectra8_status s);

/* Library version string, e.g. "2.0.0". */
const char* spectra8_version(void);

/* Thread-local default profile for integrations (e.g. ROOT RNTuple ZipSpectra8). */
void spectra8_set_thread_profile(spectra8_profile profile);
spectra8_profile spectra8_get_thread_profile(void);

/* Train a custom static profile from sample columnar data and register it under profile_id (>= 5). */
int spectra8_train_and_register_profile(unsigned profile_id, const void* sample, size_t sample_size, int stride);
int spectra8_train_and_register_profile_level(unsigned profile_id, const void* sample, size_t sample_size, int stride, int level);

/* Train a static Crystal profile model from sample columnar data.
 * If out_model is NULL, writes required size to *out_model_size and returns 1.
 * Returns 1 on success, 0 on failure. */
int spectra8_train_profile_model(const void* sample, size_t sample_size, int stride,
                                  void* out_model, size_t* out_model_size);
int spectra8_train_profile_model_level(const void* sample, size_t sample_size, int stride, int level,
                                        void* out_model, size_t* out_model_size);

/* Register a serialized profile model under profile_id (>= 5).
 * Returns 1 on success, 0 on failure. */
int spectra8_register_profile_model(unsigned profile_id, const void* model_bytes, size_t model_size);

/* Export an already-registered or builtin profile model to serialized bytes.
 * If out_model is NULL, writes required size to *out_model_size and returns 1.
 * Returns 1 on success, 0 on failure. */
int spectra8_export_profile_model(unsigned profile_id, void* out_model, size_t* out_model_size);

/* CPU decompress + H2D page pump (see spectra8/page_service.hpp). */
typedef enum spectra8_gpu_backend {
  SPECTRA8_GPU_HOST = 0,
  SPECTRA8_GPU_CUDA = 1,
  SPECTRA8_GPU_HIP = 2,
  SPECTRA8_GPU_METAL = 3,
  SPECTRA8_GPU_DEFAULT = 255
} spectra8_gpu_backend;

int spectra8_gpu_transport_available(spectra8_gpu_backend backend);
const char* spectra8_gpu_backend_name(spectra8_gpu_backend backend);

typedef struct spectra8_page_pump spectra8_page_pump;

/* unzip_threads <= 0 means auto; ring_slots 0 means the library default (3).
 * Returns NULL if the backend is unavailable. */
spectra8_page_pump* spectra8_page_pump_create(spectra8_gpu_backend backend,
                                                int unzip_threads,
                                                int ring_slots);
void spectra8_page_pump_destroy(spectra8_page_pump* p);
void* spectra8_page_pump_alloc_device(spectra8_page_pump* p, size_t n);
void spectra8_page_pump_free_device(spectra8_page_pump* p, void* d);
spectra8_status spectra8_page_pump_push_all(spectra8_page_pump* p,
                                              const void* frame,
                                              size_t frame_size,
                                              void* device_dst,
                                              size_t dst_capacity,
                                              size_t* written);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* SPECTRA8_SPECTRA8_C_H */
