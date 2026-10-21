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

// HIP GPU match finder. Compiled only when SPECTRA8_HAVE_HIP is defined, with
// a HIP-capable compiler (hipcc / clang with the HIP toolchain). Targets AMD
// (ROCm) and NVIDIA (HIP's CUDA backend) from the same source.
//
// NOTE: this backend is written to mirror the Metal backend exactly but has not
// been compiled or run in this environment (no NVIDIA/AMD GPU available here).
#ifdef SPECTRA8_HAVE_HIP

#include <hip/hip_runtime.h>

#include "gpu/gpu_match.hpp"  // kGpuTopK (must match the kernel's kHipTK)
#include "gpu/hip_lz.hpp"

namespace spectra8::detail {

// One independent block per GPU block. Threads cooperatively build a hash table
// in shared memory, keeping the EARLIEST position per bucket via atomicMin (so
// a probe yields a valid prior candidate). Each thread then computes a bounded
// best match for its position(s). 1<<13 buckets * 4 B = 32 KiB of shared mem.
namespace {
constexpr unsigned kHipHBits = 13u;
constexpr unsigned kHipHSize = 1u << kHipHBits;
constexpr unsigned kHipEmpty = 0xFFFFFFFFu;
}  // namespace

__device__ inline unsigned hip_load32(const unsigned char* p) {
  return static_cast<unsigned>(p[0]) | (static_cast<unsigned>(p[1]) << 8) |
         (static_cast<unsigned>(p[2]) << 16) |
         (static_cast<unsigned>(p[3]) << 24);
}

__device__ inline unsigned hip_hash4(unsigned x) {
  return (x * 2654435761u) >> (32u - kHipHBits);
}

__device__ inline unsigned hip_minu(unsigned a, unsigned b) {
  return a < b ? a : b;
}

__global__ void lz_find_kernel(const unsigned char* __restrict__ src,
                               unsigned short* __restrict__ out_len,
                               unsigned* __restrict__ out_off, unsigned total_n,
                               unsigned block_size, unsigned max_len) {
  __shared__ unsigned head[kHipHSize];

  const unsigned bstart = blockIdx.x * block_size;
  if (bstart >= total_n) return;
  const unsigned bend = hip_minu(bstart + block_size, total_n);
  const unsigned blen = bend - bstart;
  const unsigned tid = threadIdx.x;
  const unsigned nthr = blockDim.x;

  for (unsigned h = tid; h < kHipHSize; h += nthr) head[h] = kHipEmpty;
  __syncthreads();

  // Build: keep the earliest position that maps to each bucket.
  for (unsigned p = tid; p + 4u <= blen; p += nthr) {
    unsigned key = hip_hash4(hip_load32(src + bstart + p));
    atomicMin(&head[key], p);
  }
  __syncthreads();

  // Match: probe the bucket; accept only earlier candidates (offset > 0).
  for (unsigned p = tid; p < blen; p += nthr) {
    unsigned len = 0u, off = 0u;
    if (p + 4u <= blen) {
      unsigned key = hip_hash4(hip_load32(src + bstart + p));
      unsigned c = head[key];
      if (c != kHipEmpty && c < p) {
        unsigned maxl = hip_minu(max_len, blen - p);
        const unsigned char* a = src + bstart + p;
        const unsigned char* b = src + bstart + c;
        unsigned l = 0u;
        while (l < maxl && a[l] == b[l]) ++l;
        if (l >= 4u) {
          len = l;
          off = p - c;
        }
      }
    }
    out_len[bstart + p] = static_cast<unsigned short>(hip_minu(len, 65535u));
    out_off[bstart + p] = off;
  }
}

// Top-K variant for the optimal parser. Per bucket the shared hash keeps the
// earliest position in each of kHipTK residue classes (slot = position mod
// kHipTK), giving up to kHipTK diverse early candidates per position. atomicMin
// is order-independent, so results are deterministic. kHipTK must match
// kGpuTopK. kHipTKSize buckets * kHipTK slots * 4 B = 32 KiB of shared memory.
namespace {
constexpr unsigned kHipTKBits = 11u;
constexpr unsigned kHipTKSize = 1u << kHipTKBits;
constexpr unsigned kHipTK = 4u;
static_assert(kHipTK == static_cast<unsigned>(kGpuTopK),
              "HIP kernel TK must match kGpuTopK");
}  // namespace

__device__ inline unsigned hip_hashTK(unsigned x) {
  return (x * 2654435761u) >> (32u - kHipTKBits);
}

__global__ void lz_find_topk_kernel(const unsigned char* __restrict__ src,
                                     unsigned short* __restrict__ out_len,
                                     unsigned* __restrict__ out_off,
                                     unsigned total_n, unsigned block_size,
                                     unsigned max_len) {
  __shared__ unsigned head[kHipTKSize * kHipTK];

  const unsigned bstart = blockIdx.x * block_size;
  if (bstart >= total_n) return;
  const unsigned bend = hip_minu(bstart + block_size, total_n);
  const unsigned blen = bend - bstart;
  const unsigned tid = threadIdx.x;
  const unsigned nthr = blockDim.x;

  for (unsigned h = tid; h < kHipTKSize * kHipTK; h += nthr) head[h] = kHipEmpty;
  __syncthreads();

  for (unsigned p = tid; p + 4u <= blen; p += nthr) {
    unsigned key = hip_hashTK(hip_load32(src + bstart + p));
    unsigned slot = p & (kHipTK - 1u);
    atomicMin(&head[key * kHipTK + slot], p);
  }
  __syncthreads();

  for (unsigned p = tid; p < blen; p += nthr) {
    unsigned outbase = (bstart + p) * kHipTK;
    for (unsigned s = 0u; s < kHipTK; ++s) {
      out_len[outbase + s] = 0;
      out_off[outbase + s] = 0u;
    }
    if (p + 4u <= blen) {
      unsigned key = hip_hashTK(hip_load32(src + bstart + p));
      unsigned cnt = 0u;
      for (unsigned s = 0u; s < kHipTK; ++s) {
        unsigned c = head[key * kHipTK + s];
        if (c == kHipEmpty || c >= p) continue;
        unsigned maxl = hip_minu(max_len, blen - p);
        const unsigned char* a = src + bstart + p;
        const unsigned char* b = src + bstart + c;
        unsigned l = 0u;
        while (l < maxl && a[l] == b[l]) ++l;
        if (l < 4u) continue;
        unsigned off = p - c;
        bool dup = false;
        for (unsigned t = 0u; t < cnt; ++t)
          if (out_off[outbase + t] == off) {
            dup = true;
            break;
          }
        if (dup) continue;
        out_len[outbase + cnt] = static_cast<unsigned short>(hip_minu(l, 65535u));
        out_off[outbase + cnt] = off;
        ++cnt;
      }
    }
  }
}

struct HipJob {
  hipStream_t stream = nullptr;
  unsigned char* d_src = nullptr;
  unsigned short* d_len = nullptr;
  unsigned* d_off = nullptr;
  usize n = 0;
  usize stride = 1;  // output entries per position (1, or kGpuTopK for top-K)
};

bool hip_available() {
  int count = 0;
  hipError_t e = hipGetDeviceCount(&count);
  return e == hipSuccess && count > 0;
}

namespace {
// Allocate device buffers (`stride` output entries per byte) and async-copy the
// source. Returns nullptr on failure.
HipJob* hip_alloc_job(const u8* src, usize n, usize stride) {
  HipJob* job = new HipJob();
  job->n = n;
  job->stride = stride;
  auto fail = [&]() -> HipJob* {
    if (job->d_src) hipFree(job->d_src);
    if (job->d_len) hipFree(job->d_len);
    if (job->d_off) hipFree(job->d_off);
    if (job->stream) hipStreamDestroy(job->stream);
    delete job;
    return nullptr;
  };
  if (hipStreamCreate(&job->stream) != hipSuccess) return fail();
  if (hipMalloc(&job->d_src, n) != hipSuccess) return fail();
  if (hipMalloc(&job->d_len, n * stride * sizeof(unsigned short)) != hipSuccess)
    return fail();
  if (hipMalloc(&job->d_off, n * stride * sizeof(unsigned)) != hipSuccess)
    return fail();
  if (hipMemcpyAsync(job->d_src, src, n, hipMemcpyHostToDevice, job->stream) !=
      hipSuccess)
    return fail();
  return job;
}
}  // namespace

HipJob* hip_find_begin(const u8* src, usize n, usize block_size, int max_len) {
  if (n == 0 || block_size < 4) return nullptr;
  if (max_len < 4) max_len = 4;
  if (max_len > 65535) max_len = 65535;
  if (!hip_available()) return nullptr;

  HipJob* job = hip_alloc_job(src, n, 1);
  if (!job) return nullptr;

  const unsigned total_n = static_cast<unsigned>(n);
  const unsigned bs = static_cast<unsigned>(block_size);
  const unsigned ml = static_cast<unsigned>(max_len);
  const unsigned nblocks = static_cast<unsigned>((n + block_size - 1) / block_size);
  const unsigned tg = 256u;

  // Launched on the stream and not awaited: the GPU runs while the caller works.
  hipLaunchKernelGGL(lz_find_kernel, dim3(nblocks), dim3(tg), 0, job->stream,
                     job->d_src, job->d_len, job->d_off, total_n, bs, ml);
  // Any launch/runtime error is surfaced at synchronization time in _end.
  return job;
}

HipJob* hip_find_topk_begin(const u8* src, usize n, usize block_size,
                            int max_len) {
  if (n == 0 || block_size < 4) return nullptr;
  if (max_len < 4) max_len = 4;
  if (max_len > 65535) max_len = 65535;
  if (!hip_available()) return nullptr;

  HipJob* job = hip_alloc_job(src, n, static_cast<usize>(kGpuTopK));
  if (!job) return nullptr;

  const unsigned total_n = static_cast<unsigned>(n);
  const unsigned bs = static_cast<unsigned>(block_size);
  const unsigned ml = static_cast<unsigned>(max_len);
  const unsigned nblocks = static_cast<unsigned>((n + block_size - 1) / block_size);
  const unsigned tg = 256u;

  hipLaunchKernelGGL(lz_find_topk_kernel, dim3(nblocks), dim3(tg), 0,
                     job->stream, job->d_src, job->d_len, job->d_off, total_n,
                     bs, ml);
  return job;
}

bool hip_find_end(HipJob* job, u16* lens, u32* offs) {
  if (!job) return false;
  bool ok = true;
  const usize count = job->n * job->stride;
  if (hipMemcpyAsync(lens, job->d_len, count * sizeof(unsigned short),
                     hipMemcpyDeviceToHost, job->stream) != hipSuccess)
    ok = false;
  if (ok && hipMemcpyAsync(offs, job->d_off, count * sizeof(unsigned),
                           hipMemcpyDeviceToHost, job->stream) != hipSuccess)
    ok = false;
  if (ok && hipStreamSynchronize(job->stream) != hipSuccess) ok = false;
  if (ok && hipGetLastError() != hipSuccess) ok = false;

  if (job->d_src) hipFree(job->d_src);
  if (job->d_len) hipFree(job->d_len);
  if (job->d_off) hipFree(job->d_off);
  if (job->stream) hipStreamDestroy(job->stream);
  delete job;
  return ok;
}

bool hip_find_topk_end(HipJob* job, u16* lens, u32* offs) {
  return hip_find_end(job, lens, offs);  // copy honors job->stride
}

bool hip_find_matches(const u8* src, usize n, usize block_size, int max_len,
                      u16* lens, u32* offs) {
  HipJob* job = hip_find_begin(src, n, block_size, max_len);
  if (!job) return false;
  return hip_find_end(job, lens, offs);
}

}  // namespace spectra8::detail

#endif  // SPECTRA8_HAVE_HIP
