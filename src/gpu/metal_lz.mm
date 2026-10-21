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

// Metal GPU match finder (Objective-C++). Compiled only when SPECTRA8_HAVE_METAL
// is defined. The MSL kernel is compiled at runtime, so no offline `metal`
// toolchain is required.
#ifdef SPECTRA8_HAVE_METAL

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstring>
#include <mutex>

#include "gpu/gpu_match.hpp"  // kGpuTopK (must match the kernel's TK)
#include "gpu/metal_lz.hpp"

namespace spectra8::detail {
namespace {

// One independent block per threadgroup. Threads cooperatively build a hash
// table in threadgroup memory, keeping the EARLIEST position per bucket via
// atomic_min (so a probe yields a valid prior candidate rather than a later
// one). Each thread then computes a bounded best match for its position(s).
// 1<<13 buckets * 4 B = 32 KiB == the threadgroup-memory limit.
const char* kKernelSrc = R"MSL(
#include <metal_stdlib>
using namespace metal;

constant uint HBITS = 13u;
constant uint HSIZE = 1u << HBITS;
constant uint EMPTY = 0xFFFFFFFFu;

inline uint load32(device const uchar* p) {
  return (uint)p[0] | ((uint)p[1] << 8) | ((uint)p[2] << 16) | ((uint)p[3] << 24);
}
inline uint hash4(uint x) { return (x * 2654435761u) >> (32u - HBITS); }

kernel void lz_find(device const uchar* src   [[buffer(0)]],
                    device ushort* out_len     [[buffer(1)]],
                    device uint* out_off       [[buffer(2)]],
                    constant uint& total_n     [[buffer(3)]],
                    constant uint& block_size  [[buffer(4)]],
                    constant uint& max_len      [[buffer(5)]],
                    uint  bg   [[threadgroup_position_in_grid]],
                    uint  tid  [[thread_position_in_threadgroup]],
                    uint  nthr [[threads_per_threadgroup]]) {
  threadgroup atomic_uint head[HSIZE];

  uint bstart = bg * block_size;
  if (bstart >= total_n) return;
  uint bend = min(bstart + block_size, total_n);
  uint blen = bend - bstart;

  for (uint h = tid; h < HSIZE; h += nthr)
    atomic_store_explicit(&head[h], EMPTY, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Build: keep the earliest position that maps to each bucket.
  for (uint p = tid; p + 4u <= blen; p += nthr) {
    uint key = hash4(load32(src + bstart + p));
    atomic_fetch_min_explicit(&head[key], p, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Match: probe the bucket; accept only earlier candidates (offset > 0).
  for (uint p = tid; p < blen; p += nthr) {
    uint len = 0u, off = 0u;
    if (p + 4u <= blen) {
      uint key = hash4(load32(src + bstart + p));
      uint c = atomic_load_explicit(&head[key], memory_order_relaxed);
      if (c != EMPTY && c < p) {
        uint maxl = min(max_len, blen - p);
        device const uchar* a = src + bstart + p;
        device const uchar* b = src + bstart + c;
        uint l = 0u;
        while (l < maxl && a[l] == b[l]) ++l;
        if (l >= 4u) { len = l; off = p - c; }
      }
    }
    out_len[bstart + p] = (ushort)min(len, 65535u);
    out_off[bstart + p] = off;
  }
}

// Top-K variant for the optimal parser. The threadgroup hash keeps, per bucket,
// the earliest position in each of TK residue classes (slot = position mod TK),
// yielding up to TK diverse early candidates per position. atomic_fetch_min is
// order-independent, so results are deterministic. TK must match kGpuTopK.
// TKBITS buckets * TK slots * 4 B = 32 KiB == the threadgroup-memory limit.
constant uint TKBITS = 11u;
constant uint TKSIZE = 1u << TKBITS;
constant uint TK = 4u;

inline uint hashTK(uint x) { return (x * 2654435761u) >> (32u - TKBITS); }

kernel void lz_find_topk(device const uchar* src   [[buffer(0)]],
                         device ushort* out_len     [[buffer(1)]],
                         device uint* out_off       [[buffer(2)]],
                         constant uint& total_n     [[buffer(3)]],
                         constant uint& block_size  [[buffer(4)]],
                         constant uint& max_len      [[buffer(5)]],
                         uint  bg   [[threadgroup_position_in_grid]],
                         uint  tid  [[thread_position_in_threadgroup]],
                         uint  nthr [[threads_per_threadgroup]]) {
  threadgroup atomic_uint head[TKSIZE * TK];

  uint bstart = bg * block_size;
  if (bstart >= total_n) return;
  uint bend = min(bstart + block_size, total_n);
  uint blen = bend - bstart;

  for (uint h = tid; h < TKSIZE * TK; h += nthr)
    atomic_store_explicit(&head[h], EMPTY, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint p = tid; p + 4u <= blen; p += nthr) {
    uint key = hashTK(load32(src + bstart + p));
    uint slot = p & (TK - 1u);
    atomic_fetch_min_explicit(&head[key * TK + slot], p, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint p = tid; p < blen; p += nthr) {
    uint outbase = (bstart + p) * TK;
    for (uint s = 0u; s < TK; ++s) {
      out_len[outbase + s] = 0;
      out_off[outbase + s] = 0u;
    }
    if (p + 4u <= blen) {
      uint key = hashTK(load32(src + bstart + p));
      uint cnt = 0u;
      for (uint s = 0u; s < TK; ++s) {
        uint c = atomic_load_explicit(&head[key * TK + s], memory_order_relaxed);
        if (c == EMPTY || c >= p) continue;
        uint maxl = min(max_len, blen - p);
        device const uchar* a = src + bstart + p;
        device const uchar* b = src + bstart + c;
        uint l = 0u;
        while (l < maxl && a[l] == b[l]) ++l;
        if (l < 4u) continue;
        uint off = p - c;
        bool dup = false;
        for (uint t = 0u; t < cnt; ++t)
          if (out_off[outbase + t] == off) { dup = true; break; }
        if (dup) continue;
        out_len[outbase + cnt] = (ushort)min(l, 65535u);
        out_off[outbase + cnt] = off;
        ++cnt;
      }
    }
  }
}
)MSL";

struct MetalCtx {
  id<MTLDevice> dev = nil;
  id<MTLCommandQueue> queue = nil;
  id<MTLComputePipelineState> pso = nil;       // lz_find (single best)
  id<MTLComputePipelineState> pso_topk = nil;  // lz_find_topk
  bool ok = false;
};

MetalCtx& ctx() {
  static MetalCtx c;
  static std::once_flag once;
  std::call_once(once, [] {
    @autoreleasepool {
      c.dev = MTLCreateSystemDefaultDevice();
      if (!c.dev) return;
      NSError* err = nil;
      id<MTLLibrary> lib = [c.dev newLibraryWithSource:@(kKernelSrc)
                                               options:nil
                                                 error:&err];
      if (!lib) return;
      id<MTLFunction> fn = [lib newFunctionWithName:@"lz_find"];
      if (!fn) return;
      c.pso = [c.dev newComputePipelineStateWithFunction:fn error:&err];
      if (!c.pso) return;
      // Top-K pipeline is best-effort: if it fails to build, the single-best
      // path still works and ultra+GPU falls back to the CPU optimal parser.
      id<MTLFunction> fnk = [lib newFunctionWithName:@"lz_find_topk"];
      if (fnk) c.pso_topk = [c.dev newComputePipelineStateWithFunction:fnk
                                                                 error:&err];
      c.queue = [c.dev newCommandQueue];
      c.ok = (c.queue != nil);
    }
  });
  return c;
}

}  // namespace

bool metal_available() { return ctx().ok; }

// Holds the GPU resources for one in-flight dispatch. The strong (ARC) members
// keep the buffers and command buffer alive until metal_find_end releases them.
struct MetalJob {
  id<MTLCommandBuffer> cb = nil;
  id<MTLBuffer> bsrc = nil;
  id<MTLBuffer> blen = nil;
  id<MTLBuffer> boff = nil;
  usize n = 0;
  usize stride = 1;  // output entries per position (1, or kGpuTopK for top-K)
};

// Shared launcher: dispatches `pso` over n/block_size threadgroups, allocating
// `stride` output entries per input byte. Committed but not awaited.
static MetalJob* metal_begin_impl(const u8* src, usize n, usize block_size,
                                  int max_len, id<MTLComputePipelineState> pso,
                                  usize stride) {
  MetalCtx& c = ctx();
  if (!c.ok || !pso || n == 0 || block_size < 4) return nullptr;
  if (max_len < 4) max_len = 4;
  if (max_len > 65535) max_len = 65535;

  @autoreleasepool {
    id<MTLBuffer> bsrc = [c.dev newBufferWithBytes:src
                                            length:n
                                           options:MTLResourceStorageModeShared];
    id<MTLBuffer> blen =
        [c.dev newBufferWithLength:n * stride * sizeof(u16)
                           options:MTLResourceStorageModeShared];
    id<MTLBuffer> boff =
        [c.dev newBufferWithLength:n * stride * sizeof(u32)
                           options:MTLResourceStorageModeShared];
    if (!bsrc || !blen || !boff) return nullptr;

    uint32_t total_n = static_cast<uint32_t>(n);
    uint32_t bs = static_cast<uint32_t>(block_size);
    uint32_t ml = static_cast<uint32_t>(max_len);

    id<MTLCommandBuffer> cb = [c.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:bsrc offset:0 atIndex:0];
    [enc setBuffer:blen offset:0 atIndex:1];
    [enc setBuffer:boff offset:0 atIndex:2];
    [enc setBytes:&total_n length:4 atIndex:3];
    [enc setBytes:&bs length:4 atIndex:4];
    [enc setBytes:&ml length:4 atIndex:5];

    NSUInteger nblocks = (n + block_size - 1) / block_size;
    NSUInteger tg = 256;
    if (tg > pso.maxTotalThreadsPerThreadgroup)
      tg = pso.maxTotalThreadsPerThreadgroup;
    [enc dispatchThreadgroups:MTLSizeMake(nblocks, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc endEncoding];
    [cb commit];  // do not wait: the GPU runs while the caller keeps working

    MetalJob* job = new MetalJob();
    job->cb = cb;
    job->bsrc = bsrc;
    job->blen = blen;
    job->boff = boff;
    job->n = n;
    job->stride = stride;
    return job;
  }
}

MetalJob* metal_find_begin(const u8* src, usize n, usize block_size,
                           int max_len) {
  return metal_begin_impl(src, n, block_size, max_len, ctx().pso, 1);
}

bool metal_find_end(MetalJob* job, u16* lens, u32* offs) {
  if (!job) return false;
  bool ok = false;
  @autoreleasepool {
    [job->cb waitUntilCompleted];
    if (job->cb.status == MTLCommandBufferStatusCompleted) {
      std::memcpy(lens, job->blen.contents, job->n * job->stride * sizeof(u16));
      std::memcpy(offs, job->boff.contents, job->n * job->stride * sizeof(u32));
      ok = true;
    }
  }
  delete job;
  return ok;
}

MetalJob* metal_find_topk_begin(const u8* src, usize n, usize block_size,
                                int max_len) {
  return metal_begin_impl(src, n, block_size, max_len, ctx().pso_topk,
                          static_cast<usize>(kGpuTopK));
}

bool metal_find_topk_end(MetalJob* job, u16* lens, u32* offs) {
  return metal_find_end(job, lens, offs);  // copy honors job->stride
}

bool metal_find_matches(const u8* src, usize n, usize block_size, int max_len,
                        u16* lens, u32* offs) {
  MetalJob* job = metal_find_begin(src, n, block_size, max_len);
  if (!job) return false;
  return metal_find_end(job, lens, offs);
}

}  // namespace spectra8::detail

#endif  // SPECTRA8_HAVE_METAL
