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

#ifdef SPECTRA8_HAVE_METAL

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstring>
#include <cstdlib>
#include <mutex>
#include <new>
#include <unordered_map>

#include "gpu/h2d_backend.hpp"

namespace spectra8::detail {
namespace {

struct MetalH2d {
  id<MTLDevice> device = nil;
  std::mutex mu;
  std::unordered_map<void*, id<MTLBuffer>> bufs;
};

void* aligned_host(std::size_t n) {
  if (n == 0) n = 1;
  const std::size_t aligned = (n + 63u) & ~std::size_t{63};
  void* p = nullptr;
  if (posix_memalign(&p, 64, aligned) != 0) return nullptr;
  return p;
}

}  // namespace

bool metal_h2d_available() noexcept {
  return MTLCreateSystemDefaultDevice() != nil;
}

void* metal_h2d_open() noexcept {
  id<MTLDevice> d = MTLCreateSystemDefaultDevice();
  if (!d) return nullptr;
  auto* c = new (std::nothrow) MetalH2d();
  if (!c) return nullptr;
  c->device = d;
  return c;
}

void metal_h2d_close(void* ctx) noexcept { delete static_cast<MetalH2d*>(ctx); }

void* metal_h2d_alloc_staging(std::size_t n) noexcept { return aligned_host(n); }

void metal_h2d_free_staging(void* p) noexcept { std::free(p); }

void* metal_h2d_alloc_device(void* ctx, std::size_t n) noexcept {
  auto* c = static_cast<MetalH2d*>(ctx);
  if (!c || !c->device) return nullptr;
  const NSUInteger len = n == 0 ? 1 : static_cast<NSUInteger>(n);
  id<MTLBuffer> b = [c->device newBufferWithLength:len
                                           options:MTLResourceStorageModeShared];
  if (!b) return nullptr;
  void* p = [b contents];
  std::lock_guard<std::mutex> lock(c->mu);
  c->bufs[p] = b;
  return p;
}

void metal_h2d_free_device(void* ctx, void* p) noexcept {
  auto* c = static_cast<MetalH2d*>(ctx);
  if (!c || !p) return;
  std::lock_guard<std::mutex> lock(c->mu);
  c->bufs.erase(p);
}

Status metal_h2d_copy(void* ctx, void* dst, const void* src, std::size_t n,
                      bool to_device) noexcept {
  auto* c = static_cast<MetalH2d*>(ctx);
  if (!c || !dst || !src) return Status::InternalError;
  (void)to_device;
  std::memcpy(dst, src, n);
  return Status::Ok;
}

Status metal_h2d_sync(void* ctx) noexcept {
  (void)ctx;
  return Status::Ok;
}

}  // namespace spectra8::detail

#endif  // SPECTRA8_HAVE_METAL
