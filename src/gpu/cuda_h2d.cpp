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

#ifdef SPECTRA8_HAVE_CUDA

#include <new>

#include <cuda_runtime_api.h>

#include "gpu/h2d_backend.hpp"

namespace spectra8::detail {
namespace {

struct CudaH2d {
  cudaStream_t stream = nullptr;
};

}  // namespace

bool cuda_h2d_available() noexcept {
  int n = 0;
  return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

void* cuda_h2d_open() noexcept {
  if (!cuda_h2d_available()) return nullptr;
  auto* c = new (std::nothrow) CudaH2d();
  if (!c) return nullptr;
  if (cudaStreamCreate(&c->stream) != cudaSuccess) {
    delete c;
    return nullptr;
  }
  return c;
}

void cuda_h2d_close(void* ctx) noexcept {
  auto* c = static_cast<CudaH2d*>(ctx);
  if (!c) return;
  if (c->stream) cudaStreamDestroy(c->stream);
  delete c;
}

void* cuda_h2d_alloc_staging(std::size_t n) noexcept {
  if (n == 0) n = 1;
  void* p = nullptr;
  if (cudaHostAlloc(&p, n, cudaHostAllocDefault) != cudaSuccess) return nullptr;
  return p;
}

void cuda_h2d_free_staging(void* p) noexcept {
  if (p) cudaFreeHost(p);
}

void* cuda_h2d_alloc_device(void* ctx, std::size_t n) noexcept {
  (void)ctx;
  if (n == 0) n = 1;
  void* p = nullptr;
  if (cudaMalloc(&p, n) != cudaSuccess) return nullptr;
  return p;
}

void cuda_h2d_free_device(void* ctx, void* p) noexcept {
  (void)ctx;
  if (p) cudaFree(p);
}

Status cuda_h2d_copy(void* ctx, void* dst, const void* src, std::size_t n,
                     bool to_device) noexcept {
  auto* c = static_cast<CudaH2d*>(ctx);
  if (!c || !c->stream) return Status::InternalError;
  const auto kind = to_device ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost;
  if (cudaMemcpyAsync(dst, src, n, kind, c->stream) != cudaSuccess)
    return Status::InternalError;
  if (!to_device && cudaStreamSynchronize(c->stream) != cudaSuccess)
    return Status::InternalError;
  return Status::Ok;
}

Status cuda_h2d_sync(void* ctx) noexcept {
  auto* c = static_cast<CudaH2d*>(ctx);
  if (!c || !c->stream) return Status::InternalError;
  return cudaStreamSynchronize(c->stream) == cudaSuccess ? Status::Ok
                                                         : Status::InternalError;
}

}  // namespace spectra8::detail

#endif  // SPECTRA8_HAVE_CUDA
