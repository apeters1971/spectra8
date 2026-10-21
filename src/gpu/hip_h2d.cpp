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

#ifdef SPECTRA8_HAVE_HIP

#include <new>

#include <hip/hip_runtime_api.h>

#include "gpu/h2d_backend.hpp"

namespace spectra8::detail {
namespace {

struct HipH2d {
  hipStream_t stream = nullptr;
};

}  // namespace

bool hip_h2d_available() noexcept {
  int n = 0;
  return hipGetDeviceCount(&n) == hipSuccess && n > 0;
}

void* hip_h2d_open() noexcept {
  if (!hip_h2d_available()) return nullptr;
  auto* c = new (std::nothrow) HipH2d();
  if (!c) return nullptr;
  if (hipStreamCreate(&c->stream) != hipSuccess) {
    delete c;
    return nullptr;
  }
  return c;
}

void hip_h2d_close(void* ctx) noexcept {
  auto* c = static_cast<HipH2d*>(ctx);
  if (!c) return;
  if (c->stream) hipStreamDestroy(c->stream);
  delete c;
}

void* hip_h2d_alloc_staging(std::size_t n) noexcept {
  if (n == 0) n = 1;
  void* p = nullptr;
  if (hipHostMalloc(&p, n, hipHostMallocDefault) != hipSuccess) return nullptr;
  return p;
}

void hip_h2d_free_staging(void* p) noexcept {
  if (p) hipHostFree(p);
}

void* hip_h2d_alloc_device(void* ctx, std::size_t n) noexcept {
  (void)ctx;
  if (n == 0) n = 1;
  void* p = nullptr;
  if (hipMalloc(&p, n) != hipSuccess) return nullptr;
  return p;
}

void hip_h2d_free_device(void* ctx, void* p) noexcept {
  (void)ctx;
  if (p) hipFree(p);
}

Status hip_h2d_copy(void* ctx, void* dst, const void* src, std::size_t n,
                    bool to_device) noexcept {
  auto* c = static_cast<HipH2d*>(ctx);
  if (!c || !c->stream) return Status::InternalError;
  const auto kind = to_device ? hipMemcpyHostToDevice : hipMemcpyDeviceToHost;
  if (hipMemcpyAsync(dst, src, n, kind, c->stream) != hipSuccess)
    return Status::InternalError;
  if (!to_device && hipStreamSynchronize(c->stream) != hipSuccess)
    return Status::InternalError;
  return Status::Ok;
}

Status hip_h2d_sync(void* ctx) noexcept {
  auto* c = static_cast<HipH2d*>(ctx);
  if (!c || !c->stream) return Status::InternalError;
  return hipStreamSynchronize(c->stream) == hipSuccess ? Status::Ok
                                                       : Status::InternalError;
}

}  // namespace spectra8::detail

#endif  // SPECTRA8_HAVE_HIP
