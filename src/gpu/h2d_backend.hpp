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

// Optional H2D backends (CUDA / HIP / Metal). Host lives in page_service.cpp.
#ifndef SPECTRA8_SRC_GPU_H2D_BACKEND_HPP
#define SPECTRA8_SRC_GPU_H2D_BACKEND_HPP

#include <cstddef>

#include "spectra8/spectra8.hpp"

namespace spectra8::detail {

#if defined(SPECTRA8_HAVE_CUDA)
bool cuda_h2d_available() noexcept;
void* cuda_h2d_open() noexcept;
void cuda_h2d_close(void* ctx) noexcept;
void* cuda_h2d_alloc_staging(std::size_t n) noexcept;
void cuda_h2d_free_staging(void* p) noexcept;
void* cuda_h2d_alloc_device(void* ctx, std::size_t n) noexcept;
void cuda_h2d_free_device(void* ctx, void* p) noexcept;
Status cuda_h2d_copy(void* ctx, void* dst, const void* src, std::size_t n,
                     bool to_device) noexcept;
Status cuda_h2d_sync(void* ctx) noexcept;
#endif

#if defined(SPECTRA8_HAVE_HIP)
bool hip_h2d_available() noexcept;
void* hip_h2d_open() noexcept;
void hip_h2d_close(void* ctx) noexcept;
void* hip_h2d_alloc_staging(std::size_t n) noexcept;
void hip_h2d_free_staging(void* p) noexcept;
void* hip_h2d_alloc_device(void* ctx, std::size_t n) noexcept;
void hip_h2d_free_device(void* ctx, void* p) noexcept;
Status hip_h2d_copy(void* ctx, void* dst, const void* src, std::size_t n,
                    bool to_device) noexcept;
Status hip_h2d_sync(void* ctx) noexcept;
#endif

#if defined(SPECTRA8_HAVE_METAL)
bool metal_h2d_available() noexcept;
void* metal_h2d_open() noexcept;
void metal_h2d_close(void* ctx) noexcept;
void* metal_h2d_alloc_staging(std::size_t n) noexcept;
void metal_h2d_free_staging(void* p) noexcept;
void* metal_h2d_alloc_device(void* ctx, std::size_t n) noexcept;
void metal_h2d_free_device(void* ctx, void* p) noexcept;
Status metal_h2d_copy(void* ctx, void* dst, const void* src, std::size_t n,
                      bool to_device) noexcept;
Status metal_h2d_sync(void* ctx) noexcept;
#endif

}  // namespace spectra8::detail

#endif  // SPECTRA8_SRC_GPU_H2D_BACKEND_HPP
