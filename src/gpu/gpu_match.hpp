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

// Backend-agnostic GPU match-finder interface. Exactly one backend is selected
// at compile time (Metal on Apple, HIP on NVIDIA/AMD); the encoder uses only the
// generic gpu_* names below, so it never needs to know which one is active.
//
// Only included when SPECTRA8_HAVE_GPU is defined.
#ifndef SPECTRA8_SRC_GPU_GPU_MATCH_HPP
#define SPECTRA8_SRC_GPU_GPU_MATCH_HPP

#include "util/common.hpp"

#if defined(SPECTRA8_HAVE_METAL)
#include "gpu/metal_lz.hpp"
#elif defined(SPECTRA8_HAVE_HIP)
#include "gpu/hip_lz.hpp"
#endif

namespace spectra8::detail {

// Number of candidate matches the GPU returns per position for the optimal
// ("ultra") parser. MUST match the `TK` constant in the Metal/HIP kernels.
inline constexpr int kGpuTopK = 4;

#if defined(SPECTRA8_HAVE_METAL)

using GpuJob = MetalJob;
inline bool gpu_backend_available() { return metal_available(); }
inline GpuJob* gpu_find_begin(const u8* src, usize n, usize block_size,
                              int max_len) {
  return metal_find_begin(src, n, block_size, max_len);
}
inline bool gpu_find_end(GpuJob* job, u16* lens, u32* offs) {
  return metal_find_end(job, lens, offs);
}
// Top-K candidate variant: lens/offs hold n*kGpuTopK entries (candidate s of
// position i at index i*kGpuTopK + s).
inline GpuJob* gpu_find_topk_begin(const u8* src, usize n, usize block_size,
                                   int max_len) {
  return metal_find_topk_begin(src, n, block_size, max_len);
}
inline bool gpu_find_topk_end(GpuJob* job, u16* lens, u32* offs) {
  return metal_find_topk_end(job, lens, offs);
}

#elif defined(SPECTRA8_HAVE_HIP)

using GpuJob = HipJob;
inline bool gpu_backend_available() { return hip_available(); }
inline GpuJob* gpu_find_begin(const u8* src, usize n, usize block_size,
                              int max_len) {
  return hip_find_begin(src, n, block_size, max_len);
}
inline bool gpu_find_end(GpuJob* job, u16* lens, u32* offs) {
  return hip_find_end(job, lens, offs);
}
inline GpuJob* gpu_find_topk_begin(const u8* src, usize n, usize block_size,
                                   int max_len) {
  return hip_find_topk_begin(src, n, block_size, max_len);
}
inline bool gpu_find_topk_end(GpuJob* job, u16* lens, u32* offs) {
  return hip_find_topk_end(job, lens, offs);
}

#endif

}  // namespace spectra8::detail

#endif  // SPECTRA8_SRC_GPU_GPU_MATCH_HPP
