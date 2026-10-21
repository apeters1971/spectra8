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

// HIP (NVIDIA/AMD GPU) match finder. Optional: only built when
// SPECTRA8_HAVE_HIP is defined (a ROCm/HIP toolchain is required). The kernel
// mirrors the Metal backend: one independent block per GPU block, a hash table
// in shared memory (atomicMin keeps the earliest prior position), and a bounded
// best-match search per position. The sequential parse, entropy coding and
// framing stay on the CPU, so output is an ordinary decodable Spectra8 frame.
#ifndef SPECTRA8_SRC_GPU_HIP_LZ_HPP
#define SPECTRA8_SRC_GPU_HIP_LZ_HPP

#include "util/common.hpp"

namespace spectra8::detail {

// True if a usable HIP device is present.
bool hip_available();

// Opaque handle for an in-flight (launched but not yet synchronized) dispatch.
struct HipJob;

// Begin async match-finding for `n` bytes, treated as independent blocks of
// `block_size`. The kernel is launched on a stream and not awaited, so the GPU
// runs while the caller keeps working. Returns nullptr on failure.
HipJob* hip_find_begin(const u8* src, usize n, usize block_size, int max_len);

// Wait for `job`, copy its per-position results into lens/offs (each at least
// the job's `n` entries), and release it. Returns false on failure; the job is
// released either way.
bool hip_find_end(HipJob* job, u16* lens, u32* offs);

// Synchronous convenience wrapper (begin + end).
bool hip_find_matches(const u8* src, usize n, usize block_size, int max_len,
                      u16* lens, u32* offs);

// Top-K variant for the optimal parser: up to kGpuTopK distinct (length,
// offset) candidates per position. lens/offs must each hold n * kGpuTopK
// entries (candidate s of position i at index i*kGpuTopK + s). Same async
// begin/end contract as hip_find_begin/end.
HipJob* hip_find_topk_begin(const u8* src, usize n, usize block_size,
                            int max_len);
bool hip_find_topk_end(HipJob* job, u16* lens, u32* offs);

}  // namespace spectra8::detail

#endif  // SPECTRA8_SRC_GPU_HIP_LZ_HPP
