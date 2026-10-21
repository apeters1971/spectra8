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

// Metal (GPU) match finder. Optional: only built when SPECTRA8_HAVE_METAL is
// defined (Apple platforms with the Metal framework).
//
// The GPU performs the parallel, compute-heavy part of compression: for every
// position it finds a bounded best match within its independent block, using a
// per-threadgroup hash table. The sequential parse, entropy coding and framing
// stay on the CPU, so GPU-assisted output is an ordinary decodable Spectra8 frame.
#ifndef SPECTRA8_SRC_GPU_METAL_LZ_HPP
#define SPECTRA8_SRC_GPU_METAL_LZ_HPP

#include "util/common.hpp"

namespace spectra8::detail {

// True if a usable Metal device is present and the kernel compiled.
bool metal_available();

// Opaque handle for an in-flight (committed but not yet completed) GPU
// match-finding dispatch. Used to overlap GPU work with CPU encoding.
struct MetalJob;

// Begin async match-finding for `n` bytes, treated as independent blocks of
// `block_size` (matches never cross a block boundary). The command buffer is
// committed but not awaited, so the GPU runs while the caller does other work.
// Returns nullptr on failure (caller should fall back to the CPU finder).
MetalJob* metal_find_begin(const u8* src, usize n, usize block_size,
                           int max_len);

// Wait for `job` to finish, copy its per-position results into lens/offs (each
// at least the job's `n` entries), and release the job. For position i, lens[i]
// is the match length (0 if none) and offs[i] the back-distance. Returns false
// on failure; the job is released either way.
bool metal_find_end(MetalJob* job, u16* lens, u32* offs);

// Synchronous convenience wrapper (begin + end).
bool metal_find_matches(const u8* src, usize n, usize block_size, int max_len,
                        u16* lens, u32* offs);

// Top-K variant for the optimal parser: instead of one best match per position
// it returns up to kGpuTopK distinct (length, offset) candidates per position.
// lens/offs must each hold n * kGpuTopK entries (candidate s of position i at
// index i*kGpuTopK + s; unused slots are 0). Same async begin/end contract.
MetalJob* metal_find_topk_begin(const u8* src, usize n, usize block_size,
                                int max_len);
bool metal_find_topk_end(MetalJob* job, u16* lens, u32* offs);

}  // namespace spectra8::detail

#endif  // SPECTRA8_SRC_GPU_METAL_LZ_HPP
