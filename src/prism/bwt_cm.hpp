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

#ifndef SPECTRA8_SRC_BWT_CM_HPP
#define SPECTRA8_SRC_BWT_CM_HPP

#include <cstddef>
#include <cstdint>
#include <vector>

#include "util/common.hpp"

namespace spectra8::detail {

// BWT Context-Mixing entropy coder.
// Operates directly on the BWT L-column (no Move-to-Front).
// Combines an order-0 bit tree, order-1 & order-2 preceding-symbol bit models,
// and an adaptive probability map (APM) with run detection.
// Yields near-Shannon compression ratios on natural text and source code.

// Compress `src[0, n)` into `out`. Returns true on success.
bool bwt_cm_encode(const std::uint8_t* src, std::size_t n, std::vector<std::uint8_t>& out);

// Decompress `src[0, src_len)` into `dst[0, orig_len)`. Returns Status::Ok on success.
Status bwt_cm_decode(const std::uint8_t* src, std::size_t src_len,
                     std::uint8_t* dst, std::size_t orig_len) noexcept;

}  // namespace spectra8::detail

#endif  // SPECTRA8_SRC_BWT_CM_HPP
