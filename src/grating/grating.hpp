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

// Grating: order-0 rANS over a column of whole integers.
//
// The page is a little-endian array of 1, 2, 4 or 8 byte symbols. The encoder
// picks the width whose spectrum is cheapest, keeps the lines that earn a
// frequency slot, and writes the rest raw behind one escape symbol. Decode is
// four or eight interleaved 16-bit-renorm states (eight once the column has
// at least 4096 symbols). There are no levels: the model does not search.
#ifndef SPECTRA8_SRC_GRATING_HPP
#define SPECTRA8_SRC_GRATING_HPP

#include "util/common.hpp"

namespace spectra8::detail {

void build_grating_block(const u8* src, usize n, std::vector<u8>& payload);
Status decode_grating_block(const u8* data, usize len, u8* dst, usize n);

}  // namespace spectra8::detail

#endif  // SPECTRA8_SRC_GRATING_HPP
