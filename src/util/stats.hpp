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

// Compression statistics (optional aggregation for benchmarking / diagnostics).
#ifndef SPECTRA8_SRC_STATS_HPP
#define SPECTRA8_SRC_STATS_HPP

#include <cstddef>
#include <span>

#include "spectra8/spectra8.hpp"

namespace spectra8::detail {

// Metadata for one successful compress() call (one RNTuple page / ROOT zip block).
struct CompressObserveInfo {
  StatsWirePath wire_path = StatsWirePath::RawCodec;
  StatsPageOutcome outcome = StatsPageOutcome::Plain;
  StatsFilterKind page_filter = StatsFilterKind::None;
  bool page_used_rle = false;
  bool explicit_set = false;
  // Set when the call used Codec::Auto; records the concrete winner.
  bool has_auto_winner = false;
  Codec auto_winner = Codec::Beam;
};

void stats_reset_last_observe() noexcept;
void stats_set_last_observe(const CompressObserveInfo& info) noexcept;
[[nodiscard]] CompressObserveInfo stats_take_last_observe() noexcept;

void stats_observe_compress(std::size_t uncompressed_size, std::span<const std::byte> compressed,
                            const CompressObserveInfo& info) noexcept;

[[nodiscard]] CompressObserveInfo observe_spectra8_frame(std::span<const std::byte> frame,
                                                     bool filters_enabled) noexcept;

// One Crystal block requested with a static profile: coded with it, or
// escaped to a dynamic block (see build_crystal_profile).
void stats_observe_crystal_profile_block(bool escaped) noexcept;

}  // namespace spectra8::detail

#endif  // SPECTRA8_SRC_STATS_HPP
