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

// Minimal parallel-for over [0, n). Workers pull indices from a shared atomic
// counter; the caller thread participates too. Runs inline when threads <= 1.
#ifndef SPECTRA8_SRC_PARALLEL_HPP
#define SPECTRA8_SRC_PARALLEL_HPP

#include <atomic>
#include <thread>
#include <vector>

#include "util/common.hpp"

namespace spectra8::detail {

// Resolve a requested thread count: <= 0 means "auto" (hardware concurrency).
inline int resolve_threads(int requested) noexcept {
  if (requested > 0) return requested;
  unsigned hc = std::thread::hardware_concurrency();
  return hc == 0 ? 1 : static_cast<int>(hc);
}

template <class F>
void parallel_for(usize n, int threads, F&& f) {
  threads = resolve_threads(threads);
  if (threads < 1) threads = 1;
  if (threads == 1 || n <= 1) {
    for (usize i = 0; i < n; ++i) f(i);
    return;
  }
  if (static_cast<usize>(threads) > n) threads = static_cast<int>(n);

  std::atomic<usize> next{0};
  auto worker = [&]() {
    for (usize i = next.fetch_add(1, std::memory_order_relaxed); i < n;
         i = next.fetch_add(1, std::memory_order_relaxed)) {
      f(i);
    }
  };

  std::vector<std::thread> pool;
  pool.reserve(static_cast<usize>(threads - 1));
  for (int t = 0; t < threads - 1; ++t) pool.emplace_back(worker);
  worker();
  for (auto& th : pool) th.join();
}

}  // namespace spectra8::detail

#endif  // SPECTRA8_SRC_PARALLEL_HPP
