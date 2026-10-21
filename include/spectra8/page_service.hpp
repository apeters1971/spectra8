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

// CPU decompress + host-to-device copy for GPU consumers.
//
// Decode stays on the CPU. This header is the in-process page pump described
// in docs/cpu-h2d-page-service.md: unzip into pinned staging, then async H2D
// into device memory. There is no GPU decoder.
//
// GpuBackend::Host is always available (memcpy into a heap "device" buffer)
// and is what tests use. Cuda / Hip / Metal are compiled in when the matching
// toolkit is present; Default picks the first real GPU, else Host.
#ifndef SPECTRA8_PAGE_SERVICE_HPP
#define SPECTRA8_PAGE_SERVICE_HPP

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "spectra8/spectra8.hpp"

namespace spectra8 {

enum class GpuBackend : std::uint8_t {
  Host = 0,
  Cuda = 1,
  Hip = 2,
  Metal = 3,
  Default = 255,  // first available real GPU, else Host
};

[[nodiscard]] bool gpu_transport_available(GpuBackend b) noexcept;
[[nodiscard]] GpuBackend default_gpu_backend() noexcept;
[[nodiscard]] const char* gpu_backend_name(GpuBackend b) noexcept;

// Regenerated size of each independently addressable page. Spectra8 frames yield
// one entry per block (seek table or header scan). zstd / bzip3 / xz / L4EF
// yield a single page equal to the frame content size when that size is known.
[[nodiscard]] Result<std::vector<std::uint64_t>> get_page_sizes(
    std::span<const std::byte> frame);

// One GPU (or host-emulated) transport: staging alloc, device alloc, H2D, D2H.
// Not thread-safe. Allocations are owned by the caller (free_*).
class GpuTransport {
 public:
  GpuTransport();  // Default backend
  explicit GpuTransport(GpuBackend backend);
  GpuTransport(GpuTransport&& o) noexcept;
  GpuTransport& operator=(GpuTransport&& o) noexcept;
  ~GpuTransport();

  GpuTransport(const GpuTransport&) = delete;
  GpuTransport& operator=(const GpuTransport&) = delete;

  [[nodiscard]] bool ok() const noexcept;
  [[nodiscard]] GpuBackend backend() const noexcept;

  [[nodiscard]] void* alloc_staging(std::size_t n) noexcept;
  void free_staging(void* p) noexcept;
  [[nodiscard]] void* alloc_device(std::size_t n) noexcept;
  void free_device(void* p) noexcept;

  Status h2d_async(void* device_dst, const void* host_src,
                   std::size_t n) noexcept;
  Status d2h(void* host_dst, const void* device_src, std::size_t n) noexcept;
  Status synchronize() noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

struct PagePumpParams {
  int unzip_threads = 0;             // <= 0 => hardware concurrency
  int ring_slots = 3;                // clamped to [2, 8]
  std::size_t batch_bytes = 8 << 20; // coalesce pages up to this H2D size
};

// Binds a compressed frame and copies decompressed pages to device memory.
// Overlaps CPU unzip of batch N+1 with H2D of batch N on a pinned ring.
class PagePump {
 public:
  explicit PagePump(GpuBackend backend = GpuBackend::Default,
                    PagePumpParams params = {});
  PagePump(GpuTransport transport, PagePumpParams params);

  PagePump(PagePump&&) noexcept;
  PagePump& operator=(PagePump&&) noexcept;
  ~PagePump();

  PagePump(const PagePump&) = delete;
  PagePump& operator=(const PagePump&) = delete;

  [[nodiscard]] bool ok() const noexcept;
  [[nodiscard]] GpuTransport& transport() noexcept;
  [[nodiscard]] const GpuTransport& transport() const noexcept;

  Status bind(std::span<const std::byte> frame);
  [[nodiscard]] std::size_t page_count() const noexcept;
  [[nodiscard]] std::size_t page_bytes(std::size_t index) const noexcept;
  [[nodiscard]] std::size_t total_bytes() const noexcept;

  Result<std::size_t> push(std::size_t index, void* device_dst);
  Result<std::size_t> push_range(std::size_t begin, std::size_t end,
                                 void* device_dst);
  Result<std::size_t> push_all(void* device_dst);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Decompress `frame` into caller-owned `device_dst` (capacity `dst_cap`).
[[nodiscard]] Result<std::size_t> decompress_to_device(
    std::span<const std::byte> frame, GpuTransport& transport,
    void* device_dst, std::size_t dst_cap,
    const PagePumpParams& params = {});

}  // namespace spectra8

#endif  // SPECTRA8_PAGE_SERVICE_HPP
