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

#include "spectra8/page_service.hpp"

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <new>
#include <stdlib.h>
#include <utility>

#include "spectra8/spectra8_c.h"
#include "codec.hpp"
#include "util/common.hpp"
#include "gpu/h2d_backend.hpp"
#include "util/parallel.hpp"

namespace spectra8 {
namespace {

void* host_alloc(std::size_t n) noexcept {
  if (n == 0) n = 1;
  const std::size_t aligned = (n + 63u) & ~std::size_t{63};
  void* p = nullptr;
  if (posix_memalign(&p, 64, aligned) != 0) return nullptr;
  return p;
}

void host_free(void* p) noexcept { std::free(p); }

bool looks_like_spectra8(std::span<const std::byte> src) noexcept {
  static constexpr char kMagic[4] = {'L', 'H', 'C', '4'};
  return src.size() >= 4 && std::memcmp(src.data(), kMagic, 4) == 0;
}

int clamp_slots(int slots) noexcept {
  if (slots < 2) return 2;
  if (slots > 8) return 8;
  return slots;
}

}  // namespace

const char* gpu_backend_name(GpuBackend b) noexcept {
  switch (b) {
    case GpuBackend::Host:
      return "host";
    case GpuBackend::Cuda:
      return "cuda";
    case GpuBackend::Hip:
      return "hip";
    case GpuBackend::Metal:
      return "metal";
    case GpuBackend::Default:
      return "default";
  }
  return "?";
}

bool gpu_transport_available(GpuBackend b) noexcept {
  switch (b) {
    case GpuBackend::Host:
      return true;
    case GpuBackend::Cuda:
#if defined(SPECTRA8_HAVE_CUDA)
      return detail::cuda_h2d_available();
#else
      return false;
#endif
    case GpuBackend::Hip:
#if defined(SPECTRA8_HAVE_HIP)
      return detail::hip_h2d_available();
#else
      return false;
#endif
    case GpuBackend::Metal:
#if defined(SPECTRA8_HAVE_METAL)
      return detail::metal_h2d_available();
#else
      return false;
#endif
    case GpuBackend::Default:
      return true;
  }
  return false;
}

GpuBackend default_gpu_backend() noexcept {
  if (gpu_transport_available(GpuBackend::Cuda)) return GpuBackend::Cuda;
  if (gpu_transport_available(GpuBackend::Hip)) return GpuBackend::Hip;
  if (gpu_transport_available(GpuBackend::Metal)) return GpuBackend::Metal;
  return GpuBackend::Host;
}

Result<std::vector<std::uint64_t>> get_page_sizes(
    std::span<const std::byte> frame) {
  Result<std::vector<std::uint64_t>> r;
  if (looks_like_spectra8(frame)) {
    if (auto seek = get_seek_table(frame)) {
      r.value.reserve(seek->size());
      for (const auto& e : *seek) r.value.push_back(e.regen_size);
      return r;
    }
    std::vector<detail::u64> regen;
    const Status st = detail::list_block_regens(
        reinterpret_cast<const detail::u8*>(frame.data()), frame.size(), regen);
    if (st != Status::Ok) return {st, {}};
    r.value.assign(regen.begin(), regen.end());
    return r;
  }
  if (auto cs = get_frame_content_size(frame)) {
    r.value.push_back(*cs);
    return r;
  }
  return {Status::InvalidArgument, {}};
}

struct GpuTransport::Impl {
  GpuBackend backend = GpuBackend::Host;
  bool ok = true;
  void* ctx = nullptr;

  ~Impl() { close(); }

  void close() noexcept {
#if defined(SPECTRA8_HAVE_CUDA)
    if (backend == GpuBackend::Cuda && ctx) {
      detail::cuda_h2d_close(ctx);
      ctx = nullptr;
    }
#endif
#if defined(SPECTRA8_HAVE_HIP)
    if (backend == GpuBackend::Hip && ctx) {
      detail::hip_h2d_close(ctx);
      ctx = nullptr;
    }
#endif
#if defined(SPECTRA8_HAVE_METAL)
    if (backend == GpuBackend::Metal && ctx) {
      detail::metal_h2d_close(ctx);
      ctx = nullptr;
    }
#endif
  }
};

GpuTransport::GpuTransport() : GpuTransport(GpuBackend::Default) {}

GpuTransport::GpuTransport(GpuBackend backend) : impl_(std::make_unique<Impl>()) {
  if (backend == GpuBackend::Default) backend = default_gpu_backend();
  impl_->backend = backend;
  if (!gpu_transport_available(backend)) {
    impl_->ok = false;
    return;
  }
#if defined(SPECTRA8_HAVE_CUDA)
  if (backend == GpuBackend::Cuda) {
    impl_->ctx = detail::cuda_h2d_open();
    impl_->ok = impl_->ctx != nullptr;
    return;
  }
#endif
#if defined(SPECTRA8_HAVE_HIP)
  if (backend == GpuBackend::Hip) {
    impl_->ctx = detail::hip_h2d_open();
    impl_->ok = impl_->ctx != nullptr;
    return;
  }
#endif
#if defined(SPECTRA8_HAVE_METAL)
  if (backend == GpuBackend::Metal) {
    impl_->ctx = detail::metal_h2d_open();
    impl_->ok = impl_->ctx != nullptr;
    return;
  }
#endif
  impl_->backend = GpuBackend::Host;
  impl_->ok = true;
}

GpuTransport::GpuTransport(GpuTransport&& o) noexcept = default;
GpuTransport& GpuTransport::operator=(GpuTransport&& o) noexcept = default;
GpuTransport::~GpuTransport() = default;

bool GpuTransport::ok() const noexcept { return impl_ && impl_->ok; }

GpuBackend GpuTransport::backend() const noexcept {
  return impl_ ? impl_->backend : GpuBackend::Host;
}

void* GpuTransport::alloc_staging(std::size_t n) noexcept {
  if (!ok()) return nullptr;
#if defined(SPECTRA8_HAVE_CUDA)
  if (impl_->backend == GpuBackend::Cuda)
    return detail::cuda_h2d_alloc_staging(n);
#endif
#if defined(SPECTRA8_HAVE_HIP)
  if (impl_->backend == GpuBackend::Hip)
    return detail::hip_h2d_alloc_staging(n);
#endif
#if defined(SPECTRA8_HAVE_METAL)
  if (impl_->backend == GpuBackend::Metal)
    return detail::metal_h2d_alloc_staging(n);
#endif
  return host_alloc(n);
}

void GpuTransport::free_staging(void* p) noexcept {
  if (!p || !impl_) return;
#if defined(SPECTRA8_HAVE_CUDA)
  if (impl_->backend == GpuBackend::Cuda) {
    detail::cuda_h2d_free_staging(p);
    return;
  }
#endif
#if defined(SPECTRA8_HAVE_HIP)
  if (impl_->backend == GpuBackend::Hip) {
    detail::hip_h2d_free_staging(p);
    return;
  }
#endif
#if defined(SPECTRA8_HAVE_METAL)
  if (impl_->backend == GpuBackend::Metal) {
    detail::metal_h2d_free_staging(p);
    return;
  }
#endif
  host_free(p);
}

void* GpuTransport::alloc_device(std::size_t n) noexcept {
  if (!ok()) return nullptr;
#if defined(SPECTRA8_HAVE_CUDA)
  if (impl_->backend == GpuBackend::Cuda)
    return detail::cuda_h2d_alloc_device(impl_->ctx, n);
#endif
#if defined(SPECTRA8_HAVE_HIP)
  if (impl_->backend == GpuBackend::Hip)
    return detail::hip_h2d_alloc_device(impl_->ctx, n);
#endif
#if defined(SPECTRA8_HAVE_METAL)
  if (impl_->backend == GpuBackend::Metal)
    return detail::metal_h2d_alloc_device(impl_->ctx, n);
#endif
  return host_alloc(n);
}

void GpuTransport::free_device(void* p) noexcept {
  if (!p || !impl_) return;
#if defined(SPECTRA8_HAVE_CUDA)
  if (impl_->backend == GpuBackend::Cuda) {
    detail::cuda_h2d_free_device(impl_->ctx, p);
    return;
  }
#endif
#if defined(SPECTRA8_HAVE_HIP)
  if (impl_->backend == GpuBackend::Hip) {
    detail::hip_h2d_free_device(impl_->ctx, p);
    return;
  }
#endif
#if defined(SPECTRA8_HAVE_METAL)
  if (impl_->backend == GpuBackend::Metal) {
    detail::metal_h2d_free_device(impl_->ctx, p);
    return;
  }
#endif
  host_free(p);
}

Status GpuTransport::h2d_async(void* device_dst, const void* host_src,
                               std::size_t n) noexcept {
  if (!ok()) return Status::InternalError;
  if (n == 0) return Status::Ok;
  if (!device_dst || !host_src) return Status::InvalidArgument;
#if defined(SPECTRA8_HAVE_CUDA)
  if (impl_->backend == GpuBackend::Cuda)
    return detail::cuda_h2d_copy(impl_->ctx, device_dst, host_src, n, true);
#endif
#if defined(SPECTRA8_HAVE_HIP)
  if (impl_->backend == GpuBackend::Hip)
    return detail::hip_h2d_copy(impl_->ctx, device_dst, host_src, n, true);
#endif
#if defined(SPECTRA8_HAVE_METAL)
  if (impl_->backend == GpuBackend::Metal)
    return detail::metal_h2d_copy(impl_->ctx, device_dst, host_src, n, true);
#endif
  std::memcpy(device_dst, host_src, n);
  return Status::Ok;
}

Status GpuTransport::d2h(void* host_dst, const void* device_src,
                         std::size_t n) noexcept {
  if (!ok()) return Status::InternalError;
  if (n == 0) return Status::Ok;
  if (!host_dst || !device_src) return Status::InvalidArgument;
#if defined(SPECTRA8_HAVE_CUDA)
  if (impl_->backend == GpuBackend::Cuda)
    return detail::cuda_h2d_copy(impl_->ctx, host_dst, device_src, n, false);
#endif
#if defined(SPECTRA8_HAVE_HIP)
  if (impl_->backend == GpuBackend::Hip)
    return detail::hip_h2d_copy(impl_->ctx, host_dst, device_src, n, false);
#endif
#if defined(SPECTRA8_HAVE_METAL)
  if (impl_->backend == GpuBackend::Metal)
    return detail::metal_h2d_copy(impl_->ctx, host_dst, device_src, n, false);
#endif
  std::memcpy(host_dst, device_src, n);
  return Status::Ok;
}

Status GpuTransport::synchronize() noexcept {
  if (!ok()) return Status::InternalError;
#if defined(SPECTRA8_HAVE_CUDA)
  if (impl_->backend == GpuBackend::Cuda)
    return detail::cuda_h2d_sync(impl_->ctx);
#endif
#if defined(SPECTRA8_HAVE_HIP)
  if (impl_->backend == GpuBackend::Hip) return detail::hip_h2d_sync(impl_->ctx);
#endif
#if defined(SPECTRA8_HAVE_METAL)
  if (impl_->backend == GpuBackend::Metal)
    return detail::metal_h2d_sync(impl_->ctx);
#endif
  return Status::Ok;
}

struct PagePump::Impl {
  GpuTransport transport;
  PagePumpParams params;
  std::span<const std::byte> frame{};
  std::vector<std::uint64_t> pages;
  bool spectra8_pages = false;
  std::vector<void*> staging;
  std::size_t staging_cap = 0;
  int next_slot = 0;

  explicit Impl(GpuTransport t, PagePumpParams p)
      : transport(std::move(t)), params(p) {
    params.ring_slots = clamp_slots(p.ring_slots);
    if (params.batch_bytes < 4096) params.batch_bytes = 4096;
    params.unzip_threads = detail::resolve_threads(p.unzip_threads);
  }

  ~Impl() { release_staging(); }

  void release_staging() {
    for (void* p : staging) transport.free_staging(p);
    staging.clear();
    staging_cap = 0;
  }

  Status ensure_staging(std::size_t need) {
    if (need <= staging_cap &&
        staging.size() == static_cast<std::size_t>(params.ring_slots))
      return Status::Ok;
    if (Status s = transport.synchronize(); s != Status::Ok) return s;
    release_staging();
    staging.resize(static_cast<std::size_t>(params.ring_slots), nullptr);
    for (void*& p : staging) {
      p = transport.alloc_staging(need);
      if (!p) {
        release_staging();
        return Status::InternalError;
      }
    }
    staging_cap = need;
    next_slot = 0;
    return Status::Ok;
  }

  Result<std::size_t> unzip_one(std::size_t index, std::byte* dst,
                                std::size_t cap) {
    if (index >= pages.size()) return {Status::InvalidArgument, 0};
    const std::size_t need = static_cast<std::size_t>(pages[index]);
    if (need > cap) return {Status::BufferTooSmall, 0};
    auto out = std::span<std::byte>(dst, cap);
    if (spectra8_pages && pages.size() > 1)
      return decompress_block(frame, index, out);
    DecompressParams dp;
    dp.threads = params.unzip_threads;
    return decompress(frame, out, dp);
  }

  Result<std::size_t> unzip_batch(std::size_t begin, std::size_t end,
                                  std::byte* dst) {
    if (end <= begin || end > pages.size())
      return {Status::InvalidArgument, 0};
    if (!spectra8_pages || pages.size() == 1) {
      if (begin != 0 || end != pages.size())
        return {Status::InvalidArgument, 0};
      return unzip_one(0, dst, staging_cap);
    }
    std::vector<std::size_t> off(end - begin);
    std::size_t acc = 0;
    for (std::size_t i = begin; i < end; ++i) {
      off[i - begin] = acc;
      acc += static_cast<std::size_t>(pages[i]);
    }
    std::atomic<int> err{static_cast<int>(Status::Ok)};
    const int threads = (end - begin > 1) ? params.unzip_threads : 1;
    detail::parallel_for(end - begin, threads, [&](detail::usize i) {
      if (err.load(std::memory_order_relaxed) != 0) return;
      const std::size_t idx = begin + i;
      const std::size_t n = static_cast<std::size_t>(pages[idx]);
      auto r = decompress_block(frame, idx, std::span<std::byte>(dst + off[i], n));
      if (!r.ok()) {
        int expected = 0;
        err.compare_exchange_strong(expected, static_cast<int>(r.status));
      }
    });
    if (err.load() != 0)
      return {static_cast<Status>(err.load()), 0};
    return {Status::Ok, acc};
  }
};

PagePump::PagePump(GpuBackend backend, PagePumpParams params)
    : impl_(std::make_unique<Impl>(GpuTransport(backend), params)) {}

PagePump::PagePump(GpuTransport transport, PagePumpParams params)
    : impl_(std::make_unique<Impl>(std::move(transport), params)) {}

PagePump::PagePump(PagePump&&) noexcept = default;
PagePump& PagePump::operator=(PagePump&&) noexcept = default;
PagePump::~PagePump() = default;

bool PagePump::ok() const noexcept {
  return impl_ && impl_->transport.ok();
}

GpuTransport& PagePump::transport() noexcept { return impl_->transport; }
const GpuTransport& PagePump::transport() const noexcept {
  return impl_->transport;
}

Status PagePump::bind(std::span<const std::byte> frame) {
  if (!ok()) return Status::InternalError;
  auto sizes = get_page_sizes(frame);
  if (!sizes.ok()) return sizes.status;
  impl_->frame = frame;
  impl_->pages = std::move(sizes.value);
  impl_->spectra8_pages = looks_like_spectra8(frame);
  return Status::Ok;
}

std::size_t PagePump::page_count() const noexcept {
  return impl_ ? impl_->pages.size() : 0;
}

std::size_t PagePump::page_bytes(std::size_t index) const noexcept {
  if (!impl_ || index >= impl_->pages.size()) return 0;
  return static_cast<std::size_t>(impl_->pages[index]);
}

std::size_t PagePump::total_bytes() const noexcept {
  if (!impl_) return 0;
  std::size_t n = 0;
  for (auto s : impl_->pages) n += static_cast<std::size_t>(s);
  return n;
}

Result<std::size_t> PagePump::push(std::size_t index, void* device_dst) {
  return push_range(index, index + 1, device_dst);
}

Result<std::size_t> PagePump::push_all(void* device_dst) {
  return push_range(0, page_count(), device_dst);
}

Result<std::size_t> PagePump::push_range(std::size_t begin, std::size_t end,
                                         void* device_dst) {
  if (!ok()) return {Status::InternalError, 0};
  if (end < begin || end > impl_->pages.size())
    return {Status::InvalidArgument, 0};
  if (begin == end) return {Status::Ok, 0};
  if (!device_dst) return {Status::InvalidArgument, 0};

  std::size_t max_batch = 0;
  std::size_t total = 0;
  std::size_t run = 0;
  for (std::size_t i = begin; i < end; ++i) {
    const std::size_t n = static_cast<std::size_t>(impl_->pages[i]);
    total += n;
    if (run > 0 && run + n > impl_->params.batch_bytes) {
      if (run > max_batch) max_batch = run;
      run = 0;
    }
    run += n;
  }
  if (run > max_batch) max_batch = run;
  if (max_batch == 0) max_batch = total;
  if (Status s = impl_->ensure_staging(max_batch); s != Status::Ok)
    return {s, 0};

  std::byte* dev = static_cast<std::byte*>(device_dst);
  std::size_t written = 0;
  std::size_t i = begin;
  while (i < end) {
    std::size_t batch_end = i;
    std::size_t batch_n = 0;
    while (batch_end < end) {
      const std::size_t n = static_cast<std::size_t>(impl_->pages[batch_end]);
      if (batch_n > 0 && batch_n + n > impl_->params.batch_bytes) break;
      batch_n += n;
      ++batch_end;
    }
    const int slot = impl_->next_slot;
    impl_->next_slot = (impl_->next_slot + 1) % impl_->params.ring_slots;
    auto* host = static_cast<std::byte*>(impl_->staging[static_cast<std::size_t>(slot)]);
    auto uz = impl_->unzip_batch(i, batch_end, host);
    if (!uz.ok()) return uz;
    if (Status s = impl_->transport.h2d_async(dev + written, host, uz.value);
        s != Status::Ok)
      return {s, written};
    written += uz.value;
    i = batch_end;
    // Reusing a ring slot requires the previous H2D on that slot to finish.
    // Synchronize every full revolution (cheap on Host; needed on CUDA/HIP).
    if (impl_->next_slot == 0) {
      if (Status s = impl_->transport.synchronize(); s != Status::Ok)
        return {s, written};
    }
  }
  if (Status s = impl_->transport.synchronize(); s != Status::Ok)
    return {s, written};
  return {Status::Ok, written};
}

Result<std::size_t> decompress_to_device(std::span<const std::byte> frame,
                                         GpuTransport& transport,
                                         void* device_dst, std::size_t dst_cap,
                                         const PagePumpParams& params) {
  if (!transport.ok()) return {Status::InternalError, 0};
  if (!device_dst && dst_cap != 0) return {Status::InvalidArgument, 0};
  PagePump pump(transport.backend(), params);
  if (!pump.ok()) return {Status::InternalError, 0};
  if (Status s = pump.bind(frame); s != Status::Ok) return {s, 0};
  if (pump.total_bytes() > dst_cap) return {Status::BufferTooSmall, 0};
  return pump.push_all(device_dst);
}

}  // namespace spectra8

// ---------------------------------------------------------------------------
// C ABI
// ---------------------------------------------------------------------------
extern "C" {

struct spectra8_page_pump {
  spectra8::PagePump pump;
};

static spectra8::GpuBackend from_c_backend(spectra8_gpu_backend b) {
  switch (b) {
    case SPECTRA8_GPU_HOST:
      return spectra8::GpuBackend::Host;
    case SPECTRA8_GPU_CUDA:
      return spectra8::GpuBackend::Cuda;
    case SPECTRA8_GPU_HIP:
      return spectra8::GpuBackend::Hip;
    case SPECTRA8_GPU_METAL:
      return spectra8::GpuBackend::Metal;
    case SPECTRA8_GPU_DEFAULT:
    default:
      return spectra8::GpuBackend::Default;
  }
}

int spectra8_gpu_transport_available(spectra8_gpu_backend backend) {
  return spectra8::gpu_transport_available(from_c_backend(backend)) ? 1 : 0;
}

const char* spectra8_gpu_backend_name(spectra8_gpu_backend backend) {
  return spectra8::gpu_backend_name(from_c_backend(backend));
}

spectra8_page_pump* spectra8_page_pump_create(spectra8_gpu_backend backend,
                                                int unzip_threads,
                                                int ring_slots) {
  spectra8::PagePumpParams p;
  p.unzip_threads = unzip_threads;
  if (ring_slots != 0) p.ring_slots = ring_slots;
  auto* wrap = new (std::nothrow) spectra8_page_pump{
      spectra8::PagePump(from_c_backend(backend), p)};
  if (!wrap || !wrap->pump.ok()) {
    delete wrap;
    return nullptr;
  }
  return wrap;
}

void spectra8_page_pump_destroy(spectra8_page_pump* p) { delete p; }

void* spectra8_page_pump_alloc_device(spectra8_page_pump* p, size_t n) {
  if (!p) return nullptr;
  return p->pump.transport().alloc_device(n);
}

void spectra8_page_pump_free_device(spectra8_page_pump* p, void* d) {
  if (!p) return;
  p->pump.transport().free_device(d);
}

spectra8_status spectra8_page_pump_push_all(spectra8_page_pump* p,
                                              const void* frame,
                                              size_t frame_size,
                                              void* device_dst,
                                              size_t dst_capacity,
                                              size_t* written) {
  if (!p || !frame) return SPECTRA8_INVALID_ARGUMENT;
  auto src = std::span<const std::byte>(static_cast<const std::byte*>(frame),
                                        frame_size);
  if (auto s = p->pump.bind(src); s != spectra8::Status::Ok)
    return static_cast<spectra8_status>(static_cast<int>(s));
  if (p->pump.total_bytes() > dst_capacity) return SPECTRA8_BUFFER_TOO_SMALL;
  auto r = p->pump.push_all(device_dst);
  if (written) *written = r.value;
  return static_cast<spectra8_status>(static_cast<int>(r.status));
}

}  // extern "C"
