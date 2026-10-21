# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 CERN
# Author: Andreas Joachim Peters <andreas.joachim.peters@cern.ch>
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain
# a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.20)

# C is needed only for the vendored libsais (third_party/libsais).
project(spectra8
  VERSION 2.0.0
  DESCRIPTION "Fast modern-C++ LZ + entropy compressor (Spectra8 container format)"
  LANGUAGES C CXX)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(SPECTRA8_BUILD_CLI    "Build the spectra8 command-line tool" ON)
option(SPECTRA8_BUILD_TESTS  "Build tests"                            ON)
option(SPECTRA8_USE_SSE2NEON "Use vendored sse2neon.h on arm64"       OFF)
option(SPECTRA8_NATIVE       "Optimize for the build host (-march=native)" OFF)
# LTO is opt-in: on some toolchains (notably Apple clang/arm64) it changes
# inlining in the match-finder hot loop and measurably slows compression while
# not helping decompression. Benchmark before enabling it for your platform.
option(SPECTRA8_LTO          "Enable link-time optimization in Release"    OFF)
# Optimization level for the codec objects (non-MSVC). -O3 is the historical
# default, but -O2 is sometimes faster for hand-tuned SIMD/hot-loop code because
# it avoids -O3's extra unrolling/vectorization bloat. Benchmark both on your
# target (e.g. -DSPECTRA8_OPT_LEVEL=2). Accepts a -O suffix: 2, 3, fast, s, ...
set(SPECTRA8_OPT_LEVEL "3" CACHE STRING "Optimization level for codec objects (2, 3, fast, s)")
# Metal GPU match finder. Apple-only; compiles the MSL kernel at runtime, so no
# offline `metal` toolchain is needed (just the Metal + Foundation frameworks).
if(APPLE)
  option(SPECTRA8_METAL "Build the Metal GPU match finder (Apple only)" ON)
else()
  set(SPECTRA8_METAL OFF)
endif()
# HIP GPU match finder for NVIDIA/AMD GPUs. Off by default; requires a ROCm/HIP
# toolchain. Enable with -DSPECTRA8_HIP=ON on a machine with HIP installed.
option(SPECTRA8_HIP "Build the HIP GPU match finder (NVIDIA/AMD, needs ROCm/HIP)" OFF)
# CUDA H2D transport for the CPU page pump (no decoder kernels). AUTO links
# CUDA if a toolkit is found; ON requires it; OFF disables it.
set(SPECTRA8_CUDA "AUTO" CACHE STRING "CUDA H2D transport (AUTO/ON/OFF)")
set_property(CACHE SPECTRA8_CUDA PROPERTY STRINGS AUTO ON OFF)
# Optional zstd backend for the library API (Codec::Zstd) and CLI/bench. AUTO
# links zstd if found; ON requires it; OFF disables it.
set(SPECTRA8_ZSTD "AUTO" CACHE STRING "Link zstd backend into libspectra8 (AUTO/ON/OFF)")
set_property(CACHE SPECTRA8_ZSTD PROPERTY STRINGS AUTO ON OFF)

# Optional liblzma backend for Codec::Lzma / Codec::Auto and CLI verdict rows.
set(SPECTRA8_LZMA "AUTO" CACHE STRING
  "Link liblzma backend into libspectra8 (AUTO/ON/OFF)")
set_property(CACHE SPECTRA8_LZMA PROPERTY STRINGS AUTO ON OFF)

# Optional libbzip3 backend for the library API (Codec::Bzip3) and CLI/bench.
# Default AUTO: dynamically links a system-installed shared libbzip3 if present,
# but never compiles or embeds the LGPL-3.0 third_party/bzip3 submodule,
# ensuring the build is 100% pure Apache-2.0.
# BUNDLED: compile and statically link the LGPL-3.0 third_party/bzip3 submodule
# (or SPECTRA8_BZIP3_SOURCE). Sets package license to "Apache-2.0 AND LGPL-3.0-or-later".
# SYSTEM: require a system-installed dynamic libbzip3 (pure Apache-2.0).
# ON: require a system libbzip3.
# OFF: disable bzip3 entirely.
# SPECTRA8_BZIP3_SOURCE: optional path to a bzip3 source tree (src/libbz3.c +
# include/libbz3.h). When set, that tree is used.
set(SPECTRA8_BZIP3 "AUTO" CACHE STRING
  "Link libbzip3 backend into libspectra8 (AUTO/ON/OFF/BUNDLED/SYSTEM)")
set_property(CACHE SPECTRA8_BZIP3 PROPERTY STRINGS AUTO ON OFF BUNDLED SYSTEM)
set(SPECTRA8_BZIP3_SOURCE "" CACHE PATH
  "Optional path to a bzip3 source tree (src/libbz3.c + include/libbz3.h)")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_OBJCXX_STANDARD 20)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# ---------------------------------------------------------------------------
# Metal (optional GPU acceleration)
# ---------------------------------------------------------------------------
set(SPECTRA8_METAL_ENABLED OFF)
if(SPECTRA8_METAL AND APPLE)
  enable_language(OBJCXX)
  find_library(METAL_FRAMEWORK Metal)
  find_library(FOUNDATION_FRAMEWORK Foundation)
  if(METAL_FRAMEWORK AND FOUNDATION_FRAMEWORK)
    set(SPECTRA8_METAL_ENABLED ON)
    message(STATUS "spectra8: Metal GPU match finder enabled")
  else()
    message(STATUS "spectra8: Metal/Foundation not found; GPU path disabled")
  endif()
endif()

# HIP backend (mutually compatible with Metal, but normally used on non-Apple).
set(SPECTRA8_HIP_ENABLED OFF)
if(SPECTRA8_HIP)
  find_package(hip QUIET)
  if(hip_FOUND)
    enable_language(HIP)
    set(CMAKE_HIP_STANDARD 20)
    set(CMAKE_HIP_STANDARD_REQUIRED ON)
    set(SPECTRA8_HIP_ENABLED ON)
    message(STATUS "spectra8: HIP GPU match finder enabled")
  else()
    message(STATUS "spectra8: HIP requested but not found (need ROCm/HIP); GPU path disabled")
  endif()
endif()

# CUDA runtime for the page-pump H2D path (regular C++ + cudart; no nvcc).
set(SPECTRA8_CUDA_ENABLED OFF)
if(NOT SPECTRA8_CUDA STREQUAL "OFF")
  find_package(CUDAToolkit QUIET)
  if(CUDAToolkit_FOUND)
    set(SPECTRA8_CUDA_ENABLED ON)
    message(STATUS "spectra8: CUDA H2D transport enabled")
  elseif(SPECTRA8_CUDA STREQUAL "ON")
    message(FATAL_ERROR "spectra8: SPECTRA8_CUDA=ON but CUDA toolkit was not found")
  else()
    message(STATUS "spectra8: CUDA not found; H2D CUDA transport unavailable")
  endif()
endif()

# zstd (optional library backend + CLI/bench). Located in three ways, in
# order, so a plain system install (e.g. /usr/include/zstd.h + libzstd.so on
# Linux) is found even without a CMake config package:
#   1. find_package(zstd CONFIG)   - vcpkg/conan/recent zstd installs
#   2. pkg-config (libzstd.pc)     - most Linux distro -dev packages
#   3. find_path + find_library    - bare system header/library
set(SPECTRA8_ZSTD_ENABLED OFF)
set(SPECTRA8_ZSTD_TARGET "")
if(NOT SPECTRA8_ZSTD STREQUAL "OFF")
  find_package(zstd CONFIG QUIET)
  if(zstd_FOUND)
    # Prefer the version-agnostic interface target; fall back to the static or
    # shared imported targets exported by older zstd config packages.
    if(TARGET zstd::libzstd)
      set(SPECTRA8_ZSTD_TARGET zstd::libzstd)
    elseif(TARGET zstd::libzstd_shared)
      set(SPECTRA8_ZSTD_TARGET zstd::libzstd_shared)
    elseif(TARGET zstd::libzstd_static)
      set(SPECTRA8_ZSTD_TARGET zstd::libzstd_static)
    endif()
  endif()

  # 2. pkg-config (most Linux distributions ship libzstd.pc with libzstd-dev).
  if(NOT SPECTRA8_ZSTD_TARGET)
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
      pkg_check_modules(ZSTD_PC QUIET IMPORTED_TARGET libzstd)
      if(TARGET PkgConfig::ZSTD_PC)
        set(SPECTRA8_ZSTD_TARGET PkgConfig::ZSTD_PC)
      endif()
    endif()
  endif()

  # 3. Bare system install: search the standard header/library locations.
  if(NOT SPECTRA8_ZSTD_TARGET)
    find_path(ZSTD_INCLUDE_DIR NAMES zstd.h)
    find_library(ZSTD_LIBRARY NAMES zstd libzstd)
    if(ZSTD_INCLUDE_DIR AND ZSTD_LIBRARY)
      add_library(spectra8_zstd INTERFACE)
      target_include_directories(spectra8_zstd INTERFACE ${ZSTD_INCLUDE_DIR})
      target_link_libraries(spectra8_zstd INTERFACE ${ZSTD_LIBRARY})
      set(SPECTRA8_ZSTD_TARGET spectra8_zstd)
    endif()
  endif()

  if(SPECTRA8_ZSTD_TARGET)
    set(SPECTRA8_ZSTD_ENABLED ON)
    message(STATUS "spectra8: zstd found; Codec::Zstd enabled (${SPECTRA8_ZSTD_TARGET})")
  elseif(SPECTRA8_ZSTD STREQUAL "ON")
    message(FATAL_ERROR "spectra8: SPECTRA8_ZSTD=ON but zstd was not found via "
                        "config package, pkg-config, or system paths")
  else()
    message(STATUS "spectra8: zstd not found; Codec::Zstd unavailable")
  endif()
endif()

# liblzma / xz (optional Codec::Lzma backend). Located via pkg-config
# (liblzma.pc) first, then a bare header/library search.
set(SPECTRA8_LZMA_ENABLED OFF)
set(SPECTRA8_LZMA_TARGET "")
if(NOT SPECTRA8_LZMA STREQUAL "OFF")
  find_package(PkgConfig QUIET)
  if(PkgConfig_FOUND)
    pkg_check_modules(LZMA_PC QUIET IMPORTED_TARGET liblzma)
    if(TARGET PkgConfig::LZMA_PC)
      set(SPECTRA8_LZMA_TARGET PkgConfig::LZMA_PC)
    endif()
  endif()

  if(NOT SPECTRA8_LZMA_TARGET)
    find_path(LZMA_INCLUDE_DIR NAMES lzma.h)
    find_library(LZMA_LIBRARY NAMES lzma liblzma)
    if(LZMA_INCLUDE_DIR AND LZMA_LIBRARY)
      add_library(spectra8_lzma INTERFACE)
      target_include_directories(spectra8_lzma INTERFACE ${LZMA_INCLUDE_DIR})
      target_link_libraries(spectra8_lzma INTERFACE ${LZMA_LIBRARY})
      set(SPECTRA8_LZMA_TARGET spectra8_lzma)
    endif()
  endif()

  if(SPECTRA8_LZMA_TARGET)
    set(SPECTRA8_LZMA_ENABLED ON)
    message(STATUS "spectra8: liblzma found; Codec::Lzma enabled (${SPECTRA8_LZMA_TARGET})")
  elseif(SPECTRA8_LZMA STREQUAL "ON")
    message(FATAL_ERROR "spectra8: SPECTRA8_LZMA=ON but liblzma was not found via "
                        "pkg-config or system paths")
  else()
    message(STATUS "spectra8: liblzma not found; Codec::Lzma unavailable")
  endif()
endif()

# libbzip3 backend.
# Default AUTO and SYSTEM: system shared library only (_find_system_bzip3()).
#   Never compiles or statically links the LGPL-3.0 third_party/bzip3 submodule,
#   so libspectra8 and libspectra8.a are 100% pure Apache-2.0 binaries.
# BUNDLED:
#   Explicit opt-in to compile and statically link the LGPL-3.0 third_party/bzip3
#   submodule (or SPECTRA8_BZIP3_SOURCE), setting package license to
#   "Apache-2.0 AND LGPL-3.0-or-later".
set(SPECTRA8_BZIP3_ENABLED OFF)
set(SPECTRA8_BZIP3_TARGET "")
set(SPECTRA8_BZIP3_IS_BUNDLED OFF)

# Build libbzip3 from a source tree at `_src_root`. Sets SPECTRA8_BZIP3_* on success.
function(spectra8_try_bzip3_source _src_root _label)
  set(_c "${_src_root}/src/libbz3.c")
  set(_h "${_src_root}/include/libbz3.h")
  if(NOT EXISTS "${_c}" OR NOT EXISTS "${_h}")
    return()
  endif()
  # Prefer project(... VERSION x.y.z) over cmake_minimum_required(VERSION …).
  set(_ver "1.5.3")
  if(EXISTS "${_src_root}/CMakeLists.txt")
    file(READ "${_src_root}/CMakeLists.txt" _bzip3_cmake)
    if(_bzip3_cmake MATCHES
        "project[ \t]*\\([^)]*VERSION[ \t]+([0-9]+\\.[0-9]+(\\.[0-9]+)?)")
      set(_ver "${CMAKE_MATCH_1}")
    endif()
  endif()
  if(NOT TARGET spectra8_bzip3_bundled)
    add_library(spectra8_bzip3_bundled STATIC "${_c}")
    set_target_properties(spectra8_bzip3_bundled PROPERTIES
      POSITION_INDEPENDENT_CODE ON
      C_STANDARD 11
      C_STANDARD_REQUIRED ON)
    target_include_directories(spectra8_bzip3_bundled PUBLIC
      "${_src_root}/include")
    # libbz3.c expects VERSION as a string literal (see bz3_version()).
    target_compile_definitions(spectra8_bzip3_bundled PRIVATE VERSION=\"${_ver}\")
  endif()
  set(SPECTRA8_BZIP3_ENABLED ON PARENT_SCOPE)
  set(SPECTRA8_BZIP3_TARGET spectra8_bzip3_bundled PARENT_SCOPE)
  set(SPECTRA8_BZIP3_IS_BUNDLED ON PARENT_SCOPE)
  message(STATUS "spectra8: using bundled bzip3 from ${_label} (${_src_root}, v${_ver}); "
                 "Codec::Bzip3 enabled (LGPL-3.0-or-later)")
endfunction()

# Look for a system-installed shared libbzip3 library.
# Only shared dynamic libraries (.so / .dylib) are accepted to ensure
# libspectra8 and libspectra8.a remain 100% pure Apache-2.0.
function(_find_system_bzip3)
  set(_bzip3_include "")
  set(_bzip3_libs "")
  set(_bzip3_libdirs "")

  find_package(PkgConfig QUIET)
  if(PkgConfig_FOUND)
    pkg_check_modules(BZIP3_PC QUIET bzip3)
    if(BZIP3_PC_FOUND)
      set(_bzip3_include "${BZIP3_PC_INCLUDE_DIRS}")
      set(_bzip3_libdirs "${BZIP3_PC_LIBRARY_DIRS}")
      # Sanitize pkg-config includes if broken (e.g. Homebrew's unexpanded @extra_cflags@)
      set(_sanitized_inc "")
      foreach(_dir IN LISTS _bzip3_include)
        string(REPLACE "@extra_cflags@" "" _dir "${_dir}")
        if(EXISTS "${_dir}")
          list(APPEND _sanitized_inc "${_dir}")
        endif()
      endforeach()
      set(_bzip3_include "${_sanitized_inc}")
      # Use the full pkg-config --libs line; LIBRARIES alone is often just
      # "bzip3" and does not always propagate to the final link on EL9/Alma.
      if(BZIP3_PC_LDFLAGS)
        separate_arguments(_bzip3_libs NATIVE_COMMAND "${BZIP3_PC_LDFLAGS}")
      elseif(BZIP3_PC_LINK_LIBRARIES)
        set(_bzip3_libs "${BZIP3_PC_LINK_LIBRARIES}")
      else()
        set(_bzip3_libs "-lbzip3")
      endif()
    endif()
  endif()

  if(NOT _bzip3_include)
    find_path(BZIP3_INCLUDE_DIR NAMES libbz3.h)
    if(BZIP3_INCLUDE_DIR)
      set(_bzip3_include "${BZIP3_INCLUDE_DIR}")
    endif()
  endif()

  # Enforce searching for shared libraries only (.dylib / .so / .dll.a)
  set(_saved_lib_suffixes "${CMAKE_FIND_LIBRARY_SUFFIXES}")
  if(APPLE)
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".dylib" ".so")
  elseif(WIN32)
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".dll.a")
  else()
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".so")
  endif()
  find_library(BZIP3_LIBRARY NAMES bzip3 libbzip3
    HINTS ${_bzip3_libdirs}
    PATHS /usr/lib64 /usr/lib /lib64 /lib)
  set(CMAKE_FIND_LIBRARY_SUFFIXES "${_saved_lib_suffixes}")

  if(BZIP3_LIBRARY)
    if(BZIP3_LIBRARY MATCHES "\\.a$")
      message(STATUS "spectra8: ignoring static bzip3 library ${BZIP3_LIBRARY} to maintain pure Apache-2.0 binary")
      return()
    endif()
    set(_bzip3_libs "${BZIP3_LIBRARY}")
    get_filename_component(_bzip3_libdir "${BZIP3_LIBRARY}" DIRECTORY)
    if(_bzip3_libdir)
      set(_bzip3_libdirs "${_bzip3_libdir}")
    endif()
  endif()

  # Check that none of the link libraries are static archives (.a)
  foreach(_lib IN LISTS _bzip3_libs)
    if(_lib MATCHES "\\.a$")
      message(STATUS "spectra8: ignoring static bzip3 library ${_lib} to maintain pure Apache-2.0 binary")
      return()
    endif()
  endforeach()

  if(_bzip3_include AND _bzip3_libs)
    include(CMakePushCheckState)
    include(CheckCXXSymbolExists)
    cmake_push_check_state(RESET)
    set(CMAKE_REQUIRED_INCLUDES ${_bzip3_include})
    if(_bzip3_libdirs)
      foreach(_bzip3_ldir IN LISTS _bzip3_libdirs)
        list(APPEND CMAKE_REQUIRED_LINK_OPTIONS "-L${_bzip3_ldir}")
      endforeach()
    endif()
    foreach(_bzip3_link IN LISTS _bzip3_libs)
      if(_bzip3_link MATCHES "^-L")
        list(APPEND CMAKE_REQUIRED_LINK_OPTIONS "${_bzip3_link}")
      elseif(_bzip3_link MATCHES "^-l")
        string(SUBSTRING "${_bzip3_link}" 2 -1 _bzip3_lname)
        list(APPEND CMAKE_REQUIRED_LIBRARIES "${_bzip3_lname}")
      elseif(_bzip3_link MATCHES "^/")
        list(APPEND CMAKE_REQUIRED_LIBRARIES "${_bzip3_link}")
      endif()
    endforeach()
    # CXX-only project: CheckSymbolExists uses the C compiler and fails even
    # when libbzip3.so is present; CheckCXXSymbolExists matches the link lang.
    check_cxx_symbol_exists(bz3_compress "libbz3.h" _spectra8_bzip3_links)
    cmake_pop_check_state()
    if(NOT _spectra8_bzip3_links AND BZIP3_LIBRARY AND EXISTS "${BZIP3_LIBRARY}")
      message(STATUS "spectra8: bzip3 link probe failed; using ${BZIP3_LIBRARY} anyway")
      set(_spectra8_bzip3_links TRUE)
    endif()
    if(_spectra8_bzip3_links)
      if(NOT TARGET spectra8_bzip3)
        add_library(spectra8_bzip3 INTERFACE)
        target_include_directories(spectra8_bzip3 INTERFACE "${_bzip3_include}")
        if(_bzip3_libdirs)
          target_link_directories(spectra8_bzip3 INTERFACE ${_bzip3_libdirs})
        endif()
        target_link_libraries(spectra8_bzip3 INTERFACE ${_bzip3_libs})
      endif()
      set(SPECTRA8_BZIP3_TARGET spectra8_bzip3 PARENT_SCOPE)
      set(SPECTRA8_BZIP3_ENABLED ON PARENT_SCOPE)
      set(SPECTRA8_BZIP3_IS_BUNDLED OFF PARENT_SCOPE)
    else()
      message(STATUS "spectra8: libbz3.h found but libbzip3 does not link")
    endif()
  endif()
endfunction()

if(SPECTRA8_BZIP3_SOURCE)
  get_filename_component(_bzip3_user_src "${SPECTRA8_BZIP3_SOURCE}" ABSOLUTE)
  if(EXISTS "${_bzip3_user_src}/src/libbz3.c" AND
     EXISTS "${_bzip3_user_src}/include/libbz3.h")
    spectra8_try_bzip3_source("${_bzip3_user_src}" "SPECTRA8_BZIP3_SOURCE")
  else()
    message(FATAL_ERROR
      "spectra8: SPECTRA8_BZIP3_SOURCE=${SPECTRA8_BZIP3_SOURCE} is not a "
      "bzip3 source tree (need src/libbz3.c and include/libbz3.h)")
  endif()
elseif(SPECTRA8_BZIP3 STREQUAL "BUNDLED")
  set(_spectra8_bzip3_src "${CMAKE_CURRENT_SOURCE_DIR}/third_party/bzip3")
  set(_spectra8_bzip3_c "${_spectra8_bzip3_src}/src/libbz3.c")
  set(_spectra8_bzip3_h "${_spectra8_bzip3_src}/include/libbz3.h")
  if(NOT EXISTS "${_spectra8_bzip3_c}" OR NOT EXISTS "${_spectra8_bzip3_h}")
    find_package(Git QUIET)
    if(Git_FOUND AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.gitmodules")
      message(STATUS "spectra8: fetching bzip3 git submodule into third_party/bzip3")
      execute_process(
        COMMAND ${GIT_EXECUTABLE} submodule update --init --depth 1
                -- third_party/bzip3
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        RESULT_VARIABLE _spectra8_bzip3_sub_rc
        OUTPUT_VARIABLE _spectra8_bzip3_sub_out
        ERROR_VARIABLE _spectra8_bzip3_sub_err)
      if(NOT _spectra8_bzip3_sub_rc EQUAL 0)
        message(STATUS "spectra8: git submodule update failed: ${_spectra8_bzip3_sub_err}")
      endif()
    endif()
  endif()
  spectra8_try_bzip3_source("${_spectra8_bzip3_src}" "git submodule")
  if(NOT SPECTRA8_BZIP3_ENABLED)
    message(FATAL_ERROR
      "spectra8: SPECTRA8_BZIP3=BUNDLED requested but third_party/bzip3 was not found.\n"
      "  Run: git submodule update --init -- third_party/bzip3\n"
      "  Or pass: -DSPECTRA8_BZIP3_SOURCE=/path/to/bzip3")
  endif()
elseif(SPECTRA8_BZIP3 STREQUAL "AUTO" OR SPECTRA8_BZIP3 STREQUAL "SYSTEM" OR SPECTRA8_BZIP3 STREQUAL "ON")
  _find_system_bzip3()
  if(SPECTRA8_BZIP3_ENABLED)
    message(STATUS "spectra8: using system dynamic libbzip3; Codec::Bzip3 enabled (pure Apache-2.0)")
  elseif(SPECTRA8_BZIP3 STREQUAL "SYSTEM" OR SPECTRA8_BZIP3 STREQUAL "ON")
    message(FATAL_ERROR
      "spectra8: SPECTRA8_BZIP3=${SPECTRA8_BZIP3} but system shared libbzip3 was not found.\n"
      "  Install the system bzip3 shared library and headers (on macOS: brew install bzip3).\n"
      "  For bundled LGPL-3.0 static build: -DSPECTRA8_BZIP3=BUNDLED")
  else()
    message(STATUS "spectra8: system libbzip3 not found; Codec::Bzip3 unavailable "
                   "(default builds are pure Apache-2.0; pass -DSPECTRA8_BZIP3=BUNDLED "
                   "to compile with the LGPL-3.0 submodule, or install the system bzip3 library)")
  endif()
elseif(SPECTRA8_BZIP3 STREQUAL "OFF")
  # Disabled
else()
  message(FATAL_ERROR "spectra8: invalid SPECTRA8_BZIP3 value '${SPECTRA8_BZIP3}'. "
                      "Expected AUTO, ON, OFF, BUNDLED, or SYSTEM.")
endif()

# ---------------------------------------------------------------------------
# Architecture detection / SIMD configuration
# ---------------------------------------------------------------------------
set(SPECTRA8_ARCH "${CMAKE_SYSTEM_PROCESSOR}")
string(TOLOWER "${SPECTRA8_ARCH}" SPECTRA8_ARCH)

set(SPECTRA8_PRIVATE_DEFS "")
set(SPECTRA8_PRIVATE_INCLUDES "")

if(SPECTRA8_ARCH MATCHES "arm64|aarch64")
  if(SPECTRA8_USE_SSE2NEON)
    # Fetch sse2neon so SSE intrinsics map to NEON. Optional: the codec also
    # builds without it via portable wide-copy helpers.
    include(FetchContent)
    FetchContent_Declare(sse2neon
      GIT_REPOSITORY https://github.com/DLTcollab/sse2neon.git
      GIT_TAG master)
    FetchContent_MakeAvailable(sse2neon)
    list(APPEND SPECTRA8_PRIVATE_DEFS SPECTRA8_USE_SSE2NEON)
    list(APPEND SPECTRA8_PRIVATE_INCLUDES "${sse2neon_SOURCE_DIR}")
  endif()
elseif(SPECTRA8_ARCH MATCHES "x86_64|amd64|i.86")
  if(NOT MSVC)
    list(APPEND SPECTRA8_PRIVATE_FLAGS -mssse3 -msse4.1)
  endif()
endif()

if(SPECTRA8_NATIVE AND NOT MSVC)
  list(APPEND SPECTRA8_PRIVATE_FLAGS -march=native)
endif()

# Link-time optimization (Release only), when the toolchain supports it.
set(SPECTRA8_IPO OFF)
if(SPECTRA8_LTO)
  include(CheckIPOSupported)
  check_ipo_supported(RESULT SPECTRA8_IPO OUTPUT SPECTRA8_IPO_MSG)
  if(NOT SPECTRA8_IPO)
    message(STATUS "spectra8: IPO/LTO not supported (${SPECTRA8_IPO_MSG})")
  endif()
endif()

# ---------------------------------------------------------------------------
# Library (built as both a shared and a static target from one object library)
# ---------------------------------------------------------------------------
set(SPECTRA8_SOURCES
  src/util/huffman.cpp
  src/util/tans.cpp
  src/util/arith.cpp
  src/util/entropy.cpp
  src/prism/bwt.cpp
  src/prism/bwt_cm.cpp
  src/prism/bwt_ultra.cpp
  src/prism/lzp.cpp
  src/mosaic/mosaic.cpp
  src/oracle/oracle.cpp
  src/crystal/crystal.cpp
  src/crystal/crystal_static.cpp
  src/crystal/crystal_bwt.cpp
  src/beam/match_finder.cpp
  src/laser/laser.cpp
  src/grating/grating.cpp
  src/encoder.cpp
  src/decoder.cpp
  src/spectra8.cpp
  src/page_service.cpp
  src/ext/zstd.cpp
  src/ext/bzip3.cpp
  src/ext/lzma.cpp
  src/ext/filter_frame.cpp
  src/util/stats.cpp
  third_party/libsais/libsais.c)

if(SPECTRA8_METAL_ENABLED)
  list(APPEND SPECTRA8_SOURCES src/gpu/metal_lz.mm src/gpu/metal_h2d.mm)
  list(APPEND SPECTRA8_PRIVATE_DEFS SPECTRA8_HAVE_METAL SPECTRA8_HAVE_GPU)
endif()
if(SPECTRA8_HIP_ENABLED)
  list(APPEND SPECTRA8_SOURCES src/gpu/hip_lz.cpp src/gpu/hip_h2d.cpp)
  list(APPEND SPECTRA8_PRIVATE_DEFS SPECTRA8_HAVE_HIP SPECTRA8_HAVE_GPU)
endif()
if(SPECTRA8_CUDA_ENABLED)
  list(APPEND SPECTRA8_SOURCES src/gpu/cuda_h2d.cpp)
  list(APPEND SPECTRA8_PRIVATE_DEFS SPECTRA8_HAVE_CUDA)
endif()
if(SPECTRA8_ZSTD_ENABLED)
  list(APPEND SPECTRA8_PRIVATE_DEFS SPECTRA8_HAVE_ZSTD)
endif()
if(SPECTRA8_BZIP3_ENABLED)
  list(APPEND SPECTRA8_PRIVATE_DEFS SPECTRA8_HAVE_BZIP3)
endif()
if(SPECTRA8_LZMA_ENABLED)
  list(APPEND SPECTRA8_PRIVATE_DEFS SPECTRA8_HAVE_LZMA)
endif()

add_library(spectra8_objects OBJECT ${SPECTRA8_SOURCES})
target_include_directories(spectra8_objects
  PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
          ${CMAKE_CURRENT_SOURCE_DIR}/third_party/libsais
          ${SPECTRA8_PRIVATE_INCLUDES})
target_compile_definitions(spectra8_objects PRIVATE ${SPECTRA8_PRIVATE_DEFS})
if(SPECTRA8_ZSTD_ENABLED)
  # Propagate include dirs into the object compile; final link is on the
  # shared/static targets below.
  target_link_libraries(spectra8_objects PRIVATE
    $<BUILD_INTERFACE:${SPECTRA8_ZSTD_TARGET}>)
endif()
if(SPECTRA8_BZIP3_ENABLED AND SPECTRA8_BZIP3_TARGET)
  target_link_libraries(spectra8_objects PRIVATE
    $<BUILD_INTERFACE:${SPECTRA8_BZIP3_TARGET}>)
endif()
if(SPECTRA8_LZMA_ENABLED)
  target_link_libraries(spectra8_objects PRIVATE
    $<BUILD_INTERFACE:${SPECTRA8_LZMA_TARGET}>)
endif()
if(SPECTRA8_METAL_ENABLED)
  # ARC for the Objective-C++ host files.
  set_source_files_properties(src/gpu/metal_lz.mm src/gpu/metal_h2d.mm PROPERTIES
    COMPILE_OPTIONS "-fobjc-arc")
endif()
if(SPECTRA8_HIP_ENABLED)
  # Compile the kernel-bearing file with the HIP compiler.
  set_source_files_properties(src/gpu/hip_lz.cpp PROPERTIES LANGUAGE HIP)
endif()
if(SPECTRA8_PRIVATE_FLAGS)
  target_compile_options(spectra8_objects PRIVATE ${SPECTRA8_PRIVATE_FLAGS})
endif()
if(NOT MSVC)
  target_compile_options(spectra8_objects PRIVATE -O${SPECTRA8_OPT_LEVEL} -Wall -Wextra)
  message(STATUS "spectra8: codec objects built with -O${SPECTRA8_OPT_LEVEL}")
endif()

add_library(spectra8 SHARED $<TARGET_OBJECTS:spectra8_objects>)
add_library(spectra8_static STATIC $<TARGET_OBJECTS:spectra8_objects>)
add_library(spectra8::spectra8 ALIAS spectra8)

foreach(tgt spectra8 spectra8_static)
  target_include_directories(${tgt}
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:include>)
  target_link_libraries(${tgt} PUBLIC Threads::Threads)
  if(SPECTRA8_METAL_ENABLED)
    target_link_libraries(${tgt} PUBLIC ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK})
  endif()
  if(SPECTRA8_HIP_ENABLED)
    target_link_libraries(${tgt} PUBLIC hip::host)
  endif()
  if(SPECTRA8_CUDA_ENABLED)
    target_link_libraries(${tgt} PUBLIC CUDA::cudart)
  endif()
  # Optional backends are build-only (not part of the installed public ABI —
  # headers stay free of zstd.h / libbz3.h). BUILD_INTERFACE keeps the
  # INTERFACE helper targets out of the install(EXPORT) set.
  if(SPECTRA8_ZSTD_ENABLED)
    target_link_libraries(${tgt} PRIVATE
      $<BUILD_INTERFACE:${SPECTRA8_ZSTD_TARGET}>)
    target_compile_definitions(${tgt} PRIVATE SPECTRA8_HAVE_ZSTD)
  endif()
  if(SPECTRA8_BZIP3_ENABLED)
    target_compile_definitions(${tgt} PRIVATE SPECTRA8_HAVE_BZIP3)
    if(SPECTRA8_BZIP3_TARGET)
      target_link_libraries(${tgt} PRIVATE
        $<BUILD_INTERFACE:${SPECTRA8_BZIP3_TARGET}>)
    endif()
  endif()
  if(SPECTRA8_LZMA_ENABLED)
    target_compile_definitions(${tgt} PRIVATE SPECTRA8_HAVE_LZMA)
    target_link_libraries(${tgt} PRIVATE
      $<BUILD_INTERFACE:${SPECTRA8_LZMA_TARGET}>)
  endif()
endforeach()

set_target_properties(spectra8 PROPERTIES
  VERSION   ${PROJECT_VERSION}
  SOVERSION ${PROJECT_VERSION_MAJOR}
  OUTPUT_NAME spectra8)
if(APPLE)
  # Installed dylib carries its absolute install name (e.g.
  # /opt/homebrew/lib/libspectra8.2.dylib) so consumers of the prebuilt
  # tarball / Homebrew keg need no rpath; the build tree keeps @rpath.
  set_target_properties(spectra8 PROPERTIES
    INSTALL_NAME_DIR "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")
endif()
set_target_properties(spectra8_static PROPERTIES OUTPUT_NAME spectra8)

if(SPECTRA8_IPO)
  set_target_properties(spectra8_objects spectra8 spectra8_static
    PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
endif()

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if(SPECTRA8_BUILD_CLI)
  add_executable(spectra8_cli cli/main.cpp cli/bench.cpp cli/log.cpp)
  set_target_properties(spectra8_cli PROPERTIES OUTPUT_NAME spectra8)
  target_link_libraries(spectra8_cli PRIVATE spectra8_static)
  # The benchmark reports the auto-detected byte filter, which lives in an
  # internal (header-only) module.
  target_include_directories(spectra8_cli PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  if(SPECTRA8_LZMA_ENABLED)
    target_link_libraries(spectra8_cli PRIVATE ${SPECTRA8_LZMA_TARGET})
    target_compile_definitions(spectra8_cli PRIVATE SPECTRA8_HAVE_LZMA)
  endif()
  # Bench still calls libzstd / libbzip3 directly; the static lib link already
  # pulls the libs in, but the HAVE_* macros must be set on the CLI TU.
  if(SPECTRA8_ZSTD_ENABLED)
    target_link_libraries(spectra8_cli PRIVATE
      $<BUILD_INTERFACE:${SPECTRA8_ZSTD_TARGET}>)
    target_compile_definitions(spectra8_cli PRIVATE SPECTRA8_HAVE_ZSTD)
  endif()
  if(SPECTRA8_BZIP3_ENABLED)
    target_compile_definitions(spectra8_cli PRIVATE SPECTRA8_HAVE_BZIP3)
    if(SPECTRA8_BZIP3_TARGET)
      target_link_libraries(spectra8_cli PRIVATE
        $<BUILD_INTERFACE:${SPECTRA8_BZIP3_TARGET}>)
    endif()
  endif()
  if(SPECTRA8_IPO)
    set_target_properties(spectra8_cli PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
  endif()
endif()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(SPECTRA8_BUILD_TESTS)
  enable_testing()
  add_executable(spectra8_tests tests/roundtrip_test.cpp)
  target_link_libraries(spectra8_tests PRIVATE spectra8_static)
  target_include_directories(spectra8_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  add_test(NAME roundtrip COMMAND spectra8_tests)

  add_executable(spectra8_api_tests tests/api_test.cpp)
  target_link_libraries(spectra8_api_tests PRIVATE spectra8_static)
  # Direct unit tests for internal primitives (BWT/MTF) include src/ headers.
  target_include_directories(spectra8_api_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  add_test(NAME api COMMAND spectra8_api_tests)

  add_executable(spectra8_stream_decoder_tests tests/stream_decoder_test.cpp)
  target_link_libraries(spectra8_stream_decoder_tests PRIVATE spectra8_static)
  target_include_directories(spectra8_stream_decoder_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src)
  add_test(NAME stream_decoder COMMAND spectra8_stream_decoder_tests)

  add_executable(spectra8_log_tests tests/log_test.cpp cli/log.cpp)
  target_link_libraries(spectra8_log_tests PRIVATE spectra8_static)
  target_include_directories(spectra8_log_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/cli)
  add_test(NAME logfile COMMAND spectra8_log_tests)

  if(SPECTRA8_IPO)
    set_target_properties(spectra8_tests spectra8_api_tests
      PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
  endif()
endif()

# ---------------------------------------------------------------------------
# Install / export
# ---------------------------------------------------------------------------
include(GNUInstallDirs)
# Two install components: `runtime` (shared library + CLI) and `devel`
# (headers, static library, CMake export).
install(TARGETS spectra8
  EXPORT spectra8Targets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT runtime
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT runtime
    NAMELINK_COMPONENT devel
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT devel)
install(TARGETS spectra8_static
  EXPORT spectra8Targets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT devel)
if(SPECTRA8_BUILD_CLI)
  install(TARGETS spectra8_cli RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT runtime)
endif()
install(DIRECTORY include/spectra8 DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  COMPONENT devel)
install(EXPORT spectra8Targets
  NAMESPACE spectra8::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/spectra8
  COMPONENT devel)
install(FILES LICENSE NOTICE README.md
  DESTINATION ${CMAKE_INSTALL_DATADIR}/doc/spectra8 COMPONENT runtime)
