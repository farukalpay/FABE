cmake_minimum_required(VERSION 3.15)
project(FABE13 LANGUAGES C)

# ------------------ OPTIONS ------------------
option(FABE13_ENABLE_BENCHMARK "Build the benchmark executable" ON)
option(FABE13_ENABLE_TEST      "Build the test executable" ON)

option(FABE13_USE_AVX512 "Enable AVX-512F support (x86 only)" OFF)
option(FABE13_USE_AVX2   "Enable AVX2+FMA support (x86 only)" ON)
option(FABE13_USE_NEON   "Enable NEON support (AArch64)" ON)

# ------------------ C STANDARD ------------------
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# ------------------ SIMD FLAGS ------------------
# We'll add CPU-specific flags if the processor is x86_64 or aarch64
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    if(FABE13_USE_AVX512)
        add_compile_options(-mavx512f -mfma)
    elseif(FABE13_USE_AVX2)
        add_compile_options(-mavx2 -mfma)
    endif()
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    if(FABE13_USE_NEON)
        # NEON is typically the default on AArch64 compilers.
        # You can add additional flags if needed.
    endif()
endif()

# ------------------ INCLUDE PATHS ------------------
# So '#include "fabe13/fabe13.h"' works in your code
include_directories(${CMAKE_SOURCE_DIR}/fabe13)

# ------------------ MAIN LIBRARY ------------------
# Build a static library from fabe13/fabe13.c
add_library(fabe13 STATIC fabe13/fabe13.c)
target_include_directories(fabe13 PUBLIC ${CMAKE_SOURCE_DIR}/fabe13)

# ------------------ BENCHMARK ------------------
if(FABE13_ENABLE_BENCHMARK)
    add_executable(fabe13_benchmark fabe13/benchmark_fabe13.c)
    # If your benchmark code uses #ifdef ENABLE_FABE13_BENCHMARK
    target_compile_definitions(fabe13_benchmark PRIVATE ENABLE_FABE13_BENCHMARK=1)

    # Link the standard math library and our library
    target_link_libraries(fabe13_benchmark PRIVATE fabe13 m)

    # On non-Apple platforms, link librt for clock_gettime() etc.
    if(NOT APPLE)
        target_link_libraries(fabe13_benchmark PRIVATE rt)
    endif()
endif()

# ------------------ TEST ------------------
if(FABE13_ENABLE_TEST)
    add_executable(fabe13_test tests/test_fabe13.c)
    # Link with the library and math
    target_link_libraries(fabe13_test PRIVATE fabe13 m)
    # If needed, add include dirs:
    target_include_directories(fabe13_test PRIVATE ${CMAKE_SOURCE_DIR}/fabe13)
endif()

# ------------------ SUMMARY ------------------
message(STATUS "FABE13 Build Summary:")
message(STATUS "  FABE13_USE_AVX2:   ${FABE13_USE_AVX2}")
message(STATUS "  FABE13_USE_AVX512: ${FABE13_USE_AVX512}")
message(STATUS "  FABE13_USE_NEON:   ${FABE13_USE_NEON}")
message(STATUS "  Benchmark:         ${FABE13_ENABLE_BENCHMARK}")
message(STATUS "  Test:              ${FABE13_ENABLE_TEST}")
