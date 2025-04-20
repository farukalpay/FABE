cmake_minimum_required(VERSION 3.15)
project(FABE13 LANGUAGES C)

# ------------------ OPTIONS ------------------
option(FABE13_ENABLE_BENCHMARK "Build the benchmark executable" ON)
option(FABE13_ENABLE_TEST      "Build the unit test executable" ON)
option(FABE13_USE_AVX512       "Enable AVX-512F support (x86 only)" OFF)
option(FABE13_USE_AVX2         "Enable AVX2+FMA support (x86 only)" ON)
option(FABE13_USE_NEON         "Enable NEON support (AArch64)" ON)

# ------------------ STANDARDS ------------------
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# ------------------ OPTIMIZATION FLAGS ------------------
add_compile_options(-Ofast -funroll-loops -ffast-math)

# ------------------ ARCH DETECTION ------------------
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
  if(FABE13_USE_AVX512)
    add_compile_options(-mavx512f -mfma)
  elseif(FABE13_USE_AVX2)
    add_compile_options(-mavx2 -mfma)
  endif()
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  if(FABE13_USE_NEON)
    # NEON is usually enabled by default on AArch64
  endif()
endif()

# ------------------ SOURCE STRUCTURE ------------------
include_directories(${CMAKE_SOURCE_DIR}/fabe13)

add_library(fabe13 STATIC
  fabe13/fabe13.c
)

target_include_directories(fabe13 PUBLIC
  ${CMAKE_SOURCE_DIR}/fabe13
)

# ------------------ BENCHMARK ------------------
if(FABE13_ENABLE_BENCHMARK)
  add_executable(fabe13_benchmark fabe13/benchmark_fabe13.c)
  target_compile_definitions(fabe13_benchmark PRIVATE ENABLE_FABE13_BENCHMARK=1)
  target_link_libraries(fabe13_benchmark PRIVATE fabe13 m)
  if(NOT APPLE)
    target_link_libraries(fabe13_benchmark PRIVATE rt)
  endif()
endif()

# ------------------ UNIT TEST ------------------
if(FABE13_ENABLE_TEST)
  # Use unique name to avoid Replit collisions
  add_executable(fabe13_test_runner tests/test_fabe13.c)
  target_link_libraries(fabe13_test_runner PRIVATE fabe13 m)
  target_include_directories(fabe13_test_runner PRIVATE ${CMAKE_SOURCE_DIR}/fabe13)
  enable_testing()
  add_test(NAME FABE13_UnitTest COMMAND fabe13_test_runner)
endif()

# ------------------ Summary ------------------
message(STATUS "")
message(STATUS "FABE13-HX Configuration Summary")
message(STATUS "  AVX2 enabled   : ${FABE13_USE_AVX2}")
message(STATUS "  AVX512 enabled : ${FABE13_USE_AVX512}")
message(STATUS "  NEON enabled   : ${FABE13_USE_NEON}")
message(STATUS "  Benchmark      : ${FABE13_ENABLE_BENCHMARK}")
message(STATUS "  Unit Tests     : ${FABE13_ENABLE_TEST}")
message(STATUS "")
