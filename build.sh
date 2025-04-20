#!/usr/bin/env bash
set -euo pipefail

# --- Project Setup ---
PROJECT_ROOT="$(pwd)"
BUILD_DIR="${PROJECT_ROOT}/build"

# --- Clean Build Directory ---
echo "→ Cleaning build directory..."
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# --- Configure CMake ---
echo "→ Configuring with CMake..."
cmake "${PROJECT_ROOT}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DFABE13_ENABLE_TEST=ON \
  -DFABE13_ENABLE_BENCHMARK=ON \
  -DCMAKE_C_FLAGS="-Ofast -march=native -funroll-loops -ffast-math" \
  -DCMAKE_CXX_FLAGS="-Ofast -march=native -funroll-loops -ffast-math"

# --- Detect Parallelism ---
echo "→ Building (all cores)…"
if command -v nproc &>/dev/null; then
  PARALLEL=$(nproc)
elif command -v sysctl &>/dev/null; then
  PARALLEL=$(sysctl -n hw.ncpu)
else
  PARALLEL=4
fi
make -j"${PARALLEL}"

# --- Run Unit Tests ---
if [[ -f ./fabe13_test || -f ./fabe13_test_runner ]]; then
  echo "→ Running unit tests…"
  ctest --output-on-failure -j"${PARALLEL}"
else
  echo "⚠️ No unit test binary found. Skipping."
fi

# --- Run Sanity Test (if available) ---
if [[ -f ./fabe13_test ]]; then
  echo "→ Quick sanity test…"
  ./fabe13_test
fi

# --- Run Benchmark (if available) ---
if [[ -f ./fabe13_benchmark ]]; then
  echo "→ Running benchmark…"
  ./fabe13_benchmark
fi

echo "✅ Done."
