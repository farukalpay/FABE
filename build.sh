#!/usr/bin/env bash
set -euo pipefail

# --- Configurable Root ---
SRC_ROOT="$(dirname "$(realpath "$0")")"
BUILD_DIR="${SRC_ROOT}/build"

# --- Cleanup & Setup ---
echo "→ Cleaning build directory..."
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# --- Detect CPU Info (Optional Debug Output) ---
echo "→ Configuring with CMake..."

cmake "${SRC_ROOT}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DFABE13_ENABLE_TEST=ON \
  -DFABE13_ENABLE_BENCHMARK=ON \
  -DCMAKE_C_FLAGS="-Ofast -march=native -funroll-loops -ffast-math" \
  -DCMAKE_C_STANDARD=99

# --- Build All ---
echo "→ Building (all cores)…"
if command -v nproc &>/dev/null; then
  PARALLEL=$(nproc)
elif sysctl -n hw.ncpu &>/dev/null; then
  PARALLEL=$(sysctl -n hw.ncpu)
else
  PARALLEL=4
fi
make -j"${PARALLEL}"

# --- Run Tests ---
echo "→ Running unit tests…"
ctest --output-on-failure -j"${PARALLEL}"

# --- Quick Functional Test ---
echo "→ Quick sanity test…"
./fabe13_test

# --- Benchmark Run ---
echo "→ Running benchmark…"
./fabe13_benchmark

echo "✅ Done."
