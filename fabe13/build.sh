#!/usr/bin/env bash
set -euo pipefail

# Adjust these paths to wherever you keep your project:
SRC_ROOT="$HOME/Desktop/www/fabe_project/fabe"
BUILD_DIR="${SRC_ROOT}/build"

echo "→ Cleaning build directory..."
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

echo "→ Configuring with CMake..."
cmake "${SRC_ROOT}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DFABE13_ENABLE_TEST=ON \
  -DFABE13_ENABLE_BENCHMARK=ON \
  -DCMAKE_C_FLAGS="-Ofast -march=native -funroll-loops -ffast-math" \
  -DCMAKE_CXX_FLAGS="-Ofast -march=native -funroll-loops -ffast-math"

echo "→ Building (all cores)…"
# On Linux:
if command -v nproc &>/dev/null; then
  PARALLEL=$(nproc)
# On macOS:
elif sysctl -n hw.ncpu &>/dev/null; then
  PARALLEL=$(sysctl -n hw.ncpu)
else
  PARALLEL=4
fi
make -j"${PARALLEL}"

echo "→ Running unit tests…"
ctest --output-on-failure -j"${PARALLEL}"

echo "→ Quick sanity test…"
./fabe13_test

echo "→ Running benchmark…"
./fabe13_benchmark

echo "✅ Done."
