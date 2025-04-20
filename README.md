# FABE13-HX — Post-Polynomial SIMD Trigonometric Library

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Build](https://img.shields.io/badge/build-passing-brightgreen.svg)]()
[![Platform](https://img.shields.io/badge/platform-x86_64%20%7C%20AArch64-lightgrey.svg)]()
[![SIMD](https://img.shields.io/badge/SIMD-AVX2%2C%20AVX512%2C%20NEON-orange.svg)]()

**FABE13-HX** is a high-performance trigonometric math library in C, reimagining `sin(x)`, `cos(x)`, and `sincos(x)` using a novel rational-function foundation: the **Ψ-Hyperbasis**.

Designed for **numerical computing**, **AI acceleration**, and **scientific simulation**, it replaces traditional polynomial approximations with a fused rational + correction model that’s more efficient and vectorization-friendly.

---

## ✨ What's New in HX

- 🚀 **Ψ-Hyperbasis Core:** `Ψ(x) = x / (1 + (3/8)x²)` drives both `sin` and `cos` via correction polynomials in `Ψ²`.
- 🧠 **Post-Polynomial Philosophy:** A rational-first architecture beyond Horner or Estrin’s classical methods.
- ⚡ **Runtime SIMD Dispatch:** Supports AVX512F, AVX2+FMA (x86), NEON (ARM), or scalar fallback.
- 🧩 **Single API for All:** Scalar and vector modes unified in a clean, minimal C API.
- 🔬 **Extreme-Range Support:** Accurate up to |x| ≈ 1e308 via Payne–Hanek reduction.

---

## 📂 Project Structure

```
fabe13/                 # Core source
├── fabe13.c            # HX implementation
├── fabe13.h            # Public API
├── benchmark_fabe13.c  # Benchmark main

tests/
└── test_fabe13.c       # Optional unit tests

CMakeLists.txt          # Cross-platform CMake
Makefile                # Minimalist legacy build
build.sh                # Recommended build script (cross-platform)
```

---

## ⚙️ Build Instructions

### ✅ Recommended: `build.sh`

```bash
./build.sh
```

This script:
- Cleans and configures the build (Release mode)
- Enables both benchmarking and testing
- Compiles using aggressive `-Ofast`, `-ffast-math`, `-march=native` flags
- Runs all unit tests and benchmarks automatically

### 🛠️ Manual CMake

```bash
mkdir -p build && cd build
cmake .. -DFABE13_ENABLE_BENCHMARK=ON -DFABE13_ENABLE_TEST=ON
make
./fabe13_test
./fabe13_benchmark
```

### 🧱 Makefile (Legacy)

```bash
make all
make run-benchmark
```

---

## 🚀 Benchmark Snapshot

```txt
FABE13 Active Implementation: NEON (AArch64) (SIMD Width: 2)

Benchmarking sincos(x):
N = 1,000,000

FABE13:  0.0016 sec  |  635.81 M ops/sec
libm:    0.0071 sec  |  140.27 M ops/sec
Speedup: 4.53x

Max diff vs libm: sin=1.225e-11, cos=1.225e-11
```

---

## 🔬 Core Algorithm (Ψ-Hyperbasis)

```c
// Core rational transformation
Ψ(x) = x / (1 + (3/8)x²)

// sin(x) approximation
sin(x) ≈ Ψ ⋅ (1 - a1⋅Ψ² + a2⋅Ψ⁴ - a3⋅Ψ⁶)

// cos(x) approximation
cos(x) ≈ 1 - b1⋅Ψ² + b2⋅Ψ⁴ - b3⋅Ψ⁶
```

This allows both functions to share a unified base, optimizing performance and memory access.

---

## 📊 Public API

```c
#include "fabe13/fabe13.h"

// Scalar API
double fabe13_sin(double x);
double fabe13_cos(double x);
double fabe13_sinc(double x);  // sin(x)/x
double fabe13_tan(double x);
double fabe13_cot(double x);
double fabe13_atan(double x);
double fabe13_asin(double x);  // [-1, 1]
double fabe13_acos(double x);  // [-1, 1]

// SIMD vector API
void fabe13_sincos(const double* in, double* sin_out, double* cos_out, int n);
```

---

## 🧠 Design Highlights

- ✅ **Branchless Quadrant Correction**
- ✅ **NaN/Inf/0-safe logic**
- ✅ **Prefetch-friendly & unrolled scalar fallback**
- ✅ **SIMD-ready backend design (NEON / AVX2 / AVX512)**
- ✅ **Precision-preserving range reduction**

---

## 🛠️ Roadmap

- [ ] SIMD Ψ-Hyperbasis implementation (AVX2 / NEON / AVX512)
- [ ] `cosm1`, `expm1`, `log1p` expansions
- [ ] `float32` support (`fabe13_sinf`, etc.)
- [ ] LUT-based ultra-fast variants
- [ ] Header-only + Python / Rust bindings

---

## 📜 License

MIT License © 2025 Faruk Alpay  
See [LICENSE](fabe13-old/LICENSE)

---

## 🧬 Author

**Faruk Alpay**  
https://Frontier2075.com  
https://lightcap.ai  

> FABE13-HX is part of the **Lightcap Initiative** — building the most precise and elegant math primitives in open source.
