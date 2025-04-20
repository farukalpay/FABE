# FABE13-HX — Post-Polynomial SIMD Trigonometric Library

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Build](https://img.shields.io/badge/build-passing-brightgreen.svg)]()
[![Platform](https://img.shields.io/badge/platform-x86_64%20%7C%20AArch64-lightgrey.svg)]()
[![SIMD](https://img.shields.io/badge/SIMD-AVX2%2C%20AVX512%2C%20NEON-orange.svg)]()

**FABE13-HX** is a high-performance trigonometric math library in C, reimagining `sin(x)`, `cos(x)`, and `sincos(x)` using a novel rational-function foundation: the **Ψ-Hyperbasis**.

Designed for **numerical computing**, **AI acceleration**, and **scientific simulation**, it replaces traditional polynomial approximations with a fused rational + correction model that's more efficient and vectorization-friendly.

---

## ✨ What's New in HX

- 🚀 **Ψ-Hyperbasis Core:** `Ψ(x) = x / (1 + (3/8)x²)` drives both `sin` and `cos` via correction polynomials in `Ψ²`.
- 🧠 **Post-Polynomial Philosophy:** A rational-first architecture beyond Horner or Estrin's classical methods.
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

## 🚀 FABE13-HX vs libm — Performance Benchmarks

FABE13-HX delivers consistent speedups over standard `libm`, across platforms and input sizes. These benchmarks highlight its advantage for both cloud-based and local environments.

### 📊 Performance Overview

- 🟨 **FABE13-HX**: SIMD-accelerated (`AVX2+FMA`, Ψ-core)
- 🔴 **libm**: Standard C math (`math.h`)
- 🧠 Input size: `N ∈ [10 ... 1,000,000,000]` doubles
- ⚙️ Timing: Full-array `sincos()` throughput
- 📐 Aligned memory: 64 bytes
- 🎯 Accuracy: ≤ 2e-11 max diff (sin/cos)

---

### 🌐 Replit (Cloud / Linux, AVX2 Clang)

![FABE13-HX vs libm — Replit](https://github.com/farukalpay/FABE/blob/main/img/Performance%20Comparison%3A%20FABE13-HX%20vs%20libm%20(Platform%3A%20Replit%2C%20AVX2%20Core%2C%20CMath%20backend).png)

> ✅ **FABE13-HX is consistently faster than libm — up to 8.4× for large inputs.**

- Platform: Replit Linux
- SIMD: AVX2 + FMA
- Compiler: Clang 14 (nix)
- libm: GNU `math.h`

---

### 🍎 MacBook Pro (macOS AVX2, AppleClang)

![FABE13-HX vs libm — macOS](https://github.com/farukalpay/FABE/blob/main/img/FABE13-HX%20vs%20libm%20%E2%80%94%20Performance%20Benchmark.png)

> 🟨 **FABE13-HX outperforms libm with up to 8.4× higher throughput on AppleClang (AVX2).**

- Platform: macOS 14.x (MacBook Pro 16")
- SIMD: AVX2 + FMA
- Compiler: AppleClang 16.0
- libm: macOS system `math.h`

---

### 📊 Performance Overview

```
FABE13 Active Implementation: NEON (AArch64) (SIMD Width: 2)
Benchmark Alignment: 64 bytes
```

### 📈 Scaling with Array Size

| Array Size | FABE13 (sec) | Libm (sec) | FABE13 (M ops/sec) | Libm (M ops/sec) | Speedup |
|------------|--------------|------------|-------------------|-----------------|---------|
| 10         | 0.0000       | 0.0000     | 50.00             | 50.00           | 1.00x   |
| 100        | 0.0000       | 0.0000     | 166.67            | 71.43           | 2.33x   |
| 1,000      | 0.0000       | 0.0000     | 185.19            | 72.46           | 2.56x   |
| 10,000     | 0.0001       | 0.0001     | 173.01            | 71.02           | 2.44x   |
| 100,000    | 0.0006       | 0.0009     | 177.12            | 115.82          | 1.53x   |
| 1,000,000  | 0.0016       | 0.0072     | 614.85            | 138.34          | 4.44x   |
| 10,000,000 | 0.0164       | 0.0720     | 611.30            | 138.95          | 4.40x   |
| 100,000,000| 0.1673       | 0.7296     | 597.63            | 137.07          | 4.36x   |
| 1,000,000,000| 1.8044     | 10.4989    | 554.19            | 95.25           | 5.82x   |

### 🔍 Detailed Benchmark Snapshot (N = 1,000,000)

```
FABE13:  0.0016 sec  |  614.85 M ops/sec
libm:    0.0072 sec  |  138.34 M ops/sec
Speedup: 4.44x

Memory: Allocated 0.04 GB
        Peak RSS: ~29 MB (FABE13), ~45 MB (Libm)
CPU:    100.0% utilization for both implementations

Max diff vs libm: sin=1.224e-11, cos=1.225e-11
```

### 🔬 Precision Analysis

- All test cases maintain acceptable numerical accuracy compared to libm
- Maximum difference observed: ~10⁻¹¹ for both sin and cos operations 
- Properly handles edge cases (0, inf, nan) with correct behavior

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
