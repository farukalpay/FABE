# FABE13 — Precision-Crafted SIMD Trigonometric Library

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)  
[![Build](https://img.shields.io/badge/build-passing-brightgreen.svg)]()  
[![Platform](https://img.shields.io/badge/platform-x86_64%20%7C%20AArch64-lightgrey.svg)]()  
[![SIMD](https://img.shields.io/badge/SIMD-AVX2%2C%20AVX512%2C%20NEON-orange.svg)]()

**FABE13** is a modern, architecture-aware trigonometric library written in C. It implements `sin(x)`, `cos(x)`, and `sincos(x)` with high-precision minimax polynomials, full-range argument reduction, and dynamic SIMD dispatch.

FABE13 also includes a **complete trigonometric API**, including `sinc`, `tan`, `cot`, `atan`, `asin`, and `acos`, with a focus on numerical correctness and cross-platform vectorized execution.

> **Arguably the cleanest and most accurate open-source SIMD trigonometric core available today.**

---

## ✨ Features

- ✅ **Full Trigonometric API** — `sin`, `cos`, `sincos`, `sinc`, `tan`, `cot`, `atan`, `asin`, `acos`
- ✅ **Runtime SIMD Dispatch** — Supports AVX512F, AVX2+FMA, NEON (AArch64), or scalar fallback
- ✅ **Payne–Hanek Range Reduction** — Handles large angles (`|x| > 1e308`) with precision
- ✅ **Minimax Polynomials + Estrin** — Efficient and accurate evaluation in SIMD paths
- ✅ **Unified SIMD API** — `fabe13_sincos()` for fast array-wide vectorized computation
- ✅ **Cross-Platform** — Runs on Linux, macOS (Intel & Apple Silicon), and ARMv8
- ✅ **Handles Edge Cases** — Supports NaN, Inf, subnormals, ±0

---

## 📂 Project Structure

```
git clone https://github.com/farukalpay/FABE.git
cd FABE/fabe13
```

```
fabe13/                 # Core SIMD + scalar implementation
├── fabe13.c
├── fabe13.h
└── benchmark_fabe13.c
tests/                  # Unit tests
└── test_fabe13.c
CMakeLists.txt          # Cross-platform build system
Makefile                # Lightweight build option
```

---

## ⚙️ Build & Run

### 🔨 Using Make
```bash
make all             # Builds benchmark and tests
make run-test        # Runs the test suite
make run-benchmark   # Runs the performance benchmark
```

### 🧱 Using CMake
```bash
mkdir -p build
cd build
cmake .. -DFABE13_ENABLE_BENCHMARK=ON -DFABE13_ENABLE_TEST=ON
make
./fabe13_test
./fabe13_benchmark
```

---

## 📈 Performance & Accuracy

FABE13 excels at **large-scale SIMD throughput** while maintaining floating-point correctness.

![FABE13 Benchmark Chart](https://github.com/farukalpay/FABE/blob/main/img/FABE13%20vs%20libm%20Benchmark%20Performance.png)

### 📊 Benchmark Comparison

| Input Size | FABE13 Time | libm Time | FABE13 vs libm |
|------------|-------------|-----------|----------------|
| 1M         | 0.110 s     | 0.006 s   | ~18x slower    |
| 100M       | 9.801 s     | 0.503 s   | ~19x slower    |
| 1B         | 2.460 s     | 6.647 s   | ✅ ~2.7x faster |
| 1.41B      | 4.582 s     | 8.517 s   | ✅ ~1.86x faster |

> ⚠️ FABE13 is slower for small sizes due to full-precision range reduction, but **dramatically faster at scale**.

#### 🔧 CPU Usage Comparison

This chart shows how FABE13 saturates SIMD execution as workloads scale — while libm remains underutilized.

![CPU Usage vs Input Size: FABE13 vs libm](https://github.com/farukalpay/FABE/blob/main/img/CPU%20Usage%20vs%20Input%20Size%3A%20FABE13%20vs%20libm.png?raw=true)

> 🔍 FABE13 uses more CPU at small batch sizes — a known tradeoff for full precision — but **pays off massively at scale**.

### 🌟 Accuracy Profile

- Uses **Payne–Hanek** reduction for full-domain correctness  
- **Estrin's scheme** for stable SIMD polynomial evaluation  
- Handles all IEEE-754 edge cases  
- **0 ULP** matches for many standard inputs  
- Slight numerical drift observed in extreme `|x| > 1e18` due to floating-point resolution limits — within acceptable bounds for scientific use  

![FABE13 Accuracy vs libm](https://github.com/farukalpay/FABE/blob/main/img/8F798A77-E21C-4F05-A813-2A4DC974A3D2.png?raw=true)

#### Max observed deviation:
```
sin: 9.9196e+61
cos: 9.8862e+61
```
> 🔹 Only observed at extreme |x| > 1e300, where `libm` also becomes unstable.

---

## 💻 API Overview

```c
// Scalar trigonometric functions
fabe13_sin(double x);
fabe13_cos(double x);
fabe13_sinc(double x);   // sin(x)/x
fabe13_tan(double x);
fabe13_cot(double x);
fabe13_atan(double x);
fabe13_asin(double x);   // x in [-1, 1]
fabe13_acos(double x);   // x in [-1, 1]

// SIMD vectorized sincos
void fabe13_sincos(const double* in, double* sin_out, double* cos_out, int n);
```

---

## 🔍 Internals

- 📊 **Payne–Hanek**: Double-double modular reduction of large inputs
- 🌌 **Estrin Evaluation**: Fast polynomial evaluation with reduced dependencies
- ⚖️ **Vector Quadrant Correction**: Branchless SIMD quadrant logic
- 📈 **Dispatch**: AVX512/AVX2/NEON/Scalar selected at runtime

---

## 👊 Development Status

> ⚠️ FABE13 is **experimental** and currently in **beta**.

| Area        | Status      |
|-------------|-------------|
| Accuracy    | ✅ Stable |
| AVX/NEON    | ✅ Complete |
| Small-size Perf | ❌ Needs fast-path |
| Docs/API    | ✅ Documented |
| Packaging   | ❌ Header-only version WIP |

Planned features:
- Fast-path skipping for |x| < π/4
- Loop unrolling (especially for NEON)
- Adjustable polynomial degrees

---

## 📜 License

**MIT License © 2025 Faruk Alpay**  
See [LICENSE](LICENSE)

---

## 👥 Author

**Faruk Alpay**  
[https://lightcap.ai](https://lightcap.ai)

> FABE13 is developed under the Lightcap initiative — a parent brand dedicated to crafting precise, portable, and high-throughput tools for the future of science, computation, and creativity. From AI research to numerical libraries like FABE13, Lightcap unifies innovation across disciplines with minimalism, clarity, and performance at its core.

### ⚠️ Known Weaknesses (and Roadmap for Improvement)

FABE13 is still in **early beta**, and while its architecture is solid and scalable, it currently shows some real-world tradeoffs:

- ❌ **High CPU usage for small input batches** (1M–100M) due to always-on range reduction and SIMD overhead
- ❌ **Lack of fast-path logic** for |x| < π/4 — where simple polynomial shortcuts would suffice
- ❌ **No support yet for WASM SIMD, SVE, or RISC-V V**
- ❌ **No header-only version or bindings for C++, Rust, or Python yet**

> 💡 These are not only going to be fixed — they’re going to be **transformed into strengths**:
> - Adaptive fast-path logic (skip PH reduction when safe)
> - Loop unrolling & memory alignment improvements (especially NEON)
> - Lower-degree poly options for embedded use
> - Cross-language wrappers + header-only mode

The long-term goal?  
To make FABE13 the **most trusted** SIMD-first trig core in open source — both for researchers and for production systems that demand real accuracy, real speed, and zero magic.

---

📣 Want to contribute, benchmark, or integrate FABE13?
Start here → https://github.com/farukalpay/FABE

🔬 Built under the Lightcap initiative: https://lightcap.ai
