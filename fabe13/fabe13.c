/**
 * @file fabe13.c
 * @brief FABE13 High-Performance Trigonometric Library Implementation (FABE13-HX).
 *
 * Contains implementations for sin(x), cos(x), and sincos(x) using SIMD
 * instructions (AVX-512F, AVX2+FMA, NEON) with runtime dispatch.
 * Features a custom scalar core using the Ψ-Hyperbasis approximation (FABE13-HX)
 * for the main range, direct dispatch, alignment handling, small-N optimization,
 * prefetching, and optional FTZ/DAZ mode.
 */

// --- Compiler Optimizations & Feature Tests ---
#pragma GCC optimize("O3", "unroll-loops", "align-loops", "tree-vectorize")
// Consider adding -Ofast or -ffast-math via CFLAGS for potentially more speed
// #pragma GCC optimize("Ofast") // If using this, ensure accuracy is acceptable

#define _GNU_SOURCE      // For llrint, round if needed, constructor attribute
#define _ISOC99_SOURCE   // For isnan, isinf, NAN, INFINITY etc.
#define _POSIX_C_SOURCE 200809L // For clock_gettime, posix_memalign detection (though not used here)

#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <stdlib.h> // For NULL (though not strictly needed now)
#include <float.h>  // For DBL_MAX etc.
#include <limits.h> // For INT_MAX (used in API check)
#include <stdio.h>  // For #warning

#include "fabe13.h" // Include the public API header

// --- Platform Detection & Intrinsics ---
#undef FABE13_PLATFORM_X86
#undef FABE13_PLATFORM_ARM
#if defined(__x86_64__) || defined(_M_X64)
    #define FABE13_PLATFORM_X86 1
    #include <immintrin.h> // x86 intrinsics (AVX, AVX2, AVX512)
    #include <xmmintrin.h> // For CSR access, prefetch
    // Define restrict keyword if not using C99/C11 explicitly
    #ifndef __cplusplus
        #define FABE13_RESTRICT restrict
    #else
        #define FABE13_RESTRICT __restrict
    #endif
#elif defined(__aarch64__)
    #define FABE13_PLATFORM_ARM 1
    #include <arm_neon.h>   // ARM NEON intrinsics
    #ifndef __cplusplus
        #define FABE13_RESTRICT restrict
    #else
        #define FABE13_RESTRICT __restrict
    #endif
    // For FPSCR access if needed
    // #include <arm_acle.h>
#else
    #define SIMD_WIDTH 1
    #define FABE13_RESTRICT
#endif

// --- Constants ---
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef M_PI_2
#define M_PI_2 1.57079632679489661923
#endif
// NAN/INFINITY should be defined via math.h

#define FABE13_TINY_THRESHOLD 7.450580596923828125e-9
#define FABE13_SMALL_DELTA   0.01 // Threshold for using small polynomial in scalar core
#define SMALL_N_THRESHOLD 32 // Threshold for using dedicated small-N scalar path
// FABE13_ALIGNMENT is defined in fabe13.h

// --- Coefficients ---
// ** Ψ-Hyperbasis Coefficients (Scalar Core, |r| > SMALL_DELTA) **
// Taylor-matched coefficients for the Ψ-basis expansion
static const double FABE13_HX_A1 = 0.16666666666666666;  // ~1/6
static const double FABE13_HX_A2 = 0.008333333333333333; // ~1/120
static const double FABE13_HX_A3 = 0.0001984126984126984; // ~1/5040
static const double FABE13_HX_B1 = 0.5;                   // 1/2
static const double FABE13_HX_B2 = 0.041666666666666664; // ~1/24
static const double FABE13_HX_B3 = 0.001388888888888889; // ~1/720

// ** Small Polynomial Coefficients (Scalar Core, |r| <= SMALL_DELTA) **
static const double FABE13_SIN_COEFFS_SMALL[] = {
    -1.66666666666666657415e-01, 8.33333333333329961475e-03,
    -1.98412698412589187999e-04, 2.75573192235635111290e-06 };
static const double FABE13_COS_COEFFS_SMALL[] = {
    -4.99999999999999944489e-01, 4.16666666666664590036e-02,
    -1.38888888888829782464e-03, 2.48015873015087640936e-05 };
static const int FABE13_POLY_DEGREE_SMALL_IN_X2 = 3;

// ** Main Polynomial Coefficients (Used only by SIMD paths currently) **
// TODO: Update SIMD paths to use Ψ-Hyperbasis as well
static const double FABE13_SIN_COEFFS_MAIN[] = {
    9.99999999999999999983e-01, -1.66666666666666657415e-01,
    8.33333333333329961475e-03, -1.98412698412589187999e-04,
    2.75573192235635111290e-06, -2.50521083760783692702e-08,
    1.60590438125280493886e-10, -7.64757314471113976640e-13 };
static const double FABE13_COS_COEFFS_MAIN[] = {
    1.00000000000000000000e+00, -4.99999999999999944489e-01,
    4.16666666666664590036e-02, -1.38888888888829782464e-03,
    2.48015873015087640936e-05, -2.75573192094882420430e-07,
    2.08767569813591324530e-09, -1.14757362211242971740e-11 };
static const int FABE13_POLY_DEGREE_MAIN_IN_X2 = 7;


// Payne-Hanek Constants
static const double FABE13_TWO_OVER_PI_HI = 0x1.45f306dc9c883p-1;
static const double FABE13_TWO_OVER_PI_LO = -0x1.9f3c6a7a0b5edp-57;
static const double FABE13_PI_OVER_2_HI = 0x1.921fb54442d18p+0;
static const double FABE13_PI_OVER_2_LO = 0x1.1a62633145c07p-53;

// --- Internal Function Prototypes ---
static void fabe13_sincos_core_scalar(double x, double* s, double* c);
static void fabe13_sincos_scalar_unrolled(const double* FABE13_RESTRICT in, double* FABE13_RESTRICT sin_out, double* FABE13_RESTRICT cos_out, int n);
static void fabe13_sincos_scalar_simple(const double* in, double* sin_out, double* cos_out, int n);

#if FABE13_PLATFORM_ARM
static void fabe13_sincos_neon(const double* FABE13_RESTRICT in, double* FABE13_RESTRICT sin_out, double* FABE13_RESTRICT cos_out, int n); // No alignment flag needed
#endif
#if FABE13_PLATFORM_X86
static void fabe13_sincos_avx2(const double* FABE13_RESTRICT in, double* FABE13_RESTRICT sin_out, double* FABE13_RESTRICT cos_out, int n, bool aligned);
#if defined(__AVX512F__)
static void fabe13_sincos_avx512(const double* FABE13_RESTRICT in, double* FABE13_RESTRICT sin_out, double* FABE13_RESTRICT cos_out, int n, bool aligned);
#endif
#endif

// --- Global State ---
typedef enum { IMPL_SCALAR, IMPL_NEON, IMPL_AVX2, IMPL_AVX512 } fabe13_impl_type;
static fabe13_impl_type active_impl_type = IMPL_SCALAR;
static const char* active_impl_name = "Not Initialized";
static int active_simd_width = 1;

// --- Runtime CPU Feature Detection (x86 only) ---
#if FABE13_PLATFORM_X86
#ifdef __GNUC__
static inline bool fabe13_detect_avx512f() {
    #if defined(__AVX512F__)
        return __builtin_cpu_supports("avx512f");
    #else
        return false;
    #endif
}
static inline bool fabe13_detect_avx2() {
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
}
#elif defined(_MSC_VER)
 #include <intrin.h>
 static inline bool fabe13_detect_avx512f() {
     #if defined(__AVX512F__)
         int cpuInfo[4]; __cpuidex(cpuInfo, 7, 0); return (cpuInfo[1] & (1 << 16)) != 0;
     #else
         return false;
     #endif
 }
 static inline bool fabe13_detect_avx2() {
     int cpuInfo[4]; __cpuid(cpuInfo, 1); bool hasFMA = (cpuInfo[2] & (1 << 12)) != 0;
     __cpuidex(cpuInfo, 7, 0); bool hasAVX2 = (cpuInfo[1] & (1 << 5)) != 0;
     return hasAVX2 && hasFMA;
 }
#else
static inline bool fabe13_detect_avx512f() { #warning "AVX512 detection not implemented for this compiler. AVX512 disabled." return false; }
static inline bool fabe13_detect_avx2() { #warning "AVX2+FMA detection not implemented for this compiler. AVX2 disabled." return false; }
#endif
#endif

// --- Optional: Set FTZ/DAZ Mode ---
#ifdef FABE13_ENABLE_FAST_FP
static void fabe13_set_fast_fp_mode() {
    #if FABE13_PLATFORM_X86 && (defined(__GNUC__) || defined(__clang__) || defined(_MSC_VER))
        unsigned int mxcsr = _mm_getcsr();
        mxcsr |= (1 << 15); // FTZ
        mxcsr |= (1 << 6);  // DAZ
        _mm_setcsr(mxcsr);
    #elif FABE13_PLATFORM_ARM
        #warning "ARM FTZ/DAZ setting not fully implemented/portable yet."
    #endif
}
#endif // FABE13_ENABLE_FAST_FP


// --- Dispatcher & Constructor ---
static void fabe13_initialize_dispatcher() {
    #if FABE13_PLATFORM_ARM
        active_impl_type = IMPL_NEON; active_impl_name = "NEON (AArch64)"; active_simd_width = 2;
    #elif FABE13_PLATFORM_X86
        if (fabe13_detect_avx512f()) {
            #if defined(__AVX512F__)
                active_impl_type = IMPL_AVX512; active_impl_name = "AVX-512F"; active_simd_width = 8;
            #else
                #warning "AVX512F detected but not compiled with AVX512F support. Falling back."
                if (fabe13_detect_avx2()) { active_impl_type = IMPL_AVX2; active_impl_name = "AVX2+FMA"; active_simd_width = 4; }
                else { active_impl_type = IMPL_SCALAR; active_impl_name = "Scalar (Custom-HX, x86)"; active_simd_width = 1; }
            #endif
        } else if (fabe13_detect_avx2()) { active_impl_type = IMPL_AVX2; active_impl_name = "AVX2+FMA"; active_simd_width = 4; }
        else { active_impl_type = IMPL_SCALAR; active_impl_name = "Scalar (Custom-HX, x86)"; active_simd_width = 1; }
    #else
        active_impl_type = IMPL_SCALAR; active_impl_name = "Scalar (Custom-HX, Unknown)"; active_simd_width = 1;
    #endif

    #ifdef FABE13_ENABLE_FAST_FP
        fabe13_set_fast_fp_mode();
    #endif
}

#ifdef __GNUC__
__attribute__((constructor))
#endif
static void init_fabe13(void) {
    fabe13_initialize_dispatcher();
}

// --- Public API Implementations ---

const char* fabe13_get_active_implementation_name(void) { return active_impl_name; }
int fabe13_get_active_simd_width(void) { return active_simd_width; }

// Main dispatch function
void fabe13_sincos(const double * in, double * sin_out, double * cos_out, int n) {
    if (n <= 0) return;

    // 1. Handle Small N case separately
    if (n < SMALL_N_THRESHOLD) {
        fabe13_sincos_scalar_unrolled(in, sin_out, cos_out, n);
        return;
    }

    // 2. Check alignment for larger N (only matters for x86 paths now)
    bool aligned = (((uintptr_t)in      & (FABE13_ALIGNMENT - 1)) == 0) &&
                   (((uintptr_t)sin_out & (FABE13_ALIGNMENT - 1)) == 0) &&
                   (((uintptr_t)cos_out & (FABE13_ALIGNMENT - 1)) == 0);

    // 3. Direct dispatch based on detected type and alignment
    switch (active_impl_type) {
#if FABE13_PLATFORM_ARM
        case IMPL_NEON:
            fabe13_sincos_neon(in, sin_out, cos_out, n); // NEON path ignores 'aligned' internally
            break;
#endif
#if FABE13_PLATFORM_X86
        case IMPL_AVX2:
            fabe13_sincos_avx2(in, sin_out, cos_out, n, aligned);
            break;
#if defined(__AVX512F__)
        case IMPL_AVX512:
            fabe13_sincos_avx512(in, sin_out, cos_out, n, aligned);
            break;
#endif // __AVX512F__
#endif // FABE13_PLATFORM_X86
        case IMPL_SCALAR:
        default:
            // Use non-unrolled scalar for larger N if SIMD failed or wasn't available
            fabe13_sincos_scalar_simple(in, sin_out, cos_out, n);
            break;
    }
}

// Single value API uses the custom scalar core directly
double fabe13_sin(double x) { double s, c; fabe13_sincos_core_scalar(x, &s, &c); return s; }
double fabe13_cos(double x) { double s, c; fabe13_sincos_core_scalar(x, &s, &c); return c; }
double fabe13_sinc(double x) { if (fabs(x) < 1e-9) return 1.0; double s, c; fabe13_sincos_core_scalar(x, &s, &c); return s / x; }
double fabe13_tan(double x) { double s, c; fabe13_sincos_core_scalar(x, &s, &c); if (c == 0.0) return NAN; return s / c; }
double fabe13_cot(double x) { double s, c; fabe13_sincos_core_scalar(x, &s, &c); if (s == 0.0) return NAN; return c / s; }
double fabe13_atan(double x) { return atan(x); }
double fabe13_asin(double x) { return asin(x); }
double fabe13_acos(double x) { return acos(x); }

// --- Internal Implementation Details ---

// ** Ψ-Hyperbasis Scalar Core **
// Helper for small polynomial (used when |r| <= SMALL_DELTA)
static inline void small_sincos_poly(double r, double *s, double *c) {
    double r2 = r * r;
    double sin_poly = FABE13_SIN_COEFFS_SMALL[3];
    sin_poly = fma(r2, sin_poly, FABE13_SIN_COEFFS_SMALL[2]);
    sin_poly = fma(r2, sin_poly, FABE13_SIN_COEFFS_SMALL[1]);
    sin_poly = fma(r2, sin_poly, FABE13_SIN_COEFFS_SMALL[0]);
    double sinr = fma(r, fma(r2, sin_poly, 1.0), 0.0);

    double cos_poly = FABE13_COS_COEFFS_SMALL[3];
    cos_poly = fma(r2, cos_poly, FABE13_COS_COEFFS_SMALL[2]);
    cos_poly = fma(r2, cos_poly, FABE13_COS_COEFFS_SMALL[1]);
    cos_poly = fma(r2, cos_poly, FABE13_COS_COEFFS_SMALL[0]);
    double cosr = fma(r2, cos_poly, 1.0);
    *s = sinr; *c = cosr;
}

// Core scalar function using Ψ-Hyperbasis for main range
static void fabe13_sincos_core_scalar(double x, double* s, double* c) {
    double sin_r, cos_r;
    double ax = fabs(x);
    if (isnan(x) || isinf(x)) { *s = NAN; *c = NAN; return; }
    if (ax <= FABE13_TINY_THRESHOLD) { *s = x; *c = 1.0; return; }

    // Payne-Hanek Argument Reduction
    double p_hi = x * FABE13_TWO_OVER_PI_HI; double e1 = fma(x, FABE13_TWO_OVER_PI_HI, -p_hi);
    double p_lo = fma(x, FABE13_TWO_OVER_PI_LO, e1); double k_dd = p_hi + p_lo;
    long long k_int = llrint(k_dd); double k_dbl = (double)k_int;
    double t1 = k_dbl * FABE13_PI_OVER_2_HI; double e2 = fma(k_dbl, FABE13_PI_OVER_2_HI, -t1);
    double t2 = k_dbl * FABE13_PI_OVER_2_LO; double w = e2 + t2; double r = (x - t1) - w;
    int q = (int)(k_int & 3);
    double ar = fabs(r);

    // Select Path: Small Poly or Ψ-Hyperbasis
    if (ar <= FABE13_SMALL_DELTA) {
        small_sincos_poly(r, &sin_r, &cos_r);
    } else {
        // Ψ-Hyperbasis Calculation
        const double a1 = FABE13_HX_A1; const double a2 = FABE13_HX_A2; const double a3 = FABE13_HX_A3;
        const double b1 = FABE13_HX_B1; const double b2 = FABE13_HX_B2; const double b3 = FABE13_HX_B3;

        double r2 = r * r;
        double psi = r / (1.0 + 0.375 * r2); // 0.375 = 3/8
        double psi2 = psi * psi;

        // Evaluate sin approximation P_sin(psi2) = 1.0 + psi2*(-a1 + psi2*(a2 + psi2*(-a3)))
        double poly_sin = fma(-a3, psi2, a2);
        poly_sin = fma(poly_sin, psi2, -a1);
        poly_sin = fma(poly_sin, psi2, 1.0);
        sin_r = psi * poly_sin;

        // Evaluate cos approximation P_cos(psi2) = 1.0 + psi2*(-b1 + psi2*(b2 + psi2*(-b3)))
        double poly_cos = fma(-b3, psi2, b2);
        poly_cos = fma(poly_cos, psi2, -b1);
        poly_cos = fma(poly_cos, psi2, 1.0);
        cos_r = poly_cos;
    }

    // Quadrant Correction
    switch (q) { case 0: *s = sin_r; *c = cos_r; break; case 1: *s = cos_r; *c = -sin_r; break;
                 case 2: *s = -sin_r; *c = -cos_r; break; case 3: *s = -cos_r; *c = sin_r; break;
                 default: *s = NAN; *c = NAN; break; }
}

// Simple scalar loop (for tails of SIMD or if SIMD not available for large N)
static void fabe13_sincos_scalar_simple(const double* in, double* sin_out, double* cos_out, int n) {
    for (int i = 0; i < n; ++i) {
        fabe13_sincos_core_scalar(in[i], &sin_out[i], &cos_out[i]);
    }
}

// Unrolled scalar loop (for small N)
static void fabe13_sincos_scalar_unrolled(const double* FABE13_RESTRICT in, double* FABE13_RESTRICT sin_out, double* FABE13_RESTRICT cos_out, int n) {
    int i = 0;
    int limit4 = n & ~3; // Process chunks of 4
    for (; i < limit4; i += 4) {
        fabe13_sincos_core_scalar(in[i+0], &sin_out[i+0], &cos_out[i+0]);
        fabe13_sincos_core_scalar(in[i+1], &sin_out[i+1], &cos_out[i+1]);
        fabe13_sincos_core_scalar(in[i+2], &sin_out[i+2], &cos_out[i+2]);
        fabe13_sincos_core_scalar(in[i+3], &sin_out[i+3], &cos_out[i+3]);
    }
    for (; i < n; ++i) { // Handle remaining 1-3 elements
        fabe13_sincos_core_scalar(in[i], &sin_out[i], &cos_out[i]);
    }
}


// SIMD Polynomial Helpers (Still using Estrin/Horner for MAIN coeffs - TODO: Update to Ψ-Hyperbasis)
#if FABE13_PLATFORM_ARM
// ** NOTE: This still uses the OLD polynomial coeffs. Needs update for HX **
static inline float64x2_t fabe13_poly_neon(float64x2_t r_squared, const double* coeffs) {
    const float64x2_t C0 = vdupq_n_f64(coeffs[0]), C1 = vdupq_n_f64(coeffs[1]), C2 = vdupq_n_f64(coeffs[2]), C3 = vdupq_n_f64(coeffs[3]), C4 = vdupq_n_f64(coeffs[4]), C5 = vdupq_n_f64(coeffs[5]), C6 = vdupq_n_f64(coeffs[6]), C7 = vdupq_n_f64(coeffs[7]);
    float64x2_t z = r_squared; float64x2_t z2 = vmulq_f64(z, z); float64x2_t z4 = vmulq_f64(z2, z2);
    float64x2_t T01 = vfmaq_f64(C0, C1, z); float64x2_t T23 = vfmaq_f64(C2, C3, z); float64x2_t T45 = vfmaq_f64(C4, C5, z); float64x2_t T67 = vfmaq_f64(C6, C7, z);
    float64x2_t S03 = vfmaq_f64(T01, T23, z2); float64x2_t S47 = vfmaq_f64(T45, T67, z2);
    return vfmaq_f64(S03, S47, z4);
}
#endif
#if FABE13_PLATFORM_X86
// ** NOTE: This still uses the OLD polynomial coeffs. Needs update for HX **
static inline __m256d fabe13_poly_avx2(__m256d r_squared, const double* coeffs) {
    const __m256d C0 = _mm256_set1_pd(coeffs[0]), C1 = _mm256_set1_pd(coeffs[1]), C2 = _mm256_set1_pd(coeffs[2]), C3 = _mm256_set1_pd(coeffs[3]), C4 = _mm256_set1_pd(coeffs[4]), C5 = _mm256_set1_pd(coeffs[5]), C6 = _mm256_set1_pd(coeffs[6]), C7 = _mm256_set1_pd(coeffs[7]);
    __m256d z = r_squared; __m256d z2 = _mm256_mul_pd(z, z); __m256d z4 = _mm256_mul_pd(z2, z2);
    __m256d T01 = _mm256_fmadd_pd(C1, z, C0); __m256d T23 = _mm256_fmadd_pd(C3, z, C2); __m256d T45 = _mm256_fmadd_pd(C5, z, C4); __m256d T67 = _mm256_fmadd_pd(C7, z, C6);
    __m256d S03 = _mm256_fmadd_pd(T23, z2, T01); __m256d S47 = _mm256_fmadd_pd(T67, z2, T45);
    return _mm256_fmadd_pd(S47, z4, S03);
}
#if defined(__AVX512F__)
// ** NOTE: This still uses the OLD polynomial coeffs. Needs update for HX **
static inline __m512d fabe13_poly_avx512(__m512d r_squared, const double* coeffs) {
    const __m512d C0 = _mm512_set1_pd(coeffs[0]), C1 = _mm512_set1_pd(coeffs[1]), C2 = _mm512_set1_pd(coeffs[2]), C3 = _mm512_set1_pd(coeffs[3]), C4 = _mm512_set1_pd(coeffs[4]), C5 = _mm512_set1_pd(coeffs[5]), C6 = _mm512_set1_pd(coeffs[6]), C7 = _mm512_set1_pd(coeffs[7]);
    __m512d z = r_squared; __m512d z2 = _mm512_mul_pd(z, z); __m512d z4 = _mm512_mul_pd(z2, z2);
    __m512d T01 = _mm512_fmadd_pd(C1, z, C0); __m512d T23 = _mm512_fmadd_pd(C3, z, C2); __m512d T45 = _mm512_fmadd_pd(C5, z, C4); __m512d T67 = _mm512_fmadd_pd(C7, z, C6);
    __m512d S03 = _mm512_fmadd_pd(T23, z2, T01); __m512d S47 = _mm512_fmadd_pd(T67, z2, T45);
    return _mm512_fmadd_pd(S47, z4, S03);
}
#endif
#endif


// --- SIMD Backend Implementations (Original Logic + Alignment Flag for x86) ---
// ** NOTE: These still use the OLD polynomial helpers. Needs update for HX **

#if FABE13_PLATFORM_ARM
// NEON version - Always uses unaligned intrinsics now
static void fabe13_sincos_neon(const double* FABE13_RESTRICT in, double* FABE13_RESTRICT sin_out, double* FABE13_RESTRICT cos_out, int n) {
    const int SIMD_WIDTH = 2;
    const float64x2_t VEC_TWO_OVER_PI_HI = vdupq_n_f64(FABE13_TWO_OVER_PI_HI);
    const float64x2_t VEC_TWO_OVER_PI_LO = vdupq_n_f64(FABE13_TWO_OVER_PI_LO);
    const float64x2_t VEC_PI_OVER_2_HI = vdupq_n_f64(FABE13_PI_OVER_2_HI);
    const float64x2_t VEC_PI_OVER_2_LO = vdupq_n_f64(FABE13_PI_OVER_2_LO);
    const float64x2_t VEC_NAN = vdupq_n_f64(NAN); const float64x2_t VEC_ONE = vdupq_n_f64(1.0);
    const float64x2_t VEC_ZERO = vdupq_n_f64(0.0); const float64x2_t VEC_TINY = vdupq_n_f64(FABE13_TINY_THRESHOLD);
    const float64x2_t VEC_INF = vdupq_n_f64(INFINITY);
    const int64x2_t VEC_INT_3 = vdupq_n_s64(3), VEC_INT_0 = vdupq_n_s64(0), VEC_INT_1 = vdupq_n_s64(1), VEC_INT_2 = vdupq_n_s64(2);
    const uint64x2_t VEC_U64_ZERO = vdupq_n_u64(0);

    int i = 0;
    int limit = n - (n % SIMD_WIDTH);
    for (; i < limit; i += SIMD_WIDTH) {
        #if defined(__GNUC__) || defined(__clang__)
        __builtin_prefetch(&in[i + 64], 0, 0);
        #endif

        float64x2_t vx = vld1q_f64(&in[i]); // Unaligned load

        float64x2_t vax = vabsq_f64(vx);
        uint64x2_t non_nan_mask = vceqq_f64(vx, vx);
        uint64x2_t nan_mask = vceqq_u64(non_nan_mask, VEC_U64_ZERO);
        uint64x2_t inf_mask = vceqq_f64(vax, VEC_INF);
        uint64x2_t special_mask = vorrq_u64(inf_mask, nan_mask);
        uint64x2_t tiny_mask = vcleq_f64(vax, VEC_TINY);
        float64x2_t p_hi = vmulq_f64(vx, VEC_TWO_OVER_PI_HI);
        float64x2_t e1   = vfmsq_f64(p_hi, VEC_TWO_OVER_PI_HI, vx); e1 = vnegq_f64(e1);
        float64x2_t p_lo = vfmaq_f64(e1, vx, VEC_TWO_OVER_PI_LO);
        float64x2_t k_dd = vrndnq_f64(vaddq_f64(p_hi, p_lo));
        float64x2_t t1 = vmulq_f64(k_dd, VEC_PI_OVER_2_HI);
        float64x2_t e2 = vfmsq_f64(t1, VEC_PI_OVER_2_HI, k_dd); e2 = vnegq_f64(e2);
        float64x2_t t2 = vmulq_f64(k_dd, VEC_PI_OVER_2_LO);
        float64x2_t w = vaddq_f64(e2, t2);
        float64x2_t r = vsubq_f64(vsubq_f64(vx, t1), w);
        float64x2_t r2 = vmulq_f64(r, r);

        // *** TODO: Replace with NEON Ψ-Hyperbasis calculation ***
        float64x2_t sin_poly_r2 = fabe13_poly_neon(r2, FABE13_SIN_COEFFS_MAIN);
        float64x2_t cos_poly_r2 = fabe13_poly_neon(r2, FABE13_COS_COEFFS_MAIN);
        float64x2_t sin_r = vmulq_f64(r, sin_poly_r2);
        float64x2_t cos_r = cos_poly_r2;
        // *** END TODO ***

        int64x2_t vk_int = vcvtq_s64_f64(k_dd);
        int64x2_t vq = vandq_s64(vk_int, VEC_INT_3);
        uint64x2_t q0_mask = vceqq_s64(vq, VEC_INT_0); uint64x2_t q1_mask = vceqq_s64(vq, VEC_INT_1); uint64x2_t q2_mask = vceqq_s64(vq, VEC_INT_2);
        float64x2_t neg_sin_r = vnegq_f64(sin_r); float64x2_t neg_cos_r = vnegq_f64(cos_r);
        float64x2_t s_result = neg_cos_r; float64x2_t c_result = sin_r;
        s_result = vbslq_f64(q2_mask, neg_sin_r, s_result); c_result = vbslq_f64(q2_mask, neg_cos_r, c_result);
        s_result = vbslq_f64(q1_mask, cos_r, s_result);     c_result = vbslq_f64(q1_mask, neg_sin_r, c_result);
        s_result = vbslq_f64(q0_mask, sin_r, s_result);     c_result = vbslq_f64(q0_mask, cos_r, c_result);
        s_result = vbslq_f64(tiny_mask, vx, s_result);
        c_result = vbslq_f64(tiny_mask, VEC_ONE, c_result);
        s_result = vbslq_f64(special_mask, VEC_NAN, s_result);
        c_result = vbslq_f64(special_mask, VEC_NAN, c_result);

        vst1q_f64(&sin_out[i], s_result); // Unaligned store
        vst1q_f64(&cos_out[i], c_result); // Unaligned store
    }
    // Remainder loop using the simple scalar function
    if (i < n) {
        fabe13_sincos_scalar_simple(in + i, sin_out + i, cos_out + i, n - i);
    }
 }
#endif // FABE13_PLATFORM_ARM

#if FABE13_PLATFORM_X86
// AVX2 version - Accepts alignment flag
static void fabe13_sincos_avx2(const double* FABE13_RESTRICT in, double* FABE13_RESTRICT sin_out, double* FABE13_RESTRICT cos_out, int n, bool aligned) {
    const int SIMD_WIDTH = 4;
    const __m256d VEC_TWO_OVER_PI_HI = _mm256_set1_pd(FABE13_TWO_OVER_PI_HI);
    const __m256d VEC_TWO_OVER_PI_LO = _mm256_set1_pd(FABE13_TWO_OVER_PI_LO);
    const __m256d VEC_PI_OVER_2_HI = _mm256_set1_pd(FABE13_PI_OVER_2_HI);
    const __m256d VEC_PI_OVER_2_LO = _mm256_set1_pd(FABE13_PI_OVER_2_LO);
    const __m256d VEC_NAN = _mm256_set1_pd(NAN); const __m256d VEC_ONE = _mm256_set1_pd(1.0);
    const __m256d VEC_TINY = _mm256_set1_pd(FABE13_TINY_THRESHOLD); const __m256d VEC_ZERO = _mm256_setzero_pd();
    const __m256d VEC_SIGN_MASK = _mm256_set1_pd(-0.0); const __m256d VEC_INF = _mm256_set1_pd(INFINITY);
    const __m256d VEC_ROUND_BIAS = _mm256_set1_pd(6755399441055744.0);
    const __m256i VEC_ROUND_BIAS_I = _mm256_castpd_si256(VEC_ROUND_BIAS);
    const __m256i VEC_INT_3 = _mm256_set1_epi64x(3), VEC_INT_0 = _mm256_setzero_si256(), VEC_INT_1 = _mm256_set1_epi64x(1), VEC_INT_2 = _mm256_set1_epi64x(2);

    int i = 0;
    int limit = n - (n % SIMD_WIDTH);
    for (; i < limit; i += SIMD_WIDTH) {
        _mm_prefetch((const char*)(&in[i + 64]), _MM_HINT_T0);

        __m256d vx = aligned ? _mm256_load_pd(&in[i]) : _mm256_loadu_pd(&in[i]);

        __m256d vax = _mm256_andnot_pd(VEC_SIGN_MASK, vx);
        __m256d nan_mask = _mm256_cmp_pd(vx, vx, _CMP_UNORD_Q);
        __m256d inf_mask = _mm256_cmp_pd(vax, VEC_INF, _CMP_EQ_OQ);
        __m256d special_mask = _mm256_or_pd(nan_mask, inf_mask);
        __m256d tiny_mask = _mm256_cmp_pd(vax, VEC_TINY, _CMP_LE_OS);
        __m256d p_hi = _mm256_mul_pd(vx, VEC_TWO_OVER_PI_HI);
        __m256d e1 = _mm256_fmsub_pd(vx, VEC_TWO_OVER_PI_HI, p_hi);
        __m256d p_lo = _mm256_fmadd_pd(vx, VEC_TWO_OVER_PI_LO, e1);
        __m256d k_dd = _mm256_round_pd(_mm256_add_pd(p_hi, p_lo), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m256d t1 = _mm256_mul_pd(k_dd, VEC_PI_OVER_2_HI);
        __m256d e2 = _mm256_fmsub_pd(k_dd, VEC_PI_OVER_2_HI, t1);
        __m256d t2 = _mm256_mul_pd(k_dd, VEC_PI_OVER_2_LO);
        __m256d w = _mm256_add_pd(e2, t2);
        __m256d r = _mm256_sub_pd(_mm256_sub_pd(vx, t1), w);
        __m256d r2 = _mm256_mul_pd(r, r);

        // *** TODO: Replace with AVX2 Ψ-Hyperbasis calculation ***
        __m256d sin_poly_r2 = fabe13_poly_avx2(r2, FABE13_SIN_COEFFS_MAIN);
        __m256d cos_poly_r2 = fabe13_poly_avx2(r2, FABE13_COS_COEFFS_MAIN);
        __m256d sin_r = _mm256_mul_pd(r, sin_poly_r2);
        __m256d cos_r = cos_poly_r2;
        // *** END TODO ***

        __m256d k_plus_bias = _mm256_add_pd(k_dd, VEC_ROUND_BIAS);
        __m256i vk_int = _mm256_sub_epi64(_mm256_castpd_si256(k_plus_bias), VEC_ROUND_BIAS_I);
        __m256i vq = _mm256_and_si256(vk_int, VEC_INT_3);
        __m256d q0_mask = _mm256_castsi256_pd(_mm256_cmpeq_epi64(vq, VEC_INT_0));
        __m256d q1_mask = _mm256_castsi256_pd(_mm256_cmpeq_epi64(vq, VEC_INT_1));
        __m256d q2_mask = _mm256_castsi256_pd(_mm256_cmpeq_epi64(vq, VEC_INT_2));
        __m256d neg_sin_r = _mm256_xor_pd(sin_r, VEC_SIGN_MASK);
        __m256d neg_cos_r = _mm256_xor_pd(cos_r, VEC_SIGN_MASK);
        __m256d s_result = neg_cos_r; __m256d c_result = sin_r;
        s_result = _mm256_blendv_pd(s_result, neg_sin_r, q2_mask); c_result = _mm256_blendv_pd(c_result, neg_cos_r, q2_mask);
        s_result = _mm256_blendv_pd(s_result, cos_r, q1_mask);     c_result = _mm256_blendv_pd(c_result, neg_sin_r, q1_mask);
        s_result = _mm256_blendv_pd(s_result, sin_r, q0_mask);     c_result = _mm256_blendv_pd(c_result, cos_r, q0_mask);
        s_result = _mm256_blendv_pd(s_result, vx, tiny_mask);
        c_result = _mm256_blendv_pd(c_result, VEC_ONE, tiny_mask);
        s_result = _mm256_blendv_pd(s_result, VEC_NAN, special_mask);
        c_result = _mm256_blendv_pd(c_result, VEC_NAN, special_mask);

        if (aligned) {
            _mm256_store_pd(&sin_out[i], s_result);
            _mm256_store_pd(&cos_out[i], c_result);
        } else {
            _mm256_storeu_pd(&sin_out[i], s_result);
            _mm256_storeu_pd(&cos_out[i], c_result);
        }
    }
    // Remainder loop using the simple scalar function
    if (i < n) {
        fabe13_sincos_scalar_simple(in + i, sin_out + i, cos_out + i, n - i);
    }
 }

#if defined(__AVX512F__)
// AVX512 version - Accepts alignment flag
static void fabe13_sincos_avx512(const double* FABE13_RESTRICT in, double* FABE13_RESTRICT sin_out, double* FABE13_RESTRICT cos_out, int n, bool aligned) {
    const int SIMD_WIDTH = 8;
    const __m512d VEC_TWO_OVER_PI_HI = _mm512_set1_pd(FABE13_TWO_OVER_PI_HI);
    const __m512d VEC_TWO_OVER_PI_LO = _mm512_set1_pd(FABE13_TWO_OVER_PI_LO);
    const __m512d VEC_PI_OVER_2_HI = _mm512_set1_pd(FABE13_PI_OVER_2_HI);
    const __m512d VEC_PI_OVER_2_LO = _mm512_set1_pd(FABE13_PI_OVER_2_LO);
    const __m512d VEC_NAN = _mm512_set1_pd(NAN); const __m512d VEC_ONE = _mm512_set1_pd(1.0);
    const __m512d VEC_TINY = _mm512_set1_pd(FABE13_TINY_THRESHOLD); const __m512d VEC_ZERO = _mm512_setzero_pd();
    const __m512d VEC_INF = _mm512_set1_pd(INFINITY);
    const __m512i VEC_INT_3 = _mm512_set1_epi64(3), VEC_INT_0 = _mm512_setzero_si512(), VEC_INT_1 = _mm512_set1_epi64(1), VEC_INT_2 = _mm512_set1_epi64(2);

    int i = 0;
    int limit = n - (n % SIMD_WIDTH);
    for (; i < limit; i += SIMD_WIDTH) {
        _mm_prefetch((const char*)(&in[i + 128]), _MM_HINT_T0);

        __m512d vx = aligned ? _mm512_load_pd(&in[i]) : _mm512_loadu_pd(&in[i]);

        __m512d vax = _mm512_abs_pd(vx);
        __mmask8 nan_mask = _mm512_cmp_pd_mask(vx, vx, _CMP_UNORD_Q);
        __mmask8 inf_mask = _mm512_cmp_pd_mask(vax, VEC_INF, _CMP_EQ_OQ);
        __mmask8 special_mask = _kor_mask8(nan_mask, inf_mask);
        __mmask8 tiny_mask = _mm512_cmp_pd_mask(vax, VEC_TINY, _CMP_LE_OS);
        __m512d p_hi = _mm512_mul_pd(vx, VEC_TWO_OVER_PI_HI);
        __m512d e1 = _mm512_fmsub_pd(vx, VEC_TWO_OVER_PI_HI, p_hi);
        __m512d p_lo = _mm512_fmadd_pd(vx, VEC_TWO_OVER_PI_LO, e1);
        __m512d k_dd = _mm512_roundscale_pd(_mm512_add_pd(p_hi, p_lo), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m512d t1 = _mm512_mul_pd(k_dd, VEC_PI_OVER_2_HI);
        __m512d e2 = _mm512_fmsub_pd(k_dd, VEC_PI_OVER_2_HI, t1);
        __m512d t2 = _mm512_mul_pd(k_dd, VEC_PI_OVER_2_LO);
        __m512d w = _mm512_add_pd(e2, t2);
        __m512d r = _mm512_sub_pd(_mm512_sub_pd(vx, t1), w);
        __m512d r2 = _mm512_mul_pd(r, r);

        // *** TODO: Replace with AVX512 Ψ-Hyperbasis calculation ***
        __m512d sin_poly_r2 = fabe13_poly_avx512(r2, FABE13_SIN_COEFFS_MAIN);
        __m512d cos_poly_r2 = fabe13_poly_avx512(r2, FABE13_COS_COEFFS_MAIN);
        __m512d sin_r = _mm512_mul_pd(r, sin_poly_r2);
        __m512d cos_r = cos_poly_r2;
        // *** END TODO ***

        __m512i vk_int = _mm512_cvtpd_epi64(k_dd);
        __m512i vq = _mm512_and_si512(vk_int, VEC_INT_3);
        __mmask8 q0_mask = _mm512_cmpeq_epi64_mask(vq, VEC_INT_0);
        __mmask8 q1_mask = _mm512_cmpeq_epi64_mask(vq, VEC_INT_1);
        __mmask8 q2_mask = _mm512_cmpeq_epi64_mask(vq, VEC_INT_2);
        __m512d neg_sin_r = _mm512_sub_pd(VEC_ZERO, sin_r);
        __m512d neg_cos_r = _mm512_sub_pd(VEC_ZERO, cos_r);
        __m512d s_result = neg_cos_r; __m512d c_result = sin_r;
        s_result = _mm512_mask_blend_pd(q0_mask, s_result, sin_r);     c_result = _mm512_mask_blend_pd(q0_mask, c_result, cos_r);
        s_result = _mm512_mask_blend_pd(q1_mask, s_result, cos_r);     c_result = _mm512_mask_blend_pd(q1_mask, c_result, neg_sin_r);
        s_result = _mm512_mask_blend_pd(q2_mask, s_result, neg_sin_r); c_result = _mm512_mask_blend_pd(q2_mask, c_result, neg_cos_r);
        s_result = _mm512_mask_blend_pd(tiny_mask, s_result, vx);
        c_result = _mm512_mask_blend_pd(tiny_mask, c_result, VEC_ONE);
        s_result = _mm512_mask_blend_pd(special_mask, s_result, VEC_NAN);
        c_result = _mm512_mask_blend_pd(special_mask, c_result, VEC_NAN);

        if (aligned) {
            _mm512_store_pd(&sin_out[i], s_result);
            _mm512_store_pd(&cos_out[i], c_result);
        } else {
            _mm512_storeu_pd(&sin_out[i], s_result);
            _mm512_storeu_pd(&cos_out[i], c_result);
        }
    }
    // Remainder loop using the simple scalar function
    if (i < n) {
        fabe13_sincos_scalar_simple(in + i, sin_out + i, cos_out + i, n - i);
    }
 }
#endif // defined(__AVX512F__)
#endif // FABE13_PLATFORM_X86

/* --- NO BENCHMARK OR MAIN FUNCTION HERE --- */
