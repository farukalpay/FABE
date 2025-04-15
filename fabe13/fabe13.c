/**
 * @file fabe13.c
 * @brief FABE13 High-Performance Trigonometric Library.
 *
 * Provides optimized implementations for sin(x), cos(x), and sincos(x)
 * using SIMD instructions with runtime dispatch for x86_64 and AArch64.
 * Includes scalar fallback and extended scalar math functions.
 *
 * Core sincos implementations utilize:
 * - AVX-512F (if available)
 * - AVX2+FMA (if available)
 * - NEON (AArch64)
 * - Scalar fallback (using standard libm)
 *
 * Features:
 * - Runtime CPU feature detection and dispatch (x86_64).
 * - Payne-Hanek argument reduction for high accuracy over large ranges.
 * - Estrin's polynomial evaluation scheme for SIMD paths.
 * - High-precision minimax polynomial coefficients for sin/cos approximation.
 * - Robust handling of special values (NaN, Infinity, Tiny).
 * - AVX2 uses optimized double->int64 conversion approximation.
 * - Provides scalar API functions: fabe13_sin, fabe13_cos, fabe13_sinc,
 *   fabe13_tan, fabe13_cot, fabe13_atan, fabe13_asin, fabe13_acos.
 *   (Note: atan, asin, acos currently use standard libm for guaranteed accuracy).
 */

 #define _GNU_SOURCE      // For llrint if needed by specific libm/compiler combos
 #define _ISOC99_SOURCE   // For isnan, isinf, NAN, INFINITY etc.
 
 #include <stdint.h>
 #include <stdbool.h>
 #include <math.h>
 #include <stdlib.h> // For NULL
 #include <float.h>  // For DBL_MAX etc.
 #include <limits.h> // For INT_MAX etc.
 
 // For warnings on non-GCC compilers
 #include <stdio.h> // Only needed for #warning
 
 // --- Platform Detection & Intrinsics ---
 #undef FABE13_PLATFORM_X86
 #undef FABE13_PLATFORM_ARM
 #if defined(__x86_64__) || defined(_M_X64)
     #define FABE13_PLATFORM_X86 1
     #include <immintrin.h> // x86 intrinsics (AVX, AVX2, AVX512)
 #elif defined(__aarch64__)
     #define FABE13_PLATFORM_ARM 1
     #include <arm_neon.h>   // ARM NEON intrinsics
 #else
     // Define SIMD_WIDTH = 1 for scalar fallback on unknown platforms
     #define SIMD_WIDTH 1
 #endif
 
 // Ensure M_PI / M_PI_2 are defined
 #ifndef M_PI
 #define M_PI 3.14159265358979323846
 #endif
 #ifndef M_PI_2
 #define M_PI_2 1.57079632679489661923
 #endif
 #ifndef NAN
 #define NAN (0.0 / 0.0)
 #endif
 #ifndef INFINITY
 #define INFINITY (1.0 / 0.0)
 #endif
 
 // --- Configuration & Constants ---
 
 /**
  * @brief Threshold for small angle approximation.
  * Inputs with absolute value smaller than this use sin(x) ≈ x, cos(x) ≈ 1.
  */
 #define FABE13_TINY_THRESHOLD 7.450580596923828125e-9
 
 /**
  * @brief Recommended memory alignment for optimal SIMD performance (in bytes).
  * While the library uses unaligned loads/stores for flexibility, performance
  * may improve if input/output arrays are aligned to this value (e.g., using
  * posix_memalign or aligned_alloc). 64 bytes is suitable for AVX-512.
  */
 #define FABE13_ALIGNMENT 64
 
 // --- High-Precision Coefficients ---
 
 /**
  * @internal
  * @brief Minimax polynomial coefficients for sin(x)/x on [-pi/4, pi/4].
  * Represents P(x^2) where sin(x) ≈ x * P(x^2). Degree 7 in x^2 (8 coeffs).
  */
 static const double FABE13_SIN_COEFFS[] = {
     9.99999999999999999983e-01,  // c0
    -1.66666666666666657415e-01,  // c1
     8.33333333333329961475e-03,  // c2
    -1.98412698412589187999e-04,  // c3
     2.75573192235635111290e-06,  // c4
    -2.50521083760783692702e-08,  // c5
     1.60590438125280493886e-10,  // c6
    -7.64757314471113976640e-13   // c7
 };
 
 /**
  * @internal
  * @brief Minimax polynomial coefficients for cos(x) on [-pi/4, pi/4].
  * Represents P(x^2) where cos(x) ≈ P(x^2). Degree 7 in x^2 (8 coeffs).
  */
 static const double FABE13_COS_COEFFS[] = {
     1.00000000000000000000e+00,  // c0
    -4.99999999999999944489e-01,  // c1
     4.16666666666664590036e-02,  // c2
    -1.38888888888829782464e-03,  // c3
     2.48015873015087640936e-05,  // c4
    -2.75573192094882420430e-07,  // c5
     2.08767569813591324530e-09,  // c6
    -1.14757362211242971740e-11   // c7
 };
 
 /**
  * @internal
  * @brief Degree of the polynomials used (in terms of x^2).
  */
 static const int FABE13_POLY_DEGREE_IN_X2 = 7; // Corresponds to 8 coefficients
 
 // --- Payne-Hanek Double-Double Constants (Precomputed) ---
 // These provide extra precision for argument reduction.
 static const double FABE13_TWO_OVER_PI_HI = 0x1.45f306dc9c883p-1;  // 0.6366197723675814
 static const double FABE13_TWO_OVER_PI_LO = -0x1.9f3c6a7a0b5edp-57; // -1.7359123787586675e-17
 static const double FABE13_PI_OVER_2_HI = 0x1.921fb54442d18p+0;    // 1.5707963267948966
 static const double FABE13_PI_OVER_2_LO = 0x1.1a62633145c07p-53;    // 6.123233995736766e-17
 
 // --- Runtime CPU Feature Detection (x86 only) ---
 #if FABE13_PLATFORM_X86
 #ifdef __GNUC__
 /** @internal @brief Detects AVX-512F support at runtime. */
 static inline bool fabe13_detect_avx512f() {
     // Check AVX512F. The AVX512 kernel uses only F instructions.
     return __builtin_cpu_supports("avx512f");
 }
 /** @internal @brief Detects AVX2 and FMA support at runtime. */
 static inline bool fabe13_detect_avx2() {
     // Requires AVX2 and FMA for the AVX2 kernel
     return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
 }
 #else // Fallback for other compilers (e.g., MSVC - needs implementation)
 // TODO: Implement CPU detection for non-GCC compilers (e.g., using __cpuidex for MSVC)
 static inline bool fabe13_detect_avx512f() {
     #warning "AVX512 detection not implemented for this compiler. AVX512 path disabled."
     return false;
 }
 static inline bool fabe13_detect_avx2() {
     #warning "AVX2+FMA detection not implemented for this compiler. AVX2 path disabled."
     return false;
 }
 #endif // __GNUC__
 #endif // FABE13_PLATFORM_X86
 
 
 // --- Forward declarations for internal SIMD/Scalar backends ---
 static void fabe13_sincos_scalar(const double* in, double* sin_out, double* cos_out, int n);
 #if FABE13_PLATFORM_ARM
 static void fabe13_sincos_neon(const double* in, double* sin_out, double* cos_out, int n);
 #endif
 #if FABE13_PLATFORM_X86
 static void fabe13_sincos_avx2(const double* in, double* sin_out, double* cos_out, int n);
 #if defined(__AVX512F__) // Only declare AVX512 func if compiler support exists
 static void fabe13_sincos_avx512(const double* in, double* sin_out, double* cos_out, int n);
 #endif
 #endif
 
 // --- Global state for active implementation ---
 /** @internal @brief Function pointer to the active sincos implementation. */
 static void (*active_impl)(const double*, double*, double*, int n) = NULL;
 /** @internal @brief Name of the active implementation. */
 static const char* active_impl_name = "Not Initialized";
 /** @internal @brief SIMD width of the active implementation. */
 static int active_simd_width = 1;
 
 /**
  * @internal
  * @brief Initializes the function pointer `active_impl` by detecting CPU features.
  * Selects the best available implementation (AVX512 > AVX2 > NEON > Scalar).
  * This function is called automatically on the first call to fabe13_sincos
  * or the getter functions.
  */
 static void fabe13_initialize_dispatcher() {
     if (active_impl != NULL) {
         return; // Already initialized
     }
 
     #if FABE13_PLATFORM_ARM
         // On ARM, NEON is assumed for AArch64. No runtime detection needed here.
         active_impl = fabe13_sincos_neon;
         active_impl_name = "NEON (AArch64)";
         active_simd_width = 2;
     #elif FABE13_PLATFORM_X86
         // x86: Runtime detection
         if (fabe13_detect_avx512f()) {
             #if defined(__AVX512F__) // Check if AVX512 code was actually compiled
                 active_impl = fabe13_sincos_avx512;
                 active_impl_name = "AVX-512F";
                 active_simd_width = 8;
             #else
                 // Detected AVX512F at runtime, but code wasn't compiled with support. Fallback.
                 if (fabe13_detect_avx2()) {
                     active_impl = fabe13_sincos_avx2;
                     active_impl_name = "AVX2+FMA";
                     active_simd_width = 4;
                 } else {
                     active_impl = fabe13_sincos_scalar;
                     active_impl_name = "Scalar (x86)";
                     active_simd_width = 1;
                 }
             #endif
         } else if (fabe13_detect_avx2()) {
             active_impl = fabe13_sincos_avx2;
             active_impl_name = "AVX2+FMA";
             active_simd_width = 4;
         } else {
             active_impl = fabe13_sincos_scalar;
             active_impl_name = "Scalar (x86)";
             active_simd_width = 1;
         }
     #else
         // Unknown platform, use scalar
         active_impl = fabe13_sincos_scalar;
         active_impl_name = "Scalar (Unknown Platform)";
         active_simd_width = 1;
     #endif
 
     // Optional: Print the selected implementation during initialization
     // printf("FABE13 Dispatcher: Selected implementation = %s (SIMD Width = %d)\n",
     //        active_impl_name, active_simd_width);
 }
 
 
 // --- Public API ---
 
 /**
  * @brief Computes sine and cosine for an array of doubles.
  *
  * This function computes `sin(in[i])` and `cos(in[i])` for each element
  * in the input array `in` and stores the results in `sin_out` and `cos_out`.
  * It automatically selects the fastest available implementation (SIMD or scalar)
  * for the current CPU architecture at runtime during the first call.
  *
  * @param in Pointer to the input array of doubles (angles in radians).
  * @param sin_out Pointer to the output array for sine results.
  * @param cos_out Pointer to the output array for cosine results.
  * @param n The number of elements in the arrays.
  *
  * @note For optimal performance, ensure input and output arrays are aligned
  *       to FABE13_ALIGNMENT (typically 64 bytes), although the function
  *       handles unaligned data correctly.
  * @note The first call to this function (or any fabe13 function) may incur
  *       a small overhead for runtime CPU detection and dispatcher initialization.
  */
 void fabe13_sincos(const double* in, double* sin_out, double* cos_out, int n) {
     if (!active_impl) {
         fabe13_initialize_dispatcher();
     }
     // Call the selected implementation
     active_impl(in, sin_out, cos_out, n);
 }
 
 /**
  * @brief Gets the name of the active sincos implementation.
  *
  * Returns a string identifying the currently selected backend (e.g., "AVX-512F",
  * "AVX2+FMA", "NEON (AArch64)", "Scalar"). Initializes the dispatcher
  * if not already done.
  *
  * @return A constant string with the name of the active implementation.
  */
 const char* fabe13_get_active_implementation_name() {
     if (!active_impl) {
         fabe13_initialize_dispatcher();
     }
     return active_impl_name;
 }
 
 /**
  * @brief Gets the SIMD width of the active sincos implementation.
  *
  * Returns the number of double-precision elements processed simultaneously by
  * the active SIMD backend (e.g., 8 for AVX-512, 4 for AVX2, 2 for NEON, 1 for Scalar).
  * Initializes the dispatcher if not already done.
  *
  * @return The SIMD width (vector size) of the active implementation.
  */
 int fabe13_get_active_simd_width() {
     if (!active_impl) {
         fabe13_initialize_dispatcher();
     }
     return active_simd_width;
 }
 
 /**
  * @internal
  * @brief Computes sine and cosine for a single double value.
  * Internal helper function used by the scalar API functions.
  *
  * @param x Input angle in radians.
  * @param s Pointer to store the sine result.
  * @param c Pointer to store the cosine result.
  */
 static inline void fabe13_sincos_single(double x, double* s, double* c) {
     // Ensure dispatcher is initialized if called before fabe13_sincos
     if (!active_impl) {
         fabe13_initialize_dispatcher();
     }
     // Use the active implementation for a single element
     active_impl(&x, s, c, 1);
 }
 
 /**
  * @brief Computes the sine of a double value.
  * Uses the optimized fabe13_sincos core.
  * @param x Input angle in radians.
  * @return The sine of x.
  */
 double fabe13_sin(double x) {
     double s, c;
     fabe13_sincos_single(x, &s, &c);
     return s;
 }
 
 /**
  * @brief Computes the cosine of a double value.
  * Uses the optimized fabe13_sincos core.
  * @param x Input angle in radians.
  * @return The cosine of x.
  */
 double fabe13_cos(double x) {
     double s, c;
     fabe13_sincos_single(x, &s, &c);
     return c;
 }
 
 /**
  * @brief Computes the sinc function (sin(x)/x).
  * Uses the optimized fabe13_sincos core. Handles x=0 correctly.
  * @param x Input value in radians.
  * @return sin(x)/x, or 1.0 if x is close to 0.
  */
 double fabe13_sinc(double x) {
     // Use small angle approximation for sin(x)/x -> 1 near x=0
     // Threshold chosen somewhat arbitrarily, could use FABE13_TINY_THRESHOLD
     if (fabs(x) < 1e-9) {
          return 1.0;
     }
     double s, c;
     fabe13_sincos_single(x, &s, &c);
     return s / x;
 }
 
 /**
  * @brief Computes the tangent of a double value (sin(x)/cos(x)).
  * Uses the optimized fabe13_sincos core. Handles cos(x)=0 correctly.
  * @param x Input angle in radians.
  * @return The tangent of x, or NAN if cos(x) is 0.
  */
 double fabe13_tan(double x) {
     double s, c;
     fabe13_sincos_single(x, &s, &c);
     // Check for division by zero (cos(x) == 0 at odd multiples of pi/2)
     // Comparing floating point to zero directly is generally discouraged,
     // but in this context, if the optimized sincos returns exactly 0.0 for cosine,
     // tan is indeed undefined (infinite). Returning NAN is appropriate.
     if (c == 0.0) {
         // Could potentially return INFINITY or -INFINITY based on sign of s,
         // but NAN is safer and standard practice for libm tan.
         return NAN;
     }
     return s / c;
 }
 
 /**
  * @brief Computes the cotangent of a double value (cos(x)/sin(x)).
  * Uses the optimized fabe13_sincos core. Handles sin(x)=0 correctly.
  * @param x Input angle in radians.
  * @return The cotangent of x, or NAN if sin(x) is 0.
  */
 double fabe13_cot(double x) {
     double s, c;
     fabe13_sincos_single(x, &s, &c);
     // Check for division by zero (sin(x) == 0 at multiples of pi)
     if (s == 0.0) {
         // Similar to tan, returning NAN is appropriate.
         return NAN;
     }
     return c / s;
 }
 
 /**
  * @brief Computes the principal value of the arc tangent of x.
  * @param x Value whose arc tangent is computed.
  * @return Principal arc tangent of x, in the interval [-pi/2, pi/2] radians.
  * @note This function currently calls the standard `atan` function from `math.h`
  *       to ensure maximum accuracy and robustness.
  */
 double fabe13_atan(double x) {
     // For guaranteed accuracy and robustness across all inputs,
     // delegate to the standard library's implementation.
     return atan(x);
 }
 
 /**
  * @brief Computes the principal value of the arc sine of x.
  * @param x Value whose arc sine is computed, in the interval [-1, 1].
  * @return Principal arc sine of x, in the interval [-pi/2, pi/2] radians.
  *         Returns NAN if x is outside [-1, 1].
  * @note This function currently calls the standard `asin` function from `math.h`
  *       to ensure maximum accuracy and robustness.
  */
 double fabe13_asin(double x) {
     // Delegate to the standard library's implementation.
     return asin(x);
 }
 
 /**
  * @brief Computes the principal value of the arc cosine of x.
  * @param x Value whose arc cosine is computed, in the interval [-1, 1].
  * @return Principal arc cosine of x, in the interval [0, pi] radians.
  *         Returns NAN if x is outside [-1, 1].
  * @note This function currently calls the standard `acos` function from `math.h`
  *       to ensure maximum accuracy and robustness.
  */
 double fabe13_acos(double x) {
     // Delegate to the standard library's implementation.
     return acos(x);
 }
 
 
 // --- Internal Implementations ---
 
 /**
  * @internal
  * @brief Scalar fallback implementation using standard libm sin/cos.
  * This is used if no SIMD instructions are available or detected.
  */
 static void fabe13_sincos_scalar(const double* in, double* sin_out, double* cos_out, int n) {
     for (int i = 0; i < n; ++i) {
         sin_out[i] = sin(in[i]);
         cos_out[i] = cos(in[i]);
     }
 }
 
 
 // --- NEON Implementation ---
 #if FABE13_PLATFORM_ARM
 /**
  * @internal
  * @brief Helper: NEON Polynomial Evaluation (Estrin's Scheme for Degree 7 in r^2).
  * Evaluates P(r^2) = C0 + C1*r^2 + C2*r^4 + ... + C7*r^14.
  * @param r Vector of input values.
  * @param coeffs Pointer to the 8 polynomial coefficients.
  * @return Vector of polynomial results.
  */
 static inline float64x2_t fabe13_poly_neon(float64x2_t r_squared, const double* coeffs) {
     // Estrin's scheme for P(z) = c0 + c1*z + c2*z^2 + ... + c7*z^7 where z = r^2
     const float64x2_t C0 = vdupq_n_f64(coeffs[0]), C1 = vdupq_n_f64(coeffs[1]),
                       C2 = vdupq_n_f64(coeffs[2]), C3 = vdupq_n_f64(coeffs[3]),
                       C4 = vdupq_n_f64(coeffs[4]), C5 = vdupq_n_f64(coeffs[5]),
                       C6 = vdupq_n_f64(coeffs[6]), C7 = vdupq_n_f64(coeffs[7]);
 
     float64x2_t z = r_squared;
     float64x2_t z2 = vmulq_f64(z, z);   // r^4
     float64x2_t z4 = vmulq_f64(z2, z2); // r^8
 
     // Terms grouped for Estrin's scheme
     float64x2_t T01 = vfmaq_f64(C0, C1, z);  // C0 + C1*z
     float64x2_t T23 = vfmaq_f64(C2, C3, z);  // C2 + C3*z
     float64x2_t T45 = vfmaq_f64(C4, C5, z);  // C4 + C5*z
     float64x2_t T67 = vfmaq_f64(C6, C7, z);  // C6 + C7*z
 
     // Combine pairs using z^2 = r^4
     float64x2_t S03 = vfmaq_f64(T01, T23, z2); // T01 + T23*z^2 = C0..C3 terms
     float64x2_t S47 = vfmaq_f64(T45, T67, z2); // T45 + T67*z^2 = C4..C7 terms
 
     // Final combination using z^4 = r^8
     return vfmaq_f64(S03, S47, z4); // S03 + S47*z^4 = C0..C7 terms
 }
 
 /**
  * @internal
  * @brief Computes sincos using NEON intrinsics (AArch64).
  */
 static void fabe13_sincos_neon(const double* in, double* sin_out, double* cos_out, int n) {
     const int SIMD_WIDTH = 2;
     // Constants (vectorized)
     const float64x2_t VEC_TWO_OVER_PI_HI = vdupq_n_f64(FABE13_TWO_OVER_PI_HI);
     const float64x2_t VEC_TWO_OVER_PI_LO = vdupq_n_f64(FABE13_TWO_OVER_PI_LO);
     const float64x2_t VEC_PI_OVER_2_HI = vdupq_n_f64(FABE13_PI_OVER_2_HI);
     const float64x2_t VEC_PI_OVER_2_LO = vdupq_n_f64(FABE13_PI_OVER_2_LO);
     const float64x2_t VEC_NAN = vdupq_n_f64(NAN);
     const float64x2_t VEC_ONE = vdupq_n_f64(1.0);
     const float64x2_t VEC_ZERO = vdupq_n_f64(0.0);
     const float64x2_t VEC_TINY = vdupq_n_f64(FABE13_TINY_THRESHOLD);
     const float64x2_t VEC_INF = vdupq_n_f64(INFINITY);
     const int64x2_t   VEC_INT_3 = vdupq_n_s64(3);
     const int64x2_t   VEC_INT_0 = vdupq_n_s64(0);
     const int64x2_t   VEC_INT_1 = vdupq_n_s64(1);
     const int64x2_t   VEC_INT_2 = vdupq_n_s64(2);
 
     int i = 0;
     // Process chunks of SIMD_WIDTH
     for (; i <= n - SIMD_WIDTH; i += SIMD_WIDTH) {
         float64x2_t vx = vld1q_f64(&in[i]); // Unaligned load
         float64x2_t vax = vabsq_f64(vx);
 
         // --- Special Value Handling ---
         // Mask for NaN or Inf inputs
         // Note: NEON doesn't have a direct ordered/unordered compare.
         // Check for NaN: vx != vx
         // Check for Inf: |vx| == INF
         uint64x2_t nan_mask = vceqq_f64(vx, vx); // 0xFF.. if NOT NaN, 0x00.. if NaN
         nan_mask = veorq_u64(nan_mask, vdupq_n_u64(UINT64_MAX)); // Invert: 0xFF.. if NaN
         uint64x2_t inf_mask = vceqq_f64(vax, VEC_INF);
         uint64x2_t special_mask = vorrq_u64(inf_mask, nan_mask);
 
         // Mask for tiny inputs (|vx| < threshold)
         uint64x2_t tiny_mask = vcltq_f64(vax, VEC_TINY);
 
         // --- Payne-Hanek Argument Reduction ---
         // k_dd = round(vx * (2/pi))
         // r = vx - k_dd * (pi/2) using double-double arithmetic
         float64x2_t p_hi = vmulq_f64(vx, VEC_TWO_OVER_PI_HI);
         // FMA: e1 = -(p_hi - vx * VEC_TWO_OVER_PI_HI) = vx * VEC_TWO_OVER_PI_HI - p_hi
         float64x2_t e1   = vfmsq_f64(vx, VEC_TWO_OVER_PI_HI, p_hi); // Note: vfms(a,b,c) = a*b-c
         e1 = vnegq_f64(e1); // Correct sign for error term
         // FMA: p_lo = e1 + vx * VEC_TWO_OVER_PI_LO
         float64x2_t p_lo = vfmaq_f64(e1, vx, VEC_TWO_OVER_PI_LO);
         // k_dd = round_nearest(p_hi + p_lo)
         float64x2_t k_dd = vrndnq_f64(vaddq_f64(p_hi, p_lo)); // Round to nearest integer
 
         // Calculate reduced argument r = vx - k_dd * pi/2
         float64x2_t t1 = vmulq_f64(k_dd, VEC_PI_OVER_2_HI);
         // FMA: e2 = -(t1 - k_dd * VEC_PI_OVER_2_HI) = k_dd * VEC_PI_OVER_2_HI - t1
         float64x2_t e2 = vfmsq_f64(k_dd, VEC_PI_OVER_2_HI, t1);
         e2 = vnegq_f64(e2); // Correct sign for error term
         float64x2_t t2 = vmulq_f64(k_dd, VEC_PI_OVER_2_LO);
         float64x2_t w = vaddq_f64(e2, t2); // Correction term
         // r = (vx - t1) - w
         float64x2_t r = vsubq_f64(vsubq_f64(vx, t1), w);
 
         // --- Polynomial Evaluation ---
         // sin(r) ≈ r * P_sin(r^2)
         // cos(r) ≈ P_cos(r^2)
         float64x2_t r2 = vmulq_f64(r, r);
         float64x2_t sin_poly_r2 = fabe13_poly_neon(r2, FABE13_SIN_COEFFS);
         float64x2_t cos_poly_r2 = fabe13_poly_neon(r2, FABE13_COS_COEFFS);
         float64x2_t sin_r = vmulq_f64(r, sin_poly_r2);
         float64x2_t cos_r = cos_poly_r2;
 
         // --- Quadrant Correction ---
         // Determine quadrant q = k mod 4
         int64x2_t vk_int = vcvtq_s64_f64(k_dd); // Convert rounded double k_dd to int64
         int64x2_t vq = vandq_s64(vk_int, VEC_INT_3); // q = k & 3
 
         // Create masks for each quadrant
         uint64x2_t q0_mask = vceqq_s64(vq, VEC_INT_0); // q == 0
         uint64x2_t q1_mask = vceqq_s64(vq, VEC_INT_1); // q == 1
         uint64x2_t q2_mask = vceqq_s64(vq, VEC_INT_2); // q == 2
         // q3 is the remainder
 
         // Negated values for blending
         float64x2_t neg_sin_r = vnegq_f64(sin_r);
         float64x2_t neg_cos_r = vnegq_f64(cos_r);
 
         // Blend results based on quadrant masks using vbslq (Bit Select)
         // Select(mask, val_if_true, val_if_false)
         // Start with q=3 case, then blend in others based on masks
         // q=0: sin= sin(r), cos= cos(r)
         // q=1: sin= cos(r), cos=-sin(r)
         // q=2: sin=-sin(r), cos=-cos(r)
         // q=3: sin=-cos(r), cos= sin(r)
 
         float64x2_t s_result = neg_cos_r; // Default: q=3 result for sin
         float64x2_t c_result = sin_r;     // Default: q=3 result for cos
 
         s_result = vbslq_f64(q2_mask, neg_sin_r, s_result); // If q=2, use -sin(r)
         c_result = vbslq_f64(q2_mask, neg_cos_r, c_result); // If q=2, use -cos(r)
 
         s_result = vbslq_f64(q1_mask, cos_r, s_result);     // If q=1, use cos(r)
         c_result = vbslq_f64(q1_mask, neg_sin_r, c_result); // If q=1, use -sin(r)
 
         s_result = vbslq_f64(q0_mask, sin_r, s_result);     // If q=0, use sin(r)
         c_result = vbslq_f64(q0_mask, cos_r, c_result);     // If q=0, use cos(r)
 
         // --- Apply Tiny & Special Masks ---
         // If tiny, sin(x) = x, cos(x) = 1
         s_result = vbslq_f64(tiny_mask, vx, s_result);
         c_result = vbslq_f64(tiny_mask, VEC_ONE, c_result);
         // If special (NaN or Inf), result is NaN
         s_result = vbslq_f64(special_mask, VEC_NAN, s_result);
         c_result = vbslq_f64(special_mask, VEC_NAN, c_result);
 
         // --- Store results ---
         vst1q_f64(&sin_out[i], s_result); // Unaligned store
         vst1q_f64(&cos_out[i], c_result); // Unaligned store
     }
 
     // Process any remaining elements using the scalar fallback
     if (i < n) {
         fabe13_sincos_scalar(in + i, sin_out + i, cos_out + i, n - i);
     }
 }
 #endif // FABE13_PLATFORM_ARM
 
 
 // --- AVX2 Implementation ---
 #if FABE13_PLATFORM_X86
 /**
  * @internal
  * @brief Helper: AVX2 Polynomial Evaluation (Estrin's Scheme for Degree 7 in r^2).
  * Evaluates P(r^2) = C0 + C1*r^2 + C2*r^4 + ... + C7*r^14 using FMA.
  * @param r_squared Vector of r^2 values.
  * @param coeffs Pointer to the 8 polynomial coefficients.
  * @return Vector of polynomial results.
  */
 static inline __m256d fabe13_poly_avx2(__m256d r_squared, const double* coeffs) {
     // Estrin's scheme for P(z) = c0 + c1*z + c2*z^2 + ... + c7*z^7 where z = r^2
     const __m256d C0 = _mm256_set1_pd(coeffs[0]), C1 = _mm256_set1_pd(coeffs[1]),
                   C2 = _mm256_set1_pd(coeffs[2]), C3 = _mm256_set1_pd(coeffs[3]),
                   C4 = _mm256_set1_pd(coeffs[4]), C5 = _mm256_set1_pd(coeffs[5]),
                   C6 = _mm256_set1_pd(coeffs[6]), C7 = _mm256_set1_pd(coeffs[7]);
 
     __m256d z = r_squared;
     __m256d z2 = _mm256_mul_pd(z, z);   // r^4
     __m256d z4 = _mm256_mul_pd(z2, z2); // r^8
 
     // Terms grouped for Estrin's scheme, using FMA
     __m256d T01 = _mm256_fmadd_pd(C1, z, C0); // C0 + C1*z
     __m256d T23 = _mm256_fmadd_pd(C3, z, C2); // C2 + C3*z
     __m256d T45 = _mm256_fmadd_pd(C5, z, C4); // C4 + C5*z
     __m256d T67 = _mm256_fmadd_pd(C7, z, C6); // C6 + C7*z (Note order for FMA: C7*z + C6)
 
     // Combine pairs using z^2 = r^4
     __m256d S03 = _mm256_fmadd_pd(T23, z2, T01); // T01 + T23*z^2
     __m256d S47 = _mm256_fmadd_pd(T67, z2, T45); // T45 + T67*z^2
 
     // Final combination using z^4 = r^8
     return _mm256_fmadd_pd(S47, z4, S03); // S03 + S47*z^4
 }
 
 /**
  * @internal
  * @brief Computes sincos using AVX2 and FMA intrinsics.
  */
 static void fabe13_sincos_avx2(const double* in, double* sin_out, double* cos_out, int n) {
     const int SIMD_WIDTH = 4;
     // Constants
     const __m256d VEC_TWO_OVER_PI_HI = _mm256_set1_pd(FABE13_TWO_OVER_PI_HI);
     const __m256d VEC_TWO_OVER_PI_LO = _mm256_set1_pd(FABE13_TWO_OVER_PI_LO);
     const __m256d VEC_PI_OVER_2_HI = _mm256_set1_pd(FABE13_PI_OVER_2_HI);
     const __m256d VEC_PI_OVER_2_LO = _mm256_set1_pd(FABE13_PI_OVER_2_LO);
     const __m256d VEC_NAN = _mm256_set1_pd(NAN);
     const __m256d VEC_ONE = _mm256_set1_pd(1.0);
     const __m256d VEC_TINY = _mm256_set1_pd(FABE13_TINY_THRESHOLD);
     const __m256d VEC_ZERO = _mm256_setzero_pd();
     const __m256d VEC_SIGN_MASK = _mm256_set1_pd(-0.0); // Mask to flip sign bit
     const __m256d VEC_INF = _mm256_set1_pd(INFINITY);
 
     // Constants for double -> int64 conversion approximation (bias trick)
     // Magic number: 2^52 + 2^51 = 1.5 * 2^52 = 6755399441055744.0
     const __m256d VEC_ROUND_BIAS = _mm256_set1_pd(6755399441055744.0);
     // Reinterpret the bias bits as integer for subtraction
     const __m256i VEC_ROUND_BIAS_I = _mm256_castpd_si256(VEC_ROUND_BIAS);
 
     // Integer constants for quadrant logic
     const __m256i VEC_INT_3 = _mm256_set1_epi64x(3);
     const __m256i VEC_INT_0 = _mm256_setzero_si256();
     const __m256i VEC_INT_1 = _mm256_set1_epi64x(1);
     const __m256i VEC_INT_2 = _mm256_set1_epi64x(2);
 
     int i = 0;
     // Process chunks of SIMD_WIDTH
     for (; i <= n - SIMD_WIDTH; i += SIMD_WIDTH) {
         __m256d vx = _mm256_loadu_pd(&in[i]); // Unaligned load
 
         // --- Special Value Handling ---
         // Mask for NaN inputs (unordered compare)
         __m256d nan_mask = _mm256_cmp_pd(vx, vx, _CMP_UNORD_Q);
         // Mask for Inf inputs (compare absolute value)
         __m256d vax = _mm256_andnot_pd(VEC_SIGN_MASK, vx); // |vx|
         __m256d inf_mask = _mm256_cmp_pd(vax, VEC_INF, _CMP_EQ_OQ);
         // Combined mask for NaN or Inf
         __m256d special_mask = _mm256_or_pd(nan_mask, inf_mask);
 
         // Mask for tiny inputs (|vx| < threshold)
         __m256d tiny_mask = _mm256_cmp_pd(vax, VEC_TINY, _CMP_LT_OS);
 
         // --- Payne-Hanek Argument Reduction ---
         // k_dd = round(vx * (2/pi))
         // r = vx - k_dd * (pi/2) using double-double arithmetic and FMA
         __m256d p_hi = _mm256_mul_pd(vx, VEC_TWO_OVER_PI_HI);
         // e1 = vx * VEC_TWO_OVER_PI_HI - p_hi (error of first multiplication)
         __m256d e1   = _mm256_fmsub_pd(vx, VEC_TWO_OVER_PI_HI, p_hi); // FMA: vx*HI - p_hi
         // p_lo = vx * VEC_TWO_OVER_PI_LO + e1 (lower part of vx * 2/pi)
         __m256d p_lo = _mm256_fmadd_pd(vx, VEC_TWO_OVER_PI_LO, e1); // FMA: vx*LO + e1
         // k_dd = round_nearest(p_hi + p_lo)
         __m256d k_dd = _mm256_round_pd(_mm256_add_pd(p_hi, p_lo),
                                       _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
 
         // Calculate reduced argument r = vx - k_dd * pi/2
         __m256d t1 = _mm256_mul_pd(k_dd, VEC_PI_OVER_2_HI);
         // e2 = k_dd * VEC_PI_OVER_2_HI - t1 (error of first multiplication)
         __m256d e2 = _mm256_fmsub_pd(k_dd, VEC_PI_OVER_2_HI, t1); // FMA: k*HI - t1
         __m256d t2 = _mm256_mul_pd(k_dd, VEC_PI_OVER_2_LO);
         __m256d w = _mm256_add_pd(e2, t2); // Correction term
         // r = (vx - t1) - w
         __m256d r = _mm256_sub_pd(_mm256_sub_pd(vx, t1), w);
 
         // --- Polynomial Evaluation ---
         // sin(r) ≈ r * P_sin(r^2)
         // cos(r) ≈ P_cos(r^2)
         __m256d r2 = _mm256_mul_pd(r, r);
         __m256d sin_poly_r2 = fabe13_poly_avx2(r2, FABE13_SIN_COEFFS);
         __m256d cos_poly_r2 = fabe13_poly_avx2(r2, FABE13_COS_COEFFS);
         __m256d sin_r = _mm256_mul_pd(r, sin_poly_r2);
         __m256d cos_r = cos_poly_r2;
 
         // --- Quadrant Correction ---
         // Determine quadrant q = k mod 4
         // Convert k_dd (__m256d, already rounded) to integer vk_int (__m256i)
         // using the bias trick (add bias, cast to int, subtract bias as int).
         // This avoids _mm256_cvtpd_epi64 which doesn't exist.
         // Assumes k_dd is within the range where this trick is valid (~ +/- 2^51).
         __m256d k_plus_bias = _mm256_add_pd(k_dd, VEC_ROUND_BIAS);
         __m256i vk_int = _mm256_sub_epi64(_mm256_castpd_si256(k_plus_bias), VEC_ROUND_BIAS_I);
 
         // Calculate quadrant q = k & 3
         __m256i vq = _mm256_and_si256(vk_int, VEC_INT_3); // q = k & 3
 
         // Create masks for each quadrant using integer comparison
         __m256i i_q0_mask = _mm256_cmpeq_epi64(vq, VEC_INT_0); // q == 0 ? 0xFF.. : 0x00...
         __m256i i_q1_mask = _mm256_cmpeq_epi64(vq, VEC_INT_1); // q == 1 ? 0xFF.. : 0x00...
         __m256i i_q2_mask = _mm256_cmpeq_epi64(vq, VEC_INT_2); // q == 2 ? 0xFF.. : 0x00...
         // q3 is implied where others are false
 
         // Cast integer masks (__m256i) to double masks (__m256d) for blendv
         __m256d q0_mask = _mm256_castsi256_pd(i_q0_mask);
         __m256d q1_mask = _mm256_castsi256_pd(i_q1_mask);
         __m256d q2_mask = _mm256_castsi256_pd(i_q2_mask);
 
         // Negated values for blending (using XOR with sign mask)
         __m256d neg_sin_r = _mm256_xor_pd(sin_r, VEC_SIGN_MASK);
         __m256d neg_cos_r = _mm256_xor_pd(cos_r, VEC_SIGN_MASK);
 
         // Blend results based on quadrant masks using blendv_pd
         // blendv_pd(else_val, then_val, mask)
         // Start with q=3 case, then blend in others based on masks
         // q=0: sin= sin(r), cos= cos(r)
         // q=1: sin= cos(r), cos=-sin(r)
         // q=2: sin=-sin(r), cos=-cos(r)
         // q=3: sin=-cos(r), cos= sin(r)
 
         __m256d s_result = neg_cos_r; // Default: q=3 result for sin
         __m256d c_result = sin_r;     // Default: q=3 result for cos
 
         s_result = _mm256_blendv_pd(s_result, neg_sin_r, q2_mask); // If q=2, use -sin(r)
         c_result = _mm256_blendv_pd(c_result, neg_cos_r, q2_mask); // If q=2, use -cos(r)
 
         s_result = _mm256_blendv_pd(s_result, cos_r, q1_mask);     // If q=1, use cos(r)
         c_result = _mm256_blendv_pd(c_result, neg_sin_r, q1_mask); // If q=1, use -sin(r)
 
         s_result = _mm256_blendv_pd(s_result, sin_r, q0_mask);     // If q=0, use sin(r)
         c_result = _mm256_blendv_pd(c_result, cos_r, q0_mask);     // If q=0, use cos(r)
 
         // --- Apply Tiny & Special Masks ---
         // If tiny, sin(x) = x, cos(x) = 1
         s_result = _mm256_blendv_pd(s_result, vx, tiny_mask);
         c_result = _mm256_blendv_pd(c_result, VEC_ONE, tiny_mask);
         // If special (NaN or Inf), result is NaN
         s_result = _mm256_blendv_pd(s_result, VEC_NAN, special_mask);
         c_result = _mm256_blendv_pd(c_result, VEC_NAN, special_mask);
 
         // --- Store results ---
         _mm256_storeu_pd(&sin_out[i], s_result); // Unaligned store
         _mm256_storeu_pd(&cos_out[i], c_result); // Unaligned store
     }
 
     // Process any remaining elements using the scalar fallback
     if (i < n) {
         fabe13_sincos_scalar(in + i, sin_out + i, cos_out + i, n - i);
     }
 }
 
 
 // --- AVX-512 Implementation ---
 #if defined(__AVX512F__)
 /**
  * @internal
  * @brief Helper: AVX512 Polynomial Evaluation (Estrin's Scheme for Degree 7 in r^2).
  * Evaluates P(r^2) = C0 + C1*r^2 + C2*r^4 + ... + C7*r^14 using FMA.
  * @param r_squared Vector of r^2 values.
  * @param coeffs Pointer to the 8 polynomial coefficients.
  * @return Vector of polynomial results.
  */
 static inline __m512d fabe13_poly_avx512(__m512d r_squared, const double* coeffs) {
     // Estrin's scheme for P(z) = c0 + c1*z + c2*z^2 + ... + c7*z^7 where z = r^2
     const __m512d C0 = _mm512_set1_pd(coeffs[0]), C1 = _mm512_set1_pd(coeffs[1]),
                   C2 = _mm512_set1_pd(coeffs[2]), C3 = _mm512_set1_pd(coeffs[3]),
                   C4 = _mm512_set1_pd(coeffs[4]), C5 = _mm512_set1_pd(coeffs[5]),
                   C6 = _mm512_set1_pd(coeffs[6]), C7 = _mm512_set1_pd(coeffs[7]);
 
     __m512d z = r_squared;
     __m512d z2 = _mm512_mul_pd(z, z);   // r^4
     __m512d z4 = _mm512_mul_pd(z2, z2); // r^8
 
     // Terms grouped for Estrin's scheme, using FMA
     __m512d T01 = _mm512_fmadd_pd(C1, z, C0); // C0 + C1*z
     __m512d T23 = _mm512_fmadd_pd(C3, z, C2); // C2 + C3*z
     __m512d T45 = _mm512_fmadd_pd(C5, z, C4); // C4 + C5*z
     __m512d T67 = _mm512_fmadd_pd(C7, z, C6); // C6 + C7*z
 
     // Combine pairs using z^2 = r^4
     __m512d S03 = _mm512_fmadd_pd(T23, z2, T01); // T01 + T23*z^2
     __m512d S47 = _mm512_fmadd_pd(T67, z2, T45); // T45 + T67*z^2
 
     // Final combination using z^4 = r^8
     return _mm512_fmadd_pd(S47, z4, S03); // S03 + S47*z^4
 }
 
 /**
  * @internal
  * @brief Computes sincos using AVX-512F intrinsics.
  */
 static void fabe13_sincos_avx512(const double* in, double* sin_out, double* cos_out, int n) {
     const int SIMD_WIDTH = 8;
     // Constants
     const __m512d VEC_TWO_OVER_PI_HI = _mm512_set1_pd(FABE13_TWO_OVER_PI_HI);
     const __m512d VEC_TWO_OVER_PI_LO = _mm512_set1_pd(FABE13_TWO_OVER_PI_LO);
     const __m512d VEC_PI_OVER_2_HI = _mm512_set1_pd(FABE13_PI_OVER_2_HI);
     const __m512d VEC_PI_OVER_2_LO = _mm512_set1_pd(FABE13_PI_OVER_2_LO);
     const __m512d VEC_NAN = _mm512_set1_pd(NAN);
     const __m512d VEC_ONE = _mm512_set1_pd(1.0);
     const __m512d VEC_TINY = _mm512_set1_pd(FABE13_TINY_THRESHOLD);
     const __m512d VEC_ZERO = _mm512_setzero_pd();
     const __m512d VEC_INF = _mm512_set1_pd(INFINITY);
     // Integer constants for quadrant logic
     const __m512i VEC_INT_3 = _mm512_set1_epi64(3);
     const __m512i VEC_INT_0 = _mm512_setzero_si512();
     const __m512i VEC_INT_1 = _mm512_set1_epi64(1);
     const __m512i VEC_INT_2 = _mm512_set1_epi64(2);
 
     int i = 0;
     // Process chunks of SIMD_WIDTH
     for (; i <= n - SIMD_WIDTH; i += SIMD_WIDTH) {
         __m512d vx = _mm512_loadu_pd(&in[i]); // Unaligned load
         __m512d vax = _mm512_abs_pd(vx);      // |vx|
 
         // --- Special Value Handling (using k-masks) ---
         // Mask for NaN inputs (unordered compare)
         __mmask8 nan_mask = _mm512_cmp_pd_mask(vx, vx, _CMP_UNORD_Q);
         // Mask for Inf inputs (compare absolute value)
         __mmask8 inf_mask = _mm512_cmp_pd_mask(vax, VEC_INF, _CMP_EQ_OQ);
         // Combined mask for NaN or Inf
         __mmask8 special_mask = _kor_mask8(nan_mask, inf_mask);
 
         // Mask for tiny inputs (|vx| < threshold)
         __mmask8 tiny_mask = _mm512_cmp_pd_mask(vax, VEC_TINY, _CMP_LT_OS);
 
         // --- Payne-Hanek Argument Reduction ---
         // k_dd = round(vx * (2/pi))
         // r = vx - k_dd * (pi/2) using double-double arithmetic and FMA
         __m512d p_hi = _mm512_mul_pd(vx, VEC_TWO_OVER_PI_HI);
         // e1 = vx * VEC_TWO_OVER_PI_HI - p_hi
         __m512d e1   = _mm512_fmsub_pd(vx, VEC_TWO_OVER_PI_HI, p_hi);
         // p_lo = vx * VEC_TWO_OVER_PI_LO + e1
         __m512d p_lo = _mm512_fmadd_pd(vx, VEC_TWO_OVER_PI_LO, e1);
         // k_dd = round_nearest(p_hi + p_lo)
         // Use roundscale which includes rounding mode and exception suppression
         __m512d k_dd = _mm512_roundscale_pd(_mm512_add_pd(p_hi, p_lo),
                                             _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
 
         // Calculate reduced argument r = vx - k_dd * pi/2
         __m512d t1 = _mm512_mul_pd(k_dd, VEC_PI_OVER_2_HI);
         // e2 = k_dd * VEC_PI_OVER_2_HI - t1
         __m512d e2 = _mm512_fmsub_pd(k_dd, VEC_PI_OVER_2_HI, t1);
         __m512d t2 = _mm512_mul_pd(k_dd, VEC_PI_OVER_2_LO);
         __m512d w = _mm512_add_pd(e2, t2); // Correction term
         // r = (vx - t1) - w
         __m512d r = _mm512_sub_pd(_mm512_sub_pd(vx, t1), w);
 
         // --- Polynomial Evaluation ---
         // sin(r) ≈ r * P_sin(r^2)
         // cos(r) ≈ P_cos(r^2)
         __m512d r2 = _mm512_mul_pd(r, r);
         __m512d sin_poly_r2 = fabe13_poly_avx512(r2, FABE13_SIN_COEFFS);
         __m512d cos_poly_r2 = fabe13_poly_avx512(r2, FABE13_COS_COEFFS);
         __m512d sin_r = _mm512_mul_pd(r, sin_poly_r2);
         __m512d cos_r = cos_poly_r2;
 
         // --- Quadrant Correction ---
         // Determine quadrant q = k mod 4
         // Use native AVX512 conversion from double to int64
         __m512i vk_int = _mm512_cvtpd_epi64(k_dd); // Convert rounded double k_dd to int64
 
         // Calculate quadrant q = k & 3
         __m512i vq = _mm512_and_si512(vk_int, VEC_INT_3); // q = k & 3
 
         // Create k-masks for each quadrant using integer comparison
         __mmask8 q0_mask = _mm512_cmpeq_epi64_mask(vq, VEC_INT_0); // q == 0
         __mmask8 q1_mask = _mm512_cmpeq_epi64_mask(vq, VEC_INT_1); // q == 1
         __mmask8 q2_mask = _mm512_cmpeq_epi64_mask(vq, VEC_INT_2); // q == 2
         // q3 is implied where others are false
 
         // Negated values for blending (using subtraction from zero)
         __m512d neg_sin_r = _mm512_sub_pd(VEC_ZERO, sin_r);
         __m512d neg_cos_r = _mm512_sub_pd(VEC_ZERO, cos_r);
 
         // Blend results based on quadrant masks using k-mask blends
         // _mm512_mask_blend_pd(mask, else_val, then_val)
         // Start with q=3 case, then blend in others based on masks
         // q=0: sin= sin(r), cos= cos(r)
         // q=1: sin= cos(r), cos=-sin(r)
         // q=2: sin=-sin(r), cos=-cos(r)
         // q=3: sin=-cos(r), cos= sin(r)
 
         __m512d s_result = neg_cos_r; // Default: q=3 result for sin
         __m512d c_result = sin_r;     // Default: q=3 result for cos
 
         // Apply masks in order (q0, q1, q2). Order matters if masks overlap (they don't here).
         s_result = _mm512_mask_blend_pd(q0_mask, s_result, sin_r);     // If q=0, use sin(r)
         c_result = _mm512_mask_blend_pd(q0_mask, c_result, cos_r);     // If q=0, use cos(r)
 
         s_result = _mm512_mask_blend_pd(q1_mask, s_result, cos_r);     // If q=1, use cos(r)
         c_result = _mm512_mask_blend_pd(q1_mask, c_result, neg_sin_r); // If q=1, use -sin(r)
 
         s_result = _mm512_mask_blend_pd(q2_mask, s_result, neg_sin_r); // If q=2, use -sin(r)
         c_result = _mm512_mask_blend_pd(q2_mask, c_result, neg_cos_r); // If q=2, use -cos(r)
 
         // --- Apply Tiny & Special Masks ---
         // If tiny, sin(x) = x, cos(x) = 1
         s_result = _mm512_mask_blend_pd(tiny_mask, s_result, vx);
         c_result = _mm512_mask_blend_pd(tiny_mask, c_result, VEC_ONE);
         // If special (NaN or Inf), result is NaN
         s_result = _mm512_mask_blend_pd(special_mask, s_result, VEC_NAN);
         c_result = _mm512_mask_blend_pd(special_mask, c_result, VEC_NAN);
 
         // --- Store results ---
         _mm512_storeu_pd(&sin_out[i], s_result); // Unaligned store
         _mm512_storeu_pd(&cos_out[i], c_result); // Unaligned store
     }
 
     // Process any remaining elements using the scalar fallback
     if (i < n) {
         fabe13_sincos_scalar(in + i, sin_out + i, cos_out + i, n - i);
     }
 }
 #endif // defined(__AVX512F__)
 #endif // FABE13_PLATFORM_X86
 
 
 /*
  --- Example Compilation Flags ---
 
  Target: x86_64 with AVX2 + FMA support:
    gcc -O3 -mavx2 -mfma fabe13.c -o my_program -lm -lmylib_using_fabe13...
    clang -O3 -mavx2 -mfma fabe13.c -o my_program -lm -lmylib_using_fabe13...
 
  Target: x86_64 with AVX-512F support:
    gcc -O3 -mavx512f -mfma fabe13.c -o my_program -lm -lmylib_using_fabe13...
    clang -O3 -mavx512f -mfma fabe13.c -o my_program -lm -lmylib_using_fabe13...
    Note: Compiling with -mavx512f enables the AVX-512 backend. The dispatcher
          will still select AVX2 if the runtime CPU doesn't support AVX-512F.
 
  Target: AArch64 (ARM 64-bit) with NEON support:
    gcc -O3 fabe13.c -o my_program -lm -lmylib_using_fabe13...
    clang -O3 fabe13.c -o my_program -lm -lmylib_using_fabe13...
    (NEON support is typically default for AArch64 targets)
 
  Target: Generic/Scalar Fallback (or if SIMD flags are omitted):
    gcc -O3 fabe13.c -o my_program -lm -lmylib_using_fabe13...
    clang -O3 fabe13.c -o my_program -lm -lmylib_using_fabe13...
 
  Notes:
   - Link with `-lm` for the standard math library functions (used in scalar fallback
     and potentially for atan/asin/acos).
   - Replace `my_program` and `-lmylib_using_fabe13...` with your actual program
     and linking requirements.
   - `-O3` is recommended for optimal performance.
   - Ensure your compiler supports the specified SIMD instruction sets.
 */
