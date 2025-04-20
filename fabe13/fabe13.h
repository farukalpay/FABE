/**
 * @file fabe13.h
 * @brief Public API Header for the FABE13 High-Performance Trigonometric Library.
 *
 * Declares functions for computing sin(x), cos(x), and related trigonometric
 * functions using optimized SIMD implementations with runtime dispatch.
 */

 #ifndef FABE13_H
 #define FABE13_H
 
 #ifdef __cplusplus
 extern "C" {
 #endif
 
 // --- Constants ---
 
 /**
  * @brief Recommended memory alignment (in bytes) for input/output arrays
  *        passed to fabe13_sincos for optimal SIMD performance.
  */
 #define FABE13_ALIGNMENT 64
 
 /**
  * @brief Define FABE13_ENABLE_FAST_FP (e.g., via -DFABE13_ENABLE_FAST_FP)
  *        during compilation to enable setting Flush-To-Zero (FTZ) and
  *        Denormals-Are-Zero (DAZ) modes in the CPU's floating-point unit
  *        (primarily on x86 via MXCSR). This can improve performance when
  *        dealing with subnormal numbers but alters strict IEEE 754 compliance.
  *        Use with caution and verify numerical results. Disabled by default.
  */
 // #define FABE13_ENABLE_FAST_FP // Uncomment or define via compiler flag
 
 // --- Public API Declarations ---
 
 /**
  * @brief Computes sine and cosine for an array of doubles.
  *
  * Uses the best available SIMD implementation detected at library load time
  * or a custom scalar fallback. Dispatches internally to optimized paths based
  * on input size and pointer alignment.
  *
  * @param in Pointer to the input array of angles (in radians).
  *           Performance is best if aligned to FABE13_ALIGNMENT.
  * @param sin_out Pointer to the output array for sine results.
  *                Performance is best if aligned to FABE13_ALIGNMENT.
  * @param cos_out Pointer to the output array for cosine results.
  *                Performance is best if aligned to FABE13_ALIGNMENT.
  * @param n The number of elements in the arrays. Must not exceed INT_MAX.
  */
 void fabe13_sincos(
     const double * in,
     double       * sin_out,
     double       * cos_out,
     int             n
 );
 
 /**
  * @brief Gets the name of the active sincos implementation chosen at load time.
  * @return A constant C-string literal with the name of the active implementation.
  */
 const char* fabe13_get_active_implementation_name(void);
 
 /**
  * @brief Gets the SIMD width of the active sincos implementation.
  * @return The SIMD width (vector size) of the active implementation.
  */
 int fabe13_get_active_simd_width(void);
 
 // --- Scalar API (Unchanged, use custom scalar core) ---
 double fabe13_sin(double x);
 double fabe13_cos(double x);
 double fabe13_sinc(double x);
 double fabe13_tan(double x);
 double fabe13_cot(double x);
 double fabe13_atan(double x); // Uses standard libm
 double fabe13_asin(double x); // Uses standard libm
 double fabe13_acos(double x); // Uses standard libm
 
 
 #ifdef __cplusplus
 } // extern "C"
 #endif
 
 #endif // FABE13_H
