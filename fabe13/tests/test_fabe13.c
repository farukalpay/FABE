/**
 * @file fabe13_example.c
 * @brief Example usage of the FABE13 trigonometric library.
 *
 * Demonstrates calling scalar and array functions from fabe13.
 * Compares results with standard math library functions.
 * Assumes fabe13.h and the compiled library are accessible.
 */

// Define POSIX source level *before* including headers to ensure visibility
// of functions like posix_memalign.
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include <stdio.h>
#include <stdlib.h> // For malloc, free, posix_memalign
#include <math.h>   // For M_PI, sin, cos, etc., NAN, INFINITY, fabs, isnan
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include <stdbool.h> // For bool
#include <string.h>  // For strerror
#include <errno.h>   // For errno

// Include public API after system headers
#include "../fabe13/fabe13.h" // Adjust path relative to this example file if needed

// --- Helper for Alignment ---

#ifdef _MSC_VER
#include <malloc.h> // For _aligned_malloc, _aligned_free
#endif

/**
 * @brief Portable aligned memory allocation for the example.
 * Attempts POSIX posix_memalign, MSVC _aligned_malloc, or falls back
 * to standard malloc (alignment not guaranteed in fallback).
 * Uses the alignment specified by FABE13_ALIGNMENT from fabe13.h.
 */
void* example_aligned_malloc(size_t size, size_t alignment) {
    // Ensure alignment is a power of two and meets minimum requirements
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) alignment = 64; // Default/Fix
    if (alignment < sizeof(void*)) alignment = sizeof(void*);

    void* ptr = NULL;
    errno = 0; // Clear errno before allocation attempt

#if defined(_POSIX_C_SOURCE) && _POSIX_C_SOURCE >= 200112L || defined(__APPLE__)
    // Use POSIX posix_memalign (preferred on Linux/macOS)
    if (posix_memalign(&ptr, alignment, size) != 0) {
        // errno is set by posix_memalign on failure
        ptr = NULL; // Ensure ptr is NULL on failure
        fprintf(stderr, "WARNING: posix_memalign failed: %s\n", strerror(errno));
    }
#elif defined(_MSC_VER)
    // Use MSVC specific aligned allocation
    ptr = _aligned_malloc(size, alignment);
    if (!ptr) {
        errno = ENOMEM; // Set errno for consistency
        fprintf(stderr, "WARNING: _aligned_malloc failed\n");
    }
#else
    // Fallback: Standard malloc, alignment not guaranteed
    #warning "Aligned allocation using standard malloc, alignment not guaranteed."
    ptr = malloc(size);
    if (!ptr) {
        errno = ENOMEM;
        fprintf(stderr, "WARNING: standard malloc failed\n");
    }
#endif

    if (!ptr) {
         fprintf(stderr, "ERROR: Failed to allocate %zu bytes aligned to %zu.\n", size, alignment);
    }
    return ptr;
}

/**
 * @brief Portable free for memory allocated with example_aligned_malloc.
 */
void example_aligned_free(void* ptr) {
    if (!ptr) return;
#if defined(_MSC_VER)
    _aligned_free(ptr);
#else
    // free works for posix_memalign and malloc
    free(ptr);
#endif
}

// --- Main Example ---
int main() {
    printf("--- FABE13 Library Usage Example ---\n\n");

    printf("Checking FABE13 active implementation...\n");
    // Library initializes automatically via constructor (if supported)
    // Getters retrieve the initialized state.
    const char* impl_name = fabe13_get_active_implementation_name();
    int simd_width = fabe13_get_active_simd_width();
    printf("Active Implementation: %s (SIMD Width: %d)\n", impl_name, simd_width);
    printf("Required Alignment for fabe13_sincos: %d bytes\n\n", FABE13_ALIGNMENT);


    printf("--- Scalar Function Examples ---\n");
    // Scalar functions use the custom scalar core directly.
    double inputs[] = {
        0.0, 1.0, -1.0, M_PI / 6.0, M_PI / 4.0, M_PI / 2.0, M_PI,
        3.0 * M_PI / 2.0, 2.0 * M_PI, 1000.0, -1000.0, 1e-10, -1e-10,
        0.5, -0.5, 2.0, -2.0,
        INFINITY, -INFINITY, NAN
    };
    int num_inputs = sizeof(inputs) / sizeof(inputs[0]);

    for (int i = 0; i < num_inputs; ++i) {
        double x = inputs[i];
        printf("\nInput x = %.6g\n", x);

        // --- sin/cos ---
        double s_fabe = fabe13_sin(x);
        double c_fabe = fabe13_cos(x);
        double s_ref = sin(x);
        double c_ref = cos(x);
        bool s_nan_match = isnan(s_fabe) && isnan(s_ref);
        bool c_nan_match = isnan(c_fabe) && isnan(c_ref);
        printf("  fabe13_sin: %+.15e (ref: %+.15e, diff: %.3e)\n",
               s_fabe, s_ref, s_nan_match ? 0.0 : fabs(s_fabe - s_ref));
        printf("  fabe13_cos: %+.15e (ref: %+.15e, diff: %.3e)\n",
               c_fabe, c_ref, c_nan_match ? 0.0 : fabs(c_fabe - c_ref));

        // --- sinc ---
        double sinc_fabe = fabe13_sinc(x);
        double sinc_ref = (fabs(x) < 1e-12) ? 1.0 : (sin(x) / x);
        bool sinc_nan_match = isnan(sinc_fabe) && isnan(sinc_ref);
        printf("  fabe13_sinc:%+.15e (ref: %+.15e, diff: %.3e)\n",
               sinc_fabe, sinc_ref, sinc_nan_match ? 0.0 : fabs(sinc_fabe - sinc_ref));

        // --- tan ---
        double tan_fabe = fabe13_tan(x);
        double tan_ref = tan(x);
        bool tan_nan_match = isnan(tan_fabe) && isnan(tan_ref);
        // Avoid large diff for Inf/-Inf vs large number near pole
        double tan_diff = (isinf(tan_ref) || isinf(tan_fabe)) ? 0.0 : fabs(tan_fabe - tan_ref);
        printf("  fabe13_tan: %+.15e (ref: %+.15e, diff: %.3e)\n",
               tan_fabe, tan_ref, tan_nan_match ? 0.0 : tan_diff);

        // --- cot ---
        double cot_fabe = fabe13_cot(x);
        double cot_ref_s = sin(x);
        double cot_ref = (cot_ref_s == 0.0) ? NAN : (cos(x) / cot_ref_s);
        bool cot_nan_match = isnan(cot_fabe) && isnan(cot_ref);
        double cot_diff = (isinf(cot_ref) || isinf(cot_fabe)) ? 0.0 : fabs(cot_fabe - cot_ref);
        printf("  fabe13_cot: %+.15e (ref: %+.15e, diff: %.3e)\n",
               cot_fabe, cot_ref, cot_nan_match ? 0.0 : cot_diff);

        // --- atan --- (Delegates to libm)
        double atan_fabe = fabe13_atan(x);
        double atan_ref = atan(x);
        bool atan_nan_match = isnan(atan_fabe) && isnan(atan_ref);
        printf("  fabe13_atan:%+.15e (ref: %+.15e, diff: %.3e)\n",
               atan_fabe, atan_ref, atan_nan_match ? 0.0 : fabs(atan_fabe - atan_ref));

        // --- asin/acos --- (Delegate to libm)
        if (x >= -1.0 && x <= 1.0) {
            double asin_fabe = fabe13_asin(x);
            double asin_ref = asin(x);
            printf("  fabe13_asin:%+.15e (ref: %+.15e, diff: %.3e)\n",
                   asin_fabe, asin_ref, fabs(asin_fabe - asin_ref));
            double acos_fabe = fabe13_acos(x);
            double acos_ref = acos(x);
            printf("  fabe13_acos:%+.15e (ref: %+.15e, diff: %.3e)\n",
                   acos_fabe, acos_ref, fabs(acos_fabe - acos_ref));
        } else {
            double asin_fabe = fabe13_asin(x); // Should return NaN
            double acos_fabe = fabe13_acos(x); // Should return NaN
            printf("  fabe13_asin: %s (Input out of range [-1, 1])\n", isnan(asin_fabe) ? "NaN (Correct)" : "ERROR! Expected NaN");
            printf("  fabe13_acos: %s (Input out of range [-1, 1])\n", isnan(acos_fabe) ? "NaN (Correct)" : "ERROR! Expected NaN");
        }
    }

    printf("\n--- Array Function Example (fabe13_sincos) ---\n");
    int n_array = 100; // Small number for example output
    printf("Allocating aligned memory for %d elements (alignment: %d bytes)...\n", n_array, FABE13_ALIGNMENT);
    // Use the aligned allocator
    double* array_in = (double*)example_aligned_malloc(n_array * sizeof(double), FABE13_ALIGNMENT);
    double* array_sin = (double*)example_aligned_malloc(n_array * sizeof(double), FABE13_ALIGNMENT);
    double* array_cos = (double*)example_aligned_malloc(n_array * sizeof(double), FABE13_ALIGNMENT);

    if (!array_in || !array_sin || !array_cos) {
        fprintf(stderr, "ERROR: Failed to allocate memory for array example.\n");
        // Free any that might have succeeded
        example_aligned_free(array_in);
        example_aligned_free(array_sin);
        example_aligned_free(array_cos);
        return 1;
    }
    printf("Memory allocated.\n");

    printf("Generating %d inputs for array test...\n", n_array);
    for (int i = 0; i < n_array; ++i) {
        array_in[i] = (double)i * 0.1 - 5.0; // Range around zero: -5.0 to +4.9
    }

    printf("Calling fabe13_sincos (expects aligned pointers)...\n");
    // Pass the aligned pointers to the library function
    // Using __restrict__ here is optional but shows intent if using GCC/Clang
    #if defined(__GNUC__) || defined(__clang__)
        fabe13_sincos((const double* __restrict__)array_in,
                      (double* __restrict__)array_sin,
                      (double* __restrict__)array_cos,
                      n_array);
    #else
        fabe13_sincos(array_in, array_sin, array_cos, n_array);
    #endif
    printf("Call complete.\n");

    printf("Sample results (comparing first/last few to libm):\n");
    int check_count = (n_array < 5) ? n_array : 5;
    for (int i = 0; i < check_count; ++i) {
        double s_ref = sin(array_in[i]);
        double c_ref = cos(array_in[i]);
        printf("  in[%d]=%.3f -> sin=%.15e (diff=%.2e), cos=%.15e (diff=%.2e)\n",
               i, array_in[i],
               array_sin[i], fabs(array_sin[i] - s_ref),
               array_cos[i], fabs(array_cos[i] - c_ref));
    }

    if (n_array > check_count) {
         printf("  ...\n");
         // Check last element
         int i = n_array - 1;
         double s_ref = sin(array_in[i]);
         double c_ref = cos(array_in[i]);
         printf("  in[%d]=%.3f -> sin=%.15e (diff=%.2e), cos=%.15e (diff=%.2e)\n",
               i, array_in[i],
               array_sin[i], fabs(array_sin[i] - s_ref),
               array_cos[i], fabs(array_cos[i] - c_ref));
    }

    printf("Freeing aligned memory...\n");
    example_aligned_free(array_in);
    example_aligned_free(array_sin);
    example_aligned_free(array_cos);

    printf("\n--- Example Complete ---\n");
    return 0;
}
