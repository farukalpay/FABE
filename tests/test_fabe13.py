/**
 * @file fabe13_example.c
 * @brief Example usage of the FABE13 trigonometric library.
 *
 * Demonstrates calling scalar and array functions from fabe13.
 * Compares results with standard math library functions.
 */

 #include <stdio.h>
 #include <stdlib.h>
 #include <math.h>
 #include <float.h>
 #include <string.h>
 
 // Include public API
 #include "../fabe13/fabe13.h"
 
 // --- Helper for Alignment ---
 #ifndef FABE13_ALIGNMENT
 #define FABE13_ALIGNMENT 64
 #endif
 
 void* example_aligned_malloc(size_t size, size_t alignment) {
     if (alignment == 0 || (alignment & (alignment - 1)) != 0) alignment = 64;
     if (alignment < sizeof(void*)) alignment = sizeof(void*);
     size_t aligned_size = ((size + alignment - 1) / alignment) * alignment;
 #if defined(_ISOC11_SOURCE) || (defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L)
     return aligned_alloc(alignment, aligned_size);
 #else
     void* ptr = NULL;
     if (posix_memalign(&ptr, alignment, size) != 0) return NULL;
     return ptr;
 #endif
 }
 
 void example_aligned_free(void* ptr) {
     free(ptr);
 }
 
 // --- Main Example ---
 int main() {
     printf("--- FABE13 Library Usage Example ---\n\n");
 
     printf("Checking FABE13 active implementation...\n");
     const char* impl_name = fabe13_get_active_implementation_name();
     int simd_width = fabe13_get_active_simd_width();
     printf("Active Implementation: %s (SIMD Width: %d)\n\n", impl_name, simd_width);
 
     printf("--- Scalar Function Examples ---\n");
     double inputs[] = { 0.0, 1.0, -1.0, M_PI / 6.0, M_PI / 4.0, M_PI / 2.0, M_PI,
                         3.0 * M_PI / 2.0, 2.0 * M_PI, 1000.0, -1000.0, 1e-10,
                         0.5, -0.5, 2.0, -2.0, INFINITY, -INFINITY, NAN };
     int num_inputs = sizeof(inputs) / sizeof(inputs[0]);
 
     for (int i = 0; i < num_inputs; ++i) {
         double x = inputs[i];
         printf("\nInput x = %.6g\n", x);
         double s_fabe = fabe13_sin(x);
         double c_fabe = fabe13_cos(x);
         double s_ref = sin(x);
         double c_ref = cos(x);
         printf("  fabe13_sin: %+.15e (ref: %+.15e, diff: %.3e)\n", s_fabe, s_ref, fabs(s_fabe - s_ref));
         printf("  fabe13_cos: %+.15e (ref: %+.15e, diff: %.3e)\n", c_fabe, c_ref, fabs(c_fabe - c_ref));
 
         double sinc_fabe = fabe13_sinc(x);
         double sinc_ref = (fabs(x) < 1e-9) ? 1.0 : sin(x) / x;
         printf("  fabe13_sinc:%+.15e (ref: %+.15e, diff: %.3e)\n", sinc_fabe, sinc_ref, fabs(sinc_fabe - sinc_ref));
 
         double tan_fabe = fabe13_tan(x), tan_ref = tan(x);
         if (isnan(tan_fabe) && isnan(tan_ref))
             printf("  fabe13_tan: NAN (ref: NAN)\n");
         else
             printf("  fabe13_tan: %+.15e (ref: %+.15e, diff: %.3e)\n", tan_fabe, tan_ref, fabs(tan_fabe - tan_ref));
 
         double cot_fabe = fabe13_cot(x), cot_ref = (sin(x) == 0.0) ? NAN : cos(x) / sin(x);
         if (isnan(cot_fabe) && isnan(cot_ref))
             printf("  fabe13_cot: NAN (ref: NAN)\n");
         else
             printf("  fabe13_cot: %+.15e (ref: %+.15e, diff: %.3e)\n", cot_fabe, cot_ref, fabs(cot_fabe - cot_ref));
 
         double atan_fabe = fabe13_atan(x), atan_ref = atan(x);
         if (isnan(atan_fabe) && isnan(atan_ref))
             printf("  fabe13_atan: NAN (ref: NAN)\n");
         else
             printf("  fabe13_atan:%+.15e (ref: %+.15e, diff: %.3e)\n", atan_fabe, atan_ref, fabs(atan_fabe - atan_ref));
 
         if (x >= -1.0 && x <= 1.0) {
             double asin_fabe = fabe13_asin(x), asin_ref = asin(x);
             printf("  fabe13_asin:%+.15e (ref: %+.15e, diff: %.3e)\n", asin_fabe, asin_ref, fabs(asin_fabe - asin_ref));
             double acos_fabe = fabe13_acos(x), acos_ref = acos(x);
             printf("  fabe13_acos:%+.15e (ref: %+.15e, diff: %.3e)\n", acos_fabe, acos_ref, fabs(acos_fabe - acos_ref));
         } else {
             printf("  fabe13_asin: Input out of range [-1, 1]\n");
             printf("  fabe13_acos: Input out of range [-1, 1]\n");
         }
     }
 
     printf("\n--- Array Function Example (fabe13_sincos) ---\n");
     int n_array = 100;
     double* array_in = (double*)example_aligned_malloc(n_array * sizeof(double), FABE13_ALIGNMENT);
     double* array_sin = (double*)example_aligned_malloc(n_array * sizeof(double), FABE13_ALIGNMENT);
     double* array_cos = (double*)example_aligned_malloc(n_array * sizeof(double), FABE13_ALIGNMENT);
 
     if (!array_in || !array_sin || !array_cos) {
         fprintf(stderr, "ERROR: Failed to allocate memory for array example.\n");
         example_aligned_free(array_in);
         example_aligned_free(array_sin);
         example_aligned_free(array_cos);
         return 1;
     }
 
     for (int i = 0; i < n_array; ++i) array_in[i] = (double)i * 0.1;
 
     printf("Calling fabe13_sincos...\n");
     fabe13_sincos(array_in, array_sin, array_cos, n_array);
     printf("Call complete.\n");
 
     printf("Sample results:\n");
     for (int i = 0; i < 5 && i < n_array; ++i)
         printf("  in[%d]=%.3f -> sin=%.15e, cos=%.15e\n", i, array_in[i], array_sin[i], array_cos[i]);
 
     if (n_array > 5) {
         int i = n_array / 2;
         printf("  in[%d]=%.3f -> sin=%.15e, cos=%.15e\n", i, array_in[i], array_sin[i], array_cos[i]);
         i = n_array - 1;
         printf("  in[%d]=%.3f -> sin=%.15e, cos=%.15e\n", i, array_in[i], array_sin[i], array_cos[i]);
     }
 
     example_aligned_free(array_in);
     example_aligned_free(array_sin);
     example_aligned_free(array_cos);
 
     printf("\n--- Example Complete ---\n");
     return 0;
 }
