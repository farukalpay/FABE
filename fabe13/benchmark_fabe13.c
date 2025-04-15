#ifndef ENABLE_FABE13_BENCHMARK
#define ENABLE_FABE13_BENCHMARK 0
#endif

#if ENABLE_FABE13_BENCHMARK

/**
 * @file benchmark_fabe13.c
 * @brief Benchmark executable for the FABE13 library.
 *
 * This file contains a main() function that benchmarks FABE13's sincos
 * functions against the standard libm. It does NOT redefine the FABE13
 * library functions; it only calls them.
 *
 * To enable this benchmark, compile with:
 *   gcc -O3 -I. -DENABLE_FABE13_BENCHMARK=1 fabe13.c benchmark_fabe13.c -o fabe13_benchmark -lm
 * (On macOS, omit -lrt since it doesn't exist.)
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <math.h>
#include <errno.h>
#include <string.h>

// If you’re on Linux, you might use <unistd.h> for _SC_PAGESIZE
// #include <unistd.h> 

// Include your FABE13 library header:
#include "fabe13.h"

// On macOS, librt is unavailable, so we don't include <sys/time.h> or -lrt

// Optional alignment for benchmark buffers
#ifndef FABE13_ALIGNMENT
#define FABE13_ALIGNMENT 64
#endif

// A small helper for aligned allocation (posix_memalign).
// For Windows, you'd need a different approach (e.g. _aligned_malloc).
static void* aligned_malloc(size_t size, size_t alignment) {
    void* ptr = NULL;
    if (alignment < sizeof(void*) || (alignment & (alignment - 1)) != 0) {
        // Fallback alignment
        alignment = 64;
    }
    if (posix_memalign(&ptr, alignment, size) != 0) {
        perror("posix_memalign failed");
        return NULL;
    }
    return ptr;
}

static void aligned_free(void* ptr) {
    free(ptr);
}

/**
 * @brief Reference sin/cos using standard libm, for comparison.
 */
static void benchmark_libm_array(const double *x, double *sin_out, double *cos_out, int length) {
    for (int i = 0; i < length; i++) {
        sin_out[i] = sin(x[i]);
        cos_out[i] = cos(x[i]);
    }
}

/**
 * @brief Compute time difference in seconds (double).
 */
static double timespec_diff(struct timespec start, struct timespec end) {
    return (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
}

int main(void) {
    printf("============================================\n");
    printf("= FABE13 Benchmark (ENABLE_FABE13_BENCHMARK)\n");
    printf("============================================\n");

    // Force library to initialize its dispatcher
    const char* impl_name = fabe13_get_active_implementation_name();
    int simd_width = fabe13_get_active_simd_width();
    printf("Selected Implementation: %s (SIMD Width: %d)\n\n", impl_name, simd_width);

    // Simple example: compare performance on a large array of angles
    const int N = 1000000000;  // 1 billion
    double* angles     = (double*)aligned_malloc(N * sizeof(double), FABE13_ALIGNMENT);
    double* sin_fabe13 = (double*)aligned_malloc(N * sizeof(double), FABE13_ALIGNMENT);
    double* cos_fabe13 = (double*)aligned_malloc(N * sizeof(double), FABE13_ALIGNMENT);
    double* sin_libm   = (double*)aligned_malloc(N * sizeof(double), FABE13_ALIGNMENT);
    double* cos_libm   = (double*)aligned_malloc(N * sizeof(double), FABE13_ALIGNMENT);

    if (!angles || !sin_fabe13 || !cos_fabe13 || !sin_libm || !cos_libm) {
        fprintf(stderr, "Allocation failed.\n");
        return 1;
    }

    // Fill angles with some random or linearly increasing data
    for (int i = 0; i < N; i++) {
        // from -100000.0 to +100000.0
        angles[i] = ((double) i / (N - 1)) * 200000.0 - 100000.0;
    }

    struct timespec start, end;

    // --- Benchmark FABE13 ---
    clock_gettime(CLOCK_MONOTONIC, &start);
    fabe13_sincos(angles, sin_fabe13, cos_fabe13, N);
    clock_gettime(CLOCK_MONOTONIC, &end);
    double sec_fabe13 = timespec_diff(start, end);

    // --- Benchmark standard libm ---
    clock_gettime(CLOCK_MONOTONIC, &start);
    benchmark_libm_array(angles, sin_libm, cos_libm, N);
    clock_gettime(CLOCK_MONOTONIC, &end);
    double sec_libm = timespec_diff(start, end);

    printf("FABE13 time for %d sincos calls: %.3f seconds\n", N, sec_fabe13);
    printf("libm   time for %d sincos calls: %.3f seconds\n\n", N, sec_libm);

    // Quick accuracy check
    double max_sin_diff = 0.0;
    double max_cos_diff = 0.0;
    for (int i = 0; i < N; i++) {
        double diff_sin = fabs(sin_fabe13[i] - sin_libm[i]);
        double diff_cos = fabs(cos_fabe13[i] - cos_libm[i]);
        if (diff_sin > max_sin_diff) max_sin_diff = diff_sin;
        if (diff_cos > max_cos_diff) max_cos_diff = diff_cos;
    }

    printf("Max difference vs. libm: sin=%.4e, cos=%.4e\n", max_sin_diff, max_cos_diff);

    aligned_free(angles);
    aligned_free(sin_fabe13);
    aligned_free(cos_fabe13);
    aligned_free(sin_libm);
    aligned_free(cos_libm);

    return 0;
}

#endif // ENABLE_FABE13_BENCHMARK
