/* --- Optional: Benchmark main() --- */
/*
 * This section provides a benchmark executable if ENABLE_FABE13_BENCHMARK is set to 1.
 * Compile the library (fabe13.c) first, e.g., into libfabe13.a or fabe13.o.
 * Then compile this benchmark code separately, linking against the library.
 *
 * Example Compilation (assuming library object fabe13.o exists):
 *   gcc -O3 -march=native -mfma -funroll-loops -DENABLE_FABE13_BENCHMARK=1 \
 *       benchmark_main.c fabe13.o -o fabe13_benchmark -lm -I/path/to/fabe13_header
 * (Replace benchmark_main.c with the file containing this code if separated)
 * (Replace /path/to/fabe13_header with the directory containing fabe13.h)
 * (Adjust -march, -mfma flags based on the target CPU and library compilation)
 */
#ifndef ENABLE_FABE13_BENCHMARK
#define ENABLE_FABE13_BENCHMARK 0 // Default to disabled
#endif

#if ENABLE_FABE13_BENCHMARK

// --- Includes required for the benchmark ---
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stdbool.h>
#include <errno.h>
#include <string.h>
#include <stdint.h>
#include <limits.h>

// Include the public API header for the library being benchmarked
#include "fabe13.h" // Assumes fabe13.h is in the include path

// --- Platform Specific Includes for Resource Usage ---
#if defined(__unix__) || defined(__APPLE__)
    #include <sys/time.h>     // For timeval struct used by rusage
    #include <sys/resource.h> // For getrusage
    #define CAN_MEASURE_RESOURCE_USAGE 1
#else
    // Windows would require <windows.h>, GetProcessTimes, GetProcessMemoryInfo etc.
    #define CAN_MEASURE_RESOURCE_USAGE 0
    #warning "Resource usage measurement (CPU time, RAM) via getrusage is not available on this platform."
#endif


// --- High-Resolution Timer ---
#if (defined(_POSIX_TIMERS) && _POSIX_TIMERS > 0 && defined(CLOCK_MONOTONIC)) || defined(__APPLE__)
    // <time.h> included above or via sys/time.h
    typedef struct timespec high_res_time_t;
    #define get_high_res_time(t) clock_gettime(CLOCK_MONOTONIC, t)
    static double high_res_time_diff(high_res_time_t start, high_res_time_t end) {
         return (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec) / 1e9;
    }
    #define HIGH_RES_TIMER_USED "clock_gettime(CLOCK_MONOTONIC)"
#else
    #warning "clock_gettime(CLOCK_MONOTONIC) not detected, benchmark timing might be less accurate using clock()."
    // <time.h> included above
    typedef clock_t high_res_time_t;
    #define get_high_res_time(t) (*(t) = clock())
    #define high_res_time_diff(start, end) ((double)((end) - (start)) / CLOCKS_PER_SEC)
    #define HIGH_RES_TIMER_USED "clock()"
#endif

// --- Resource Usage Helpers (POSIX) ---
#if CAN_MEASURE_RESOURCE_USAGE
// Helper to convert timeval to seconds (double)
static double timeval_to_sec(struct timeval tv) {
    return (double)tv.tv_sec + (double)tv.tv_usec / 1e6;
}

// Helper to calculate difference between two timevals
static double timeval_diff_sec(struct timeval start, struct timeval end) {
    return timeval_to_sec(end) - timeval_to_sec(start);
}
#endif // CAN_MEASURE_RESOURCE_USAGE


// --- Aligned Memory Allocation for Benchmark ---
// Uses FABE13_ALIGNMENT defined in fabe13.h

#ifdef _MSC_VER
#include <malloc.h> // For _aligned_malloc, _aligned_free
#endif

static void* benchmark_aligned_malloc(size_t size, size_t alignment) {
    void *ptr = NULL;
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) alignment = 64;
    if (alignment < sizeof(void*)) alignment = sizeof(void*);
    errno = 0;

#if defined(_POSIX_C_SOURCE) && _POSIX_C_SOURCE >= 200112L || defined(__APPLE__)
    if (posix_memalign(&ptr, alignment, size) != 0) {
        ptr = NULL; fprintf(stderr, "WARNING: posix_memalign failed: %s\n", strerror(errno));
    }
#elif defined(_MSC_VER)
    ptr = _aligned_malloc(size, alignment);
    if (!ptr) { errno = ENOMEM; fprintf(stderr, "WARNING: _aligned_malloc failed\n"); }
#else
    #warning "Aligned memory allocation using standard malloc, alignment not guaranteed."
    ptr = malloc(size);
    if (!ptr) { errno = ENOMEM; fprintf(stderr, "WARNING: standard malloc failed\n"); }
#endif
    if (!ptr) { fprintf(stderr, "ERROR: Failed to allocate %zu bytes aligned to %zu.\n", size, alignment); }
    return ptr;
}

static void benchmark_aligned_free(void *ptr) {
    if (!ptr) return;
#if defined(_MSC_VER)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

/** @brief Reference sin/cos using standard libm. */
static void benchmark_libm_array(const double *x, double *sin_out, double *cos_out, long long length) {
    for (long long i = 0; i < length; i++) {
        sin_out[i] = sin(x[i]);
        cos_out[i] = cos(x[i]);
    }
}

/** @brief Helper to format large numbers. */
static char* format_lld(long long n) {
    static char buf[32]; sprintf(buf, "%lld", n); return buf;
}

// --- Benchmark Main Function ---
int main(void) {
    printf("============================================\n");
    printf("= FABE13 Benchmark (vs. Standard Libm)\n");
    printf("============================================\n");

    const char* impl_name = fabe13_get_active_implementation_name();
    int simd_width = fabe13_get_active_simd_width();
    printf("FABE13 Active Implementation: %s (SIMD Width: %d)\n", impl_name, simd_width);
    printf("Benchmark Alignment: %d bytes\n", FABE13_ALIGNMENT);
    printf("Using Timer: %s\n", HIGH_RES_TIMER_USED);
    #if CAN_MEASURE_RESOURCE_USAGE
        printf("Resource Usage Measurement: Enabled (getrusage)\n");
    #else
        printf("Resource Usage Measurement: Disabled (Platform not supported)\n");
    #endif
    printf("NOTE: Benchmark uses aligned memory allocation.\n");
    printf("NOTE: GPU usage is not measured (CPU-only benchmark).\n");
    printf("\n");
    printf("Benchmarking sincos(x) for various array sizes (N):\n");
    printf("---------------------------------------------------------------------------------------------------------------------------------------\n");
    printf("%-15s | %-15s | %-15s | %-15s | %-15s | %-10s | %-15s | %-15s\n",
           "N", "FABE13 (sec)", "Libm (sec)", "FABE13 (M/sec)", "Libm (M/sec)", "Speedup", "FABE13 CPU(%)", "Libm CPU(%)");
    printf("---------------------------------------------------------------------------------------------------------------------------------------\n");

    const long long max_n = 1000000000LL;
    const double max_ram_gb = 64.0;
    const int num_reps = 5;

    srand((unsigned int)time(NULL));

    #if CAN_MEASURE_RESOURCE_USAGE
        struct rusage usage_start, usage_end;
        long peak_rss_kb_fabe13 = 0;
        long peak_rss_kb_libm = 0;
    #endif

    for (long long n = 10; n <= max_n; n *= 10) {
        size_t bytes_per_array = n * sizeof(double);
        size_t total_bytes = 5 * bytes_per_array;
        double total_gb = (double)total_bytes / (1024.0 * 1024.0 * 1024.0);

        printf("%-15s |", format_lld(n));

        if (total_gb > max_ram_gb) { printf(" Skipped (Requires > %.1f GB RAM)\n", max_ram_gb); continue; }
        if (n > INT_MAX) { printf(" Skipped (N > INT_MAX)\n"); continue; }

        double* angles     = (double*)benchmark_aligned_malloc(bytes_per_array, FABE13_ALIGNMENT);
        double* sin_fabe13 = (double*)benchmark_aligned_malloc(bytes_per_array, FABE13_ALIGNMENT);
        double* cos_fabe13 = (double*)benchmark_aligned_malloc(bytes_per_array, FABE13_ALIGNMENT);
        double* sin_libm   = (double*)benchmark_aligned_malloc(bytes_per_array, FABE13_ALIGNMENT);
        double* cos_libm   = (double*)benchmark_aligned_malloc(bytes_per_array, FABE13_ALIGNMENT);

        if (!angles || !sin_fabe13 || !cos_fabe13 || !sin_libm || !cos_libm) {
            printf(" Allocation Failed (Req: %.1f GB)\n", total_gb);
            benchmark_aligned_free(angles); benchmark_aligned_free(sin_fabe13); benchmark_aligned_free(cos_fabe13);
            benchmark_aligned_free(sin_libm); benchmark_aligned_free(cos_libm);
            printf("Stopping benchmark due to allocation failure.\n"); break;
        }

        for (long long i = 0; i < n; i++) { angles[i] = (2.0 * ((double)rand() / RAND_MAX) - 1.0) * 100000.0 * M_PI; }

        // Warm-up
        int warmup_n = (n < 1000) ? (int)n : 1000;
        if (warmup_n > 0) {
            fabe13_sincos(angles, sin_fabe13, cos_fabe13, warmup_n);
            benchmark_libm_array(angles, sin_libm, cos_libm, warmup_n);
        }

        high_res_time_t start_wall, end_wall;
        double sec_fabe13_total = 0, sec_libm_total = 0;
        double cpu_user_sec_fabe13 = 0, cpu_sys_sec_fabe13 = 0;
        double cpu_user_sec_libm = 0, cpu_sys_sec_libm = 0;

        // --- Benchmark FABE13 ---
        #if CAN_MEASURE_RESOURCE_USAGE
            getrusage(RUSAGE_SELF, &usage_start);
        #endif
        get_high_res_time(&start_wall);

        for(int rep = 0; rep < num_reps; ++rep) { // Inner loop for timing only
            #if defined(__GNUC__) || defined(__clang__)
                fabe13_sincos((const double* __restrict__)angles, (double* __restrict__)sin_fabe13, (double* __restrict__)cos_fabe13, (int)n);
            #else
                fabe13_sincos(angles, sin_fabe13, cos_fabe13, (int)n);
            #endif
        }

        get_high_res_time(&end_wall);
        #if CAN_MEASURE_RESOURCE_USAGE
            getrusage(RUSAGE_SELF, &usage_end);
            cpu_user_sec_fabe13 = timeval_diff_sec(usage_start.ru_utime, usage_end.ru_utime);
            cpu_sys_sec_fabe13 = timeval_diff_sec(usage_start.ru_stime, usage_end.ru_stime);
            peak_rss_kb_fabe13 = usage_end.ru_maxrss; // Peak RSS for the process *up to this point*
        #endif
        sec_fabe13_total = high_res_time_diff(start_wall, end_wall);
        double sec_fabe13_avg = sec_fabe13_total / num_reps;


        // --- Benchmark standard libm ---
        #if CAN_MEASURE_RESOURCE_USAGE
            getrusage(RUSAGE_SELF, &usage_start);
        #endif
        get_high_res_time(&start_wall);

        for(int rep = 0; rep < num_reps; ++rep) { // Inner loop for timing only
             benchmark_libm_array(angles, sin_libm, cos_libm, n);
        }

        get_high_res_time(&end_wall);
        #if CAN_MEASURE_RESOURCE_USAGE
            getrusage(RUSAGE_SELF, &usage_end);
            cpu_user_sec_libm = timeval_diff_sec(usage_start.ru_utime, usage_end.ru_utime);
            cpu_sys_sec_libm = timeval_diff_sec(usage_start.ru_stime, usage_end.ru_stime);
            peak_rss_kb_libm = usage_end.ru_maxrss; // Peak RSS for the process *up to this point*
        #endif
        sec_libm_total = high_res_time_diff(start_wall, end_wall);
        double sec_libm_avg = sec_libm_total / num_reps;


        // --- Calculate & Print Results ---
        double mops_fabe13 = (sec_fabe13_avg > 1e-9) ? (double)n / sec_fabe13_avg / 1e6 : 0;
        double mops_libm = (sec_libm_avg > 1e-9) ? (double)n / sec_libm_avg / 1e6 : 0;
        double speedup = (mops_libm > 0 && mops_fabe13 > 0) ? mops_fabe13 / mops_libm : 0;

        // CPU Utilization (%) = (Total CPU Time / Wall Clock Time) * 100
        // Note: Measures utilization over the *entire* multi-rep timing block
        double cpu_total_sec_fabe13 = cpu_user_sec_fabe13 + cpu_sys_sec_fabe13;
        double cpu_util_fabe13 = (sec_fabe13_total > 1e-9) ? (cpu_total_sec_fabe13 / sec_fabe13_total) * 100.0 : 0.0;
        double cpu_total_sec_libm = cpu_user_sec_libm + cpu_sys_sec_libm;
        double cpu_util_libm = (sec_libm_total > 1e-9) ? (cpu_total_sec_libm / sec_libm_total) * 100.0 : 0.0;

        #if CAN_MEASURE_RESOURCE_USAGE
            printf(" %-15.4f | %-15.4f | %-15.2f | %-15.2f | %-10.2fx | %-15.1f | %-15.1f\n",
                   sec_fabe13_avg, sec_libm_avg, mops_fabe13, mops_libm, speedup, cpu_util_fabe13, cpu_util_libm);
            // Report memory separately
            printf("%-15s | Allocated: %.2f GB. Peak RSS: ~%ld KB (FABE13), ~%ld KB (Libm)\n",
                   "", total_gb, peak_rss_kb_fabe13, peak_rss_kb_libm);
        #else
            // Print without CPU/RAM usage columns
             printf(" %-15.4f | %-15.4f | %-15.2f | %-15.2f | %-10.2fx | %-15s | %-15s\n",
                   sec_fabe13_avg, sec_libm_avg, mops_fabe13, mops_libm, speedup, "N/A", "N/A");
             printf("%-15s | Allocated: %.2f GB. (Resource usage N/A)\n", "", total_gb);
        #endif


        // --- Accuracy Check (Limited size) ---
        const long long accuracy_check_limit = 1000000LL;
        if (n <= accuracy_check_limit) {
            #if defined(__GNUC__) || defined(__clang__)
                fabe13_sincos((const double* __restrict__)angles, (double* __restrict__)sin_fabe13, (double* __restrict__)cos_fabe13, (int)n);
            #else
                fabe13_sincos(angles, sin_fabe13, cos_fabe13, (int)n);
            #endif
            benchmark_libm_array(angles, sin_libm, cos_libm, n);
            double max_sin_diff = 0.0, max_cos_diff = 0.0; long long check_count = 0;
            for (long long i = 0; i < n; i++) {
                bool skip = isnan(sin_fabe13[i]) || isnan(sin_libm[i]) || isinf(sin_fabe13[i]) || isinf(sin_libm[i]) ||
                            isnan(cos_fabe13[i]) || isnan(cos_libm[i]) || isinf(cos_fabe13[i]) || isinf(cos_libm[i]);
                if (skip) continue; check_count++;
                double diff_sin = fabs(sin_fabe13[i] - sin_libm[i]); double diff_cos = fabs(cos_fabe13[i] - cos_libm[i]);
                if (diff_sin > max_sin_diff) max_sin_diff = diff_sin; if (diff_cos > max_cos_diff) max_cos_diff = diff_cos;
            }
             if (check_count > 0) printf("%-15s | Max diff vs libm (on %lld samples): sin=%.3e, cos=%.3e\n", "", check_count, max_sin_diff, max_cos_diff);
             else if (n > 0) printf("%-15s | Accuracy check skipped (all samples were NaN/Inf?)\n", "");
        } else {
             printf("%-15s | Accuracy check skipped for N > %lld\n", "", accuracy_check_limit);
        }

        benchmark_aligned_free(angles); benchmark_aligned_free(sin_fabe13); benchmark_aligned_free(cos_fabe13);
        benchmark_aligned_free(sin_libm); benchmark_aligned_free(cos_libm);
        if (n < max_n) printf("%-15s |\n", "");
    } // End loop over n

    printf("---------------------------------------------------------------------------------------------------------------------------------------\n");
    printf("Benchmark finished.\n");
    return 0;
}

#endif // ENABLE_FABE13_BENCHMARK
