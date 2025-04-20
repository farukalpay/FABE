// fabe13/fabe13.h
#ifndef FABE13_H
#define FABE13_H

#ifdef __cplusplus
extern "C" {
#endif

void fabe13_sincos(const double* in, double* sin_out, double* cos_out, int n);
const char* fabe13_get_active_implementation_name(void);
int fabe13_get_active_simd_width(void);
double fabe13_sin(double x);
double fabe13_cos(double x);
double fabe13_sinc(double x);
double fabe13_tan(double x);
double fabe13_cot(double x);
double fabe13_atan(double x);
double fabe13_asin(double x);
double fabe13_acos(double x);

#ifdef __cplusplus
}
#endif

#endif // FABE13_H
