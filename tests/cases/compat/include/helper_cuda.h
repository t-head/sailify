#ifndef COMMON_HELPER_CUDA_H_
#define COMMON_HELPER_CUDA_H_

#include <stdio.h>
#include <stdlib.h>
#include <helper_string.h>

#define EXIT_WAIVED 2

#define MAX(a, b) (a > b ? a : b)
#define MIN(a, b) (a < b ? a : b)

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)
#define getLastCudaError(msg) __getLastCudaError(msg, __FILE__, __LINE__)
#define printLastCudaError(msg) __printLastCudaError(msg, __FILE__, __LINE__)

typedef int cudaError_stub_t;

template <typename T>
void check(T result, char const *const func, const char *const file, int const line) {
  (void) result;
  (void) func;
  (void) file;
  (void) line;
}

inline void __getLastCudaError(const char *errorMessage, const char *file, int line) {
  (void) errorMessage;
  (void) file;
  (void) line;
}

inline void __printLastCudaError(const char *errorMessage, const char *file, int line) {
  (void) errorMessage;
  (void) file;
  (void) line;
}

inline int _ConvertSMVer2Cores(int major, int minor) {
  (void) major;
  (void) minor;
  return 128;
}

inline const char *_ConvertSMVer2ArchName(int major, int minor) {
  (void) major;
  (void) minor;
  return "ppu";
}

inline int gpuGetMaxGflopsDeviceId() {
  return 0;
}

inline int gpuDeviceInit(int devID) {
  return devID;
}

inline int findCudaDevice(int argc, const char **argv) {
  (void) argc;
  (void) argv;
  return 0;
}

inline int findIntegratedGPU() {
  return 0;
}

inline bool checkCudaCapabilities(int major_version, int minor_version) {
  (void) major_version;
  (void) minor_version;
  return true;
}

#endif
