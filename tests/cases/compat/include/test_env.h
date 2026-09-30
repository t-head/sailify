#pragma once
#include "test_logger.h"
#include "gpu_neutral.h"
#ifndef __USE_PPUSDK__
  #include <cuda_runtime.h>
#else
  #include <hggcrt.h>
#endif

namespace testenv {

extern std::vector<long> env_seeds;
extern std::string env_goldenpath;
extern int env_gpuid;
extern bool goldenfile_path_specified;
extern bool env_verbose;

class CudaTestEnv : public testing::Environment {
  public:
    void SetUp() override {
      _InitRandomSeed();
      _InitGoldenFilePath();
      _InitGpuIdSelect();
      _InitVerbose();
    }
    bool seed_timer_based {false};
  private:
    void _InitRandomSeed() {
      if (gtestutil::GTEST_UTIL_FLAG(random_seed).size() == 0) {
        env_seeds.push_back(time(nullptr));
        seed_timer_based = true;
      } else {
        env_seeds = gtestutil::GTEST_UTIL_FLAG(random_seed);
      }
    }
    void _InitGoldenFilePath() {
      if (gtestutil::GTEST_UTIL_FLAG(golden_filepath) != "") {
        goldenfile_path_specified = true;
      }
      env_goldenpath = gtestutil::GTEST_UTIL_FLAG(golden_filepath);
      if (goldenfile_path_specified) {
        gtestutil::PrintColorFormatted(("@bB-- Golden file path: " + env_goldenpath + "\n").c_str());
      } else {
        gtestutil::PrintColorFormatted("@bB-- Golden file path will be specified per tests.\n");
      }
    }
    void _InitGpuIdSelect() {
      if (gtestutil::GTEST_UTIL_FLAG(select_gpuid) >= 0) {
        env_gpuid = gtestutil::GTEST_UTIL_FLAG(select_gpuid);
      } else {
        int device_count = 0;
        auto cuda_ret = gpuGetDeviceCount(&device_count);
        if (cuda_ret) {
          auto err = gpuGetLastError();
          TLOG_ERROR << "Get GPU count error: " << gpuGetErrorString(err) << "\n";
          exit(-1);
        }
        int max_smid = 0;
        int max_smdev = -1;
        for (int cur_dev = 0; cur_dev < device_count; cur_dev ++) {
          int compute_mode = -1;
          cuda_ret = gpuDeviceGetAttribute(&compute_mode, gpuDevAttrComputeMode, cur_dev);
          if (cuda_ret) {
            auto err = gpuGetLastError();
            TLOG_ERROR << "Get GPU compute mode error: " << gpuGetErrorString(err) << "\n";
            exit(-1);
          }
          if (compute_mode == gpuComputeModeProhibited) {
            continue;
          }
          int major = 0, minor = 0;
          cuda_ret = gpuDeviceGetAttribute(&major, gpuDevAttrComputeCapabilityMajor, cur_dev);
          auto cuda_ret2 = gpuDeviceGetAttribute(&minor, gpuDevAttrComputeCapabilityMinor, cur_dev);
          if (cuda_ret || cuda_ret2) {
            auto err = gpuGetLastError();
            TLOG_ERROR << "Get GPU SM ID error: " << gpuGetErrorString(err) << "\n";
            exit(-1);
          }
          int sm_id = major * 16 + minor;
          if (sm_id > max_smid) {
            max_smid = sm_id;
            max_smdev = cur_dev;
          }
        }
        TLOG_INFO << "Picked default GPU ID: " << max_smdev << ", SM: " << std::hex << max_smid << "\n";
        env_gpuid = max_smdev;
      }
      auto cuda_ret = gpuSetDevice(env_gpuid);
      gpuDeviceProp prop;
      auto cuda_ret2 = gpuGetDeviceProperties(&prop, env_gpuid);
      if (cuda_ret || cuda_ret2) {
        auto err = gpuGetLastError();
        TLOG_ERROR << "Set GPU id error: " << gpuGetErrorString(err) << "\n";
        exit(-1);
      }
      gtestutil::PrintColorFormatted(("@bB-- Select GPU id: " + std::to_string(env_gpuid) +
        ", GPU name: " + prop.name + "\n").c_str());
    }
    void _InitVerbose() {
      env_verbose = gtestutil::GTEST_UTIL_FLAG(verbose);
      TLOG_INFO << "Output of test details is " << (env_verbose ? "not " : "") << "suppressed.\n";
    }
};
}