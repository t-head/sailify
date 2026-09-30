#define NVRTC_GET_TYPE_NAME 1
#include <nvrtc.h>
#include "jittify.hpp"
#include "test_logger.h"
#include "test_env.h"
#include "cuda.h"
#include <algorithm>

#define CHECK_NVRTC_CALL(v)        \
  do {                             \
    nvrtcResult result = v;        \
    if (result != NVRTC_SUCCESS) { \
      TLOG_ERROR << "RTC failed with error " << nvrtcGetErrorString(result) << "\n"; \
      exit(-1);\
    } \
    TLOG_INFO << "RTC operation " << #v << " succeeded !\n"; \
  } while(0)
#define CHECK_CUDA_CALL(v)         \
  do {                             \
    CUresult result = v;           \
    if (result != CUDA_SUCCESS) {  \
      const char *errorStr = NULL; \
      cuGetErrorString(result, &errorStr); \
      TLOG_ERROR << "CUDA func call failed with error " << errorStr << "\n"; \
      exit(-1); \
    } \
    TLOG_INFO << "CUDA func call " << #v << " succeeded !\n"; \
  } while(0)

#define VAL(str) #str
#define MACRO_TO_STR(str) VAL(str)

namespace rtc {

void CompileModule(const std::string& source, const std::vector<const char*>& hdr,
const std::vector<const char*>& opts, const std::string& name, const std::string& kname,
CUmodule &modRes, std::string& lowername) {
  if (source == "") {
    TLOG_INFO << "No source code provided. Skip compiling...\n";
    return;
  }

  std::vector<const char*> hdr_codes;
  auto sys_hdr_map = get_jit_headers_map();
  int num_hdrs = hdr.size();
  for (auto& hdrname : hdr) {
    hdr_codes.push_back(sys_hdr_map[hdrname].c_str());
  }

  TLOG_DETAIL << "# of sys headers: " << num_hdrs << "\n";
  for(auto& v : hdr_codes) {
    TLOG_DETAIL << "##hdr code: " << v << "\n";
  }

  nvrtcProgram prog;
  CHECK_NVRTC_CALL(nvrtcCreateProgram(&prog, source.c_str(), name.c_str(), num_hdrs,
    hdr_codes.data(), hdr.data()));

  std::vector<const char*> compileOptions = opts;
#ifdef CUDA_INCLUDE_PATH
  char cuda_include_path[] = MACRO_TO_STR(CUDA_INCLUDE_PATH);
  TLOG_INFO << "CUDA SDK INCLUDE PATH " << cuda_include_path << " included to rtc.\n";
  std::string clp = std::string("--include-path=") + cuda_include_path;
  compileOptions.push_back(clp.c_str());
#endif

  const char *cuda_home = getenv("CUDA_HOME");
  std::string cld;
  std::string cccl_path;
  if (cuda_home) {
    cld = std::string("--include-path=") + std::string(cuda_home) + "/include";
    #if (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 0)
      cccl_path = std::string("--include-path=") + std::string(cuda_home) + "/include/cccl";
      compileOptions.push_back(cccl_path.c_str());
      compileOptions.push_back(cld.c_str());
      TLOG_DETAIL << "Adding include path: " << cccl_path << "\n";
      TLOG_DETAIL << "Adding include path: " << cld << "\n";
    #else 
      compileOptions.push_back(cld.c_str());
    #endif 
  }


  CHECK_NVRTC_CALL(nvrtcAddNameExpression(prog, kname.c_str()));

  TLOG_DETAIL << "##rtc options:\n";
  for (auto& opt : compileOptions) {
    TLOG_DETAIL << opt << "\n";
  }
  TLOG_DETAIL << "end of rtc options\n##src code: " << source << "\n";
  
  nvrtcCompileProgram(prog, compileOptions.size(), compileOptions.data());

  size_t logSize;
  CHECK_NVRTC_CALL(nvrtcGetProgramLogSize(prog, &logSize));
  char* log = new char[logSize + 1];
  CHECK_NVRTC_CALL(nvrtcGetProgramLog(prog, log));
  log[logSize] = '\x0';
  TLOG_INFO << "RTC log begin ---\n" << log << "\n --- end log\n";
  delete[] log;

  size_t codeSize;
  CHECK_NVRTC_CALL(nvrtcGetCUBINSize(prog, &codeSize));
  char* code = new char[codeSize];
  CHECK_NVRTC_CALL(nvrtcGetCUBIN(prog, code));

  const char* lowered_kernel_name = nullptr;
  CHECK_NVRTC_CALL(nvrtcGetLoweredName(prog, kname.c_str(), &lowered_kernel_name));
  std::copy(lowered_kernel_name, lowered_kernel_name + strlen(lowered_kernel_name), std::back_inserter(lowername));
  TLOG_INFO << "Lowered kernel name: " << lowername << "\n";

  CHECK_NVRTC_CALL(nvrtcDestroyProgram(&prog));

  CHECK_CUDA_CALL(cuModuleLoadData(&modRes, code));
  delete[] code;
}

CUfunction GetKernel(const CUmodule& mod, const std::string& name) {
  CUfunction kernel = nullptr;
  CHECK_CUDA_CALL(cuModuleGetFunction(&kernel, mod, name.c_str()));
  return kernel;
}

struct Kernel{
  Kernel() :mod_{nullptr} {}
  template<bool use_coop, typename... Args>
  typename std::enable_if<!use_coop, void>::type launch(CUfunction& ker,
      dim3 gdim, dim3 bdim, size_t smem, CUstream stream, Args... args) {
    void *kernelArgs[] = {(void*)&args...};
    CHECK_CUDA_CALL(cuLaunchKernel(ker, gdim.x, gdim.y, gdim.z, bdim.x, bdim.y,
      bdim.z, smem, stream, kernelArgs, NULL));
  }
  template<bool use_coop, typename... Args>
  typename std::enable_if<use_coop, void>::type launch(CUfunction& ker,
      dim3 gdim, dim3 bdim, size_t smem, CUstream stream, Args... args) {
    void *kernelArgs[] = {(void*)&args...};
    CHECK_CUDA_CALL(cuLaunchCooperativeKernel(ker, gdim.x, gdim.y, gdim.z, bdim.x, bdim.y,
      bdim.z, smem, stream, kernelArgs));
  }
  template<bool use_coop, typename... Args>
  void compile_and_launch(const std::string& code, const std::vector<const char*>& hdr,
      const std::string& kname, dim3 gdim, dim3 bdim, size_t smem, CUstream stream, Args... args) {
    if (code == "") {
      TLOG_INFO << "No source code provided. Skip launching...\n";
      return;
    }
    TLOG_INFO << "Compiling kernel with name " << kname << " ...\n";
    std::string cname = kname + "_rtc";
    std::string lowername;
    CompileModule(code, hdr, rtc_options, cname, kname, mod_, lowername);
    CUfunction kfunc = GetKernel(mod_, lowername);
    launch<use_coop>(kfunc, gdim, bdim, smem, stream, args...);
  }
  template<bool use_coop, typename... Args>
  void compile_and_launch(const std::string& code, const std::vector<const char*>& hdr,
      const std::string& kname, const std::string& cname, dim3 gdim, dim3 bdim, size_t smem,
      CUstream stream, Args... args) {
    if (code == "") {
      TLOG_INFO << "No source code provided. Skip launching...\n";
      return;
    }
    TLOG_INFO << "Compiling kernel with name " << kname << " ...\n";
    TLOG_INFO << "Providing program filename " << cname << " ...\n";
    std::string lowername;
    CompileModule(code, hdr, rtc_options, cname, kname, mod_, lowername);
    CUfunction kfunc = GetKernel(mod_, lowername);
    launch<use_coop>(kfunc, gdim, bdim, smem, stream, args...);
  }
  void add_rtc_option(const char* opt) {
    rtc_options.push_back(opt);
  }
  void remove_rtc_option(const char* opt) {
    rtc_options.erase(std::remove(rtc_options.begin(), rtc_options.end(), opt),
      rtc_options.end());
  }
  ~Kernel() {
    if (mod_) {
      CHECK_CUDA_CALL(cuModuleUnload(mod_));
    }
  }
private:
  CUmodule mod_;
    std::vector<const char*> rtc_options{
    "--std=c++14",
    "--device-as-default-execution-space",
    "--gpu-architecture=sm_80"};
};
}