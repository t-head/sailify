#pragma once
#include <fstream>
#include <assert.h>
#include <stdlib.h>
#include <algorithm>
#include <cxxabi.h>
#include <cstdint>
#include <string>
#include "vector_utils.h"
#include "gpu_neutral.h"

extern std::string golden_prefix = "/ppusw/resource/CompilerTest/ppu1.0/";

enum class memory_scope {device, host, pinned};
template<typename T, memory_scope>
struct testBuf;

extern char **Gargv;

template<typename T>
struct testBuf<T, memory_scope::device> {
  using type = T;
  friend class testBuf<T, memory_scope::host>;
  testBuf(size_t num_element, std::string name) : size{num_element * sizeof(T)}, bufname{name} {
    gpuMalloc((void**)&ptr, size);
  }
  testBuf(size_t num_element) : testBuf(num_element, "") {}
  testBuf(testBuf&& other) : ptr{other.ptr}, size{other.size}, bufname{other.bufname} {
    other.ptr = nullptr;
    other.size = 0;
  }
  ~testBuf() {
    if (ptr) {
      gpuFree(ptr);
    }
  }
  T* get_pointer() {
    return ptr;
  }
  template<class FUNC>
  void initialize(FUNC f) {
    std::vector<T> vec(get_size());
    std::for_each(vec.begin(),vec.end(), 
      [&vec,&f](auto& elem){ elem = f(&elem - &vec[0]);});
    gpuMemcpy(ptr, vec.data(), size, gpuMemcpyHostToDevice);
  }
  void clear() {
    gpuMemset(ptr, 0, size);
  }
  T* get_initialized_pointer() {
    gpuMemset(ptr, 0, size);
    return ptr;
  }
  size_t get_size() {
    return size / sizeof(T);
  }
  operator T*() {
    return ptr;
  }
  testBuf<T, memory_scope::device> clone(bool copy_val=false) {
    testBuf<T, memory_scope::device> copy = testBuf<T, memory_scope::device>(size / sizeof(T));
    if (copy_val) {
      gpuMemcpy(copy.ptr, ptr, size, gpuMemcpyDeviceToDevice);
    }
    return copy;
  }
  testBuf<T, memory_scope::device>& operator= (const testBuf<T, memory_scope::host>& buf) {
    size_t copysize = std::min(size, buf.size);
    gpuMemcpy(ptr, buf.ptr, copysize, gpuMemcpyHostToDevice);
    return *this;
  }
  template<typename U>
  testBuf<T, memory_scope::device>& operator= (const std::initializer_list<U> & list) {
    std::vector<U> vec = list;
    size_t copysize = std::min(size, vec.size() * sizeof(U));
    gpuMemcpy(ptr, vec.data(), copysize, gpuMemcpyHostToDevice);
    return *this;
  }
  template<typename U>
  testBuf<T, memory_scope::device>& operator= (const std::vector<U> & vec) {
    size_t copysize = std::min(size, vec.size() * sizeof(U));
    gpuMemcpy(ptr, vec.data(), copysize, gpuMemcpyHostToDevice);
    return *this;
  }
  void fromhostptr(void* hptr, size_t hsize) {
    gpuMemcpy(ptr, hptr, hsize, gpuMemcpyHostToDevice);
  }
private:
  T* ptr;
  size_t size;
  std::string bufname;
};

template<typename T>
struct testBuf<T, memory_scope::host> {
  using type = T;
  friend class testBuf<T, memory_scope::device>;
  testBuf(size_t num_element, std::string name) : size{num_element * sizeof(T)}, bufname{name} {
    ptr = (T*)malloc(size);
  }
  testBuf(size_t num_element) : testBuf(num_element, "") {}
  testBuf(testBuf&& other) : ptr{other.ptr}, size{other.size}, bufname{other.bufname} {
    other.ptr = nullptr;
    other.size = 0;
  }
  ~testBuf() {
    if (ptr) {
      free(ptr);
    }
  }
  T* get_pointer() {
    return ptr;
  }
  testBuf<T, memory_scope::host>& clear() {
    memset(ptr, 0, size);
    return *this;
  }
  testBuf<T, memory_scope::host>& sort() {
    std::sort(ptr, ptr + get_size());
    return *this;
  }
  T* get_initialized_pointer() {
    memset(ptr, 0, size);
    return ptr;
  }
  size_t get_size() {
    return size / sizeof(T);
  }
  template<class FUNC>
  void foreach(FUNC f) {
    for (int i = 0; i < size/sizeof(T); i ++) {
      f(ptr[i], i);
    }
  }
  template<class FUNC>
  testBuf<T, memory_scope::host>& initialize(FUNC f) {
    foreach([&f](T& v, int i){v=f(i);});
    return *this;
  }
  void setname(std::string name) {
    bufname = name;
  }
  testBuf<T, memory_scope::host> clone(std::string name="", bool copy_val=false) {
    testBuf<T, memory_scope::host> copy = testBuf<T, memory_scope::host>(size / sizeof(T));
    if (copy_val) {
      memcpy(copy.ptr, ptr, size);
    }
    if (name != "") {
      copy.setname(name);
    } else {
      copy.setname(bufname + "_c");
    }
    return copy;
  }
  testBuf<T, memory_scope::host>& operator= (const testBuf<T, memory_scope::device>& buf) {
    size_t copysize = std::min(size, buf.size);
    gpuMemcpy(ptr, buf.ptr, copysize, gpuMemcpyDeviceToHost);
    return *this;
  }
  template<typename U>
  testBuf<T, memory_scope::host>& operator= (const std::vector<U> & vec) {
    size_t copysize = std::min(get_size(), vec.size());
    for (size_t i = 0; i < copysize; i ++) {
      ptr[i] = *const_cast<T*>(reinterpret_cast<const T*>(vec.data() + i));
    }
    return *this;
  }
  template<typename U>
  testBuf<T, memory_scope::host>& operator= (const std::initializer_list<U> & list) {
    std::vector<U> vec = list;
    size_t copysize = std::min(size, vec.size() * sizeof(U));
    memcpy(ptr, vec.data(), copysize);
    return *this;
  }
  bool operator== (const testBuf<T, memory_scope::host> & buf) {
    assert (size <= buf.size);
    return memcmp(ptr, buf.ptr, size) == 0;
  }
  T& operator[] (int index) {
    assert (index < size / sizeof(T));
    return ptr[index];
  }
  operator T*() {
    return ptr;
  }

  void printv(int vsize, bool hex=false) {
    _printvec(hex, vsize);
  }
  template<typename U=T>
  typename std::enable_if<sizeof(U)==1, void>::type print(bool hex=false) {
    _print(true, hex);
  }
  template<typename U=T>
  typename std::enable_if<sizeof(U)!=1, void>::type print(bool hex=false) {
    _print(false, hex);
  }
  void tofile(std::string filename) {
    std::string golden_filepath = _get_golden_path();
    std::string cmd = "mkdir -p " + golden_filepath;
    std::system(cmd.c_str());
    golden_filepath += filename + ".bin";
    auto goldenfile = std::ofstream(golden_filepath, std::ios::binary);
    if (goldenfile.good()) {
      goldenfile.write(reinterpret_cast<char*>(ptr), size);
      goldenfile.close();
    } else {
      TLOG_ERROR << "open file failed! " << golden_filepath << "\n";
    }
  }
  void tofile(std::string path, std::string filename) {
    std::string golden_filepath = path;
    std::string cmd = "mkdir -p " + golden_filepath;
    std::system(cmd.c_str());
    golden_filepath += filename + ".bin";
    auto goldenfile = std::ofstream(golden_filepath, std::ios::binary);
    if (goldenfile.good()) {
      goldenfile.write(reinterpret_cast<char*>(ptr), size);
      goldenfile.close();
    } else {
      TLOG_ERROR << "open file failed! " << golden_filepath << "\n";
    }
  }
  void fromfile(std::string filename) {
    std::string golden_filepath = _get_golden_path();
    golden_filepath += filename + ".bin";
    auto goldenfile = std::ifstream(golden_filepath, std::ios::binary);
    if(goldenfile.good()) {
      goldenfile.read(reinterpret_cast<char*>(ptr), size);
      goldenfile.close();
    } else {
      TLOG_ERROR << "golden file not exist! " << golden_filepath << "\n";
      exit(-1);
    }
  }
  void fromfile(std::string path, std::string filename) {
    std::string golden_filepath = path;
    golden_filepath += filename + ".bin";
    auto goldenfile = std::ifstream(golden_filepath, std::ios::binary);
    if(goldenfile.good()) {
      goldenfile.read(reinterpret_cast<char*>(ptr), size);
      goldenfile.close();
    } else {
      TLOG_ERROR << "golden file not exist! " << golden_filepath << "\n";
      exit(-1);
    }
  }
  void fromdevptr(T* dptr, size_t dsize) {
    gpuMemcpy(ptr, dptr, dsize, gpuMemcpyDeviceToHost);
  }
private:
  std::string _get_golden_path() {
    std::string golden_filepath;
    char full_path[PATH_MAX];
    realpath(Gargv[0], full_path);
    std::string golden_fullpath = full_path;
    auto build_pos = golden_fullpath.find("/build");
    if (testenv::goldenfile_path_specified) {
      golden_filepath = testenv::env_goldenpath;
    } else {
      golden_filepath = golden_fullpath.substr(0, build_pos) + "/golden/";
    }
    if (golden_filepath.back() != '/') {
      golden_filepath += "/";
    }
    std::string partial_path = golden_fullpath.substr(build_pos + 7);
    partial_path = partial_path.substr(partial_path.find("/") + 1);
    golden_filepath += partial_path.substr(0, partial_path.find("/")) + "/";
    golden_filepath += partial_path.substr(partial_path.find_last_of("/") + 1) + "/";
    return golden_filepath;
  }
  void _printvec(bool hex, int vecsize) {
    auto logger = testlogger::TestInfoLogger<testlogger::loglevel::INFO>();
    T *p = ptr;
    logger << "\n====== ##BUF " << bufname << " begins ======\n";
    logger.flush();
    std::string pfx = hex ? "0x" : "";
    for (size_t i = 0; i < size; i += sizeof(T)) {
      logger << "Idx " << (i / sizeof(T)) << ": x = ";
      if (hex) {
        logger << std::hex;
      }
      logger << pfx << (p -> x) << ", y = " << pfx << (p -> y);
      if (vecsize >= 3) {
        logger << ", z = " << pfx << (p -> z);
      }
      if (vecsize >= 4) {
        logger << ", w = " << pfx << (p -> w);
      }
      logger << "\n";
      logger.flush(true);
      p ++;
      if (hex) {
        logger << std::dec;
      }
    }
    logger << "\n====== ##BUF " << bufname << " ends ======\n\n";
    logger.flush(true);
  }
  void _print(bool tob32, bool hex) {
    auto logger = testlogger::TestInfoLogger<testlogger::loglevel::INFO>();
    T *p = ptr;
    logger << "\n====== ##BUF " << bufname << " begins ======\n";
    logger.flush();
    for (size_t i = 0; i < size; i += sizeof(T)) {
      logger << "Idx " << (i / sizeof(T));
      logger << ": " << (hex ? "0x": "");
      if (hex) {
        logger << std::hex;
      }
      if (!tob32) {
        logger << (*(p ++));
      } else {
        logger << (uint32_t)(*(p ++));
      }
      logger << "  ";
      if ((i / sizeof(T) + 1) % 10 == 0) {
        logger << "\n";
        logger.flush(true);
      }
      if (hex) {
        logger << std::dec;
      }
    }
    logger << "\n====== ##BUF " << bufname << " ends ======\n\n";
    logger.flush(true);
  }
  T* ptr;
  size_t size;
  std::string bufname;
};

template<typename T>
struct testBuf<T, memory_scope::pinned> {
  using type = T;
  testBuf(size_t num_element, std::string name) : size{num_element * sizeof(T)}, bufname{name} {
    gpuMallocHost((void**)&ptr, size);
    gpuHostGetDevicePointer((void**)&dptr, ptr, 0);
  }
  testBuf(size_t num_element) : testBuf(num_element, "") {}
  testBuf(testBuf&& other) : ptr{other.ptr}, size{other.size}, bufname{other.bufname} {
    other.ptr = nullptr;
    other.size = 0;
  }
  ~testBuf() {
    if (ptr) {
      gpuFreeHost(ptr);
    }
  }
  T* get_pointer() {
    return dptr;
  }
  void clear() {
    gpuMemset(ptr, 0, size);
  }
  T* get_initialized_pointer() {
    gpuMemset(dptr, 0, size);
    return dptr;
  }
  size_t get_size() {
    return size / sizeof(T);
  }
  T* get_host_pointer() {
    return ptr;
  }
private:
  T* ptr;
  T* dptr;
  size_t size;
  std::string bufname;
};

template<typename T>
std::string print_types() {
  std::string tmp = typeid(T).name();
  int status;
  char *s = abi::__cxa_demangle(tmp.c_str(), nullptr, nullptr, &status);
  if (!status) {
    std::string ss = std::string(s);
    free(s);
    return ss;
  } else {
    return "";
  }
}
