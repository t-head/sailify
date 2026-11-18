#pragma once
#include "gtest_utils.h"

namespace testenv {
  extern bool env_verbose;
}

namespace testlogger {

enum class loglevel {
  INFO,
  DETAIL,
  ERROR
};

template<loglevel l>
class TestInfoLogger {
  public:
    template<typename T>
    TestInfoLogger& operator<<(T input) {
      if (l != loglevel::DETAIL || testenv::env_verbose) {
        s << input;
        empty = false;
      }
      return *this;
    }
    void flush(bool toggle=false) {
      std::string buf;
      if (l == loglevel::INFO) {
        buf = "@C";
      } else if (l == loglevel::ERROR) {
        buf = "@R";
      } else if (l == loglevel::DETAIL) {
        buf = "@Y";
      }
      if (!toggle){
        buf += (l == loglevel::INFO) ? "[TESTINFO] " :
          (l == loglevel::ERROR) ? "[TESTERROR] " : "[TESTDETAILS] ";
      }
      buf += s.str();
      gtestutil::PrintColorFormatted(buf.c_str());
      s.str("");
      s.clear();
      empty = true;
    }
    ~TestInfoLogger() {
      if (!empty) {
        flush();
      }
    }
    TestInfoLogger() = default;
    TestInfoLogger(const TestInfoLogger&) {}
    TestInfoLogger& operator=(const TestInfoLogger&) {};
  private:
    std::ostringstream s;
    bool empty {true};
};

#define TLOG_INFO testlogger::TestInfoLogger<testlogger::loglevel::INFO>()
#define TLOG_ERROR testlogger::TestInfoLogger<testlogger::loglevel::ERROR>()
#define TLOG_DETAIL testlogger::TestInfoLogger<testlogger::loglevel::DETAIL>()
}