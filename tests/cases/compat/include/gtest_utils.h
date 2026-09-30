#pragma once
#include <vector>
#include <sstream>
#include <iostream>
#include <unistd.h>
#include <stdio.h>
#include <stdarg.h>
#include <gtest/gtest.h>

#define GTEST_UTIL_FLAG(name) UTIL_FLAGS_##name
#define GTEST_UTIL_DEFINE_VEC(name, etype) std::vector<etype> GTEST_UTIL_FLAG(name);
#define GTEST_UTIL_DEFINE_STR(name) std::string GTEST_UTIL_FLAG(name);
#define GTEST_UTIL_DEFINE_INT32(name, defval) int GTEST_UTIL_FLAG(name) = defval;
#define GTEST_UTIL_DEFINE_BOOL(name, defval) bool GTEST_UTIL_FLAG(name) = defval;

namespace gtestutil {

static GTEST_UTIL_DEFINE_VEC(random_seed, long)
static GTEST_UTIL_DEFINE_STR(golden_filepath)
static GTEST_UTIL_DEFINE_INT32(select_gpuid, -1)
static GTEST_UTIL_DEFINE_BOOL(verbose, false)

static const char gtestUtilSupportedFlagMessage[] =
"@bCSupported flags:\n"
"  @bM--randomly-seed=val1[val2,val3,...]\n"
"   @MSpecify global random seed list that could be used by some tests.\n"
"   If not specified, use current time based seed.\n"
"  @bM--goldenfile-path=goldenfilepath\n"
"   @MSpecify golden file path that could be used by some tests.\n"
"   If not specified, tests specify their own golden file path.\n"
"  @bM--select-gpuid=gpuid\n"
"   @MSpecify gpu id to run cuda tests.\n"
"   If not specified, by default pick the device id with highest SM.\n"
"  @bM--verbose\n"
"   @MSpecify whether outputing details of tests.\n"
"   If not specified, by default details will not be shown in the output.\n";

static const char* ParseFlag(const char *str, const char *flag, bool flag_only) {
  if (str == NULL || flag == NULL) {
    return NULL;
  }
  const std::string flag_str = std::string("--") + flag;
  const size_t flag_len = flag_str.length();
  if (strncmp(str, flag_str.c_str(), flag_len) != 0) {
    return NULL;
  }
  const char* flag_end = str + flag_len;
  if (flag_only) {
    return flag_end;
  }
  if (flag_end[0] != '=') {
    return NULL;
  }
  return flag_end + 1;
}

static void trim(std::string &str) {
  auto pos = str.find_last_not_of(" \n\r\t");
  str.erase(pos + 1);
  pos = str.find_first_not_of(" \n\r\t");
  str.erase(0, pos);
}

static std::vector<std::string> SplitStrWithDelim(const char *str, const char delim) {
  std::vector<std::string> substrings;
  std::istringstream iss(str);
  std::string tmp;
  while (std::getline(iss, tmp, delim)) {
    trim(tmp);
    if (tmp != "") {
      substrings.emplace_back(std::move(tmp));
    }
  }
  return substrings;
}

static bool ParseIntListFlag(const char *str, const char *flag, std::vector<long> &vec) {
  const char* value_p = ParseFlag(str, flag, false);
  if (value_p == NULL) {
    return false;
  }
  auto tokens = SplitStrWithDelim(value_p, ',');
  for (auto &s : tokens) {
    long long_val = strtol(s.c_str(), NULL, 0); 
    vec.push_back(long_val);
  }
  return true;
}

static bool ParseInt32Flag(const char *str, const char *flag, int *val) {
  const char *value_p = ParseFlag(str, flag, false);
  if (value_p == NULL) {
    return false;
  }
  *val = strtol(value_p, NULL, 10);
  return true;
}

static bool ParseStringFlag(const char *str, const char *flag, std::string *value) {
  const char *value_p = ParseFlag(str, flag, false);
  if (value_p == NULL) {
    return false;
  }
  *value = value_p;
  return true;
}

static bool ParseOptionFlag(const char *str, const char *flag, bool *val) {
  const char *value_p = ParseFlag(str, flag, true);
  if (value_p == NULL) {
    return false;
  }
  *val = true;
  return true;
}

static void PrintWithColor(int cfmt, int color, const char* fmt, ...) {
  const char *term = getenv("TERM");
  bool term_supports_color = term && (!strncmp(term, "xterm", 5) ||
    !strncmp(term, "screen", 6) || !strncmp(term, "linux", 5));
  bool is_tty = isatty(fileno(stdout));
  va_list vargs;
  va_start(vargs, fmt);
  if (term_supports_color && is_tty) {
    printf("\033[%d;3%dm", cfmt, color);
    vprintf(fmt, vargs);
    printf("\033[m");
  } else {
    vprintf(fmt, vargs);
  }
  va_end(vargs);
}

static void PrintColorFormatted(const char *str) {
  int cfmt = 0;
  int color = 0;
  while(true) {
    const char *p = strchr(str, '@');
    if (!p) {
      PrintWithColor(cfmt, color, "%s", str);
      return;
    }
    PrintWithColor(cfmt, color, "%s", std::string(str, p).c_str());
    char ch = p[1];
    str = p;
    if (ch == 'b') {
      cfmt = 1;
      str ++;
      ch = p[2];
    } else {
      cfmt = 0;
    }
    bool nofmt = false;
    switch (ch) {
      case 'D':
        color = 0;
        break;
      case 'R':
        color = 1;
        break;
      case 'G':
        color = 2;
        break;
      case 'Y':
        color = 3;
        break;
      case 'B':
        color = 4;
        break;
      case 'M':
        color = 5;
        break;
      case 'C':
        color = 6;
        break;
      default:
        nofmt = true;
        break;
    }
    if (!nofmt) {
      str += 2;
    } else {
      str += 1;
    }
  }
}

static void ParseCommandLineFlags(int argc, char **argv) {
  for (int i = 1; i < argc; i ++) {
    std::string arg_str {argv[i]};
    if (arg_str == "--list_flags") {
      PrintColorFormatted(gtestUtilSupportedFlagMessage);
      exit(0);
    } else {
      if (ParseIntListFlag(argv[i], "randomly-seed",
                           GTEST_UTIL_FLAG(random_seed))) {
        PrintColorFormatted("@B--Parsed random seed--\n");
        continue;
      }
      if (ParseStringFlag(argv[i], "goldenfile-path",
                          &GTEST_UTIL_FLAG(golden_filepath))) {
        PrintColorFormatted("@B--Parsed golden file path--\n");
        continue;
      }
      if (ParseInt32Flag(argv[i], "select-gpuid",
                         &GTEST_UTIL_FLAG(select_gpuid))) {
        PrintColorFormatted("@B--Parsed select pguid--\n");
        continue;
      }
      if (ParseOptionFlag(argv[i], "verbose",
                         &GTEST_UTIL_FLAG(verbose))) {
        PrintColorFormatted("@B--Parsed verbose--\n");
        continue;
      }
      PrintColorFormatted(("@Y--Skip unrecognized flag: " + arg_str + "\n").c_str());
    }
  }
}

template <typename Parent, typename Factory>
static ::testing::TestInfo* Registertest(const char* test_suite_name, const char* test_name,
                       const char* type_param, const char* value_param,
                       const char* file, int line, Factory factory) {
  using TestT = typename std::remove_pointer<decltype(factory())>::type;

  class FactoryImpl : public testing::internal::TestFactoryBase {
   public:
    explicit FactoryImpl(Factory f) : factory_(std::move(f)) {}
    testing::Test* CreateTest() override { return factory_(); }

   private:
    Factory factory_;
  };

  return testing::internal::MakeAndRegisterTestInfo(
      test_suite_name, test_name, type_param, value_param,
      testing::internal::GetTypeId<Parent>(),
      TestT::SetUpTestCase, TestT::TearDownTestCase,
      new FactoryImpl{std::move(factory)});
}

}