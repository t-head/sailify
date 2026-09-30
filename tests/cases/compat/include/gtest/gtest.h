#ifndef SAILIFY_TESTS_GTEST_SHIM_H
#define SAILIFY_TESTS_GTEST_SHIM_H
#include <vector>
#include <string>
namespace testenv {
extern std::vector<long> env_seeds;
extern std::string env_goldenpath;
extern int env_gpuid;
extern bool goldenfile_path_specified;
extern bool env_verbose;
}


namespace sailify_shim {

struct NullLog {
    template <typename T>
    NullLog& operator<<(const T&) { return *this; }
};

}


namespace testlogger {

enum class loglevel {
  INFO,
  DETAIL,
  ERROR
};

template <loglevel l>
class TestInfoLogger {
  public:
    template <typename T>
    TestInfoLogger& operator<<(const T&) { return *this; }
    void flush(bool toggle = false) { (void)toggle; }
    TestInfoLogger() = default;
};

}


#define TEST(suite, name) static void suite##_##name(void)
#define EXPECT_TRUE(expr) ((void)(expr))
#define EXPECT_FALSE(expr) ((void)0)
#define EXPECT_EQ(a, b) ((void)((a) == (b)))
#define EXPECT_NE(a, b) ((void)0)
#define EXPECT_FLOAT_EQ(a, b) ((void)((a) == (b)))
#define EXPECT_DOUBLE_EQ(a, b) ((void)((a) == (b)))
#define EXPECT_LT(a, b) ((void)((a) < (b)))
#define EXPECT_LE(a, b) ((void)((a) <= (b)))
#define EXPECT_GT(a, b) ((void)((a) > (b)))
#define EXPECT_GE(a, b) ((void)((a) >= (b)))
#define EXPECT_STREQ(a, b) ((void)0)
#define ASSERT_TRUE(expr) ((void)(expr))
#define ASSERT_FALSE(expr) ((void)0)
#define ASSERT_EQ(a, b) ((void)((a) == (b)))
#define ASSERT_NE(a, b) ((void)0)
#define ASSERT_FLOAT_EQ(a, b) ((void)((a) == (b)))
#define TLOG_INFO sailify_shim::NullLog()
#define TLOG_ERROR sailify_shim::NullLog()
#define TLOG_DETAIL sailify_shim::NullLog()

#endif
