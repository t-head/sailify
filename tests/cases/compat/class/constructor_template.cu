#include "test_buf.h"

class A {
public:
  int v{0};
  template<typename T>
  A(T x) {v=sizeof(x);}
};

template<>
A::A<int>(int x) {v=1;}

const int a[] = {0, 1, 2, 3};

TEST(class, constructor_template) {
  A a1(3);
  EXPECT_EQ(a1.v, 1);
  A a2(2.0);
  EXPECT_EQ(a2.v, 8);
  A a3(a);
  EXPECT_EQ(a3.v, 8);
}
