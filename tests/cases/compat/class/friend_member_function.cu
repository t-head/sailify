#include "test_buf.h"

class PixelRGBA {
public:
    __device__ PixelRGBA(): r_(0), g_(0), b_(0), a_(0) { }

    __device__ PixelRGBA(unsigned char r, unsigned char g,
                         unsigned char b, unsigned char a = 255):
                         r_(r), g_(g), b_(b), a_(a) { }

    __device__ uint32_t get_hex() {
      return (uint32_t(r_) << 24) | (uint32_t(g_) << 16) | (uint32_t(b_) << 8) | a_;
    }

private:
    unsigned char r_, g_, b_, a_;

    friend PixelRGBA operator+(const PixelRGBA&, const PixelRGBA&);
};

__device__
PixelRGBA operator+(const PixelRGBA& p1, const PixelRGBA& p2)
{
    return PixelRGBA(p1.r_ + p2.r_, p1.g_ + p2.g_,
                     p1.b_ + p2.b_, p1.a_ + p2.a_);
}

__global__ void kernel(uint32_t *dp)
{
    PixelRGBA p1{1, 2, 3}, p2{10, 20, 30, 0};
    PixelRGBA p3 = p1 + p2;
    dp[0] = p3.get_hex();
}

TEST(class, friend_member_function) {
  auto testDBuf = testBuf<uint32_t, memory_scope::device>(1);
  auto testHBuf = testBuf<uint32_t, memory_scope::host>(1);
  kernel<<<1, 1>>>(testDBuf);
  testHBuf = testDBuf;
  EXPECT_EQ(testHBuf[0], 0xB1621FF);
}