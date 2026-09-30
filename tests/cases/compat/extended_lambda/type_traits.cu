#define IS_D_LAMBDA(X) __nv_is_extended_device_lambda_closure_type(X)
#define IS_HD_LAMBDA(X) __nv_is_extended_host_device_lambda_closure_type(X)

auto lam0 = [] __host__ __device__ { };

void foo(void) {
  auto lam1 = [] { };
  auto lam2 = [] __device__ { };
  auto lam3 = [] __host__ __device__ { };

  static_assert(!IS_D_LAMBDA(decltype(lam0)), "");
  static_assert(!IS_HD_LAMBDA(decltype(lam0)), "");

  static_assert(!IS_D_LAMBDA(decltype(lam1)), "");
  static_assert(!IS_HD_LAMBDA(decltype(lam1)), "");

  static_assert(IS_D_LAMBDA(decltype(lam2)), "");
  static_assert(!IS_HD_LAMBDA(decltype(lam2)), "");

  static_assert(!IS_D_LAMBDA(decltype(lam3)), "");
  static_assert(IS_HD_LAMBDA(decltype(lam3)), "");

  auto lam4 = [] __device__ { return 0; };
  {
    auto lam5 = [] __device__ { return 0; };
    auto lam6 = [] __device__ __host__ { return 0; };
    static_assert(IS_D_LAMBDA(decltype(lam5)), "");
    static_assert(!IS_HD_LAMBDA(decltype(lam5)), "");
    static_assert(!IS_D_LAMBDA(decltype(lam6)), "");
    static_assert(IS_HD_LAMBDA(decltype(lam6)), "");
  }
}
