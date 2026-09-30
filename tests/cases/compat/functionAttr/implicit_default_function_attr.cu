struct S1 {
  __host__ S1() = default;
};

__device__ void foo1() {
  S1 s1;
}