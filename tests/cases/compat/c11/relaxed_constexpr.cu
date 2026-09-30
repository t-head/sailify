__host__ constexpr int array_size2 (int x) { 
  return x+1; 
} 
__global__ void kernel() {
 int array[array_size2(10)];     
}

__device__ constexpr int darray_size(int x) {
    return x + 1;
}

int g() {
    int array[darray_size(10)];
    return 0;
}