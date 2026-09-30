#define V(x) static_assert(x, "");

struct S{
  int x;
  int y;
};
struct S2 {
  int x;
  char v;
};
class C {
public:
  C(const C&) {}
  int v;
  int n;
};
class C2 {
public:
  C2(const C2&)=default;
  int v;
  int n;
};
class C3 {
public:
  int v;
  int n;
};
class C4 {
public:
  int v;
  int n;
  ~C4() {}
};
union U {
  int x;
  int y;
};

__global__ void kernel() {
  V(__has_unique_object_representations(S))
  V(!__has_unique_object_representations(S2))
  V(!__has_unique_object_representations(float))
  V(!__has_unique_object_representations(double))
  V(__has_unique_object_representations(int))
  V(!__has_unique_object_representations(C))
  V(__has_unique_object_representations(C2))
  V(__has_unique_object_representations(C3))
  V(!__has_unique_object_representations(C4))
  V(__has_unique_object_representations(U))
}

void f() {
  V(__has_unique_object_representations(S))
  V(!__has_unique_object_representations(S2))
  V(!__has_unique_object_representations(float))
  V(!__has_unique_object_representations(double))
  V(__has_unique_object_representations(int))
  V(!__has_unique_object_representations(C))
  V(__has_unique_object_representations(C2))
  V(__has_unique_object_representations(C3))
  V(!__has_unique_object_representations(C4))
  V(__has_unique_object_representations(U))
}