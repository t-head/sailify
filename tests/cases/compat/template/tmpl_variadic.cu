template <int... Ns> struct Sum;
template <> struct Sum<> { static const int value = 0; };
template <int N, int... Rest> struct Sum<N, Rest...> { static const int value = N + Sum<Rest...>::value; };
typedef char variadic_ok[(Sum<1, 2, 3, 4, 5>::value == 15) ? 1 : -1];
int variadicCheck(void) { return Sum<3, 4>::value; }
