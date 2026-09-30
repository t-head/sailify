template <bool B, typename T> struct EnableIf;
template <typename T> struct EnableIf<true, T> { typedef T type; };
template <typename T> typename EnableIf<(sizeof(T) < 8), T>::type smallOnly(T v) { return v; }
template <typename T> typename EnableIf<(sizeof(T) >= 8), T>::type bigOnly(T v) { return v; }
int enableIfCheck(void) { return smallOnly(3) + (int)bigOnly(2.5); }
