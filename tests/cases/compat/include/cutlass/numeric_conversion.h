#pragma once

#include <cutlass/array.h>

namespace cutlass {

template <typename DstT, typename SrcT, int N>
struct NumericArrayConverter {
  Array<DstT, N> operator()(Array<SrcT, N> const& src) const {
    Array<DstT, N> dst;
    for (int i = 0; i < N; ++i) {
      dst[i] = static_cast<DstT>(src[i]);
    }
    return dst;
  }
};

}
