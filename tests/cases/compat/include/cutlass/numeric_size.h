#pragma once

namespace cutlass {

template <typename T>
struct sizeof_bits {
  static constexpr int value = static_cast<int>(sizeof(T) * 8);
};

}
