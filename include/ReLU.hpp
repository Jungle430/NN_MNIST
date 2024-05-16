#pragma once

#include <algorithm>
#include <type_traits>

namespace NN::ActivationFunction {
template <typename dtype = double>
struct ReLU {
  static_assert(std::is_floating_point<dtype>::value,
                "dtype must be a floating point type.");

  [[nodiscard]] auto operator()(dtype x,
                                bool differential = true) const noexcept
      -> dtype {
    return differential ? std::max(static_cast<dtype>(0.0), x)
                        : (x >= 0 ? 1 : 0);
  };
};
}  // namespace NN::ActivationFunction