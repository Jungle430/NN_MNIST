#pragma once

#include <cmath>
#include <type_traits>

namespace NN::ActivationFunction {
template <typename dtype>
struct Sigmoid {
  static_assert(std::is_floating_point<dtype>::value,
                "dtype must be a floating point type.");
  [[nodiscard]] auto operator()(dtype x) const noexcept -> dtype {
    return 1 / (1 + std::exp(-x));
  };
};
}  // namespace NN::ActivationFunction