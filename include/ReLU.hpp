#pragma once

#include <algorithm>
#include <type_traits>

namespace NN::ActivationFunction {
template <typename dtype>
struct ReLU {
  static_assert(std::is_floating_point<dtype>::value,
                "dtype must be a floating point type.");
  [[nodiscard]] auto operator()(dtype x) const noexcept -> dtype {
    return std::max(static_cast<dtype>(0.0), x);
  };
};
}  // namespace NN::ActivationFunction