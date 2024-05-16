#pragma once

#include <algorithm>
#include <type_traits>

#include "NNActivationFunction.hpp"

namespace NN::ActivationFunction {
template <typename dtype = double>
class ReLU : public BaseActivationFunction<dtype> {
  static_assert(std::is_floating_point<dtype>::value,
                "dtype must be a floating point type.");

  [[nodiscard]] auto apply(dtype x, bool differential) const noexcept
      -> dtype override {
    return differential ? std::max(static_cast<dtype>(0.0), x)
                        : (x >= 0 ? 1 : 0);
  };
};
}  // namespace NN::ActivationFunction