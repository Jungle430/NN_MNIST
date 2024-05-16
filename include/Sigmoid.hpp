#pragma once

#include <cmath>
#include <type_traits>

#include "NNActivationFunction.hpp"

namespace NN::ActivationFunction {
template <typename dtype = double>
class Sigmoid : public BaseActivationFunction<dtype> {
  static_assert(std::is_floating_point<dtype>::value,
                "dtype must be a floating point type.");

  [[nodiscard]] auto apply(dtype x, bool differential) const noexcept
      -> dtype override {
    return differential ? 1 / (1 + std::exp(-x))
                        : std::exp(-x) / std::pow((1 + std::exp(-x)), 2);
  };
};
}  // namespace NN::ActivationFunction