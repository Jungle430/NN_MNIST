#pragma once

#include <cmath>
#include <type_traits>

namespace NN::ActivationFunction {
template <typename dtype = double>
struct Sigmoid {
  static_assert(std::is_floating_point<dtype>::value,
                "dtype must be a floating point type.");

  [[nodiscard]] auto operator()(dtype x,
                                bool differential = true) const noexcept
      -> dtype {
    return differential ? 1 / (1 + std::exp(-x))
                        : std::exp(-x) / std::pow((1 + std::exp(-x)), 2);
  };
};
}  // namespace NN::ActivationFunction