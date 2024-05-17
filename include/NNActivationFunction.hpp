#pragma once

#include <stdexcept>
#include <type_traits>

namespace NN::ActivationFunction {
template <typename dtype = double>
class BaseActivationFunction {
  static_assert(std::is_floating_point<dtype>::value,
                "dtype must be a floating point type.");

 public:
  /**
   * @throw if it's not override, it will throw `std::logic_error`
   */
  [[nodiscard]] virtual auto apply(dtype /*unused*/, bool /*unused*/) const
      noexcept(false) -> dtype {
    throw std::logic_error("is the virtual function");
  };

  [[nodiscard]] virtual auto differentialByCurrentValue(
      dtype /*current_value*/) const noexcept(false) -> dtype {
    throw std::logic_error("is the virtual function");
  }
};
}  // namespace NN::ActivationFunction