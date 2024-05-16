#pragma once

#include "ReLU.hpp"
#include "Sigmoid.hpp"

namespace NN::ActivationFunction {
template <class Func>
struct is_ActivationFunction {
  constexpr static bool value = false;
};

template <typename dtype>
struct is_ActivationFunction<ReLU<dtype>> {
  constexpr static bool value = true;
};

template <typename dtype>
struct is_ActivationFunction<Sigmoid<dtype>> {
  constexpr static bool value = true;
};
}  // namespace NN::ActivationFunction