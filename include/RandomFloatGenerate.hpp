#pragma once

#include <random>
#include <stdexcept>
#include <type_traits>

template <typename dtype = double>
class RandomFloatGenerate {
  static_assert(std::is_floating_point<dtype>::value,
                "dtype must be a floating point type.");

 private:
  dtype min_;

  dtype max_;

  std::random_device rd;

  std::mt19937 gen;

  std::uniform_real_distribution<dtype> custom_dis;

 public:
  /**
   * @throw if min > max, throw `std::invalid_argument`
   */
  RandomFloatGenerate(dtype min, dtype max) : rd(), gen(rd()) {
    if (min > max) {
      throw std::invalid_argument(
          "Invalid range: min must be less than or equal to max");
    }

    min_ = min;
    max_ = max;
    custom_dis = std::uniform_real_distribution<dtype>(min_, max_);
  }

  [[nodiscard]] auto operator()() -> dtype { return custom_dis(gen); }
};