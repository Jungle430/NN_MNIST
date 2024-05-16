#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <iterator>
#include <stdexcept>
#include <type_traits>

#include "DataMatrix.hpp"
#include "NNActivationFunction.h"
#include "RandomFloatGenerate.hpp"
#include "config.h"

template <typename dtype = double,
          typename ActivationFunction = NN::ActivationFunction::Sigmoid<dtype>>
class NeuralNetworkLayer {
  static_assert(
      NN::ActivationFunction::is_ActivationFunction<ActivationFunction>::value,
      "Function must be network activation function.");

  static_assert(std::is_floating_point<dtype>::value,
                "dtype must be a floating point type.");

 private:
  DataMatrix<dtype> *data_layer;

  NeuralNetworkLayer<dtype, ActivationFunction> *prev_layer;

  NeuralNetworkLayer<dtype, ActivationFunction> *next_layer;

  std::vector<dtype> data;

  std::vector<std::vector<dtype>> w;

  std::vector<dtype> b;

  const ActivationFunction activation_function;

 public:
  NeuralNetworkLayer() = default;

  NeuralNetworkLayer(std::size_t n, DataMatrix<dtype> *data_layer,
                     NeuralNetworkLayer<dtype, ActivationFunction> *prev_layer,
                     NeuralNetworkLayer<dtype, ActivationFunction> *next_layer,
                     ActivationFunction activation_function) noexcept
      : data_layer(data_layer),
        prev_layer(prev_layer),
        next_layer(next_layer),
        activation_function(activation_function) {
    data = std::vector<dtype>(n, 0.0);
    std::size_t last_layer_domain;
    if (data_layer != nullptr) {
      last_layer_domain = data_layer->size();
    } else {
      last_layer_domain = prev_layer->size();
    }

    RandomFloatGenerate<dtype> randomFloatGenerate(
        static_cast<dtype>(NN::RANDOM_PARAM_MIN),
        static_cast<dtype>(NN::RANDOM_PARAM_MAX));
    w = std::vector<std::vector<dtype>>(data.size(), std::vector<dtype>());
    for (std::size_t i = 0; i < data.size(); i++) {
      w[i] = std::vector<dtype>(last_layer_domain, 0.0);
      for (std::size_t j = 0; j < last_layer_domain; j++) {
        w[i][j] = randomFloatGenerate();
      }
    }
    b = std::vector<dtype>(data.size(), 0.0);
    for (dtype &num : b) {
      num = randomFloatGenerate();
    }
  }

  [[nodiscard]] auto size() const noexcept -> std::size_t {
    return data.size();
  }

  auto operator[](std::size_t i) -> dtype & { return data[i]; }

  auto forward() -> void {
    if (data_layer != nullptr) {
      for (std::size_t i = 0; i < data.size(); i++) {
        dtype s = 0.0;
        for (std::size_t j = 0; j < w[i].size(); j++) {
          s += w[i][j] * ((*data_layer)[j]);
        }
        data[i] = activation_function(s + b[i]);
      }
    } else {
      for (std::size_t i = 0; i < data.size(); i++) {
        dtype s = 0.0;
        for (std::size_t j = 0; j < w[i].size(); j++) {
          s += w[i][j] * ((*prev_layer)[j]);
        }
        data[i] = activation_function(s + b[i]);
      }
    }
    if (next_layer != nullptr) {
      next_layer->forward();
    }
  }

  [[nodiscard]] auto isOutput() const noexcept -> bool {
    return next_layer == nullptr;
  }

  [[nodiscard]] auto forecast() const noexcept -> std::size_t {
    return std::distance(data.cbegin(),
                         std::max_element(data.cbegin(), data.cend()));
  }

  auto setDataLayer(DataMatrix<dtype> *data_layer) noexcept -> void {
    this->data_layer = data_layer;
  }

  auto setPrevLayer(
      NeuralNetworkLayer<dtype, ActivationFunction> *prev_layer) noexcept
      -> void {
    this->prev_layer = prev_layer;
  }

  auto setNextLayer(
      NeuralNetworkLayer<dtype, ActivationFunction> *next_layer) noexcept
      -> void {
    this->next_layer = next_layer;
  }
};