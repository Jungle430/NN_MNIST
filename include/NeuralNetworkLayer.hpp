#pragma once

#include <array>
#include <cstddef>
#include <stdexcept>
#include <type_traits>

#include "DataMatrix.hpp"
#include "RandomFloatGenerate.hpp"
#include "config.h"

template <typename dtype>
class NeuralNetworkLayer {
 private:
  static_assert(std::is_floating_point<dtype>::value,
                "dtype must be a floating point type.");

  bool is_data_layer_next;

  DataMatrix<dtype> const *data_layer;

  bool is_last;

  std::vector<dtype> data;

  std::vector<std::vector<dtype>> w;

  std::vector<dtype> b;

  NeuralNetworkLayer<dtype> const *front_layer;

 public:
  NeuralNetworkLayer() = default;

  NeuralNetworkLayer(std::size_t n, bool is_last, DataMatrix<dtype> *data_layer,
                     bool is_data_layer_next,
                     NeuralNetworkLayer<dtype> *front_layer)
      : is_data_layer_next(is_data_layer_next),
        data_layer(data_layer),
        is_last(is_last),
        front_layer(front_layer) {
    if ((is_data_layer_next && data_layer == nullptr) ||
        (!is_data_layer_next && front_layer == nullptr)) {
      throw std::invalid_argument("front layer can't be null");
    }

    data = std::vector<dtype>(n, 0.0);
    std::size_t last_layer_domain;
    if (is_data_layer_next) {
      last_layer_domain = data_layer->size();
    } else {
      last_layer_domain = front_layer->size();
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
};