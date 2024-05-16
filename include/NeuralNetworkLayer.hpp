#pragma once

#include <algorithm>
#include <type_traits>

#include "DataMatrix.hpp"
#include "NNActivationFunction.hpp"
#include "RandomFloatGenerate.hpp"
#include "ReLU.hpp"
#include "Sigmoid.hpp"
#include "config.h"

template <typename dtype = double>
class NeuralNetworkLayer {
  static_assert(std::is_floating_point<dtype>::value,
                "dtype must be a floating point type.");

 private:
  constexpr static NN::ActivationFunction::ReLU<dtype> relu =
      NN::ActivationFunction::ReLU<dtype>();  //<-----------------------
  //                                                                   |
  constexpr static NN::ActivationFunction::Sigmoid<dtype> sigmoid =  //|
      NN::ActivationFunction::Sigmoid<dtype>();  //<--------------------
  //                                                                   |
  NN::ActivationFunction::BaseActivationFunction<dtype> const  //      |
      *activation_function;  //----------------------------------------|

  constexpr static dtype DEFAULT_ALPHA = 0.01;

  dtype alpha = DEFAULT_ALPHA;

  DataMatrix<dtype> *data_layer;

  NeuralNetworkLayer<dtype> *prev_layer;

  NeuralNetworkLayer<dtype> *next_layer;

  std::vector<dtype> data;

  std::vector<std::vector<dtype>> w;

  std::vector<dtype> b;

 public:
  NeuralNetworkLayer() = default;

  NeuralNetworkLayer(std::size_t n, DataMatrix<dtype> *data_layer,
                     NeuralNetworkLayer<dtype> *prev_layer,
                     NeuralNetworkLayer<dtype> *next_layer,
                     const std::string &activation_function_type) noexcept
      : data_layer(data_layer), prev_layer(prev_layer), next_layer(next_layer) {
    /**
     * @brief
     *  static
     |    ReLU   | <-----------------------------
     ------------                               |
     |  Sigmoid  |  activation_function ---------
     */
    if (activation_function_type == "ReLU") {
      activation_function = &NeuralNetworkLayer<dtype>::relu;
    } else {
      /**
       * @brief
     *  static
     |    ReLU   | activation_function ----------
     ------------                               |
     |  Sigmoid  | <-----------------------------
     */
      activation_function = &NeuralNetworkLayer<dtype>::sigmoid;
    }

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
        data[i] = activation_function->apply(s + b[i], true);
      }
    } else {
      for (std::size_t i = 0; i < data.size(); i++) {
        dtype s = 0.0;
        for (std::size_t j = 0; j < w[i].size(); j++) {
          s += w[i][j] * ((*prev_layer)[j]);
        }
        data[i] = activation_function->apply(s + b[i], true);
      }
    }
    if (next_layer != nullptr) {
      next_layer->forward();
    }
  }

  auto backward(const std::vector<dtype> &labels) -> void {
    if (next_layer == nullptr) {
      for (std::size_t i = 0; i < data.size(); i++) {
        // update b
        // dloss/dy_hat = y_hat - y
        dtype d_loss_d_y_hat = data[i] - labels[i];
        // y_hat = f(w+b) => w1*a1+w2*a2...
        // dy_hat/db = f'(k+b)
        dtype w_and_b = 0.0;
        if (data_layer != nullptr) {
          for (std::size_t j = 0; j < w[i].size(); j++) {
            w_and_b += w[i][j] * ((*data_layer)[j]);
          }
        } else {
          for (std::size_t j = 0; j < w[i].size(); j++) {
            w_and_b += w[i][j] * ((*prev_layer)[j]);
          }
        }
        w_and_b += b[i];
        dtype dy_hat_db = activation_function->apply(w_and_b, false);
        b[i] -= alpha * d_loss_d_y_hat * dy_hat_db;

        for (std::size_t j = 0; j < w[i].size(); j++) {
          // update [w1,w2...,wn]
          // y_hat = f(w1 * a1 + w2 * a2 + ...)
          // dy_hat/dw = a1 * f'
          // a1 = image[i] or prev_layer[i]
          dtype dy_hat_dw = (data_layer != nullptr ? ((*data_layer)[j])
                                                   : ((*prev_layer)[j])) *
                            activation_function->apply(w_and_b, false);
          w[i][j] -= alpha * d_loss_d_y_hat * dy_hat_dw;
        }
      }
    }
  }

  [[nodiscard]] auto isOutput() const noexcept -> bool {
    return next_layer == nullptr;
  }

  [[nodiscard]] auto forecast() const noexcept -> std::size_t {
    return std::distance(data.cbegin(),
                         std::max_element(data.cbegin(), data.cend()));
  }

  [[nodiscard]] auto forecastData() const noexcept
      -> const std::vector<dtype> & {
    return data;
  }

  auto setDataLayer(DataMatrix<dtype> *data_layer) noexcept -> void {
    this->data_layer = data_layer;
  }

  auto setPrevLayer(NeuralNetworkLayer<dtype> *prev_layer) noexcept -> void {
    this->prev_layer = prev_layer;
  }

  auto setNextLayer(NeuralNetworkLayer<dtype> *next_layer) noexcept -> void {
    this->next_layer = next_layer;
  }

  auto setAlpha(dtype alpha) noexcept -> void { this->alpha = alpha; }
};