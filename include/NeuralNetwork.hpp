#pragma once

#include <array>
#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "DataMatrix.hpp"
#include "NeuralNetworkLayer.hpp"
#include "spdlog/fmt/bundled/core.h"

template <typename dtype = double, std::size_t N = 2>
class NeuralNetwork {
  static_assert(N >= 2, "layers size must >= 2");

  static_assert(std::is_floating_point<dtype>::value,
                "dtype must be a floating point type.");

 private:
  std::array<NeuralNetworkLayer<dtype>, N> neural_network_layers;

  DataMatrix<dtype> data_matrix;

 public:
  NeuralNetwork(std::size_t data_matrix_domain,
                std::initializer_list<std::pair<std::size_t, std::string>>
                    params) noexcept(false)
      : neural_network_layers(std::array<NeuralNetworkLayer<dtype>, N>()),
        data_matrix(DataMatrix<dtype>(data_matrix_domain)) {
    if (params.size() != N) {
      throw std::invalid_argument("param.size() != neural_network_layers");
    }
    for (std::size_t i = 0; i < N; i++) {
      if (i == 0) {
        neural_network_layers[i] = NeuralNetworkLayer<dtype>(
            (params.begin() + i)->first, &data_matrix, nullptr, nullptr,
            (params.begin() + i)->second);
      } else {
        neural_network_layers[i] = NeuralNetworkLayer<dtype>(
            (params.begin() + i)->first, nullptr, &neural_network_layers[i - 1],
            nullptr, (params.begin() + i)->second);
        neural_network_layers[i - 1].setNextLayer(&neural_network_layers[i]);
      }
    }
  }

  auto forward(const std::string &data_value) noexcept(false) -> void {
    data_matrix.setData(data_value);
    neural_network_layers[0].forward();
  }

  auto backward(const std::vector<dtype> &labels) -> void {
    neural_network_layers[N - 1].backward(labels);
  }

  auto operator[](std::size_t i) const -> dtype {
    return neural_network_layers[N - 1][i];
  }

  [[nodiscard]] auto forecast() const noexcept -> std::size_t {
    return neural_network_layers[N - 1].forecast();
  }

  [[nodiscard]] constexpr inline auto size() const noexcept -> std::size_t {
    return N;
  }

  [[nodiscard]] auto toString() const noexcept -> std::string {
    std::string ans("The NeuralNetwork:\n");

    ans += fmt::format("\tThe input layer size {}\n", data_matrix.size());

    ans += "\tThe hidden layer: \n";
    for (std::size_t i = 0; i < N - 1; i++) {
      ans += fmt::format("\t\tThe hidden {}, size: {}\n", i + 1,
                         neural_network_layers[i].size());
    }

    ans += fmt::format("\tThe output layer size: {}",
                       neural_network_layers[N - 1].size());

    return ans;
  }
};