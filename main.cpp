#include <cmath>

#include "include/AsyncFileReader.h"
#include "include/MNIST.hpp"
#include "include/NeuralNetwork.hpp"
#include "spdlog/spdlog.h"

auto main() -> int {
  NeuralNetwork<long double, 4> neural_network(
      MNIST::DOMAIN, {{NN::DEFAULT_NODE_SIZE, "Sigmoid"},
                      {NN::DEFAULT_NODE_SIZE, "Sigmoid"},
                      {NN::DEFAULT_NODE_SIZE, "Sigmoid"},
                      {MNIST::NUMBER_SIZE, "Sigmoid"}});

  MNIST::testNeuralNetwork(neural_network);

  spdlog::info("begin training...");

  for (std::size_t i = 0; i < MNIST::DEFAULT_TRAIN_COUNT; i++) {
    spdlog::info("The {:d}st, train", i + 1);
    MNIST::trainNeuralNetwork(neural_network);
    MNIST::testNeuralNetwork(neural_network);
  }
  return 0;
}