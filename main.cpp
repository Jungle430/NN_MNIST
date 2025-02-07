#include "include/MNIST.hpp"
#include "include/NeuralNetwork.hpp"
#include "spdlog/spdlog.h"

auto main() -> int {
  NeuralNetwork<long double, 5> neural_network(
      MNIST::IMAGE_DOMAIN, {{NN::DEFAULT_NODE_SIZE, "Sigmoid"},
                            {NN::DEFAULT_NODE_SIZE >> 1, "Sigmoid"},
                            {NN::DEFAULT_NODE_SIZE >> 2, "Sigmoid"},
                            {NN::DEFAULT_NODE_SIZE >> 3, "Sigmoid"},
                            {MNIST::NUMBER_SIZE, "Sigmoid"}});

  spdlog::info("The neural network scale:\n" + neural_network.toString());

  MNIST::testNeuralNetwork(neural_network);

  spdlog::info("begin training...");
  for (std::size_t i = 0; i < MNIST::DEFAULT_TRAIN_COUNT; i++) {
    spdlog::info("The {:d}st, train", i + 1);
    MNIST::trainNeuralNetwork(neural_network);
    MNIST::testNeuralNetwork(neural_network);
  }
  return 0;
}