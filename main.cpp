#include <cmath>

#include "include/AsyncFileReader.h"
#include "include/MNIST.hpp"
#include "include/NeuralNetwork.hpp"
#include "spdlog/spdlog.h"

auto main() -> int {
  NeuralNetwork<long double, 2> neural_network(
      MNIST::DOMAIN, {{10, "Sigmoid"},
                      
                      {MNIST::NUMBER_SIZE, "Sigmoid"}});

  MNIST::testNeuralNetwork(neural_network);

  spdlog::info("begin training...");

  for (std::size_t i = 0; i < MNIST::DEFAULT_TRAIN_COUNT; i++) {
    std::cout << "The " << i + 1 << "st, train" << std::endl;
    MNIST::trainNeuralNetwork(neural_network);
    MNIST::testNeuralNetwork(neural_network);
  }
  return 0;
}