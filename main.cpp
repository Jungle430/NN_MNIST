#include "include/AsyncFileReader.h"
#include "include/DataMatrix.hpp"
#include "include/NeuralNetworkLayer.hpp"
#include "include/ReLU.hpp"
#include "include/Sigmoid.hpp"
#include "include/config.h"

auto main() -> int {
  AsyncFileReader async_reader = AsyncFileReader(
      MNIST::TRAIN_IMAGES, NN::DEFAULT_BATCH_SIZE, NN::DEFAULT_BUFFER_MAX_SIZE);
  auto read = async_reader.getLines();
  DataMatrix<double> dm(read.value()[0]);
  NeuralNetworkLayer<double> nn(16, true, &dm, true, nullptr);

  RandomFloatGenerate<double> rm(NN::RANDOM_PARAM_MIN, NN::RANDOM_PARAM_MAX);

  for (std::size_t i = 0; i < 10; i++) {
    double num = rm();
    auto relu = NN::ActivationFunction::ReLU<double>();
    auto sigmoid = NN::ActivationFunction::Sigmoid<double>();
    std::cout << num << " " << relu(num) << " " << sigmoid(num) << std::endl;
  }

  return 0;
}