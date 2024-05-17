#pragma once

#include "AsyncFileReader.h"
#include "NeuralNetwork.hpp"

namespace MNIST {
template <typename dtype = double, std::size_t N = 2>
auto NeuralNetworkTrain(NeuralNetwork<dtype, N> &neural_network,
                        const std::string &data_value,
                        const std::vector<dtype> &labels) noexcept(false)
    -> void {
  neural_network.forward(data_value);
  neural_network.backward(labels);
}

template <typename dtype = double, std::size_t N = 2>
auto NeuralNetworkForecast(NeuralNetwork<dtype, N> &neural_network,
                           const std::string &data_value) noexcept(false)
    -> std::size_t {
  neural_network.forward(data_value);
  return neural_network.forecast();
}

template <typename dtype = double, std::size_t N = 2>
auto testNeuralNetwork(NeuralNetwork<dtype, N> &neural_network) -> void {
  long double loss = 0.0;
  long double right = 0.0;
  long double count = 0.0;
  AsyncFileReader test_data_set_reader = AsyncFileReader(
      MNIST::TEST_IMAGES, NN::DEFAULT_BATCH_SIZE, NN::DEFAULT_BUFFER_MAX_SIZE);

  AsyncFileReader test_label_set_read = AsyncFileReader(
      MNIST::TEST_LABELS, NN::DEFAULT_BATCH_SIZE, NN::DEFAULT_BUFFER_MAX_SIZE);

  while (true) {
    auto test_image = test_data_set_reader.getLines();
    auto test_labels = test_label_set_read.getLines();

    if (!test_image) {
      break;
    }

    for (std::size_t j = 0; j < test_image.value().size(); j++) {
      auto real_num = static_cast<std::size_t>(test_labels.value()[j][0] - '0');
      auto real_data = std::vector<long double>(MNIST::NUMBER_SIZE, 0.0);
      real_data[real_num] = 1;
      neural_network.forward(test_image.value()[j]);
      for (std::size_t k = 0; k < real_data.size(); k++) {
        loss += (neural_network[k] - real_data[k]) *
                (neural_network[k] - real_data[k]);
      }
      right += neural_network.forecast() == real_num ? 1 : 0;
      count++;
    }
  }
  std::cout << "test" << std::endl;
  std::cout << "loss: " << loss << std::endl;
  std::cout << "right/count:" << right / count << std::endl;
}

template <typename dtype = double, std::size_t N = 2>
auto trainNeuralNetwork(NeuralNetwork<dtype, N> &neural_network) -> void {
  AsyncFileReader train_data_set_reader = AsyncFileReader(
      MNIST::TRAIN_IMAGES, NN::DEFAULT_BATCH_SIZE, NN::DEFAULT_BUFFER_MAX_SIZE);
  AsyncFileReader train_label_set_read = AsyncFileReader(
      MNIST::TRAIN_LABELS, NN::DEFAULT_BATCH_SIZE, NN::DEFAULT_BUFFER_MAX_SIZE);

  while (true) {
    auto train_image = train_data_set_reader.getLines();
    auto train_labels = train_label_set_read.getLines();

    if (!train_image) {
      break;
    }
    for (std::size_t j = 0; j < train_image.value().size(); j++) {
      auto real_num =
          static_cast<std::size_t>(train_labels.value()[j][0] - '0');
      auto real_data = std::vector<long double>(MNIST::NUMBER_SIZE, 0.0);
      real_data[real_num] = 1;
      neural_network.forward(train_image.value()[j]);
      neural_network.backward(real_data);
    }
  }
}
}  // namespace MNIST