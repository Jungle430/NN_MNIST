#include <memory>

#include "include/AsyncFileReader.h"
#include "include/DataMatrix.hpp"
#include "include/NNActivationFunction.hpp"
#include "include/NeuralNetworkLayer.hpp"
#include "include/ReLU.hpp"
#include "include/Sigmoid.hpp"
#include "include/config.h"

auto main() -> int {
  AsyncFileReader train_data_set_reader = AsyncFileReader(
      MNIST::TRAIN_IMAGES, NN::DEFAULT_BATCH_SIZE, NN::DEFAULT_BUFFER_MAX_SIZE);
  AsyncFileReader train_label_set_read = AsyncFileReader(
      MNIST::TRAIN_LABELS, NN::DEFAULT_BATCH_SIZE, NN::DEFAULT_BUFFER_MAX_SIZE);
  double right = 0.0;
  double count = 0.0;
  while (true) {
    auto train_set_read = train_data_set_reader.getLines();
    auto train_label_read = train_label_set_read.getLines();
    if (!train_set_read) {
      break;
    }
    DataMatrix<double> dm(train_set_read.value()[0]);
    NeuralNetworkLayer<double> nn1(NN::DEFAULT_NODE_SIZE, &dm, nullptr, nullptr,
                                   "ReLU");
    NeuralNetworkLayer<double> nn2(NN::DEFAULT_NODE_SIZE, nullptr, &nn1,
                                   nullptr, "ReLU");
    NeuralNetworkLayer<double> nn3(MNIST::NUMBER_SIZE, nullptr, &nn2, nullptr,
                                   "ReLU");
    nn1.setNextLayer(&nn2);
    nn2.setNextLayer(&nn3);
    for (std::size_t i = 0; i < train_set_read.value().size(); i++) {
      dm = DataMatrix<double>(train_set_read.value()[i]);
      nn1.setDataLayer(&dm);
      nn1.forward();
      std::cout << nn3.forecast() << " " << train_label_read.value()[i]
                << std::endl;
      if (nn3.forecast() ==
          static_cast<std::size_t>(train_label_read.value()[i][0] - '0')) {
        right++;
      }
      count++;
    }
  }
  std::cout << right / count << std::endl;
  return 0;
}