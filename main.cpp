#include <memory>

#include "include/AsyncFileReader.h"
#include "include/DataMatrix.hpp"
#include "include/NNActivationFunction.hpp"
#include "include/NeuralNetworkLayer.hpp"
#include "include/ReLU.hpp"
#include "include/Sigmoid.hpp"
#include "include/config.h"

auto main() -> int {
  DataMatrix<double> dm(MNIST::DOMAIN);
  NeuralNetworkLayer<double> nn1(NN::DEFAULT_NODE_SIZE, &dm, nullptr, nullptr,
                                 "Sigmoid");
  NeuralNetworkLayer<double> nn2(MNIST::NUMBER_SIZE, nullptr, &nn2, nullptr,
                                 "Sigmoid");

  nn1.setNextLayer(&nn2);

  for (std::size_t i = 0; i < 10; i++) {
    double loss = 0.0;
    double right = 0.0;
    double count = 0.0;

    AsyncFileReader train_data_set_reader =
        AsyncFileReader(MNIST::TRAIN_IMAGES, NN::DEFAULT_BATCH_SIZE,
                        NN::DEFAULT_BUFFER_MAX_SIZE);
    AsyncFileReader train_label_set_read =
        AsyncFileReader(MNIST::TRAIN_LABELS, NN::DEFAULT_BATCH_SIZE,
                        NN::DEFAULT_BUFFER_MAX_SIZE);

    while (true) {
      auto train_image = train_data_set_reader.getLines();
      auto train_labels = train_label_set_read.getLines();

      if (!train_image) {
        break;
      }
      for (std::size_t j = 0; j < train_image.value().size(); j++) {
        auto real_num =
            static_cast<std::size_t>(train_labels.value()[j][0] - '0');
        auto real_data = std::vector<double>(MNIST::NUMBER_SIZE, 0.0);
        real_data[real_num] = 1;
        dm = DataMatrix<double>(train_image.value()[j]);
        nn1.setDataLayer(&dm);
        nn1.forward();
        right += nn2.forecast() == real_num ? 1 : 0;
        count++;
        for (std::size_t k = 0; k < MNIST::NUMBER_SIZE; k++) {
          loss += nn2[i] - real_data[i];
        }
        nn2.backward(real_data);
      }
    }
    std::cout << loss << std::endl;
    std::cout << right / count << std::endl;
  }
  return 0;
}