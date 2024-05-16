#include "include/AsyncFileReader.h"
#include "include/DataMatrix.hpp"
#include "include/NNActivationFunction.hpp"
#include "include/NeuralNetworkLayer.hpp"
#include "include/ReLU.hpp"
#include "include/Sigmoid.hpp"
#include "include/config.h"

auto main() -> int {
  DataMatrix<long double> dm(MNIST::DOMAIN);
  NeuralNetworkLayer<long double> nn1(NN::DEFAULT_NODE_SIZE, &dm, nullptr,
                                      nullptr, "Sigmoid");
  NeuralNetworkLayer<long double> nn2(MNIST::NUMBER_SIZE, nullptr, &nn1,
                                      nullptr, "Sigmoid");

  nn1.setNextLayer(&nn2);

  {
    long double loss = 0.0;
    long double right = 0.0;
    long double count = 0.0;
    AsyncFileReader test_data_set_reader =
        AsyncFileReader(MNIST::TEST_IMAGES, NN::DEFAULT_BATCH_SIZE,
                        NN::DEFAULT_BUFFER_MAX_SIZE);

    AsyncFileReader test_label_set_read =
        AsyncFileReader(MNIST::TEST_LABELS, NN::DEFAULT_BATCH_SIZE,
                        NN::DEFAULT_BUFFER_MAX_SIZE);
    while (true) {
      auto test_image = test_data_set_reader.getLines();
      auto test_labels = test_label_set_read.getLines();

      if (!test_image) {
        break;
      }

      for (std::size_t j = 0; j < test_image.value().size(); j++) {
        auto real_num =
            static_cast<std::size_t>(test_labels.value()[j][0] - '0');
        auto real_data = std::vector<long double>(MNIST::NUMBER_SIZE, 0.0);
        real_data[real_num] = 1;
        dm = DataMatrix<long double>(test_image.value()[j]);
        nn1.setDataLayer(&dm);
        nn1.forward();
        for (std::size_t k = 0; k < real_data.size(); k++) {
          // std::cout << nn2[k] << " ";

          loss += (nn2[k] - real_data[k]) * (nn2[k] - real_data[k]);
        }

        // std::cout << nn2.forecast() << " " << real_num << std::endl;

        right += nn2.forecast() == real_num ? 1 : 0;
        count++;
      }
    }
    std::cout << "when no training..." << std::endl;
    std::cout << loss << std::endl;
    std::cout << right / count << std::endl;
  }

  std::cout << "begin training..." << std::endl;

  for (std::size_t i = 0; i < 10; i++) {
    long double loss = 0.0;
    long double right = 0.0;
    long double count = 0.0;
    {
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
          auto real_data = std::vector<long double>(MNIST::NUMBER_SIZE, 0.0);
          real_data[real_num] = 1;
          dm = DataMatrix<long double>(train_image.value()[j]);
          nn1.setDataLayer(&dm);
          nn1.forward();
          nn2.backward(real_data);
        }
      }
    }

    {
      AsyncFileReader test_data_set_reader =
          AsyncFileReader(MNIST::TEST_IMAGES, NN::DEFAULT_BATCH_SIZE,
                          NN::DEFAULT_BUFFER_MAX_SIZE);

      AsyncFileReader test_label_set_read =
          AsyncFileReader(MNIST::TEST_LABELS, NN::DEFAULT_BATCH_SIZE,
                          NN::DEFAULT_BUFFER_MAX_SIZE);
      while (true) {
        auto test_image = test_data_set_reader.getLines();
        auto test_labels = test_label_set_read.getLines();

        if (!test_image) {
          break;
        }

        for (std::size_t j = 0; j < test_image.value().size(); j++) {
          auto real_num =
              static_cast<std::size_t>(test_labels.value()[j][0] - '0');
          auto real_data = std::vector<long double>(MNIST::NUMBER_SIZE, 0.0);
          real_data[real_num] = 1;
          dm = DataMatrix<long double>(test_image.value()[j]);
          nn1.setDataLayer(&dm);
          nn1.forward();
          for (std::size_t k = 0; k < real_data.size(); k++) {
            if (i == 9) {
              std::cout << nn2[k] << " ";
            }
            loss += (nn2[k] - real_data[k]) * (nn2[k] - real_data[k]);
          }
          if (i == 9) {
            std::cout << nn2.forecast() << " " << real_num << std::endl;
          }
          right += nn2.forecast() == real_num ? 1 : 0;
          count++;
        }
      }
    }

    std::cout << loss << std::endl;
    std::cout << right / count << std::endl;
  }
  return 0;
}