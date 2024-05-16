#pragma once

#include <cstddef>

namespace MNIST {
constexpr const char* TRAIN_IMAGES = "../mnist_data/train_images.csv";

constexpr const char* TRAIN_LABELS = "../mnist_data/train_labels.csv";

constexpr const char* TEST_IMAGES = "../mnist_data/test_images.csv";

constexpr const char* TEST_LABELS = "../mnist_data/test_labels.csv";

constexpr const std::size_t NUMBER_SIZE = 10;
}  // namespace MNIST

// param for CNN
namespace CNN {
constexpr std::size_t DEFAULT_BATCH_SIZE = 1000;

constexpr std::size_t DEFAULT_BUFFER_MAX_SIZE = 2000;
}  // namespace CNN

// param for NN
namespace NN {
constexpr std::size_t DEFAULT_BATCH_SIZE = 1000;

constexpr std::size_t DEFAULT_BUFFER_MAX_SIZE = 2000;

constexpr double RANDOM_PARAM_MIN = -1.0;

constexpr double RANDOM_PARAM_MAX = 1.0;

constexpr std::size_t DEFAULT_NODE_SIZE = 16;
}  // namespace NN