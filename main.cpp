#include <iostream>

#include "include/AsyncFileReader.h"
#include "include/config.h"

auto main() -> int {
  AsyncFileReader reader("../test.txt", CNN::DEFAULT_BATCH_SIZE,
                         CNN::DEFAULT_BUFFER_MAX_SIZE);

  while (true) {
    std::optional<std::vector<std::string>> read_data = reader.getLines();
    if (!read_data) {
      break;
    }
    for (const std::string &s : read_data.value()) {
      std::cout << s << std::endl;
    }
    std::cout << std::endl;
  }
  return 0;
}