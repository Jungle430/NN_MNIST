#include <iostream>

#include "include/AsyncFileReader.h"

auto main() -> int {
  AsyncFileReader fileReader("../test.txt");

  while (true) {
    auto lines = fileReader.getLines(5);
    if (lines.empty()) {
      break;
    }
    for (const auto &line : lines) {
      std::cout << line << std::endl;
    }
  }

  return 0;
}