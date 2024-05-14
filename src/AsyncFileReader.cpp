#include "../include/AsyncFileReader.h"

#include <chrono>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>

AsyncFileReader::AsyncFileReader(std::string &&file_name)
    : file_name(file_name), stop_flag(false) {
  worker_thread = std::thread(&AsyncFileReader::readFile, this);
}

AsyncFileReader::~AsyncFileReader() {
  {
    std::lock_guard<std::mutex> lk(mu);
    stop_flag = true;
  }
  cv.notify_all();
  worker_thread.join();
}

auto AsyncFileReader::readFile() -> void {
  std::ifstream file(file_name);
  std::string line;

  while (std::getline(file, line)) {
    {
      std::lock_guard<std::mutex> lk(mu);
      lines_queue.push(line);
    }
    cv.notify_all();
    std::this_thread::sleep_for(
        std::chrono::microseconds(READ_SLEEP_INTERVAL_TIME));
  }
  {
    std::lock_guard<std::mutex> lk(mu);
    stop_flag = true;
  }
  cv.notify_all();
}

auto AsyncFileReader::getLines(std::size_t num_lines)
    -> std::vector<std::string> {
  std::vector<std::string> lines;
  std::unique_lock<std::mutex> lk(mu);
  cv.wait(lk, [this, num_lines]() -> bool {
    return stop_flag || lines_queue.size() >= num_lines;
  });
  for (size_t i = 0; i < num_lines && !lines_queue.empty(); ++i) {
    lines.push_back(lines_queue.front());
    lines_queue.pop();
  }

  return lines;
}