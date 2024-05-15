#include "../include/AsyncFileReader.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

AsyncFileReader::AsyncFileReader(std::string &&file_name,
                                 std::size_t batch_size,
                                 std::size_t max_buffer_size) noexcept
    : file_name(file_name),
      batch_size(batch_size),
      max_buffer_size(std::max(max_buffer_size, batch_size)),
      stop_flag(false) {
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

  // then wait and notify_all
  while (true) {
    // lock to read and write
    std::unique_lock<std::mutex> lk(mu);
    // check the stop_flag is true, if true(~AsyncFileReader has running, then
    // stop)
    if (stop_flag) {
      break;
    }
    // continue read the file until
    // 2. buffer size >= buffer_max_size
    cv.wait(lk, [this]() -> bool {
      return stop_flag || buffer.size() < max_buffer_size;
    });
    while (buffer.size() < max_buffer_size) {
      // 1. finish read the file
      if (!std::getline(file, line)) {
        stop_flag = true;
        break;
      }
      buffer.emplace(std::move(line));
      line = std::string();
    }
    cv.notify_all();
  }
}

[[nodiscard]] auto AsyncFileReader::getLines()
    -> std::optional<std::vector<std::string>> {
  std::unique_lock<std::mutex> lk(mu);
  // if the file has been finished reading and the buffer is empty, return null;
  if (stop_flag && buffer.empty()) {
    cv.notify_all();
    return std::nullopt;
  }
  // wait until
  // 1. file has been finished
  // 2. buffer size >= batch size
  cv.wait(lk, [this]() -> bool {
    return stop_flag || buffer.size() >= batch_size;
  });
  // read buffer data
  auto ans = std::vector<std::string>();
  std::size_t read_count = std::min(batch_size, buffer.size());
  for (std::size_t i = 0; i < read_count; i++) {
    ans.emplace_back(buffer.front());
    buffer.pop();
  }
  // notify the read_thread
  cv.notify_all();
  return std::make_optional(ans);
}