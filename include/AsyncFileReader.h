#pragma once

#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <optional>
#include <queue>
#include <string>
#include <thread>

/**
  @brief Async read file, solve the OOM problem
*/
class AsyncFileReader {
 private:
  // The file name
  std::string file_name;

  // work thread to async read the file
  std::thread worker_thread;

  // lock to protect the file data
  std::mutex mu;

  // buffer to save the file data
  //                             buffer
  // other thread: read <- [data1,data2,....] <- worker thread
  std::queue<std::string> buffer;  //<----------------------------|
  //                                                              |
  // max buffer size ---------------------------------------------|
  std::size_t max_buffer_size;

  // condition value
  std::condition_variable cv;  //

  // flag to express if the file has been finished reading
  bool stop_flag;

  // the size that every read
  std::size_t batch_size;

  auto readFile() -> void;

 public:
  AsyncFileReader() = delete;

  /**
   * @param file_name read file name
   * @param batch_size every read size
   * @param max_buffer_size the buffer to save file data max size, if it <=
   * batch_size, it will be batch_size
   */
  explicit AsyncFileReader(std::string &&file_name, std::size_t batch_size,
                           std::size_t max_buffer_size) noexcept;
  /**
   * @brief async read the file data
   * @return if has been finished reading file, return null
   * @return else return the batch size file data
   */
  [[nodiscard]] auto getLines() -> std::optional<std::vector<std::string>>;

  ~AsyncFileReader();
};