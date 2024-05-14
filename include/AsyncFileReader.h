#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <queue>
#include <string>
#include <thread>

class AsyncFileReader {
 private:
  constexpr static std::size_t READ_SLEEP_INTERVAL_TIME = 1;

  std::string file_name;

  std::thread worker_thread;

  std::mutex mu;

  std::queue<std::string> lines_queue;

  std::condition_variable cv;

  bool stop_flag;

  auto readFile() -> void;

 public:
  explicit AsyncFileReader(std::string &&file_name);

  auto getLines(std::size_t num_lines) -> std::vector<std::string>;

  ~AsyncFileReader();
};