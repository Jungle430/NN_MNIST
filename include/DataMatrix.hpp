#pragma once

#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

template <typename dtype = double>
class DataMatrix {
  static_assert(std::is_floating_point<dtype>::value,
                "dtype must be a floating point type.");

 private:
  constexpr static dtype IMAGE_VALUE_MAX = 255.0;

  constexpr static std::size_t DEFAULT_INIT_CAP = 100;

  std::vector<dtype> data;

 public:
  DataMatrix() = default;

  explicit DataMatrix(const std::string &data_value,
                      std::size_t init_cap = DEFAULT_INIT_CAP)
      : data(std::vector<dtype>()) {
    data.reserve(init_cap);

    std::stringstream ss(data_value);
    std::string token;
    while (std::getline(ss, token, ',')) {
      auto pixelValue = static_cast<dtype>(std::stod(token));
      pixelValue /= IMAGE_VALUE_MAX;
      data.emplace_back(pixelValue);
    }
  }

  explicit DataMatrix(std::size_t size,
                      std::size_t init_cap = DEFAULT_INIT_CAP) noexcept
      : data(std::vector<dtype>()) {
    data.reserve(std::max(size, init_cap));
    data.resize(size);
  }

  auto setData(const std::string &data_value) -> void {
    std::vector<dtype> new_data;
    std::stringstream ss(data_value);
    std::string token;
    while (std::getline(ss, token, ',')) {
      auto pixelValue = static_cast<dtype>(std::stod(token));
      pixelValue /= IMAGE_VALUE_MAX;
      new_data.emplace_back(pixelValue);
    }
    if (new_data.size() != data.size()) {
      throw std::logic_error("new data size != old data size");
    }
    data = new_data;
  }

  auto operator[](std::size_t i) -> dtype & { return data[i]; }

  [[nodiscard]] auto size() const noexcept -> std::size_t {
    return data.size();
  }

  [[nodiscard]] auto toString() const noexcept -> std::string {
    auto ans = std::string("[");
    for (const dtype &num : data) {
      ans += std::to_string(num);
      ans += ',';
    }
    if (ans.size() != 1) {
      ans[ans.size() - 1] = ']';
    } else {
      ans += ']';
    }
    return ans;
  }
};