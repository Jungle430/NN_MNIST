cmake_minimum_required(VERSION 3.21)
project(NN_MNIST VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/spdlog/include)

add_subdirectory(src)
add_executable(NN_MNIST main.cpp)
target_link_libraries(${PROJECT_NAME} SRC)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})