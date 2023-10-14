cmake_minimum_required(VERSION 3.25)
project(punica_ops_bench CUDA CXX)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)
set(CMAKE_CUDA_ARCHITECTURES native)

find_package(CUDAToolkit REQUIRED)
list(APPEND CMAKE_PREFIX_PATH "${CUDAToolkit_LIBRARY_DIR}/cmake/thrust")
find_package(Thrust REQUIRED CONFIG)
thrust_create_target(Thrust)


add_subdirectory(third_party/nvbench EXCLUDE_FROM_ALL)

add_subdirectory(benchmarks/nvbench)
