cmake_minimum_required(VERSION 3.12)
project(ft LANGUAGES CXX)
include(FetchContent)

# FasterTransformer
FetchContent_Declare(
    fastertransformer
    GIT_REPOSITORY https://github.com/void-main/FasterTransformer.git
    GIT_TAG        e770ddf2bc66217034b6e9e3b0c3256ebf1c1b40
)
FetchContent_MakeAvailable(fastertransformer)
cmake_policy(SET CMP0079 NEW)
target_link_libraries(cuda_driver_wrapper PUBLIC -lcublas -lcudart -ldl)
find_package(CUDA REQUIRED)

# ft
add_library(ft SHARED ft_llama.cc)
target_include_directories(ft PUBLIC
    ${CUDA_TOOLKIT_ROOT_DIR}/include
    ${FasterTransformer_SOURCE_DIR}
    ${FasterTransformer_SOURCE_DIR}/3rdparty/cutlass/include
)
target_link_directories(ft PUBLIC
    ${CUDA_TOOLKIT_ROOT_DIR}/lib64
)
target_link_libraries(ft PUBLIC
    -lcublas -lcublasLt -lcudart
    Llama
)
