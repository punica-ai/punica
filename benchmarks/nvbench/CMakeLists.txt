set(bench_srcs
    sgmv.cu
)

foreach(bench_src IN LISTS bench_srcs)
    get_filename_component(bench_name "${bench_src}" NAME_WLE)
    string(PREPEND bench_name "nvbench_")
    add_executable(${bench_name} "${bench_src}")
    target_include_directories(${bench_name} PUBLIC ${PROJECT_SOURCE_DIR}/punica/ops/csrc)
    target_include_directories(${bench_name} PUBLIC ${PROJECT_SOURCE_DIR}/third_party/nvbench)
    target_include_directories(${bench_name} PUBLIC ${PROJECT_SOURCE_DIR}/third_party/cutlass/include)
    target_include_directories(${bench_name} PUBLIC ${PROJECT_SOURCE_DIR}/third_party/cutlass/tools/util/include)
    target_link_libraries(${bench_name} PUBLIC nvbench::main Thrust)
    set_target_properties(${bench_name} PROPERTIES CXX_STANDARD 17)
    set_target_properties(${bench_name} PROPERTIES CUDA_STANDARD 17)
endforeach()
