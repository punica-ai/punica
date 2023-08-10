#include "ft_llama.h"
#include <pybind11/pybind11.h>

namespace {
void run()
{
    srand(0);
    const char* data_type = "float16";

    FtLlama llama(32,     // num_heads
                  128,    // head_dim
                  11008,  // inter_size
                  32,     // num_layers
                  data_type);

    std::vector<std::vector<int>> input_ids = {
        {0, 37, 92, 26, 66, 36, 55, 70, 73, 15, 36, 51, 34, 52, 29},
        {0, 37, 92, 70, 73, 15, 66, 93, 34, 52, 99},
        {0, 92, 16, 66, 16, 45, 70, 93, 11, 36, 53, 30, 52, 29},
        {0, 37, 92, 26, 66, 36, 55, 70, 23, 23},
        {0, 37, 92, 26, 66, 36, 55, 70, 29, 15, 34, 52, 23},
        {0, 73},
        {0, 37, 92, 15, 66, 93, 34, 52, 23},
        {0, 70, 73, 15, 66, 93, 30, 92, 29},
    };
    llama.forward(input_ids, 10);
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("run", &run);
}
