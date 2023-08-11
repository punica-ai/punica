#include "ft_llama.h"
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    py::class_<FtLlama>(m, "FtLlama")
        .def(py::init<size_t, size_t, size_t, size_t, const char*, int>(),
             py::arg("num_heads"),
             py::arg("head_dim"),
             py::arg("inter_size"),
             py::arg("num_layers"),
             py::arg("dtype"),
             py::arg("device_id") = 0)
        .def("forward",
             &FtLlama::forward,
             py::arg("input_ids"),
             py::arg("request_output_len"),
             py::arg("callback") = nullptr);
}
