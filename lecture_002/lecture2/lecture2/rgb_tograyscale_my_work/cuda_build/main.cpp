#include <torch/extension.h>
torch::Tensor rgb_to_grayscale(torch::Tensor image);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("rgb_to_grayscale", torch::wrap_pybind_function(rgb_to_grayscale), "rgb_to_grayscale");
}