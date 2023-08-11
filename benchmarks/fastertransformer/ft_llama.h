#include <cstddef>
#include <functional>
#include <vector>

class FtLlama {
    enum class DataType
    {
        FP32,
        FP16,
        BF16
    } dtype_;
    void* impl_;

public:
    FtLlama(
        size_t num_heads, size_t head_dim, size_t inter_size, size_t num_layers, const char* data_type, int device_id);
    ~FtLlama();
    void
    forward(const std::vector<std::vector<int>>& input_ids, size_t request_output_len, std::function<void()> callback);
};
