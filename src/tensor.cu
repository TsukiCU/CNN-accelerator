#include "include/tensor.h"

namespace cuda
{
    
Tensor::Tensor(uint32_t dim, const std::vector<uint32_t> shape, DataType dtype, DeviceType device = DeviceType::CPU) :
    dim_(dim), shape_(shape), device_(device), dtype_(dtype)
{
    create_tensor(nullptr, false);
}

Tensor::Tensor(uint32_t dim, const std::vector<uint32_t> shape, DataType dtype,
    void* data, bool copy, DeviceType device = DeviceType::CPU) :
    dim_(dim), shape_(shape), device_(device), dtype_(dtype)
{
    create_tensor(data, copy);
}

void Tensor::create_tensor(void* data, bool copy) {
    uint32_t data_size = get_data_size(dtype_);
    size_ = 1;
    stride_.resize(dim_);
    for (int i = dim_-1; i >= 0; --i) {
        stride_[i] = size_;
        size_ *= shape_[i];
    }

    if (data) {
        if (copy) {
            // Log info.
            buffer_ = std::make_shared<MemoryBuffer> (data_size * size_, device_);
            // buffer_->copy_from({data, data_size * size_, device_});
            auto temp_buffer = std::make_shared<MemoryBuffer>(data_size * size_, device_, data);
            buffer_->copy_from(*temp_buffer);
        }
        else {
            // Log warn. Use data directly.
            // buffer_ = std::make_shared<MemoryBuffer>(data, data_size * size_, device_);
            buffer_ = std::make_shared<MemoryBuffer> (MemoryBuffer::create_from_existing(data, data_size * size_, device_));
        }
    }
    else {
        // Log info.
        buffer_ = std::make_shared<MemoryBuffer> (data_size * size_, device_);
    }
}

Tensor::~Tensor() {
    // No specific things need to be done as MemoryBuffer is handled by smart pointer.
}

bool Tensor::operator== (const Tensor& other) {
    if (size_ != other.size_ || dim_ != other.dim_ ||
    stride_.size() != other.stride_.size() || shape_.size() != other.shape_.size())
        return false;
}

bool Tensor::operator!= (const Tensor& other) {
    return !(*this == other);
}

} // cuda