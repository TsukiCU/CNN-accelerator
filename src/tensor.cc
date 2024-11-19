#include "../include/tensor.h"

namespace cuda
{
    
Tensor::Tensor(const std::vector<uint32_t> shape, DataType dtype, DeviceType device)
    : dim_(shape.size()), shape_(shape), dtype_(dtype), device_(device)
{
    create_tensor(nullptr, false);
}

Tensor::Tensor(const std::vector<uint32_t> shape, DataType dtype, DeviceType device, void* data, bool copy)
    : dim_(shape.size()), shape_(shape), dtype_(dtype), device_(device)
{
    if (!data && copy) {
        // Log fatal.
        throw std::invalid_argument("Cannot copy from a null data pointer.");
    }
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
            // auto temp_buffer = std::make_shared<MemoryBuffer>(data_size * size_, data);
            // buffer_->copy_from(*temp_buffer);
            buffer_->copy_from({data, data_size * size_, device_});
        } else {
            // Log warn. Use data directly.
            // buffer_ = std::make_shared<MemoryBuffer>(data, data_size * size_);
            buffer_ = std::make_shared<MemoryBuffer> (MemoryBuffer::create_from_existing(data, data_size * size_, device_));
        }
    } else {
        // Log info.
        buffer_ = std::make_shared<MemoryBuffer> (data_size * size_, device_);
    }
}

Tensor::Tensor(Tensor&& other) noexcept
    : dim_(other.dim_), size_(other.size_), shape_(std::move(other.shape_)),
      stride_(std::move(other.stride_)), buffer_(std::move(other.buffer_)), dtype_(other.dtype_)
{
    other.dim_ = 0;
    other.size_ = 0;
    other.dtype_ = DataType::DataTypeUnkown;
}

Tensor::~Tensor() {
    // No specific things need to be done as MemoryBuffer is handled by smart pointer.
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (*this != other) {
        // TODO: Log info. May need to release resource first.
        dim_ = other.dim_;
        size_ = other.size_;
        dtype_ = other.dtype_;
        shape_ = std::move(other.shape_);
        buffer_ = std::move(other.buffer_);
        stride_ = std::move(other.stride_);

        other.dim_ = 0;
        other.size_ = 0;
        other.dtype_ = DataType::DataTypeUnkown;
    }
    return *this;
}

template <typename T>
bool arithmetic_generic(int type, void* data, const void* other_data, uint32_t size, float scale) {
    bool ret = true;
    T* typed_data = static_cast<T*>(data);
    const T* typed_other_data = nullptr;
    if (other_data)
        typed_other_data = static_cast<const T*>(other_data);

    switch (type) {
        case TENSOR_ADD:
            for (uint32_t i = 0; i < size; ++i)
                typed_data[i] += typed_other_data[i];
            break;
        case TENSOR_SUB:
            for (uint32_t i = 0; i < size; ++i)
                typed_data[i] -= typed_other_data[i];
            break;
        case TENSOR_MUL:
            if (other_data) {
                for (uint32_t i = 0; i < size; ++i)
                    typed_data[i] *= typed_other_data[i];
            }
            else {
                for (uint32_t i = 0; i < size; ++i)
                    typed_data[i] *= static_cast<T>(scale);
            }
            break;
        case TENSOR_DIV:
            if (other_data) {
                for (uint32_t i = 0; i < size; ++i)
                    typed_data[i] /= typed_other_data[i];
            }
            else {
                for (uint32_t i = 0; i < size; ++i)
                    typed_data[i] /= static_cast<T>(scale);
            }
            break;
        case TENSOR_EQL:
            for (uint32_t i = 0; i < size; ++i) {
                if (typed_data[i] != typed_other_data[i]) {
                    ret = false;
                    break;
                }
            }
            break;
        default:
            // Log error
            throw std::invalid_argument("Unsupported arithmetic type. ");
    }
    return ret;
}

bool Tensor::operator== (const Tensor& other) {
    if (size_ != other.size_ || dim_ != other.dim_ || stride_.size() != other.stride_.size() || dtype_ != other.dtype_)
            return false;
    for (uint32_t i = 0; i < shape_.size(); ++i)
        if (shape_[i] != other.shape_[i] || stride_[i] != other.stride_[i])
            return false;
    // compare element by element.
    void* data_ptr = buffer_->data();
    const void* other_ptr = buffer_->data();
    if (!data_ptr || !other_ptr) {
        // Log error.
        throw std::runtime_error("can't compare between empty tensors.");
    }

    bool equal = false;
    switch (dtype_) {
        case DataType::DataTypeInt8:
            return arithmetic_generic<int8_t>(TENSOR_EQL, data_ptr, other_ptr, size_, 0);
        case DataType::DataTypeInt16:
            return arithmetic_generic<int16_t>(TENSOR_EQL, data_ptr, other_ptr, size_, 0);
        case DataType::DataTypeInt32:
            return arithmetic_generic<int32_t>(TENSOR_EQL, data_ptr, other_ptr, size_, 0);
        case DataType::DataTypeFloat32:
            return arithmetic_generic<float>(TENSOR_EQL, data_ptr, other_ptr, size_, 0);
        case DataType::DataTypeFloat64:
            return arithmetic_generic<double>(TENSOR_EQL, data_ptr, other_ptr, size_, 0);
        default:
            throw std::invalid_argument("Unsupported data type for comparison.");
    }
}

bool Tensor::operator!= (const Tensor& other) {
    return !(*this == other);
}

Tensor Tensor::operator+ (const Tensor& other) {
    if (size_ != other.size_ || dim_ != other.dim_ || stride_.size() != other.stride_.size())
        // TODO: Log error
        throw std::runtime_error("Unable to perform add. ");
    Tensor ans = clone();
    void* data_ptr = ans.buffer_->data();
    const void* other_ptr = other.buffer_->data();

    switch (dtype_) {
        case DataType::DataTypeInt8:
            arithmetic_generic<int8_t>(TENSOR_ADD, data_ptr, other_ptr, size_, 0);
            break;
        case DataType::DataTypeInt16:
            arithmetic_generic<int16_t>(TENSOR_ADD, data_ptr, other_ptr, size_, 0);
            break;
        case DataType::DataTypeInt32:
            arithmetic_generic<int32_t>(TENSOR_ADD, data_ptr, other_ptr, size_, 0);
            break;
        case DataType::DataTypeFloat32:
            arithmetic_generic<float>(TENSOR_ADD, data_ptr, other_ptr, size_, 0);
            break;
        case DataType::DataTypeFloat64:
            arithmetic_generic<double>(TENSOR_ADD, data_ptr, other_ptr, size_, 0);
            break;
        default:
            // Log error
            throw std::invalid_argument("Unsupported type for multiplication! ");
    }

    return ans;
}

Tensor Tensor::operator- (const Tensor& other) {
    if (size_ != other.size_ || dim_ != other.dim_ || stride_.size() != other.stride_.size())
        // TODO: Log error
        throw std::runtime_error("Unable to perform add. ");
    Tensor ans = clone();
    void* data_ptr = ans.buffer_->data();
    const void* other_ptr = other.buffer_->data();

    switch (dtype_) {
        case DataType::DataTypeInt8:
            arithmetic_generic<int8_t>(TENSOR_SUB, data_ptr, other_ptr, size_, 0);
            break;
        case DataType::DataTypeInt16:
            arithmetic_generic<int16_t>(TENSOR_SUB, data_ptr, other_ptr, size_, 0);
            break;
        case DataType::DataTypeInt32:
            arithmetic_generic<int32_t>(TENSOR_SUB, data_ptr, other_ptr, size_, 0);
            break;
        case DataType::DataTypeFloat32:
            arithmetic_generic<float>(TENSOR_SUB, data_ptr, other_ptr, size_, 0);
            break;
        case DataType::DataTypeFloat64:
            arithmetic_generic<double>(TENSOR_SUB, data_ptr, other_ptr, size_, 0);
            break;
        default:
            // Log error
            throw std::invalid_argument("Unsupported type for multiplication! ");
    }

    return ans;
}

Tensor Tensor::operator* (const Tensor& other) {
    if (size_ != other.size_ || dim_ != other.dim_ || stride_.size() != other.stride_.size())
        // TODO: Log error
        throw std::runtime_error("Unable to perform multiplication. ");
    return multiply_generic(0, &other);
}

Tensor Tensor::operator* (float scale) {
    return multiply_generic(scale, nullptr);
}

Tensor Tensor::multiply_generic(float scale, const Tensor* other) {
    if (other == nullptr) {
        // TODO: Log error.
        throw std::runtime_error("Multiply with a null Tensor is not allowed.");
    }
    // Tensor ans = *this;
    Tensor ans = clone();
    void* data_ptr = ans.buffer_->data();
    void* other_ptr = other->buffer_->data();

    switch (dtype_) {
        case DataType::DataTypeInt8:
            arithmetic_generic<int8_t>(TENSOR_MUL, data_ptr, other_ptr, size_, scale);
            break;
        case DataType::DataTypeInt16:
            arithmetic_generic<int16_t>(TENSOR_MUL, data_ptr, other_ptr, size_, scale);
            break;
        case DataType::DataTypeInt32:
            arithmetic_generic<int32_t>(TENSOR_MUL, data_ptr, other_ptr, size_, scale);
            break;
        case DataType::DataTypeFloat32:
            arithmetic_generic<float>(TENSOR_MUL, data_ptr, other_ptr, size_, scale);
            break;
        case DataType::DataTypeFloat64:
            arithmetic_generic<double>(TENSOR_MUL, data_ptr, other_ptr, size_, scale);
            break;
        default:
            // Log error
            throw std::invalid_argument("Unsupported type for multiplication! ");
    }

    return ans;
}

Tensor Tensor::operator/ (const Tensor& other) {
    if (size_ != other.size_ || dim_ != other.dim_ || stride_.size() != other.stride_.size())
        // TODO: Log error
        throw std::runtime_error("Unable to perform division. ");
    return divide_generic(0, &other);
}

Tensor Tensor::operator/ (float scale) {
    return divide_generic(scale, nullptr);
}

Tensor Tensor::divide_generic(float scale, const Tensor* other) {
    if (other == nullptr) {
        // TODO: Log error.
        throw std::runtime_error("Divide by a null Tensor is not allowed.");
    }
    // Tensor ans = *this;
    Tensor ans = clone();
    void* data_ptr = ans.buffer_->data();
    void* other_ptr = other->buffer_->data();

    switch (dtype_) {
        case DataType::DataTypeInt8:
            arithmetic_generic<int8_t>(TENSOR_DIV, data_ptr, other_ptr, size_, scale);
            break;
        case DataType::DataTypeInt16:
            arithmetic_generic<int16_t>(TENSOR_DIV, data_ptr, other_ptr, size_, scale);
            break;
        case DataType::DataTypeInt32:
            arithmetic_generic<int32_t>(TENSOR_DIV, data_ptr, other_ptr, size_, scale);
            break;
        case DataType::DataTypeFloat32:
            arithmetic_generic<float>(TENSOR_DIV, data_ptr, other_ptr, size_, scale);
            break;
        case DataType::DataTypeFloat64:
            arithmetic_generic<double>(TENSOR_DIV, data_ptr, other_ptr, size_, scale);
            break;
        default:
            // Log error
            throw std::invalid_argument("Unsupported type for division! ");
    }

    return ans;
}

Tensor Tensor::reshape(const std::vector<uint32_t>& new_shape) {
    Tensor ans = clone();
    uint32_t elem_count = 1;

    for (const uint32_t& n : new_shape)
        elem_count *= n;
    if (elem_count != size_) {
        // TODO: Log error.
        throw std::runtime_error("Reshape failed: New shape doesn't fit. ");
    }

    int tmp_size = 1;
    ans.shape_ = std::move(new_shape);
    ans.dim_ = new_shape.size();
    ans.stride_.resize(dim_);
    for (int i = dim_-1; i >= 0; --i) {
        ans.stride_[i] = tmp_size;
        tmp_size *= shape_[i];
    }
    return ans;
}

Tensor Tensor::clone() const {
    Tensor new_tensor(shape_, dtype_, device_, buffer_->data(), true);
    return new_tensor;
}

bool Tensor::check_indices(std::vector<uint32_t> indices) {
    if (indices.size() != dim_)
        return false;
    for (uint32_t i = 0; i < dim_; ++i)
        if (indices[i] >= shape_[i])
            return false;
    return true;
}

uint32_t Tensor::compute_offset(std::vector<uint32_t> indices) {
    // TODO: handle potential overflow here.
    uint32_t offset = 0;
    if (!check_indices(indices))
        return -1;
    for (uint32_t i = 0; i < dim_; ++i)
        offset += stride_[i] * indices[i];
    return offset;
}

template<typename F>
void Tensor::dispatch_type(DataType dtype, F&& func) {
    switch(dtype) {
        case DataType::DataTypeInt8:
            func(static_cast<int8_t*>(buffer_->data()));
            break;
        case DataType::DataTypeInt16:
            func(static_cast<int16_t*>(buffer_->data()));
            break;
        case DataType::DataTypeInt32:
            func(static_cast<int32_t*>(buffer_->data()));
            break;
        case DataType::DataTypeFloat32:
            func(static_cast<float*>(buffer_->data()));
            break;
        case DataType::DataTypeFloat64:
            func(static_cast<double*>(buffer_->data()));
            break;
        case DataType::DataTypeUnkown:
            // Log error.
            throw std::runtime_error("Unsupported data type while dispatching. ");
        default:
            // Log error.
            throw std::runtime_error("Unsupported data type while dispatching. ");
    }
}

double Tensor::at(const std::vector<uint32_t>& indices) {
    uint32_t offset = compute_offset(indices);
    if (offset == -1) {
        // Log error.
        throw std::invalid_argument("indice out of range. ");
    }
    double result = 0;
    dispatch_type(dtype_, [&](auto* data) {
        result = static_cast<double>(data[offset]);
    });
    return result;
}

void Tensor::fill(float value) {
    dispatch_type(dtype_, [&](auto* data) {
        using data_type = std::decay_t<decltype(*data)>;
        for (uint32_t i = 0; i < size_; ++i) {
            data[i] = static_cast<data_type>(value);
        }
    });
}

} // cuda
