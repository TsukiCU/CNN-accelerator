#include "../include/tensor.h"
#include "../include/log.h"
using namespace std;
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
        LOG_FATAL("Cannot copy from a null data pointer.");
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
            LOG_INFO("Creating a tensor by copying another one.");
            // buffer_->copy_from({data, data_size * size_, device_});

            // buffer_ = std::make_shared<MemoryBuffer> (data, data_size * size_, device_);
            buffer_ = std::make_shared<MemoryBuffer>(data_size * size_, device_);
            std::memcpy(buffer_->data(), data, data_size * size_);
        } else {
            LOG_WARN("Creating tensors by directly using another one.");
            // TBH this should never get called.
            buffer_ = std::make_shared<MemoryBuffer> (MemoryBuffer::create_from_existing(data, data_size * size_, device_));
        }
    } else {
        LOG_INFO("Creating a new tensor.");
        buffer_ = std::make_shared<MemoryBuffer> (data_size * size_, device_);
    }
}

Tensor::Tensor(const Tensor& other) :
    dim_(other.dim()), shape_(std::move(other.shape_)), dtype_(other.dtype_), device_(other.device_)
{
    create_tensor(other.buffer_->data(), true);
}

Tensor::Tensor(Tensor&& other) noexcept
    : dim_(other.dim_), size_(other.size_), shape_(std::move(other.shape_)),
      stride_(std::move(other.stride_)), buffer_(std::move(other.buffer_)), dtype_(other.dtype_)
{
    other.dim_ = 0;
    other.size_ = 0;
    other.dtype_ = DataType::DataTypeUnknown;
    other.buffer_.reset();
}

Tensor::~Tensor() {
    // No specific things need to be done as MemoryBuffer is handled by smart pointer.
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        buffer_.reset();

        dim_ = other.dim_;
        size_ = other.size_;
        dtype_ = other.dtype_;
        device_ = other.device_;
        shape_ = other.shape_;
        stride_ = other.stride_;

        if (other.buffer_) {
            void* other_ptr = other.buffer_->data();
            uint32_t size = other.buffer_->size();
            if (!other_ptr) {
                LOG_FATAL("Copy constructor failed.");
            }
            // buffer_ = std::make_shared<MemoryBuffer> (other_ptr, size, device_);
            buffer_ = std::make_shared<MemoryBuffer>(size, device_);
            std::memcpy(buffer_->data(), other.buffer_->data(), size);
        }
        else
            buffer_ = nullptr;
    }

    return *this;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        buffer_.reset();

        dim_ = other.dim_;
        size_ = other.size_;
        dtype_ = other.dtype_;
        device_ = other.device_;
        shape_ = other.shape_;
        stride_ = other.stride_;
        buffer_ = other.buffer_;

        other.dim_ = 0;
        other.size_ = 0;
        other.dtype_ = DataType::DataTypeUnknown;
        other.buffer_.reset();
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
                for (uint32_t i = 0; i < size; ++i) {
                    typed_data[i] *= static_cast<T>(scale);
                }
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
            LOG_ERROR("Unsupported arithmetic type.");
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
        LOG_ERROR("can't compare between empty tensors.");
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
            LOG_ERROR("Unsupported data type for comparison.");
    }
}

bool Tensor::operator!= (const Tensor& other) {
    return !(*this == other);
}

Tensor Tensor::operator+ (const Tensor& other) {
    if (size_ != other.size_ || dim_ != other.dim_ || stride_.size() != other.stride_.size())
        LOG_ERROR("Unable to perform add.");
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
            LOG_ERROR("Unsupported type for multiplication!");
    }

    return ans;
}

Tensor Tensor::operator- (const Tensor& other) {
    if (size_ != other.size_ || dim_ != other.dim_ || stride_.size() != other.stride_.size())
        LOG_ERROR("Unable to perform add.");

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
            LOG_ERROR("Unsupported type for multiplication!");
    }

    return ans;
}

Tensor Tensor::operator* (const Tensor& other) {
    if (size_ != other.size_ || dim_ != other.dim_ || stride_.size() != other.stride_.size()) {
        LOG_ERROR("Unable to perform multiplication.");
    }

    return multiply_generic(0, &other);
}

Tensor Tensor::operator* (float scale) {
    return multiply_generic(scale, nullptr);
}

Tensor Tensor::multiply_generic(float scale, const Tensor* other) {
    // Tensor ans = *this;
    Tensor ans = clone();
    void* data_ptr = ans.buffer_->data(), *other_ptr = nullptr;
    if (other)
        other_ptr = other->buffer_->data();

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
            LOG_ERROR("Unsupported type for multiplication!");
    }

    return ans;
}

Tensor Tensor::operator/ (const Tensor& other) {
    if (size_ != other.size_ || dim_ != other.dim_ || stride_.size() != other.stride_.size()) {
        LOG_ERROR("Unable to perform division.");
    }

    return divide_generic(0, &other);
}

Tensor Tensor::operator/ (float scale) {
    return divide_generic(scale, nullptr);
}

Tensor Tensor::divide_generic(float scale, const Tensor* other) {
    // Tensor ans = *this;
    Tensor ans = clone();
    void* data_ptr = ans.buffer_->data(), *other_ptr = nullptr;
    if (other)
        other_ptr = other->buffer_->data();

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
            LOG_ERROR("Unsupported type for division!");
    }

    return ans;
}

Tensor Tensor::reshape(const std::vector<uint32_t>& new_shape) {
    Tensor ans = clone();
    uint32_t elem_count = 1;

    for (const uint32_t& n : new_shape)
        elem_count *= n;
    if (elem_count != size_) {
        LOG_ERROR("Reshape failed: New shape doesn't fit.");
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

// typename... Args for extensibility
template<typename F, typename... Args>
void Tensor::dispatch_type(DataType dtype, F&& func, Args&&... args) {
    switch(dtype) {
        case DataType::DataTypeInt8:
            func(static_cast<int8_t*>(buffer_->data()), std::forward<Args>(args)...);
            break;
        case DataType::DataTypeInt16:
            func(static_cast<int16_t*>(buffer_->data()), std::forward<Args>(args)...);
            break;
        case DataType::DataTypeInt32:
            func(static_cast<int32_t*>(buffer_->data()), std::forward<Args>(args)...);
            break;
        case DataType::DataTypeFloat32:
            func(static_cast<float*>(buffer_->data()), std::forward<Args>(args)...);
            break;
        case DataType::DataTypeFloat64:
            func(static_cast<double*>(buffer_->data()), std::forward<Args>(args)...);
            break;
        case DataType::DataTypeUnknown:
            LOG_ERROR("Unsupported data type while dispatching.");
        default:
            LOG_ERROR("Unsupported data type while dispatching.");
    }
}

double Tensor::at(const std::vector<uint32_t>& indices) {
    uint32_t offset = compute_offset(indices);
    if (offset == -1) {
        LOG_ERROR("Tensor.at() : Indice out of range.");
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

template <typename T>
T& Tensor::operator()(const std::vector<uint32_t>& indices) {
    // Need find a way to check the data type.
    uint32_t offset = compute_offset(indices);
    if (offset >= size_) {
        LOG_ERROR("Tensor.operator() : Indice out of range.");
    }
    T* data = static_cast<T*>(buffer_->data());
    return data[offset];
}

Tensor Tensor::rand(const std::vector<uint32_t>& shape, double lower=0.0, double upper=1.0) {
    int size = 1;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(lower, upper);
    for (uint32_t n : shape) {
        size *= n;
    }
    double* data = (double*)malloc(size * get_data_size(DataType::DataTypeFloat64));
    if (!data) {
        LOG_FATAL("Tensor.rand() : malloc failed.");
    }
    for (uint32_t i=0; i < size; ++i) {
        data[i] = dist(gen);
    }
    Tensor ret(shape, DataType::DataTypeFloat64, DeviceType::CPU, (void*)data, true);
    free(data); // This is safe.
    return ret;
}

template <typename T>
void Tensor::matmul_impl(void* data_a, void* data_b, void* data_c, uint32_t M, uint32_t K, uint32_t N) const {
    T* typed_a = static_cast< T* >(data_a);
    T* typed_b = static_cast< T* >(data_b);
    T* typed_c = static_cast< T* >(data_c);

    for (uint32_t i = 0; i < M; ++i) {
        for (uint32_t j = 0; j < N; ++j) {
            T sum = 0;
            for (uint32_t k = 0; k < K; ++k)
                sum += typed_a[i * K + k] * typed_b[k * N + j];
            typed_c[i * N + j] = sum;
        }
    }
}

Tensor Tensor::matmul(const Tensor& other) const {
    if (dim_ != 2 || other.dim() != 2) {
        LOG_ERROR("matmul only supports 2D tensors (matrices).");
    }
    if (shape_[1] != other.shape_[0]) {
        LOG_ERROR("matmul fails because of imcompatible matrices.");
    }
    std::vector<uint32_t> res_shape = { shape_[0], other.shape_[1] };
    Tensor res(res_shape, dtype_, device_);

    uint32_t M = shape_[0];
    uint32_t K = shape_[1];
    uint32_t N = other.shape_[1];
    void* data_a = buffer_->data();
    void* data_b = other.buffer_->data();
    void* data_c = res.buffer_->data();

    switch (dtype_) {
        case DataType::DataTypeInt8:
            matmul_impl<int8_t>(data_a, data_b, data_c, M, K, N);
            break;
        case DataType::DataTypeInt16:
            matmul_impl<int16_t>(data_a, data_b, data_c, M, K, N);
            break;
        case DataType::DataTypeInt32:
            matmul_impl<int32_t>(data_a, data_b, data_c, M, K, N);
            break;
        case DataType::DataTypeFloat32:
            matmul_impl<float>(data_a, data_b, data_c, M, K, N);
            break;
        case DataType::DataTypeFloat64:
            matmul_impl<double>(data_a, data_b, data_c, M, K, N);
            break;
        default:
            LOG_ERROR("Unsupported data type in matmul.");
    }

    return res;
}

// Only supports 2d tensors. Might need to apply to multi-dimension tensors.
Tensor Tensor::transpose() {
    if (dim_ != 2) {
        LOG_ERROR("Tensor::transpose : For now, transpose only supports 2d tensors.");
    }

    uint32_t rows = shape_[0], cols = shape_[1];
    Tensor transposed_tensor({cols, rows}, dtype_, device_);

    dispatch_type(dtype_, [&](auto* src_ptr) {
        // using T = decltype(*src_ptr);
        using T = std::remove_reference_t<decltype(*src_ptr)>; // erase reference.
        T* dst_ptr = static_cast<T*>(transposed_tensor.buffer_->data());

        for (uint32_t i = 0; i < rows; ++i) {
            for (uint32_t j = 0; j < cols; ++j) {
                dst_ptr[j * rows + i] = src_ptr[i * cols + j];
            }
        }
    });

    return transposed_tensor;
}

} // cuda


int main()
{
    cuda::Tensor t1 = cuda::Tensor::rand({2,2}, 2.5, 4.5);

    cout << t1.at({0,0}) << ", " << t1.at({1,0})<< endl;
    cout << t1.at({1,0}) << ", " << t1.at({1,1})<< endl;

    t1 = t1.transpose();

    cout << t1.at({0,0}) << ", " << t1.at({1,0})<< endl;
    cout << t1.at({1,0}) << ", " << t1.at({1,1})<< endl;

    return 0;
}