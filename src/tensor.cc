#include "tensor.h"

namespace snnf {

template <typename T>
Tensor<T>::Tensor(const std::vector<uint32_t>& shape)
    : shape_(shape)
{
    compute_stride();
    size_ = std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<uint32_t>());
    data_.resize(size_);
}

template <typename T>
Tensor<T>::Tensor(const std::vector<uint32_t>& shape, const std::vector<T>& data)
    : shape_(shape), data_(data)
{
    compute_stride();
    size_ = std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<uint32_t>());
    if (data_.size() != size_) {
        LOG_ERROR(std::invalid_argument, "Tensor::Tensor : Data size does not match tensor shape.");
    }
}

template <typename T>
void Tensor<T>::compute_stride() {
    stride_.resize(shape_.size());
    uint32_t stride = 1;
    for (int i = shape_.size() - 1; i >= 0; --i) {
        stride_[i] = stride;
        stride *= shape_[i];
    }
}

/**
 * @brief : Compute offset for data access.
 */
template <typename T>
uint32_t Tensor<T>::compute_offset(const std::vector<uint32_t>& indices) const {
    if (indices.size() != shape_.size()) {
        LOG_ERROR(std::invalid_argument, "Indices size does not match tensor dimensions.");
    }
    uint32_t offset = 0;
    for (size_t i = 0; i < shape_.size(); ++i) {
        if (indices[i] >= shape_[i]) {
            LOG_ERROR(std::out_of_range, "Tensor::computer_offset : Index out of range.");
        }
        offset += indices[i] * stride_[i];
    }
    return offset;
}

template <typename T>
void Tensor<T>::check_shape(const Tensor<T>& other) const {
    if (shape_ != other.shape_) {
        LOG_ERROR(std::invalid_argument, "Tensor shapes do not match.");
    }
}

/**
 * @brief : Element access.
 */
template <typename T>
T& Tensor<T>::at(const std::vector<uint32_t>& indices) {
    uint32_t offset = compute_offset(indices);
    return data_[offset];
}

template <typename T>
const T& Tensor<T>::at(const std::vector<uint32_t>& indices) const {
    uint32_t offset = compute_offset(indices);
    return data_[offset];
}

template <typename T>
void Tensor<T>::fill(T value) {
    std::fill(data_.begin(), data_.end(), value);
}

template <typename T>
void Tensor<T>::reshape(const std::vector<uint32_t>& new_shape) {
    uint32_t new_size = std::accumulate(new_shape.begin(), new_shape.end(), 1, std::multiplies<uint32_t>());
    if (new_size != size_) {
        LOG_ERROR(std::invalid_argument, "Tensor::reshape : New shape size must match original size.");
    }
    shape_ = new_shape;
    compute_stride();
}

/**
 * @brief : Arithmetic operations. Supports broadcasting but do it outside.
 */
template <typename T>
Tensor<T> Tensor<T>::add(const Tensor<T>& other) const {
    check_shape(other);
    Tensor<T> result(shape_);
    for (uint32_t i = 0; i < size_; ++i) {
        result.data_[i] = data_[i] + other.data_[i];
    }
    return result;
}

template <typename T>
Tensor<T> Tensor<T>::subtract(const Tensor<T>& other) const {
    check_shape(other);
    Tensor<T> result(shape_);
    for (uint32_t i = 0; i < size_; ++i) {
        result.data_[i] = data_[i] - other.data_[i];
    }
    return result;
}

template <typename T>
Tensor<T> Tensor<T>::multiply(const Tensor<T>& other) const {
    check_shape(other);
    Tensor<T> result(shape_);
    for (uint32_t i = 0; i < size_; ++i) {
        result.data_[i] = data_[i] * other.data_[i];
    }
    return result;
}

template <typename T>
Tensor<T> Tensor<T>::divide(const Tensor<T>& other) const {
    check_shape(other);
    Tensor<T> result(shape_);
    for (uint32_t i = 0; i < size_; ++i) {
        if (other.data_[i] == static_cast<T>(0)) {
            LOG_ERROR(std::runtime_error, "Tensor::divide : Division by zero.");
        }
        result.data_[i] = data_[i] / other.data_[i];
    }
    return result;
}

template <typename T>
Tensor<T> Tensor<T>::matmul(const Tensor<T>& other) const {
    if (shape_.size() != 2 || other.shape_.size() != 2) {
        LOG_ERROR(std::invalid_argument, "Tensor::matmul : Matmul only supports 2D tensors.");
    }
    if (shape_[1] != other.shape_[0]) {
        LOG_ERROR(std::invalid_argument, "Tensor::matmul : Inner dimensions do not match for matmul.");
    }

    uint32_t M = shape_[0];
    uint32_t K = shape_[1];
    uint32_t N = other.shape_[1];

    Tensor<T> result({M, N});
    for (uint32_t i = 0; i < M; ++i) {
        for (uint32_t j = 0; j < N; ++j) {
            T sum = static_cast<T>(0);
            for (uint32_t k = 0; k < K; ++k) {
                sum += at({i, k}) * other.at({k, j});
            }
            result.at({i, j}) = sum;
        }
    }
    return result;
}

/**
 * @brief : Element access. Only supports 2d tensors for now.
 */
template <typename T>
Tensor<T> Tensor<T>::transpose() const {
    if (shape_.size() != 2) {
        LOG_ERROR(std::invalid_argument, "Tensor::transpose : Transpose only supports 2D tensors.");
    }
    uint32_t rows = shape_[0];
    uint32_t cols = shape_[1];

    Tensor<T> result({cols, rows});
    for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t j = 0; j < cols; ++j) {
            result.at({j, i}) = at({i, j});
        }
    }
    return result;
}

/**
 * @brief : Sum tensor by a given dimension. Reduces tensor size.
 */
template <typename T>
Tensor<T> Tensor<T>::sum(int dim) const {
    if (dim < 0 || dim >= static_cast<int>(shape_.size())) {
        LOG_ERROR(std::invalid_argument, "Tensor::sum : Invalid dimension for sum.");
    }
    std::vector<uint32_t> result_shape = shape_;
    result_shape.erase(result_shape.begin() + dim);

    if (result_shape.empty()) {
        // Return scalar
        T total = std::accumulate(data_.begin(), data_.end(), static_cast<T>(0));
        return Tensor<T>({}, {total});
    }

    Tensor<T> result(result_shape);
    result.fill(static_cast<T>(0));

    for (uint32_t i = 0; i < size_; ++i) {
        // Compute multi-dimensional indices
        std::vector<uint32_t> indices(shape_.size());
        uint32_t temp = i;
        for (int j = 0; j < static_cast<int>(shape_.size()); ++j) {
            indices[j] = temp / stride_[j];
            temp = temp % stride_[j];
        }
        // Remove the specified dimension
        indices.erase(indices.begin() + dim);
        uint32_t offset = result.compute_offset(indices);
        result.data_[offset] += data_[i];
    }
    return result;
}

/**
 * @brief : Broadcast to a new shape.
 */
template <typename T>
Tensor<T> Tensor<T>::broadcast_to(const std::vector<uint32_t>& target_shape) const {
    if (shape_.size() > target_shape.size()) {
        LOG_ERROR(std::invalid_argument, "Tensor::broadcast_to : Cannot broadcast to fewer dimensions.");
    }

    std::vector<uint32_t> new_shape = shape_;
    while (new_shape.size() < target_shape.size()) {
        new_shape.insert(new_shape.begin(), 1);
    }

    for (size_t i = 0; i < target_shape.size(); ++i) {
        if (new_shape[i] != target_shape[i] && new_shape[i] != 1) {
            LOG_ERROR(std::invalid_argument, "Tensor::broadcast_to : Shapes cannot be broadcasted.");
        }
    }

    // Broadcasting is simulated by repeating data; actual implementation can be optimized
    Tensor<T> result(target_shape);
    std::vector<uint32_t> indices(new_shape.size(), 0);

    for (uint32_t i = 0; i < result.size_; ++i) {
        // Compute indices in the broadcasted tensor
        uint32_t temp = i;
        for (int j = 0; j < static_cast<int>(result.shape_.size()); ++j) {
            indices[j] = temp / result.stride_[j];
            temp = temp % result.stride_[j];
        }

        // Map indices back to the original tensor
        std::vector<uint32_t> orig_indices = indices;
        for (size_t j = 0; j < new_shape.size(); ++j) {
            if (new_shape[j] == 1) {
                orig_indices[j] = 0;
            }
        }

        // Compute offsets
        uint32_t orig_offset = compute_offset(orig_indices);
        result.data_[i] = data_[orig_offset];
    }
    return result;
}

/**
 * @brief : Gradient support.
 */
template <typename T>
Tensor<T>& Tensor<T>::grad() {
    if (!grad_) {
        grad_ = std::make_shared<Tensor<T>>(shape_);
        grad_->fill(static_cast<T>(0));
    }
    return *grad_;
}

template <typename T>
void Tensor<T>::zero_grad() {
    if (grad_) {
        grad_->fill(static_cast<T>(0));
    }
}

template <typename T>
void Tensor<T>::random(T lower, T upper) {
    if (lower >= upper) {
        LOG_ERROR(std::invalid_argument, "Tensor::random : Lower bound must be less than upper bound.");
    }

    std::default_random_engine engine(std::random_device{}());
    std::uniform_real_distribution<T> dist(lower, upper);
    for (auto& elem : data_) {
        elem = dist(engine);
    }
}

template <typename T>
void Tensor<T>::random_normal(T mean, T stddev) {
    if (stddev <= static_cast<T>(0)) {
        LOG_ERROR(std::invalid_argument, "Tensor::random_normal : Standard deviation must be positive.");
    }

    std::default_random_engine engine(std::random_device{}());
    std::normal_distribution<T> dist(mean, stddev);
    for (auto& elem : data_) {
        elem = dist(engine);
    }
}

// Explicit template instantiation.
template class Tensor<float>;
template class Tensor<double>;

// template class Tensor<uint32_t>;
// Now only supports floating type because of the existence of Tensor::random_normal and Tensor::random.

} // snnf