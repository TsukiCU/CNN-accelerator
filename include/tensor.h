#pragma once

#include "utils.h"
#include "log.h"

namespace snnf {

template <typename T>
class Tensor {
public:
    Tensor() {}
    Tensor(const std::vector<uint32_t>& shape);
    Tensor(const std::vector<uint32_t>& shape, const std::vector<T>& data);

    const std::vector<uint32_t>& shape() const { return shape_; }
    const std::vector<uint32_t>& stride() const { return stride_; }
    uint32_t size() const { return size_; }
    uint32_t dim() const { return shape_.size(); }
    const std::vector<T>& data() const { return data_; }
    std::vector<T>& data() { return data_; }

    // Basic operations.
    void fill(T value);
    void reshape(const std::vector<uint32_t>& new_shape);
    T& at(const std::vector<uint32_t>& indices);
    const T& at(const std::vector<uint32_t>& indices) const;

    Tensor<T> matmul(const Tensor<T>& other) const;
    Tensor<T> transpose() const;
    Tensor<T> add(const Tensor<T>& other) const;
    Tensor<T> subtract(const Tensor<T>& other) const;
    Tensor<T> multiply(const Tensor<T>& other) const;
    Tensor<T> divide(const Tensor<T>& other) const;
    Tensor<T> sum(int dim) const;
    Tensor<T> broadcast_to(const std::vector<uint32_t>& target_shape) const;

    // Gradient support.
    Tensor<T>& grad();
    void zero_grad();
    void set_grad(const Tensor<T>& gradient);

    // randomized initialization.
    void random(T lower = static_cast<T>(0), T upper = static_cast<T>(1));          // Uniform
    void random_normal(T mean = static_cast<T>(0), T stddev = static_cast<T>(1));   // Normal 
    
private:
    std::vector<uint32_t> shape_;
    std::vector<uint32_t> stride_;
    uint32_t size_;
    std::vector<T> data_;   // stores actual data.
    std::shared_ptr<Tensor<T>> grad_;

    // Helper functions.
    void compute_stride();
    uint32_t compute_offset(const std::vector<uint32_t>& indices) const;
    void check_shape(const Tensor<T>& other) const;
};

}  // snnf