#ifndef _CUDA_TENSOR_H_
#define _CUDA_TENSOR_H_

#include "common.h"
#include "memory.h"
#include "threadpool.h"

#define TENSOR_ADD 1
#define TENSOR_SUB 2
#define TENSOR_MUL 3
#define TENSOR_DIV 4
#define TENSOR_EQL 5

namespace cuda {

/*
 * @brief : Tensor class. Support all common tensor operations.
 * @todo  : at, gradient, sigmoid, relu..
 */
class Tensor {
public:
    Tensor() {}
    Tensor(std::vector<uint32_t> shape, DataType dtype, DeviceType device);
    Tensor(std::vector<uint32_t> shape, DataType dtype, DeviceType device, void* data, bool copy);
    Tensor(const Tensor& other);
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(const Tensor& other);
    Tensor& operator=(Tensor&& other) noexcept;
    ~Tensor();

    uint32_t dim() const { return dim_; }
    uint32_t size() const { return size_; }
    DeviceType device() const { return device_; }
    std::vector<uint32_t> shape() const { return shape_; }
    std::vector<uint32_t> stride() const { return stride_; }

    Tensor reshape(const std::vector<uint32_t>& new_shape);

    // Arithmetic functions.
    bool operator== (const Tensor &other);
    bool operator!= (const Tensor &other);
    Tensor operator+ (const Tensor &other);
    Tensor operator- (const Tensor &other);

    Tensor operator* (float scale);
    Tensor operator/ (float scale);
    Tensor operator* (const Tensor &other);
    Tensor operator/ (const Tensor &other);

    // Neural networks fundamentals
    Tensor gradient();
    Tensor relu();
    Tensor sigmoid();
    Tensor matmul(const Tensor& other) const;
    template <typename T>
    void matmul_impl(void* data_a, void* data_b, void* data_c, uint32_t M, uint32_t K, uint32_t N) const;

    // template<typename... Indices>
    // avoid operator()(Indices... indices);
    template <typename T>
    T& operator()(const std::vector<uint32_t>& indices);
    double at(const std::vector<uint32_t>& indices);    // Return type is set to double.

    // Helper functions
    Tensor multiply_generic(float scale, const Tensor* other);
    Tensor divide_generic(float scale, const Tensor* other);
    Tensor transpose();
    Tensor clone() const;  // For deep copy only.

    // Static functions. double if dtype is not specified.
    static Tensor zeros(const std::vector<uint32_t>& shape, DataType dtype);    // Not implemented.
    static Tensor ones(const std::vector<uint32_t>& shape, DataType dtype);     // Not implemented.
    static Tensor rand(const std::vector<uint32_t>& shape, double lower, double upper); // Double by default.

    // Debug
    void print() const;

private:
    uint32_t dim_;
    uint32_t size_;
    DataType dtype_;
    DeviceType device_;
    std::vector<uint32_t> shape_;
    std::vector<uint32_t> stride_;
    std::shared_ptr<MemoryBuffer> buffer_;

    // Helper functions
    void create_tensor(void* data, bool copy);
    bool check_indices(std::vector<uint32_t> indices);
    uint32_t compute_offset(std::vector<uint32_t> indice);
    void fill(float value); // data is set to float but can fit with any data type.

    template <typename T>
    DataType get_dtype(T t) {
        if constexpr (std::is_same_v<T, int8_t>) {
            return DataType::DataTypeInt8;
        } else if constexpr (std::is_same_v<T, int16_t>) {
            return DataType::DataTypeInt16;
        } else if constexpr (std::is_same_v<T, int32_t>) {
            return DataType::DataTypeInt32;
        } else if constexpr (std::is_same_v<T, float>) {
            return DataType::DataTypeFloat32;
        } else if constexpr (std::is_same_v<T, double>) {
            return DataType::DataTypeFloat64;
        } else {
            return DataType::DataTypeUnknown;
        }
    }

    template<typename F, typename... Args>
    void dispatch_type(DataType dtype, F&& func, Args&&... args);
};

} // cuda

#endif // _CUDA_TENSOR_H_