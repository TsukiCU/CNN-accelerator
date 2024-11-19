#ifndef _CUDA_TENSOR_H_
#define _CUDA_TENSOR_H_

#include "common.h"
#include "memory.h"

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
    Tensor(std::vector<uint32_t> shape, DataType dtype, DeviceType device);
    Tensor(std::vector<uint32_t> shape, DataType dtype, DeviceType device, void* data, bool copy);
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;
    ~Tensor();

    uint32_t dim() const { return dim_; }
    uint32_t size() const { return size_; }
    DeviceType device() const { return device_; }
    std::vector<uint32_t> shape() const { return shape_; }
    std::vector<uint32_t> stride() const { return stride_; }

    Tensor reshape(const std::vector<uint32_t>& new_shape);

    bool operator== (const Tensor &other);
    bool operator!= (const Tensor &other);
    Tensor operator+ (const Tensor &other);
    Tensor operator- (const Tensor &other);

    Tensor operator* (float scale);
    Tensor operator/ (float scale);
    Tensor operator* (const Tensor &other);
    Tensor operator/ (const Tensor &other);

    double at(const std::vector<uint32_t>& indices); // Return type is set to double

    // Helper functions
    Tensor multiply_generic(float scale, const Tensor* other);
    Tensor divide_generic(float scale, const Tensor* other);
    Tensor clone() const;  // For deep copy only.
    void print() const;

    // Static functions
    static Tensor zeros(const std::vector<uint32_t>& shape, DataType dtype);
    static Tensor ones(const std::vector<uint32_t>& shape, DataType dtype);
    static Tensor random(const std::vector<uint32_t>& shape, DataType dtype);

private:
    uint32_t dim_;
    uint32_t size_;
    DataType dtype_;
    DeviceType device_;
    std::vector<uint32_t> shape_;
    std::vector<uint32_t> stride_;
    std::shared_ptr<MemoryBuffer> buffer_;

    // Neural networks fundamentals
    Tensor gradient();
    Tensor relu();
    Tensor sigmoid();
    Tensor matmul(const Tensor& other) const;

    // Helper functions
    void create_tensor(void* data, bool copy);
    bool check_indices(std::vector<uint32_t> indices);
    uint32_t compute_offset(std::vector<uint32_t> indice);

    void fill(float value); // data is set to float but can fit with any data type.

    template<typename F>
    void dispatch_type(DataType dtype, F&& func);
};

} // cuda

#endif // _CUDA_TENSOR_H_