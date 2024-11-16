#ifndef _CUDA_TENSOR_H_
#define _CUDA_TENSOR_H_

#include "common.h"
#include "memory.h"

#define TENSOR_ADD 1
#define TENSOR_SUB 2
#define TENSOR_MUL 3
#define TENSOR_DIV 4

namespace cuda {

/*
 * @brief : Tensor class. Support all common tensor operations.
 * @todo  : resize, broadcast, print, operator(), Log.
 */
class Tensor {
public:
    Tensor(std::vector<uint32_t> shape, DataType dtype, DeviceType device);
    Tensor(std::vector<uint32_t> shape, DataType dtype, void* data, bool copy, DeviceType device);
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;
    ~Tensor();

    uint32_t dim() const { return dim_; }
    uint32_t size() const { return size_; }
    std::vector<uint32_t> shape() const { return shape_; }
    std::vector<uint32_t> stride() const { return stride_; }

    void reshape(const std::vector<uint32_t>& new_shape);
    void resize(const std::vector<uint32_t>& shape);
    Tensor broadcast(const std::vector<uint32_t> shape, std::vector<uint32_t> is_broadcast);

    bool operator== (const Tensor &other);
    bool operator!= (const Tensor &other);
    Tensor operator+ (const Tensor &other);
    Tensor operator- (const Tensor &other);

    Tensor operator* (float scale);
    Tensor operator/ (float scale);
    Tensor operator* (const Tensor &other);
    Tensor operator/ (const Tensor &other);

    Tensor multiply_generic(float scale, const Tensor* other);
    Tensor divide_generic(float scale, const Tensor* other);

    uint32_t operator() (const Tensor &other) const;

    Tensor clone() const;  // For deep copy only.
    void print() const;

private:
    uint32_t dim_;
    uint32_t size_;
    std::vector<uint32_t> shape_;
    std::vector<uint32_t> stride_;
    std::shared_ptr<MemoryBuffer> buffer_;

    DeviceType device_;
    DataType dtype_;

    void create_tensor(void* data, bool copy);
};

} // cuda

#endif // _CUDA_TENSOR_H_