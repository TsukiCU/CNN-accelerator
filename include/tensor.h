#ifndef _CUDA_TENSOR_H_
#define _CUDA_TENSOR_H_

#include "util.h"

namespace cuda {

class Tensor {
public:
    Tensor();
    uint32_t dim();
    uint32_t size();
    std::vector<uint32_t> shape();
    std::vector<uint32_t> stride();

private:
    uint32_t dim_;
    std::vector<uint32_t> shape_;
    std::vector<uint32_t> stride_;
};

} // cuda

#endif _CUDA_TENSOR_H_