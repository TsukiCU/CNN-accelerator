#include "include/tensor.h"

namespace cuda
{

uint32_t Tensor::dim()
{
    return dim_;
}

uint32_t Tensor::size()
{
    return shape_[0] * stride_[0];
}

std::vector<uint32_t> Tensor::shape()
{
    return shape_;
}

} // cuda