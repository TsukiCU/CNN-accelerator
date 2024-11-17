#include <gtest/gtest.h>
#include "include/tensor.h"


/********** Tensor creation and basics **********/
TEST(TensorTest, TensorBasic) {
    std::vector<uint32_t> shape = {1, 2, 3};
    cuda::Tensor(shape, cuda::DataType::DataTypeInt32);
}