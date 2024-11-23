#include <gtest/gtest.h>
#include "../include/tensor.h"


/********** Tensor creation and basics **********/
TEST(TensorTest, TensorBasic) {
    std::vector<uint32_t> shape = {2, 3, 4};
    cuda::Tensor t1(shape, cuda::DataType::DataTypeInt32, cuda::DeviceType::CPU);
    EXPECT_EQ(t1.dim(), 3);
    EXPECT_EQ(t1.size(), 24);
    EXPECT_EQ(t1.device(), cuda::DeviceType::CPU);
    EXPECT_EQ(t1.stride()[0], 12);
    EXPECT_EQ(t1.stride()[1], 4);
    EXPECT_EQ(t1.stride()[2], 1);
}

/********** Tensor creation and basics **********/