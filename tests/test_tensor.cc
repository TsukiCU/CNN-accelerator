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

/********** Tensor data access and modification **********/
TEST(TensorTest, TensorDataAccess) {
    std::vector<uint32_t> shape = {2, 3};
    cuda::Tensor t(shape, cuda::DataType::DataTypeFloat32, cuda::DeviceType::CPU);
    t.fill(1.0f);

    // Verify initial values
    for (uint32_t i = 0; i < shape[0]; ++i) {
        for (uint32_t j = 0; j < shape[1]; ++j) {
            EXPECT_EQ(t.at({i, j}), 1.0f);
        }
    }

    // Modify values
    t.fill(2.0f);
    for (uint32_t i = 0; i < shape[0]; ++i) {
        for (uint32_t j = 0; j < shape[1]; ++j) {
            EXPECT_EQ(t.at({i, j}), 2.0f);
        }
    }
}

/********** Tensor arithmetic operations **********/
TEST(TensorTest, TensorAddition) {
    std::vector<uint32_t> shape = {2, 2};
    cuda::Tensor t1(shape, cuda::DataType::DataTypeInt32, cuda::DeviceType::CPU);
    cuda::Tensor t2(shape, cuda::DataType::DataTypeInt32, cuda::DeviceType::CPU);

    t1.fill(3);
    t2.fill(5);

    cuda::Tensor t3 = t1 + t2;

    for (uint32_t i = 0; i < shape[0]; ++i)
        for (uint32_t j = 0; j < shape[1]; ++j)
            EXPECT_EQ(t3.at({i, j}), 8);
}

TEST(TensorTest, TensorSubtraction) {
    std::vector<uint32_t> shape = {2, 2};
    cuda::Tensor t1(shape, cuda::DataType::DataTypeInt32, cuda::DeviceType::CPU);
    cuda::Tensor t2(shape, cuda::DataType::DataTypeInt32, cuda::DeviceType::CPU);

    t1.fill(10);
    t2.fill(4);

    cuda::Tensor t3 = t1 - t2;

    for (uint32_t i = 0; i < shape[0]; ++i)
        for (uint32_t j = 0; j < shape[1]; ++j)
            EXPECT_EQ(t3.at({i, j}), 6);
}

TEST(TensorTest, TensorMultiplication) {
    std::vector<uint32_t> shape = {2, 2};
    cuda::Tensor t1(shape, cuda::DataType::DataTypeFloat32, cuda::DeviceType::CPU);
    cuda::Tensor t2(shape, cuda::DataType::DataTypeFloat32, cuda::DeviceType::CPU);

    t1.fill(2.0f);
    t2.fill(3.5f);

    cuda::Tensor t3 = t1 * t2;

    for (uint32_t i = 0; i < shape[0]; ++i)
        for (uint32_t j = 0; j < shape[1]; ++j)
            EXPECT_FLOAT_EQ(t3.at({i, j}), 7.0f);
}

TEST(TensorTest, TensorScalarMultiplication) {
    std::vector<uint32_t> shape = {3, 3};
    cuda::Tensor t(shape, cuda::DataType::DataTypeFloat32, cuda::DeviceType::CPU);

    t.fill(4.0f);

    cuda::Tensor result = t * 2.5f;

    for (uint32_t i = 0; i < shape[0]; ++i)
        for (uint32_t j = 0; j < shape[1]; ++j)
            EXPECT_FLOAT_EQ(result.at({i, j}), 10.0f);
}

TEST(TensorTest, TensorDivision) {
    std::vector<uint32_t> shape = {2, 2};
    cuda::Tensor t1(shape, cuda::DataType::DataTypeFloat32, cuda::DeviceType::CPU);
    cuda::Tensor t2(shape, cuda::DataType::DataTypeFloat32, cuda::DeviceType::CPU);

    t1.fill(9.0f);
    t2.fill(3.0f);

    cuda::Tensor t3 = t1 / t2;

    for (uint32_t i = 0; i < shape[0]; ++i)
        for (uint32_t j = 0; j < shape[1]; ++j)
            EXPECT_FLOAT_EQ(t3.at({i, j}), 3.0f);
}

TEST(TensorTest, TensorScalarDivision) {
    std::vector<uint32_t> shape = {2, 3};
    cuda::Tensor t(shape, cuda::DataType::DataTypeFloat32, cuda::DeviceType::CPU);

    t.fill(12.0f);

    cuda::Tensor result = t / 4.0f;

    for (uint32_t i = 0; i < shape[0]; ++i)
        for (uint32_t j = 0; j < shape[1]; ++j)
            EXPECT_FLOAT_EQ(result.at({i, j}), 3.0f);
}

/********** Tensor reshape and clone **********/
TEST(TensorTest, TensorReshape) {
    std::vector<uint32_t> shape = {2, 3};
    cuda::Tensor t(shape, cuda::DataType::DataTypeInt32, cuda::DeviceType::CPU);
    t.fill(1);

    std::vector<uint32_t> new_shape = {3, 2};
    cuda::Tensor reshaped = t.reshape(new_shape);

    EXPECT_EQ(reshaped.dim(), 2);
    EXPECT_EQ(reshaped.size(), 6);
    EXPECT_EQ(reshaped.shape()[0], 3);
    EXPECT_EQ(reshaped.shape()[1], 2);

    for (uint32_t i = 0; i < new_shape[0]; ++i)
        for (uint32_t j = 0; j < new_shape[1]; ++j)
            EXPECT_EQ(reshaped.at({i, j}), 1);
}

TEST(TensorTest, TensorClone) {
    std::vector<uint32_t> shape = {4, 4};
    cuda::Tensor t(shape, cuda::DataType::DataTypeInt32, cuda::DeviceType::CPU);
    t.fill(5);

    cuda::Tensor clone = t.clone();

    EXPECT_EQ(clone.dim(), t.dim());
    EXPECT_EQ(clone.size(), t.size());
    EXPECT_EQ(clone.dtype(), t.dtype());
    EXPECT_EQ(clone.device(), t.device());

    for (uint32_t i = 0; i < shape[0]; ++i)
        for (uint32_t j = 0; j < shape[1]; ++j)
            EXPECT_EQ(clone.at({i, j}), 5);
}

/********** Tensor equality and inequality **********/
TEST(TensorTest, TensorEquality) {
    std::vector<uint32_t> shape = {2, 2};
    cuda::Tensor t1(shape, cuda::DataType::DataTypeInt32, cuda::DeviceType::CPU);
    cuda::Tensor t2(shape, cuda::DataType::DataTypeInt32, cuda::DeviceType::CPU);

    t1.fill(7);
    t2.fill(7);

    EXPECT_TRUE(t1 == t2);
}

TEST(TensorTest, TensorInequality) {
    std::vector<uint32_t> shape = {2, 2};
    cuda::Tensor t1(shape, cuda::DataType::DataTypeInt32, cuda::DeviceType::CPU);
    cuda::Tensor t2(shape, cuda::DataType::DataTypeInt32, cuda::DeviceType::CPU);

    t1.fill(7);
    t2.fill(8);

    EXPECT_TRUE(t1 != t2);
}

/********** Tensor exception handling **********/
TEST(TensorTest, InvalidReshape) {
    std::vector<uint32_t> shape = {2, 2};
    cuda::Tensor t(shape, cuda::DataType::DataTypeFloat32, cuda::DeviceType::CPU);

    std::vector<uint32_t> invalid_shape = {3, 2};  // Total size doesn't match

    EXPECT_THROW({
        t.reshape(invalid_shape);
    }, std::runtime_error);
}

TEST(TensorTest, MismatchedShapesInOperations) {
    std::vector<uint32_t> shape1 = {2, 2};
    std::vector<uint32_t> shape2 = {2, 3};

    cuda::Tensor t1(shape1, cuda::DataType::DataTypeFloat32, cuda::DeviceType::CPU);
    cuda::Tensor t2(shape2, cuda::DataType::DataTypeFloat32, cuda::DeviceType::CPU);

    t1.fill(1.0f);
    t2.fill(2.0f);

    EXPECT_THROW({
        cuda::Tensor t3 = t1 + t2;
    }, std::runtime_error);

    EXPECT_THROW({
        cuda::Tensor t4 = t1 - t2;
    }, std::runtime_error);

    EXPECT_THROW({
        cuda::Tensor t5 = t1 * t2;
    }, std::runtime_error);

    EXPECT_THROW({
        cuda::Tensor t6 = t1 / t2;
    }, std::runtime_error);
}

TEST(TensorTest, InvalidIndexAccess) {
    std::vector<uint32_t> shape = {2, 2};
    cuda::Tensor t(shape, cuda::DataType::DataTypeInt32, cuda::DeviceType::CPU);
    t.fill(1);

    EXPECT_THROW({
        int value = t.at({2, 0});  // Out of bounds
    }, std::invalid_argument);

    EXPECT_THROW({
        int value = t.at({0, 2});  // Out of bounds
    }, std::invalid_argument);

    EXPECT_THROW({
        int value = t.at({0});  // Insufficient indices
    }, std::invalid_argument);

    EXPECT_THROW({
        int value = t.at({0, 0, 0});  // Too many indices
    }, std::invalid_argument);
}

TEST(TensorTest, NullDataPointer) {
    std::vector<uint32_t> shape = {2, 2};
    void* data = nullptr;

    EXPECT_THROW({
        cuda::Tensor t(shape, cuda::DataType::DataTypeFloat32, cuda::DeviceType::CPU, data, true);
    }, std::invalid_argument);
}

/********** Tensor fill functionality **********/
TEST(TensorTest, TensorFill) {
    std::vector<uint32_t> shape = {2, 2, 2};
    cuda::Tensor t(shape, cuda::DataType::DataTypeFloat32, cuda::DeviceType::CPU);

    t.fill(3.14f);

    for (uint32_t i = 0; i < shape[0]; ++i)
        for (uint32_t j = 0; j < shape[1]; ++j)
            for (uint32_t k = 0; k < shape[2]; ++k)
                EXPECT_FLOAT_EQ(t.at({i, j, k}), 3.14f);
}
