#include <gtest/gtest.h>
#include "tensor.h"

using namespace snnf;

// ========== Tensor Constructors ============

TEST(TensorTest, ConstructorWithShape) {
    Tensor<float> tensor({2, 3});
    EXPECT_EQ(tensor.shape(), (std::vector<uint32_t>{2, 3}));
    EXPECT_EQ(tensor.size(), 6);
    EXPECT_EQ(tensor.data().size(), 6);
}

TEST(TensorTest, ConstructorWithShapeAndData) {
    std::vector<float> data = {1, 2, 3, 4, 5, 6};
    Tensor<float> tensor({2, 3}, data);
    EXPECT_EQ(tensor.shape(), (std::vector<uint32_t>{2, 3}));
    EXPECT_EQ(tensor.size(), 6);
    EXPECT_EQ(tensor.data(), data);
}

TEST(TensorTest, ConstructorDataSizeMismatch) {
    std::vector<float> data = {1, 2, 3};
    EXPECT_THROW(Tensor<float> tensor({2, 3}, data), std::invalid_argument);
}

// ========== Element Access ============

TEST(TensorTest, AtFunction) {
    Tensor<float> tensor({2, 2});
    tensor.fill(0.0f);
    tensor.at({0, 1}) = 5.0f;
    EXPECT_EQ(tensor.at({0, 1}), 5.0f);
    EXPECT_THROW(tensor.at({2, 0}), std::out_of_range);
    EXPECT_THROW(tensor.at({0}), std::invalid_argument);
}

TEST(TensorTest, AtFunctionConst) {
    const Tensor<float> tensor({2, 2}, {1, 2, 3, 4});
    EXPECT_EQ(tensor.at({1, 0}), 3.0f);
}

// ========== Basic Operations ============

TEST(TensorTest, AddFunction) {
    Tensor<float> tensor1({2, 2}, {1, 2, 3, 4});
    Tensor<float> tensor2({2, 2}, {5, 6, 7, 8});
    Tensor<float> result = tensor1.add(tensor2);
    std::vector<float> expected_data = {6, 8, 10, 12};
    EXPECT_EQ(result.data(), expected_data);
}

TEST(TensorTest, SubtractFunction) {
    Tensor<float> tensor1({2, 2}, {5, 6, 7, 8});
    Tensor<float> tensor2({2, 2}, {1, 2, 3, 4});
    Tensor<float> result = tensor1.subtract(tensor2);
    std::vector<float> expected_data = {4, 4, 4, 4};
    EXPECT_EQ(result.data(), expected_data);
}

TEST(TensorTest, MultiplyFunction) {
    Tensor<float> tensor1({2, 2}, {1, 2, 3, 4});
    Tensor<float> tensor2({2, 2}, {2, 0.5, 1.5, 2});
    Tensor<float> result = tensor1.multiply(tensor2);
    std::vector<float> expected_data = {2, 1, 4.5, 8};
    EXPECT_EQ(result.data(), expected_data);
}

TEST(TensorTest, DivideFunction) {
    Tensor<float> tensor1({2, 2}, {2, 4, 6, 8});
    Tensor<float> tensor2({2, 2}, {2, 2, 2, 2});
    Tensor<float> result = tensor1.divide(tensor2);
    std::vector<float> expected_data = {1, 2, 3, 4};
    EXPECT_EQ(result.data(), expected_data);
}

TEST(TensorTest, DivideByZero) {
    Tensor<float> tensor1({2, 2}, {2, 4, 6, 8});
    Tensor<float> tensor2({2, 2}, {1, 0, 1, 1});
    EXPECT_THROW(tensor1.divide(tensor2), std::runtime_error);
}

// ========== Matrix Multiplication ============

TEST(TensorTest, MatmulFunction) {
    Tensor<float> tensor1({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor<float> tensor2({3, 2}, {7, 8, 9, 10, 11, 12});
    Tensor<float> result = tensor1.matmul(tensor2);
    std::vector<float> expected_data = {58, 64, 139, 154};
    EXPECT_EQ(result.shape(), (std::vector<uint32_t>{2, 2}));
    EXPECT_EQ(result.data(), expected_data);
}

TEST(TensorTest, MatmulDimensionMismatch) {
    Tensor<float> tensor1({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor<float> tensor2({2, 2}, {7, 8, 9, 10});
    EXPECT_THROW(tensor1.matmul(tensor2), std::invalid_argument);
}

TEST(TensorTest, MatmulNon2D) {
    Tensor<float> tensor1({2}, {1, 2});
    Tensor<float> tensor2({2}, {3, 4});
    EXPECT_THROW(tensor1.matmul(tensor2), std::invalid_argument);
}

// ========== Transpose Function ============

TEST(TensorTest, TransposeFunction) {
    Tensor<float> tensor({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor<float> result = tensor.transpose();
    std::vector<float> expected_data = {1, 4, 2, 5, 3, 6};
    EXPECT_EQ(result.shape(), (std::vector<uint32_t>{3, 2}));
    EXPECT_EQ(result.data(), expected_data);
}

TEST(TensorTest, TransposeNon2D) {
    Tensor<float> tensor({2, 2, 2});
    EXPECT_THROW(tensor.transpose(), std::invalid_argument);
}

// ========== Sum Function ============

TEST(TensorTest, SumFunction) {
    Tensor<float> tensor({2, 2}, {1, 2, 3, 4});
    Tensor<float> result0 = tensor.sum(0);
    Tensor<float> result1 = tensor.sum(1);
    EXPECT_EQ(result0.shape(), (std::vector<uint32_t>{2}));
    EXPECT_EQ(result0.data(), (std::vector<float>{4, 6}));
    EXPECT_EQ(result1.shape(), (std::vector<uint32_t>{2}));
    EXPECT_EQ(result1.data(), (std::vector<float>{3, 7}));
}

TEST(TensorTest, SumInvalidDimension) {
    Tensor<float> tensor({2, 2});
    EXPECT_THROW(tensor.sum(2), std::invalid_argument);
}

// ========== Broadcasting ============

TEST(TensorTest, BroadcastToFunction) {
    Tensor<float> tensor({1, 3}, {1, 2, 3});
    Tensor<float> result = tensor.broadcast_to({2, 3});
    std::vector<float> expected_data = {1, 2, 3, 1, 2, 3};
    EXPECT_EQ(result.shape(), (std::vector<uint32_t>{2, 3}));
    EXPECT_EQ(result.data(), expected_data);
}

TEST(TensorTest, BroadcastInvalidShape) {
    Tensor<float> tensor({2, 1});
    EXPECT_THROW(tensor.broadcast_to({2}), std::invalid_argument);
}

// ========== Fill and Reshape Functions ============

TEST(TensorTest, FillFunction) {
    Tensor<float> tensor({2, 2});
    tensor.fill(3.14f);
    std::vector<float> expected_data = {3.14f, 3.14f, 3.14f, 3.14f};
    EXPECT_EQ(tensor.data(), expected_data);
}

TEST(TensorTest, ReshapeFunction) {
    Tensor<float> tensor({2, 3}, {1, 2, 3, 4, 5, 6});
    tensor.reshape({3, 2});
    EXPECT_EQ(tensor.shape(), (std::vector<uint32_t>{3, 2}));
    EXPECT_EQ(tensor.data(), (std::vector<float>{1, 2, 3, 4, 5, 6}));
}

TEST(TensorTest, ReshapeInvalidSize) {
    Tensor<float> tensor({2, 3});
    EXPECT_THROW(tensor.reshape({4, 2}), std::invalid_argument);
}

// ========== Gradient Support ============

TEST(TensorTest, GradFunction) {
    Tensor<float> tensor({2, 2});
    tensor.fill(1.0f);
    tensor.grad().fill(0.5f);
    EXPECT_EQ(tensor.grad().data(), (std::vector<float>{0.5f, 0.5f, 0.5f, 0.5f}));
}

TEST(TensorTest, ZeroGradFunction) {
    Tensor<float> tensor({2, 2});
    tensor.grad().fill(0.5f);
    tensor.zero_grad();
    EXPECT_EQ(tensor.grad().data(), (std::vector<float>{0.0f, 0.0f, 0.0f, 0.0f}));
}