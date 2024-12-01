#include <gtest/gtest.h>
#include "../include/layer.h"

using namespace snnf;

// ========== LinearLayer Constructor Test ============

TEST(LinearLayerTest, ConstructorInitializesParameters) {
    LinearLayer<float> layer(4, 3);
    EXPECT_EQ(layer.get_parameters().size(), 4);  // weights, bias, grad_weights, grad_bias

    auto weights = layer.get_parameters()[0];
    auto bias = layer.get_parameters()[1];
    EXPECT_EQ(weights->shape(), (std::vector<uint32_t>{4, 3}));
    EXPECT_EQ(bias->shape(), (std::vector<uint32_t>{1, 3}));
}

// ========== LinearLayer Forward Test ============

TEST(LinearLayerTest, ForwardPass) {
    LinearLayer<float> layer(4, 3);
    Tensor<float> input({2, 4}, {1, 2, 3, 4, 5, 6, 7, 8});  // Shape [2, 4]
    Tensor<float> output = layer.forward(input);
    EXPECT_EQ(output.shape(), (std::vector<uint32_t>{2, 3}));  // Output shape [2, 3]
}

// ========== LinearLayer Backward Test ============

TEST(LinearLayerTest, BackwardPass) {
    LinearLayer<float> layer(4, 3);
    Tensor<float> input({2, 4}, {1, 2, 3, 4, 5, 6, 7, 8});  // Shape [2, 4]
    Tensor<float> grad_output({2, 3}, {0.1, 0.2, 0.3, 0.4, 0.5, 0.6});  // Shape [2, 3]

    layer.forward(input);  // Cache input for backward pass
    Tensor<float> grad_input = layer.backward(grad_output);

    EXPECT_EQ(grad_input.shape(), (std::vector<uint32_t>{2, 4}));  // Grad input shape [2, 4]

    auto grad_weights = layer.get_parameters()[2];
    auto grad_bias = layer.get_parameters()[3];

    EXPECT_EQ(grad_weights->shape(), (std::vector<uint32_t>{4, 3}));  // Grad weights shape [4, 3]
    EXPECT_EQ(grad_bias->shape(), (std::vector<uint32_t>{1, 3}));    // Grad bias shape [1, 3]
}

// ========== LinearLayer ZeroGrad Test ============

TEST(LinearLayerTest, ZeroGrad) {
    LinearLayer<float> layer(4, 3);
    Tensor<float> grad_output({2, 3}, {0.1, 0.2, 0.3, 0.4, 0.5, 0.6});

    layer.forward(Tensor<float>({2, 4}, {1, 2, 3, 4, 5, 6, 7, 8}));
    layer.backward(grad_output);

    auto grad_weights = layer.get_parameters()[2];
    auto grad_bias = layer.get_parameters()[3];

    layer.zero_grad();

    EXPECT_EQ(grad_weights->data(), std::vector<float>(grad_weights->size(), 0.0f));
    EXPECT_EQ(grad_bias->data(), std::vector<float>(grad_bias->size(), 0.0f));
}

// ========== LinearLayer Integration Test ============

TEST(LinearLayerTest, FullIntegration) {
    LinearLayer<float> layer(2, 2);
    Tensor<float> input({2, 2}, {1.0, 2.0, 3.0, 4.0});
    Tensor<float> grad_output({2, 2}, {0.1, 0.2, 0.3, 0.4});

    Tensor<float> output = layer.forward(input);
    Tensor<float> grad_input = layer.backward(grad_output);

    EXPECT_EQ(output.shape(), (std::vector<uint32_t>{2, 2}));
    EXPECT_EQ(grad_input.shape(), (std::vector<uint32_t>{2, 2}));

    auto grad_weights = layer.get_parameters()[2];
    auto grad_bias = layer.get_parameters()[3];

    EXPECT_EQ(grad_weights->shape(), (std::vector<uint32_t>{2, 2}));
    EXPECT_EQ(grad_bias->shape(), (std::vector<uint32_t>{1, 2}));
}