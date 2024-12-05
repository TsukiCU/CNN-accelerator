#include <gtest/gtest.h>
#include "layer.h"

using namespace snnf;
using namespace snnf::layer;

/// @brief Linear Layer Test Unit.

// ========== LinearLayer Constructor Test ============

TEST(LinearLayerTest, ConstructorInitializesParameters) {
    LinearLayer<float> layer(4, 3);
    EXPECT_EQ(layer.get_parameters().size(), 2);  // weights, bias.

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

    auto grad_weights = layer.get_parameters()[0]->grad();
    auto grad_bias = layer.get_parameters()[1]->grad();

    EXPECT_EQ(grad_weights.shape(), (std::vector<uint32_t>{4, 3}));  // Grad weights shape [4, 3]
    EXPECT_EQ(grad_bias.shape(), (std::vector<uint32_t>{1, 3}));    // Grad bias shape [1, 3]
}

// ========== LinearLayer ZeroGrad Test ============

TEST(LinearLayerTest, ZeroGrad) {
    LinearLayer<float> layer(4, 3);
    Tensor<float> grad_output({2, 3}, {0.1, 0.2, 0.3, 0.4, 0.5, 0.6});

    layer.forward(Tensor<float>({2, 4}, {1, 2, 3, 4, 5, 6, 7, 8}));
    layer.backward(grad_output);

    layer.zero_grad();
    auto grad_weights = layer.get_parameters()[0]->grad();
    auto grad_bias = layer.get_parameters()[1]->grad();

    EXPECT_EQ(grad_weights.data(), std::vector<float>(grad_weights.size(), 0.0f));
    EXPECT_EQ(grad_bias.data(), std::vector<float>(grad_bias.size(), 0.0f));
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

    auto grad_weights = layer.get_parameters()[0]->grad();
    auto grad_bias = layer.get_parameters()[1]->grad();

    EXPECT_EQ(grad_weights.shape(), (std::vector<uint32_t>{2, 2}));
    EXPECT_EQ(grad_bias.shape(), (std::vector<uint32_t>{1, 2}));
}


/// @brief Activation Layer Test Unit.

// ========== ReLU Layer Forward Test ===========

TEST(ReLULayerTest, ForwardPass) {
    ReLULayer<float> relu;
    Tensor<float> input({2, 3}, {-1.0, 0.0, 1.0, 2.0, -2.0, 3.0});
    Tensor<float> output = relu.forward(input);

    std::vector<float> expected = {0.0, 0.0, 1.0, 2.0, 0.0, 3.0};
    for (uint32_t i = 0; i < output.size(); ++i) {
        EXPECT_EQ(output.data()[i], expected[i]);
    }
}

// ========== ReLU Layer Backward Test ===========

TEST(ReLULayerTest, BackwardPass) {
    ReLULayer<float> relu;
    Tensor<float> input({2, 3}, {-1.0, 0.0, 1.0, 2.0, -2.0, 3.0});
    relu.forward(input);

    Tensor<float> grad_output({2, 3}, {1.0, 1.0, 1.0, 1.0, 1.0, 1.0});
    Tensor<float> grad_input = relu.backward(grad_output);

    std::vector<float> expected = {0.0, 0.0, 1.0, 1.0, 0.0, 1.0};
    for (uint32_t i = 0; i < grad_input.size(); ++i) {
        EXPECT_EQ(grad_input.data()[i], expected[i]);
    }
}

// ========== Sigmoid Layer Forward Test ===========

TEST(SigmoidLayerTest, ForwardPass) {
    SigmoidLayer<float> sigmoid;
    Tensor<float> input({2, 3}, {0.0, 1.0, -1.0, 2.0, -2.0, 3.0});
    Tensor<float> output = sigmoid.forward(input);

    for (uint32_t i = 0; i < output.size(); ++i) {
        float expected = 1.0f / (1.0f + std::exp(-input.data()[i]));
        EXPECT_NEAR(output.data()[i], expected, 1e-5);
    }
}

// ========== Sigmoid Layer Backward Test ===========

TEST(SigmoidLayerTest, BackwardPass) {
    SigmoidLayer<float> sigmoid;
    Tensor<float> input({2, 3}, {0.0, 1.0, -1.0, 2.0, -2.0, 3.0});
    Tensor<float> output = sigmoid.forward(input);

    Tensor<float> grad_output({2, 3}, {1.0, 1.0, 1.0, 1.0, 1.0, 1.0});
    Tensor<float> grad_input = sigmoid.backward(grad_output);

    for (uint32_t i = 0; i < grad_input.size(); ++i) {
        float sig = output.data()[i];
        float expected = grad_output.data()[i] * sig * (1.0f - sig);
        EXPECT_NEAR(grad_input.data()[i], expected, 1e-5);
    }
}

// ========== Softmax Layer Backward Test ===========

TEST(SoftmaxLayerTest, ForwardPass) {
    SoftmaxLayer<float> softmax;
    Tensor<float> input({2, 3}, {1.0, 2.0, 3.0, 1.0, 2.0, 3.0});
    Tensor<float> output = softmax.forward(input);

    for (uint32_t i = 0; i < input.shape()[0]; ++i) {
        float sum_exp = 0;
        for (uint32_t j = 0; j < input.shape()[1]; ++j) {
            sum_exp += std::exp(input.at({i, j}));
        }
        for (uint32_t j = 0; j < input.shape()[1]; ++j) {
            float expected = std::exp(input.at({i, j})) / sum_exp;
            EXPECT_NEAR(output.at({i, j}), expected, 1e-5);
        }
    }
}

TEST(SoftmaxLayerTest, BackwardPass) {
    SoftmaxLayer<float> softmax;
    Tensor<float> input({1, 3}, {1.0, 2.0, 3.0});
    Tensor<float> output = softmax.forward(input);

    Tensor<float> grad_output({1, 3}, {0.1, 0.2, 0.3});
    Tensor<float> grad_input = softmax.backward(grad_output);

    // Only verify shape here as well.
    EXPECT_EQ(grad_input.shape(), input.shape());
}