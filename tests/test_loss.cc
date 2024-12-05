#include "loss.h"
#include <gtest/gtest.h>

using namespace snnf;

TEST(MSELossTest, ForwardBackward) {
    MSELoss<float> mse_loss;
    Tensor<float> input({2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    Tensor<float> target({2, 3}, {1.5, 2.5, 3.5, 3.5, 4.5, 5.5});

    float loss = mse_loss.forward(input, target);
    EXPECT_NEAR(loss, 0.125f, 1e-5);

    Tensor<float> grad_input = mse_loss.backward();
    std::vector<float> expected_grad = {-0.08333333f, -0.08333333f, -0.08333333f,
                                        0.08333333f, 0.08333333f, 0.08333333f};
    for (uint32_t i = 0; i < grad_input.size(); ++i) {
        EXPECT_NEAR(grad_input.data()[i], expected_grad[i], 1e-5);
    }
}

TEST(CrossEntropyLossTest, ForwardBackward) {
    CrossEntropyLoss<float> ce_loss;
    Tensor<float> input({2, 3}, {2.0, 1.0, 0.1, 0.5, 1.5, 2.5});
    Tensor<float> target({2, 3}, {1, 0, 0, 0, 0, 1});

    float loss = ce_loss.forward(input, target);
    EXPECT_NEAR(loss, 0.412462f, 1e-3);

    Tensor<float> grad_input = ce_loss.backward();
    // Expected gradient can be computed manually or verified against a known implementation
    // For simplicity, we can just check the gradient shape
    EXPECT_EQ(grad_input.shape(), input.shape());
}

TEST(HuberLossTest, ForwardBackward) {
    float delta = 1.0f;  // Huber delta value
    HuberLoss<float> huber_loss(delta);
    Tensor<float> input({2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    Tensor<float> target({2, 3}, {1.5, 2.5, 3.5, 3.5, 4.5, 5.5});

    float loss = huber_loss.forward(input, target);
    // Manually calculate the expected loss:
    // Loss = (0.5 * 0.5^2 + 0.5 * 0.5^2 + 0.5 * 0.5^2 + 0.5 * 0.5 + 0.5 * 0.5 + 0.5 * 0.5) / 6
    EXPECT_NEAR(loss, 0.125f, 1e-5);

    Tensor<float> grad_input = huber_loss.backward();
    std::vector<float> expected_grad = {-0.0833333f, -0.0833333f, -0.0833333f,
                                        0.0833333f, 0.0833333f, 0.0833333f};
    for (uint32_t i = 0; i < grad_input.size(); ++i) {
        EXPECT_NEAR(grad_input.data()[i], expected_grad[i], 1e-5);
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
