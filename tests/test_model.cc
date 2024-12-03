#include <gtest/gtest.h>
#include "../include/model.h"
#include "../include/layer.h"
#include "../include/loss.h"

#include <iostream>

using namespace snnf;
using namespace snnf::layer;

TEST(ModelTest, ForwardBackward) {
    Tensor<float> input({1, 2}, {1.0f, 2.0f});
    Tensor<float> target({1, 1}, {0.5f});

    Model<float> model;
    MSELoss<float> loss_fn;
    model.add_layer(std::make_shared<LinearLayer<float>>(2, 3));
    model.add_layer(std::make_shared<ReLULayer<float>>());
    model.add_layer(std::make_shared<LinearLayer<float>>(3, 1));

    SGD<float> optimizer(0.01f);
    optimizer.add_parameters(model.get_parameters());

    optimizer.zero_grad();
    Tensor<float> output = model.forward(input);

    // float loss = loss_fn.forward(output, target), expected = 0.783039;
    // EXPECT_NEAR(loss, 0.783039, 1e-5);

    Tensor<float> loss_grad = loss_fn.backward();
    model.backward(loss_grad);

    optimizer.step();

    // Check the output.
    EXPECT_EQ(output.shape(), (std::vector<uint32_t>{1, 1}));
}