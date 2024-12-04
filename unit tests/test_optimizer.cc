#include <gtest/gtest.h>
#include "layer.h"
#include "optimizer.h"

using namespace snnf;
using namespace snnf::layer;

TEST(SGDOptimizerTest, ParameterUpdate) {
    // Manually set w.b.g for a simple linear layer.
    LinearLayer<float> layer(2, 2);
    layer.get_parameters()[0]->data() = {1.0, 2.0, 3.0, 4.0};
    layer.get_parameters()[1]->data() = {0.5, -0.5};
    layer.get_parameters()[0]->grad().data() = {0.1, 0.1, 0.1, 0.1};
    layer.get_parameters()[1]->grad().data() = {0.05, -0.05};

    SGD<float> optimizer(0.1f);
    optimizer.add_parameters(layer.get_parameters());

    optimizer.step();

    // Check updated weights and biases
    std::vector<float> expected_weights = {0.99f, 1.99f, 2.99f, 3.99f};
    std::vector<float> expected_biases = {0.495f, -0.495f};
    auto weights = layer.get_parameters()[0];
    auto biases = layer.get_parameters()[1];

    for (uint32_t i = 0; i < weights->size(); ++i) {
        EXPECT_NEAR(weights->data()[i], expected_weights[i], 1e-5);
    }
    for (uint32_t i = 0; i < biases->size(); ++i) {
        EXPECT_NEAR(biases->data()[i], expected_biases[i], 1e-5);
    }
}