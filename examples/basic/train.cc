#include "model.h"
#include "loss.h"
#include "dataset.h"

using namespace snnf;
using namespace snnf::layer;

const int num_epochs = 100;
const float learning_rate = 0.01;

int main()
{
    Model<float> model;
    model.add_layer(std::make_shared<LinearLayer<float>>(2, 4));
    model.add_layer(std::make_shared<ReLULayer<float>>());
    model.add_layer(std::make_shared<LinearLayer<float>>(4, 2));

    MSELoss<float> loss_fn;
    SGD<float> optimizer(learning_rate);
    optimizer.add_parameters(model.get_parameters());

    Tensor<float> input({1, 2});
    Tensor<float> target({1, 2});

    input.random(-2, 2);

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        Tensor<float> output = model.forward(input);

        float loss = loss_fn.forward(output, target);

        model.zero_grad();

        Tensor<float> loss_grad = loss_fn.backward();
        model.backward(loss_grad);

        optimizer.step();

        std::cout << "Epoch " << epoch << ", Loss: " << loss << std::endl;
    }
}