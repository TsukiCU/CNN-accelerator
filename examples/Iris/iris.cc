#include <iostream>
#include "model.h"
#include "loss.h"
#include "optimizer.h"

#include "iris_dataset.h"

using namespace snnf;
using namespace snnf::layer;
using namespace snnf::dataset;

const int num_epochs = 20;
const float train_pct = 0.8f;
const float learning_rate = 0.01f;

int main() {
    IrisDataset<float> dataset;
    dataset.load_data("data/iris.txt");

    // Load data, set batch size to 16, num_worker to 1, and shuffle.
    DataLoader<float> train_loader(dataset, 16, true, 1);

    Model<float> model;
    model.add_layer(std::make_shared<LinearLayer<float>>(4, 16));
    model.add_layer(std::make_shared<ReLULayer<float>>());
    model.add_layer(std::make_shared<LinearLayer<float>>(16, 3));
    model.add_layer(std::make_shared<SigmoidLayer<float>>());

    SGD<float> optimizer(learning_rate);
    optimizer.add_parameters(model.get_parameters());

    MSELoss<float> loss_fn;

    for (size_t epoch = 0; epoch < num_epochs; ++epoch) {
        train_loader.reset();
        float total_loss = 0.0f;
        size_t batch_count = 0;

        while (train_loader.has_next()) {
            auto batch = train_loader.next_batch();
            auto& input = batch.first;
            auto& target = batch.second;

            optimizer.zero_grad();

            Tensor<float> output = model.forward(input);
            float loss = loss_fn.forward(output, target);

            total_loss += loss;
            batch_count++;

            Tensor<float> loss_grad = loss_fn.backward();
            model.backward(loss_grad);

            optimizer.step();

        }

        std::cout << "Epoch " << epoch + 1 << " of " << num_epochs
                  << ", Loss: " << total_loss / batch_count << std::endl;
    }

    return 0;
}