#include <iostream>
#include "model.h"
#include "loss.h"
#include "optimizer.h"

#include "mnist_dataset.h"

using namespace snnf;
using namespace snnf::layer;
using namespace snnf::dataset;

const float learning_rate = 0.01;
const int num_epochs = 1;

const std::string train_file = "data/train-images-idx3-ubyte";
const std::string label_file = "data/train-labels-idx1-ubyte";

int main()
{
    MNISTDataset<float> dataset;
    dataset.load_data(train_file, label_file);
    DataLoader<float> train_loader(dataset, 64, true);

    Model<float> model;
    model.add_layer(std::make_shared<LinearLayer<float>>(784, 64));
    model.add_layer(std::make_shared<ReLULayer<float>>());
    model.add_layer(std::make_shared<LinearLayer<float>>(64, 10));
    model.add_layer(std::make_shared<ReLULayer<float>>());

    SGD<float> optimizer(learning_rate);
    optimizer.add_parameters(model.get_parameters());
    MSELoss<float> loss_fn;

    for (size_t epoch = 0; epoch < num_epochs; ++epoch) {
        train_loader.reset();
        float total_loss = 0.0f;
        size_t batch_count = 0;

        int round = 0;

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
            if (!(round % 10)) {
                std::cout << "Epoch " << epoch + 1 << " Round " << round
                  << ", Loss: " << total_loss / batch_count << std::endl;
            }
            ++round;
        }

        // std::cout << "Epoch " << epoch + 1 << " of " << num_epochs
        //           << ", Loss: " << total_loss / batch_count << std::endl;
    }
}