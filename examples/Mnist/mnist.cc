#include <iostream>
#include "model.h"
#include "loss.h"
#include "optimizer.h"

#include "mnist_dataset.h"

using namespace snnf;
using namespace snnf::layer;
using namespace snnf::dataset;

const float learning_rate = 0.01;
const int num_epochs = 0;

const std::string train_image = "data/train-images-idx3-ubyte";
const std::string train_label = "data/train-labels-idx1-ubyte";
const std::string test_image  = "data/t10k-images-idx3-ubyte";
const std::string test_label  = "data/t10k-labels-idx1-ubyte";

int main()
{
    // ******************* training *******************
    MNISTDataset<float> train_dataset;
    train_dataset.load_data(train_image, train_label);
    train_dataset.normalize(0.5, 0.5);
    DataLoader<float> train_loader(train_dataset, 64, true);

    Model<float> model;
    model.add_layer(std::make_shared<LinearLayer<float>>(784, 128, InitMethod::Kaiming));
    model.add_layer(std::make_shared<ReLULayer<float>>());
    model.add_layer(std::make_shared<LinearLayer<float>>(128, 64, InitMethod::Kaiming));
    model.add_layer(std::make_shared<ReLULayer<float>>());
    model.add_layer(std::make_shared<LinearLayer<float>>(64, 10));
    model.add_layer(std::make_shared<SoftmaxLayer<float>>());

    SGD<float> optimizer(learning_rate);
    optimizer.add_parameters(model.get_parameters());
    CrossEntropyLoss<float> loss_fn;

    for (size_t epoch = 0; epoch < num_epochs; ++epoch) {
        std::cout << "\e[1;36m\nEpoch " << epoch + 1 << " training starts.\e[0m\n" << std::endl;
        train_loader.reset();
        float total_loss = 0.0f;
        size_t batch_cnt = 0;

        while (train_loader.has_next()) {
            auto batch = train_loader.next_batch();
            auto& input = batch.first;
            auto& target = batch.second;

            optimizer.zero_grad();

            Tensor<float> output = model.forward(input);
            float loss = loss_fn.forward(output, target);

            total_loss += loss;

            Tensor<float> loss_grad = loss_fn.backward();
            model.backward(loss_grad);

            optimizer.step();
            if (batch_cnt % 200 == 0) {
                std::cout << "Epoch " << epoch + 1 << " Batch " << batch_cnt
                  << ", Loss: " << total_loss / batch_cnt << std::endl;
            }
            ++batch_cnt;
        }
    }

    // ******************* testing *******************
    MNISTDataset<float> test_dataset;
    test_dataset.load_data(test_image, test_label);
    test_dataset.normalize(0.5, 0.5);
    DataLoader<float> test_loader(test_dataset, 1, false);

    int correct = 0, total = 0;
    while (test_loader.has_next()) {
        auto batch = test_loader.next_batch();
        auto& input = batch.first;
        auto& target = batch.second;
        Tensor<float> output = model.forward(input);

        uint32_t pred_index = 0;
        float max_val = output.at({0, 0});
        for (uint32_t i = 0; i < 10; ++i) {
            if (output.at({0, i}) > max_val) {
                max_val = output.at({0, i});
                pred_index = i;
            }
        }

        uint32_t target_index = 0;
        for (uint32_t i = 0; i < 10; ++i) {
            if (target.at({0, i}) == 1) {
                target_index = i;
                break;
            }
        }

        if (target_index == pred_index)
            ++correct;
        ++total;
    }

    float accuracy = static_cast<float>(correct) / static_cast<float>(total);
    std::cout << "\e[1;31m\nTest Accuracy: " << accuracy * 100.0f << "%\n\e[0m" << std::endl;

    model.save_parameters("param/mnist.bin");

    return 0;
}