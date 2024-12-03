#pragma once

#include "tensor.h"
#include "layer.h"
#include "optimizer.h"

namespace snnf {

namespace layer {

template <typename T>
class Model {
public:
    Model() = default;
    ~Model() = default;

    /// @brief: Add a layer to the model
    void add_layer(std::shared_ptr<Layer<T>> layer);

    /// @brief: Forward pass
    Tensor<T> forward(const Tensor<T>& input);

    /// @brief: Backward pass
    void backward(const Tensor<T>& loss_grad);

    /// @brief: Get model parameters
    std::vector<Tensor<T>*> get_parameters();

    /// @brief: Zero gradients
    void zero_grad();

private:
    std::vector<std::shared_ptr<Layer<T>>> layers_;
    Tensor<T> output_; // Stores the output of the last layer.
};

} // layer

} // snnf