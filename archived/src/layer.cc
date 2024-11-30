#include "../include/layer.h"
#include "../include/tensor.h"

namespace cuda {

LinearLayer::LinearLayer(uint32_t in_features, uint32_t out_features) {
    // randomized initialization.
    weight_ = Tensor::rand({in_features, out_features}, -0.1, 0.1);
    bias_ = Tensor::rand({out_features}, -0.1, 0.1);
    add_parameter("weight", weight_);   // weight_ and bias_ need to be optimized.
    add_parameter("bias", bias_);
}

Tensor LinearLayer::forward(const Tensor& input) {
    input_ = input;
    Tensor output = input.matmul(weight_);
    output = output.add(bias_);
    return output;
}

void LinearLayer::backward(const Tensor& grad_output) {
    Tensor grad_weight = input_.transpose().matmul(grad_output);
    Tensor grad_bias = grad_output.sum(0);

    weight_.grad() = grad_weight;
    bias_.grad() = grad_bias;

    Tensor grad_input = grad_output.matmul(weight_.transpose());
}


Tensor ReLULayer::forward(const Tensor& input) {
    mask_ = input > 0;

    Tensor output = input * mask_;

    return output;
}

void ReLULayer::backward(const Tensor& grad_output) {
    Tensor grad_input = grad_output * mask_;
}


}