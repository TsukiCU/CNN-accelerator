#pragma once

#include "log.h"
#include "utils.h"
#include "tensor.h"

namespace snnf {

/*
 * @brief : base class for all layers.
 */
template <typename T>
class Layer {
public:
    virtual ~Layer() = default;

    virtual Tensor<T> forward(const Tensor<T>& input) = 0;

    virtual Tensor<T> backward(const Tensor<T>& grad_output) = 0;

    virtual std::vector<Tensor<T>*> get_parameters() { return {}; }

    virtual void zero_grad() {}
};

/*
 * @brief : Linear class for fully connected layer.
 */
template <typename T>
class LinearLayer : public Layer<T> {
public:
    LinearLayer(uint32_t in_features, uint32_t out_features);

    Tensor<T> forward(const Tensor<T>& input) override;
    Tensor<T> backward(const Tensor<T>& grad_output) override;

    std::vector<Tensor<T>*> get_parameters() override;
    void zero_grad() override;

private:
    uint32_t in_features_;
    uint32_t out_features_;
    Tensor<T> weights_;
    Tensor<T> bias_;
    Tensor<T> input_;

    Tensor<T> grad_weights_;
    Tensor<T> grad_bias_;
};


/*********************** Impl of LinearLayer methods ***********************/

/* 
 * @brief : initialzie weights randomly and bias to 0.
 * @todo : support other initialization methods. 
 */
template <typename T>
LinearLayer<T>::LinearLayer(uint32_t in_features, uint32_t out_features)
    : in_features_(in_features),
      out_features_(out_features),
      weights_({in_features, out_features}),
      bias_({1, out_features}),
      input_({1, in_features}),
      grad_weights_({in_features, out_features}),
      grad_bias_({1, out_features})
{
    weights_.random();
    bias_.fill(static_cast<T>(0));
}

/*
 * @brief : Forward propagation
 */
template <typename T>
Tensor<T> LinearLayer<T>::forward(const Tensor<T>& input)
{
    input_ = input; 
    Tensor<T> output = input.matmul(weights_);  // Linear transformation.
    if (output.shape() == bias_.shape()) {
        output = output.add(bias_);
    } else {
        LOG_INFO("Layer::forward : Broadcasting bias to a new shape.");
        Tensor<T> new_bias = bias_.broadcast_to(output.shape());
        output = output.add(new_bias);
    }
    return output;
}

/*
 * @brief : Backward propagation
 */
template <typename T>
Tensor<T> LinearLayer<T>::backward(const Tensor<T>& grad_output)
{
    // Compute gradients w.r.t weights and bias
    grad_weights_ = input_.transpose().matmul(grad_output);
    grad_bias_ = grad_output.sum(0);
    grad_bias_.reshape({1, grad_output.shape()[1]});

    // Compute gradient w.r.t input to pass to previous layer
    Tensor<T> grad_input = grad_output.matmul(weights_.transpose());
    return grad_input;
}

/*
 * @brief : Get weights and bisa
 */
template <typename T>
std::vector<Tensor<T>*> LinearLayer<T>::get_parameters()
{
    return { &weights_, &bias_, &grad_weights_, &grad_bias_ };
}

/*
 * @brief : Clear up grad_weights and grad_bias.
 */
template <typename T>
void LinearLayer<T>::zero_grad()
{
    grad_weights_.fill(static_cast<T>(0));
    grad_bias_.fill(static_cast<T>(0));
}

} // snnf

