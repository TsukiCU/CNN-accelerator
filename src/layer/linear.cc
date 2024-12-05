#include "layer.h"

namespace snnf {

namespace layer {

/**
 * @brief : Initialzie weights randomly and bias to 0.
 * @param : Size of the input and output features.
 * @todo : Support other initialization methods. 
 */
template <typename T>
LinearLayer<T>::LinearLayer(uint32_t in_features, uint32_t out_features, InitMethod init)
    : in_features_(in_features),
      out_features_(out_features),
      weights_({in_features, out_features}),
      bias_({1, out_features}),
      input_({1, in_features}),
      grad_weights_({in_features, out_features}),
      grad_bias_({1, out_features})
{
    switch (init) {
        case InitMethod::Uniform:
            weights_.random(-0.1, 0.1);
            break;
        case InitMethod::Gaussian:
            weights_.random_normal(0, 0.01);
            break;
        case InitMethod::Kaiming: {
            float stddev = std::sqrt(2.0f / in_features);
            weights_.random_normal(0, stddev);
            break;
        }
        case InitMethod::Xavier: {
            float limit = std::sqrt(6.0f / (in_features + out_features));
            weights_.random(-limit, limit);
            break;
        }
        default:
            LOG_ERROR(std::invalid_argument, "LinearLayer::LinearLayer : Unknown initialization method.");
    }

    bias_.fill(static_cast<T>(0));
}


/**
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

/**
 * @brief : Backward propagation
 */
template <typename T>
Tensor<T> LinearLayer<T>::backward(const Tensor<T>& grad_output)
{
    // Compute gradients w.r.t weights and bias
    grad_weights_ = input_.transpose().matmul(grad_output);
    grad_bias_ = grad_output.sum(0);
    grad_bias_.reshape({1, grad_output.shape()[1]});

    // Get the gradient by calling grad() instead of accessing directly.
    weights_.set_grad(grad_weights_);
    bias_.set_grad(grad_bias_);

    // Compute gradient w.r.t input to pass to previous layer
    Tensor<T> grad_input = grad_output.matmul(weights_.transpose());
    return grad_input;
}

/**
 * @brief : Get weights and bias for each layer.
 */
template <typename T>
std::vector<Tensor<T>*> LinearLayer<T>::get_parameters()
{
    return { &weights_, &bias_ };
}

/**
 * @brief : Clear up grad_weights and grad_bias.
 */
template <typename T>
void LinearLayer<T>::zero_grad()
{
    weights_.zero_grad();
    bias_.zero_grad();
    grad_weights_.fill(static_cast<T>(0));
    grad_bias_.fill(static_cast<T>(0));
}

// Explicit template instatiation
/// @todo : Include more types.

template class LinearLayer<float>;
template class LinearLayer<double>;

} // layer

} // snnf