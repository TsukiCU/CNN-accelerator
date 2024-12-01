#include "../include/layer.h"

namespace snnf {

namespace layer {

/**
 * @brief: Forward propagation for ReLU Layer.
 */
template <typename T>
Tensor<T> ReLULayer<T>::forward(const Tensor<T>& input) {
    input_ = input;  // Cache input for backward pass
    Tensor<T> output = input;
    // Apply ReLU activation: output = max(0, input)
    for (uint32_t i = 0; i < input.size(); ++i) {
        output.data()[i] = std::max(static_cast<T>(0), input.data()[i]);
    }
    return output;
}

/**
 * @brief: Backward propagation for ReLU Layer.
 */
template <typename T>
Tensor<T> ReLULayer<T>::backward(const Tensor<T>& grad_output) {
    Tensor<T> grad_input = grad_output;
    // Compute gradient: grad_input = grad_output * (input > 0)
    for (uint32_t i = 0; i < input_.size(); ++i) {
        grad_input.data()[i] = input_.data()[i] > static_cast<T>(0) ? grad_output.data()[i] : static_cast<T>(0);
    }
    return grad_input;
}


/**
 * @brief: Forward propagation for Sigmoid Layer.
 */
template <typename T>
Tensor<T> SigmoidLayer<T>::forward(const Tensor<T>& input) {
    output_ = input;
    // Apply Sigmoid activation: output = 1 / (1 + exp(-input))
    for (uint32_t i = 0; i < input.size(); ++i) {
        output_.data()[i] = static_cast<T>(1) / (static_cast<T>(1) + std::exp(-input.data()[i]));
    }
    return output_;
}

/**
 * @brief: Backward propagation for Sigmoid Layer.
 */
template <typename T>
Tensor<T> SigmoidLayer<T>::backward(const Tensor<T>& grad_output) {
    Tensor<T> grad_input(grad_output.shape());
    // Compute gradient: grad_input = grad_output * output * (1 - output)
    for (uint32_t i = 0; i < output_.size(); ++i) {
        grad_input.data()[i] = grad_output.data()[i] * output_.data()[i] * (static_cast<T>(1) - output_.data()[i]);
    }
    return grad_input;
}

// Explicit template instantiation
template class ReLULayer<float>;
template class ReLULayer<double>;
template class SigmoidLayer<float>;
template class SigmoidLayer<double>;

} // layer

} // snnf
