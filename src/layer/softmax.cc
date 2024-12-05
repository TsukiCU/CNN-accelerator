#include "layer.h"

namespace snnf {

namespace layer {

template <typename T>
Tensor<T> SoftmaxLayer<T>::forward(const Tensor<T>& input) {
    output_ = input;
    uint32_t batch_size = input.shape()[0];
    uint32_t num_classes = input.shape()[1];

    // Same trick as in CrossEntropyLoss. Avoid exponential overflow.
    for (uint32_t i = 0; i < batch_size; ++i) {
        T max_val = input.at({i, 0});
        for (uint32_t j = 1; j < num_classes; ++j) {
            T val = input.at({i, j});
            if (val > max_val) {
                max_val = val;
            }
        }
        T sum_exp = static_cast<T>(0);
        for (uint32_t j = 0; j < num_classes; ++j) {
            T exp_val = std::exp(input.at({i, j}) - max_val);
            output_.data()[i * num_classes + j] = exp_val;
            sum_exp += exp_val;
        }
        for (uint32_t j = 0; j < num_classes; ++j) {
            output_.data()[i * num_classes + j] /= sum_exp;
        }
    }
    return output_;
}

template <typename T>
Tensor<T> SoftmaxLayer<T>::backward(const Tensor<T>& grad_output) {
    uint32_t batch_size = output_.shape()[0];
    uint32_t num_classes = output_.shape()[1];
    Tensor<T> grad_input(output_.shape());

    for (uint32_t i = 0; i < batch_size; ++i) {
        for (uint32_t j = 0; j < num_classes; ++j) {
            T grad = static_cast<T>(0);
            for (uint32_t k = 0; k < num_classes; ++k) {
                T delta = (j == k) ? static_cast<T>(1) : static_cast<T>(0);
                T partial = output_.at({i, j}) * (delta - output_.at({i, k}));
                grad += grad_output.at({i, k}) * partial;
            }
            grad_input.data()[i * num_classes + j] = grad;
        }
    }

    return grad_input;
}

template class SoftmaxLayer<float>;
template class SoftmaxLayer<double>;

} // layer

} // snnf