// optimizer.cc

#include "optimizer.h"

namespace snnf {

template <typename T>
void Optimizer<T>::add_parameters(const std::vector<Tensor<T>*>& params) {
    parameters_.insert(parameters_.end(), params.begin(), params.end());
}

template <typename T>
void Optimizer<T>::zero_grad() {
    for (auto& param : parameters_) {
        param->grad().fill(static_cast<T>(0));
    }
}


template <typename T>
SGD<T>::SGD(T learning_rate)
    : learning_rate_(learning_rate) {}

template <typename T>
void SGD<T>::step() {
    for (auto& param : this->parameters_) {
        // Update parameters : param -= learning_rate * param.grad()
        for (uint32_t i = 0; i < param->size(); ++i) {
            param->data()[i] -= learning_rate_ * param->grad().data()[i];
        }
    }
}

template class Optimizer<float>;
template class Optimizer<double>;
template class SGD<float>;
template class SGD<double>;

} // snnf