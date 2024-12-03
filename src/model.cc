#include "model.h"

namespace snnf {

namespace layer {

template <typename T>
void Model<T>::add_layer(std::shared_ptr<Layer<T>> layer) {
    layers_.push_back(layer);
}

template <typename T>
Tensor<T> Model<T>::forward(const Tensor<T>& input) {
    Tensor<T> x = input;
    for (auto& layer : layers_) {
        x = layer->forward(x);
    }
    output_ = x;  // Cache output for potential use
    return output_;
}

template <typename T>
void Model<T>::backward(const Tensor<T>& loss_grad) {
    Tensor<T> grad = loss_grad;
    // Iterate layers in reverse order
    for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) {
        grad = (*it)->backward(grad);
    }
}

template <typename T>
std::vector<Tensor<T>*> Model<T>::get_parameters() {
    std::vector<Tensor<T>*> params;
    for (auto& layer : layers_) {
        auto layer_params = layer->get_parameters();
        params.insert(params.end(), layer_params.begin(), layer_params.end());
    }
    return params;
}

template <typename T>
void Model<T>::zero_grad() {
    for (auto& layer : layers_) {
        layer->zero_grad();
    }
}

template class Model<float>;
template class Model<double>;

} // layer

} // snnf