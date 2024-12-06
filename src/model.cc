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

template <typename T>
void Model<T>::save_parameters(const std::string& filename) {
    auto params = get_parameters();
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs.is_open()) {
        LOG_ERROR(std::runtime_error, "Model::save_parameters: Failed to open file " + filename);
    }

    uint32_t num_params = static_cast<uint32_t>(params.size());
    ofs.write(reinterpret_cast<const char*>(&num_params), sizeof(num_params));

    for (auto* param : params) {
        auto shape = param->shape();
        uint32_t dim = static_cast<uint32_t>(shape.size());
        ofs.write(reinterpret_cast<const char*>(&dim), sizeof(dim));

        // Size of each dimension.
        for (auto s : shape) {
            ofs.write(reinterpret_cast<const char*>(&s), sizeof(s));
        }

        uint32_t data_size = param->size();
        ofs.write(reinterpret_cast<const char*>(param->data().data()), data_size * sizeof(T));
    }

    ofs.close();
}

template <typename T>
void Model<T>::load_parameters(const std::string& filename) {
    auto params = get_parameters();
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs.is_open()) {
        throw std::runtime_error("Model::load_parameters: Failed to open file " + filename);
    }

    uint32_t num_params = 0;
    ifs.read(reinterpret_cast<char*>(&num_params), sizeof(num_params));
    if (num_params != params.size()) {
        LOG_ERROR(std::runtime_error, "Model::load_parameters: number of parameters mismatch.");
    }

    for (auto* param : params) {
        uint32_t dim = 0;
        ifs.read(reinterpret_cast<char*>(&dim), sizeof(dim));

        std::vector<uint32_t> loaded_shape(dim);
        for (uint32_t i = 0; i < dim; ++i) {
            ifs.read(reinterpret_cast<char*>(&loaded_shape[i]), sizeof(loaded_shape[i]));
        }

        auto current_shape = param->shape();
        if (current_shape.size() != dim) {
            LOG_ERROR(std::runtime_error, "Model::load_parameters: dimension mismatch.");
        }
        for (uint32_t i = 0; i < dim; ++i) {
            if (current_shape[i] != loaded_shape[i]) {
                LOG_ERROR(std::runtime_error, "Model::load_parameters: shape mismatch.");
            }
        }

        uint32_t data_size = param->size();
        ifs.read(reinterpret_cast<char*>(param->data().data()), data_size * sizeof(T));
    }

    ifs.close();
}

template class Model<float>;
template class Model<double>;

} // layer

} // snnf