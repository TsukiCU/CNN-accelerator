#include "iris_dataset.h"

namespace snnf {

namespace dataset {

template <typename T>
void IrisDataset<T>::load_data(const std::string& file_path) {
    std::ifstream file(file_path);
    std::string line;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::vector<T> features;
        std::string value;
        for (int i = 0; i < 4; ++i) {
            std::getline(ss, value, ',');
            features.push_back(static_cast<T>(std::stod(value)));
        }
        std::getline(ss, value, ',');
        int label = std::stoi(value);

        data_.emplace_back(std::vector<uint32_t>{1, 4}, features);
        // One-hot encodes the label -> [0, 0, 1], [0, 1, 0], [1, 0, 0].
        std::vector<T> one_hot_label(3, static_cast<T>(0));
        one_hot_label[label] = static_cast<T>(1);
        labels_.emplace_back(std::vector<uint32_t>{1, 3}, one_hot_label);
    }
}

template <typename T>
size_t IrisDataset<T>::size() const {
    return data_.size();
}

template <typename T>
std::pair<Tensor<T>, Tensor<T>> IrisDataset<T>::get_item(size_t index) const {
    Tensor<T> data = data_[index];
    Tensor<T> label = labels_[index];

    // Yuck!! Ik this stinks but at least it's clear.
    if (data.dim() >= 2) {
        if (data.shape()[0] != 1 || data.dim() > 2) {
            LOG_ERROR(std::runtime_error, "IrisDataset::get_item : Weird things happened (data).");
        }
        data.reshape({data.shape()[1]});
    }

    if (label.dim() >= 2) {
        if (label.shape()[0] != 1 || label.dim() > 2) {
            LOG_ERROR(std::runtime_error, "IrisDataset::get_item : Weird things happened (label).");
        }
        label.reshape({label.shape()[1]});
    }

    return {data, label};
}

template class IrisDataset<float>;
template class IrisDataset<double>;

} // dataset

} // snnf