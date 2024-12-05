#include "iris_dataset.h"

namespace snnf {

namespace dataset {

template <typename T>
IrisDataset<T>::IrisDataset() : in_features(4), out_features(3) {}

template <typename T>
void IrisDataset<T>::load_data(const std::string& file_path) {
    std::ifstream file(file_path);
    std::string line;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::vector<T> features;
        std::string value;
        for (uint32_t i = 0; i < in_features; ++i) {
            std::getline(ss, value, ',');
            features.push_back(static_cast<T>(std::stod(value)));
        }
        std::getline(ss, value, ',');
        int label = std::stoi(value);

        // One-hot encodes the label -> [0, 0, 1], [0, 1, 0], [1, 0, 0].
        data_.emplace_back(std::vector<uint32_t>{in_features}, features);
        std::vector<T> one_hot_label(out_features, static_cast<T>(0));
        one_hot_label[label] = static_cast<T>(1);
        labels_.emplace_back(std::vector<uint32_t>{out_features}, one_hot_label);
    }
}

template <typename T>
size_t IrisDataset<T>::size() const {
    return data_.size();
}

template <typename T>
std::pair<Tensor<T>, Tensor<T>> IrisDataset<T>::get_item(size_t index) const {
    return { data_[index], labels_[index] };
}

template class IrisDataset<float>;
template class IrisDataset<double>;

} // dataset

} // snnf