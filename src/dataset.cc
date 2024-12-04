#include "dataset.h"

namespace snnf {

namespace dataset {

template <typename T>
DataLoader<T>::DataLoader(const Dataset<T>& dataset, size_t batch_size, bool shuffle)
    : dataset_(dataset), batch_size_(batch_size), shuffle_(shuffle), current_index_(0) {
    // Initialize indices
    indices_.resize(dataset_.size());
    for (size_t i = 0; i < indices_.size(); ++i) {
        indices_[i] = i;
    }
    if (shuffle_) {
        std::random_shuffle(indices_.begin(), indices_.end());
    }
}

template <typename T>
void DataLoader<T>::reset() {
    current_index_ = 0;
    if (shuffle_) {
        std::random_shuffle(indices_.begin(), indices_.end());
    }
}

template <typename T>
bool DataLoader<T>::has_next() const {
    return current_index_ < indices_.size();
}

template <typename T>
std::pair<Tensor<T>, Tensor<T>> DataLoader<T>::next_batch() {
    size_t end_index = std::min(current_index_ + batch_size_, indices_.size());
    std::vector<T> batch_data;
    std::vector<T> batch_labels;
    std::vector<uint32_t> data_shape;
    std::vector<uint32_t> label_shape;

    // Assuming all data samples have the same shape
    for (size_t i = current_index_; i < end_index; ++i) {
        auto item = dataset_.get_item(indices_[i]);
        auto& data = item.first;
        auto& label = item.second;

        if (data_shape.empty()) {
            data_shape = data.shape();
            data_shape.insert(data_shape.begin(), 0);  // Add batch dimension
        }
        if (label_shape.empty()) {
            label_shape = label.shape();
            label_shape.insert(label_shape.begin(), 0);  // Add batch dimension
        }

        batch_data.insert(batch_data.end(), data.data().begin(), data.data().end());
        batch_labels.insert(batch_labels.end(), label.data().begin(), label.data().end());
    }

    // Update batch size in shape
    data_shape[0] = static_cast<uint32_t>(end_index - current_index_);
    label_shape[0] = static_cast<uint32_t>(end_index - current_index_);

    current_index_ = end_index;

    Tensor<T> batch_data_tensor(data_shape, batch_data);
    Tensor<T> batch_labels_tensor(label_shape, batch_labels);

    return {batch_data_tensor, batch_labels_tensor};
}

template class Dataset<float>;
template class Dataset<double>;
template class DataLoader<float>;
template class DataLoader<double>;

} // dataset

} // snnf