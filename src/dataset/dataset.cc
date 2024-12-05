#include "dataset.h"

namespace snnf {

namespace dataset {

template <typename T>
DataLoader<T>::DataLoader(const Dataset<T>& dataset, size_t batch_size, bool shuffle, int num_workers)
    : dataset_(dataset), batch_size_(batch_size), shuffle_(shuffle), current_index_(0) {
    // Initialize indices
    indices_.resize(dataset_.size());
    for (size_t i = 0; i < indices_.size(); ++i) {
        indices_[i] = i;
    }
    if (shuffle_) {
        std::random_shuffle(indices_.begin(), indices_.end());
    }

    for (int i = 0; i < num_workers_; ++i) {
        worker_threads_.emplace_back(&DataLoader<T>::worker_thread, this, i);
    }
}

template <typename T>
DataLoader<T>::~DataLoader() {
    {
        std::unique_lock<std::mutex> lock(mutex_);
        stop_ = true;
        cv_.notify_all();
    }
    for (auto& t : worker_threads_) {
        if (t.joinable()) {
            t.join();
        }
    }
}

template <typename T>
void DataLoader<T>::reset() {
    std::unique_lock<std::mutex> lock(mutex_);
    current_index_ = 0;
    if (shuffle_) {
        std::random_shuffle(indices_.begin(), indices_.end());
    }
    // Empty the queue.
    std::queue<std::pair<Tensor<T>, Tensor<T>>> empty_queue;
    std::swap(batch_queue_, empty_queue);
    cv_.notify_all();
}

template <typename T>
bool DataLoader<T>::has_next() {
    std::unique_lock<std::mutex> lock(mutex_);
    return !batch_queue_.empty() || current_index_ < indices_.size();
}

template <typename T>
std::pair<Tensor<T>, Tensor<T>> DataLoader<T>::next_batch() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this] { return !batch_queue_.empty() || stop_; });

    if (!batch_queue_.empty()) {
        auto batch = batch_queue_.front();
        batch_queue_.pop();
        cv_.notify_all(); // Notify the worker thread to keep producing batches.
        return batch;
    } else {
        // There's nothing to be read.
        return {};
    }
}

template <typename T>
void DataLoader<T>::worker_thread(int worker_id) {
    while (true) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] {
            return batch_queue_.size() < max_queue_size_ || stop_;
        });

        if (stop_) {
            break;
        }

        if (current_index_ >= indices_.size()) {
            cv_.notify_all();
            break;
        }

        size_t end_index = std::min(current_index_ + batch_size_, indices_.size());
        /// @note : Temporarily store the current_index_ to prevent data racing.
        size_t start_index = current_index_;
        current_index_ = end_index;

        // Release the lock so that other threads might proceed.
        lock.unlock();

        // Must not hold lock when reading data.
        std::vector<T> batch_data;
        std::vector<T> batch_labels;
        std::vector<uint32_t> data_shape;
        std::vector<uint32_t> label_shape;

        // Batch for this round : [current_index , end_index].
        for (size_t i = start_index; i < end_index; ++i) {
            auto item = dataset_.get_item(indices_[i]);
            auto& data = item.first;
            auto& label = item.second;

            if (data_shape.empty()) {
                data_shape = data.shape();
                data_shape.insert(data_shape.begin(), 0);
            }
            if (label_shape.empty()) {
                label_shape = label.shape();
                label_shape.insert(label_shape.begin(), 0);
            }

            batch_data.insert(batch_data.end(), data.data().begin(), data.data().end());
            batch_labels.insert(batch_labels.end(), label.data().begin(), label.data().end());
        }

        data_shape[0] = static_cast<uint32_t>(end_index - start_index);
        label_shape[0] = static_cast<uint32_t>(end_index - start_index);

        Tensor<T> batch_data_tensor(data_shape, batch_data);
        Tensor<T> batch_labels_tensor(label_shape, batch_labels);

        {
            std::unique_lock<std::mutex> lk(mutex_);
            batch_queue_.emplace(batch_data_tensor, batch_labels_tensor);
            cv_.notify_all(); // Notify the main thread to consume data batches.
        }
    }
}

template class Dataset<float>;
template class Dataset<double>;
template class DataLoader<float>;
template class DataLoader<double>;

} // dataset

} // snnf