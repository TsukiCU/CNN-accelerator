#pragma once

#include "tensor.h"

namespace snnf {

namespace dataset {

template <typename T>
class Dataset {
public:
    Dataset() = default;
    ~Dataset() = default;

    ///@brief : Get the size of the dataset
    virtual size_t size() const = 0;

    ///@brief : Get data and labels
    virtual std::pair<Tensor<T>, Tensor<T>> get_item(size_t index) const = 0;

    /**
     * @brief : Load data from file or other sources.
     * @param : <training data> <label data> (optional).
     * @note  : For now, each item in every dataset must be one-dimensional.
    */
    virtual void load_data(const std::string& data_path) {}
    virtual void load_data(const std::string& train_file, const std::string& label_file) {}
};

/**
 * @brief : input Tensor : [batch_size, feature].
*/
template <typename T>
class DataLoader {
public:
    DataLoader(const Dataset<T>& dataset, size_t batch_size, bool shuffle = true);
    ~DataLoader() = default;

    void reset();
    bool has_next() const;
    std::pair<Tensor<T>, Tensor<T>> next_batch(); // get next batch

private:
    const Dataset<T>& dataset_;
    size_t batch_size_;
    bool shuffle_;
    std::vector<size_t> indices_;
    size_t current_index_;
};

}

} // snnf