#pragma once

#include "dataset.h"

namespace snnf {

namespace dataset {

/**
 * @brief : Mnist dataset.
*/
template <typename T>
class MNISTDataset : public Dataset<T> {
public:
    MNISTDataset();
    ~MNISTDataset() = default;

    void load_data(const std::string& train_file, const std::string& label_file) override;
    size_t size() const override;
    std::pair<Tensor<T>, Tensor<T>> get_item(size_t index) const override;

private:
    std::vector<Tensor<T>> images_;
    std::vector<Tensor<T>> labels_;

    void parse_images(const std::string& file_path);
    void parse_labels(const std::string& file_path);

    uint32_t in_features;
    uint32_t out_features;
};

} // dataset

} // snnf