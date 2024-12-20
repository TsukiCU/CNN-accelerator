#pragma once

#include "dataset.h"

namespace snnf {

namespace dataset {

/**
 * @brief : Iris dataset. Four features and one label. 
*/
template <typename T>
class IrisDataset : public Dataset<T> {
public:
    IrisDataset();
    ~IrisDataset() = default;

    void load_data(const std::string& data_path) override;
    size_t size() const override;
    std::pair<Tensor<T>, Tensor<T>> get_item(size_t index) const override;

private:
    std::vector<Tensor<T>> data_;
    std::vector<Tensor<T>> labels_;

    uint32_t in_features;
    uint32_t out_features;
};

} // dataset

} // snnf