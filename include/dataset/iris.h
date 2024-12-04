#pragma once

#include "../dataset.h"

namespace snnf {

namespace dataset {

template <typename T>
class IrisDataset : public Dataset<T> {
public:
    IrisDataset() = default;
    ~IrisDataset() = default;

    void load_data(const std::string& file_path) override;
    size_t size() const override;
    std::pair<Tensor<T>, Tensor<T>> get_item(size_t index) const override;

private:
    std::vector<Tensor<T>> data_;
    std::vector<Tensor<T>> labels_;
};

} // dataset

} // snnf