#ifndef _CUDA_DATA_H
#define _CUDA_DATA_H

#include "module.h"
#include "layer.h"

namespace cuda {

class Dataset {
public:
    virtual ~Dataset();
    virtual std::pair<Tensor, Tensor> get_item(uint32_t index) = 0;
    virtual uint32_t size() const = 0;
};

class DataLoader {
public:
    DataLoader(Dataset& dataset, uint32_t batch_size);
    virtual ~DataLoader();

    void reset();
    bool has_next() const;
    std::pair<Tensor, Tensor> next();

private:
    Dataset& dataset_;
    uint32_t batch_size_;
    uint32_t current_index_;
};

}

#endif