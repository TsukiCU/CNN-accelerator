#ifndef _CUDA_MODEL_H
#define _CUDA_MODEL_H

#include "module.h"
#include "layer.h"

namespace cuda {

class Model : public Module {
public:
    Model();
    virtual ~Model();

    virtual Tensor forward(const Tensor& input) override;
    virtual void backward(const Tensor& grad_output) override;

    void add_layer(const std::string& name, std::shared_ptr<Layer> layer);
    std::shared_ptr<Layer> get_layer(const std::string& name);

private:
    std::vector<std::shared_ptr<Layer>> layers_;
};

} // cuda

#endif