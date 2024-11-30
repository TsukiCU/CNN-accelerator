#ifndef _CUDA_MODULE_H
#define _CUDA_MODULE_H

#include "tensor.h"
#include "common.h"

namespace cuda {

// Base class for all other components.
class Module {
public:
    Module();
    virtual ~Module();

    virtual Tensor forward(const Tensor& input) = 0;
    virtual void backward(const Tensor& grad_output) = 0;

    void add_parameter(const std::string& name, Tensor& param);
    Tensor& get_parameter(const std::string& name);

    void add_module(const std::string& name, std::shared_ptr<Module> module);
    std::shared_ptr<Module> get_module(const std::string& name);

    std::vector<Tensor*> parameters();
    void zero_grad();

protected:
    std::unordered_map<std::string, Tensor> parameters_;
    std::unordered_map<std::string, std::shared_ptr<Module>> modules_;
};

} // cuda

#endif // _CUDA_MODULE_H