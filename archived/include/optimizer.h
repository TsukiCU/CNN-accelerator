#ifndef _CUDA_LOSS_H
#define _CUDA_LOSS_H

#include "module.h"

namespace cuda {

class Optimizer {
public:
    Optimizer(std::vector<Tensor*> parameters, double learning_rate);
    virtual ~Optimizer();

    virtual void step() = 0;
    void zero_grad();

protected:
    std::vector<Tensor*> parameters_;
    double learning_rate_;
};


class SGDOptimizer : public Optimizer {
public:
    SGDOptimizer(std::vector<Tensor*> parameters, double learning_rate);
    virtual ~SGDOptimizer();

    virtual void step() override;
};

} // cuda

#endif