#ifndef _CUDA_LOSS_H
#define _CUDA_LOSS_H

#include "module.h"

namespace cuda {

class Loss {
public:
    virtual ~Loss();

    virtual double forward(const Tensor& prediction, const Tensor& target) = 0;
    virtual Tensor backward() = 0;

protected:
    Tensor prediction_;
    Tensor target_;
};


class MSELoss : public Loss {
public:
    virtual ~MSELoss();

    virtual double forward(const Tensor& prediction, const Tensor& target) override;
    virtual Tensor backward() override;
};

};

#endif