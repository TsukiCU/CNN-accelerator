#ifndef _CUDA_LAYER_H
#define _CUDA_LAYER_H

#include "module.h"

namespace cuda {

class Layer : public Module {
public:
    Layer();
    virtual ~Layer();

    virtual Tensor forward(const Tensor& input) override = 0;
    virtual void backward(const Tensor& grad_output) override = 0;
};


// Fully connected layer.
class LinearLayer : public Layer {
public:
    LinearLayer(uint32_t in_features, uint32_t out_features);
    virtual ~LinearLayer();

    virtual Tensor forward(const Tensor& input) override;
    virtual void backward(const Tensor& grad_output) override;

private:
    Tensor weight_;
    Tensor bias_;
    Tensor input_;  // for backward propagation.
};


// Activation layer.
class ActivationLayer : public Layer {
public:
    virtual ~ActivationLayer();
};

class ReLULayer : public ActivationLayer {
public:
    ReLULayer();
    virtual ~ReLULayer();

    virtual Tensor forward(const Tensor& input) override;
    virtual void backward(const Tensor& grad_output) override;

private:
    Tensor mask_;
};

} // cuda

#endif