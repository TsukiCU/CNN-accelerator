#pragma once

#include "log.h"
#include "utils.h"
#include "tensor.h"

namespace snnf {

enum class InitMethod { Uniform, Gaussian, Kaiming, Xavier };

namespace layer {

/**
 * @brief : base class for all other layers.
 */
template <typename T>
class Layer {
public:
    Layer() = default;
    virtual ~Layer() = default;

    virtual Tensor<T> forward(const Tensor<T>& input) = 0;

    virtual Tensor<T> backward(const Tensor<T>& grad_output) = 0;

    virtual std::vector<Tensor<T>*> get_parameters() { return {}; }

    virtual void zero_grad() {}
};


/**
 * @brief : Linear layer class for fully connected layer.
 * @todo : Include more types of activation layer.
 */
template <typename T>
class LinearLayer : public Layer<T> {
public:
    LinearLayer(uint32_t in_features, uint32_t out_features, InitMethod init=InitMethod::Kaiming);

    Tensor<T> forward(const Tensor<T>& input) override;
    Tensor<T> backward(const Tensor<T>& grad_output) override;

    std::vector<Tensor<T>*> get_parameters() override;
    void zero_grad() override;

private:
    uint32_t in_features_;
    uint32_t out_features_;
    Tensor<T> weights_;
    Tensor<T> bias_;
    Tensor<T> input_;

    // The direction and size of the weights and biases 
    // that need to be adjusted in the current training batch.
    Tensor<T> grad_weights_;
    Tensor<T> grad_bias_;
};


/**
 * @brief : Abstract activation layer base class.
 */
template <typename T>
class ActivationLayer : public Layer<T> {
public:
    ActivationLayer() = default;
    virtual ~ActivationLayer() = default;

    // Must instantiate a specific activation layer.
    virtual Tensor<T> forward(const Tensor<T>& input) override = 0;
    virtual Tensor<T> backward(const Tensor<T>& grad_output) override = 0;
};


/**
 * @brief : ReLU layer class.
 */
template <typename T>
class ReLULayer : public ActivationLayer<T> {
public:
    ReLULayer() = default;
    ~ReLULayer() = default;

    Tensor<T> forward(const Tensor<T>& input) override;
    Tensor<T> backward(const Tensor<T>& grad_output) override;

private:
    Tensor<T> input_;  // Cache input for backward pass
};


/**
 * @brief : Sigmoid layer class.
 */
template <typename T>
class SigmoidLayer : public ActivationLayer<T> {
public:
    SigmoidLayer() = default;
    ~SigmoidLayer() = default;

    Tensor<T> forward(const Tensor<T>& input) override;
    Tensor<T> backward(const Tensor<T>& grad_output) override;

private:
    Tensor<T> output_;  // Cache for backward pass
};

/**
 * @brief : Softmax layer for classification tasks.
*/
template <typename T>
class SoftmaxLayer : public Layer<T> {
public:
    SoftmaxLayer() = default;
    ~SoftmaxLayer() = default;

    Tensor<T> forward(const Tensor<T>& input) override;
    Tensor<T> backward(const Tensor<T>& grad_output) override;

private:
    Tensor<T> output_; // Cache for backward pass
};

} // layer

} // snnf