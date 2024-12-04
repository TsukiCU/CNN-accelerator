#pragma once

#include "utils.h"
#include "tensor.h"

namespace snnf {

template <typename T>
class Loss {
public:
    virtual ~Loss() = default;

    // Compute the loss value
    virtual T forward(const Tensor<T>& input, const Tensor<T>& target) = 0;

    // Compute the gradient w.r.t input
    virtual Tensor<T> backward() = 0;
};

template <typename T>
class MSELoss : public Loss<T> {
public:
    MSELoss() = default;
    ~MSELoss() = default;

    T forward(const Tensor<T>& input, const Tensor<T>& target) override;
    Tensor<T> backward() override;

private:
    Tensor<T> input_;   // Cache input for backward pass
    Tensor<T> target_;  // Cache target for backward pass
};

template <typename T>
class CrossEntropyLoss : public Loss<T> {
public:
    CrossEntropyLoss() = default;
    ~CrossEntropyLoss() = default;

    T forward(const Tensor<T>& input, const Tensor<T>& target) override;
    Tensor<T> backward() override;

private:
    Tensor<T> input_;    // Cache input (logits) for backward pass
    Tensor<T> target_;   // Cache target for backward pass
    Tensor<T> softmax_;  // Cache softmax output for backward pass
};

// Huber Loss Class
template <typename T>
class HuberLoss : public Loss<T> {
public:
    HuberLoss(T delta) : delta_(delta) {}
    ~HuberLoss() = default;

    T forward(const Tensor<T>& input, const Tensor<T>& target) override;
    Tensor<T> backward() override;

private:
    T delta_;          // Threshold parameter for Huber loss
    Tensor<T> input_;   // Cache input for backward pass
    Tensor<T> target_;  // Cache target for backward pass
};

} // snnf
