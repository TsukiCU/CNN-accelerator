
# SNNF

## Introduction

SNNF, which is short for Simple Neural Network Framework, is a simplified neural network training framework written in C++, which is akin a subset of PyTorch. It provides core functionalities such as defining neural network layers (linear and activation layers), training loops, loss computation, parameter optimization, and data loading with multi-threading support.

The primary goal of this framework is to offer a modular, extensible codebase that can be easily understood, tested, and integrated with other components.

## High-Level Architecture

Key Concepts & Components:

####   Tensor
The Tensor class is a fundamental building block representing multi-dimensional arrays of data.

1. Store data (e.g., model parameters, inputs, outputs).
2. Provide shape, stride information, and basic operations (e.g., matmul, add, sum).
3. Implemented as a template class (Tensor<T>) to support different data types (float, double, etc).
4. Extensible for potential CPU/GPU backends in the future.

####	Layer & Activation Classes
The framework models neural network layers (e.g., LinearLayer, ReLULayer, SigmoidLayer, SoftmaxLayer) as classes derived from a base Layer or ActivationLayer.

1.  Define forward and backward methods to compute outputs and gradients.
2.  Store parameters (weights, biases) when applicable (e.g., LinearLayer).
3.  Each layer provides forward(const Tensor<T>&) and backward(const Tensor<T>&) methods.
4.  Parameters are stored as Tensor<T> objects for easy retrieval and updating.

####	Model Class
The Model class manages a collection of layers, providing a top-level interface for forward and backward propagation.

1.  Compose layers into a sequence to form a model architecture.
2.  Invoke each layer’s forward and backward methods during training.
3.  Provide get_parameters() to access all parameters for optimization.
4.  Offer save_parameters() and load_parameters() for serialization and deserialization.
5.  Hold  ```std::vector<std::shared_ptr<Layer<T>>>``` to manage a stack of layers.

####	Loss Functions
The framework supports different loss functions (e.g., MSELoss, CrossEntropyLoss).

1.  Compute scalar loss given output and target.
2.  Compute gradients w.r.t. output for backward propagation.
3.  Each loss class provides forward to return the loss value and backward to return gradient wrt the model output.
4.  Ensures numerical stability (particularly in CrossEntropyLoss with Softmax).

####	Optimizers
The Optimizer base class and its implementations (e.g., SGD) update model parameters based on computed gradients.

1.  Extract parameters from Model (via get_parameters()).
2.  Update parameters according to optimization logic (e.g., param = param - lr * grad).
3.  SGD is a simple optimizer that performs vanilla stochastic gradient descent.
4.  Can be extended to implement momentum, Adam, RMSProp, etc.

####	Dataset & DataLoader
The Dataset interface and DataLoader classes handle data provisioning.

1.  Dataset: Provides raw access to samples and labels (e.g., MNISTDataset).
2.  DataLoader: Fetches mini-batches from the Dataset, handles shuffling, and, optionally, multi-threaded pre-fetching of batches.
3.  DataLoader supports a configurable number of worker threads (num_workers) to load data in parallel.
4.  A thread-safe queue (batch_queue_) buffers batches from producer threads to the main training loop.

## Data Flow

1.	**Forward Pass**:
Input data flows from the DataLoader to Model::forward(), passing through each Layer sequentially. Activation layers transform data non-linearly. At the output layer, predictions are generated.

2.	**Loss Computation**:
The computed predictions and the target labels from the Dataset are passed to the Loss function to produce a scalar loss. The loss is a measure of how far predictions deviate from the target.

3.	**Backward Pass**:
Loss::backward() gives gradients w.r.t. model output. Model::backward() propagates these gradients through each layer in reverse order. Each layer’s backward() updates grad() fields in their parameters.

4.	**Parameter Update**:
Optimizer::step() retrieves gradients from all parameters and applies update rules (e.g., SGD). Updated parameters reflect a step toward minimizing the loss.

5.	**Multi-threaded Data Loading**:
**Not fully tested.** So it's currently now merged into the main branch.
Multiple worker threads continuously fetch and prepare the next batches. The main training loop consumes these batches from batch_queue_. When batch_queue_ is empty but data is still available, the main thread waits. If all data is exhausted, training proceeds to the next epoch or terminates.

## Concurrency Considerations

•	**DataLoader and num_workers:**
The DataLoader spawns multiple worker threads. Each worker fetches samples and packs them into batches. A thread-safe queue and condition variables ensure safe producer-consumer interaction.

•	**Parameters and Thread Safety:**
Parameter updates typically happen in a single-threaded context (the main training loop). If parallel updates are desired, additional synchronization or careful partitioning of parameters is needed.

## Save & Load Parameters

####  File Format
The model parameters are stored in a binary file. This file contains:
•	The number of parameters.
•	For each parameter: dimension count, dimension sizes, and raw data array.

####  Workflow
•	Saving Parameters:
Model::save_parameters(filename) opens a binary ofstream, writes parameter metadata and data arrays.
•	Loading Parameters:
Model::load_parameters(filename) opens a binary ifstream, reads parameter metadata, and data arrays. It checks shape consistency with the current model. If mismatched, it reports log error.

## Extensibility

•	Additional Layers:
New layers can be added by deriving from Layer<T>, implementing forward() and backward().
	•	New Loss Functions:
Implement new loss functions by providing corresponding forward() and backward() methods.
	•	Custom Optimizers:
Extend Optimizer class to implement advanced optimization techniques.
	•	GPU & Other Devices:
The design currently targets CPU execution. Extending Tensor operations to GPUs involves device-specific memory management and kernels.
	•	Data Augmentation & Preprocessing:
Additional transformations can be integrated into Dataset or DataLoader to preprocess samples (e.g., normalization, augmentation).

## Testing and Debugging

•	Unit Tests:
Each component (Tensor, Layers, Loss, Optimizer, DataLoader) can be unit-tested independently.
•	Logging & Assertions:
Use logging macros (e.g., LOG_ERROR) and assertions to detect shape mismatches, file I/O failures, and other runtime issues early.
•	Profiling:
Profile the code to identify bottlenecks in data loading or arithmetic operations. Adjust num_workers or enable OpenMP optimizations accordingly.

## Summary