#include "../include/memory.h"

namespace cuda
{

MemoryBuffer::MemoryBuffer(uint32_t size, DeviceType device) :
data_(nullptr), size_(size), device_(device) {
    allocate();
}

MemoryBuffer::MemoryBuffer(void* data, uint32_t size, DeviceType device) :
data_(nullptr), size_(size), device_(device)
{
    allocate();
}

MemoryBuffer::~MemoryBuffer() {
    deallocate();
}

void MemoryBuffer::allocate() {
    if (size_ == 0) {
        // TODO : Log info.
    }
    if (device_ != DeviceType::CPU && device_ != DeviceType::GPU) {
        // TODO: Log error.
        throw std::invalid_argument("Unknown device in allocate");
    }

    if (device_ == DeviceType::CPU) {
        data_ = malloc(size_);
        if (!data_)
            // TODO: Log fatal.
            throw std::bad_alloc();
    }
    else if (device_ == DeviceType::GPU) {
        cudaError_t err = cudaMalloc(&data_, size_);
        if (err != cudaSuccess) {
            // TODO: Log fatal.
            throw std::runtime_error("Failed to allocate GPU memory");
        }
    }
}

void MemoryBuffer::deallocate() {
    if (device_ == DeviceType::CPU)
        free(data_);
    else if (device_ == DeviceType::GPU) {
        cudaError_t err = cudaFree(data_);
        if (err != cudaSuccess)
            // TODO: Log error
            throw std::runtime_error("Failed to free GPU memory.");
    }
    else
        // TODO: Log error.
        throw std::invalid_argument("Unknown type in deallocate");
    data_ = nullptr;
}

void MemoryBuffer::copy_from(const MemoryBuffer& src) {
    if (!data_ || !src.data_) {
        // Log error.
        throw std::runtime_error("Null pointers.");
    }

    cudaMemcpyKind kind;
    if (device_ == DeviceType::CPU && src.device_ == DeviceType::GPU) {
        // TODO: Log debug
        kind = cudaMemcpyDeviceToHost;
    }
    else if (device_ == DeviceType::GPU && src.device_ == DeviceType::CPU) {
        // TODO: Log debug
        kind = cudaMemcpyHostToDevice;
    }
    else if (device_ == DeviceType::GPU && src.device_ == DeviceType::GPU) {
        // TODO: Log debug
        kind = cudaMemcpyDeviceToDevice;
    }
    else if (device_ == DeviceType::CPU && src.device_ == DeviceType::CPU) {
        // TODO: Log info, copying between hosts, using memcpy.
        std::memcpy(data_, src.data_, size_);
        return;
    }
    else {
        // TODO: Log error.
        throw std::invalid_argument("Invalid device types for copy");
    }

    cudaError_t err = cudaMemcpy(data_, src.data_, size_, kind);
    if (err != cudaSuccess) {
        // TODO: Log error.
        throw std::runtime_error("Failed to copy memory");
    }
}

void MemoryBuffer::copy_to(MemoryBuffer& dst) const {
    dst.copy_from(*this);
}

} // cuda
