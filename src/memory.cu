#include "../include/memory.h"

namespace cuda
{

MemoryBuffer::MemoryBuffer(uint32_t size, DeviceType device) :
ptr_(nullptr), size_(size), device_type_(device) {
    allocate();
}

MemoryBuffer::~MemoryBuffer() {
    deallocate();
}

void MemoryBuffer::allocate() {
    if (size_ == 0) {
        // TODO : Log info.
    }
    if (device_type_ == DeviceType::CPU) {
        ptr_ = malloc(size_);
        if (!ptr_)
            // TODO: Log fatal.
            throw std::bad_alloc();
    }
    else if (device_type_ == DeviceType::GPU) {
        cudaError_t err = cudaMalloc(&ptr_, size_);
        if (err != cudaSuccess) {
            // TODO: Log fatal.
            throw std::runtime_error("Failed to allocate GPU memory");
        }
    }
    else {
        // TODO: Log error.
        throw std::invalid_argument("Unknown device in allocate");
    }
}

void MemoryBuffer::deallocate() {
    if (device_type_ == DeviceType::CPU)
        free(ptr_);
    else if (device_type_ == DeviceType::GPU) {
        cudaError_t err = cudaFree(ptr_);
        if (err != cudaSuccess)
            // TODO: Log error
            throw std::runtime_error("Failed to free GPU memory.");
    }
    else
        // TODO: Log error.
        throw std::runtime_error("Unknown type in deallocate");
    ptr_ = nullptr;
}

void MemoryBuffer::copy_from(const MemoryBuffer& src) {
    if (!ptr_ || !src.ptr_) {
        // Log error.
        throw std::runtime_error("Null pointers.");
    }

    cudaMemcpyKind kind;
    if (device_type_ == DeviceType::CPU && src.device_type_ == DeviceType::GPU) {
        // TODO: Log debug
        kind = cudaMemcpyDeviceToHost;
    }
    else if (device_type_ == DeviceType::GPU && src.device_type_ == DeviceType::CPU) {
        // TODO: Log debug
        kind = cudaMemcpyHostToDevice;
    }
    else if (device_type_ == DeviceType::GPU && src.device_type_ == DeviceType::GPU) {
        // TODO: Log debug
        kind = cudaMemcpyDeviceToDevice;
    }
    else if (device_type_ == DeviceType::CPU && src.device_type_ == DeviceType::CPU) {
        // TODO: Log info, copying between hosts, using memcpy.
        std::memcpy(ptr_, src.ptr_, size_);
        return;
    }
    else {
        // TODO: Log error.
        throw std::invalid_argument("Invalid device types for copy");
    }

    cudaError_t err = cudaMemcpy(ptr_, src.ptr_, size_, kind);
    if (err != cudaSuccess) {
        // TODO: Log error.
        throw std::runtime_error("Failed to copy memory");
    }
}

void MemoryBuffer::copy_to(MemoryBuffer& dst) const {
    dst.copy_from(*this);
}

} // cuda
