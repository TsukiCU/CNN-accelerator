#include "../include/memory.h"

namespace cuda
{

MemoryBuffer::MemoryBuffer(uint32_t size, DeviceType device) :
    data_(nullptr), size_(size), device_(device)
{
    allocate(nullptr);
}

MemoryBuffer::MemoryBuffer(void* data, uint32_t size, DeviceType device) :
    data_(nullptr), size_(size), device_(device)
{
    allocate(data);
}

MemoryBuffer::~MemoryBuffer() {
    deallocate();
}

void MemoryBuffer::allocate(void* data) {
    if (size_ == 0) {
        // TODO : Log info.
    }
    if (device_ != DeviceType::CPU) {
        // Log error.
        throw std::invalid_argument("Not implemented yet.");
    }
    data_ = malloc(size_);
    if (!data_) {
        // TODO : Log fatal
        throw std::bad_alloc();
    }
    if (data)
        memcpy(data_, data, size_);
}

void MemoryBuffer::deallocate() {
    free(data_);
    data_ = nullptr;
}

void MemoryBuffer::copy_from(const MemoryBuffer& src) {
    if (!data_ || !src.data_) {
        // Log error.
        throw std::runtime_error("Null pointers.");
    }
    std::memcpy(data_, src.data_, size_);
}

void MemoryBuffer::copy_to(MemoryBuffer& dst) const {
    dst.copy_from(*this);
}

void MemoryBuffer::resize(uint32_t new_size) {
    if (!data_) {
        // Log error. Shouldn't get here anyway.
        throw std::runtime_error("Can't resize when buffer isn't allocated.");
    }
    deallocate();
    size_ = new_size;
    allocate(nullptr);
}

} // cuda
