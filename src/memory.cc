#include "../include/memory.h"

namespace cuda
{

MemoryBuffer::MemoryBuffer(uint32_t size) : data_(nullptr), size_(size) {
    allocate();
}

MemoryBuffer::MemoryBuffer(void* data, uint32_t size) : data_(nullptr), size_(size) {
    allocate();
}

MemoryBuffer::~MemoryBuffer() {
    deallocate();
}

void MemoryBuffer::allocate() {
    if (size_ == 0) {
        // TODO : Log info.
    }
    data_ = malloc(size_);
    if (!data_)
        // TODO : Log fatal
        throw std::bad_alloc();
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

} // cuda
