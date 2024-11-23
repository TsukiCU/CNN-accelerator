#include "../include/memory.h"

namespace cuda
{

void* MemoryBuffer::allocate_memory() {
    void* buffer = malloc(size_);
    if (!buffer) {
        // Log error.
        throw std::bad_alloc();
    }
    return buffer;
}

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

MemoryBuffer::MemoryBuffer(const MemoryBuffer& other) :
    size_(other.size()), device_(other.device())
{
    if (other.data()) {
        allocate(nullptr);
        std::memcpy(data_.get(), other.data(), size_);
        // allocate(other.data());
    }
    else {
        // Log warn.
        std::cerr << "Might not supposed to reach here." << std::endl;
    }
}

MemoryBuffer& MemoryBuffer::operator=(const MemoryBuffer& other) {
    if (this != &other) {
        data_.reset();
        device_ = other.device();
        if (other.data()) {
            allocate(nullptr);
            std::memcpy(data_.get(), other.data(), size_);
        }
        else {
            // Log warn.
            std::cerr << "Might not supposed to reach here." << std::endl;
        }
    }
    return *this;
}

void MemoryBuffer::allocate(void* data) {
    if (size_ == 0) {
        // TODO : Log info.
    }
    if (device_ != DeviceType::CPU) {
        // Log error.
        throw std::invalid_argument("Not implemented yet.");
    }
    if (data) {
        data_ = std::unique_ptr<void, Deleter>(data, Deleter{});
    } else {
        data_ = std::unique_ptr<void, Deleter>(allocate_memory(), Deleter{});
    }
}

void MemoryBuffer::deallocate() {
    data_.reset();
}

void MemoryBuffer::copy_from(const MemoryBuffer& src) {
    if (!data_ || !src.data_) {
        // Log error.
        throw std::runtime_error("Null pointers.");
    }
    if (src.size() > size_) {
        // Log error.
        throw std::runtime_error("Source buffer is larger than destination buffer");
    }
    std::memcpy(data_.get(), src.data(), size_);
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
