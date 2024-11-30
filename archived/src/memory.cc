#include "../include/log.h"
#include "../include/memory.h"

namespace cuda
{

void* MemoryBuffer::allocate_memory() {
    void* buffer = malloc(size_);
    if (!buffer) {
        LOG_FATAL("Memory::allocate_memory() : failed to allocate memory.");
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
        LOG_WARN("Might not supposed to reach here.");
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
            LOG_WARN("Might not supposed to reach here.");
        }
    }
    return *this;
}

void MemoryBuffer::allocate(void* data) {
    if (size_ == 0) {
       LOG_INFO("Allocating a piece of memory with size of 0.");
    }
    if (device_ != DeviceType::CPU) {
        LOG_ERROR("MemoryBuffer.allocate() : Not implemented yet.");
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
        LOG_ERROR("Null pointers.");
    }
    if (src.size() > size_) {
        LOG_ERROR("Source buffer is larger than destination buffer");
    }
    std::memcpy(data_.get(), src.data(), size_);
}

void MemoryBuffer::copy_to(MemoryBuffer& dst) const {
    dst.copy_from(*this);
}

void MemoryBuffer::resize(uint32_t new_size) {
    if (!data_) {
        LOG_ERROR("Can't resize when buffer isn't allocated.");
    }
    deallocate();
    size_ = new_size;
    allocate(nullptr);
}

} // cuda
