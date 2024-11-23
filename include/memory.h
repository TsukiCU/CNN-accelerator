#ifndef _CUDA_MEMORY_H
#define _CUDA_MEMORY_H

#include "common.h"
#include "noncopyable.h"

namespace cuda
{

/*
 * @brief : Create a memory block regardless of its location.
 * @todo  : Concurrency (urgent), memory alignment and log.
 */
class MemoryBuffer {
public:
    // Rule of five.
    MemoryBuffer(uint32_t size, DeviceType device);
    MemoryBuffer(void* data, uint32_t size, DeviceType device);
    MemoryBuffer(const MemoryBuffer& other);
    MemoryBuffer& operator=(const MemoryBuffer& other);
    // MemoryBuffer(MemoryBuffer&& other) noexcept;
    // MemoryBuffer& operator=(MemoryBuffer&& other) noexcept;
    ~MemoryBuffer();

    void* data() const { return data_.get(); }
    uint32_t size() const { return size_; }
    DeviceType device() const { return device_; }

    void resize(uint32_t new_size);
    void copy_to(MemoryBuffer& dst) const;
    void copy_from(const MemoryBuffer& src);

    static MemoryBuffer create_from_existing(void* data, uint32_t size, DeviceType device) {
        // MemoryBuffer buffer(data, size, device);
        MemoryBuffer buffer(size, device);
        buffer.data_ = std::unique_ptr<void, Deleter>(nullptr, Deleter{});
        return buffer;
    }

private:
    struct Deleter {
        void operator()(void* ptr) {
            if (ptr) {
                std::free(ptr); // same as free
            }
        }
    };

    std::unique_ptr<void, Deleter> data_;
    uint32_t size_;
    DeviceType device_;

    // Helper functions.
    void allocate(void* data);
    void deallocate();
    void* allocate_memory();
};

} // cuda

#endif // _CUDA_MEMORY_H