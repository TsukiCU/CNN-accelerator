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
    MemoryBuffer(uint32_t size, DeviceType device);
    MemoryBuffer(void* data, uint32_t size, DeviceType device);
    ~MemoryBuffer();

    void* data() const { return data_; }
    uint32_t get_size() const { return size_; }

    void resize(uint32_t new_size);
    void copy_to(MemoryBuffer& dst) const;
    void copy_from(const MemoryBuffer& src);

    static MemoryBuffer create_from_existing(void* data, uint32_t size, DeviceType device) {
        MemoryBuffer buffer(size, device);
        buffer.data_ = data;
        return buffer;
    }

private:
    void* data_;
    uint32_t size_;
    DeviceType device_;

    void allocate();
    void deallocate();
};

} // cuda

#endif // _CUDA_MEMORY_H