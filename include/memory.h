#ifndef _CUDA_MEMORY_H
#define _CUDA_MEMORY_H

#include "common.h"
#include "noncopyable.h"
#include <cuda_runtime.h>

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
    DeviceType get_device_type() const { return device_; }

    void copy_from(const MemoryBuffer& src);
    void copy_to(MemoryBuffer& dst) const;

    static MemoryBuffer create_from_existing(void* data, uint32_t size, DeviceType device) {
        MemoryBuffer buffer(size, device);
        buffer.data_ = data;
        return buffer;
    }

private:
    void* data_;
    uint32_t size_;
    DeviceType device_;

    void allocate(void* data);
    void deallocate();
};

} // cuda

#endif // _CUDA_MEMORY_H