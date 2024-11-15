#ifndef _CUDA_MEMORY_H
#define _CUDA_MEMORY_H

#include "util.h"
#include "noncopyable.h"
#include <cuda_runtime.h>

namespace cuda
{

class MemoryBuffer : public Noncopyable {
/*
 * @brief : Create a memory block regardless of its location.
 * @todo : Concurrency (urgent), memory alignment and log.
 */
public:
    MemoryBuffer(uint32_t size, DeviceType device);
    ~MemoryBuffer();

    void* data() const { return ptr_; }
    uint32_t get_size() const { return size_; }
    DeviceType get_device_type() const { return device_type_; }

    void copy_from(const MemoryBuffer& src);
    void copy_to(MemoryBuffer& dst) const;

private:
    void* ptr_;
    uint32_t size_;
    DeviceType device_type_;

    void allocate();
    void deallocate();
};

} // cuda

#endif // _CUDA_MEMORY_H