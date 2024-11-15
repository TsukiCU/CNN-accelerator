#ifndef _CUDA_UTIL_H_
#define _CUDA_UTIL_H_

#include <cstddef>
#include <vector>
#include <memory>
#include <iostream>
#include <spdlog/spdlog.h> 

namespace cuda
{

enum class DeviceType {
    CPU,
    GPU,
    UNKNOWN,
};

} // cuda

#endif // _CUDA_UTIL_H_