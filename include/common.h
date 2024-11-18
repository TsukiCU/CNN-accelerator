#ifndef _CUDA_UTIL_H_
#define _CUDA_UTIL_H_

#include <cstddef>
#include <vector>
#include <cstring>
#include <memory>
#include <iostream>
#include <unordered_map>
#include <spdlog/spdlog.h>

namespace cuda
{

enum class DataType {
    DataTypeUnkown = 0,
    DataTypeInt8 = 1,
    DataTypeInt16 = 2,
    DataTypeInt32 = 4,
    DataTypeFloat32 = 8,
    DataTypeFloat64 = 16,
};

enum class DeviceType {
    CPU = 0,
    GPU = 1,
    UNKNOWN = 2,
};

uint32_t get_data_size(DataType type);

} // cuda

#endif // _CUDA_UTIL_H_