#include "../include/common.h"

namespace cuda
{

uint32_t get_data_size(DataType type) {
    uint32_t ret;

    switch (type) {
        case DataType::DataTypeUnkown:
            // Log error.
            ret = 0;
            break;
        case DataType::DataTypeInt8:
            ret = 1;
            break;
        case DataType::DataTypeInt16:
            ret = 2;
            break;
        case DataType::DataTypeInt32:
            ret = 4;
            break;
        case DataType::DataTypeFloat32:
            ret = 4;
            break;
        case DataType::DataTypeFloat64:
            ret = 8;
            break;
        default:
            // Log fatal.
            throw std::invalid_argument("Invalid data type! ");
    }

    return ret;
}

} // cuda