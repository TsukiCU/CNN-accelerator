#include "../include/common.h"
#include "../include/log.h"

namespace cuda
{

uint32_t get_data_size(DataType type) {
    uint32_t ret;

    switch (type) {
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
        case DataType::DataTypeUnknown:
            LOG_ERROR("get_data_size() : Unknown data type.");
        default:
            LOG_ERROR("get_data_size() : Invalid data type.");
    }

    return ret;
}

} // cuda