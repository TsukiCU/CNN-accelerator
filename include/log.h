#pragma once

#include <iostream>

namespace cuda {

#define RED   "\e[1;31m"    // Log Error
#define YELLOW "\e[1;33m"   // Log Warning
#define WHITE "\e[1;37m"    // Log Info
#define PURPLE "\e[0;35m"   // Unknown type
#define DEFA  "\e[0m"       // Ending place

#ifndef RECORD
#define RECORD 0
#endif

#define LOG_ERROR(type, msg) do { \
    if (RECORD) { \
        std::cerr << RED << "ERROR : " << msg << DEFA << std::endl; \
    } \
    if constexpr (std::is_same<type, std::runtime_error>::value || \
                  std::is_same<type, std::invalid_argument>::value || \
                  std::is_same<type, std::out_of_range>::value) { \
        throw type(msg); \
    } else { \
        std::cerr << PURPLE << "Invalid exception type passed to LOG_ERROR." << DEFA << std::endl; \
        throw std::runtime_error("Invalid exception type in LOG_ERROR"); \
    } \
} while (0)

#define LOG_WARN(msg) do { \
    if (RECORD) \
        std::cout << YELLOW << "WARNING : " << msg << DEFA << std::endl; \
} while (0)

#define LOG_INFO(msg) do { \
    if (RECORD) \
        std::cout << WHITE << "INFO : " << msg << DEFA << std::endl; \
} while (0)

} // cuda