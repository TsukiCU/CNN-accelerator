#pragma once

#include <iostream>

namespace snnf {

#define RED   "\e[1;31m"    // Log Error
#define YELLOW "\e[1;33m"   // Log Warning
#define WHITE "\e[1;37m"    // Log Info
#define PURPLE "\e[0;35m"   // Unknown type
#define DEFA  "\e[0m"

#define ERROR_MSG 1
#define WARN_MSG  0
#define INFO_MSG  0

#define LOG_ERROR(type, msg) do { \
    if (ERROR_MSG) { \
        std::cerr << RED << "ERROR : " << msg << DEFA << std::endl; \
    } \
    if constexpr (std::is_same<type, std::runtime_error>::value || \
                  std::is_same<type, std::invalid_argument>::value || \
                  std::is_same<type, std::out_of_range>::value) { \
        throw type(msg); \
    } else { \
        std::cerr << PURPLE << "Invalid exception type passed to LOG_ERROR." << DEFA << std::endl; \
        throw std::runtime_error(""); \
    } \
} while (0)

#define LOG_WARN(msg) do { \
    if (WARN_MSG) \
        std::cout << YELLOW << "WARNING : " << msg << DEFA << std::endl; \
} while (0)

#define LOG_INFO(msg) do { \
    if (INFO_MSG) \
        std::cout << WHITE << "INFO : " << msg << DEFA << std::endl; \
} while (0)

} // snnf