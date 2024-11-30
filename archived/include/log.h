#ifndef _CUDA_LOG_H
#define _CUDA_LOG_H

#include <iostream>

namespace cuda {

#define CYAN    "\e[1;36m"  // Log info
#define YELLOW  "\e[1;33m"  // Log warning
#define RED     "\e[1;31m"  // Log Fatal & Log error
#define DEFA    "\e[0m"

#define LOG_FATAL(msg) do { \
    std::cerr << RED << "FATAL : " << msg << DEFA << std::endl; \
    throw std::runtime_error(""); \
} while (0)

#define LOG_ERROR(msg) do { \
    std::cerr << RED << "ERROR : " << msg << DEFA << std::endl; \
    throw std::runtime_error(""); \
} while (0)

#define LOG_WARN(msg) do { std::cout << YELLOW << "WARNING : " << msg << DEFA << std::endl; } while (0)
#define LOG_INFO(msg) do { std::cout << CYAN << "INFO : " << msg << DEFA << std::endl; } while (0)

}

#endif