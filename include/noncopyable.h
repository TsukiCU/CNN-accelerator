#ifndef _CUDA_NONCOPYABLE_H__
#define _CUDA_NONCOPYABLE_H__

namespace cuda {

class Noncopyable {
public:
    Noncopyable() = default;
    ~Noncopyable() = default;
    Noncopyable(const Noncopyable&) = delete;
    Noncopyable& operator=(const Noncopyable&) = delete;
};

}

#endif // _CUDA_NONCOPYABLE_H__