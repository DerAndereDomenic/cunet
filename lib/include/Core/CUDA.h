#ifndef __CUNET_CORE_CUDA_H
#define __CUNET_CORE_CUDA_H

#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace cunet
{
    inline
        void check(cudaError_t error, char const* const func, const char* const file, int const line)
    {
        if (error != cudaSuccess)
        {
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
                    static_cast<unsigned int>(error), cudaGetErrorString(error), func);
        }
    }

    #ifdef _DEBUG
    #define cudaSafeCall(val) cunet::check((val), #val, __FILE__, __LINE__)
    #else
    #define cudaSafeCall(val) val
    #endif
}

#endif