#pragma once

#include "../Core/CUDA.h"

namespace cunet
{
    template<typename T>
    struct ReLU
    {
        ReLU() = default;

        __device__
        T operator()(const T& input);

        __device__
        Tensor<T>& operator()(Tensor<T>& input);

        __device__
        void operator()(T* input, T* output, const uint32_t& size);
    };

    template<typename T>
    struct Tanh
    {
        Tanh() = default;

        __device__
        T operator()(const T& input);

        __device__
        Tensor<T>& operator()(Tensor<T>& input);

        __device__
        void operator()(T* input, T* output, const uint32_t& size);
    };

    template<typename T>
    struct Sigmoid
    {
        Sigmoid() = default;

        __device__
        T operator()(const T& input);

        __device__
        Tensor<T>& operator()(Tensor<T>& input);

        __device__
        void operator()(T* input, T* output, const uint32_t& size);
    };

    template<typename T>
    struct Softplus
    {
        Softplus(const T& beta = 1);

        __device__
        T operator()(const T& input);

        __device__
        Tensor<T>& operator()(Tensor<T>& input);

        __device__
        void operator()(T* input, T* output, const uint32_t& size);

        private:
        T beta;
    };

    template<typename T, class Activation>
    __device__
    void forward(Tensor<T>& input, Activation& act_function);
}

#include "../../src/Layer/ActivationsDetail.h"