#pragma once

#include <cmath>

namespace cunet
{
    template<typename T>
    __device__
    T 
    ReLU<T>::operator()(const T& input)
    {
        return fmaxf(0, input);
    }

    template<typename T>
    __device__
    Tensor<T>& 
    ReLU<T>::operator()(Tensor<T>& input)
    {
        forward(input, *this);
        return input;
    }

    template<typename T>
    __device__
    T 
    Tanh<T>::operator()(const T& input)
    {
        T e = expf(input);
        T me = expf(-input);
        return (e - me) / (e + me);
    }

    template<typename T>
    __device__
    Tensor<T>& 
    Tanh<T>::operator()(Tensor<T>& input)
    {
        forward(input, *this);
        return input;
    }

    template<typename T>
    __device__
    T 
    Sigmoid<T>::operator()(const T& input)
    {
        return static_cast<T>(1.0) / (static_cast<T>(1.0) + expf(-input));
    }

    template<typename T>
    __device__
    Tensor<T>& 
    Sigmoid<T>::operator()(Tensor<T>& input)
    {
        forward(input, *this);
        return input;
    }

    template<typename T>
    Softplus<T>::Softplus(const T& beta)
        :beta(beta)
    {
        
    }

    template<typename T>
    __device__
    T 
    Softplus<T>::operator()(const T& input)
    {
        return logf(static_cast<T>(1) + expf(beta * input)) / beta;
    }

    template<typename T>
    __device__
    Tensor<T>& 
    Softplus<T>::operator()(Tensor<T>& input)
    {
        forward(input, *this);
        return input;
    }

    template<typename T, class Activation>
    __device__
    void 
    forward(Tensor<T>& input, Activation& act_function)
    {
        for(uint32_t i = 0; i < input.size(); ++i)
        {
            input[i] = act_function(input[i]);
        }
    }
}