#pragma once

#include "../Core/Tensor.h"

namespace cunet
{
    template<typename T>
    class Linear
    {
        public:

        Linear() = default;

        Linear(const uint32_t& in, const uint32_t& out, T* weight, T* bias);

        ~Linear();

        __device__
        Tensor<T>&
        operator()(Tensor<T>& input);

        __device__
        void
        operator()(T* input, T* output);

        private:
        T* _weight;
        T* _bias;
        uint32_t _size_in;
        uint32_t _size_out;
        Tensor<T> _output;
    };

    using LinearFloat = Linear<float>;
}


#include "../../src/Layer/LinearDetail.h"