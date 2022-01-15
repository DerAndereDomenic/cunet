#pragma once

#include "CUDA.h"

namespace cunet
{
    template <typename T>
    class Tensor
    {
        public:
        Tensor() = default;

        ~Tensor();

        Tensor(T* data, const uint32_t& size);

        //Creates tensor with zeros
        Tensor(const uint32_t& size);

        __device__
        T* data();

        __device__
        T& operator[](unsigned int index);

        __device__
        uint32_t size() const;

        __device__
        void merge(Tensor<T>& t1, Tensor<T>& t2);

        private:
        T* _data = nullptr;
        uint32_t _size = 0;
    };

    using TensorFloat = Tensor<float>;
}

#include "../../src/Core/TensorDetail.h"