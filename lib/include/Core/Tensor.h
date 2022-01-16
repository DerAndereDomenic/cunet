#pragma once

#include "CUDA.h"

namespace cunet
{
    template <typename T>
    class Tensor
    {
        public:
        Tensor() = default;

        //These methods are to create presized tensors on the cpu
        static Tensor<T>
        createDeviceTensor(T* data, const uint32_t& size);

        //Creates tensor with zeros
        static Tensor<T>
        createDeviceTensor(const uint32_t& size);

        static void
        destroyDeviceTensor(Tensor<T>& tensor);

        /// 
        /// This creates a temporary device tensor.
        /// This tensor is only valid as long as 'data' is alive
        /// Only the pointer to the data is stored, no copy is created
        /// 
        template<uint32_t N>
        __device__
        void
        setData(T data[N]);

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