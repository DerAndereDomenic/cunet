#pragma once

#include "../../include/Core/Memory.h"

namespace cunet
{
    template<typename T>
    Tensor<T>
    Tensor<T>::createDeviceTensor(T* data, const uint32_t& size)
    {
        Tensor<T> result;
        result._data = Memory::createDeviceArray<T>(size);
        Memory::copyHost2DeviceArray<T>(size, data, result._data);
        result._size = size;
        return result;
    }
    
    template<typename T>
    Tensor<T>
    Tensor<T>::createDeviceTensor(const uint32_t& size)
    {
        Tensor<T> result;
        result._data = Memory::createDeviceArray<T>(size);
        result._size = size;

        return result;
    }

    template<typename T>
    void
    Tensor<T>::destroyDeviceTensor(Tensor<T>& t)
    {
        Memory::destroyDeviceArray<T>(t._data);
    }

    template<typename T>
    template<uint32_t N>
    __device__
    void
    Tensor<T>::setData<N>(T data[N])
    {
        _data = data;
        _size = N;
    }

    template<typename T>
    __device__
    T* 
    Tensor<T>::data()
    {
        return _data;
    }
    
    template<typename T>
    __device__
    T& 
    Tensor<T>::operator[](unsigned int index)
    {
        return _data[index];
    }
    
    template<typename T>
    __device__
    uint32_t 
    Tensor<T>::size() const
    {
        return _size;
    }

    template<typename T>
    __device__
    void 
    Tensor<T>::merge(Tensor<T>& t1, Tensor<T>& t2)
    {
        uint32_t index = 0;

        for(uint32_t i = 0; i < t1.size(); ++i)
        {
            _data[index] = t1[i];
            ++index;
        }

        for(uint32_t i = 0; i < t2.size(); ++i)
        {
            _data[index] = t2[i];
            ++index;
        }

    }

}