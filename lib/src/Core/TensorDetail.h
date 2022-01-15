#pragma once

#include "../../include/Core/Memory.h"

namespace cunet
{
    template<typename T>
    Tensor<T>::Tensor(T* data, const uint32_t& size)
        :_size(size)
    {
        _data = Memory::createDeviceArray<T>(size);
        Memory::copyHost2DeviceArray<T>(size, data, _data);
    }
    
    template<typename T>
    Tensor<T>::Tensor(const uint32_t& size)
        :_size(size)
    {
        _data = Memory::createDeviceArray<T>(size);
    }

    template<typename T>
    Tensor<T>::~Tensor()
    {
        //TODO
        //Memory::destroyDeviceArray<T>(_data);
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