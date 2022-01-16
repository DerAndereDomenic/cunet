#pragma once

namespace cunet
{
    template<typename T>
    Linear<T>::Linear(const uint32_t& in, const uint32_t& out, T* weights, T* bias)
        :_size_in(in),
         _size_out(out),
         _output(Tensor<T>::createDeviceTensor(out))
    {
        _weight = Memory::createDeviceArray<T>(in * out);
        Memory::copyHost2DeviceArray<T>(in * out, weights, _weight);
        _bias = Memory::createDeviceArray<T>(out);
        Memory::copyHost2DeviceArray<T>(out, bias, _bias);
    }

    template<typename T>
    Linear<T>::~Linear()
    {
        //TODO
        //Memory::destroyDeviceArray<T>(_weight);
        //Memory::destroyDeviceArray<T>(_bias);
    }

    template<typename T>
    __device__
    Tensor<T>&
    Linear<T>::operator()(Tensor<T>& input)
    {
        for(uint32_t i = 0; i < _size_out; ++i) //Rows
        {
            T result = 0;
            for(uint32_t j = 0; j < _size_in; ++j) //Cols
            {
                result += _weight[i * _size_in + j] * input[j];
            }
            _output[i] = result + _bias[i];
        }

        return _output;
    }
}