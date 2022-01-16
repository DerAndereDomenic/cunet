#include "../lib/include/CUNET.h"
#include "TestModel.h"


__global__ void test(cunet::TestModel mlp)
{
    printf("Start\n");
    float input[] = { 1,2,3,4,5,6,7,8,9,10 };
    cunet::Tensor<float> inp;
    inp.setData<10>(input);
    cunet::Tensor<float>& output = mlp(inp);

    for(uint32_t i = 0; i < output.size(); ++i)
    {
        printf("%f\n", output[i]);
    }
    printf("End\n");
}

int main()
{
    cunet::TestModel mlp;

    test << <1, 1 >> > (mlp);
    cudaSafeCall(cudaDeviceSynchronize());

    return 0;
}