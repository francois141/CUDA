#include <iostream>

// I am following this tutorial to improve my cuda skills https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf

__global__ void reduce0(int *in_data,int *out_data)
{
    __shared__ int s[1024];

    unsigned int index = threadIdx.x;
    s[index] = in_data[index];

    __syncthreads();

    for(unsigned int i = 1; i < blockDim.x; i *= 2)
    {
        if(index % (2*i) == 0)
        {
            s[index] += s[index+i];
        }
        __syncthreads();
    }

    if(index == 0){
        *out_data = s[0];
    }   
}

__global__ void reduce1(int *in_data,int *out_data)
{
    __shared__ int s[1024];

    unsigned int index = threadIdx.x;
    s[index] = in_data[index];

    __syncthreads();

    for(unsigned int i = 1; i < blockDim.x; i *= 2)
    {
        unsigned int currentIndex = 2 * i * index;
        if(currentIndex < blockDim.x)
        {
            s[currentIndex] += s[currentIndex + i];
        }
        __syncthreads();
    }

    if(index == 0){
        *out_data = s[0];
    }   
}

__global__ void reduce2(int *in_data,int *out_data)
{
    __shared__ int s[1024];

    unsigned int index = threadIdx.x;
    s[index] = in_data[index];

    __syncthreads();

    for(unsigned int i = blockDim.x / 2; i > 0; i >>= 1)
    {
        if(index < i) {
            s[index] += s[index + i];
        }
        __syncthreads();
    }

    if(index == 0){
        *out_data = s[0];
    }   
}

__global__ void reduce3(int *in_data,int *out_data)
{
    __shared__ int s[1024];

    unsigned int index = threadIdx.x;
    s[index] = in_data[index];
    s[index+512] = in_data[index+512];

    __syncthreads();

    for(unsigned int i = blockDim.x; i > 0; i >>= 1)
    {
        if(index < i) {
            s[index] += s[index + i];
        }
        __syncthreads();
    }

    if(index == 0){
        *out_data = s[0];
    } 
}

__device__ void warpReduce(volatile int*s, int index)
{
    s[index] += s[index+32];
    s[index] += s[index+16];
    s[index] += s[index+8];
    s[index] += s[index+4];
    s[index] += s[index+2];
    s[index] += s[index+1];
}

__global__ void reduce4(int *in_data,int *out_data)
{
    __shared__ int s[1024];

    unsigned int index = threadIdx.x;
    s[index] = in_data[index];
    s[2*index] = in_data[2*index];

    __syncthreads();

    for(unsigned int i = blockDim.x; i > 32; i >>= 1)
    {
        if(index < i) {
            s[index] += s[index + i];
        }
        __syncthreads();
    }

    if(index < 32) warpReduce(s,index);

    if(index == 0){
        *out_data = s[0];
    } 
}

int main(void)
{
    unsigned int size = 1024;

    int array[size];
    for(int i = 0; i < size;i++) {
        array[i] = 1;
    }

    int *cudaArray;
    int *cudaOut;
    int output[1];

    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMalloc(&cudaArray,size*sizeof(int));
    cudaMalloc(&cudaOut,sizeof(int));

    cudaMemcpy(cudaArray,array,size*sizeof(int),cudaMemcpyHostToDevice);

    cudaEventRecord(start);
    reduce4<<<1,512>>>(cudaArray,cudaOut);
    cudaEventRecord(stop);

    cudaMemcpy(output,cudaOut,sizeof(int),cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Time taken with GPU : " << milliseconds << " ms" << std::endl;

    std::cout << output[0] << std::endl;

    return 0;
}
