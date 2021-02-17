#include <iostream>
#include <math.h>
#include <chrono>

template<typename T> __host__ __device__ inline void op(T &a,T &b)
{
  a = a + b;
}

__global__ void add(int n,float *x, float *y)
{
  int stride = blockDim.x * gridDim.x;
  int index  = blockIdx.x * blockDim.x + threadIdx.x;
  for(int i = index; i < n;i+=stride)
  {
    op<float>(y[i],x[i]);
  }
}

void addNormal(int n,float *x,float *y)
{
  for(int i = 0; i <n;i++) {
    op<float>(x[i],y[i]);
  }
}

int main(void)
{
  /** CHECK DEVICES FIRST **/

  int numberDevices;
  cudaGetDeviceCount(&numberDevices);

  for(int i = 0; i < numberDevices;i++)
  {
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop,i);

    if(err != cudaSuccess)
      std::cout << "There is an error ! " << std::endl;

    std::cout << "Device : " << i << std::endl;
    std::cout << "Name : " << prop.name <<  std::endl;
    std::cout << "Total global memory : " << prop.totalGlobalMem << std::endl;
    std::cout << "Max Threads per blocks : " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "Major : " << prop.major << std::endl;
    std::cout << "Minor : " << prop.minor << std::endl;
    std::cout << std::endl;
  }

  /** TEST CUDA WITH A SIMPLE CODE **/

  int N = 1<<20;
  float *x, *y;

  cudaEvent_t start,stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));

  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  cudaEventRecord(start);

  dim3 gridSize = dim3(1024,0,0);
  add<<<N/1024, gridSize>>>(N, x, y);
  
  cudaEventRecord(stop);

  cudaDeviceSynchronize();

  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  std::cout << "Time take with GPU : " << milliseconds << " ms" << std::endl;

  cudaFree(x);
  cudaFree(y);

  cudaError_t errSync  = cudaGetLastError();
  cudaError_t errAsync = cudaDeviceSynchronize();

  x = (float*)malloc(N*sizeof(float));
  y = (float*)malloc(N*sizeof(float));

  auto start2 = std::chrono::steady_clock::now();
  addNormal(N,x,y);
  auto end2 = std::chrono::steady_clock::now();

  auto diff2 = end2 - start2;
  std::cout << "Time take with CPU : " << std::chrono::duration <double, std::milli> (diff2).count() << " ms" << std::endl;

  return 0;
}
