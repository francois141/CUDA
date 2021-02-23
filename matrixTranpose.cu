#include <stdio.h>
#include <assert.h>
#include <iostream>

const unsigned int TILE_DIM = 32;
const unsigned int BLOCK_ROWS = 32;

const unsigned int SIZE = 256;

// In this case we transpose the programm with a reduced amount of bank conflicts and coalescing
__global__ void transposeOptmized(int *odata,int *idata)
{
    __shared__ float tile[TILE_DIM][TILE_DIM+1];

    // Setup some variables
    int sizeX = blockIdx.x * TILE_DIM + threadIdx.x;
    int sizeY = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;

    // Copy to the shared memory
    tile[threadIdx.y][threadIdx.x] = idata[(sizeY)*width + sizeX];
    
    // Add a barrier
    __syncthreads();

    // Compute index
    sizeX = blockIdx.y * TILE_DIM + threadIdx.x;  
    sizeY = blockIdx.x * TILE_DIM + threadIdx.y;

    // Tranpose back
    odata[(sizeY)*width + sizeX] = tile[threadIdx.x][threadIdx.y];
}

// Naive implementation - not very efficient
__global__ void transposeNaive(int *odata,int *idata)
{
    // Compute Index
    unsigned int indexX = blockIdx.x * TILE_DIM + threadIdx.x;
    unsigned int indexY = blockIdx.y * TILE_DIM + threadIdx.y;
    unsigned int width  = gridDim.x * TILE_DIM;

    // Perform a dummy swap
    for(int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        odata[indexX*width + indexY + j] = idata[(indexY+j)*width + indexX];
    }

}

int main() {

    dim3 dimGrid(SIZE/TILE_DIM,SIZE/TILE_DIM);
    dim3 dimBlock(TILE_DIM,BLOCK_ROWS,1);

    int matrix[SIZE][SIZE];

    for(int i = 0; i < SIZE;i++)
    {
        for(int j = 0; j < SIZE;j++)
        {
            matrix[i][j] = i;
        }
    }

    /** CHECK INPUT **/
    for(int i = 0; i < 10;i++) {
        for(int j = 0; j < 10;j++) {
            std::cout << matrix[i][j];
        } std::cout << std::endl;
    } std::cout << std::endl;

    // copy memory to the device

    int *cudaMatrix1;
    int *cudaMatrix2;

    cudaMalloc(&cudaMatrix1,SIZE*SIZE*sizeof(int));
    cudaMalloc(&cudaMatrix2,SIZE*SIZE*sizeof(int));

    cudaMemcpy(cudaMatrix1,matrix,sizeof(int)*SIZE*SIZE,cudaMemcpyHostToDevice);
    transposeOptmized<<<dimGrid,dimBlock>>>(cudaMatrix2,cudaMatrix1);
    cudaMemcpy(matrix,cudaMatrix2,sizeof(int)*SIZE*SIZE,cudaMemcpyDeviceToHost);

    /** CHECK OUTPUT **/
    for(int i = 0; i < 10;i++) {
        for(int j = 0; j < 10;j++) {
            std::cout << matrix[i][j];
        } std::cout << std::endl;
    }
}   
