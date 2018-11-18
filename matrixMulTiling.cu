/*
============================================================================
Name        : CUDA_MatrixMul.cu
Author      : Liuzzo Mauro
Version     :
Copyright   :
Description : CUDA multiply matrices
============================================================================
*/

#include <iostream>
#include <numeric>
#include <stdlib.h>
#define TILE_WIDTH 16

static void CheckCudaErrorAux(const char *, unsigned, const char *,
    cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

/**
* Check the return value of the CUDA runtime API call and exit
* the application if the call has failed.
*/
static void CheckCudaErrorAux(const char *file, unsigned line,
    const char *statement, cudaError_t err) {
    if (err == cudaSuccess)
        return;
    std::cerr << statement << " returned " << cudaGetErrorString(err) << "("
        << err << ") at " << file << ":" << line << std::endl;
    exit(1);
}

float *generate_random_data(int length) {
    float *data = (float *)malloc(sizeof(float) * length);
    for (int i = 0; i < length; i++) {
        data[i] = ((float)(rand() % 20) - 5) / 5.0f;
    }
    return data;
}

float *generate_fixed_data(int length) {
    float *data = (float *)malloc(sizeof(float) * length);
    int value = 1;
    for (int i = 0; i < length; i++) {
        data[i] = value++;
    }
    return data;
}

// Compute C = A * B
__global__ void matrixMultiply(float *A, float *B, float *C, int numARows,
    int numAColumns, int numBRows, int numBColumns) {

    // matrix portions to be loaded into the shared memory
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    // thread coordinates computation
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    // some useful variables
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // intermediate value for the calculation of the product
    float Pvalue = 0;

    for (int ph = 0; ph < (numAColumns-1) / TILE_WIDTH + 1; ph++) {
        // verify if the thread has some data to load
        if (row < numARows && ph * TILE_WIDTH + tx < numAColumns) {
            Mds[ty][tx] = A[row * numAColumns + ph * TILE_WIDTH + tx];
        }
        else {
            Mds[ty][tx] = 0.0;
        }

        if (col < numBColumns && ph * TILE_WIDTH + ty < numBRows) {
            Nds[ty][tx] = B[(ph * TILE_WIDTH + threadIdx.y) * numBColumns + col];
        }
        else {
            Nds[ty][tx] = 0.0;
        }

       __syncthreads();

       for (int i = 0; i < TILE_WIDTH; i++) {
    	   Pvalue += Mds[ty][i] * Nds[i][tx];
       }

        __syncthreads();
    }

    // verify if the thread has some data to write, based on output organization
    if (row < numARows && col < numBColumns){
        C[row*numBColumns + col] = Pvalue;
    }
}

void print_matrix(float* matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main() {
    float *hostA; // The A matrix
    float *hostB; // The B matrix
    float *hostC; // The output C matrix
    float *deviceA;
    float *deviceB;
    float *deviceC;
    int numARows = 2;    // number of rows in the matrix A
    int numAColumns = 3; // number of columns in the matrix A
    int numBRows = 3;    // number of rows in the matrix B
    int numBColumns = 4; // number of columns in the matrix B
    int numCRows = numARows;
    int numCColumns = numBColumns;
    // allocate host memory
    hostA = generate_fixed_data(numARows * numAColumns);
    hostB = generate_fixed_data(numBRows * numBColumns);
    hostC = (float*)malloc(numCRows * numCColumns * sizeof(float));
    // allocate device memory
    CUDA_CHECK_RETURN(
        cudaMalloc((void **)&deviceA,
            sizeof(float) * numARows * numAColumns));
    CUDA_CHECK_RETURN(
        cudaMalloc((void **)&deviceB,
            sizeof(float) * numBRows * numBColumns));
    CUDA_CHECK_RETURN(
        cudaMalloc((void **)&deviceC,
            sizeof(float) * numCRows * numCColumns));
    // copy from host to device memory
    CUDA_CHECK_RETURN(
        cudaMemcpy(deviceA, hostA, numARows * numAColumns * sizeof(float),
            cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(
        cudaMemcpy(deviceB, hostB, numBRows * numBColumns * sizeof(float),
            cudaMemcpyHostToDevice));
    // organize grid
    dim3 blockDim(16, 16);
    dim3 gridDim(ceil(((float)numAColumns) / blockDim.x),
        ceil(((float)numBRows) / blockDim.y));
    // execute kernel
    matrixMultiply << <gridDim, blockDim >> >(deviceA, deviceB, deviceC, numARows,
        numAColumns, numBRows, numBColumns);
    cudaDeviceSynchronize();
    // copy results from device memory to host
    CUDA_CHECK_RETURN(
        cudaMemcpy(hostC, deviceC, numARows * numBColumns * sizeof(float),
            cudaMemcpyDeviceToHost));
    print_matrix(hostA, numARows, numAColumns);
    print_matrix(hostB, numBRows, numBColumns);
    print_matrix(hostC, numCRows, numCColumns);
    // Free the GPU memory here
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);
    // Free host memory
    free(hostA);
    free(hostB);
    free(hostC);
}
