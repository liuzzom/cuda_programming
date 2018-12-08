/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#define N 512

void fillArray(int *array, int size){
	int i;
	for(i = 0; i < size; i++){
		array[i] = 1;
	}
}

void printInFile(int *array, int size){
	int i;
	FILE *fPtr;

	if((fPtr = fopen("result.txt", "w")) == NULL){
		puts("error on file...");
		exit(1);
	}
	printf("printing on file\n");

	for(i = 0; i < size; i++){
		fprintf(fPtr, "%d ", array[i]);
	}
	fprintf(fPtr, "\n");

	fclose(fPtr);
}

void printArray(int *array, int size){
	int i;
	for(i = 0; i < size; i++){
		printf("%d ", array[i]);
	}
	printf("\n");
}

// runs on device, called by host
__global__ void gpuAdd(int *a, int *b, int *c){
	c[threadIdx.x] = a[threadIdx.x]  + b[threadIdx.x];
}

int main(int argc, char **argv)
{
	int size = N * sizeof(int);
	// host variables
	int *a = (int *)malloc(size);
	int *b = (int *)malloc(size);
	int *c = (int *)malloc(size);
	int *d_a, *d_b, *d_c; // device variables
	int i;


	// allocate device memory
	cudaMalloc((void**)&d_a, size);
	cudaMalloc((void**)&d_b, size);
	cudaMalloc((void**)&d_c, size);

	fillArray(a, N);
	fillArray(b, N);

	// copy inputs to device
	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

	// kernel launch
	puts("Launch kernel");
	// 1 block of 512 threads (1D threads vector)
	gpuAdd<<<1,N>>>(d_a, d_b, d_c);

	// copy output to host
	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
	printInFile(c, N);

	// free host memory
	free(a);
	free(b);
	free(c);

	// free device memory
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	return 0;
}
