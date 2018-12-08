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

// runs on device, called by host
__global__ void gpuAdd(int *a, int *b, int *c){
	*c = *a + *b;
}

int main(int argc, char **argv)
{
	int a, b, c; // host variables
	int *d_a, *d_b, *d_c; // device variables

	// allocate device memory
	cudaMalloc((void**)&d_a, sizeof(int));
	cudaMalloc((void**)&d_b, sizeof(int));
	cudaMalloc((void**)&d_c, sizeof(int));

	printf("insert first value: ");
	scanf("%d", &a);
	printf("insert second value: ");
	scanf("%d", &b);

	// copy inputs to device
	cudaMemcpy(d_a, &a, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, &b, sizeof(int), cudaMemcpyHostToDevice);

	// kernel launch
	puts("Launch kernel");
	gpuAdd<<<1,1>>>(d_a, d_b, d_c);

	// copy output to host
	cudaMemcpy(&c, d_c, sizeof(int), cudaMemcpyDeviceToHost);

	printf("%d+%d=%d", a, b, c);

	// free device memory
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	return 0;
}
