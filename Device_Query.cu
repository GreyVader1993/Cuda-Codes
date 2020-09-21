/**
/* * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
#include <stdio.h>
#include <stdlib.h>

static const int WORK_SIZE = 256;

/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 ///
#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }

__device__ unsigned int bitreverse(unsigned int number) {
	number = ((0xf0f0f0f0 & number) >> 4) | ((0x0f0f0f0f & number) << 4);
	number = ((0xcccccccc & number) >> 2) | ((0x33333333 & number) << 2);
	number = ((0xaaaaaaaa & number) >> 1) | ((0x55555555 & number) << 1);
	return number;
}

/**
 * CUDA kernel function that reverses the order of bits in each element of the array.
 *
__global__ void bitreverse(void *data) {
	unsigned int *idata = (unsigned int*) data;
	idata[threadIdx.x] = bitreverse(idata[threadIdx.x]);
}

/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 *
int main(void) {
	void *d = NULL;
	int i;
	unsigned int idata[WORK_SIZE], odata[WORK_SIZE];

	for (i = 0; i < WORK_SIZE; i++)
		idata[i] = (unsigned int) i;

	CUDA_CHECK_RETURN(cudaMalloc((void**) &d, sizeof(int) * WORK_SIZE));
	CUDA_CHECK_RETURN(
			cudaMemcpy(d, idata, sizeof(int) * WORK_SIZE, cudaMemcpyHostToDevice));

	bitreverse<<<1, WORK_SIZE, WORK_SIZE * sizeof(int)>>>(d);

	CUDA_CHECK_RETURN(cudaThreadSynchronize());	// Wait for the GPU launched work to complete
	CUDA_CHECK_RETURN(cudaGetLastError());
	CUDA_CHECK_RETURN(cudaMemcpy(odata, d, sizeof(int) * WORK_SIZE, cudaMemcpyDeviceToHost));

	for (i = 0; i < WORK_SIZE; i++)
		printf("Input value: %u, device output: %u\n", idata[i], odata[i]);

	CUDA_CHECK_RETURN(cudaFree((void*) d));
	CUDA_CHECK_RETURN(cudaDeviceReset());

	return 0;
}*/

#include<stdio.h>
#include<cuda.h>
#include<cuda_runtime_api.h>

void printDevProp(cudaDeviceProp devProp)
{
    printf("Major revision number:         %d\n",  devProp.major);
    printf("Minor revision number:         %d\n",  devProp.minor);
    printf("Name:                          %s\n",  devProp.name);
    printf("Total global memory:           %u\n",  devProp.totalGlobalMem);
    printf("Total shared memory per block: %u\n",  devProp.sharedMemPerBlock);
    printf("Total registers per block:     %d\n",  devProp.regsPerBlock);
    printf("Warp size:                     %d\n",  devProp.warpSize);
    printf("Maximum memory pitch:          %u\n",  devProp.memPitch);
    printf("Maximum threads per block:     %d\n",  devProp.maxThreadsPerBlock);
    for (int i = 0; i < 3; ++i)
    printf("Maximum dimension %d of block:  %d\n", i, devProp.maxThreadsDim[i]);
    for (int i = 0; i < 3; ++i)
    printf("Maximum dimension %d of grid:   %d\n", i, devProp.maxGridSize[i]);
    printf("Clock rate:                    %d\n",  devProp.clockRate);
    printf("Total constant memory:         %u\n",  devProp.totalConstMem);
    printf("Texture alignment:             %u\n",  devProp.textureAlignment);
    printf("Concurrent copy and execution: %s\n",  (devProp.deviceOverlap ? "Yes" : "No"));
    printf("Number of multiprocessors:     %d\n",  devProp.multiProcessorCount);
    printf("Kernel execution timeout:      %s\n",  (devProp.kernelExecTimeoutEnabled ? "Yes" : "No"));
    return;
}




int main()
{

int devCount;

cudaGetDeviceCount(&devCount);
printf("\nHello From Main code:......\n");
printf("CUDA Device Query...\n");
printf("There are %d CUDA devices.\n", devCount);

for (int i = 0; i < devCount; ++i)
    {
        // Get device properties
        printf("\nCUDA Device #%d\n", i);
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, i);
        printDevProp(devProp);
    }

    printf("\nPress any key to exit...");
    char c;
    scanf("%c", &c);


return 0;
}
/*
// Print device properties
void printDevProp(cudaDeviceProp devProp)
{
    printf("Major revision number:         %d\n",  devProp.major);
    printf("Minor revision number:         %d\n",  devProp.minor);
    printf("Name:                          %s\n",  devProp.name);
    printf("Total global memory:           %u\n",  devProp.totalGlobalMem);
    printf("Total shared memory per block: %u\n",  devProp.sharedMemPerBlock);
    printf("Total registers per block:     %d\n",  devProp.regsPerBlock);
    printf("Warp size:                     %d\n",  devProp.warpSize);
    printf("Maximum memory pitch:          %u\n",  devProp.memPitch);
    printf("Maximum threads per block:     %d\n",  devProp.maxThreadsPerBlock);
    for (int i = 0; i < 3; ++i)
    printf("Maximum dimension %d of block:  %d\n", i, devProp.maxThreadsDim[i]);
    for (int i = 0; i < 3; ++i)
    printf("Maximum dimension %d of grid:   %d\n", i, devProp.maxGridSize[i]);
    printf("Clock rate:                    %d\n",  devProp.clockRate);
    printf("Total constant memory:         %u\n",  devProp.totalConstMem);
    printf("Texture alignment:             %u\n",  devProp.textureAlignment);
    printf("Concurrent copy and execution: %s\n",  (devProp.deviceOverlap ? "Yes" : "No"));
    printf("Number of multiprocessors:     %d\n",  devProp.multiProcessorCount);
    printf("Kernel execution timeout:      %s\n",  (devProp.kernelExecTimeoutEnabled ? "Yes" : "No"));
    return;
}

int main()
{
    // Number of CUDA devices
    int devCount;
    cudaGetDeviceCount(&devCount);
    printf("CUDA Device Query...\n");
    printf("There are %d CUDA devices.\n", devCount);

    // Iterate through devices
    for (int i = 0; i < devCount; ++i)
    {
        // Get device properties
        printf("\nCUDA Device #%d\n", i);
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, i);
        printDevProp(devProp);
    }

    printf("\nPress any key to exit...");
    char c;
    scanf("%c", &c);

    return 0;
}




*/








