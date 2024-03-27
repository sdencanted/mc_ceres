// This program performs sum reduction with an optimization
// removing warp bank conflicts
// By: Nick from CoffeeBeforeArch

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <iostream>

#include <jetson-utils/cudaMappedMemory.h>
using namespace cooperative_groups;

// Reduces a thread group to a single element
__device__ float reduce_sum(thread_group g, float *temp, float val){
	int lane = g.thread_rank();

	// Each thread adds its partial sum[i] to sum[lane+i]
	for (int i = g.size() / 2; i > 0; i /= 2){
		temp[lane] = val;
		// wait for all threads to store
		g.sync();
		if (lane < i) {
			val += temp[lane + i];
		}
		// wait for all threads to load
		g.sync();
	}
	// note: only thread 0 will return full sum
	return val; 
}

// Creates partials sums from the original array
__device__ float thread_sum(float *input, int n){
	float sum = 0;
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	for (int i = tid; i < n / 2; i += blockDim.x * gridDim.x){
		// Cast as int4 
		float2 in = ((float2*)input)[i];
		sum += in.x + in.y;
	}
	return sum;
}

__global__ void sum_reduction(float *sum, float *input, int n){
	// Create partial sums from the array
	float my_sum = thread_sum(input, n);

	// Dynamic shared memory allocation
	extern __shared__ float temp[];
	
	// Identifier for a TB
	auto g = this_thread_block();
	
	// Reudce each TB
	float block_sum = reduce_sum(g, temp, my_sum);

	// Collect the partial result from each TB
	if (g.thread_rank() == 0) {
		atomicAdd(sum, block_sum);
	}
}

float gpu_sum_reduce(float* d_in, unsigned int d_in_len){

	// result vector
    float sum_cpu = 0;
	float *sum;


    cudaMalloc((void **)&sum, (unsigned int)sizeof(float));
    cudaMemcpy( sum,&sum_cpu, sizeof(float), cudaMemcpyHostToDevice);

	// TB Size
	int TB_SIZE = 128;

	// Grid Size (cut in half)
	int GRID_SIZE = (d_in_len + TB_SIZE - 1) / TB_SIZE;

	// Call kernel with dynamic shared memory (Could decrease this to fit larger data)
	sum_reduction <<<GRID_SIZE, TB_SIZE, d_in_len * sizeof(float)>>> (sum, d_in, d_in_len);

	// Synchronize the kernel
	// cudaDeviceSynchronize();
    cudaMemcpy( &sum_cpu,sum, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(sum);
	return sum_cpu;
}