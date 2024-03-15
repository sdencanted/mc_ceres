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
__device__ double reduce_sum(thread_group g, double *temp, double val){
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
__device__ double thread_sum(double *input, int n){
	double sum = 0;
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	for (int i = tid; i < n / 2; i += blockDim.x * gridDim.x){
		// Cast as int4 
		double2 in = ((double2*)input)[i];
		sum += in.x + in.y;
	}
	return sum;
}

__global__ void sum_reduction(double *sum, double *input, int n){
	// Create partial sums from the array
	double my_sum = thread_sum(input, n);

	// Dynamic shared memory allocation
	extern __shared__ double temp[];
	
	// Identifier for a TB
	auto g = this_thread_block();
	
	// Reudce each TB
	double block_sum = reduce_sum(g, temp, my_sum);

	// Collect the partial result from each TB
	if (g.thread_rank() == 0) {
		atomicAdd(sum, block_sum);
	}
}

double gpu_sum_reduce(double* d_in, unsigned int d_in_len){

	// result vector
    double sum_cpu = 0;
	double *sum;


    cudaMalloc((void **)&sum, (unsigned int)sizeof(double));
    cudaMemcpy( sum,&sum_cpu, sizeof(double), cudaMemcpyHostToDevice);

	// TB Size
	int TB_SIZE = 128;

	// Grid Size (cut in half)
	int GRID_SIZE = (d_in_len + TB_SIZE - 1) / TB_SIZE;

	// Call kernel with dynamic shared memory (Could decrease this to fit larger data)
	sum_reduction <<<GRID_SIZE, TB_SIZE, d_in_len * sizeof(double)>>> (sum, d_in, d_in_len);

	// Synchronize the kernel
	// cudaDeviceSynchronize();
    cudaMemcpy( &sum_cpu,sum, sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(sum);
	return sum_cpu;
}