#ifndef REDUCE_H__
#define REDUCE_H__

#define MAX_BLOCK_SZ 1024

float gpu_sum_reduce(float* d_in, unsigned int d_in_len);
void debug();

#endif // !REDUCE_H__