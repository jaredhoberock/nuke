#include <cstdio>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>
#include <thrust/fill.h>
#include <nuke/atomic.hpp>

__global__ void inc_kernel(nuke::atomic_int *a_ptr)
{
  (*a_ptr)++;
}

__global__ void dec_kernel(nuke::atomic_int *a_ptr)
{
  (*a_ptr)--;
}

__global__ void print_kernel(nuke::atomic_int *a_ptr)
{
  printf("result is %d\n", (int)*a_ptr);
}

int main()
{
  thrust::device_ptr<nuke::atomic_int> a_ptr = thrust::device_malloc<nuke::atomic_int>(1);

  thrust::fill(a_ptr, a_ptr + 1, 0);

  inc_kernel<<<2,512>>>(a_ptr.get());

  print_kernel<<<1,1>>>(a_ptr.get());

  dec_kernel<<<2,512>>>(a_ptr.get());

  print_kernel<<<1,1>>>(a_ptr.get());

  thrust::device_free(a_ptr);

  return 0;
}

