#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_SIZE 32

__host__ __device__ size_t ceilDiv(size_t x, size_t y) {
  return 1 + ((x - 1) / y);
}

__global__ void conv_forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    y - output
    x - input
    k - kernel
    B - batch_size (number of images in x)
    M - number of output feature maps
    C - number of input feature maps
    H - input height dimension
    W - input width dimension
    K - kernel height and width (K x K)
    */

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = y4d(0,0,0,0)
    // y4d(0,0,0,0) = a

#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
#define k2d(i1, i0) k_shared[(i1) * K + (i0)]
#define x2d(i1, i0) x_shared[(i1) * W_x + (i0)]

    // Insert your GPU convolution kernel code here
    const size_t b = blockIdx.x;
    const size_t m = blockIdx.y;
    const size_t W_grid = ceilDiv(W_out, TILE_SIZE);
    const size_t W_x = TILE_SIZE + K - 1;
    const size_t H_x = TILE_SIZE + K - 1;
    const size_t h_base = TILE_SIZE * (blockIdx.z / W_grid);
    const size_t w_base = TILE_SIZE * (blockIdx.z % W_grid);
    const size_t h_thread = threadIdx.y;
    const size_t w_thread = threadIdx.x;
    const size_t h = h_base + h_thread;
    const size_t w = w_base + w_thread;

    extern __shared__ float shared[];
    float *x_shared = &shared[0];
    float *k_shared = &shared[W_x * H_x];

    float res = 0.0;
    for (size_t c = 0; c < C; ++c) {
      // Load slice of kernel for m, c
      __syncthreads();
      if (h_thread < K && w_thread < K)
        k2d(h_thread, w_thread) = k4d(m, c, h_thread, w_thread);

      // Load slice of x for b, c
      for (size_t i = h_thread; i < H_x; i += TILE_SIZE)
        for (size_t j = w_thread; j < W_x; j += TILE_SIZE)
          x2d(i, j) = x4d(b, c, h_base + i, w_base + j);

      __syncthreads();
      for (size_t p = 0; p < K; ++p) { 
        for (size_t q = 0; q < K; ++q) {
          res += x2d(h_thread + p, w_thread + q) * k2d(p, q);
        }
      }
    }

    if (h < H_out && w < W_out)
      y4d(b, m, h, w) = res;

#undef y4d
#undef x4d
#undef k4d
#undef k2d
#undef x2d
}

__host__ void checkError()
{
  cudaError_t error = cudaGetLastError();
  if(error != cudaSuccess)
  {
    std::cerr<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    exit(-1);
  }
}

__host__ void GPUInterface::conv_forward_gpu(float *host_y, const float *host_x, const float *host_k, const int B, const int M, const int C, const int H, const int W, const int K)
{
  // Declare relevant device pointers
  const size_t H_out = H - K + 1;
  const size_t W_out = W - K + 1;
  const size_t elems_y = B * M * H_out * W_out;
  const size_t size_y = sizeof(float) * elems_y;
  const size_t elems_x = B * C * H * W;
  const size_t size_x = sizeof(float) * elems_x;
  const size_t elems_k = M*C*K*K;
  const size_t size_k = sizeof(float) * elems_k;

  float *device_y, *device_x, *device_k;

  // Allocate memory and copy over the relevant data structures to the GPU
  cudaMalloc(&device_y, size_y);
  cudaMalloc(&device_x, size_x);
  cudaMalloc(&device_k, size_k);
  

  cudaMemcpy(device_x, host_x, size_x, cudaMemcpyHostToDevice);
  cudaMemcpy(device_k, host_k, size_k, cudaMemcpyHostToDevice);
  

  // Set the kernel dimensions and call the kernel
  const size_t W_block = TILE_SIZE;
  const size_t H_block = W_block; // square
  const size_t Z_grid = ceilDiv(W_out, W_block) * ceilDiv(H_out, H_block);

  const size_t elems_x_shared = (W_block + K - 1) * (H_block + K - 1);
  const size_t elems_k_shared = K*K;
  const size_t size_shared = sizeof(float) * (elems_x_shared + elems_k_shared);

  const dim3 blockDim(W_block, H_block);
  const dim3 gridDim(B, M, Z_grid);

  conv_forward_kernel<<<gridDim, blockDim, size_shared>>>(device_y, device_x, device_k, B, M, C, H, W, K);
  

  // Copy the output back to host
  cudaMemcpy(host_y, device_y, size_y, cudaMemcpyDeviceToHost);
  

  // Free device memory
  cudaFree(device_y);
  cudaFree(device_x);
  cudaFree(device_k);
  

  // Useful snippet for error checking
  cudaError_t error = cudaGetLastError();
  if(error != cudaSuccess)
  {
      std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
      exit(-1);
  }
}

__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}
