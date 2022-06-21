//#include <torch/torch.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
//#include <torch/serialize/tensor.h>
//#include <ATen/ATen.h>
//#include <ATen/cuda/CUDAContext.h>

#define CUDA_NUM_THREADS 256 
#define THREADS_PER_BLOCK 64 

#define BLOCK_H 4
#define BLOCK_W 8
#define BLOCK_HW BLOCK_H * BLOCK_W
#define CHANNEL_STRIDE 32
#define FEATURE_SIZE 192

#define DIM0(TENSOR) ((TENSOR).x)
#define DIM1(TENSOR) ((TENSOR).y)
#define DIM2(TENSOR) ((TENSOR).z)
#define DIM3(TENSOR) ((TENSOR).w)

#define DIM3_INDEX(TENSOR, xx, yy, zz, ww) ((TENSOR)[((xx) * (TENSOR##_stride.x)) + ((yy) * (TENSOR##_stride.y)) + ((zz) * (TENSOR##_stride.z)) + ((ww) * (TENSOR##_stride.w))])

// #ifdef __cplusplus
//     extern "C" {
// #endif

__forceinline__ __device__
bool within_bounds(int h, int w, int H, int W) {
  return h >= 0 && h < H && w >= 0 && w < W;
}

template <typename scalar_t>
__global__ void max_avg_forward_kernel (
  const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> img1_features_l0,
  const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> img2_features_lk,
  torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> max_avg_output_u,
  torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> max_avg_output_v)
{

  const int B = img1_features_l0.size(0); // batch size
  const int H1 = img1_features_l0.size(1); // ht of img1
  const int W1 = img1_features_l0.size(2); // wd of img1
  const int H2 = img2_features_lk.size(1); // ht of current lvl: ht/2**i
  const int W2 = img2_features_lk.size(2); // wd of current lvl: wd/2**i
  const int C = img1_features_l0.size(3); // image feature dimension

  const int b = blockIdx.x; // current batch

  // global starting x/y value for this block
  const int h0 = blockIdx.y * blockDim.x;   // block_i * BLOCK_H
  const int w0 = blockIdx.z * blockDim.y;   // block_j * BLOCK_W

  // local thread index in block
  const int tIdx = threadIdx.x * blockDim.y + threadIdx.y;

  // global current x/y value for this block
  const int hc0 = h0 + threadIdx.x;
  const int wc0 = w0 + threadIdx.y;

  if (within_bounds(hc0, wc0, H1, W1)) {

    auto g1_ref = img1_features_l0[b][hc0][wc0];

    // __shared__ scalar_t g1_block[BLOCK_HW][FEATURE_SIZE];
    // scalar_t * g1 = g1_block[tIdx];
    // for(int i = 0; i<C; ++i){
    //   g1[i] = g1_ref[i];
    // }

    for (int v = 0; v<W2; ++v){
      auto max_u_ref = max_avg_output_u[b][hc0][wc0][0];
      auto avg_u_ref = max_avg_output_u[b][hc0][wc0][1];
      auto max_v_ref = max_avg_output_v[b][hc0][wc0][0];
      auto avg_v_ref = max_avg_output_v[b][hc0][wc0][1];
      
      for(int u = 0; u<H2; ++u){

        auto g2_ref = img2_features_lk[b][u][v];

        scalar_t cval = 0;
        for(int i = 0; i<C; ++i){
          cval += g1_ref[i]*g2_ref[i];  
        }

        max_u_ref [u] = max(max_u_ref [u], cval);
        avg_u_ref [u] += cval / W2;

        max_v_ref [v] = max(max_v_ref [v], cval);
        avg_v_ref [v] += cval / H2;

      }
    }
  }

}

template <typename scalar_t>
__global__ void max_avg_forward_kernel_unoptimized (
  const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> img1_features_l0,
  const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> img2_features_lk,
  torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> max_avg_output_u,
  torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> max_avg_output_v)
{
  const int B = img1_features_l0.size(0); // batch size
  const int H1 = img1_features_l0.size(1); // ht of img1
  const int W1 = img1_features_l0.size(2); // wd of img1
  const int H2 = img2_features_lk.size(1); // ht of current lvl: ht/2**i
  const int W2 = img2_features_lk.size(2); // wd of current lvl: wd/2**i
  const int C = img1_features_l0.size(3); // image feature dimension

  const int b = blockIdx.x; // current batch

  // global starting x/y value for this block
  const int h0 = blockIdx.y * blockDim.x;   // block_i * BLOCK_H
  const int w0 = blockIdx.z * blockDim.y;   // block_j * BLOCK_W

  // global current x/y value for this block
  const int hc0 = h0 + threadIdx.x;
  const int wc0 = w0 + threadIdx.y;

  if (within_bounds(hc0, wc0, H1, W1)) {

    auto g1_ref = img1_features_l0[b][hc0][wc0];

    for (int v = 0; v<W2; ++v){
      auto max_u_ref = max_avg_output_u[b][hc0][wc0][0];
      auto avg_u_ref = max_avg_output_u[b][hc0][wc0][1];
      auto max_v_ref = max_avg_output_v[b][hc0][wc0][0];
      auto avg_v_ref = max_avg_output_v[b][hc0][wc0][1];
      
      for(int u = 0; u<H2; ++u){

        auto g2_ref = img2_features_lk[b][u][v];

        scalar_t cval = 0;
        for(int i = 0; i<C; ++i){
          cval += g1_ref[i]*g2_ref[i];  
        }

        max_u_ref [u] = max(max_u_ref [u], cval);
        avg_u_ref [u] += cval / W2;

        max_v_ref [v] = max(max_v_ref [v], cval);
        avg_v_ref [v] += cval / H2;

      }
    }
  }

}

template <typename scalar_t>
__global__ void compression_forward_kernel_unoptimized (
      const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> img1_features_l0,
      const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> img2_features_lk,
      const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> attention_weights_u,
      const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> attention_weights_v,
      torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> compression_output_u,
      torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> compression_output_v)
{
  const int B = img1_features_l0.size(0); // batch size
  const int H1 = img1_features_l0.size(1); // ht of img1
  const int W1 = img1_features_l0.size(2); // wd of img1
  const int H2 = img2_features_lk.size(1); // ht of current lvl: ht/2**i
  const int W2 = img2_features_lk.size(2); // wd of current lvl: wd/2**i
  const int C = img1_features_l0.size(3); // image feature dimension
  const int K = attention_weights_u.size(3); // K-2 self-compressed features

  const int b = blockIdx.x; // current batch

  // global starting x/y value for this block
  const int h0 = blockIdx.y * blockDim.x;   // block_i * BLOCK_H
  const int w0 = blockIdx.z * blockDim.y;   // block_j * BLOCK_W

  // global current x/y value for this block
  const int hc0 = h0 + threadIdx.x;
  const int wc0 = w0 + threadIdx.y;

  if (within_bounds(hc0, wc0, H1, W1)) {

    auto g1_ref = img1_features_l0[b][hc0][wc0];
    auto a_u_ref = attention_weights_u[b][hc0][wc0];
    auto a_v_ref = attention_weights_v[b][hc0][wc0];
    
    auto compressed_u_ref = compression_output_u[b][hc0][wc0];
    auto compressed_v_ref = compression_output_v[b][hc0][wc0];

    for (int v = 0; v<W2; ++v){
        
      for(int u = 0; u<H2; ++u){

        auto g2_ref = img2_features_lk[b][u][v];

        scalar_t cval = 0;
        for(int i = 0; i<C; ++i){
          cval += g1_ref[i]*g2_ref[i];  
        }
        for (int k = 0; k<K; ++k){
          
          compressed_u_ref [k][u] += cval*a_u_ref[k][v];
          compressed_v_ref [k][v] += cval*a_v_ref[k][u];
          
        }
      }
    }
  }

}

template <typename scalar_t>
__global__ void testkernel (
  const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> img1_features_l0,
  const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> img2_features_lk,
  torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> max_avg_output_u,
  torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> max_avg_output_v)
{

  const int B = img1_features_l0.size(0); // batch size
  const int H1 = img1_features_l0.size(1); // ht of img1
  const int W1 = img1_features_l0.size(2); // wd of img1
  const int H2 = img2_features_lk.size(1); // ht of current lvl: ht/2**i
  const int W2 = img2_features_lk.size(2); // wd of current lvl: wd/2**i
  const int C = img1_features_l0.size(3); // image feature dimension

  const int b = blockIdx.x; // current batch

  // global starting x/y value for this block
  const int h0 = blockIdx.y * blockDim.x;   // block_i * BLOCK_H
  const int w0 = blockIdx.z * blockDim.y;   // block_j * BLOCK_W

  // global current x/y value for this block
  const int hc0 = h0 + threadIdx.x;
  const int wc0 = w0 + threadIdx.y;
  
  auto g1_ref = img1_features_l0[b][hc0][wc0];

  for (int v = 0; v<W2; v+=1){
    auto max_u_ref = max_avg_output_u[b][hc0][wc0][0];
    auto avg_u_ref = max_avg_output_u[b][hc0][wc0][1];
    auto max_v_ref = max_avg_output_v[b][hc0][wc0][0];
    auto avg_v_ref = max_avg_output_v[b][hc0][wc0][1];
    
    for(int u = 0; u<H2; u+=1){

      auto g2_ref = img2_features_lk[b][u][v];

      max_u_ref [u] = max(max_u_ref [u], (1+b)*10000.0 + (1+hc0)*1000.0 + (1+wc0)*100.0 + (1+u)*10.0 + (1+v)*1.0);
      avg_u_ref [u] += 1;

      max_v_ref [v] = max(max_u_ref [v], (1+b)*10000.0 + (1+hc0)*1000.0 + (1+wc0)*100.0 + (1+u)*10.0 + (1+v)*1.0);
      avg_v_ref [v] += 1;

    }
  }

}

std::vector<torch::Tensor> max_avg_cuda_forward (
        at::Tensor img1_features_l0, 
        at::Tensor img2_features_lk){

  const int B = img1_features_l0.size(0); // batch size
  const int H1 = img1_features_l0.size(1); // ht of img1
  const int W1 = img1_features_l0.size(2); // wd of img1
  const int H2 = img2_features_lk.size(1); // ht of current lvl: ht/2**i
  const int W2 = img2_features_lk.size(2); // wd of current lvl: wd/2**i
  const int C = img1_features_l0.size(3); // image feature dimension

  // properties of the input
  auto opts = img1_features_l0.options();

  // allocate storage for result
  auto max_avg_output_u = torch::cat(
    {
      torch::full({B, H1, W1, 1, H2}, -INFINITY, opts),
      torch::zeros({B, H1, W1, 1, H2}, opts)
    }, 
    3);
  auto max_avg_output_v = torch::cat(
    {
      torch::full({B, H1, W1, 1, W2}, -INFINITY, opts),
      torch::zeros({B, H1, W1, 1, W2}, opts)
    }, 
    3);
  
  // number of blocks
  // shape: (batch, (ht+4-1)/4, (wd+8-1)/8) = (batch, ceil(ht/4), ceil(ht/8))
  const dim3 blocks(B, (H1+BLOCK_H-1)/BLOCK_H, (W1+BLOCK_W-1)/BLOCK_W);
  
  // number of threads per block
  // shape: (4,8)
  const dim3 threads(BLOCK_H, BLOCK_W);
  AT_DISPATCH_FLOATING_TYPES(img1_features_l0.type(), "max_avg_cuda_forward", ([&] {
    max_avg_forward_kernel_unoptimized <scalar_t> <<< blocks, threads >>> (
      img1_features_l0.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
      img2_features_lk.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
      max_avg_output_u.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
      max_avg_output_v.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>()
    );
  }));

  return {max_avg_output_u, max_avg_output_v};
}

std::vector<torch::Tensor> compression_cuda_forward (
        at::Tensor img1_features_l0, 
        at::Tensor img2_features_lk,
        at::Tensor attention_weights_u,
        at::Tensor attention_weights_v){

  const int B = img1_features_l0.size(0); // batch size
  const int H1 = img1_features_l0.size(1); // ht of img1
  const int W1 = img1_features_l0.size(2); // wd of img1
  const int H2 = img2_features_lk.size(1); // ht of current lvl: ht/2**i
  const int W2 = img2_features_lk.size(2); // wd of current lvl: wd/2**i
  const int C = img1_features_l0.size(3); // image feature dimension
  const int K = attention_weights_u.size(3); // K-2 self-compressed features

  // properties of the input
  auto opts = img1_features_l0.options();

  // allocate storage for result
  auto compression_output_u = torch::zeros({B, H1, W1, K, H2}, opts);
  auto compression_output_v = torch::zeros({B, H1, W1, K, W2}, opts);
  
  // number of blocks
  // shape: (batch, (ht+4-1)/4, (wd+8-1)/8) = (batch, ceil(ht/4), ceil(ht/8))
  const dim3 blocks(B, (H1+BLOCK_H-1)/BLOCK_H, (W1+BLOCK_W-1)/BLOCK_W);
  
  // number of threads per block
  // shape: (4,8)
  const dim3 threads(BLOCK_H, BLOCK_W);

  cudaDeviceSetLimit(cudaLimitMallocHeapSize, 128*1024*1024);

  AT_DISPATCH_FLOATING_TYPES(img1_features_l0.type(), "compression_cuda_forward", ([&] {
    compression_forward_kernel_unoptimized <scalar_t> <<< blocks, threads >>> (
      img1_features_l0.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
      img2_features_lk.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
      attention_weights_u.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
      attention_weights_v.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
      compression_output_u.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
      compression_output_v.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>()
    );
  }));

  return {compression_output_u, compression_output_v};
}




// #ifdef __cplusplus
//     }
// #endif
