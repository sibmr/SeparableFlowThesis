//#include <torch/torch.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdexcept>
#include <string>

#define BLOCK_H 4
#define BLOCK_W 8
#define BLOCK_HW BLOCK_H * BLOCK_W

#define BLOCK_H_DIV8 8
#define BLOCK_W_DIV8 8
#define BLOCK_HW_DIV8 BLOCK_H_DIV8 * BLOCK_W_DIV8

#define CHANNEL_STRIDE 32
#define FEATURE_SIZE 256
#define FEATURE_SPLIT_SIZE 32

// define number of self-adaptive compression to be 2
#define K_VAL 2

__forceinline__ __device__
bool within_bounds(int h, int w, int H, int W) {
  return h >= 0 && h < H && w >= 0 && w < W;
}

template <typename scalar_t>
__global__ void compression_backward_kernel_unoptimized (
      const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> img1_features_l0        ,
      const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> img2_features_lk        ,
      const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> attention_weights_u     ,
      const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> attention_weights_v     ,
      const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> grad_compressed_output_u,
      const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> grad_compressed_output_v,
      torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> grad_img1_features_l0,
      torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> grad_img2_features_lk,
      torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> grad_attention_u     ,
      torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> grad_attention_v     )
{
  const int B = img1_features_l0.size(0); // batch size
  const int H1 = img1_features_l0.size(1); // ht of img1
  const int W1 = img1_features_l0.size(2); // wd of img1
  const int H2 = img2_features_lk.size(1); // ht of current lvl: ht/2**i
  const int W2 = img2_features_lk.size(2); // wd of current lvl: wd/2**i
  const int C = img1_features_l0.size(3); // image feature dimension
  const int K = attention_weights_u.size(3); // image feature dimension
  
  const int b = blockIdx.x; // current batch

  // global starting x/y value for this block
  const int h0 = blockIdx.y * blockDim.x;   // block_i * BLOCK_H
  const int w0 = blockIdx.z * blockDim.y;   // block_j * BLOCK_W

  // global current x/y value for this block
  const int hc0 = h0 + threadIdx.x;
  const int wc0 = w0 + threadIdx.y;

  if (within_bounds(hc0, wc0, H1, W1)) {

    for(int k = 0; k < K; ++k){
      auto g1_ref = img1_features_l0[b][hc0][wc0];
      auto grad_g1_ref = grad_img1_features_l0[b][hc0][wc0];
      
      auto attention_weights_u_ref = attention_weights_u[b][hc0][wc0][k];
      auto attention_weights_v_ref = attention_weights_v[b][hc0][wc0][k];

      auto grad_attention_u_ref = grad_attention_u[b][hc0][wc0][k];
      auto grad_attention_v_ref = grad_attention_v[b][hc0][wc0][k];

      auto grad_compressed_output_u_ref = grad_compressed_output_u[b][hc0][wc0][k];
      auto grad_compressed_output_v_ref = grad_compressed_output_v[b][hc0][wc0][k];

      // (L -> cuk)(cuk -> f1) + (L -> cvk)(cvk -> f1)
      for(int u = 0; u < H2; ++u){
        for(int v = 0; v < W2; ++v){
          auto g2_ref = img2_features_lk[b][u][v];
          const scalar_t factor_u = attention_weights_u_ref[v] * grad_compressed_output_u_ref[u];
          const scalar_t factor_v = attention_weights_v_ref[u] * grad_compressed_output_v_ref[v];
          const scalar_t factor_sum = factor_u + factor_v;
          for(int f = 0; f < C; ++f){
            grad_g1_ref[f] += g2_ref[f] * factor_sum;
          }
        }
      }

      // (L -> cuk)(cuk -> f2) + (L -> cvk)(cvk -> f2)
      for(int u = 0; u < H2; ++u){
        for(int v = 0; v < W2; ++v){
          auto grad_g2_ref = grad_img2_features_lk[b][u][v];
          const scalar_t factor_u = attention_weights_u_ref[v] * grad_compressed_output_u_ref[u];
          const scalar_t factor_v = attention_weights_v_ref[u] * grad_compressed_output_v_ref[v];
          const scalar_t factor_sum = factor_u + factor_v;
          for(int f = 0; f < C; ++f){
            scalar_t * inc_val = &grad_g2_ref[f];
            atomicAdd(inc_val, g1_ref[f] * factor_sum);
          }
        }
      }

      // (L -> cuk)(cuk -> a_u)
      // (L -> cvk)(cvk -> a_v)
      for(int u = 0; u < H2; ++u){
        for(int v = 0; v < W2; ++v){
          auto g2_ref = img2_features_lk[b][u][v];
          scalar_t cval = 0;
          for(int f = 0; f < C; ++f){
            cval += g1_ref[f]*g2_ref[f];
          }
          grad_attention_u_ref[v] += cval*grad_compressed_output_u_ref[u];
          grad_attention_v_ref[u] += cval*grad_compressed_output_v_ref[v];
        }
      }


    }
  }
    
}


std::vector<torch::Tensor> compression_cuda_backward (
        at::Tensor img1_features_l0, 
        at::Tensor img2_features_lk,
        at::Tensor attention_weights_u,
        at::Tensor attention_weights_v,
        at::Tensor grad_compressed_output_u, 
        at::Tensor grad_compressed_output_v){
  
  const int B = img1_features_l0.size(0); // batch size
  const int H1 = img1_features_l0.size(1); // ht of img1
  const int W1 = img1_features_l0.size(2); // wd of img1
  const int H2 = img2_features_lk.size(1); // ht of current lvl: ht/2**i
  const int W2 = img2_features_lk.size(2); // wd of current lvl: wd/2**i
  const int C = img1_features_l0.size(3); // image feature dimension
  const int K = attention_weights_u.size(3); // image feature dimension

  // properties of the input
  auto opts = img1_features_l0.options();

  // allocate storage for result
  auto grad_img1_features_l0 = torch::zeros({B, H1, W1, C}, opts);
  auto grad_img2_features_lk = torch::zeros({B, H2, W2, C}, opts);
  auto grad_attention_u = torch::zeros({B, H1, W1, K, W2}, opts);
  auto grad_attention_v = torch::zeros({B, H1, W1, K, H2}, opts);
  
  // number of blocks
  // shape: (batch, (ht+4-1)/4, (wd+8-1)/8) = (batch, ceil(ht/4), ceil(ht/8))
  const dim3 blocks(B, (H1+BLOCK_H-1)/BLOCK_H, (W1+BLOCK_W-1)/BLOCK_W);
  
  // number of threads per block
  // shape: (4,8)
  const dim3 threads(BLOCK_H, BLOCK_W);
  
  compression_backward_kernel_unoptimized <float> <<< blocks, threads >>> (
    // inputs
    img1_features_l0          .packed_accessor32<float,4,torch::RestrictPtrTraits>(), 
    img2_features_lk          .packed_accessor32<float,4,torch::RestrictPtrTraits>(),
    attention_weights_u       .packed_accessor32<float,5,torch::RestrictPtrTraits>(),
    attention_weights_v       .packed_accessor32<float,5,torch::RestrictPtrTraits>(),
    grad_compressed_output_u  .packed_accessor32<float,5,torch::RestrictPtrTraits>(),
    grad_compressed_output_v  .packed_accessor32<float,5,torch::RestrictPtrTraits>(),
    // outputs
    grad_img1_features_l0     .packed_accessor32<float,4,torch::RestrictPtrTraits>(),
    grad_img2_features_lk     .packed_accessor32<float,4,torch::RestrictPtrTraits>(),
    grad_attention_u          .packed_accessor32<float,5,torch::RestrictPtrTraits>(),
    grad_attention_v          .packed_accessor32<float,5,torch::RestrictPtrTraits>()
  );

  // cudaError_t err = cudaThreadSynchronize();
  // printf("Run kernel: %s\n", cudaGetErrorString(err));

  return {grad_img1_features_l0, grad_img2_features_lk, grad_attention_u, grad_attention_v};
}