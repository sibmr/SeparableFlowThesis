//#include <torch/torch.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define BLOCK_H 4
#define BLOCK_W 8
#define BLOCK_HW BLOCK_H * BLOCK_W
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
__global__ void max_argmax_avg_backward_kernel_unoptimized (
      const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> img1_features_l0,
      const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> img2_features_lk,
      const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> max_avg_output_u,
      const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> max_avg_output_v,
      const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> argmax_output_u ,
      const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> argmax_output_v ,
      const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> grad_MaxAvg_u   ,
      const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> grad_MaxAvg_v   ,
      torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> grad_img1_features_l0,
      torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> grad_img2_features_lk)
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
    auto grad_g1_ref = grad_img1_features_l0[b][hc0][wc0];
    
    auto max_avg_output_u_ref = max_avg_output_u[b][hc0][wc0];
    auto max_avg_output_v_ref = max_avg_output_v[b][hc0][wc0];
    auto argmax_output_u_ref = argmax_output_u[b][hc0][wc0][0];
    auto argmax_output_v_ref = argmax_output_v[b][hc0][wc0][0];
    auto grad_MaxAvg_u_ref = grad_MaxAvg_u[b][hc0][wc0];
    auto grad_MaxAvg_v_ref = grad_MaxAvg_v[b][hc0][wc0];

    // (L/cumax)(cumax/fmap1)
    // (L/cuavg)(cuavg/fmap1)
    for(int u = 0; u<H2; ++u){
      
      const int vmax = argmax_output_u_ref[u];
      
      const scalar_t grad_cumax = grad_MaxAvg_u_ref[0][u];
      const scalar_t grad_cuavg = grad_MaxAvg_u_ref[1][u];
      const scalar_t grad_factor = grad_cuavg / W2;
      
      auto g2_avg_ref = img2_features_lk[b][u];
      auto g2_max_ref = g2_avg_ref[vmax];
      for(int f = 0; f<C; ++f){
        grad_g1_ref[f] += g2_max_ref[f] * grad_cumax;
      }
      for(int v = 0; v<W2; ++v){
        for(int f = 0; f<C; ++f){
          grad_g1_ref[f] += g2_avg_ref[v][f] * grad_factor;
        }
      }
    }

    // (L/cvmax)(cvmax/fmap1)
    // (L/cvavg)(cvavg/fmap1)
    for(int v = 0; v<W2; ++v){

      const int umax = argmax_output_v_ref[v];
      
      const scalar_t grad_cvmax = grad_MaxAvg_v_ref[0][v];
      const scalar_t grad_cvavg = grad_MaxAvg_v_ref[1][v];
      const scalar_t grad_factor = grad_cvavg / H2;
      
      auto g2_max_ref = img2_features_lk[b][umax][v];
      for(int f = 0; f<C; ++f){
        grad_g1_ref[f] += g2_max_ref[f] * grad_cvmax;
      }
      for(int u = 0; u<H2; ++u){
        for(int f = 0; f<C; ++f){
          grad_g1_ref[f] += img2_features_lk[b][u][v][f] * grad_factor;
        }
      }
    }

    // (L/cumax)(cumax/fmap2)
    // (L/cuavg)(cuavg/fmap2)
    for(int u = 0; u < H2; ++u){

      const int vmax = argmax_output_u_ref[u];
      
      const scalar_t grad_cumax = grad_MaxAvg_u_ref[0][u];
      const scalar_t grad_cuavg = grad_MaxAvg_u_ref[1][u];
      const scalar_t grad_factor = grad_cuavg/W2;

      auto grad_g2_avg_ref = grad_img2_features_lk[b][u];
      auto grad_g2_max_ref = grad_g2_avg_ref[vmax];

      for(int f = 0; f<C; ++f){
        // NOTE: parallel write conflict: Need to use atomic operation
        scalar_t * inc_val = &grad_g2_max_ref[f];
        atomicAdd(inc_val, g1_ref[f] * grad_cumax);
        // grad_g2_max_ref[f] += g1_ref[f] * grad_cumax;
      }
      for(int v = 0; v < W2; ++v){
        for(int f = 0; f<C; ++f){
          // NOTE: parallel write conflict: Need to use atomic operation
          scalar_t * inc_val = &grad_g2_avg_ref[v][f];
          atomicAdd(inc_val, g1_ref[f] * grad_factor);
          // grad_g2_avg_ref[v][f] += g1_ref[f] * grad_factor;
        }
      }
    }

    // (L/cvmax)(cvmax/fmap2)
    // (L/cvavg)(cvavg/fmap2)
    for(int v = 0; v < W2; ++v){

      const int umax = argmax_output_v_ref[v];
      
      const scalar_t grad_cvmax = grad_MaxAvg_v_ref[0][v];
      const scalar_t grad_cvavg = grad_MaxAvg_v_ref[1][v];
      const scalar_t grad_factor = grad_cvavg/H2;

      auto grad_g2_max_ref = grad_img2_features_lk[b][umax][v];

      for(int f = 0; f<C; ++f){
        // NOTE: parallel write conflict: Need to use atomic operation
        scalar_t * inc_val = &grad_g2_max_ref[f];
        atomicAdd(inc_val, g1_ref[f] * grad_cvmax);
        // grad_g2_max_ref[f] += g1_ref[f] * grad_cvmax;
      }
      for(int u = 0; u < W2; ++u){
        for(int f = 0; f<C; ++f){
          // NOTE: parallel write conflict: Need to use atomic operation
          scalar_t * inc_val = &grad_img2_features_lk[b][u][v][f];
          atomicAdd(inc_val, g1_ref[f] * grad_factor);
          // grad_img2_features_lk[b][u][v][f] += g1_ref[f] * grad_factor;
        }
      }
    }
  }
}

std::vector<torch::Tensor> max_argmax_avg_cuda_backward (
        at::Tensor img1_features_l0, 
        at::Tensor img2_features_lk,
        at::Tensor max_avg_output_u,
        at::Tensor max_avg_output_v,
        at::Tensor argmax_output_u,
        at::Tensor argmax_output_v,
        at::Tensor grad_MaxAvg_u,
        at::Tensor grad_MaxAvg_v){

  const int B = img1_features_l0.size(0); // batch size
  const int H1 = img1_features_l0.size(1); // ht of img1
  const int W1 = img1_features_l0.size(2); // wd of img1
  const int H2 = img2_features_lk.size(1); // ht of current lvl: ht/2**i
  const int W2 = img2_features_lk.size(2); // wd of current lvl: wd/2**i
  const int C = img1_features_l0.size(3); // image feature dimension

  // properties of the input
  auto opts = img1_features_l0.options();

  // allocate storage for result
  auto grad_img1_features_l0 = torch::zeros({B, H1, W1, C}, opts);
  auto grad_img2_features_lk = torch::zeros({B, H2, W2, C}, opts);
  
  // number of blocks
  // shape: (batch, (ht+4-1)/4, (wd+8-1)/8) = (batch, ceil(ht/4), ceil(ht/8))
  const dim3 blocks(B, (H1+BLOCK_H-1)/BLOCK_H, (W1+BLOCK_W-1)/BLOCK_W);
  
  // number of threads per block
  // shape: (4,8)
  const dim3 threads(BLOCK_H, BLOCK_W);
  
  max_argmax_avg_backward_kernel_unoptimized <float> <<< blocks, threads >>> (
    // inputs
    img1_features_l0  .packed_accessor32<float,4,torch::RestrictPtrTraits>(), 
    img2_features_lk  .packed_accessor32<float,4,torch::RestrictPtrTraits>(),
    max_avg_output_u  .packed_accessor32<float,5,torch::RestrictPtrTraits>(),
    max_avg_output_v  .packed_accessor32<float,5,torch::RestrictPtrTraits>(),
    argmax_output_u   .packed_accessor32<float,5,torch::RestrictPtrTraits>(),
    argmax_output_v   .packed_accessor32<float,5,torch::RestrictPtrTraits>(),
    grad_MaxAvg_u     .packed_accessor32<float,5,torch::RestrictPtrTraits>(),
    grad_MaxAvg_v     .packed_accessor32<float,5,torch::RestrictPtrTraits>(),
    // outputs
    grad_img1_features_l0     .packed_accessor32<float,4,torch::RestrictPtrTraits>(),
    grad_img2_features_lk     .packed_accessor32<float,4,torch::RestrictPtrTraits>()
  );

  cudaError_t err = cudaThreadSynchronize();
  printf("Run kernel: %s\n", cudaGetErrorString(err));

  return {grad_img1_features_l0, grad_img2_features_lk};
}

std::vector<torch::Tensor> compression_cuda_backward (
        at::Tensor img1_features_l0, 
        at::Tensor img2_features_lk){

  return {};
}