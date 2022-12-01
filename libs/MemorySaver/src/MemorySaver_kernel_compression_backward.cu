#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdexcept>
#include <string>

// block dimension size constants
#define BLOCK_H 4
#define BLOCK_W 8
#define BLOCK_HW BLOCK_H * BLOCK_W

#define BLOCK_H_DIV8 8
#define BLOCK_W_DIV8 8
#define BLOCK_HW_DIV8 BLOCK_H_DIV8 * BLOCK_W_DIV8

#define CHANNEL_STRIDE 32
#define FEATURE_SIZE 256

// size of the feature split (cache features dimension size)
#define FEATURE_SPLIT_SIZE 32

// define number of self-adaptive compression to be 2
#define K_VAL 2

__forceinline__ __device__
bool within_bounds(int h, int w, int H, int W) {
  return h >= 0 && h < H && w >= 0 && w < W;
}

/**
 * @brief More time-efficient implementation of the self-adaptive compression backpropagation
 *  
 * does not work, ununsed
 * 
 */
template <typename scalar_t, int CONST_K>
__global__ void compression_backward_kernel_optimized_arch_indep (
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
  
  const int b = blockIdx.x; // current batch

  // global starting x/y value for this block
  const int h0 = blockIdx.y * blockDim.x;   // block_i * BLOCK_H
  const int w0 = blockIdx.z * blockDim.y;   // block_j * BLOCK_W

  // global current x/y value for this block
  const int hc0 = h0 + threadIdx.x;
  const int wc0 = w0 + threadIdx.y;

  const int uBlockLimit = (H2 + BLOCK_H - 1) / BLOCK_H;
  const int vBlockLimit = (W2 + BLOCK_W - 1) / BLOCK_W;

  const int fBlockLimit = (C + FEATURE_SPLIT_SIZE - 1) / FEATURE_SPLIT_SIZE;

  // 4*4096 Bytes = 16384 Bytes
  __shared__ scalar_t fcache1[BLOCK_H][BLOCK_W][FEATURE_SPLIT_SIZE];
  __shared__ scalar_t fcache2[BLOCK_H][BLOCK_W][FEATURE_SPLIT_SIZE];
  __shared__ scalar_t grad_fcache1[BLOCK_H][BLOCK_W][FEATURE_SPLIT_SIZE];
  __shared__ scalar_t grad_fcache2[BLOCK_H][BLOCK_W][FEATURE_SPLIT_SIZE];
  
  // CONST_K * (512 Bytes + 1024 Bytes) = CONST_K * 1536 Bytes
  // CONST_K = 2 : 3072 Bytes
  __shared__ scalar_t grad_compression_cache_u[BLOCK_H][BLOCK_W][CONST_K][BLOCK_H];
  __shared__ scalar_t grad_compression_cache_v[BLOCK_H][BLOCK_W][CONST_K][BLOCK_W];
 
  // CONST_K * (1024 Bytes + 512 Bytes + 1024 Bytes + 512 Bytes) = CONST_K * 3072 Bytes
  // CONST_K = 2 : 6144 Bytes
  __shared__ scalar_t attention_weights_cache_u[BLOCK_H][BLOCK_W][CONST_K][BLOCK_W];
  __shared__ scalar_t attention_weights_cache_v[BLOCK_H][BLOCK_W][CONST_K][BLOCK_H];
  __shared__ scalar_t grad_attention_cache_u[BLOCK_H][BLOCK_W][CONST_K][BLOCK_W];
  __shared__ scalar_t grad_attention_cache_v[BLOCK_H][BLOCK_W][CONST_K][BLOCK_H];

  scalar_t * fcache1_ptr = fcache1                                [threadIdx.x][threadIdx.y];
  scalar_t * fcache2_ptr = fcache2                                [threadIdx.x][threadIdx.y];
  scalar_t * grad_fcache1_ptr = grad_fcache1                      [threadIdx.x][threadIdx.y];
  scalar_t * grad_fcache2_ptr = grad_fcache2                      [threadIdx.x][threadIdx.y];
  
  scalar_t (* grad_compression_u_ptr) [BLOCK_H] = grad_compression_cache_u    [threadIdx.x][threadIdx.y];
  scalar_t (* grad_compression_v_ptr) [BLOCK_W] = grad_compression_cache_v    [threadIdx.x][threadIdx.y];

  scalar_t (* attention_weights_u_ptr) [BLOCK_W] = attention_weights_cache_u  [threadIdx.x][threadIdx.y];
  scalar_t (* attention_weights_v_ptr) [BLOCK_H] = attention_weights_cache_v  [threadIdx.x][threadIdx.y];
  scalar_t (* grad_attention_u_ptr) [BLOCK_W] = grad_attention_cache_u        [threadIdx.x][threadIdx.y];
  scalar_t (* grad_attention_v_ptr) [BLOCK_H] = grad_attention_cache_v        [threadIdx.x][threadIdx.y];

  auto g1_ref = img1_features_l0[b][hc0][wc0];
  auto grad_g1_ref = grad_img1_features_l0[b][hc0][wc0];
  auto grad_g2_ref = grad_img2_features_lk[b][hc0][wc0];
  
  auto grad_compressed_output_u_ref = grad_compressed_output_u[b][hc0][wc0];
  auto grad_compressed_output_v_ref = grad_compressed_output_v[b][hc0][wc0];

  auto attention_weights_u_ref = attention_weights_u[b][hc0][wc0];
  auto attention_weights_v_ref = attention_weights_u[b][hc0][wc0];
  auto grad_attention_u_ref = grad_attention_u[b][hc0][wc0];
  auto grad_attention_v_ref = grad_attention_v[b][hc0][wc0];

  const bool withinBoundsHc0Wc0 = within_bounds(hc0, wc0, H1, W1);

  for(int v_block = 0; v_block < vBlockLimit; ++v_block){
    
    const int v_offset = v_block * BLOCK_W;

    for(int k_inc = 0; k_inc < CONST_K; ++k_inc){
      scalar_t * k_attention_weights_u_ptr = attention_weights_u_ptr[k_inc];
      auto k_attention_weights_u_ref = attention_weights_u_ref[k_inc];
      for(int v_inc = 0; v_inc < BLOCK_W; ++v_inc){
        if(withinBoundsHc0Wc0 && v_offset + v_inc < W2){
          k_attention_weights_u_ptr[v_inc] = k_attention_weights_u_ref[v_offset + v_inc];
        }else{
          k_attention_weights_u_ptr[v_inc] = 0.0;
        }
      }
    }

    for(int k_inc = 0; k_inc < CONST_K; ++k_inc){
      scalar_t * k_grad_compression_v_ptr = grad_compression_v_ptr[k_inc];
      auto k_grad_compressed_output_v_ref = grad_compressed_output_v_ref[k_inc];
      for(int v_inc = 0; v_inc < BLOCK_W; ++v_inc){
        if(withinBoundsHc0Wc0 && v_offset + v_inc < W2){
          k_grad_compression_v_ptr[v_inc] = k_grad_compressed_output_v_ref[v_offset + v_inc];
        }else{
          k_grad_compression_v_ptr[v_inc] = 0.0;
        }
      }
    }

    // printf("check init 1\n");
    
    for(int u_block = 0; u_block < uBlockLimit; ++u_block){

      const int u_offset = u_block * BLOCK_H;

      for(int k_inc = 0; k_inc < CONST_K; ++k_inc){
        scalar_t * k_attention_weights_v_ptr = attention_weights_v_ptr[k_inc];
        auto k_attention_weights_v_ref = attention_weights_v_ref[k_inc];
        for(int u_inc = 0; u_inc < BLOCK_H; ++u_inc){
          if(withinBoundsHc0Wc0 && u_offset + u_inc < H2){
            k_attention_weights_v_ptr[u_inc] = k_attention_weights_v_ref[u_offset + u_inc];
          }else{
            k_attention_weights_v_ptr[u_inc] = 0.0;
          }
        }
      }

      for(int k_inc = 0; k_inc < CONST_K; ++k_inc){
        scalar_t * k_grad_compression_u_ptr = grad_compression_u_ptr[k_inc];
        auto k_grad_compressed_output_u_ref = grad_compressed_output_u_ref[k_inc];
        for(int u_inc = 0; u_inc < BLOCK_H; ++u_inc){
          if(withinBoundsHc0Wc0 && u_offset + u_inc < H2){
            k_grad_compression_u_ptr[u_inc] = k_grad_compressed_output_u_ref[u_offset + u_inc];
          }else{
            k_grad_compression_u_ptr[u_inc] = 0.0;
          }
        }
      }

       // printf("check init 2\n");

      auto g2_ref = img2_features_lk[b][u_offset + threadIdx.x][v_offset + threadIdx.y];
      
      for(int f_block = 0; f_block < fBlockLimit; ++f_block){
        
        const int f_offset = f_block * FEATURE_SPLIT_SIZE;

        // printf("check init sub 1\n");

        // reset values in temporary result cache (each thread resets its own)
        for(int f_inc = 0; f_inc < FEATURE_SPLIT_SIZE; ++f_inc){
          grad_fcache1_ptr[f_inc] = 0.0;
          grad_fcache2_ptr[f_inc] = 0.0;
        }

        // printf("check init sub 2\n");

        // reset values in result gradient attention chache (each thread resets its own)
        for(int k_inc = 0; k_inc < CONST_K; ++k_inc){
          scalar_t * k_grad_attention_u_ptr = grad_attention_u_ptr[k_inc];
          scalar_t * k_grad_attention_v_ptr = grad_attention_v_ptr[k_inc];
          for(int u_inc = 0; u_inc < BLOCK_H; ++u_inc){
            k_grad_attention_v_ptr[u_inc] = 0.0;
          }
          for(int v_inc = 0; v_inc < BLOCK_W; ++v_inc){
            k_grad_attention_u_ptr[v_inc] = 0.0;
          }
        }

        // printf("check init sub 3\n");

        // load image 1 features to its cache (fcache1)
        for(int f_inc = 0; f_inc < FEATURE_SPLIT_SIZE; ++f_inc){
          if(withinBoundsHc0Wc0 && f_offset + f_inc < C){
            fcache1_ptr[f_inc] = g1_ref[f_offset + f_inc];
          }else{
            fcache1_ptr[f_inc] = 0.0;
          }
        }

        // printf("check init sub 4\n");

        // load image 2 features to its cache (fcache2)
        for(int f_inc = 0; f_inc < FEATURE_SPLIT_SIZE; ++f_inc){
          if(within_bounds(u_offset+threadIdx.x, v_offset+threadIdx.y, H2, W2) && f_offset + f_inc < C){
            fcache2_ptr[f_inc] = g2_ref[f_offset + f_inc];
          }else{
            fcache2_ptr[f_inc] = 0.0;
          }
        }

        // printf("check init finish\n");

        __syncthreads();
        
        // (L/cuk)(cuk/fmap1)
        // (L/cvk)(cvk/fmap1)
        for(int k_inc = 0; k_inc < CONST_K; ++k_inc){
          for(int u_inc = 0; u_inc < BLOCK_H; ++u_inc){

            const scalar_t grad_cu_k = grad_compression_u_ptr[k_inc][u_inc];

            for(int v_inc = 0; v_inc < BLOCK_W; ++v_inc){

              const scalar_t grad_cv_k = grad_compression_v_ptr[k_inc][v_inc];

              const scalar_t factor = 
                  attention_weights_u_ptr[k_inc][v_inc] * grad_cu_k
                + attention_weights_v_ptr[k_inc][u_inc] * grad_cv_k;

              // printf("%i, %i, %i, %f, %f, %f, %f, %f\n",k_inc, u_inc, v_inc, attention_weights_u_ptr[k_inc][v_inc], attention_weights_v_ptr[k_inc][u_inc], grad_cu_k, grad_cv_k, factor);

              scalar_t * local_fcache2_ptr = fcache2[u_inc][v_inc];

              for(int f_inc = 0; f_inc < FEATURE_SPLIT_SIZE; ++f_inc){
                // printf("%i, %i, %i, %i, %f, %f, %f\n",k_inc, u_inc, v_inc, f_inc, local_fcache2_ptr[f_inc], factor, local_fcache2_ptr[f_inc] * factor);
                grad_fcache1_ptr[f_inc] += local_fcache2_ptr[f_inc] * factor;
              }
            }
          }
        }

        // (L/cuk)(cuk/fmap2)
        // (L/cvk)(cvk/fmap2)
        for(int k_inc = 0; k_inc < CONST_K; ++k_inc){
          for(int u_sync = 0; u_sync < BLOCK_H; ++u_sync){
            for(int v_sync = 0; v_sync < BLOCK_W; ++v_sync){

              const int u_inc = (threadIdx.x + u_sync) % BLOCK_H;
              const int v_inc = (threadIdx.y + v_sync) % BLOCK_W;
              
              const scalar_t grad_cu_k = grad_compression_u_ptr[k_inc][u_inc];
              const scalar_t grad_cv_k = grad_compression_u_ptr[k_inc][u_inc];
              const scalar_t factor = 
                  grad_cu_k * attention_weights_u_ptr[k_inc][v_inc]
                + grad_cv_k * attention_weights_v_ptr[k_inc][u_inc];

              scalar_t * local_grad_fcache2_ptr = grad_fcache2[u_inc][v_inc];

              // sync warps iterations progress to avoid parallel write conflict
              // only necessary if there are multiple warps in a block, i.e. BLOCK_HW > 32
              __syncthreads();

              for(int f_inc = 0; f_inc < FEATURE_SPLIT_SIZE; ++f_inc){
                // NOTE: parallel write conflict: Need to use atomic operation
                // scalar_t * inc_val = &grad_fcache2[u_inc][v_inc][f_inc];
                // atomicAdd(inc_val, fcache1[threadIdx.x][threadIdx.y][f_inc] * total_grad_factor);
                local_grad_fcache2_ptr[f_inc] += fcache1_ptr[f_inc] * factor;
              }

              // sync warps iterations progress to avoid parallel write conflict
              // only necessary if there are multiple warps in a block, i.e. BLOCK_HW > 32
              __syncthreads();
            }
          }
        }
          

        for(int u_inc = 0; u_inc < BLOCK_H; ++u_inc){
          for(int v_inc = 0; v_inc < BLOCK_W; ++v_inc){
            scalar_t * fcache2_uv_ptr = fcache2[u_inc][v_inc];
            scalar_t corr_val = 0;
            for(int f_inc = 0; f_inc < FEATURE_SPLIT_SIZE; ++f_inc){
              corr_val += fcache1_ptr[f_inc]*fcache2_uv_ptr[f_inc];
            }
            for(int k_inc = 0; k_inc < CONST_K; ++k_inc){

              const scalar_t grad_cu_k = grad_compression_u_ptr[k_inc][u_inc];
              const scalar_t grad_cv_k = grad_compression_v_ptr[k_inc][v_inc];

              // debug code
              // if(withinBoundsHc0Wc0 && u_offset + u_inc < H2 && v_offset + v_inc < W2){
              //   if(corr_val == 0 || grad_cu_k == 0 || grad_cv_k == 0){
              //     printf("hc0: %i, wc0: %i, uc0: %i, vc0: %i, %f, %f, %f\n", hc0, wc0, u_offset + u_inc, v_offset + v_inc, corr_val, grad_cu_k, grad_cv_k);
              //   }
              // }
            }
          }
        }

        // (L/cuk)(cuk/auk)
        // (L/cvk)(cvk/avk)
        for(int u_inc = 0; u_inc < BLOCK_H; ++u_inc){
          for(int v_inc = 0; v_inc < BLOCK_W; ++v_inc){
            scalar_t * fcache2_uv_ptr = fcache2[u_inc][v_inc];
            scalar_t corr_val = 0;
            for(int f_inc = 0; f_inc < FEATURE_SPLIT_SIZE; ++f_inc){
              corr_val += fcache1_ptr[f_inc]*fcache2_uv_ptr[f_inc];
            }
            for(int k_inc = 0; k_inc < CONST_K; ++k_inc){

              const scalar_t grad_cu_k = grad_compression_u_ptr[k_inc][u_inc];
              const scalar_t grad_cv_k = grad_compression_v_ptr[k_inc][v_inc];

              // debug code
              // if(withinBoundsHc0Wc0 && u_offset + u_inc < H2 && v_offset + v_inc < W2){
              //   // if(corr_val == 0 || grad_cu_k == 0 || grad_cv_k == 0){
              //     printf("hc0: %i, wc0: %i, uc0: %i, vc0: %i, %f, %f, %f\n", hc0, wc0, u_offset + u_inc, v_offset + v_inc, corr_val, grad_cu_k, grad_cv_k);
              //   // }
              // }

              grad_attention_u_ptr[k_inc][v_inc] += corr_val * grad_cu_k;
              grad_attention_v_ptr[k_inc][u_inc] += corr_val * grad_cv_k;
            }
          }
        }

         // printf("start writeback\n");

        // write back results to global memory (for this block of u,v,f values)

        // grad_fmap1
        for(int f_inc = 0; f_inc < FEATURE_SPLIT_SIZE; ++f_inc){
          if(withinBoundsHc0Wc0 && f_offset + f_inc < C){
            grad_g1_ref[f_offset + f_inc] += grad_fcache1_ptr[f_inc];
          }
        }

        // grad_attention_v and grad_attention_u
        for(int k_inc = 0; k_inc < CONST_K; ++k_inc){
          auto k_grad_attention_v_ref = grad_attention_v_ref[k_inc];
          for(int u_inc = 0; u_inc < BLOCK_H; ++u_inc){
            k_grad_attention_v_ref[u_offset + u_inc] += grad_attention_v_ptr[k_inc][u_inc];
            
            // debug
            // if(withinBoundsHc0Wc0 && u_offset + u_inc < H2){
              // if(k_grad_attention_v_ref[u_offset + u_inc] == 0 || grad_attention_v_ptr[k_inc][u_inc] == 0){
              //   printf("hc0: %i, wc0: %i, uc0: %i, %f, %f\n", hc0, wc0, u_offset + u_inc, k_grad_attention_v_ref[u_offset + u_inc], grad_attention_v_ptr[k_inc][u_inc]);
              // }
            // }

          }
          auto k_grad_attention_u_ref = grad_attention_u_ref[k_inc];
          for(int v_inc = 0; v_inc < BLOCK_W; ++v_inc){
            k_grad_attention_u_ref[v_offset + v_inc] += grad_attention_u_ptr[k_inc][v_inc];

            // debug
            // if(withinBoundsHc0Wc0 && v_offset + v_inc < W2){
              // if(k_grad_attention_u_ref[v_offset + v_inc] == 0 || grad_attention_u_ptr[k_inc][v_inc] == 0){
              //   printf("hc0: %i, wc0: %i, vc0: %i, %f, %f\n", hc0, wc0, v_offset + v_inc, k_grad_attention_u_ref[v_offset + v_inc], grad_attention_u_ptr[k_inc][v_inc]);
              // }
            // }

          }
        }

        // threads within the block need to be synced because the following section read-accesses 
        // the shared memory written by other threads
        __syncthreads();

        // write back results to global memory (for this block of u,v,f values)
        for(int f_inc = 0; f_inc < FEATURE_SPLIT_SIZE; ++f_inc){
          if(within_bounds(u_offset+threadIdx.x, v_offset+threadIdx.y, H2, W2) && f_offset + f_inc < C){
            scalar_t * inc_val = &grad_img2_features_lk[b][u_offset+threadIdx.x][v_offset+threadIdx.y][f_offset + f_inc];
            atomicAdd(inc_val, grad_fcache2_ptr[f_inc]);
          }
        }

        // sync threads before overwriting fcache2
        __syncthreads();
      }
    }
  }
}

/**
 * @brief Kernel Computation of the self-adaptive compression backward pass
 * 
 * @tparam scalar_t                 either float or double (type used in comptuation)
 * @param img1_features_l0          features of image 1
 * @param img2_features_lk          features of image 2
 * @param attention_weights_u       attention weights for C_u (one weight for every i,j,v)
 * @param attention_weights_v       attention weights for C_v (one weight for every i,j,u)
 * @param grad_compressed_output_u  incoming gradient of the C^(2:k)_u output
 * @param grad_compressed_output_v  incoming gradient of the C^(2:k)_v output
 * @param grad_img1_features_l0     resulting img1 features gradient
 * @param grad_img2_features_lk     resulting img2 features gradient
 * @param grad_attention_u          resulting attention weights u gradient
 * @param grad_attention_v          resulting attention weights v gradient
 * @return __global__ 
 */
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

/**
 * @brief Kernel-start function for the unoptimized self-adaptive compression backward pass
 * 
 * @param img1_features_l0              features of image 1
 * @param img2_features_lk              features of image 2
 * @param attention_weights_u           attention weights for C_u (one weight for every i,j,v) 
 * @param attention_weights_v           attention weights for C_v (one weight for every i,j,u)
 * @param grad_compressed_output_u      incoming gradient of the C^(2:k)_u output
 * @param grad_compressed_output_v      incoming gradient of the C^(2:k)_v output
 * @return std::vector<torch::Tensor>   results of backward pass:
 *                                      grad_img1_features_l0, grad_img2_features_lk, grad_attention_u, grad_attention_v
 */
std::vector<torch::Tensor> compression_cuda_backward_unoptimized (
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

/**
 * @brief Kernel-start function for the optimized self-adaptive compression backward pass
 * 
 * does not work, ununsed
 */
std::vector<torch::Tensor> compression_cuda_backward_optimized (
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

  if(K != K_VAL){
    printf("K(%i) is not equal to K_VAL(%i)\n", K, K_VAL);
  }
  
  compression_backward_kernel_optimized_arch_indep <float, K_VAL> <<< blocks, threads >>> (
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

  cudaError_t err = cudaThreadSynchronize();
  printf("Run kernel: %s\n", cudaGetErrorString(err));

  return {grad_img1_features_l0, grad_img2_features_lk, grad_attention_u, grad_attention_v};
}

/**
 * @brief Choice helper function (This is called from outside)
 * 
 * @param img1_features_l0              features of image 1
 * @param img2_features_lk              features of image 2
 * @param attention_weights_u           attention weights for C_u (one weight for every i,j,v) 
 * @param attention_weights_v           attention weights for C_v (one weight for every i,j,u)
 * @param grad_compressed_output_u      incoming gradient of the C^(2:k)_u output
 * @param grad_compressed_output_v      incoming gradient of the C^(2:k)_v output
 * @return std::vector<torch::Tensor>   results of backward pass:
 *                                      grad_img1_features_l0, grad_img2_features_lk, grad_attention_u, grad_attention_v
 */
std::vector<torch::Tensor> compression_cuda_backward (
        at::Tensor img1_features_l0, 
        at::Tensor img2_features_lk,
        at::Tensor attention_weights_u,
        at::Tensor attention_weights_v,
        at::Tensor grad_compressed_output_u, 
        at::Tensor grad_compressed_output_v){

  return compression_cuda_backward_unoptimized(
    img1_features_l0,
    img2_features_lk,
    attention_weights_u,
    attention_weights_v,
    grad_compressed_output_u,
    grad_compressed_output_v);
}