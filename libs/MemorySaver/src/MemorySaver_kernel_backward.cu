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
__global__ void max_argmax_avg_backward_kernel_optimized_arch_indep (
      const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> img1_features_l0,
      const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> img2_features_lk,
      const torch::PackedTensorAccessor32<int32_t,5,torch::RestrictPtrTraits> argmax_output_u ,
      const torch::PackedTensorAccessor32<int32_t,5,torch::RestrictPtrTraits> argmax_output_v ,
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

  const int uBlockLimit = (H2 + BLOCK_H - 1) / BLOCK_H;
  const int vBlockLimit = (W2 + BLOCK_W - 1) / BLOCK_W;

  const int fBlockLimit = (C + FEATURE_SPLIT_SIZE - 1) / FEATURE_SPLIT_SIZE;

  // 2*4096 Bytes
  __shared__ scalar_t fcache1[BLOCK_H][BLOCK_W][FEATURE_SPLIT_SIZE];
  __shared__ scalar_t fcache2[BLOCK_H][BLOCK_W][FEATURE_SPLIT_SIZE];
  __shared__ scalar_t grad_fcache1[BLOCK_H][BLOCK_W][FEATURE_SPLIT_SIZE];
  __shared__ scalar_t grad_fcache2[BLOCK_H][BLOCK_W][FEATURE_SPLIT_SIZE];
  
  // 1536 Bytes + 3072 Bytes = 4608 Bytes
  __shared__ int32_t argmax_cache_u[BLOCK_H][BLOCK_W][BLOCK_H];
  __shared__ int32_t argmax_cache_v[BLOCK_H][BLOCK_W][BLOCK_W];
  __shared__ scalar_t grad_maxavg_cache_u[BLOCK_H][BLOCK_W][2][BLOCK_H];
  __shared__ scalar_t grad_maxavg_cache_v[BLOCK_H][BLOCK_W][2][BLOCK_W];

  scalar_t * fcache1_ptr = fcache1                [threadIdx.x][threadIdx.y];
  scalar_t * fcache2_ptr = fcache2                [threadIdx.x][threadIdx.y];
  scalar_t * grad_fcache1_ptr = grad_fcache1      [threadIdx.x][threadIdx.y];
  scalar_t * grad_fcache2_ptr = grad_fcache2      [threadIdx.x][threadIdx.y];
  int32_t  * argmax_u_ptr = argmax_cache_u        [threadIdx.x][threadIdx.y];
  int32_t  * argmax_v_ptr = argmax_cache_v        [threadIdx.x][threadIdx.y];
  scalar_t * grad_max_u_ptr = grad_maxavg_cache_u [threadIdx.x][threadIdx.y][0];
  scalar_t * grad_max_v_ptr = grad_maxavg_cache_v [threadIdx.x][threadIdx.y][0];
  scalar_t * grad_avg_u_ptr = grad_maxavg_cache_u [threadIdx.x][threadIdx.y][1];
  scalar_t * grad_avg_v_ptr = grad_maxavg_cache_v [threadIdx.x][threadIdx.y][1];

  auto g1_ref = img1_features_l0[b][hc0][wc0];

  auto grad_g1_ref = grad_img1_features_l0[b][hc0][wc0];
  auto grad_g2_ref = grad_img2_features_lk[b][hc0][wc0];
  
  auto argmax_output_u_ref = argmax_output_u[b][hc0][wc0][0];
  auto argmax_output_v_ref = argmax_output_v[b][hc0][wc0][0];
  auto grad_MaxAvg_u_ref = grad_MaxAvg_u[b][hc0][wc0];
  auto grad_MaxAvg_v_ref = grad_MaxAvg_v[b][hc0][wc0];

  const bool withinBoundsHc0Wc0 = within_bounds(hc0, wc0, H1, W1);

  for(int v_block = 0; v_block < vBlockLimit; ++v_block){
    
    const int v_offset = v_block * BLOCK_W;

    for(int v_inc = 0; v_inc < BLOCK_W; ++v_inc){
      if(withinBoundsHc0Wc0 && v_offset + v_inc < W2){
        argmax_v_ptr[v_inc] = argmax_output_v_ref[v_offset + v_inc];
      }else{
        argmax_v_ptr[v_inc] = -1;
      }
    }
    for(int v_inc = 0; v_inc < BLOCK_W; ++v_inc){
      if(withinBoundsHc0Wc0 && v_offset + v_inc < W2){
        grad_max_v_ptr[v_inc] = grad_MaxAvg_v_ref[0][v_offset + v_inc];
      }else{
        grad_max_v_ptr[v_inc] = 0.0;
      }
    }
    for(int v_inc = 0; v_inc < BLOCK_W; ++v_inc){
      if(withinBoundsHc0Wc0 && v_offset + v_inc < W2){
        grad_avg_v_ptr[v_inc] = grad_MaxAvg_v_ref[1][v_offset + v_inc];
      }else{
        grad_avg_v_ptr[v_inc] = 0.0;
      }
    }
    
    for(int u_block = 0; u_block < uBlockLimit; ++u_block){

      const int u_offset = u_block * BLOCK_H;

      for(int u_inc = 0; u_inc < BLOCK_H; ++u_inc){
        if(withinBoundsHc0Wc0 && u_offset + u_inc < H2){
          argmax_u_ptr[u_inc] = argmax_output_u_ref[u_offset + u_inc];
        }else{
          argmax_u_ptr[u_inc] = -1;
        }
      }
      for(int u_inc = 0; u_inc < BLOCK_H; ++u_inc){
        if(withinBoundsHc0Wc0 && u_offset + u_inc < H2){
          grad_max_u_ptr[u_inc] = grad_MaxAvg_u_ref[0][u_offset + u_inc];
        }else{
          grad_max_u_ptr[u_inc] = 0.0;
        }
      }
      for(int u_inc = 0; u_inc < BLOCK_H; ++u_inc){
        if(withinBoundsHc0Wc0 && u_offset + u_inc < H2){
          grad_avg_u_ptr[u_inc] = grad_MaxAvg_u_ref[1][u_offset + u_inc];
        }else{
          grad_avg_u_ptr[u_inc] = 0.0;
        }
      }

      auto g2_ref = img2_features_lk[b][u_offset + threadIdx.x][v_offset + threadIdx.y];
      for(int f_block = 0; f_block < fBlockLimit; ++f_block){
        
        const int f_offset = f_block * FEATURE_SPLIT_SIZE;

        // reset values in temporary result cache (each thread resets its own)
        for(int f_inc = 0; f_inc < FEATURE_SPLIT_SIZE; ++f_inc){
          grad_fcache1_ptr[f_inc] = 0.0;
          grad_fcache2_ptr[f_inc] = 0.0;
        }

        // load image 1 features to its cache (fcache1)
        for(int f_inc = 0; f_inc < FEATURE_SPLIT_SIZE; ++f_inc){
          if(withinBoundsHc0Wc0 && f_offset + f_inc < C){
            fcache1_ptr[f_inc] = g1_ref[f_offset + f_inc];
          }else{
            fcache1_ptr[f_inc] = 0.0;
          }
        }

        // load image 2 features to its cache (fcache2)
        for(int f_inc = 0; f_inc < FEATURE_SPLIT_SIZE; ++f_inc){
          if(within_bounds(u_offset+threadIdx.x, v_offset+threadIdx.y, H2, W2) && f_offset + f_inc < C){
            fcache2_ptr[f_inc] = g2_ref[f_offset + f_inc];
          }else{
            fcache2_ptr[f_inc] = 0.0;
          }
        }

        __syncthreads();
        
        // (L/cumax)(cumax/fmap1)
        // (L/cuavg)(cuavg/fmap1)
        for(int u_inc = 0; u_inc < BLOCK_H; ++u_inc){
          
          int vmax = argmax_u_ptr[u_inc];
          const bool vmax_valid = (vmax >= 0);
          // if vmax not valid, set out-of-bounds index -1 to 0 (always in bounds)
          vmax = vmax * vmax_valid;
          
          // if vmax not valid, ignore out-of-bounds index by setting contribution to 0
          const scalar_t grad_cumax = grad_max_u_ptr[u_inc] * vmax_valid;
          const scalar_t grad_cuavg = grad_avg_u_ptr[u_inc] / W2;

          scalar_t (* g2_avg_ptr) [FEATURE_SPLIT_SIZE] = fcache2[u_inc];
          for(int v_inc = 0; v_inc < BLOCK_W; ++v_inc){
            if(vmax == v_offset + v_inc){
              for(int f_inc = 0; f_inc < FEATURE_SPLIT_SIZE; ++f_inc){
                grad_fcache1_ptr[f_inc] += g2_avg_ptr[v_inc][f_inc] * grad_cumax;
              }
            }
          }
          for(int v_inc = 0; v_inc < BLOCK_W; ++v_inc){
            for(int f_inc = 0; f_inc < FEATURE_SPLIT_SIZE; ++f_inc){
              grad_fcache1_ptr[f_inc] += g2_avg_ptr[v_inc][f_inc] * grad_cuavg;
            }
          }
        }

        // (L/cvmax)(cvmax/fmap1)
        // (L/cvavg)(cvavg/fmap1)
        for(int v_inc = 0; v_inc < BLOCK_W; ++v_inc){
          
          int umax = argmax_v_ptr[v_inc];
          const bool umax_valid = (umax >= 0);
          // if umax not valid, set out-of-bounds index -1 to 0 (always in bounds)
          umax = umax * umax_valid;

          // if umax not valid, ignore out-of-bounds index by setting contribution to 0
          const scalar_t grad_cvmax = grad_max_v_ptr[v_inc] * umax_valid;
          const scalar_t grad_cvavg = grad_avg_v_ptr[v_inc] / H2;

          for(int u_inc = 0; u_inc < BLOCK_H; ++u_inc){
            if(umax == u_offset + u_inc){
              for(int f_inc = 0; f_inc < FEATURE_SPLIT_SIZE; ++f_inc){
                grad_fcache1_ptr[f_inc] += fcache2[u_inc][v_inc][f_inc] * grad_cvmax;
              }
            }
          }
          for(int u_inc = 0; u_inc < BLOCK_H; ++u_inc){
            for(int f_inc = 0; f_inc < FEATURE_SPLIT_SIZE; ++f_inc){
              grad_fcache1_ptr[f_inc] += fcache2[u_inc][v_inc][f_inc] * grad_cvavg;
            }
          }
        }

        for(int u_sync = 0; u_sync < BLOCK_H; ++u_sync){
          for(int v_sync = 0; v_sync < BLOCK_W; ++v_sync){

            const int u_inc = (threadIdx.x + u_sync) % BLOCK_H;
            const int v_inc = (threadIdx.y + v_sync) % BLOCK_W;

            const int vmax = argmax_u_ptr[u_inc];
            const bool vmax_valid = (vmax >= 0);
            
            // if vmax not valid, ignore out-of-bounds index by setting contribution to 0
            const scalar_t grad_cumax = grad_max_u_ptr[u_inc] * vmax_valid;
            const scalar_t grad_cuavg = grad_avg_u_ptr[u_inc] / W2;

            const int umax = argmax_v_ptr[v_inc];
            const bool umax_valid = (umax >= 0);
            
            // if vmax not valid, ignore out-of-bounds index by setting contribution to 0
            const scalar_t grad_cvmax = grad_max_v_ptr[v_inc] * umax_valid;
            const scalar_t grad_cvavg = grad_avg_v_ptr[v_inc] / H2;

            const scalar_t total_grad_factor = grad_cumax * (vmax==(v_offset+v_inc)) 
              + grad_cuavg + grad_cvmax * (umax==(u_offset+u_inc)) + grad_cvavg;

            // sync warps iterations progress to avoid parallel write conflict
            // only necessary if there are multiple warps in a block, i.e. BLOCK_HW > 32
            __syncthreads();

            for(int f_inc = 0; f_inc < FEATURE_SPLIT_SIZE; ++f_inc){
              // NOTE: parallel write conflict: Need to use atomic operation
              // scalar_t * inc_val = &grad_fcache2[u_inc][v_inc][f_inc];
              // atomicAdd(inc_val, fcache1[threadIdx.x][threadIdx.y][f_inc] * total_grad_factor);
              grad_fcache2[u_inc][v_inc][f_inc] += fcache1[threadIdx.x][threadIdx.y][f_inc] * total_grad_factor;
            }

            // sync warps iterations progress to avoid parallel write conflict
            // only necessary if there are multiple warps in a block, i.e. BLOCK_HW > 32
            __syncthreads();
          }
        }

        // write back results to global memory (for this block of u,v,f values)
        for(int f_inc = 0; f_inc < FEATURE_SPLIT_SIZE; ++f_inc){
          if(withinBoundsHc0Wc0 && f_offset + f_inc < C){
            grad_g1_ref[f_offset + f_inc] += grad_fcache1_ptr[f_inc];
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

template <typename scalar_t>
__global__ void max_argmax_avg_fmap1_backward_kernel_optimized_arch_indep (
      const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> img1_features_l0,
      const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> img2_features_lk,
      const torch::PackedTensorAccessor32<int32_t,5,torch::RestrictPtrTraits> argmax_output_u ,
      const torch::PackedTensorAccessor32<int32_t,5,torch::RestrictPtrTraits> argmax_output_v ,
      const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> grad_MaxAvg_u   ,
      const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> grad_MaxAvg_v   ,
      torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> grad_img1_features_l0)
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

  // 4096 Bytes
  __shared__ scalar_t fcache2[BLOCK_H][BLOCK_W][FEATURE_SPLIT_SIZE];
  __shared__ scalar_t grad_fcache1[BLOCK_H][BLOCK_W][FEATURE_SPLIT_SIZE];
  
  // 1536 Bytes + 3072 Bytes = 4608 Bytes
  __shared__ int32_t argmax_cache_u[BLOCK_H][BLOCK_W][BLOCK_H];
  __shared__ int32_t argmax_cache_v[BLOCK_H][BLOCK_W][BLOCK_W];
  __shared__ scalar_t grad_maxavg_cache_u[BLOCK_H][BLOCK_W][2][BLOCK_H];
  __shared__ scalar_t grad_maxavg_cache_v[BLOCK_H][BLOCK_W][2][BLOCK_W];

  scalar_t * fcache2_ptr = fcache2                [threadIdx.x][threadIdx.y];
  scalar_t * grad_fcache1_ptr = grad_fcache1      [threadIdx.x][threadIdx.y];
  int32_t  * argmax_u_ptr = argmax_cache_u        [threadIdx.x][threadIdx.y];
  int32_t  * argmax_v_ptr = argmax_cache_v        [threadIdx.x][threadIdx.y];
  scalar_t * grad_max_u_ptr = grad_maxavg_cache_u [threadIdx.x][threadIdx.y][0];
  scalar_t * grad_max_v_ptr = grad_maxavg_cache_v [threadIdx.x][threadIdx.y][0];
  scalar_t * grad_avg_u_ptr = grad_maxavg_cache_u [threadIdx.x][threadIdx.y][1];
  scalar_t * grad_avg_v_ptr = grad_maxavg_cache_v [threadIdx.x][threadIdx.y][1];

  auto grad_g1_ref = grad_img1_features_l0[b][hc0][wc0];
  
  auto argmax_output_u_ref = argmax_output_u[b][hc0][wc0][0];
  auto argmax_output_v_ref = argmax_output_v[b][hc0][wc0][0];
  auto grad_MaxAvg_u_ref = grad_MaxAvg_u[b][hc0][wc0];
  auto grad_MaxAvg_v_ref = grad_MaxAvg_v[b][hc0][wc0];

  const bool withinBoundsHc0Wc0 = within_bounds(hc0, wc0, H1, W1);

  for(int v_block = 0; v_block < vBlockLimit; ++v_block){
    
    const int v_offset = v_block * BLOCK_W;

    for(int v_inc = 0; v_inc < BLOCK_W; ++v_inc){
      if(withinBoundsHc0Wc0 && v_offset + v_inc < W2){
        argmax_v_ptr[v_inc] = argmax_output_v_ref[v_offset + v_inc];
      }else{
        argmax_v_ptr[v_inc] = -1;
      }
    }
    for(int v_inc = 0; v_inc < BLOCK_W; ++v_inc){
      if(withinBoundsHc0Wc0 && v_offset + v_inc < W2){
        grad_max_v_ptr[v_inc] = grad_MaxAvg_v_ref[0][v_offset + v_inc];
      }else{
        grad_max_v_ptr[v_inc] = 0.0;
      }
    }
    for(int v_inc = 0; v_inc < BLOCK_W; ++v_inc){
      if(withinBoundsHc0Wc0 && v_offset + v_inc < W2){
        grad_avg_v_ptr[v_inc] = grad_MaxAvg_v_ref[1][v_offset + v_inc];
      }else{
        grad_avg_v_ptr[v_inc] = 0.0;
      }
    }
    
    for(int u_block = 0; u_block < uBlockLimit; ++u_block){

      const int u_offset = u_block * BLOCK_H;

      for(int u_inc = 0; u_inc < BLOCK_H; ++u_inc){
        if(withinBoundsHc0Wc0 && u_offset + u_inc < H2){
          argmax_u_ptr[u_inc] = argmax_output_u_ref[u_offset + u_inc];
        }else{
          argmax_u_ptr[u_inc] = -1;
        }
      }
      for(int u_inc = 0; u_inc < BLOCK_H; ++u_inc){
        if(withinBoundsHc0Wc0 && u_offset + u_inc < H2){
          grad_max_u_ptr[u_inc] = grad_MaxAvg_u_ref[0][u_offset + u_inc];
        }else{
          grad_max_u_ptr[u_inc] = 0.0;
        }
      }
      for(int u_inc = 0; u_inc < BLOCK_H; ++u_inc){
        if(withinBoundsHc0Wc0 && u_offset + u_inc < H2){
          grad_avg_u_ptr[u_inc] = grad_MaxAvg_u_ref[1][u_offset + u_inc];
        }else{
          grad_avg_u_ptr[u_inc] = 0.0;
        }
      }

      auto g2_ref = img2_features_lk[b][u_offset + threadIdx.x][v_offset + threadIdx.y];
      for(int f_block = 0; f_block < fBlockLimit; ++f_block){
        
        const int f_offset = f_block * FEATURE_SPLIT_SIZE;

        // reset values in temporary result cache (each thread resets its own)
        for(int f_inc = 0; f_inc < FEATURE_SPLIT_SIZE; ++f_inc){
          grad_fcache1_ptr[f_inc] = 0.0;
        }

        // load image 2 features to its cache (fcache2)
        for(int f_inc = 0; f_inc < FEATURE_SPLIT_SIZE; ++f_inc){
          if(within_bounds(u_offset+threadIdx.x, v_offset+threadIdx.y, H2, W2) && f_offset + f_inc < C){
            fcache2_ptr[f_inc] = g2_ref[f_offset + f_inc];
          }else{
            fcache2_ptr[f_inc] = 0.0;
          }
        }

        __syncthreads();
        
        // (L/cumax)(cumax/fmap1)
        // (L/cuavg)(cuavg/fmap1)
        for(int u_inc = 0; u_inc < BLOCK_H; ++u_inc){
          
          int vmax = argmax_u_ptr[u_inc];
          const bool vmax_valid = (vmax >= 0);
          // if vmax not valid, set out-of-bounds index -1 to 0 (always in bounds)
          vmax = vmax * vmax_valid;
          
          // if vmax not valid, ignore out-of-bounds index by setting contribution to 0
          const scalar_t grad_cumax = grad_max_u_ptr[u_inc] * vmax_valid;
          const scalar_t grad_cuavg = grad_avg_u_ptr[u_inc] / W2;

          scalar_t (* g2_avg_ptr) [FEATURE_SPLIT_SIZE] = fcache2[u_inc];
          for(int v_inc = 0; v_inc < BLOCK_W; ++v_inc){
            if(vmax == v_offset + v_inc){
              for(int f_inc = 0; f_inc < FEATURE_SPLIT_SIZE; ++f_inc){
                grad_fcache1_ptr[f_inc] += g2_avg_ptr[v_inc][f_inc] * grad_cumax;
              }
            }
          }
          for(int v_inc = 0; v_inc < BLOCK_W; ++v_inc){
            for(int f_inc = 0; f_inc < FEATURE_SPLIT_SIZE; ++f_inc){
              grad_fcache1_ptr[f_inc] += g2_avg_ptr[v_inc][f_inc] * grad_cuavg;
            }
          }
        }

        // (L/cvmax)(cvmax/fmap1)
        // (L/cvavg)(cvavg/fmap1)
        for(int v_inc = 0; v_inc < BLOCK_W; ++v_inc){
          
          int umax = argmax_v_ptr[v_inc];
          const bool umax_valid = (umax >= 0);
          // if umax not valid, set out-of-bounds index -1 to 0 (always in bounds)
          umax = umax * umax_valid;

          // if umax not valid, ignore out-of-bounds index by setting contribution to 0
          const scalar_t grad_cvmax = grad_max_v_ptr[v_inc] * umax_valid;
          const scalar_t grad_cvavg = grad_avg_v_ptr[v_inc] / H2;

          for(int u_inc = 0; u_inc < BLOCK_H; ++u_inc){
            if(umax == u_offset + u_inc){
              for(int f_inc = 0; f_inc < FEATURE_SPLIT_SIZE; ++f_inc){
                grad_fcache1_ptr[f_inc] += fcache2[u_inc][v_inc][f_inc] * grad_cvmax;
              }
            }
          }
          for(int u_inc = 0; u_inc < BLOCK_H; ++u_inc){
            for(int f_inc = 0; f_inc < FEATURE_SPLIT_SIZE; ++f_inc){
              grad_fcache1_ptr[f_inc] += fcache2[u_inc][v_inc][f_inc] * grad_cvavg;
            }
          }
        }

        // write back results to global memory (for this block of u,v,f values)
        for(int f_inc = 0; f_inc < FEATURE_SPLIT_SIZE; ++f_inc){
          if(withinBoundsHc0Wc0 && f_offset + f_inc < C){
            grad_g1_ref[f_offset + f_inc] += grad_fcache1_ptr[f_inc];
          }
        }

        // sync threads before overwriting fcache2
        __syncthreads();
      }
    }
  }
}


template <typename scalar_t, int l, int div_l>
__global__ void max_argmax_avg_fmap2_backward_kernel_optimized_arch_indep (
      const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> img1_features_l0,
      const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> img2_features_lk,
      const torch::PackedTensorAccessor32<int32_t,5,torch::RestrictPtrTraits> argmax_output_u ,
      const torch::PackedTensorAccessor32<int32_t,5,torch::RestrictPtrTraits> argmax_output_v ,
      const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> grad_MaxAvg_u   ,
      const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> grad_MaxAvg_v   ,
      torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> grad_img2_features_lk)
{
  const int B = img1_features_l0.size(0); // batch size
  const int H1 = img1_features_l0.size(1); // ht of img1
  const int W1 = img1_features_l0.size(2); // wd of img1
  const int H2 = img2_features_lk.size(1); // ht of current lvl: ht/2**i
  const int W2 = img2_features_lk.size(2); // wd of current lvl: wd/2**i
  const int C = img1_features_l0.size(3); // image feature dimension
  
  const int b = blockIdx.x; // current batch

  // global starting x,y value for this block
  const int h0 = blockIdx.y * blockDim.x;   // block_i * BLOCK_H_DIV8
  const int w0 = blockIdx.z * blockDim.y;   // block_j * BLOCK_W_DIV8

  // global current x,y value for this block
  const int hc0 = h0 + threadIdx.x;
  const int wc0 = w0 + threadIdx.y;

  // size of the sub-block in (u,v)-direction
  constexpr int sub_block_size_u = div_l;
  constexpr int sub_block_size_v = div_l;

  // number of sub-blocks inside a block in each direction
  // sub-block: block where all threads have the same u,v-value
  constexpr int sub_blocks_num_u = BLOCK_H_DIV8/sub_block_size_u;
  constexpr int sub_blocks_num_v = BLOCK_W_DIV8/sub_block_size_v; 

  // global starting u,v value for this block
  const int u0 = blockIdx.y * sub_blocks_num_u;
  const int v0 = blockIdx.z * sub_blocks_num_v;
  
  // local u,v value for this thread
  const int thread_u = threadIdx.x / div_l;
  const int thread_v = threadIdx.y / div_l;

  // global u,v values of this thread
  const int uc0 = u0 + thread_u;
  const int vc0 = v0 + thread_v;

  // number of i,j blocks if level is 0
  const int iBlockLimit = (H1 + BLOCK_H_DIV8 - 1) / BLOCK_H_DIV8;
  const int jBlockLimit = (W1 + BLOCK_W_DIV8 - 1) / BLOCK_W_DIV8;

  // local i,j indices inside (u,v) sub-block
  const int sub_local_i = threadIdx.x % sub_block_size_u;
  const int sub_local_j = threadIdx.y % sub_block_size_v;

  // i,j-range of the sub-block that this thread is responsible for
  const int limit_low_i  = sub_blocks_num_u * sub_local_i;
  const int limit_low_j  = sub_blocks_num_v * sub_local_j;
  const int limit_high_i = sub_blocks_num_u * (sub_local_i + 1);
  const int limit_high_j = sub_blocks_num_v * (sub_local_j + 1);

  const int fBlockLimit = (C + FEATURE_SPLIT_SIZE - 1) / FEATURE_SPLIT_SIZE;

  // 8192 Bytes
  __shared__ scalar_t fcache1[BLOCK_H_DIV8][BLOCK_W_DIV8][FEATURE_SPLIT_SIZE];
  // for l>0: multiple fields per (u,v)
  __shared__ scalar_t grad_fcache2[BLOCK_H_DIV8][BLOCK_W_DIV8][FEATURE_SPLIT_SIZE];
  
  // 12288 Bytes
  __shared__ int32_t argmax_cache_u[BLOCK_H_DIV8][BLOCK_W_DIV8][BLOCK_H_DIV8];
  __shared__ int32_t argmax_cache_v[BLOCK_H_DIV8][BLOCK_W_DIV8][BLOCK_W_DIV8];
  __shared__ scalar_t grad_maxavg_cache_u[BLOCK_H_DIV8][BLOCK_W_DIV8][2][BLOCK_H_DIV8];
  __shared__ scalar_t grad_maxavg_cache_v[BLOCK_H_DIV8][BLOCK_W_DIV8][2][BLOCK_W_DIV8];

  scalar_t * fcache1_ptr = fcache1                [threadIdx.x][threadIdx.y];
  scalar_t * grad_fcache2_ptr = grad_fcache2      [threadIdx.x][threadIdx.y];

  auto g1_ref = img1_features_l0[b][hc0][wc0];
  auto grad_g2_ref = grad_img2_features_lk[b][uc0][vc0];
  
  
  const bool withinBoundsHc0Wc0 = within_bounds(hc0, wc0, H1, W1);
  const bool withinBoundsUc0Vc0 = within_bounds(uc0, vc0, H2, W2);
 
  // printf("u0: %i, v0: %i, uc0: %i, vc0: %i, thread_u: %i, thread_v: %i, sub_local_i: %i, sub_local_j: %i, limit_low_i: %i, limit_high_i: %i, limit_low_j: %i, limit_high_j: %i, sb_size_u: %i, sb_size_v: %i\n",
  //   u0, v0, uc0, vc0, thread_u, thread_v, sub_local_i, sub_local_j, limit_low_i, limit_high_i, limit_low_j, limit_high_j, sub_block_size_u, sub_block_size_v);

  for(int j_block = 0; j_block < jBlockLimit; ++j_block){

    const int j_offset = j_block * BLOCK_W_DIV8;

    for(int i_block = 0; i_block < iBlockLimit; ++i_block){
    
      const int i_offset = i_block * BLOCK_H_DIV8;

      bool withinBoundsLoadingThread = within_bounds(i_offset+threadIdx.x, j_offset+threadIdx.y, H1, W1);

      auto argmax_output_u_ref = argmax_output_u[b][i_offset + threadIdx.x][j_offset + threadIdx.y][0];
      auto argmax_output_v_ref = argmax_output_v[b][i_offset + threadIdx.x][j_offset + threadIdx.y][0];
      auto grad_MaxAvg_u_ref   = grad_MaxAvg_u  [b][i_offset + threadIdx.x][j_offset + threadIdx.y];
      auto grad_MaxAvg_v_ref   = grad_MaxAvg_v  [b][i_offset + threadIdx.x][j_offset + threadIdx.y];

      // printf("uc0: %i, vc0: %i, thread_u: %i, thread_v: %i, i_block: %i, j_block: %i, i_offset: %i, j_offset: %i, loading_i: %i, loading_j: %i\n",
      //   uc0, vc0, thread_u, thread_v, i_block, j_block, i_offset, j_offset, i_offset+threadIdx.x, j_offset+threadIdx.y);

      for(int v_inc = 0; v_inc < sub_blocks_num_v; ++v_inc){
        if(withinBoundsLoadingThread && v0+v_inc < W2){
          argmax_cache_v[threadIdx.x][threadIdx.y][v_inc] = argmax_output_v_ref[v0+v_inc];
        }else{
          argmax_cache_v[threadIdx.x][threadIdx.y][v_inc] = -1;
        }
      }
      for(int v_inc = 0; v_inc < sub_blocks_num_v; ++v_inc){
        if(withinBoundsLoadingThread && v0+v_inc < W2){
          grad_maxavg_cache_v[threadIdx.x][threadIdx.y][0][v_inc] = grad_MaxAvg_u_ref[0][v0+v_inc];
        }else{
          grad_maxavg_cache_v[threadIdx.x][threadIdx.y][0][v_inc] = 0.0;
        }
      }
      for(int v_inc = 0; v_inc < sub_blocks_num_v; ++v_inc){
        if(withinBoundsLoadingThread && v0+v_inc < W2){
          grad_maxavg_cache_v[threadIdx.x][threadIdx.y][1][v_inc] = grad_MaxAvg_u_ref[1][v0+v_inc];;
        }else{
          grad_maxavg_cache_v[threadIdx.x][threadIdx.y][1][v_inc] = 0.0;
        }
      }

      for(int u_inc = 0; u_inc < sub_blocks_num_u; ++u_inc){
        if(withinBoundsLoadingThread && u0+u_inc < H2){
          argmax_cache_u[threadIdx.x][threadIdx.y][u_inc] = argmax_output_u_ref[u0+u_inc];
        }else{
          argmax_cache_u[threadIdx.x][threadIdx.y][u_inc] = -1;
        }
      }
      for(int u_inc = 0; u_inc < sub_blocks_num_u; ++u_inc){
         if(withinBoundsLoadingThread && u0+u_inc < H2){
          grad_maxavg_cache_u[threadIdx.x][threadIdx.y][0][u_inc] = grad_MaxAvg_u_ref[0][u0+u_inc];
        }else{
          grad_maxavg_cache_u[threadIdx.x][threadIdx.y][0][u_inc] = 0.0;
        }
      }
      for(int u_inc = 0; u_inc < sub_blocks_num_u; ++u_inc){
        if(withinBoundsLoadingThread && u0+u_inc < H2){
          grad_maxavg_cache_u[threadIdx.x][threadIdx.y][1][u_inc] = grad_MaxAvg_u_ref[1][u0+u_inc];
        }else{
          grad_maxavg_cache_u[threadIdx.x][threadIdx.y][1][u_inc] = 0.0;
        }
      }

      for(int f_block = 0; f_block < fBlockLimit; ++f_block){
        
        const int f_offset = f_block * FEATURE_SPLIT_SIZE;

        // each thread resets its part of grad_fcache2
        for(int f_inc = 0; f_inc < FEATURE_SPLIT_SIZE; ++f_inc){
          grad_fcache2_ptr[f_inc] = 0.0;
        }

        // load image 1 features to cache
        for(int f_inc = 0; f_inc < FEATURE_SPLIT_SIZE; ++f_inc){
          if(withinBoundsLoadingThread && f_offset + f_inc < C){
            fcache1_ptr[f_inc] = img1_features_l0[b][i_offset+threadIdx.x][j_offset+threadIdx.y][f_offset + f_inc];
          }else{
            fcache1_ptr[f_inc] = 0.0;
          }
        }

        // sync threads so the complete block is loaded to shared memory
        // this means indices u0...u0+sub_blocks_num_u, v0...v0+sub_blocks_num_v, i_offset...i_offset+BLOCK_H_DIV8, j_offset...j_offset+BLOCK_W_DIV8
        __syncthreads();
        
        // calculation starts here
        // the following local indices are processed:
        // u = thread_u, v = thread_v, i=limit_low_i...limit_high_i, j=limit_low_j...limit_high_j
        // corresponding global indices are as follows:
        // u = uc0, v = vc0, i=i_offset+(limit_low_i...limit_high_i), j=j_offset+(limit_low_j...limit_high_j)

        // (L/cumax)(cumax/fmap2)
        // (L/cuavg)(cuavg/fmap2)
        // (L/cvmax)(cvmax/fmap2)
        // (L/cvavg)(cvavg/fmap2)
        for(int i_inc = limit_low_i; i_inc < limit_high_i; ++i_inc){
          for(int j_inc = limit_low_j; j_inc < limit_high_j; ++j_inc){
            
            const int vmax = argmax_cache_u[i_inc][j_inc][thread_u];
            const bool vmax_valid = (vmax >= 0) && (vmax == vc0);
           
            // if vmax not valid, ignore out-of-bounds index by setting contribution to 0
            const scalar_t grad_cumax = grad_maxavg_cache_u[i_inc][j_inc][0][thread_u] * ((scalar_t)vmax_valid);
            const scalar_t grad_cuavg = grad_maxavg_cache_u[i_inc][j_inc][1][thread_u] / W2;

            const int umax = argmax_cache_v[i_inc][j_inc][thread_v];
            const bool umax_valid = (umax >= 0) && (umax == uc0);
            
            // if umax not valid, ignore out-of-bounds index by setting contribution to 0
            const scalar_t grad_cvmax = grad_maxavg_cache_v[i_inc][j_inc][0][thread_v] * ((scalar_t)umax_valid);
            const scalar_t grad_cvavg = grad_maxavg_cache_v[i_inc][j_inc][1][thread_v] / H2;
            
            const scalar_t grad_total = grad_cvmax + grad_cvavg + grad_cumax + grad_cuavg;

            // printf("uc0: %i, vc0: %i, i: %i, j: %i, grad_total: %f\n", uc0, vc0, i_offset+i_inc, j_offset+j_inc, grad_total);

            // img1_features * (grad_cumax + grad_cuavg + grad_cvmax + grad_cvavg)
            for(int f_inc = 0; f_inc < FEATURE_SPLIT_SIZE; ++f_inc){
              // grad_fcache2_ptr[f_inc] += fcache1[i_inc][j_inc][f_inc] * grad_total;
              grad_fcache2_ptr[f_inc] += fcache1[i_inc][j_inc][f_inc] * grad_cumax;
              grad_fcache2_ptr[f_inc] += fcache1[i_inc][j_inc][f_inc] * grad_cuavg;
              grad_fcache2_ptr[f_inc] += fcache1[i_inc][j_inc][f_inc] * grad_cvmax;
              grad_fcache2_ptr[f_inc] += fcache1[i_inc][j_inc][f_inc] * grad_cvavg;
            }
          }
        }

        // need to sync so that all threads are finished with computation before corner thread accumulates their results 
        __syncthreads();

        // if true: this thread is the corner pixel of the sub_local_i, sub_local_j block
        if(sub_local_i == 0 && sub_local_j == 0 && withinBoundsUc0Vc0){
          // accumulate / write back sub-results to global memory
          // no parallel write conflicts because each (uc0, vc0) is processed by exactly one thread 
          for(int f_inc = 0; f_inc < FEATURE_SPLIT_SIZE; ++f_inc){
            scalar_t temp_f_res = 0;
            for(int i_split = threadIdx.x; i_split < threadIdx.x+sub_block_size_u; ++i_split){
              for(int j_split = threadIdx.y; j_split < threadIdx.y+sub_block_size_v; ++j_split){
              
                // if(grad_fcache2[i_split][j_split][f_inc] != 0.0) {
                //   printf("uc0: %i, vc0: %i, i_block: %i, j_block: %i, f_block: %i, i_split: %i, j_split: %i, f_inc: %i, val: %f\n",
                //     uc0, vc0, i_block, j_block, f_block, i_split, j_split, f_inc, grad_fcache2[i_split][j_split][f_inc]);
                // }
                temp_f_res += grad_fcache2[i_split][j_split][f_inc];
              }
            }
            // scalar_t * inc_val = &grad_g2_ref[f_offset + f_inc];
            // atomicAdd(inc_val, temp_f_res);
            if (f_offset + f_inc < C){
              grad_g2_ref[f_offset + f_inc] += temp_f_res;
            }
          }
        }

        // sync makes sure that cached results are not overwritten before they are added to global memory
        __syncthreads();


      }
    }
  }
}

template <typename scalar_t>
__global__ void max_argmax_avg_backward_kernel_unoptimized (
      const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> img1_features_l0,
      const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> img2_features_lk,
      const torch::PackedTensorAccessor32<int32_t,5,torch::RestrictPtrTraits> argmax_output_u ,
      const torch::PackedTensorAccessor32<int32_t,5,torch::RestrictPtrTraits> argmax_output_v ,
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
      for(int u = 0; u < H2; ++u){
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

std::vector<torch::Tensor> max_argmax_avg_cuda_backward_unoptimized (
        at::Tensor img1_features_l0, 
        at::Tensor img2_features_lk,
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
    argmax_output_u   .packed_accessor32<int32_t,5,torch::RestrictPtrTraits>(),
    argmax_output_v   .packed_accessor32<int32_t,5,torch::RestrictPtrTraits>(),
    grad_MaxAvg_u     .packed_accessor32<float,5,torch::RestrictPtrTraits>(),
    grad_MaxAvg_v     .packed_accessor32<float,5,torch::RestrictPtrTraits>(),
    // outputs
    grad_img1_features_l0     .packed_accessor32<float,4,torch::RestrictPtrTraits>(),
    grad_img2_features_lk     .packed_accessor32<float,4,torch::RestrictPtrTraits>()
  );

  // cudaError_t err = cudaThreadSynchronize();
  // printf("Run kernel: %s\n", cudaGetErrorString(err));

  return {grad_img1_features_l0, grad_img2_features_lk};
}

std::vector<torch::Tensor> max_argmax_avg_cuda_backward_optimized (
        at::Tensor img1_features_l0, 
        at::Tensor img2_features_lk,
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
  
  max_argmax_avg_backward_kernel_optimized_arch_indep <float> <<< blocks, threads >>> (
    // inputs
    img1_features_l0  .packed_accessor32<float,4,torch::RestrictPtrTraits>(), 
    img2_features_lk  .packed_accessor32<float,4,torch::RestrictPtrTraits>(),
    argmax_output_u   .packed_accessor32<int32_t,5,torch::RestrictPtrTraits>(),
    argmax_output_v   .packed_accessor32<int32_t,5,torch::RestrictPtrTraits>(),
    grad_MaxAvg_u     .packed_accessor32<float,5,torch::RestrictPtrTraits>(),
    grad_MaxAvg_v     .packed_accessor32<float,5,torch::RestrictPtrTraits>(),
    // outputs
    grad_img1_features_l0     .packed_accessor32<float,4,torch::RestrictPtrTraits>(),
    grad_img2_features_lk     .packed_accessor32<float,4,torch::RestrictPtrTraits>()
  );

  // cudaError_t err = cudaThreadSynchronize();
  // printf("Run kernel: %s\n", cudaGetErrorString(err));

  return {grad_img1_features_l0, grad_img2_features_lk};
}

std::vector<torch::Tensor> max_argmax_avg_cuda_backward_optimized_separate (
        at::Tensor img1_features_l0, 
        at::Tensor img2_features_lk,
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

  const int l = log2(((int)(H1 / H2))); // current level

  printf("level: %i\n", l);

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
  
  max_argmax_avg_fmap1_backward_kernel_optimized_arch_indep <float> <<< blocks, threads >>> (
    // inputs
    img1_features_l0  .packed_accessor32<float,4,torch::RestrictPtrTraits>(), 
    img2_features_lk  .packed_accessor32<float,4,torch::RestrictPtrTraits>(),
    argmax_output_u   .packed_accessor32<int32_t,5,torch::RestrictPtrTraits>(),
    argmax_output_v   .packed_accessor32<int32_t,5,torch::RestrictPtrTraits>(),
    grad_MaxAvg_u     .packed_accessor32<float,5,torch::RestrictPtrTraits>(),
    grad_MaxAvg_v     .packed_accessor32<float,5,torch::RestrictPtrTraits>(),
    // outputs
    grad_img1_features_l0     .packed_accessor32<float,4,torch::RestrictPtrTraits>()
  );

  cudaError_t err = cudaThreadSynchronize();
  printf("Run kernel: %s\n", cudaGetErrorString(err));


  const dim3 blocks_div8(B, (H1+BLOCK_H_DIV8-1)/BLOCK_H_DIV8, (W1+BLOCK_W_DIV8-1)/BLOCK_W_DIV8);
  const dim3 threads_div8(BLOCK_H_DIV8, BLOCK_W_DIV8);
  
  switch (l)
  {
  case 0:
    max_argmax_avg_fmap2_backward_kernel_optimized_arch_indep <float, 0, 1> <<< blocks_div8, threads_div8 >>> (
      // inputs
      img1_features_l0  .packed_accessor32<float,4,torch::RestrictPtrTraits>(), 
      img2_features_lk  .packed_accessor32<float,4,torch::RestrictPtrTraits>(),
      argmax_output_u   .packed_accessor32<int32_t,5,torch::RestrictPtrTraits>(),
      argmax_output_v   .packed_accessor32<int32_t,5,torch::RestrictPtrTraits>(),
      grad_MaxAvg_u     .packed_accessor32<float,5,torch::RestrictPtrTraits>(),
      grad_MaxAvg_v     .packed_accessor32<float,5,torch::RestrictPtrTraits>(),
      // outputs
      grad_img2_features_lk     .packed_accessor32<float,4,torch::RestrictPtrTraits>()
    );
    break;
  case 1:
    max_argmax_avg_fmap2_backward_kernel_optimized_arch_indep <float, 1, 2> <<< blocks_div8, threads_div8 >>> (
      // inputs
      img1_features_l0  .packed_accessor32<float,4,torch::RestrictPtrTraits>(),
      img2_features_lk  .packed_accessor32<float,4,torch::RestrictPtrTraits>(), 
      argmax_output_u   .packed_accessor32<int32_t,5,torch::RestrictPtrTraits>(),
      argmax_output_v   .packed_accessor32<int32_t,5,torch::RestrictPtrTraits>(),
      grad_MaxAvg_u     .packed_accessor32<float,5,torch::RestrictPtrTraits>(),
      grad_MaxAvg_v     .packed_accessor32<float,5,torch::RestrictPtrTraits>(),
      // outputs
      grad_img2_features_lk     .packed_accessor32<float,4,torch::RestrictPtrTraits>()
    );
    break;
  case 2:
    max_argmax_avg_fmap2_backward_kernel_optimized_arch_indep <float, 2, 4> <<< blocks_div8, threads_div8 >>> (
      // inputs
      img1_features_l0  .packed_accessor32<float,4,torch::RestrictPtrTraits>(), 
      img2_features_lk  .packed_accessor32<float,4,torch::RestrictPtrTraits>(),
      argmax_output_u   .packed_accessor32<int32_t,5,torch::RestrictPtrTraits>(),
      argmax_output_v   .packed_accessor32<int32_t,5,torch::RestrictPtrTraits>(),
      grad_MaxAvg_u     .packed_accessor32<float,5,torch::RestrictPtrTraits>(),
      grad_MaxAvg_v     .packed_accessor32<float,5,torch::RestrictPtrTraits>(),
      // outputs
      grad_img2_features_lk     .packed_accessor32<float,4,torch::RestrictPtrTraits>()
    );
    break;
  case 3:
    max_argmax_avg_fmap2_backward_kernel_optimized_arch_indep <float, 3, 8> <<< blocks_div8, threads_div8 >>> (
      // inputs
      img1_features_l0  .packed_accessor32<float,4,torch::RestrictPtrTraits>(),
      img2_features_lk  .packed_accessor32<float,4,torch::RestrictPtrTraits>(), 
      argmax_output_u   .packed_accessor32<int32_t,5,torch::RestrictPtrTraits>(),
      argmax_output_v   .packed_accessor32<int32_t,5,torch::RestrictPtrTraits>(),
      grad_MaxAvg_u     .packed_accessor32<float,5,torch::RestrictPtrTraits>(),
      grad_MaxAvg_v     .packed_accessor32<float,5,torch::RestrictPtrTraits>(),
      // outputs
      grad_img2_features_lk     .packed_accessor32<float,4,torch::RestrictPtrTraits>()
    );
    break;
  
  default:
    throw std::invalid_argument( "level of img2_features_lk is " + std::to_string(l) + "and not included in set {0,1,2,3}" );
  }
  

  err = cudaThreadSynchronize();
  printf("Run kernel: %s\n", cudaGetErrorString(err));

  return {grad_img1_features_l0, grad_img2_features_lk};
}

std::vector<torch::Tensor> max_argmax_avg_cuda_backward (
        at::Tensor img1_features_l0, 
        at::Tensor img2_features_lk,
        at::Tensor argmax_output_u,
        at::Tensor argmax_output_v,
        at::Tensor grad_MaxAvg_u,
        at::Tensor grad_MaxAvg_v){

  // return max_argmax_avg_cuda_backward_unoptimized(
  //   img1_features_l0,
  //   img2_features_lk,
  //   argmax_output_u,
  //   argmax_output_v,
  //   grad_MaxAvg_u,
  //   grad_MaxAvg_v);
  return max_argmax_avg_cuda_backward_optimized(
    img1_features_l0,
    img2_features_lk,
    argmax_output_u,
    argmax_output_v,
    grad_MaxAvg_u,
    grad_MaxAvg_v);
  // return max_argmax_avg_cuda_backward_optimized_separate(
  //   img1_features_l0,
  //   img2_features_lk,
  //   argmax_output_u,
  //   argmax_output_v,
  //   grad_MaxAvg_u,
  //   grad_MaxAvg_v);
}