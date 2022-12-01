#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

// definitions of constants

// block dimension sizes
#define BLOCK_H 4
#define BLOCK_W 8
#define BLOCK_HW BLOCK_H * BLOCK_W

#define CHANNEL_STRIDE 32
#define FEATURE_SIZE 256

// size of the feature split (cache features dimension size)
#define FEATURE_SPLIT_SIZE 32

// define number of self-adaptive compression to be 2
#define K_VAL 2

#define DIM0(TENSOR) ((TENSOR).x)
#define DIM1(TENSOR) ((TENSOR).y)
#define DIM2(TENSOR) ((TENSOR).z)
#define DIM3(TENSOR) ((TENSOR).w)

#define DIM3_INDEX(TENSOR, xx, yy, zz, ww) ((TENSOR)[((xx) * (TENSOR##_stride.x)) + ((yy) * (TENSOR##_stride.y)) + ((zz) * (TENSOR##_stride.z)) + ((ww) * (TENSOR##_stride.w))])

/**
 * @brief check if index (h,w) is within a 2D range of (0,0) to (H,W)
 */
__forceinline__ __device__
bool within_bounds(int h, int w, int H, int W) {
  return h >= 0 && h < H && w >= 0 && w < W;
}

/**
 * @brief Compute the maximum and average channels as first attempt to use shared memory
 *        Unused
 * 
 * @tparam scalar_t           datatype of the tensors (4 Byte float)
 * @param img1_features_l0    image one feature tensor at pyramid level 0
 * @param img2_features_lk    image two feature tensor at pyramid level l 
 * @param max_avg_output_u    output tensor for the maximum and average channels C_u^{max,avg} 
 * @param max_avg_output_v    output tensor for the maximum and average channels C_v^{max,avg} 
 * @return __global__ 
 */
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

  if( C != FEATURE_SIZE ) {
    printf("Feature size has to be %i but is %i", FEATURE_SIZE, C);
  }

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

    __shared__ scalar_t g1_block[BLOCK_HW][FEATURE_SIZE];
    scalar_t * g1 = g1_block[tIdx];
    for(int i = 0; i<C; ++i){
      g1[i] = g1_ref[i];
    }

    for (int v = 0; v<W2; ++v){
      auto max_u_ref = max_avg_output_u[b][hc0][wc0][0];
      auto avg_u_ref = max_avg_output_u[b][hc0][wc0][1];
      auto max_v_ref = max_avg_output_v[b][hc0][wc0][0];
      auto avg_v_ref = max_avg_output_v[b][hc0][wc0][1];
      
      for(int u = 0; u<H2; ++u){

        auto g2_ref = img2_features_lk[b][u][v];

        scalar_t cval = 0;
        for(int i = 0; i<C; ++i){
          cval += g1[i]*g2_ref[i];  
        }

        max_u_ref [u] = max(max_u_ref [u], cval);
        avg_u_ref [u] += cval / W2;

        max_v_ref [v] = max(max_v_ref [v], cval);
        avg_v_ref [v] += cval / H2;

      }
    }
  }

}

/**
 * @brief Optimized Max-Avg-Kernel that is architecture independent with respect to Shared
 *        Memory Requirement (15360Bytes)
 * 
 * @tparam scalar_t         datatype of the arrays
 * @param img1_features_l0  image1 features at level 0 of shape (batch, h1, w1, fdim)
 * @param img2_features_lk  image2 features at level k of shape (batch, h1/2**i, w1/2**i, fdim)
 * @param max_avg_output_u  output for max/avg values of C_u at level k of shape (batch, h1, w1, 2, h1/2**k)
 * @param max_avg_output_v  output for max/avg values of C_v at level k of shape (batch, h1, w1, 2, w1/2**k)
 * @return __global__ 
 */
template <typename scalar_t>
__global__ void max_avg_forward_kernel_optimized_arch_indep (
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

  // global current x/y value for this thread (pixel in img1)
  const int hc0 = h0 + threadIdx.x;
  const int wc0 = w0 + threadIdx.y;

  // number of "blocks" in u,v and feature direction
  // not cuda blocks, but loaded into shared memory at the same time
  const int uBlockLimit = (H2 + BLOCK_H - 1) / BLOCK_H;
  const int vBlockLimit = (W2 + BLOCK_W - 1) / BLOCK_W;
  const int fBlockLimit = (C + FEATURE_SPLIT_SIZE - 1) / FEATURE_SPLIT_SIZE;

  // The sum should be below 49152 Bytes to be architecture independent

  // 4096 Bytes
  __shared__ scalar_t corr[BLOCK_H][BLOCK_W][BLOCK_H][BLOCK_W];
  // 4096 Bytes
  __shared__ scalar_t fcache1[BLOCK_H][BLOCK_W][FEATURE_SPLIT_SIZE];
  // 4096 Bytes
  __shared__ scalar_t fcache2[BLOCK_H][BLOCK_W][FEATURE_SPLIT_SIZE];
  // 2 * 512 = 1024 Bytes
  __shared__ scalar_t uMax[BLOCK_H][BLOCK_W][BLOCK_H];
  __shared__ scalar_t uAvg[BLOCK_H][BLOCK_W][BLOCK_H];
  // 2 * 1024 = 2048 Bytes
  __shared__ scalar_t vMax[BLOCK_H][BLOCK_W][BLOCK_W];
  __shared__ scalar_t vAvg[BLOCK_H][BLOCK_W][BLOCK_W];

  // references into relevant array dimensions for this thread
  auto g1_ref = img1_features_l0[b][hc0][wc0];
  auto max_u_global_ref = max_avg_output_u[b][hc0][wc0][0];
  auto avg_u_global_ref = max_avg_output_u[b][hc0][wc0][1];
  auto max_v_global_ref = max_avg_output_v[b][hc0][wc0][0];
  auto avg_v_global_ref = max_avg_output_v[b][hc0][wc0][1];

  // pointers into relevant array dimensions for this thread
  scalar_t (* corr_ref) [BLOCK_W] = corr [threadIdx.x][threadIdx.y];
  scalar_t * uMax_ref = uMax [threadIdx.x][threadIdx.y];
  scalar_t * uAvg_ref = uAvg [threadIdx.x][threadIdx.y];
  scalar_t * vMax_ref = vMax [threadIdx.x][threadIdx.y];
  scalar_t * vAvg_ref = vAvg [threadIdx.x][threadIdx.y];
  
  bool withinBoundsHc0Wc0 = within_bounds(hc0, wc0, H1, W1);

  for (int v_block = 0; v_block < vBlockLimit; ++v_block){

    const int v_offset = v_block * BLOCK_W;

    // load v values
    for (int v_inc = 0; v_inc < BLOCK_W; ++v_inc){
      if(withinBoundsHc0Wc0 && v_offset + v_inc < W2){
        vMax_ref [v_inc] = max_v_global_ref[v_offset + v_inc];
        vAvg_ref [v_inc] = avg_v_global_ref[v_offset + v_inc];
      }else{
        vMax_ref [v_inc] = -INFINITY;
        vAvg_ref [v_inc] = 0;
      }
    }

    for (int u_block = 0; u_block < uBlockLimit; ++u_block){

      const int u_offset = u_block * BLOCK_H;

      // reset correlation volume at the start of the uv-block (each thread resets one part)
      for (int i = 0; i < BLOCK_H; ++i){
        for (int j = 0; j < BLOCK_W; ++j){
          corr_ref[i][j] = 0.0;
        }
      }

      __syncthreads();

      // compute block correlation volume in splits
      for (int f_block = 0; f_block < fBlockLimit; ++f_block){
        
        int f_offset = f_block*FEATURE_SPLIT_SIZE;
        
        // load fmap1 split into shared memory (every thread is responsible for one)
        scalar_t * fcache1_ptr = fcache1[threadIdx.x][threadIdx.y];
        for(int i = 0; i<FEATURE_SPLIT_SIZE; ++i){
          if(withinBoundsHc0Wc0 && f_offset+i < C){
            fcache1_ptr[i] = g1_ref[f_offset+i];
          }else{
            fcache1_ptr[i] = 0;
          }
        }

        // load fmap2 split into shared memory (every thread is responsible for one)
        auto g2_ref = img2_features_lk[b][u_offset + threadIdx.x][v_offset + threadIdx.y];
        scalar_t * fcache2_ptr = fcache2[threadIdx.x][threadIdx.y];
        for(int i = 0; i<FEATURE_SPLIT_SIZE; ++i){
          if(within_bounds(u_offset+threadIdx.x, v_offset+threadIdx.y, H2, W2) && f_offset+i < C){
            fcache2_ptr[i] = g2_ref[f_offset+i];
          }else{
            fcache2_ptr[i] = 0.0;
          }
        }

        // sync to prevent threads from using last iterations fcache2 values for corr computation
        __syncthreads();

        // accumulate block correlation volume
        for (int u_inc = 0; u_inc < BLOCK_H; ++u_inc){
          for (int v_inc = 0; v_inc < BLOCK_W; ++v_inc){
            for(int f_inc = 0; f_inc<FEATURE_SPLIT_SIZE; ++f_inc){
              corr_ref[u_inc][v_inc] += fcache1[threadIdx.x][threadIdx.y][f_inc]*fcache2[u_inc][v_inc][f_inc];
            }
          }
        }

        // sync to prevent threads from re-writing fcache2 before all threads are done with it
        __syncthreads();
      }

      // now corr holds the values of the correlation volume for (block_u, block_v)

      // load u values from global memory to shared memory
      for (int u_inc = 0; u_inc < BLOCK_H; ++u_inc){
        if(withinBoundsHc0Wc0 && u_offset+u_inc < H2){
          uMax_ref [u_inc] = max_u_global_ref[u_offset + u_inc];
          uAvg_ref [u_inc] = avg_u_global_ref[u_offset + u_inc];
        }else{
          uMax_ref [u_inc] = -INFINITY;
          uAvg_ref [u_inc] = 0;
        }
      }

      // compute umax,uavg,vmax,vavg
      for (int u_inc = 0; u_inc < BLOCK_H; ++u_inc){
        for (int v_inc = 0; v_inc < BLOCK_W; ++v_inc){
          scalar_t cval = corr_ref[u_inc][v_inc];

          // check for out-of-bounds correlation fields (they are 0-valued)
          // replace default-zero value by default-negative-infinity
          // negative infinity is always rejected by max and thus does not influence it
          scalar_t cval_checked = cval;
          if(!(withinBoundsHc0Wc0 && u_offset+u_inc < H2 && v_offset+v_inc < W2)){
            cval_checked = -INFINITY;
          }

          uMax_ref [u_inc] = max(uMax_ref [u_inc], cval_checked);
          uAvg_ref [u_inc] += cval / W2;

          vMax_ref [v_inc] = max(vMax_ref [v_inc], cval_checked);
          vAvg_ref [v_inc] += cval;
        }
      }

      // store u values into global memory from shared memory
      for (int u_inc = 0; u_inc < BLOCK_H; ++u_inc){
        if(withinBoundsHc0Wc0 && u_offset + u_inc < H2){
          max_u_global_ref[u_offset + u_inc] = uMax_ref [u_inc];
          avg_u_global_ref[u_offset + u_inc] = uAvg_ref [u_inc];
        }
      }

    }

    // store v values into global memory from shared memory
    for (int v_inc = 0; v_inc < BLOCK_W; ++v_inc){
      if(withinBoundsHc0Wc0 && v_offset + v_inc < W2){
        max_v_global_ref[v_offset + v_inc] = vMax_ref [v_inc];
        avg_v_global_ref[v_offset + v_inc] = vAvg_ref [v_inc] / H2;
      }
    }

  }
}

/**
 * @brief Optimized Max-Argmax-Avg-Kernel that is architecture independent with respect to Shared
 *        Memory Requirement (16896 Bytes)
 * 
 * @tparam scalar_t         datatype of the arrays
 * @param img1_features_l0  image1 features at level 0 of shape (batch, h1, w1, fdim)
 * @param img2_features_lk  image2 features at level k of shape (batch, h1/2**i, w1/2**i, fdim)
 * @param max_avg_output_u  output for max/avg values of C_u at level k of shape (batch, h1, w1, 2, h1/2**k)
 * @param max_avg_output_v  output for max/avg values of C_v at level k of shape (batch, h1, w1, 2, w1/2**k)
 * @param argmax_output_u  output for argmax values of C_u at level k of shape (batch, h1, w1, 1, h1/2**k)
 * @param argmax_output_v  output for argmax values of C_v at level k of shape (batch, h1, w1, 1, w1/2**k)
 * @return __global__ 
 */
template <typename scalar_t>
__global__ void max_argmax_avg_forward_kernel_optimized_arch_indep (
  const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> img1_features_l0,
  const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> img2_features_lk,
  torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> max_avg_output_u,
  torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> max_avg_output_v,
  torch::PackedTensorAccessor32<int32_t,5,torch::RestrictPtrTraits> argmax_output_u,
  torch::PackedTensorAccessor32<int32_t,5,torch::RestrictPtrTraits> argmax_output_v)
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

  // global current x/y value for this thread (pixel in img1)
  const int hc0 = h0 + threadIdx.x;
  const int wc0 = w0 + threadIdx.y;

  // number of "blocks" in u,v and feature direction
  // not cuda blocks, but loaded into shared memory at the same time
  const int uBlockLimit = (H2 + BLOCK_H - 1) / BLOCK_H;
  const int vBlockLimit = (W2 + BLOCK_W - 1) / BLOCK_W;
  const int fBlockLimit = (C + FEATURE_SPLIT_SIZE - 1) / FEATURE_SPLIT_SIZE;

  // The sum should be below 49152 Bytes to be architecture independent

  // 4096 Bytes
  __shared__ scalar_t corr[BLOCK_H][BLOCK_W][BLOCK_H][BLOCK_W];
  // 4096 Bytes
  __shared__ scalar_t fcache1[BLOCK_H][BLOCK_W][FEATURE_SPLIT_SIZE];
  // 4096 Bytes
  __shared__ scalar_t fcache2[BLOCK_H][BLOCK_W][FEATURE_SPLIT_SIZE];
  // 3 * 512 = 1536 Bytes
  __shared__ scalar_t uMax[BLOCK_H][BLOCK_W][BLOCK_H];
  __shared__ int32_t uArgmax[BLOCK_H][BLOCK_W][BLOCK_H];
  __shared__ scalar_t uAvg[BLOCK_H][BLOCK_W][BLOCK_H];
  // 3 * 1024 = 3072 Bytes
  __shared__ scalar_t vMax[BLOCK_H][BLOCK_W][BLOCK_W];
  __shared__ int32_t vArgmax[BLOCK_H][BLOCK_W][BLOCK_W];
  __shared__ scalar_t vAvg[BLOCK_H][BLOCK_W][BLOCK_W];

  // references into relevant array dimensions for this thread
  auto g1_ref = img1_features_l0[b][hc0][wc0];
  auto max_u_global_ref = max_avg_output_u[b][hc0][wc0][0];
  auto avg_u_global_ref = max_avg_output_u[b][hc0][wc0][1];
  auto argmax_u_global_ref = argmax_output_u[b][hc0][wc0][0];
  auto max_v_global_ref = max_avg_output_v[b][hc0][wc0][0];
  auto avg_v_global_ref = max_avg_output_v[b][hc0][wc0][1];
  auto argmax_v_global_ref = argmax_output_v[b][hc0][wc0][0];

  // pointers into relevant array dimensions for this thread
  scalar_t (* corr_ref) [BLOCK_W] = corr [threadIdx.x][threadIdx.y];
  scalar_t * uMax_ref = uMax [threadIdx.x][threadIdx.y];
  int32_t * uArgmax_ref = uArgmax [threadIdx.x][threadIdx.y];
  scalar_t * uAvg_ref = uAvg [threadIdx.x][threadIdx.y];
  scalar_t * vMax_ref = vMax [threadIdx.x][threadIdx.y];
  int32_t * vArgmax_ref = vArgmax [threadIdx.x][threadIdx.y];
  scalar_t * vAvg_ref = vAvg [threadIdx.x][threadIdx.y];
  
  bool withinBoundsHc0Wc0 = within_bounds(hc0, wc0, H1, W1);

  for (int v_block = 0; v_block < vBlockLimit; ++v_block){

    const int v_offset = v_block * BLOCK_W;

    // load v values
    for (int v_inc = 0; v_inc < BLOCK_W; ++v_inc){
      if(withinBoundsHc0Wc0 && v_offset + v_inc < W2){
        vMax_ref [v_inc] = max_v_global_ref[v_offset + v_inc];
        vArgmax_ref [v_inc] = argmax_v_global_ref[v_offset + v_inc];
        vAvg_ref [v_inc] = avg_v_global_ref[v_offset + v_inc];
      }else{
        vMax_ref [v_inc] = -INFINITY;
        vArgmax_ref [v_inc] = -1;
        vAvg_ref [v_inc] = 0;
      }
    }

    for (int u_block = 0; u_block < uBlockLimit; ++u_block){

      const int u_offset = u_block * BLOCK_H;

      // reset correlation volume at the start of the uv-block (each thread resets one part)
      for (int i = 0; i < BLOCK_H; ++i){
        for (int j = 0; j < BLOCK_W; ++j){
          corr_ref[i][j] = 0.0;
        }
      }

      __syncthreads();

      // compute block correlation volume in splits
      for (int f_block = 0; f_block < fBlockLimit; ++f_block){
        
        int f_offset = f_block*FEATURE_SPLIT_SIZE;
        
        // load fmap1 split into shared memory (every thread is responsible for one)
        scalar_t * fcache1_ptr = fcache1[threadIdx.x][threadIdx.y];
        for(int i = 0; i<FEATURE_SPLIT_SIZE; ++i){
          if(withinBoundsHc0Wc0 && f_offset+i < C){
            fcache1_ptr[i] = g1_ref[f_offset+i];
          }else{
            fcache1_ptr[i] = 0;
          }
        }

        // load fmap2 split into shared memory (every thread is responsible for one)
        auto g2_ref = img2_features_lk[b][u_offset + threadIdx.x][v_offset + threadIdx.y];
        scalar_t * fcache2_ptr = fcache2[threadIdx.x][threadIdx.y];
        for(int i = 0; i<FEATURE_SPLIT_SIZE; ++i){
          if(within_bounds(u_offset+threadIdx.x, v_offset+threadIdx.y, H2, W2) && f_offset+i < C){
            fcache2_ptr[i] = g2_ref[f_offset+i];
          }else{
            fcache2_ptr[i] = 0.0;
          }
        }

        // sync to prevent threads from using last iterations fcache2 values for corr computation
        __syncthreads();

        // accumulate block correlation volume
        for (int u_inc = 0; u_inc < BLOCK_H; ++u_inc){
          for (int v_inc = 0; v_inc < BLOCK_W; ++v_inc){
            for(int f_inc = 0; f_inc<FEATURE_SPLIT_SIZE; ++f_inc){
              corr_ref[u_inc][v_inc] += fcache1[threadIdx.x][threadIdx.y][f_inc]*fcache2[u_inc][v_inc][f_inc];
            }
          }
        }

        // sync to prevent threads from re-writing fcache2 before all threads are done with it
        __syncthreads();
      }

      // now corr holds the values of the correlation volume for (block_u, block_v)

      // load u values from global memory to shared memory
      for (int u_inc = 0; u_inc < BLOCK_H; ++u_inc){
        if(withinBoundsHc0Wc0 && u_offset+u_inc < H2){
          uMax_ref [u_inc] = max_u_global_ref[u_offset + u_inc];
          uArgmax_ref [u_inc] = argmax_u_global_ref[u_offset + u_inc];
          uAvg_ref [u_inc] = avg_u_global_ref[u_offset + u_inc];
        }else{
          uMax_ref [u_inc] = -INFINITY;
          uArgmax_ref [u_inc] = -1;
          uAvg_ref [u_inc] = 0;
        }
      }

      // compute umax,uavg,vmax,vavg
      for (int u_inc = 0; u_inc < BLOCK_H; ++u_inc){
        for (int v_inc = 0; v_inc < BLOCK_W; ++v_inc){
          scalar_t cval = corr_ref[u_inc][v_inc];
          
          // check for out-of-bounds correlation fields (they are 0-valued)
          // replace default-zero value by default-negative-infinity
          // negative infinity is always rejected by max and thus does not influence it
          scalar_t cval_checked = cval;
          if(!(withinBoundsHc0Wc0 && u_offset+u_inc < H2 && v_offset+v_inc < W2)){
            cval_checked = -INFINITY;
          }

          if(uMax_ref [u_inc] < cval_checked){
            uMax_ref [u_inc] = cval_checked;
            uArgmax_ref [u_inc] = v_offset + v_inc;
          }
          uAvg_ref [u_inc] += cval / W2;

          if(vMax_ref [v_inc] < cval_checked){
            vMax_ref [v_inc] = cval_checked;
            vArgmax_ref [v_inc] = u_offset + u_inc;
          }
          vAvg_ref [v_inc] += cval;
        }
      }

      // store u values into global memory from shared memory
      for (int u_inc = 0; u_inc < BLOCK_H; ++u_inc){
        if(withinBoundsHc0Wc0 && u_offset + u_inc < H2){
          max_u_global_ref[u_offset + u_inc] = uMax_ref [u_inc];
          argmax_u_global_ref[u_offset + u_inc] = uArgmax_ref [u_inc];
          avg_u_global_ref[u_offset + u_inc] = uAvg_ref [u_inc];
        }
      }

    }

    // store v values into global memory from shared memory
    for (int v_inc = 0; v_inc < BLOCK_W; ++v_inc){
      if(withinBoundsHc0Wc0 && v_offset + v_inc < W2){
        max_v_global_ref[v_offset + v_inc] = vMax_ref [v_inc];
        argmax_v_global_ref[v_offset + v_inc] = vArgmax_ref [v_inc];
        avg_v_global_ref[v_offset + v_inc] = vAvg_ref [v_inc] / H2;
      }
    }

  }
}


/**
 * @brief Optimized Compression-Kernel that is architecture independent with respect to Shared
 *        Memory Requirement (18432 Bytes for CONST_K = 2)
 * 
 * @tparam scalar_t             datatype of the arrays
 * @tparam CONST_K              number of correlation features
 * @param img1_features_l0      image1 features at level 0 of shape (batch, h1, wd, fdim)
 * @param img2_features_lk      image2 features at level k of shape (batch, h1/2**i, w1/2**i, fdim)
 * @param attention_weights_u   attention weights for compression of shape (batch, h1, wd, K-2, w1/2**i)
 * @param attention_weights_v   attention weights for compression of shape (batch, h1, wd, K-2, h1/2**i)
 * @param compression_output_u  output for compression values of C_u at level k of shape (batch, h1, w1, K-2, h1/2**k)
 * @param compression_output_v  output for compression values of C_v at level k of shape (batch, h1, w1, K-2, w1/2**k)
 * @return __global__ 
 */
template <typename scalar_t, int CONST_K>
__global__ void compression_forward_kernel_optimized_arch_indep (
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

  const int uBlockLimit = (H2 + BLOCK_H - 1) / BLOCK_H;
  const int vBlockLimit = (W2 + BLOCK_W - 1) / BLOCK_W;

  const int fBlockLimit = (C + FEATURE_SPLIT_SIZE - 1) / FEATURE_SPLIT_SIZE;

  // The sum should be below 49152 Bytes to be architecture independent
  // The sum is 12288 + 3072*CONST_K
  // For K=2: 18432 Bytes

  // 4096 Bytes
  __shared__ scalar_t corr[BLOCK_H][BLOCK_W][BLOCK_H][BLOCK_W];
  // 4096 Bytes
  __shared__ scalar_t fcache1[BLOCK_H][BLOCK_W][FEATURE_SPLIT_SIZE];
  // 4096 Bytes
  __shared__ scalar_t fcache2[BLOCK_H][BLOCK_W][FEATURE_SPLIT_SIZE];
  
  // (512 + 1024) * 2 * CONST_K Bytes
  __shared__ scalar_t uAtt[BLOCK_H][BLOCK_W][CONST_K][BLOCK_W];
  __shared__ scalar_t vAtt[BLOCK_H][BLOCK_W][CONST_K][BLOCK_H];
  __shared__ scalar_t uCom[BLOCK_H][BLOCK_W][CONST_K][BLOCK_H];
  __shared__ scalar_t vCom[BLOCK_H][BLOCK_W][CONST_K][BLOCK_W];

  // references into relevant array dimensions for this thread
  auto g1_ref = img1_features_l0[b][hc0][wc0];
  auto attention_u_global_ref = attention_weights_u[b][hc0][wc0];
  auto attention_v_global_ref = attention_weights_v[b][hc0][wc0];
  auto compressed_u_global_ref = compression_output_u[b][hc0][wc0];
  auto compressed_v_global_ref = compression_output_v[b][hc0][wc0];
  
  // pointers into relevant array dimensions for this thread
  scalar_t (* corr_ref) [BLOCK_W] = corr [threadIdx.x][threadIdx.y];
  scalar_t (* uAtt_ref) [BLOCK_W] = uAtt [threadIdx.x][threadIdx.y];
  scalar_t (* vAtt_ref) [BLOCK_H] = vAtt [threadIdx.x][threadIdx.y];
  scalar_t (* uCom_ref) [BLOCK_H] = uCom [threadIdx.x][threadIdx.y];
  scalar_t (* vCom_ref) [BLOCK_W] = vCom [threadIdx.x][threadIdx.y];
  
  bool withinBoundsHc0Wc0 = within_bounds(hc0, wc0, H1, W1);

  for (int v_block = 0; v_block < vBlockLimit; ++v_block){

    const int v_offset = v_block * BLOCK_W;

    // load v values
    for(int k_inc = 0; k_inc < CONST_K; ++k_inc){
      auto k_attention_u_global_ref = attention_u_global_ref[k_inc];
      auto k_compressed_v_global_ref = compressed_v_global_ref[k_inc];
      scalar_t * k_uAtt_ref = uAtt_ref[k_inc];
      scalar_t * k_vCom_ref = vCom_ref[k_inc];
      for (int v_inc = 0; v_inc < BLOCK_W; ++v_inc){
        if(withinBoundsHc0Wc0 && v_offset + v_inc < W2){
          k_uAtt_ref [v_inc] = k_attention_u_global_ref[v_offset + v_inc];
          k_vCom_ref [v_inc] = k_compressed_v_global_ref[v_offset + v_inc];
        }else{
          k_uAtt_ref [v_inc] = 0;
          k_vCom_ref [v_inc] = 0;
        }
      }
    }

    for (int u_block = 0; u_block < uBlockLimit; ++u_block){

      const int u_offset = u_block * BLOCK_H;

      // reset correlation volume at the start of the uv-block (each thread resets one part)
      for (int i = 0; i < BLOCK_H; ++i){
        for (int j = 0; j < BLOCK_W; ++j){
          corr_ref[i][j] = 0.0;
        }
      }

      __syncthreads();

      // compute block correlation volume in splits
      for (int f_block = 0; f_block < fBlockLimit; ++f_block){
        
        int f_offset = f_block*FEATURE_SPLIT_SIZE;
        
        // load fmap1 split into shared memory (every thread is responsible for one)
        scalar_t * fcache1_ptr = fcache1[threadIdx.x][threadIdx.y];
        for(int i = 0; i<FEATURE_SPLIT_SIZE; ++i){
          if(withinBoundsHc0Wc0 && f_offset+i < C){
            fcache1_ptr[i] = g1_ref[f_offset+i];
          }else{
            fcache1_ptr[i] = 0.0;
          }
        }

        // load fmap2 split into shared memory (every thread is responsible for one)
        auto g2_ref = img2_features_lk[b][u_offset + threadIdx.x][v_offset + threadIdx.y];
        scalar_t * fcache2_ptr = fcache2[threadIdx.x][threadIdx.y];
        for(int i = 0; i<FEATURE_SPLIT_SIZE; ++i){
          if(within_bounds(u_offset+threadIdx.x, v_offset+threadIdx.y, H2, W2) && f_offset+i < C){
            fcache2_ptr[i] = g2_ref[f_offset+i];
          }else{
            fcache2_ptr[i] = 0.0;
          }
        }

        // sync to prevent threads from using last iterations fcache2 values for corr computation
        __syncthreads();

        // accumulate block correlation volume
        for (int u_inc = 0; u_inc < BLOCK_H; ++u_inc){
          for (int v_inc = 0; v_inc < BLOCK_W; ++v_inc){
            for(int f_inc = 0; f_inc<FEATURE_SPLIT_SIZE; ++f_inc){
              corr_ref[u_inc][v_inc] += fcache1[threadIdx.x][threadIdx.y][f_inc]*fcache2[u_inc][v_inc][f_inc];
            }
          }
        }

        // sync to prevent threads from re-writing fcache2 before all threads are done with it
        __syncthreads();
      }

      // now corr holds the values of the correlation volume for (block_u, block_v)

      // load u values from global memory to shared memory
      for(int k_inc = 0; k_inc < CONST_K; ++k_inc){
        auto k_attention_v_global_ref = attention_v_global_ref[k_inc];
        auto k_compressed_u_global_ref = compressed_u_global_ref[k_inc];
        scalar_t * k_vAtt_ref = vAtt_ref[k_inc];
        scalar_t * k_uCom_ref = uCom_ref[k_inc];
        for (int u_inc = 0; u_inc < BLOCK_H; ++u_inc){
          if(withinBoundsHc0Wc0 && u_offset+u_inc < H2){
            k_vAtt_ref [u_inc] = k_attention_v_global_ref[u_offset + u_inc];
            k_uCom_ref [u_inc] = k_compressed_u_global_ref[u_offset + u_inc];
          }else{
            k_vAtt_ref [u_inc] = 0;
            k_uCom_ref [u_inc] = 0;
          }
        }
      }

      // compute compressed u and v values for each self-adaptive correlation feature channel
      for(int k_inc = 0; k_inc < CONST_K; ++k_inc){
        // create refs for current k value
        scalar_t * k_uAtt_ref = uAtt_ref[k_inc];
        scalar_t * k_uCom_ref = uCom_ref[k_inc];
        scalar_t * k_vAtt_ref = vAtt_ref[k_inc];
        scalar_t * k_vCom_ref = vCom_ref[k_inc];
        for (int u_inc = 0; u_inc < BLOCK_H; ++u_inc){
          for (int v_inc = 0; v_inc < BLOCK_W; ++v_inc){
            scalar_t cval = corr_ref[u_inc][v_inc];

            k_uCom_ref [u_inc] += cval * k_uAtt_ref [v_inc];
            k_vCom_ref [v_inc] += cval * k_vAtt_ref [u_inc];
          }
        }
      }

      // store u values into global memory from shared memory
      for(int k_inc = 0; k_inc < CONST_K; ++k_inc){
        auto k_compressed_u_global_ref = compressed_u_global_ref[k_inc];
        scalar_t * k_uCom_ref = uCom_ref[k_inc];
        for (int u_inc = 0; u_inc < BLOCK_H; ++u_inc){
          if(withinBoundsHc0Wc0 && u_offset + u_inc < H2){
            k_compressed_u_global_ref[u_offset + u_inc] = k_uCom_ref [u_inc];
          }
        }
      }

    }

    // store v values into global memory from shared memory
    for(int k_inc = 0; k_inc < CONST_K; ++k_inc){
      auto k_compressed_v_global_ref = compressed_v_global_ref[k_inc];
      scalar_t * k_vCom_ref = vCom_ref[k_inc];
      for (int v_inc = 0; v_inc < BLOCK_W; ++v_inc){
        if(withinBoundsHc0Wc0 && v_offset + v_inc < W2){
          k_compressed_v_global_ref [v_offset + v_inc] = k_vCom_ref [v_inc];
        }
      }
    }
  
  }
}


/**
 * @brief kernel for computing the maximum and average channels 
 * 
 * @tparam scalar_t         datatype of the tensors (4 Byte float)
 * @param img1_features_l0  image one features at pyramid level 0
 * @param img2_features_lk  image two features at pyramid level l
 * @param max_avg_output_u  output tensor of the maximum and average channels C_u^{max,avg}
 * @param max_avg_output_v  output tensor of the maximum and average channels C_v^{max,avg} 
 * @return __global__ 
 */
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


/**
 * @brief kernel for computing the attention-based channels
 * 
 * @tparam scalar_t 
 * @param img1_features_l0      image one features tensor at pyramid level 0
 * @param img2_features_lk      image two features tensor at pyramid level l
 * @param attention_weights_u   attention weights tensor for Cu
 * @param attention_weights_v   attention weights tensor for Cv
 * @param compression_output_u  output tensor of attention-based channels C_u^{k+2}
 * @param compression_output_v  output tensor of attention-based channels C_v^{k+2}
 * @return __global__ 
 */
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


/**
 * @brief Responsible for starting the kernel for the maximum and average channels
 *        Allocates storage for the output tensor
 * 
 * @param img1_features_l0  image one input tensor
 * @param img2_features_lk  image two input tensor
 * @return std::vector<torch::Tensor>     vector containing two tensors with maximum and average 
 *                                        correlation volume channels C_u^{max,avg} and C_v^{max,avg}
 */
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
  
  // starts the optimized maximum and average kernel
  max_avg_forward_kernel_optimized_arch_indep <float> <<< blocks, threads >>> (
    img1_features_l0.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
    img2_features_lk.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
    max_avg_output_u.packed_accessor32<float,5,torch::RestrictPtrTraits>(),
    max_avg_output_v.packed_accessor32<float,5,torch::RestrictPtrTraits>()
  );

  // cudaError_t err = cudaThreadSynchronize();
  // printf("Run kernel: %s\n", cudaGetErrorString(err));

  return {max_avg_output_u, max_avg_output_v};
}

/**
 * @brief Responsible for starting the kernel for the maximum and average channels,
 *        including the argmax, i.e. the index of each maximum
 *        Allocates storage for the output tensor
 * 
 * @param img1_features_l0  image one input tensor
 * @param img2_features_lk  image two input tensor
 * @return std::vector<torch::Tensor>     vector containing two tensors with maximum and average 
 *                                        correlation volume channels C_u^{max,avg} and C_v^{max,avg}
 *                                        as well as C_u^{argmax} and C_v^{argmax}
 */
std::vector<torch::Tensor> max_argmax_avg_cuda_forward (
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

  auto optionsInt = torch::TensorOptions(opts).dtype(torch::kInt32);
  auto argmax_output_u = torch::zeros({B, H1, W1, 1, H2}, optionsInt);
  auto argmax_output_v = torch::zeros({B, H1, W1, 1, W2}, optionsInt);
  
  // number of blocks
  // shape: (batch, (ht+4-1)/4, (wd+8-1)/8) = (batch, ceil(ht/4), ceil(ht/8))
  const dim3 blocks(B, (H1+BLOCK_H-1)/BLOCK_H, (W1+BLOCK_W-1)/BLOCK_W);
  
  // number of threads per block
  // shape: (4,8)
  const dim3 threads(BLOCK_H, BLOCK_W);
  
  // starts the optimized maximum and average kernel with argmax
  max_argmax_avg_forward_kernel_optimized_arch_indep <float> <<< blocks, threads >>> (
    img1_features_l0.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
    img2_features_lk.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
    max_avg_output_u.packed_accessor32<float,5,torch::RestrictPtrTraits>(),
    max_avg_output_v.packed_accessor32<float,5,torch::RestrictPtrTraits>(),
    argmax_output_u.packed_accessor32<int32_t,5,torch::RestrictPtrTraits>(),
    argmax_output_v.packed_accessor32<int32_t,5,torch::RestrictPtrTraits>()
  );

  // cudaError_t err = cudaThreadSynchronize();
  // printf("Run kernel: %s\n", cudaGetErrorString(err));

  return {max_avg_output_u, max_avg_output_v, argmax_output_u, argmax_output_v};
}

/**
 * @brief Responsible for starting the kernel to compute the attention-based compression channels
 *        Allocates memory for the kernel output
 * 
 * @param img1_features_l0      image one input tensor 
 * @param img2_features_lk      image two input tensor 
 * @param attention_weights_u   Cu attention weights input tensor
 * @param attention_weights_v   Cv attention weights input tensor 
 * @return std::vector<torch::Tensor>     vector containing two tensors with the attention-based
 *                                        correlation volume channels C_u^{k+2} and C_v^{k+2}
 */
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
    
  // starts the optimized attention-based compression kernel
  compression_forward_kernel_optimized_arch_indep <float, K_VAL> <<< blocks, threads >>> (
    img1_features_l0.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
    img2_features_lk.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
    attention_weights_u.packed_accessor32<float,5,torch::RestrictPtrTraits>(),
    attention_weights_v.packed_accessor32<float,5,torch::RestrictPtrTraits>(),
    compression_output_u.packed_accessor32<float,5,torch::RestrictPtrTraits>(),
    compression_output_v.packed_accessor32<float,5,torch::RestrictPtrTraits>()
  );

  // cudaError_t err = cudaThreadSynchronize();
  // printf("Run kernel: %s\n", cudaGetErrorString(err));
  
  return {compression_output_u, compression_output_v};
}