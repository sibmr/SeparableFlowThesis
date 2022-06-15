//#include <torch/torch.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
//#include <torch/serialize/tensor.h>
//#include <ATen/ATen.h>
//#include <ATen/cuda/CUDAContext.h>

#define CUDA_NUM_THREADS 256 
#define THREADS_PER_BLOCK 64 

#define BLOCK_H 4
#define BLOCK_W 8
#define BLOCK_HW BLOCK_H * BLOCK_W
#define CHANNEL_STRIDE 32

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

  // global current x/y value for this block
  const int hc0 = h0 + threadIdx.x;
  const int wc0 = w0 + threadIdx.y;

  // image 1 feature at this threads pixel index
  // copy feature to local thread memory
  auto g1_ref = img1_features_l0[B][H1][W1];
  scalar_t * g1 = new scalar_t[C];
  for(int i = 0; i<C; ++i){
    g1[i] = g1_ref[i];
  }

  const scalar_t H2_inv = 1/H2;
  const scalar_t W2_inv = 1/W2;

  // local 
  scalar_t * tres_max_u = new scalar_t[H2];
  scalar_t * tres_avg_u = new scalar_t[H2];
  scalar_t * tres_max_v = new scalar_t[W2];
  scalar_t * tres_avg_v = new scalar_t[W2];
  
  for (int v = 0; v<W2; ++v){
    scalar_t max_v = -INFINITY;
    scalar_t avg_v = 0;
    for(int u = 0; u<H2; ++u){
      tres_max_u[u] = -INFINITY;
      tres_avg_u[u] = 0;
    }
    for(int u = 0; u<H2; ++u){
      // maybe this can be somehow shared between threads in block?
      // would require large block shared memory
      auto g2_ref = img2_features_lk[b][u][v];

      scalar_t cval = 0;
      for(int i = 0; i<C; ++i){
        cval += g1[i]*g2_ref[i];  
      }

      tres_max_u [u] = max(tres_max_u [u], cval);
      tres_avg_u [u] += cval;

      max_v = max(max_v, cval);
      avg_v += cval;

    }
    
    tres_max_v [v] = max_v;
    tres_avg_v [v] = avg_v;
  }

  // copy result from local memory to global output
  scalar_t* max_output_u_ptr = &max_avg_output_u[b][hc0][wc0][0][0];
  scalar_t* avg_output_u_ptr = &max_avg_output_u[b][hc0][wc0][1][0];
  for(int u = 0; u<H2; ++u){
    // *(max_output_u_ptr + u) = tres_max_u[u];
    // *(avg_output_u_ptr + u) = tres_avg_u[u]/H2;
    max_avg_output_u[b][hc0][wc0][0][u] = tres_max_u[u];
    max_avg_output_u[b][hc0][wc0][1][u] = tres_avg_u[u]/H2;
  }
  scalar_t* max_output_v_ref = &max_avg_output_v[b][hc0][wc0][0][0];
  scalar_t* avg_output_v_ref = &max_avg_output_v[b][hc0][wc0][1][0];
  for (int v = 0; v<W2; ++v){
    // max_output_v_ref[v] = tres_max_v[v];
    // avg_output_v_ref[v] = tres_avg_v[v]/W2;
    max_avg_output_v[b][hc0][wc0][0][v] = 3;//tres_max_v[v];
    max_avg_output_v[b][hc0][wc0][1][v] = 3;//tres_avg_v[v]/H2;
  }

  delete[] g1;
  delete[] tres_max_u;
  delete[] tres_avg_u;
  delete[] tres_max_v;
  delete[] tres_avg_v;

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
  AT_DISPATCH_FLOATING_TYPES(img1_features_l0.type(), "max_avg_forward_cuda", ([&] {
    max_avg_forward_kernel_unoptimized <scalar_t> <<< blocks, threads >>> (
      img1_features_l0.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
      img2_features_lk.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
      max_avg_output_u.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
      max_avg_output_v.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>()
    );
  }));

  return {max_avg_output_u, max_avg_output_v};
}




// #ifdef __cplusplus
//     }
// #endif
