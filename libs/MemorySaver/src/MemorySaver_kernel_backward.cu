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

std::vector<torch::Tensor> max_avg_cuda_backward (
        at::Tensor img1_features_l0, 
        at::Tensor img2_features_lk){

  return {};
}

std::vector<torch::Tensor> max_argmax_avg_cuda_backward (
        at::Tensor img1_features_l0, 
        at::Tensor img2_features_lk){

  return {};
}