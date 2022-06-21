#include <torch/extension.h>

// #ifdef __cplusplus
//     extern "C" {
// #endif

// declarations of cuda functions to access from cpp

std::vector<torch::Tensor> max_avg_cuda_forward (
	at::Tensor img1_features_l0, 
	at::Tensor img2_features_lk);

std::vector<torch::Tensor> compression_cuda_forward (
  at::Tensor img1_features_l0, 
  at::Tensor img2_features_lk,
  at::Tensor attention_weights_u,
  at::Tensor attention_weights_v);


// #ifdef __cplusplus
//     }
// #endif
