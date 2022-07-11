#include <torch/extension.h>


// declarations of cuda functions to access from cpp

std::vector<torch::Tensor> max_avg_cuda_forward (
	at::Tensor img1_features_l0, 
	at::Tensor img2_features_lk);

std::vector<torch::Tensor> max_argmax_avg_cuda_forward (
	at::Tensor img1_features_l0, 
	at::Tensor img2_features_lk);

std::vector<torch::Tensor> compression_cuda_forward (
  at::Tensor img1_features_l0, 
  at::Tensor img2_features_lk,
  at::Tensor attention_weights_u,
  at::Tensor attention_weights_v);

std::vector<torch::Tensor> max_argmax_avg_cuda_backward (
	at::Tensor img1_features_l0, 
  at::Tensor img2_features_lk,
  at::Tensor argmax_output_u,
  at::Tensor argmax_output_v,
  at::Tensor grad_MaxAvg_u,
  at::Tensor grad_MaxAvg_v);

std::vector<torch::Tensor> compression_cuda_backward (
  at::Tensor img1_features_l0, 
  at::Tensor img2_features_lk,
  at::Tensor attention_weights_u,
  at::Tensor attention_weights_v,
  at::Tensor grad_compressed_output_u, 
  at::Tensor grad_compressed_output_v);