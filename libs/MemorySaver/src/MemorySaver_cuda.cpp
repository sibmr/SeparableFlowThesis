//#include <torch/torch.h>
#include <torch/extension.h>
#include "MemorySaver_kernel.h"

// C++ interface
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> max_avg_forward (
  at::Tensor img1_features_l0, 
  at::Tensor img2_features_lk)
{

  CHECK_INPUT(img1_features_l0);
  CHECK_INPUT(img2_features_lk);

  return max_avg_cuda_forward(
    img1_features_l0,
    img2_features_lk);
}

std::vector<torch::Tensor> max_argmax_avg_forward (
  at::Tensor img1_features_l0, 
  at::Tensor img2_features_lk)
{

  CHECK_INPUT(img1_features_l0);
  CHECK_INPUT(img2_features_lk);

  return max_argmax_avg_cuda_forward(
    img1_features_l0,
    img2_features_lk);
}

std::vector<torch::Tensor> compression_forward (
  at::Tensor img1_features_l0, 
  at::Tensor img2_features_lk,
  at::Tensor attention_weights_u,
  at::Tensor attention_weights_v)
{

  CHECK_INPUT(img1_features_l0);
  CHECK_INPUT(img2_features_lk);
  CHECK_INPUT(attention_weights_u);
  CHECK_INPUT(attention_weights_v);

  return compression_cuda_forward(
    img1_features_l0,
    img2_features_lk,
    attention_weights_u,
    attention_weights_v);
}

PYBIND11_MODULE (TORCH_EXTENSION_NAME, MemorySaver)
{
  MemorySaver.def ("max_avg_forward", &max_avg_forward, "max avg forward (CUDA)");
  MemorySaver.def ("max_argmax_avg_forward", &max_argmax_avg_forward, "max argmax avg forward (CUDA)");
  MemorySaver.def ("compression_forward", &compression_forward, "compression forward (CUDA)");
}

