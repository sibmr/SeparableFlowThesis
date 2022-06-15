import torch
from torch.autograd import Function
from ..build.lib import MemorySaver
from torch.autograd import Variable

class ComputeMaxAvgFunction(Function):
    @staticmethod
    def forward(ctx, img1_features_l0, img2_features_lk):
        """ forward method for 

        Args:
            ctx (torch.autograd.FunctionCtx): context for backward pass
            img1_features_l0 (torch.Tensor): image1 feature tensor of shape (batch, ht0, wd0, ht0, fdim)
            img2_features_lk (_type_): image2 feature tensor of shape (batch, ht0, wd0, ht0/2**i, fdim)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: max and avg output for each pixel and u- or v-value
                shape: ((batch, ht0, wd0, 2, htl), (batch, ht0, wd0, 2, wdl))
        """
        assert(img1_features_l0.is_contiguous() == True and img2_features_lk.is_contiguous() == True)
        with torch.cuda.device_of(img1_features_l0):
            
            max_avg_output_u, max_avg_output_v = MemorySaver.max_avg_forward(img1_features_l0, img2_features_lk)
            
            max_avg_output_u = max_avg_output_u.contiguous()
            max_avg_output_v = max_avg_output_v.contiguous()
        
        ctx.save_for_backward(img1_features_l0, img2_features_lk, max_avg_output_u, max_avg_output_v)
        
        return max_avg_output_u, max_avg_output_v
    
    @staticmethod
    def backward(ctx, gradOutput_u, gradOutput_v):
        img1_features_l0, img2_features_lk, max_avg_output_u, max_avg_output_v = ctx.saved_tensors
        assert(gradOutput_u.is_contiguous() == True and gradOutput_v.is_contiguous() == True)
        with torch.cuda.device_of(gradOutput_u):

            gradInput1 = img1_features_l0.new_zeros()
            gradInput2 = img2_features_lk.new_zeros()
            
            raise NotImplementedError("max avg backward pass is not implemented")
            MemorySaver.max_avg_forward(img1_features_l0, img2_features_lk, max_avg_output_u, max_avg_output_v, gradOutput_u, gradOutput_v, gradInput1, gradInput2)
            
            gradInput = gradInput.contiguous()

        return gradInput1, gradInput2
