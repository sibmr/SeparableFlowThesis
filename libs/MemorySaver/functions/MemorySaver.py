import torch
from torch.autograd import Function
from ..build.lib import MemorySaver
from torch.autograd import Variable

class ComputeMaxAvgFunction(Function):
    @staticmethod
    def forward(ctx, img1_features_l0, img2_features_lk):
        """ forward method for maximum and mean value computation of the 4d correlation volume (without storing it)

        Args:
            ctx (torch.autograd.FunctionCtx): context for backward pass
            img1_features_l0 (torch.Tensor): image1 feature tensor of shape (batch, ht0, wd0, fdim)
            img2_features_lk (torch.Tensor): image2 feature tensor of shape (batch, ht0/2**i, wd0/2**i, fdim)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: max and avg output for each pixel and u- or v-value
                shape: ((batch, ht0, wd0, 2, htl), (batch, ht0, wd0, 2, wdl))
        """
        assert(img1_features_l0.is_contiguous() == True and img2_features_lk.is_contiguous() == True)
        with torch.cuda.device_of(img1_features_l0):
            
            max_avg_output_u, max_avg_output_v = MemorySaver.max_avg_forward(img1_features_l0, img2_features_lk)
            
            max_avg_output_u = max_avg_output_u.contiguous()
            max_avg_output_v = max_avg_output_v.contiguous()
        
        return max_avg_output_u, max_avg_output_v
    
    @staticmethod
    def backward(ctx, gradOutput_u, gradOutput_v):
        """ backward not implemented, since argmax is not computed

        Raises:
            NotImplementedError: backward pass not implemented for max-avg-kernel
        """

        raise NotImplementedError("max avg backward pass is not implemented: use max argmax avg")

class ComputeMaxArgmaxAvgFunction(Function):
    @staticmethod
    def forward(ctx, img1_features_l0, img2_features_lk):
        """ forward method for maximum and mean value computation of the 4d correlation volume (without storing it)

        Args:
            ctx (torch.autograd.FunctionCtx): context for backward pass
            img1_features_l0 (torch.Tensor): image1 feature tensor of shape (batch, ht0, wd0, fdim)
            img2_features_lk (torch.Tensor): image2 feature tensor of shape (batch, ht0/2**i, wd0/2**i, fdim)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: max and avg output for each pixel and u- or v-value
                shape: ((batch, ht0, wd0, 2, htl), (batch, ht0, wd0, 2, wdl))
        """
        assert(bool(img1_features_l0.is_contiguous()) and bool(img2_features_lk.is_contiguous()))
        with torch.cuda.device_of(img1_features_l0):
            
            # run the kernel to compute max argmax avg
            max_avg_output_u, max_avg_output_v, argmax_output_u, argmax_output_v = MemorySaver.max_argmax_avg_forward(img1_features_l0, img2_features_lk)
            
            max_avg_output_u = max_avg_output_u.contiguous()
            max_avg_output_v = max_avg_output_v.contiguous()
            argmax_output_u = argmax_output_u.contiguous()
            argmax_output_v = argmax_output_v.contiguous()
        
        # save the image features and argmax for the backward pass
        ctx.save_for_backward(img1_features_l0, img2_features_lk, argmax_output_u, argmax_output_v)

        return max_avg_output_u, max_avg_output_v
    
    @staticmethod
    def backward(ctx, grad_MaxAvg_u, grad_MaxAvg_v):
        """ computes the contribution of C_u^{max,avg} and C_v^{max,avg} to the gradient 
            of the loss wrt fmap1_l0, fmap2_lk

        Args:
            ctx (object): stores inputs and outputs of the forward pass
            grad_MaxAvg_u (torch.Tensor): gradient of the loss wrt C_u^{max,avg} of shape (batch, ht0, wd0, 2, htl)
            grad_MaxAvg_v (torch.Tensor): gradient of the loss wrt C_v^{max,avg} of shape (batch, ht0, wd0, 2, wdl)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: gradient of the loss wrt to the image features fmap1_l0, fmap2_lk
                                                not full gradient, only contribution of C_u^{max,avg} and C_v^{max,avg}
                                                of shape (batch, ht0, wd0, fdim) and (batch, ht0//2**i, wd0//2**i, fdim)
        """

        # get the image feature tensors and argmax from the forward pass
        img1_features_l0, img2_features_lk, argmax_output_u, argmax_output_v = ctx.saved_tensors

        grad_MaxAvg_u = grad_MaxAvg_u.contiguous()
        grad_MaxAvg_v = grad_MaxAvg_v.contiguous()

        assert(bool(grad_MaxAvg_u.is_contiguous()) and bool(grad_MaxAvg_v.is_contiguous()))
        assert(bool(img1_features_l0.is_contiguous())       and bool(img2_features_lk.is_contiguous()   ))
        assert(bool(argmax_output_u.is_contiguous())    and bool(argmax_output_v.is_contiguous()))
        
        with torch.cuda.device_of(grad_MaxAvg_u):
            
            # run the max argmax avg backward pass kernel
            grad_fmap1_l0, grad_fmap2_lk = MemorySaver.max_argmax_avg_backward(img1_features_l0, img2_features_lk,
            argmax_output_u, argmax_output_v, grad_MaxAvg_u, grad_MaxAvg_v)
            
            grad_fmap1_l0 = grad_fmap1_l0.contiguous()
            grad_fmap2_lk = grad_fmap2_lk.contiguous()

        return grad_fmap1_l0, grad_fmap2_lk

class ComputeSelfCompressionFunction(Function):
    @staticmethod
    def forward(ctx, img1_features_l0, img2_features_lk, attention_weights_u, attention_weights_v):
        """ forward method for the self-compression of the 4d correlation volume (without storing it) using an attention weighted sum

        Args:
            ctx (torch.autograd.FunctionCtx): context for backward pass
            img1_features_l0 (torch.Tensor): image1 feature tensor of shape (batch, ht0, wd0, fdim)
            img2_features_lk (torch.Tensor): image2 feature tensor of shape (batch, ht0/2**i, wd0/2**i, fdim)
            attention_weights_u (torch.Tensor): weights for self-attention compression of C_u of shape (batch, ht0, wd0, K-2, wd0/2**i)
            attention_weights_v (torch.Tensor): weights for self-attention compression of C_v of shape (batch, ht0, wd0, K-2, ht0/2**i)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: max and avg output for each pixel and u- or v-value
                shape: ((batch, ht0, wd0, 2, htl), (batch, ht0, wd0, 2, wdl))
        """
        assert(bool(img1_features_l0.is_contiguous())       and bool(img2_features_lk.is_contiguous()   ))
        assert(bool(attention_weights_u.is_contiguous())    and bool(attention_weights_v.is_contiguous()))
        with torch.cuda.device_of(img1_features_l0):
            
            compressed_output_u, compressed_output_v = MemorySaver.compression_forward(
                img1_features_l0, img2_features_lk, attention_weights_u, attention_weights_v)
            
            compressed_output_u = compressed_output_u.contiguous()
            compressed_output_v = compressed_output_v.contiguous()
        
        ctx.save_for_backward(
            img1_features_l0, img2_features_lk, 
            attention_weights_u, attention_weights_v)
        
        return compressed_output_u, compressed_output_v
    
    @staticmethod
    def backward(ctx, grad_compressed_output_u, grad_compressed_output_v):
        """ computes the contribution of C_u^{3:K} and C_v^{3:K} 
            to the gradient of the loss wrt fmap1_l0, fmap2_lk, a_u, a_v

        Args:
            ctx (object): stores inputs and outputs of the forward pass
            grad_compressed_output_u (torch.Tensor): gradient of the loss wrt C_u^{3:K} of shape (batch, ht0, wd0, 2, htl)
            grad_compressed_output_v (torch.Tensor): gradient of the loss wrt C_u^{3:K} of shape (batch, ht0, wd0, 2, htl)

        Returns:
            Tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]: 
                gradInput1: gradient of the loss wrt fmap1_l0 (batch, ht0, wd0, fdim)
                gradInput2: gradient of the loss wrt fmap2_lk (batch, ht0//2**i, wd0//2**i, fdim)
                gradAttentionU: gradient of the loss wrt attention weights a_u of shape (batch, ht0, wd0, K-2, wd0//2**i)
                gradAttentionV: gradient of the loss wrt attention weights a_v of shape (batch, ht0, wd0, K-2, ht0//2**i)
        """
        img1_features_l0, img2_features_lk, attention_weights_u, attention_weights_v = ctx.saved_tensors
        
        grad_compressed_output_u = grad_compressed_output_u.contiguous()
        grad_compressed_output_v = grad_compressed_output_v.contiguous()
        
        assert(bool(grad_compressed_output_u.is_contiguous()) and bool(grad_compressed_output_v.is_contiguous()))
        assert(bool(img1_features_l0.is_contiguous())       and bool(img2_features_lk.is_contiguous()   ))
        assert(bool(attention_weights_u.is_contiguous())    and bool(attention_weights_v.is_contiguous()))

        with torch.cuda.device_of(grad_compressed_output_u):
            
            gradInput1, gradInput2, gradAttentionU, gradAttentionV = MemorySaver.compression_backward(
                img1_features_l0, img2_features_lk,
                attention_weights_u, attention_weights_v,
                grad_compressed_output_u, grad_compressed_output_v)
            
            gradInput1 = gradInput1.contiguous()
            gradInput2 = gradInput2.contiguous()
            gradAttentionU = gradAttentionU.contiguous()
            gradAttentionV = gradAttentionV.contiguous()

        return gradInput1, gradInput2, gradAttentionU, gradAttentionV
