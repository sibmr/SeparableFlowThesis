"""
functions to verify the equations for forward and backward pass during alternative cost volume separation 
for the self adaptive compression attention-based channels 

by sketching a basic algorithm as a python program, 
the equations can be checked can be checked
by comparing results with the standard implementation
"""

from collections import namedtuple
from pathlib import Path
import sys
import os
from timeit import timeit
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
sys.path.append(os.path.join(path_root, "core"))

from typing import Callable
import torch
from torch.autograd import Function
import torch.nn.functional as F
from torch.utils import benchmark

from libs.MemorySaver.functions.MemorySaver import ComputeSelfCompressionFunction

class CUCompression(Function):
    @staticmethod
    def forward(ctx, fmap1_l0, fmap2_lk, a_u):
        batch, ht0, wd0, fdim = fmap1_l0.shape
        batch, htl, wdl, fdim = fmap2_lk.shape

        cucompressed = torch.zeros((batch, ht0, wd0, htl)).cuda()
        
        for b in range(batch):
            for i in range(ht0):
                for j in range(wd0):
                    for u in range(htl):
                        for v in range(wdl):
                            cval = fmap1_l0[b,i,j].dot(fmap2_lk[b,u,v])
                            cucompressed[b,i,j,u] += cval*a_u[b,i,j,0,v]

        ctx.save_for_backward(fmap1_l0, fmap2_lk, a_u, cucompressed)

        return cucompressed

    @staticmethod
    def backward(ctx, grad_output_cucompressed):
        fmap1_l0, fmap2_lk, a_u, cucompressed = ctx.saved_tensors
        batch, ht0, wd0, fdim = fmap1_l0.shape
        batch, htl, wdl, fdim = fmap2_lk.shape
        grad_fmap1_l0 = torch.zeros_like(fmap1_l0)
        grad_fmap2_lk = torch.zeros_like(fmap2_lk)
        grad_a_u = torch.zeros_like(a_u)

        for b in range(batch):
            for i in range(ht0):
                for j in range(wd0):
                    # accumulate gradient for each b,i,j
                    for u in range(htl):
                        for v in range(wdl):
                            grad_fmap1_l0[b,i,j] += fmap2_lk[b,u,v]*a_u[b,i,j,0,v]*grad_output_cucompressed[b,i,j,u]
        
        for b in range(batch):
            for u in range(htl):
                for v in range(wdl):
                    # accumulate gradient for each b,u,v
                    for i in range(ht0):
                        for j in range(wd0):
                            grad_fmap2_lk[b,u,v] += fmap1_l0[b,i,j]*a_u[b,i,j,0,v]*grad_output_cucompressed[b,i,j,u]
        
        for b in range(batch):
            for i in range(ht0):
                for j in range(wd0):
                    for v in range(wdl):
                        # accumulate gradient for each b,i,j,v
                        for u in range(htl):    
                            grad_a_u[b,i,j,0,v] += fmap1_l0[b,i,j].dot(fmap2_lk[b,u,v])*grad_output_cucompressed[b,i,j,u]

        return grad_fmap1_l0, grad_fmap2_lk, grad_a_u

class CVCompression(Function):
    @staticmethod
    def forward(ctx, fmap1_l0, fmap2_lk, a_v):
        batch, ht0, wd0, fdim = fmap1_l0.shape
        batch, htl, wdl, fdim = fmap2_lk.shape

        cvcompression = torch.zeros((batch, ht0, wd0, wdl)).cuda()

        for b in range(batch):
            for i in range(ht0):
                for j in range(wd0):
                    for u in range(htl):
                        for v in range(wdl):
                            cval = fmap1_l0[b,i,j].dot(fmap2_lk[b,u,v])
                            cvcompression[b,i,j,v] += cval*a_v[b,i,j,0,u]
                            
        ctx.save_for_backward(fmap1_l0, fmap2_lk, a_v, cvcompression)

        return cvcompression

    @staticmethod
    def backward(ctx, grad_output_cvcompressed):
        fmap1_l0, fmap2_lk, a_v, cvcompression = ctx.saved_tensors
        batch, ht0, wd0, fdim = fmap1_l0.shape
        batch, htl, wdl, fdim = fmap2_lk.shape
        grad_fmap1_l0 = torch.zeros_like(fmap1_l0)
        grad_fmap2_lk = torch.zeros_like(fmap2_lk)
        grad_a_v = torch.zeros_like(a_v)

        for b in range(batch):
            for i in range(ht0):
                for j in range(wd0):
                    # accumulate gradient for each b,i,j
                    for u in range(htl):
                        for v in range(wdl):
                            grad_fmap1_l0[b,i,j] += fmap2_lk[b,u,v]*a_v[b,i,j,0,u]*grad_output_cvcompressed[b,i,j,v]
        
        for b in range(batch):
            for u in range(htl):
                for v in range(wdl):
                    # accumulate gradient for each b,u,v
                    for i in range(ht0):
                        for j in range(wd0):
                            grad_fmap2_lk[b,u,v] += fmap1_l0[b,i,j]*a_v[b,i,j,0,u]*grad_output_cvcompressed[b,i,j,v]
        
        for b in range(batch):
            for i in range(ht0):
                for j in range(wd0):
                    for u in range(htl):    
                        # accumulate gradient for each b,i,j,u
                        for v in range(wdl):
                            grad_a_v[b,i,j,0,u] += fmap1_l0[b,i,j].dot(fmap2_lk[b,u,v])*grad_output_cvcompressed[b,i,j,v]

        return grad_fmap1_l0, grad_fmap2_lk, grad_a_v



def compute_compression_torch(img1_features_l0 : torch.Tensor, img2_features_l0 : torch.Tensor, attention_weights_u, attention_weights_v, level=0):
    batch_size, ht, wd, fdim = img1_features_l0.shape
    
    fmap1 = img1_features_l0.reshape(batch_size, ht*wd, fdim)
    fmap2 = img2_features_l0.reshape(batch_size, ht*wd, fdim)
    
    corr = torch.matmul(fmap1, fmap2.transpose(1,2))
    
    # downsample correlation volume to match specified pyramid level
    corr = corr.reshape(batch_size, ht*wd, ht, wd)

    for _ in range(level):
        corr = F.avg_pool2d(corr, 2, stride=2)

    _, _, htl, wdl = corr.shape

    corr = corr.reshape(batch_size, ht, wd, htl, wdl)

    # print(torch.stack(torch.where(corr[0]==1)).permute(1,0))

    # shape: (batch, ht, wd, ht/2**i, wd/2**i) -> (batch, ht, wd, 1, ht/2**i, wd/2**i)
    shaped_corr = corr.view((batch_size, ht, wd, 1, htl, wdl))
    
    # shape: (batch, ht, wd, 1, ht/2**i, wd/2**i) -> (batch, 1, ht/2**i, wd/2**i, ht, wd)
    shaped_corr = shaped_corr.permute((0,3,4,5,1,2))

    # shape: (batch, ht, wd, corr_channels-2, wd/2**i) -> (batch, corr_channels-2, wd/2**i, ht, wd)
    a_u = attention_weights_u.permute(0,3,4,1,2)

    # shape: (batch, corr_channels-2, wd/2**i, ht, wd) -> (batch, corr_channels-2, 1, wd/2**i, ht, wd)
    a_u = a_u.unsqueeze(dim=2)

    # apply softmax over v-dimension
    # a_u = a_u.softmax(dim=3)

    # shape: (batch, ht, wd, corr_channels-2, ht/2**i) -> (batch, corr_channels-2, ht/2**i, ht, wd)
    a_v = attention_weights_v.permute(0,3,4,1,2)
    
    # shape: (batch, corr_channels-2, ht/2**i, ht, wd) -> (batch, corr_channels-2, ht/2**i, 1, ht, wd)
    a_v = a_v.unsqueeze(dim=3)
    # a_v = a_v.softmax(dim=2)

    # shape:
    #       (batch, corr_channels-2,    ht/2**i,    1,          ht, wd)    attention
    #   *   (batch, 1,                  ht/2**i,    wd/2**i,    ht, wd)    4d correlation volume
    #   ->  (batch, corr_channels-2,                wd/2**i,    ht, wd)
    adaptive_corr_v = torch.einsum('bcuvij,bcuvij->bcvij',a_v, shaped_corr)
    
    # shape:
    #       (batch, corr_channels-2,    1,          wd/2**i, ht, wd)    attention
    #   *   (batch, 1,                  ht/2**i,    wd/2**i, ht, wd)    4d correlation volume
    #   ->  (batch, corr_channels-2,    ht/2**i,             ht, wd)    
    adaptive_corr_u = torch.einsum('bcuvij,bcuvij->bcuij',a_u, shaped_corr)

    return adaptive_corr_u, adaptive_corr_v

def pool_fmap_lk(fmap_l0, level):

    fmap_lk = fmap_l0.permute(0,3,1,2)
    for _ in range(level):
        fmap_lk = F.avg_pool2d(fmap_lk, 2, stride=2)
    fmap_lk = fmap_lk.permute(0,2,3,1).contiguous()

    return fmap_lk

def get_input(batch_size, ht, wd, fdim, val_low=-2, val_high=2):
    img1_features_l0 = torch.rand((batch_size, ht, wd, fdim)).cuda() * (val_high - val_low) + val_low
    img2_features_l0 = torch.rand((batch_size, ht, wd, fdim)).cuda() * (val_high - val_low) + val_low

    return img1_features_l0, img2_features_l0

def get_attention(batch_size, ht, wd, htl, wdl, K=1, val_low=-2, val_high=2):
    a_u = torch.rand((batch_size, ht, wd, K, wdl)).cuda() * (val_high - val_low) + val_low
    a_v = torch.rand((batch_size, ht, wd, K, htl)).cuda() * (val_high - val_low) + val_low

    return a_u, a_v

def loss_fn(prediction):
    target = torch.zeros_like(prediction)
    mse_loss = torch.nn.MSELoss()
    return mse_loss(prediction, target)

def get_gradient_fullMem(fmap1_l0 : torch.Tensor, fmap2_l0 : torch.Tensor, a_u : torch.Tensor, a_v : torch.Tensor, level : int, fn : Callable):
    
    fmap1_l0 = fmap1_l0.clone().detach()
    fmap2_l0 = fmap2_l0.clone().detach()
    a_u = a_u.clone().detach()
    a_v = a_v.clone().detach()

    fmap1_l0.requires_grad = True
    fmap2_l0.requires_grad = True
    a_u.requires_grad = True
    a_v.requires_grad = True

    target = fn(fmap1_l0, fmap2_l0, a_u, a_v, level)

    torch_loss = loss_fn(target)
    
    torch_loss.backward()

    return fmap1_l0.grad, fmap2_l0.grad, a_u.grad, a_v.grad

def get_gradient_lowMem(fmap1_l0 : torch.Tensor, fmap2_l0 : torch.Tensor, attention : torch.Tensor, level : int, fn : Callable):

    fmap1_l0 = fmap1_l0.clone().detach()
    fmap2_l0 = fmap2_l0.clone().detach()
    attention = attention.clone().detach()

    fmap1_l0.requires_grad = True
    fmap2_l0.requires_grad = True
    attention.requires_grad = True

    fmap2_lk = pool_fmap_lk(fmap2_l0, level)

    result = fn(fmap1_l0, fmap2_lk, attention)

    torch_loss = loss_fn(result)
    
    torch_loss.backward()

    return fmap1_l0.grad, fmap2_l0.grad, attention.grad

def test_cucompression(batch, ht, wd, fdim, level):
    batch, ht, wd, fdim = (2,5,10,7)
    level = 2

    fmap1_l0, fmap2_l0 = get_input(batch, ht, wd, fdim)

    htl = ht//2**level
    wdl = wd//2**level

    a_u, a_v = get_attention(batch, ht, wd, htl, wdl)
   
    grad_target_fmap1_l0, grad_target_fmap2_l0, grad_target_a_u, _ = get_gradient_fullMem(fmap1_l0, fmap2_l0, a_u, a_v, level,
        lambda fm1,fm2,au,av,lvl : compute_compression_torch(fm1,fm2,au,av,lvl)[0])
    grad_fmap1_l0, grad_fmap2_l0, grad_a_u = get_gradient_lowMem(fmap1_l0, fmap2_l0, a_u,level,
        lambda fm1,fm2,au : CUCompression.apply(fm1,fm2,au))
    
    print(grad_target_fmap1_l0.abs().max())
    print(grad_fmap1_l0.abs().max())
    print(grad_fmap2_l0.abs().max())
    print(grad_a_u.abs().max())
    print((grad_target_fmap1_l0 - grad_fmap1_l0).abs().max())
    print((grad_target_fmap2_l0 - grad_fmap2_l0).abs().max())
    print((grad_target_a_u - grad_a_u).abs().max())
    print((grad_target_fmap1_l0==grad_fmap1_l0).float().sum())

def test_cvcompression(batch, ht, wd, fdim, level):

    fmap1_l0, fmap2_l0 = get_input(batch, ht, wd, fdim)

    htl = ht//2**level
    wdl = wd//2**level

    a_u, a_v = get_attention(batch, ht, wd, htl, wdl)
   
    grad_target_fmap1_l0, grad_target_fmap2_l0, _, grad_target_a_v = get_gradient_fullMem(fmap1_l0, fmap2_l0, a_u, a_v, level,
        lambda fm1,fm2,au,av,lvl : compute_compression_torch(fm1,fm2,au,av,lvl)[1])
    grad_fmap1_l0, grad_fmap2_l0, grad_a_v = get_gradient_lowMem(fmap1_l0, fmap2_l0, a_v, level,
        lambda fm1,fm2,av: CVCompression.apply(fm1,fm2,av))
    
    print(grad_target_fmap1_l0.abs().max())
    print(grad_fmap1_l0.abs().max())
    print(grad_fmap2_l0.abs().max())
    print(grad_a_v.abs().max())
    print((grad_target_fmap1_l0 - grad_fmap1_l0).abs().max())
    print((grad_target_fmap2_l0 - grad_fmap2_l0).abs().max())
    print((grad_target_a_v - grad_a_v).abs().max())
    print((grad_target_fmap1_l0==grad_fmap1_l0).float().sum())

def check_forward_pass(batch, ht, wd, fdim, level):

    fmap1_l0, fmap2_l0 = get_input(batch, ht, wd, fdim, level)

    fmap2_lk = pool_fmap_lk(fmap2_l0, level)

    _, htl, wdl, _ = fmap2_lk.shape

    a_u, a_v = get_attention(batch, ht, wd, htl, wdl)
    
    with torch.no_grad():
        # shape: (batch, 1, htl, ht, wd), (batch, 1, wdl, ht, wd)
        target_compression_u, target_compression_v = compute_compression_torch(fmap1_l0, fmap2_l0, a_u, a_v, level)

        # shape: (batch, ht, wd, htl)
        pred_compression_u = CUCompression.apply(fmap1_l0, fmap2_lk, a_u).permute(0,3,1,2)[:,None]
        pred_compression_v = CVCompression.apply(fmap1_l0, fmap2_lk, a_v).permute(0,3,1,2)[:,None]

        print((target_compression_u-pred_compression_u).abs().max())
        print((target_compression_v-pred_compression_v).abs().max())

def gradcheck_cucompression(batch, ht, wd, fdim, level):

    fmap1_l0, fmap2_l0 = get_input(batch, ht, wd, fdim, level)
    fmap1_l0 = fmap1_l0.double()
    fmap2_l0 = fmap2_l0.double()

    fmap2_lk = pool_fmap_lk(fmap1_l0, level)

    _, htl, wdl, _ = fmap2_lk.shape

    a_u, a_v = get_attention(batch, ht, wd, htl, wdl)

    fmap1_l0.requires_grad = True
    fmap2_lk.requires_grad = True

    return torch.autograd.gradcheck(CUCompression.apply, (fmap1_l0, fmap2_lk, a_u), eps=1e-5)

def get_overall_grad_lowMem(fmap1_l0, fmap2_l0, a_u, a_v, level):
    fmap1_l0 = fmap1_l0.clone().detach()
    fmap2_l0 = fmap2_l0.clone().detach()
    a_u = a_u.clone().detach()
    a_v = a_v.clone().detach()

    fmap1_l0.requires_grad = True
    fmap2_l0.requires_grad = True
    a_u.requires_grad = True
    a_v.requires_grad = True

    fmap2_lk = pool_fmap_lk(fmap2_l0, level)

    lowMem_loss = torch.Tensor([0]).cuda()

    _, _, _, K, _ = a_u.shape

    for i in range(K):

        corr_u = CUCompression.apply(fmap1_l0, fmap2_lk, a_u[:,:,:,i:i+1,:])
    
        corr_v = CVCompression.apply(fmap1_l0, fmap2_lk, a_v[:,:,:,i:i+1,:])
    
        lowMem_loss += loss_fn(corr_u) + loss_fn(corr_v)

    lowMem_loss.backward()

    return fmap1_l0.grad, fmap2_l0.grad, a_u.grad, a_v.grad

def get_overall_grad_fullMem(fmap1_l0, fmap2_l0, a_u, a_v, level):
    fmap1_l0 = fmap1_l0.clone().detach()
    fmap2_l0 = fmap2_l0.clone().detach()
    a_u = a_u.clone().detach()
    a_v = a_v.clone().detach()

    fmap1_l0.requires_grad = True
    fmap2_l0.requires_grad = True
    a_u.requires_grad = True
    a_v.requires_grad = True

    corr_u, corr_v = compute_compression_torch(fmap1_l0, fmap2_l0, a_u, a_v, level)

    corr_u = corr_u.permute(0,3,4,1,2)
    corr_v = corr_v.permute(0,3,4,1,2)

    torch_loss = torch.Tensor([0]).cuda()

    _, _, _, K, _ = a_u.shape

    for i in range(K):

        torch_loss += loss_fn(corr_u[:,:,:,i]) + loss_fn(corr_v[:,:,:,i])
    
    torch_loss.backward()

    return fmap1_l0.grad, fmap2_l0.grad, a_u.grad, a_v.grad

def get_overall_grad_cuda(fmap1_l0, fmap2_l0, a_u, a_v, level):
    fmap1_l0 = fmap1_l0.clone().detach()
    fmap2_l0 = fmap2_l0.clone().detach()
    a_u = a_u.clone().detach()
    a_v = a_v.clone().detach()

    fmap1_l0.requires_grad = True
    fmap2_l0.requires_grad = True
    a_u.requires_grad = True
    a_v.requires_grad = True

    fmap2_lk = pool_fmap_lk(fmap2_l0, level)

    corr_u, corr_v = ComputeSelfCompressionFunction.apply(fmap1_l0, fmap2_lk, a_u, a_v)

    cuda_loss = torch.Tensor([0]).cuda()

    _, _, _, K, _ = a_u.shape

    for i in range(K):

        cuda_loss += loss_fn(corr_u[:,:,:,i]) + loss_fn(corr_v[:,:,:,i])
    
    cuda_loss.backward()

    return fmap1_l0.grad, fmap2_l0.grad, a_u.grad, a_v.grad

def torch_grad_comparison_test():
    batch, ht, wd, fdim = (2,8,8,5)
    for level in range(4):
        print(f"--------------(level {level})--------------")
        print("forward max difference:")
        check_forward_pass(batch, ht, wd, fdim, level)
        print("--------------cucompression--------------")
        test_cucompression(batch, ht, wd, fdim, level)
        print("--------------cvcompression--------------")
        test_cvcompression(batch, ht, wd, fdim, level)

def overall_grad_test(batch, ht, wd, fdim, lowMem=False):
    for level in range(4):
        
        fmap1_l0, fmap2_l0 = get_input(batch, ht, wd, fdim)

        htl = ht//2**level
        wdl = wd//2**level

        a_u, a_v = get_attention(batch, ht, wd, htl, wdl, K=2)
        
        grad_fmap1_fullMem, grad_fmap2_fullMem, grad_a_u_fullMem,   grad_a_v_fullMem    = get_overall_grad_fullMem  (fmap1_l0, fmap2_l0, a_u, a_v, level)
        if lowMem:
            grad_fmap1_lowMem,  grad_fmap2_lowMem,  grad_a_u_lowMem,    grad_a_v_lowMem     = get_overall_grad_lowMem   (fmap1_l0, fmap2_l0, a_u, a_v, level)
        grad_fmap1_cuda,    grad_fmap2_cuda,    grad_a_u_cuda,      grad_a_v_cuda       = get_overall_grad_cuda     (fmap1_l0, fmap2_l0, a_u, a_v, level)
        print(f"--------------(level {level})--------------")
        print(f"--------------loss values--------------")
        print(grad_fmap1_cuda.abs().max())
        if lowMem:
            print(grad_fmap1_lowMem.abs().max())
        print(grad_fmap1_fullMem.abs().max())
        print(grad_fmap2_cuda.abs().max())
        if lowMem:
            print(grad_fmap2_lowMem.abs().max())
        print(grad_fmap2_fullMem.abs().max())
        print(grad_a_u_cuda.abs().max())
        if lowMem:
            print(grad_a_u_lowMem.abs().max())
        print(grad_a_u_fullMem.abs().max())
        print(grad_a_v_cuda.abs().max())
        if lowMem:
            print(grad_a_v_lowMem.abs().max())
        print(grad_a_v_fullMem.abs().max())
        print(f"--------------loss differences--------------")
        print((grad_fmap1_cuda-grad_fmap1_fullMem).abs().max())
        print((grad_fmap2_cuda-grad_fmap2_fullMem).abs().max())
        if lowMem:
            print((grad_fmap1_lowMem-grad_fmap1_fullMem).abs().max())
            print((grad_fmap2_lowMem-grad_fmap2_fullMem).abs().max())
        print((grad_a_u_cuda-grad_a_u_fullMem).abs().max())
        print((grad_a_v_cuda-grad_a_v_fullMem).abs().max())
        if lowMem:
            print((grad_a_u_lowMem-grad_a_u_fullMem).abs().max())
            print((grad_a_v_lowMem-grad_a_v_fullMem).abs().max())

def benchmark_grad(batch, ht, wd, fdim, level, num_threads=1, label=None, iterations=10, lowMem=False, fullMem=False, cuda=False):
    fmap1_l0, fmap2_l0 = get_input(batch, ht, wd, fdim)

    htl = ht//2**level
    wdl = wd//2**level

    a_u, a_v = get_attention(batch, ht, wd, htl, wdl, K=2)

    sub_label = f"{(batch, ht, wd, fdim, level)}"
    t0 = benchmark.Timer(
        stmt='get_overall_grad_fullMem(fmap1_l0, fmap2_l0, a_u, a_v, lvl)',
        setup='from __main__ import get_overall_grad_fullMem',
        globals={'fmap1_l0' : fmap1_l0, 'fmap2_l0' : fmap2_l0, 'a_u' : a_u, 'a_v' : a_v, 'lvl' : level},
        num_threads=num_threads,
        label=label,
        sub_label=sub_label,
        description='fullMem',
        )
    t1 = benchmark.Timer(
        stmt='get_overall_grad_lowMem(fmap1_l0, fmap2_l0, a_u, a_v, lvl)',
        setup='from __main__ import get_overall_grad_lowMem',
        globals={'fmap1_l0' : fmap1_l0, 'fmap2_l0' : fmap2_l0, 'a_u' : a_u, 'a_v' : a_v, 'lvl' : level},
        num_threads=num_threads,
        label=label,
        sub_label=sub_label,
        description='lowMem',
        )
    t2 = benchmark.Timer(
        stmt='get_overall_grad_cuda(fmap1_l0, fmap2_l0, a_u, a_v, lvl)',
        setup='from __main__ import get_overall_grad_cuda',
        globals={'fmap1_l0' : fmap1_l0, 'fmap2_l0' : fmap2_l0, 'a_u' : a_u, 'a_v' : a_v, 'lvl' : level},
        num_threads=num_threads,
        label=label,
        sub_label=sub_label,
        description='cuda',
        )

    results = ([t0.timeit(iterations)] if fullMem else []) + ([t1.timeit(iterations)] if lowMem else []) + ([t2.timeit(iterations)] if cuda else [])

    return results

def debug_backward_custom_data(batch, ht, wd, fdim, level):
    htl = ht // 2**level
    wdl = wd // 2**level
    img1_features_l0    = torch.ones((batch, ht,  wd,  fdim)).cuda().contiguous()
    img2_features_lk    = torch.ones((batch, htl, wdl, fdim)).cuda().contiguous()
    attention_weights_u     = torch.ones((batch, ht, wd, 2, wdl)).cuda().contiguous()
    attention_weights_v     = torch.ones((batch, ht, wd, 2, htl)).cuda().contiguous()
    grad_compression_u       = torch.ones((batch, ht, wd, 2, htl)).cuda().contiguous()
    grad_compression_v       = torch.ones((batch, ht, wd, 2, wdl)).cuda().contiguous()

    # for i in range(wd):
    #     attention_weights_u[:,i%ht,i,:,:] = 0
    #     attention_weights_v[:,i%ht,i,:,:] = 0


    # img1_features_l0 = img1_features_l0.permute(0,2,3,1)
    # img1_features_l0[:,:,:,:] = torch.arange(ht)
    # img1_features_l0 = img1_features_l0.permute(0,3,1,2).cuda().contiguous()
    
    # grad_MaxAvg_u[:,:,:,:,:] = torch.arange(htl)
    # grad_MaxAvg_v[:,:,:,:,:] = torch.arange(wdl)

    # img2_features_lk[0,:,2,:] = 1
    # grad_MaxAvg_u[0, :, :, :, :] = 1
    # grad_MaxAvg_v[0, :, :, :, :] = 1
    
    ctx_dict = {"saved_tensors" : (img1_features_l0, img2_features_lk, attention_weights_u, attention_weights_v)}
    ctx = namedtuple("ctx", ctx_dict.keys())(*ctx_dict.values())
    grad_fmap1, grad_fmap2, grad_attention_u, grad_attention_v = ComputeSelfCompressionFunction.backward(ctx, grad_compression_u, grad_compression_v)
    print(grad_attention_v)
    print(grad_attention_v.shape)
    print(grad_attention_v[0,:,:,0,:])

if __name__ == "__main__":
    #torch_grad_comparison_test()
    overall_grad_test(batch=1,ht=128,wd=16,fdim=4)
    #debug_backward_custom_data(1,5,8,4,0)
    #benchmark.Compare(benchmark_grad(2,51,117,256,0, fullMem=True, cuda=True)).print()