"""
Functions to verify the equations for forward and backward pass during alternative cost volume separation 
for the maximum and average channels.

By sketching a basic algorithm as a python program, 
the equations can be checked can be checked
by comparing results with the standard implementation.
"""
from collections import namedtuple
from pathlib import Path
import sys
import os
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
sys.path.append(os.path.join(path_root, "core"))

from typing import Callable
import torch
from torch.autograd import Function
import torch.nn.functional as F
from torch.utils import benchmark

from libs.MemorySaver.functions.MemorySaver import ComputeMaxArgmaxAvgFunction

class CUMax(Function):
    @staticmethod
    def forward(ctx, fmap1_l0, fmap2_lk):
        batch, ht0, wd0, fdim = fmap1_l0.shape
        batch, htl, wdl, fdim = fmap2_lk.shape

        cumax = torch.full((batch, ht0, wd0, htl), -torch.inf).cuda()
        cuargmax = torch.zeros((batch, ht0, wd0, htl), dtype=torch.long).cuda()

        for b in range(batch):
            for i in range(ht0):
                for j in range(wd0):
                    for u in range(htl):
                        for v in range(wdl):
                            cval = fmap1_l0[b,i,j].dot(fmap2_lk[b,u,v])
                            if cumax[b,i,j,u] < cval:
                                cumax[b,i,j,u] = cval
                                cuargmax[b,i,j,u] = v

        ctx.save_for_backward(fmap1_l0, fmap2_lk, cumax, cuargmax)

        return cumax

    @staticmethod
    def backward(ctx, grad_output_cumax):
        fmap1_l0, fmap2_lk, cumax, cuargmax = ctx.saved_tensors
        batch, ht0, wd0, fdim = fmap1_l0.shape
        batch, htl, wdl, fdim = fmap2_lk.shape
        grad_fmap1_l0 = torch.zeros_like(fmap1_l0)
        grad_fmap2_lk = torch.zeros_like(fmap2_lk)

        for b in range(batch):
            for i in range(ht0):
                for j in range(wd0):
                    # accumulate gradient for each b,i,j
                    for u in range(htl):
                        argmax_corr_biju = cuargmax[b,i,j,u]
                        grad_fmap1_l0[b,i,j] += fmap2_lk[b,u,argmax_corr_biju]*grad_output_cumax[b,i,j,u]
        
        for b in range(batch):
            for u in range(htl):
                for v in range(wdl):
                    # accumulate gradient for each b,u,v
                    for i in range(ht0):
                        for j in range(wd0):
                            argmax_corr_biju = cuargmax[b,i,j,u]
                            if argmax_corr_biju == v:
                                grad_fmap2_lk[b,u,v] += fmap1_l0[b,i,j]*grad_output_cumax[b,i,j,u]
                                pass

        return grad_fmap1_l0, grad_fmap2_lk

class CUAvg(Function):
    @staticmethod
    def forward(ctx, fmap1_l0, fmap2_lk):
        batch, ht0, wd0, fdim = fmap1_l0.shape
        batch, htl, wdl, fdim = fmap2_lk.shape

        cuavg = torch.zeros((batch, ht0, wd0, htl)).cuda()

        for b in range(batch):
            for i in range(ht0):
                for j in range(wd0):
                    for u in range(htl):
                        for v in range(wdl):
                            cval = fmap1_l0[b,i,j].dot(fmap2_lk[b,u,v])
                            cuavg[b,i,j,u] += cval/wdl
                            
        ctx.save_for_backward(fmap1_l0, fmap2_lk, cuavg)

        return cuavg

    @staticmethod
    def backward(ctx, grad_output_cuavg):
        fmap1_l0, fmap2_lk, cuavg = ctx.saved_tensors
        batch, ht0, wd0, fdim = fmap1_l0.shape
        batch, htl, wdl, fdim = fmap2_lk.shape
        grad_fmap1_l0 = torch.zeros_like(fmap1_l0)
        grad_fmap2_lk = torch.zeros_like(fmap2_lk)

        for b in range(batch):
            for i in range(ht0):
                for j in range(wd0):
                    # accumulate gradient for each b,i,j
                    for u in range(htl):
                        for v in range(wdl):
                            grad_fmap1_l0[b,i,j] += (fmap2_lk[b,u,v]*grad_output_cuavg[b,i,j,u])/wdl
        
        for b in range(batch):
            for u in range(htl):
                for v in range(wdl):
                    # accumulate gradient for each b,u,v
                    for i in range(ht0):
                        for j in range(wd0):
                            grad_fmap2_lk[b,u,v] += (fmap1_l0[b,i,j]*grad_output_cuavg[b,i,j,u])/wdl

        return grad_fmap1_l0, grad_fmap2_lk


class CVMax(Function):
    @staticmethod
    def forward(ctx, fmap1_l0, fmap2_lk):
        batch, ht0, wd0, fdim = fmap1_l0.shape
        batch, htl, wdl, fdim = fmap2_lk.shape

        cvmax = torch.full((batch, ht0, wd0, wdl), -torch.inf).cuda()
        cvargmax = torch.zeros((batch, ht0, wd0, wdl), dtype=torch.long).cuda()

        for b in range(batch):
            for i in range(ht0):
                for j in range(wd0):
                    for u in range(htl):
                        for v in range(wdl):
                            cval = fmap1_l0[b,i,j].dot(fmap2_lk[b,u,v])
                            if cvmax[b,i,j,v] < cval:
                                cvmax[b,i,j,v] = cval
                                cvargmax[b,i,j,v] = u

        ctx.save_for_backward(fmap1_l0, fmap2_lk, cvmax, cvargmax)

        return cvmax

    @staticmethod
    def backward(ctx, grad_output_cvmax):
        fmap1_l0, fmap2_lk, cvmax, cvargmax = ctx.saved_tensors
        batch, ht0, wd0, fdim = fmap1_l0.shape
        batch, htl, wdl, fdim = fmap2_lk.shape
        grad_fmap1_l0 = torch.zeros_like(fmap1_l0)
        grad_fmap2_lk = torch.zeros_like(fmap2_lk)

        for b in range(batch):
            for i in range(ht0):
                for j in range(wd0):
                    # accumulate gradient for each b,i,j
                    for v in range(wdl):
                        argmax_corr_bijv = cvargmax[b,i,j,v]
                        grad_fmap1_l0[b,i,j] += fmap2_lk[b,argmax_corr_bijv,v]*grad_output_cvmax[b,i,j,v]
        
        for b in range(batch):
            for u in range(htl):
                for v in range(wdl):
                    # accumulate gradient for each b,u,v
                    for i in range(ht0):
                        for j in range(wd0):
                            argmax_corr_bijv = cvargmax[b,i,j,v]
                            if argmax_corr_bijv == u:
                                grad_fmap2_lk[b,u,v] += fmap1_l0[b,i,j]*grad_output_cvmax[b,i,j,v]
                                pass

        return grad_fmap1_l0, grad_fmap2_lk

class CVAvg(Function):
    @staticmethod
    def forward(ctx, fmap1_l0, fmap2_lk):
        batch, ht0, wd0, fdim = fmap1_l0.shape
        batch, htl, wdl, fdim = fmap2_lk.shape

        cvavg = torch.zeros((batch, ht0, wd0, wdl)).cuda()

        for b in range(batch):
            for i in range(ht0):
                for j in range(wd0):
                    for u in range(htl):
                        for v in range(wdl):
                            cval = fmap1_l0[b,i,j].dot(fmap2_lk[b,u,v])
                            cvavg[b,i,j,v] += cval/htl
                            
        ctx.save_for_backward(fmap1_l0, fmap2_lk, cvavg)

        return cvavg

    @staticmethod
    def backward(ctx, grad_output_cvavg):
        fmap1_l0, fmap2_lk, cuavg = ctx.saved_tensors
        batch, ht0, wd0, fdim = fmap1_l0.shape
        batch, htl, wdl, fdim = fmap2_lk.shape
        grad_fmap1_l0 = torch.zeros_like(fmap1_l0)
        grad_fmap2_lk = torch.zeros_like(fmap2_lk)

        for b in range(batch):
            for i in range(ht0):
                for j in range(wd0):
                    # accumulate gradient for each b,i,j
                    for u in range(htl):
                        for v in range(wdl):
                            grad_fmap1_l0[b,i,j] += (fmap2_lk[b,u,v]*grad_output_cvavg[b,i,j,v])/htl
        
        for b in range(batch):
            for u in range(htl):
                for v in range(wdl):
                    # accumulate gradient for each b,u,v
                    for i in range(ht0):
                        for j in range(wd0):
                            grad_fmap2_lk[b,u,v] += (fmap1_l0[b,i,j]*grad_output_cvavg[b,i,j,v])/htl

        return grad_fmap1_l0, grad_fmap2_lk


def compute_maxavg_torch(img1_features_l0 : torch.Tensor, img2_features_l0 : torch.Tensor, level=0):
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

    target_max_u, _ = corr.max(dim=4)
    target_max_u = target_max_u.permute(0,3,1,2)
    target_avg_u = corr.mean(dim=4).permute(0,3,1,2)
    
    target_max_v, _ = corr.max(dim=3)
    target_max_v = target_max_v.permute(0,3,1,2)
    target_avg_v = corr.mean(dim=3).permute(0,3,1,2)

    return target_max_u, target_avg_u, target_max_v, target_avg_v

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

def loss_fn(prediction):
    target = torch.zeros_like(prediction)
    mse_loss = torch.nn.MSELoss()
    return mse_loss(prediction, target)

def get_gradient_fullMem(fmap1_l0 : torch.Tensor, fmap2_l0 : torch.Tensor, level : int, fn : Callable):
    
    fmap1_l0 = fmap1_l0.clone().detach()
    fmap2_l0 = fmap2_l0.clone().detach()

    fmap1_l0.requires_grad = True
    fmap2_l0.requires_grad = True

    target = fn(fmap1_l0, fmap2_l0, level)

    torch_loss = loss_fn(target)
    
    torch_loss.backward()

    return fmap1_l0.grad, fmap2_l0.grad

def get_gradient_lowMem(fmap1_l0 : torch.Tensor, fmap2_l0 : torch.Tensor, level : int, fn : Callable):

    fmap1_l0 = fmap1_l0.clone().detach()
    fmap2_l0 = fmap2_l0.clone().detach()

    fmap1_l0.requires_grad = True
    fmap2_l0.requires_grad = True

    fmap2_lk = pool_fmap_lk(fmap2_l0, level)

    result = fn(fmap1_l0, fmap2_lk)

    torch_loss = loss_fn(result)
    
    torch_loss.backward()

    return fmap1_l0.grad, fmap2_l0.grad

def test_cumax(batch, ht, wd, fdim, level):

    fmap1_l0, fmap2_l0 = get_input(batch, ht, wd, fdim)
   
    grad_target_fmap1_l0, grad_target_fmap2_l0 = get_gradient_fullMem(fmap1_l0, fmap2_l0, level,
        lambda fm1,fm2,lvl : compute_maxavg_torch(fm1,fm2,lvl)[0])
    grad_fmap1_l0, grad_fmap2_l0 = get_gradient_lowMem(fmap1_l0, fmap2_l0, level,
        lambda fm1,fm2 : CUMax.apply(fm1,fm2))
    
    print(grad_target_fmap1_l0.abs().max())
    print(grad_fmap1_l0.abs().max())
    print((grad_target_fmap1_l0 - grad_fmap1_l0).abs().max())
    print((grad_target_fmap2_l0 - grad_fmap2_l0).abs().max())
    print((grad_target_fmap1_l0==grad_fmap1_l0).float().sum())

def test_cuavg(batch, ht, wd, fdim, level):

    fmap1_l0, fmap2_l0 = get_input(batch, ht, wd, fdim)
   
    grad_target_fmap1_l0, grad_target_fmap2_l0 = get_gradient_fullMem(fmap1_l0, fmap2_l0, level,
        lambda fm1,fm2,lvl : compute_maxavg_torch(fm1,fm2,lvl)[1])
    grad_fmap1_l0, grad_fmap2_l0 = get_gradient_lowMem(fmap1_l0, fmap2_l0, level,
        lambda fm1,fm2 : CUAvg.apply(fm1,fm2))
    
    print(grad_target_fmap1_l0.abs().max())
    print(grad_fmap1_l0.abs().max())
    print((grad_target_fmap1_l0 - grad_fmap1_l0).abs().max())
    print((grad_target_fmap2_l0 - grad_fmap2_l0).abs().max())
    print((grad_target_fmap1_l0==grad_fmap1_l0).float().sum())

def test_cvmax(batch, ht, wd, fdim, level):
    
    fmap1_l0, fmap2_l0 = get_input(batch, ht, wd, fdim)
   
    grad_target_fmap1_l0, grad_target_fmap2_l0 = get_gradient_fullMem(fmap1_l0, fmap2_l0, level,
        lambda fm1,fm2,lvl : compute_maxavg_torch(fm1,fm2,lvl)[2])
    grad_fmap1_l0, grad_fmap2_l0 = get_gradient_lowMem(fmap1_l0, fmap2_l0, level,
        lambda fm1,fm2 : CVMax.apply(fm1,fm2))
    
    print(grad_target_fmap1_l0.abs().max())
    print(grad_fmap1_l0.abs().max())
    print((grad_target_fmap1_l0 - grad_fmap1_l0).abs().max())
    print((grad_target_fmap2_l0 - grad_fmap2_l0).abs().max())
    print((grad_target_fmap1_l0==grad_fmap1_l0).float().sum())

def test_cvavg(batch, ht, wd, fdim, level):
    
    fmap1_l0, fmap2_l0 = get_input(batch, ht, wd, fdim)
   
    grad_target_fmap1_l0, grad_target_fmap2_l0 = get_gradient_fullMem(fmap1_l0, fmap2_l0, level,
        lambda fm1,fm2,lvl : compute_maxavg_torch(fm1,fm2,lvl)[3])
    grad_fmap1_l0, grad_fmap2_l0 = get_gradient_lowMem(fmap1_l0, fmap2_l0, level,
        lambda fm1,fm2 : CVAvg.apply(fm1,fm2))
    
    print(grad_target_fmap1_l0.abs().max())
    print(grad_fmap1_l0.abs().max())
    print((grad_target_fmap1_l0 - grad_fmap1_l0).abs().max())
    print((grad_target_fmap2_l0 - grad_fmap2_l0).abs().max())
    print((grad_target_fmap1_l0==grad_fmap1_l0).float().sum())

def check_forward_pass(batch, ht, wd, fdim, level):

    fmap1_l0, fmap2_l0 = get_input(batch, ht, wd, fdim, level)

    fmap2_lk = pool_fmap_lk(fmap2_l0, level)
    
    with torch.no_grad():
        target_max_u, target_avg_u, target_max_v, target_avg_v = compute_maxavg_torch(fmap1_l0, fmap2_l0, level)

        pred_max_u = CUMax.apply(fmap1_l0, fmap2_lk).permute(0,3,1,2)
        pred_avg_u = CUAvg.apply(fmap1_l0, fmap2_lk).permute(0,3,1,2)
        pred_max_v = CVMax.apply(fmap1_l0, fmap2_lk).permute(0,3,1,2)
        pred_avg_v = CVAvg.apply(fmap1_l0, fmap2_lk).permute(0,3,1,2)

        print((target_max_u-pred_max_u).abs().max())
        print((target_avg_u-pred_avg_u).abs().max())
        print((target_max_v-pred_max_v).abs().max())
        print((target_avg_v-pred_avg_v).abs().max())

def gradcheck_cumax():

    batch, ht, wd, fdim = (2,3,4,5)
    level = 0

    fmap1_l0, fmap2_l0 = get_input(batch, ht, wd, fdim, level)
    fmap1_l0 = fmap1_l0.double()
    fmap2_l0 = fmap2_l0.double()

    fmap2_lk = pool_fmap_lk(fmap1_l0, level)

    fmap1_l0.requires_grad = True
    fmap2_lk.requires_grad = True

    return torch.autograd.gradcheck(CUAvg.apply, (fmap1_l0, fmap2_lk))

def get_overall_grad_lowMem(fmap1_l0, fmap2_l0, level):
    fmap1_l0 = fmap1_l0.clone().detach()
    fmap2_l0 = fmap2_l0.clone().detach()

    fmap1_l0.requires_grad = True
    fmap2_l0.requires_grad = True

    fmap2_lk = pool_fmap_lk(fmap2_l0, level)

    umax = CUMax.apply(fmap1_l0, fmap2_lk)
    
    uavg = CUAvg.apply(fmap1_l0, fmap2_lk)
    
    vmax = CVMax.apply(fmap1_l0, fmap2_lk)

    vavg = CVAvg.apply(fmap1_l0, fmap2_lk)

    lowMem_loss = loss_fn(umax) + loss_fn(uavg) + loss_fn(vmax) + loss_fn(vavg)

    lowMem_loss.backward()

    return fmap1_l0.grad, fmap2_l0.grad

def get_overall_grad_fullMem(fmap1_l0, fmap2_l0, level):
    fmap1_l0 = fmap1_l0.clone().detach()
    fmap2_l0 = fmap2_l0.clone().detach()

    fmap1_l0.requires_grad = True
    fmap2_l0.requires_grad = True

    umax, uavg, vmax, vavg = compute_maxavg_torch(fmap1_l0, fmap2_l0, level)

    torch_loss = loss_fn(umax) + loss_fn(uavg) + loss_fn(vmax) + loss_fn(vavg)
    
    torch_loss.backward()

    return fmap1_l0.grad, fmap2_l0.grad

def get_overall_grad_cuda(fmap1_l0, fmap2_l0, level):
    fmap1_l0 = fmap1_l0.clone().detach()
    fmap2_l0 = fmap2_l0.clone().detach()

    fmap1_l0.requires_grad = True
    fmap2_l0.requires_grad = True

    fmap2_lk = pool_fmap_lk(fmap2_l0, level)

    corr_u, corr_v = ComputeMaxArgmaxAvgFunction.apply(fmap1_l0, fmap2_lk)

    umax = corr_u[:,:,:,0]
    uavg = corr_u[:,:,:,1]
    vmax = corr_v[:,:,:,0]
    vavg = corr_v[:,:,:,1]

    cuda_loss = loss_fn(umax) + loss_fn(uavg) + loss_fn(vmax) + loss_fn(vavg)
    
    cuda_loss.backward()

    return fmap1_l0.grad, fmap2_l0.grad
    

def torch_grad_comparison_test():
    batch, ht, wd, fdim = (2,8,8,7)
    for level in range(4):
        print(f"--------------(level {level})--------------")
        print("forward max difference:")
        check_forward_pass(batch, ht, wd, fdim, level)
        print("--------------cumax--------------")
        test_cumax(batch, ht, wd, fdim, level)
        print("--------------cuavg--------------")
        test_cuavg(batch, ht, wd, fdim, level)
        print("--------------cvmax--------------")
        test_cvmax(batch, ht, wd, fdim, level)
        print("--------------cvavg--------------")
        test_cvavg(batch, ht, wd, fdim, level)

def overall_grad_test(batch, ht, wd, fdim, lowMem=False):
    for level in range(4):
        
        fmap1_l0, fmap2_l0 = get_input(batch, ht, wd, fdim)
        
        grad_fmap1_fullMem, grad_fmap2_fullMem = get_overall_grad_fullMem(fmap1_l0, fmap2_l0, level)
        if lowMem:
            grad_fmap1_lowMem, grad_fmap2_lowMem = get_overall_grad_lowMem(fmap1_l0, fmap2_l0, level)
        grad_fmap1_cuda, grad_fmap2_cuda = get_overall_grad_cuda(fmap1_l0, fmap2_l0, level)
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
        print(f"--------------loss differences--------------")
        print((grad_fmap1_cuda-grad_fmap1_fullMem).abs().max())
        print((grad_fmap2_cuda-grad_fmap2_fullMem).abs().max())
        if lowMem:
            print((grad_fmap1_lowMem-grad_fmap1_fullMem).abs().max())
            print((grad_fmap2_lowMem-grad_fmap2_fullMem).abs().max())

def benchmark_grad(batch, ht, wd, fdim, level, num_threads=1, label=None, iterations=10, lowMem=False, fullMem=False, cuda=False):
    fmap1_l0, fmap2_l0 = get_input(batch, ht, wd, fdim)
    sub_label = f"{(batch, ht, wd, fdim, level)}"
    t0 = benchmark.Timer(
        stmt='get_overall_grad_fullMem(fmap1_l0, fmap2_l0, lvl)',
        setup='from __main__ import get_overall_grad_fullMem',
        globals={'fmap1_l0' : fmap1_l0, 'fmap2_l0' : fmap2_l0, 'lvl' : level},
        num_threads=num_threads,
        label=label,
        sub_label=sub_label,
        description='fullMem',
        )
    t1 = benchmark.Timer(
        stmt='get_overall_grad_lowMem(fmap1_l0, fmap2_l0, lvl)',
        setup='from __main__ import get_overall_grad_lowMem',
        globals={'fmap1_l0' : fmap1_l0, 'fmap2_l0' : fmap2_l0, 'lvl' : level},
        num_threads=num_threads,
        label=label,
        sub_label=sub_label,
        description='lowMem',
        )
    t2 = benchmark.Timer(
        stmt='get_overall_grad_cuda(fmap1_l0, fmap2_l0, lvl)',
        setup='from __main__ import get_overall_grad_cuda',
        globals={'fmap1_l0' : fmap1_l0, 'fmap2_l0' : fmap2_l0, 'lvl' : level},
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
    argmax_output_u     = torch.ones((batch, ht, wd, 1, htl), dtype=torch.int32).cuda().contiguous()
    argmax_output_v     = torch.ones((batch, ht, wd, 1, wdl), dtype=torch.int32).cuda().contiguous()
    grad_MaxAvg_u       = torch.ones((batch, ht, wd, 2, htl)).cuda().contiguous()
    grad_MaxAvg_v       = torch.ones((batch, ht, wd, 2, wdl)).cuda().contiguous()

    img1_features_l0 = img1_features_l0.permute(0,2,3,1)
    img1_features_l0[:,:,:,:] = torch.arange(ht)
    img1_features_l0 = img1_features_l0.permute(0,3,1,2).cuda().contiguous()
    
    grad_MaxAvg_u[:,:,:,:,:] = torch.arange(htl)
    grad_MaxAvg_v[:,:,:,:,:] = torch.arange(wdl)

    # img2_features_lk[0,:,2,:] = 1
    # grad_MaxAvg_u[0, :, :, :, :] = 1
    # grad_MaxAvg_v[0, :, :, :, :] = 1
    
    ctx_dict = {"saved_tensors" : (img1_features_l0, img2_features_lk, argmax_output_u, argmax_output_v)}
    ctx = namedtuple("ctx", ctx_dict.keys())(*ctx_dict.values())
    grad_fmap1, grad_fmap2 = ComputeMaxArgmaxAvgFunction.backward(ctx, grad_MaxAvg_u, grad_MaxAvg_v)
    print(grad_fmap2)


if __name__ == "__main__":
    # torch_grad_comparison_test()
    
    # debug_backward_custom_data(batch=1, ht=15, wd=15, fdim=3, level=0)
    overall_grad_test(batch=3, ht=61, wd=121, fdim=117)

    benchmark.Compare(
            benchmark_grad(2,61,121,256,0, label="forward+backward", iterations=10, fullMem=True, cuda=True)).print()

    # benchmark.Compare(
    #         benchmark_grad(2,100,200,256,0, label="forward+backward", iterations=10, cuda=True)
    #     +   benchmark_grad(2,100,200,256,1, label="forward+backward", iterations=10, cuda=True)
    #     +   benchmark_grad(2,100,200,256,2, label="forward+backward", iterations=10, cuda=True)
    #     +   benchmark_grad(2,100,200,256,3, label="forward+backward", iterations=10, cuda=True)).print()