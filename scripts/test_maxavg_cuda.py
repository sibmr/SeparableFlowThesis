from pathlib import Path
import sys
import os
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
sys.path.append(os.path.join(path_root, "core"))

import torch
import torch.nn.functional as F

from itertools import product
import torch.utils.benchmark as benchmark

from libs.MemorySaver.modules.MemorySaver import ComputeMaxAvgModule
from libs.MemorySaver.functions.MemorySaver import ComputeMaxAvgFunction

from core.corr import CorrBlock, CorrBlock1D

def tensor_value_eq(a : torch.Tensor, b : torch.Tensor) -> torch.Tensor:
    return (a-b).abs().max()

def tensor_shape_eq(a : torch.Tensor, b : torch.Tensor):
    return (a.shape, b.shape)

def run_comparison(batch_size, image_size, feature_size, num_levels, corr_channels):
    with torch.no_grad():
        img1_features_l0 = torch.rand((batch_size, feature_size, *image_size)).cuda()
        img2_features_l0 = torch.rand((batch_size, feature_size, *image_size)).cuda()

        attention1 = torch.nn.Conv3d(2, corr_channels-2, 3, padding=1).cuda()
        attention2 = torch.nn.Conv3d(2, corr_channels-2, 3, padding=1).cuda()
        attention_weights = (attention1, attention2)
        
        # compute correlation volume pyramid
        corr_fn = CorrBlock(img1_features_l0, img2_features_l0, guid=None)
        
        # compute correlation volumes
        corr1, corr2 = corr_fn(None, True, attention_weights)

        # compute image feature pyramid
        # shape at level i: ((batch, fdim, ht/2**i, wd/2**i),(batch, fdim, ht/2**i, wd/2**i))
        fmap1 = img1_features_l0    
        fmap2 = img2_features_l0
        pyramid = [(fmap1, fmap2)]
        for i in range(num_levels-1):
            fmap1 = F.avg_pool2d(fmap1, 2, stride=2)
            fmap2 = F.avg_pool2d(fmap2, 2, stride=2)
            pyramid.append((fmap1, fmap2))

        # compute 3d correlation volume on all levels of the 4d correlation volume pyramid
        batch, h1, w1, h2, w2 = corr_fn.shape
        sep_u_lvls = []
        sep_v_lvls = []
        for level in range(num_levels):
            
            img2_features_lk = pyramid[level][1]
            
            # shape: (batch, fdim, ht, wd) -> (batch, ht, wd, fdim)
            input_img1_features_l0 = img1_features_l0.permute(0,2,3,1).contiguous()
            input_img2_features_lk = img2_features_lk.permute(0,2,3,1).contiguous()

            # apply cuda function
            # shape: ((batch, ht0, wd0, 2, htl), (batch, ht0, wd0, 2, wdl))
            output_maxavg_u, output_maxavg_v = ComputeMaxAvgFunction.apply(input_img1_features_l0, input_img2_features_lk)
            
            # permute: ((batch, 2, htl, ht0, wd0), (batch, 2, wdl, ht0, wd0))
            # divide max and avg by the same factor as the 4d correlation volume is divided by
            maxavg_u = output_maxavg_u.permute(0,3,4,1,2) / torch.sqrt(torch.tensor(feature_size).float())
            maxavg_v = output_maxavg_v.permute(0,3,4,1,2) / torch.sqrt(torch.tensor(feature_size).float())
            
            a_u = attention_weights[0](maxavg_v)
            a_v = attention_weights[1](maxavg_u)

            # TODO: add self-adaptive correlation implementation
            # (batch, corr_channels-2,                wd/2**i,    ht, wd)
            adaptive_u = torch.zeros(batch, corr_channels-2, pyramid[level][1].shape[2], h1, w1).cuda()
            adaptive_v = torch.zeros(batch, corr_channels-2, pyramid[level][1].shape[3], h1, w1).cuda()

            print(maxavg_u.shape)
            print(maxavg_v.shape)
            print(adaptive_u.shape)
            print(adaptive_v.shape)
            sep_u = torch.cat((maxavg_u, adaptive_u), dim=1)
            sep_v = torch.cat((maxavg_v, adaptive_v), dim=1)

            sep_u = F.interpolate(sep_u, [w2, h1, w1], mode='trilinear', align_corners=True)
            sep_u_lvls.append(sep_u)
            sep_v = F.interpolate(sep_v, [h2, h1, w1], mode='trilinear', align_corners=True)
            sep_v_lvls.append(sep_v)
        
        corr1_memsave = torch.cat(sep_u_lvls, dim=1)
        corr2_memsave = torch.cat(sep_v_lvls, dim=1)

        diff_u = (corr1_memsave - corr1).abs().sum(dim=(0,2,3,4))
        diff_v = (corr2_memsave - corr2).abs().sum(dim=(0,2,3,4))

        print(diff_u)
        print(diff_v)

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

def compute_maxavg_cuda(img1_features_l0 : torch.Tensor, img2_features_lk : torch.Tensor):
    maxavg_u, maxavg_v = ComputeMaxAvgFunction.apply(img1_features_l0, img2_features_lk)
    
    maxavg_u = maxavg_u.permute(0,3,4,1,2)
    maxavg_v = maxavg_v.permute(0,3,4,1,2)
    max_u = maxavg_u[:,0]
    avg_u = maxavg_u[:,1]
    max_v = maxavg_v[:,0]
    avg_v = maxavg_v[:,1]
    return max_u, avg_u, max_v, avg_v

def test_example_same_size(batch_size, ht, wd, fdim):
    # shape: (batch, ht, wd, fdim)
    img1_features_l0 = torch.rand((batch_size, ht, wd, fdim)).cuda() * 5 - 2.5
    img2_features_l0 = torch.rand((batch_size, ht, wd, fdim)).cuda() * 5 - 2.5

    target_max_u, target_avg_u, target_max_v, target_avg_v = compute_maxavg_torch(img1_features_l0, img2_features_l0)

    max_u, avg_u, max_v, avg_v = compute_maxavg_cuda(img1_features_l0, img2_features_l0)

    shape_diff = {
        "max_u" : tensor_shape_eq(max_u, target_max_u),
        "avg_u" : tensor_shape_eq(avg_u, target_avg_u),
        "max_v" : tensor_shape_eq(max_v, target_max_v),
        "avg_v" : tensor_shape_eq(avg_v, target_avg_v),
    }   
    value_diff = {
        "max_u" : tensor_value_eq(max_u, target_max_u),
        "avg_u" : tensor_value_eq(avg_u, target_avg_u),
        "max_v" : tensor_value_eq(max_v, target_max_v),
        "avg_v" : tensor_value_eq(avg_v, target_avg_v),
    }

    return shape_diff, value_diff

def test_example_diff_size(batch_size, ht, wd, fdim, level):
    img1_features_l0 = torch.rand((batch_size, ht, wd, fdim)).cuda() * 5 - 2.5
    img2_features_l0 = torch.rand((batch_size, ht, wd, fdim)).cuda() * 5 - 2.5

    img2_features_lk = img2_features_l0.permute(0,3,1,2)
    for _ in range(level):
        img2_features_lk = F.avg_pool2d(img2_features_lk, 2, stride=2)
    img2_features_lk = img2_features_lk.permute(0,2,3,1).contiguous()

    target_max_u, target_avg_u, target_max_v, target_avg_v = compute_maxavg_torch(
        img1_features_l0, img2_features_l0, level=level)

    max_u, avg_u, max_v, avg_v = compute_maxavg_cuda(img1_features_l0, img2_features_lk)

    print(max_u.shape)
    print(target_max_u.shape)

    shape_diff = {
        "max_u" : tensor_shape_eq(max_u, target_max_u),
        "avg_u" : tensor_shape_eq(avg_u, target_avg_u),
        "max_v" : tensor_shape_eq(max_v, target_max_v),
        "avg_v" : tensor_shape_eq(avg_v, target_avg_v),
    }   
    value_diff = {
        "max_u" : tensor_value_eq(max_u, target_max_u),
        "avg_u" : tensor_value_eq(avg_u, target_avg_u),
        "max_v" : tensor_value_eq(max_v, target_max_v),
        "avg_v" : tensor_value_eq(avg_v, target_avg_v),
    }

    return shape_diff, value_diff

def random_same_shape_testing_maxavg(iterations):
    results = [
        test_example_same_size(
            batch_size =    torch.randint(1,  4,(1,)).item(),
            ht =            torch.randint(1,200,(1,)).item(),
            wd =            torch.randint(1,200,(1,)).item(),
            fdim =          torch.randint(1,300,(1,)).item()) for _ in range(iterations)]

    max_diff = 0
    for result in results:
        for key in result[1].keys():
            max_diff = max(max_diff, result[1][key].item())

    print(max_diff)

def benchmark_maxavg(batch_size, ht, wd, fdim, num_threads=1):

    results = []
    label = 'maxavg'
    sub_label = f'[{batch_size}, {ht}, {wd}, {fdim}]'

    with torch.no_grad():
        img1_features_l0 = torch.rand((batch_size, ht, wd, fdim)).cuda() * 5 - 2.5
        img2_features_l0 = torch.rand((batch_size, ht, wd, fdim)).cuda() * 5 - 2.5

        t0 = benchmark.Timer(
            stmt='compute_maxavg_torch(x, y)',
            setup='from __main__ import compute_maxavg_torch',
            globals={'x' : img1_features_l0, 'y' : img2_features_l0},
            num_threads=num_threads,
            label=label,
            sub_label=sub_label,
            description='torch',
            )

        # for batch_size=5, ht=200,wd=100,fdim=100
        # 9613MiB GPU memory usage of python process
        # 130.49 ms duration
        results.append(t0.timeit(10))

        t1 = benchmark.Timer(
            stmt='compute_maxavg_cuda(x, y)',
            setup='from __main__ import compute_maxavg_cuda',
            globals={'x' : img1_features_l0, 'y' : img2_features_l0},
            num_threads=num_threads,
            label=label,
            sub_label=sub_label,
            description='cuda',
            )

        # batch_size=5, ht=200,wd=100,fdim=100
        # 1745MiB GPU memory usage of python process
        # 10.04 s duration: can be optimized
        results.append(t1.timeit(10))

        return results

def comparison_benchmark_maxavg():

    # Compare takes a list of measurements which we'll save in results.
    results = []

    batch_sizes = [1,3]
    img_sizes = [(80,160),(160,80),(130,160)]
    fdim_sizes = [50,200]
    for b, htwd, fdim in product(batch_sizes, img_sizes, fdim_sizes):
        # label and sub_label are the rows
        # description is the column
        for num_threads in [1, 16]:
            results += benchmark_maxavg(b, *htwd, fdim, num_threads)
    compare = benchmark.Compare(results)
    compare.print()

    # [------------------ maxavg -----------------]
    #                         |  torch  |   cuda 
    # 1 threads: ----------------------------------
    #     [1, 80, 160, 50]    |    7.6  |   413.0
    #     [1, 80, 160, 200]   |    9.9  |   752.3
    #     [1, 160, 80, 50]    |    7.5  |   406.9
    #     [1, 160, 80, 200]   |   10.7  |   770.0
    #     [1, 130, 160, 50]   |   17.9  |  1265.4
    #     [1, 130, 160, 200]  |   25.5  |  2414.3
    #     [3, 80, 160, 50]    |   20.8  |  1633.8
    #     [3, 80, 160, 200]   |   39.9  |  3287.4
    #     [3, 160, 80, 50]    |   22.8  |  1537.7
    #     [3, 160, 80, 200]   |   42.3  |  3305.0
    #     [3, 130, 160, 50]   |   61.6  |  4820.8
    #     [3, 130, 160, 200]  |  116.3  |  9397.4
    # 16 threads: ---------------------------------
    #     [1, 80, 160, 50]    |    6.9  |   411.8
    #     [1, 80, 160, 200]   |   10.0  |   776.5
    #     [1, 160, 80, 50]    |    7.5  |   406.7
    #     [1, 160, 80, 200]   |   10.7  |   770.0
    #     [1, 130, 160, 50]   |   17.9  |  1264.0
    #     [1, 130, 160, 200]  |   25.8  |  2413.8
    #     [3, 80, 160, 50]    |   20.8  |  1632.1
    #     [3, 80, 160, 200]   |   39.9  |  3294.0
    #     [3, 160, 80, 50]    |   22.9  |  1541.6
    #     [3, 160, 80, 200]   |   42.3  |  3298.4
    #     [3, 130, 160, 50]   |   61.3  |  4817.2
    #     [3, 130, 160, 200]  |  116.3  |  9392.4

    # Times are in milliseconds (ms).


if __name__ == "__main__":
    # run_comparison(10, (15,30), 20, 4, 4)
    # benchmark.Compare(benchmark_maxavg(batch_size=5, ht=200, wd=100, fdim=100)).print()
    # comparison_benchmark_maxavg()
    print(test_example_same_size(5, 7, 14, 100))
    print(test_example_diff_size(3, 80, 160, 100, 3))
    # random_same_shape_testing_maxavg(10)