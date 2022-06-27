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
from libs.MemorySaver.functions.MemorySaver import ComputeMaxAvgFunction, ComputeSelfCompressionFunction

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

def compute_compression_cuda(img1_features_l0 : torch.Tensor, img2_features_lk : torch.Tensor, attention_weights_u, attention_weights_v):
    adaptive_corr_u, adaptive_corr_v = ComputeSelfCompressionFunction.apply(img1_features_l0, img2_features_lk, attention_weights_u, attention_weights_v)
    
    adaptive_corr_u = adaptive_corr_u.permute(0,3,4,1,2)
    adaptive_corr_v = adaptive_corr_v.permute(0,3,4,1,2)

    return adaptive_corr_u, adaptive_corr_v

def test_example_same_size(batch_size, ht, wd, fdim, corr_channels):
    # shape: (batch, ht, wd, fdim)
    img1_features_l0 = torch.rand((batch_size, ht, wd, fdim)).cuda() * 5 - 2.5
    img2_features_l0 = torch.rand((batch_size, ht, wd, fdim)).cuda() * 5 - 2.5

    maxavg_u = torch.rand((batch_size, 2, ht, ht, wd)).cuda() * 5 - 2.5
    maxavg_v = torch.rand((batch_size, 2, wd, ht, wd)).cuda() * 5 - 2.5

    attention1 = torch.nn.Conv3d(2, corr_channels-2, 3, padding=1).cuda()
    attention2 = torch.nn.Conv3d(2, corr_channels-2, 3, padding=1).cuda()

    attention_weights_u = attention1(maxavg_v)
    attention_weights_v = attention2(maxavg_u)

    attention_weights_u = attention_weights_u.permute(0,3,4,1,2).contiguous()
    attention_weights_v = attention_weights_v.permute(0,3,4,1,2).contiguous()

    target_compression_u, target_compression_v = compute_compression_torch(
        img1_features_l0, img2_features_l0,
        attention_weights_u, attention_weights_v)

    compression_u, compression_v = compute_compression_cuda(
        img1_features_l0, img2_features_l0,
        attention_weights_u, attention_weights_v)

    shape_diff = {}
    for k in range(corr_channels-2):
        shape_diff.update({
            f"compression_u:{k}" : tensor_shape_eq(compression_u[:,k], target_compression_u[:,k]),
            f"compression_v:{k}" : tensor_shape_eq(compression_v[:,k], target_compression_v[:,k]),
        })
    
    value_diff = {}
    for k in range(corr_channels-2):
        value_diff.update({
            f"compression_u:{k}" : tensor_value_eq(compression_u[:,k], target_compression_u[:,k]),
            f"compression_v:{k}" : tensor_value_eq(compression_v[:,k], target_compression_v[:,k]),
        })
    
    return shape_diff, value_diff

def test_example_structured():

    batch_size, ht, wd, fdim = (1, 4, 8, 10)
    corr_channels = 4

    # shape: (batch, ht, wd, fdim)
    img1_features_l0 = torch.ones((batch_size, ht, wd, fdim)).cuda()
    img2_features_l0 = torch.ones((batch_size, ht, wd, fdim)).cuda()

    maxavg_u = torch.rand((batch_size, 2, ht, ht, wd)).cuda()
    maxavg_v = torch.rand((batch_size, 2, wd, ht, wd)).cuda()

    attention1 = torch.nn.Conv3d(2, corr_channels-2, 3, padding=1).cuda()
    attention2 = torch.nn.Conv3d(2, corr_channels-2, 3, padding=1).cuda()

    attention_weights_u = torch.ones_like(attention1(maxavg_v))
    attention_weights_v = torch.ones_like(attention2(maxavg_u))

    attention_weights_u = attention_weights_u.permute(0,3,4,1,2).contiguous()
    attention_weights_v = attention_weights_v.permute(0,3,4,1,2).contiguous()

    target_compression_u, target_compression_v = compute_compression_torch(
        img1_features_l0, img2_features_l0,
        attention_weights_u, attention_weights_v)

    compression_u, compression_v = compute_compression_cuda(
        img1_features_l0, img2_features_l0,
        attention_weights_u, attention_weights_v)

    print(compression_u)
    print(target_compression_u)
    print(compression_v)
    print(target_compression_v)

def test_example_diff_size(batch_size, ht, wd, fdim, corr_channels, level):
    # shape: (batch, ht, wd, fdim)
    img1_features_l0 = torch.rand((batch_size, ht, wd, fdim)).cuda() * 5 - 2.5
    img2_features_l0 = torch.rand((batch_size, ht, wd, fdim)).cuda() * 5 - 2.5

    img2_features_lk = img2_features_l0.permute(0,3,1,2)
    for _ in range(level):
        img2_features_lk = F.avg_pool2d(img2_features_lk, 2, stride=2)
    img2_features_lk = img2_features_lk.permute(0,2,3,1).contiguous()

    _, htl, wdl, _ = img2_features_lk.shape

    maxavg_u = torch.rand((batch_size, 2, htl, ht, wd)).cuda() * 5 - 2.5
    maxavg_v = torch.rand((batch_size, 2, wdl, ht, wd)).cuda() * 5 - 2.5

    attention1 = torch.nn.Conv3d(2, corr_channels-2, 3, padding=1).cuda()
    attention2 = torch.nn.Conv3d(2, corr_channels-2, 3, padding=1).cuda()

    attention_weights_u = attention1(maxavg_v)
    attention_weights_v = attention2(maxavg_u)

    attention_weights_u = attention_weights_u.permute(0,3,4,1,2).contiguous()
    attention_weights_v = attention_weights_v.permute(0,3,4,1,2).contiguous()

    target_compression_u, target_compression_v = compute_compression_torch(
        img1_features_l0, img2_features_l0,
        attention_weights_u, attention_weights_v,
        level=level)

    compression_u, compression_v = compute_compression_cuda(
        img1_features_l0, img2_features_lk,
        attention_weights_u, attention_weights_v)

    shape_diff = {}
    for k in range(corr_channels-2):
        shape_diff.update({
            f"compression_u:{k}" : tensor_shape_eq(compression_u[:,k], target_compression_u[:,k]),
            f"compression_v:{k}" : tensor_shape_eq(compression_v[:,k], target_compression_v[:,k]),
        })
    value_diff = {}
    for k in range(corr_channels-2):
        value_diff.update({
            f"compression_u:{k}" : tensor_value_eq(compression_u[:,k], target_compression_u[:,k]),
            f"compression_v:{k}" : tensor_value_eq(compression_v[:,k], target_compression_v[:,k]),
        })

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

def benchmark_compression(batch_size, ht, wd, fdim, corr_channels, num_threads=1, num_iters=10):

    results = []
    label = 'compress'
    sub_label = f'[{batch_size}, {ht}, {wd}, {fdim}]'

    with torch.no_grad():
        img1_features_l0 = torch.rand((batch_size, ht, wd, fdim)).cuda() * 5 - 2.5
        img2_features_l0 = torch.rand((batch_size, ht, wd, fdim)).cuda() * 5 - 2.5

        maxavg_u = torch.rand((batch_size, 2, ht, ht, wd)).cuda() * 5 - 2.5
        maxavg_v = torch.rand((batch_size, 2, wd, ht, wd)).cuda() * 5 - 2.5

        attention1 = torch.nn.Conv3d(2, corr_channels-2, 3, padding=1).cuda()
        attention2 = torch.nn.Conv3d(2, corr_channels-2, 3, padding=1).cuda()

        attention_weights_u = attention1(maxavg_v)
        attention_weights_v = attention2(maxavg_u)

        attention_weights_u = attention_weights_u.permute(0,3,4,1,2).contiguous()
        attention_weights_v = attention_weights_v.permute(0,3,4,1,2).contiguous()


        t0 = benchmark.Timer(
            stmt='compute_compression_torch(x, y, a_u, a_v)',
            setup='from __main__ import compute_compression_torch',
            globals={'x' : img1_features_l0, 'y' : img2_features_l0, 'a_u' : attention_weights_u, 'a_v' : attention_weights_v},
            num_threads=num_threads,
            label=label,
            sub_label=sub_label,
            description='torch',
            )

        # for batch_size=5, ht=200,wd=100,fdim=100
        # 9613MiB GPU memory usage of python process
        # 130.49 ms duration
        results.append(t0.timeit(num_iters))

        t1 = benchmark.Timer(
            stmt='compute_compression_cuda(x, y, a_u, a_v)',
            setup='from __main__ import compute_compression_cuda',
            globals={'x' : img1_features_l0, 'y' : img2_features_l0, 'a_u' : attention_weights_u, 'a_v' : attention_weights_v},
            num_threads=num_threads,
            label=label,
            sub_label=sub_label,
            description='cuda',
            )

        # batch_size=5, ht=200,wd=100,fdim=100
        # 1745MiB GPU memory usage of python process
        # 10.04 s duration: can be optimized
        results.append(t1.timeit(num_iters))

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
            results += benchmark_compression(b, *htwd, fdim, num_threads)
    compare = benchmark.Compare(results)
    compare.print()


if __name__ == "__main__":
    # run_comparison(10, (15,30), 20, 4, 4)
    benchmark.Compare(benchmark_compression(batch_size=5, ht=200, wd=100, fdim=100, corr_channels=4, num_iters=30)).print()
    #comparison_benchmark_maxavg()
    # test_example_structured()
    print(test_example_same_size(5, 117, 217, 128, 4))
    print(test_example_diff_size(3, 80, 160, 100, 4, 3))
    # random_same_shape_testing_maxavg(10)