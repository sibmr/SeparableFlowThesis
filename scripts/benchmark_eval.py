from pathlib import Path
import sys
import os
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
sys.path.append(os.path.join(path_root, "core"))

from PIL import Image
import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import datasets
from utils import flow_viz
from utils import frame_utils

from sepflow import SepFlow
from utils.utils import InputPadder, forward_interpolate

from tqdm import tqdm

import torch.utils.benchmark as benchmark

@torch.no_grad()
def validate_chairs(model, iters=24):
    """ Perform evaluation on the FlyingChairs (test) split """
    model.eval()
    epe_list = []

    val_dataset = datasets.FlyingChairs(split='validation')
    for val_id in tqdm(range(len(val_dataset))):
        image1, image2, flow_gt, _ = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        _, flow_pr = model(image1, image2, iters=iters)
        epe = torch.sum((flow_pr[0].cpu() - flow_gt)**2, dim=0).sqrt()
        epe_list.append(epe.view(-1).numpy())

    epe = np.mean(np.concatenate(epe_list))
    print("Validation Chairs EPE: %f" % epe)
    return {'chairs': epe}


def get_val_dataset(image_size, num_samples):
    val_dataset = []
    for i in range(num_samples):
        img1 = torch.rand(3,*image_size)
        img2 = torch.rand(3,*image_size)
        flow = torch.rand(2,*image_size)
        val_dataset.append((img1, img2, flow))
    return val_dataset

@torch.no_grad()
def manual_eval_speed(model, refinement_iters=32, eval_iters=100, image_size=(384, 512)):

    model.eval()
    results = []
    epe_list = []

    val_dataset = get_val_dataset(image_size, 1)
    val_id = 0
    num_threads=8
    
    image1, image2, flow_gt = val_dataset[val_id]
    image1 = image1[None].cuda()
    image2 = image2[None].cuda()

    padder = InputPadder(image1.shape)
    image1, image2 = padder.pad(image1, image2)

    print(f"img1: {image1.shape} img2: {image2.shape}")

    start = torch.cuda.Event(enable_timing=True)
    end =   torch.cuda.Event(enable_timing=True)

    start.record()
    for i in range(eval_iters):
        model(image1, image2, iters=refinement_iters)
    end.record()
    torch.cuda.synchronize()

    print(start.elapsed_time(end))

    elapsed_time = 0.0
    for i in range(eval_iters):
        start.record()
        model(image1, image2, iters=refinement_iters)
        end.record()
        torch.cuda.synchronize()

        elapsed_time += start.elapsed_time(end)
    print(elapsed_time) 



@torch.no_grad()
def profile_eval_speed(model, refinement_iters=32, eval_iters=100, image_size=(384, 512)):

    model.eval()
    results = []
    epe_list = []

    val_dataset = get_val_dataset(image_size, 1)
    val_id = 0
    num_threads=8
    
    image1, image2, flow_gt = val_dataset[val_id]
    image1 = image1[None].cuda()
    image2 = image2[None].cuda()

    padder = InputPadder(image1.shape)
    image1, image2 = padder.pad(image1, image2)

    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        for i in range(eval_iters):
            model(image1, image2, iters=refinement_iters)

    print(prof)



@torch.no_grad()
def benchmark_eval_speed(model, refinement_iters=32, eval_iters=100, image_size=(384, 512)):

    model.eval()
    results = []
    epe_list = []

    val_dataset = get_val_dataset(image_size, 1)
    val_id = 0
    num_threads=8
    
    image1, image2, flow_gt = val_dataset[val_id]
    image1 = image1[None].cuda()
    image2 = image2[None].cuda()

    padder = InputPadder(image1.shape)
    image1, image2 = padder.pad(image1, image2)


    # benchmark on the same random images and flow
    t0 = benchmark.Timer(
        stmt='flow_low, flow_pr = model(image1, image2, iters=refinement_iters)',
        setup='',
        globals={'model' : model, 'image1' : image1, 'image2' : image2, 'refinement_iters' : refinement_iters},
        num_threads=num_threads,
        label="model evaluation",
        sub_label=f"img:{str(image_size)}, refinement_iters:{refinement_iters}",
        description='torch',
    )

    results.append(t0.timeit(eval_iters))

    compare = benchmark.Compare(results)
    compare.print()



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--no_4d_corr', action='store_true', help='whether to use the 4d correlation volume directly')
    parser.add_argument('--num_corr_channels', type=int, default=2)
    parser.add_argument('--no_4d_agg', action='store_true', help='whether to use the 4d correlation volume directly')
    parser.add_argument('--use_gma', action='store_true', help='whether to use Global Motion Aggregation')
    parser.add_argument('--position_only', default=False, action='store_true',
                            help='only use position-wise attention')
    parser.add_argument('--position_and_content', default=False, action='store_true',
                        help='use position and content-wise attention')
    parser.add_argument('--num_heads', default=1, type=int,
                        help='number of heads in attention and aggregation')
    parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])
    parser.add_argument('--refinement_iters', default=32, type=int)
    parser.add_argument('--evaluation_iters', default=1000, type=int)
    args = parser.parse_args()

    model = torch.nn.DataParallel(SepFlow(args))
    print(f"number of parameters: {count_parameters(model)}")
    checkpoint = torch.load(args.model)
    msg = model.load_state_dict(checkpoint['state_dict'], strict=False)
    print(msg)

    model.cuda()
    model.eval()

    with torch.no_grad():
        manual_eval_speed(model, image_size=args.image_size, 
                eval_iters=args.evaluation_iters,
                refinement_iters=args.refinement_iters)
        profile_eval_speed(model, image_size=args.image_size, 
                eval_iters=args.evaluation_iters,
                refinement_iters=args.refinement_iters)
        benchmark_eval_speed(model, image_size=args.image_size, 
                eval_iters=args.evaluation_iters,
                refinement_iters=args.refinement_iters)
