"""
    File for: 
        - testing the correctness of the alternatvie separation forward and backward pass 
            by comparing the results to the standard forward and backward pass
        - measuring the memory consumption during training and inference
"""

from pathlib import Path
import sys
import os
from timeit import timeit
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
sys.path.append(os.path.join(path_root, "core"))

import torch
import torch.nn.functional as F
import torch.optim as optim

from sepflow import SepFlow

def sequence_loss(flow_preds, flow_gt, valid, gamma=0.8, max_flow=400):
    """ Loss function defined over sequence of flow predictions

    Args:
        flow_preds (list): list of flow predictions of shape (batch, 2, HT, WD)
        flow_gt (torch.Tensor): ground truth flow image of shape (batch, 2, HT, WD)
        valid (torch.Tensor):  whether flow_gt is valid at each pixel of shape (batch, HT, WD)
        gamma (float, optional): weight decay factor for flow refinement. Defaults to 0.8.
        max_flow (float, optional): threshold value for maximum flow value. Defaults to MAX_FLOW.

    Returns:
        Tuple[torch.Tensor,dict]: scalar tensor loss, flow quality metrics
    """

    n_predictions = len(flow_preds)    
    flow_loss = 0.0

    # set pixels where flow magnitude is too large as invalid
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)

    # in raft: weights[i] = gamma**(n_predictions-i), where i is the iteration of refinement
    # in this case: iteration 0,1 and 2 belong to inital flow regression
    #               refinement iterations start at 3
    #                       regression      refinement      
    # weights for n_img_refinement = 4:
    #   [0.1, 0.3, 0.5, 0.6024, 0.7304, 0.8904, 1.0903999999999998]
    weights = [0.1, 0.3, 0.5]
    base = weights[2] - gamma ** (n_predictions - 3)
    for i in range(n_predictions - 3):
        weights.append( base + gamma**(n_predictions - i - 4) )



    # loss based on weights array
    for i in range(n_predictions):
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += weights[i] * (valid[:, None] * i_loss).mean()

    # end point error calculation for the final flow output
    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]
    loss_value = flow_loss.detach()
    rate0 = (epe > 1).float().mean()
    rate1 = (epe > 3).float().mean()
    error3 = epe.mean()
    
    # end point error for calculation for initial flow regression images
    epe = torch.sum((flow_preds[1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]
    error1 = epe.mean()
    epe = torch.sum((flow_preds[0] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]
    error0 = epe.mean()
    epe = torch.sum((flow_preds[2] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]
    error2 = epe.mean()

    metrics = {
        'epe0': error0.item(),
        'epe1': error1.item(),
        'epe2': error2.item(),
        'epe3': error3.item(),
        '1px': rate0.item(),
        '3px': rate1.item(),
        'loss': loss_value.item()
    }
    return flow_loss, metrics

class DictObject:
    """ object to act as commandline args of SepFlow model
    """
    def __init__(self, **entries):
        self.__dict__.update(entries)
    def __iter__(self):
        return iter(self.__dict__)



def create_model(opt_args={}) -> torch.nn.Module:
    """ create a model with standard args with option to overwrite args

    Args:
        opt_args (Dict, optional): dict to overwrite/add arguments

    Return 
        torch.nn.Module: Separable Flow module with specified arguments
    """
    args = {
        "image_size" : [384,512],
        "resume" : "",
        "weights" : "",
        "batchSize" : 1,
        "testBatchSize" : 1,
        "lr" : 0.001,
        "threads" : 1,
        "manual_seed" : 1234,
        "shift" : 0,
        "gpu" : "0,1",
        "workers" : 16,
        "world_size" : 1,
        "rank" : 0,
        "dist_backend" : "nccl",
        "dist_url" : "tcp://127.0.0.1:6789",
        "distributed" : 0,
        "sync_bn" : 0,
        "multiprocessing_distributed" : 0,
        "freeze_bn" : 0,
        "start_epoch" : 0,
        "stage" : "chairs",
        "validation" : "",
        "num_steps" : 100000,
        "mixed_precision" : False,
        "iters" : 12,
        "wdecay" : .00005,
        "epsilon" : 1e-8,
        "clip" : 1.0,
        "dropout" : 0.0,
        "gamma" : 0.8,
        "add_noise" : False,
        "alternate_corr" : False,
        "alternate_corr_backward" : False,
        
        # added params
        "no_4d_corr" : False,
        "num_corr_channels" : 2,
        "no_4d_agg" : False,
        "use_gma" : False,
        "run_name" : "unnamed",
        "experiment_name" : "unnamed"
    }
    args.update(opt_args)
    
    # print("----------args going in----------")
    # print(args)

    model = SepFlow(DictObject(**args))

    # print("----------model args----------")
    # print(dict(model.args.__dict__))

    return model

def get_model(opt_args, state_dict=None):
    """ get model with sepecified commandline args and parameters

    Args:
        opt_args (Dict): arguments for model
        state_dict (Dict, optional): model parameters for initialization

    Return:
        torch.nn.Module:    fresh model according to args 
                            if state_dict not None: initialize model with specified parameters
    """
    model = create_model(opt_args)
    if state_dict is not None:
        model.load_state_dict(state_dict)

    return model

def get_forward_result(model : torch.nn.Module, img1 : torch.Tensor, img2 : torch.Tensor):
    return model(img1, img2)

def get_backward_result(model : torch.nn.Module, img1 : torch.Tensor, img2 : torch.Tensor):
    result = model(img1, img2)
    loss = F.mse_loss(result, torch.zeros_like(result))
    loss.backward()
    return img1.grad, img2.grad

def get_state_dict(opt_args):
    model = create_model(opt_args)
    state_dict = model.state_dict()
    return state_dict

def get_gradient(model, img1, img2, target, valid):
    img1 = img1.clone().detach()
    img2 = img2.clone().detach()
    target = target.clone().detach()
    valid = valid.clone().detach()

    img1.requires_grad = True
    img2.requires_grad = True

    flow_preds = model.forward(img1, img2)
    
    loss, metrics = sequence_loss(flow_preds, target, valid)

    loss.backward()

    return img1.grad, img2.grad

def do_optimization_step(model, img1, img2, target, valid):
    img1 = img1.clone().detach()
    img2 = img2.clone().detach()
    target = target.clone().detach()
    valid = valid.clone().detach()

    optimizer = torch.optim.SGD(model.parameters(), 0.1)
    optimizer.zero_grad()

    flow_preds = model.forward(img1, img2)

    loss, metrics = sequence_loss(flow_preds, target, valid)

    loss.backward()

    optimizer.step()



def dict_compare(dict1, dict2):
    if (type(dict1) == dict) and (type(dict2) == dict):
        shared_keys = []
        for key in dict1.keys():
            if not (key in dict2.keys()):
                print(f"missing key in dict2: {key}")
            else:
                shared_keys.append(key)
        for key in dict2.keys():
            if not (key in dict1.keys()):
                print(f"missing key in dict1: {key}")

        for key in shared_keys:
            dict_compare(dict1[key],dict2[key])
    else:
        print("at bottom:")
        print(dict1)
        print(dict2)

def compare_fn(t1 : torch.Tensor, t2 : torch.Tensor):
    abs_err = (t1-t2).abs().max()
    max_abs_val = torch.max(t1.abs().max(), t2.abs().max())
    rel_err = abs_err / max_abs_val
    
    return abs_err, rel_err

def list_compare(params1, params2, compare_grad=False):
    lparams1 = list(params1)
    lparams2 = list(params2)
    for i in range(len(lparams1)):
        t1 = lparams1[i]
        t2 = lparams2[i]

        if compare_grad:    
            t1 = t1.grad
            t2 = t2.grad
        
        abs_err, rel_err = compare_fn(t1, t2)
        
        print(f"abs {abs_err} rel {rel_err}")

def do_tests():
    """ Run tests to compare the standard and alternative model results for the
        gradients and parameters after optimization
    """

    opt_args_no_memsave = {
            "alternate_corr" : False,
            "alternate_corr_backward" : False,
            "no_4d_corr" : True,
            "num_corr_channels" : 4,
            "no_4d_agg" : True,
        }
    opt_args_memsave = {
            "alternate_corr_backward" : True,
            "no_4d_corr" : True,
            "num_corr_channels" : 4,
            "no_4d_agg" : True,
        }
    
    state_dict = get_state_dict(opt_args_no_memsave)

    batch, channels, ht, wd = (2,3,64*5,64*10)
    img1 = torch.rand((batch, channels, ht, wd)).cuda()
    img2 = torch.rand((batch, channels, ht, wd)).cuda()
    target = torch.rand((batch, 2, ht, wd)).cuda()
    valid = torch.full((batch, ht, wd), True).cuda()

    alternate_model = get_model(opt_args_memsave, state_dict).cuda().train()

    img1_grad, img2_grad = get_gradient(model, img1, img2, target, valid)
    img1_grad_alt, img2_grad_alt = get_gradient(alternate_model, img1, img2, target, valid)
    
    print("absolute maximum difference between gradients")
    abs_err, rel_err = compare_fn(img1_grad, img1_grad_alt)
    print(f"abs {abs_err} rel {rel_err}")
    abs_err, rel_err = compare_fn(img2_grad, img2_grad_alt)
    print(f"abs {abs_err} rel {rel_err}")

    print("compare fnet grad:")
    list_compare(model.fnet.parameters(), alternate_model.fnet.parameters(), compare_grad=True)

    do_optimization_step(model, img1, img2, target, valid)
    do_optimization_step(alternate_model, img1, img2, target, valid)

    print("compare fnet params after optimization:")
    list_compare(model.fnet.parameters(), alternate_model.fnet.parameters())

def measure_mem_fw(model, image_tensor_size, refine_iters=12):
    """ Measure and print the memory consumption of the model without backward pass and gradients
        enabled

    Args:
        model (torch.nn.Module): version of model to test
        img_tensor_size (Tuple): size of the input frame pair patches: (batch, 3, HT, WD)
        refine_iters (int, optional): number of refinement iterations
    """

    batch, channels, ht, wd = image_tensor_size
    img1 = torch.rand((batch, channels, ht, wd)).cuda()
    img2 = torch.rand((batch, channels, ht, wd)).cuda()
    target = torch.rand((batch, 2, ht, wd)).cuda()
    valid = torch.full((batch, ht, wd), True).cuda()

    model.eval()
    used_mem_list = []
    with torch.no_grad():
        loss = 0
        for i in range(10):
            low_flow, flow_pred = model(img1, img2, iters=refine_iters)
            loss += (flow_pred-target).abs().sum().detach().cpu().item()

            free_mem, total_mem = torch.cuda.mem_get_info()
            used_mem_list.append(total_mem-free_mem)

            torch.cuda.synchronize()

    print(f"min: {min(used_mem_list)/((1024)**3)} max: {max(used_mem_list)/((1024)**3)}")

def measure_mem_fw_bw(model, image_tensor_size, refine_iters=12):
    """ Measure and print the memory consumption of the model with backward pass and gradients
        enabled
    Args:
        model (torch.nn.Module): version of model to test
        img_tensor_size (Tuple): size of the input frame pair patches: (batch, 3, HT, WD)
        refine_iters (int, optional): number of refinement iterations
    """

    batch, channels, ht, wd = image_tensor_size
    img1 = torch.rand((batch, channels, ht, wd)).cuda()
    img2 = torch.rand((batch, channels, ht, wd)).cuda()
    target = torch.rand((batch, 2, ht, wd)).cuda()
    valid = torch.full((batch, ht, wd), True).cuda()

    model.train()
    used_mem_list = []
    for i in range(10):
        flow_preds = model(img1, img2, iters=refine_iters)
        loss, metrics = sequence_loss(flow_preds, target, valid)

        free_mem, total_mem = torch.cuda.mem_get_info()
        used_mem_list.append(total_mem-free_mem)

        loss.backward()

        free_mem, total_mem = torch.cuda.mem_get_info()
        used_mem_list.append(total_mem-free_mem)

        torch.cuda.synchronize()

    print(f"min: {min(used_mem_list)/((1024)**3)} max: {max(used_mem_list)/((1024)**3)}")


def measure_normal_fw(do_bw_pass=False, img_tensor_size=(12,3,320,448), refine_iters=12):
    """ Measure and print the memory consumption of the model without alternative separation

    Args:
        do_bw_pass (bool, optional): whether to do the backward pass and enable gradients
        img_tensor_size (Tuple): size of the input frame pair patches: (batch, 3, HT, WD)
        refine_iters (int, optional): number of refinement iterations
    """

    opt_args = {
        "alternate_corr" : False,
        "alternate_corr_backward" : False,
        "no_4d_corr" : True,
        "num_corr_channels" : 4,
        "no_4d_agg" : True,
        "iters" : refine_iters,
        }
    model = get_model(opt_args).cuda().train()
    if do_bw_pass:
        measure_mem_fw_bw(model, img_tensor_size, refine_iters=refine_iters)
    else:
        measure_mem_fw(model, img_tensor_size, refine_iters=refine_iters)


def measure_alt_fw(do_bw_pass=False, img_tensor_size=(12,3,320,448), refine_iters=12):
    """ Measure and print the memory consumption of the model with alternative separation

    Args:
        do_bw_pass (bool, optional): whether to do the backward pass and enable gradients
        img_tensor_size (Tuple): size of the input frame pair patches: (batch, 3, HT, WD)
        refine_iters (int, optional): number of refinement iterations
    """

    opt_args = {
        "alternate_corr_backward" : True,
        "no_4d_corr" : True,
        "num_corr_channels" : 4,
        "no_4d_agg" : True,
        "iters" : refine_iters,
        }
    model = get_model(opt_args).cuda().train()
    if do_bw_pass:
        measure_mem_fw_bw(model, img_tensor_size, refine_iters=refine_iters)
    else:
        measure_mem_fw(model, img_tensor_size, refine_iters=refine_iters)

def measure_all():
    print("12,3,320,448")
    measure_normal_fw(do_bw_pass=False, img_tensor_size=(12,3,320,448))
    torch.cuda.empty_cache()
    measure_alt_fw   (do_bw_pass=False, img_tensor_size=(12,3,320,448))
    torch.cuda.empty_cache()
    measure_normal_fw(do_bw_pass=True , img_tensor_size=(12,3,320,448))
    torch.cuda.empty_cache()
    measure_alt_fw   (do_bw_pass=True , img_tensor_size=(12,3,320,448))
    torch.cuda.empty_cache()

    print("1,3,512,1024")
    measure_normal_fw(do_bw_pass=False, img_tensor_size=(1,3,512,1024))
    torch.cuda.empty_cache()
    measure_alt_fw   (do_bw_pass=False, img_tensor_size=(1,3,512,1024))
    torch.cuda.empty_cache()
    measure_normal_fw(do_bw_pass=True , img_tensor_size=(1,3,512,1024))
    torch.cuda.empty_cache()
    measure_alt_fw   (do_bw_pass=True , img_tensor_size=(1,3,512,1024))
    torch.cuda.empty_cache()

    print("2,3,512,1024")
    measure_normal_fw(do_bw_pass=False, img_tensor_size=(2,3,512,1024))
    torch.cuda.empty_cache()
    measure_alt_fw   (do_bw_pass=False, img_tensor_size=(2,3,512,1024))
    torch.cuda.empty_cache()
    measure_normal_fw(do_bw_pass=True , img_tensor_size=(2,3,512,1024))
    torch.cuda.empty_cache()
    measure_alt_fw   (do_bw_pass=True , img_tensor_size=(2,3,512,1024))

def measure_inference_bs1():
    print("all inference batch size 1 - 12 iters")
    torch.cuda.empty_cache()
    measure_normal_fw(do_bw_pass=False, img_tensor_size=(1,3,320, 448))
    torch.cuda.empty_cache()
    measure_alt_fw   (do_bw_pass=False, img_tensor_size=(1,3,320, 448))
    torch.cuda.empty_cache()
    measure_normal_fw(do_bw_pass=False, img_tensor_size=(1,3,512,1024))
    torch.cuda.empty_cache()
    measure_alt_fw   (do_bw_pass=False, img_tensor_size=(1,3,512,1024))

    print("all inference batch size 1 - 32 iters")
    torch.cuda.empty_cache()
    measure_normal_fw(do_bw_pass=False, img_tensor_size=(1,3,320, 448), refine_iters=32)
    torch.cuda.empty_cache()
    measure_alt_fw   (do_bw_pass=False, img_tensor_size=(1,3,320, 448), refine_iters=32)
    torch.cuda.empty_cache()
    measure_normal_fw(do_bw_pass=False, img_tensor_size=(1,3,512,1024), refine_iters=32)
    torch.cuda.empty_cache()
    measure_alt_fw   (do_bw_pass=False, img_tensor_size=(1,3,512,1024), refine_iters=32)

def measure_more_refine_iters():
    print("1,3,512,1024")
    torch.cuda.empty_cache()
    measure_normal_fw(do_bw_pass=True , img_tensor_size=(1,3,512,1024),refine_iters=32)
    torch.cuda.empty_cache()
    measure_alt_fw   (do_bw_pass=True , img_tensor_size=(1,3,512,1024),refine_iters=32)
    torch.cuda.empty_cache()

def measure_only_needed_train():
    """
    measure memory results for table at tranining time (fw+bw)
    """
    torch.cuda.empty_cache()
    measure_normal_fw(do_bw_pass=True,  img_tensor_size=(12,3,320,448))
    torch.cuda.empty_cache()
    measure_alt_fw   (do_bw_pass=True,  img_tensor_size=(12,3,320,448))
    torch.cuda.empty_cache()
    measure_normal_fw(do_bw_pass=True,  img_tensor_size=( 1,3,512,1024))
    torch.cuda.empty_cache()
    measure_alt_fw   (do_bw_pass=True,  img_tensor_size=( 1,3,512,1024))
    torch.cuda.empty_cache()
    
def measure_only_needed_test():
    """
    measure memory results for table at inference time (fw)
    """
    torch.cuda.empty_cache()
    measure_normal_fw(do_bw_pass=False, img_tensor_size=( 1,3,320,448))
    torch.cuda.empty_cache()
    measure_alt_fw   (do_bw_pass=False, img_tensor_size=( 1,3,320,448))
    torch.cuda.empty_cache()
    measure_normal_fw(do_bw_pass=False, img_tensor_size=( 1,3,512,1024))
    torch.cuda.empty_cache()
    measure_alt_fw   (do_bw_pass=False, img_tensor_size=( 1,3,512,1024))
    torch.cuda.empty_cache()



if __name__ == "__main__":
    # do_tests()
    # measure_only_needed_train()
    measure_only_needed_test()
