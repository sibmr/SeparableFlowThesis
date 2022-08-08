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
    def __init__(self, **entries):
        self.__dict__.update(entries)
    def __iter__(self):
        return iter(self.__dict__)



def create_model(opt_args={}) -> torch.nn.Module:
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

    optimizer = torch.optim.SGD(model.parameters(), 0.01)
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

def list_compare(params1, params2):
    lparams1 = list(params1)
    lparams2 = list(params2)
    for i in range(len(lparams1)):
        t1 = lparams1[i]
        t2 = lparams2[i]
        abs_err = (t1-t2).abs().max()
        max_abs_val = torch.max(t1.abs().max(), t2.abs().max())
        rel_err = abs_err / max_abs_val
        print(f"abs {abs_err} rel {rel_err}")

if __name__ == "__main__":
    
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

    model = get_model(opt_args_no_memsave, state_dict).cuda().train()
    alternate_model = get_model(opt_args_memsave, state_dict).cuda().train()

    img1_grad, img2_grad = get_gradient(model, img1, img2, target, valid)
    img1_grad_alt, img2_grad_alt = get_gradient(alternate_model, img1, img2, target, valid)
    
    print(img1_grad.abs().max())
    print(img1_grad_alt.abs().max())
    print(img2_grad.abs().max())
    print(img2_grad_alt.abs().max())
    print("absolute maximum difference between gradients")
    print((img1_grad-img1_grad_alt).abs().max())
    print((img2_grad-img2_grad_alt).abs().max())


    do_optimization_step(model, img1, img2, target, valid)
    do_optimization_step(alternate_model, img1, img2, target, valid)

    print("compare fnet params:")
    list_compare(model.fnet.parameters(), alternate_model.fnet.parameters())