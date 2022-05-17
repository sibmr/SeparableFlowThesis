from turtle import color
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

def fetch_optimizer(args):
    """ Create the optimizer and learning rate scheduler

    Args:
        args (argparse.Namespace): model parameters
        model (torch.nn.Module): model to be trained

    Returns:
        Tuple[torch.optim.AdamW, torch.optim.lr_scheduler.OneCycleLR]: create optimizer
            and learning rate scheduler for the model
    """
    modules_ori = [torch.nn.Linear(2,2),torch.nn.Linear(2,2),torch.nn.Linear(2,2),
                    torch.nn.Linear(2,2),torch.nn.Linear(2,2),torch.nn.Linear(2,2)]

    modules_new = [torch.nn.Linear(2,2),torch.nn.Linear(2,2)]
    params_list = []
    for module in modules_ori:
        params_list.append(dict(params=module.parameters(), lr=args.lr))
    for module in modules_new:
        # TODO: this seems to do the same as adjust_learning_rate(), why duplicate this?
        params_list.append(dict(params=module.parameters(), lr=args.lr * 2.5))
    #optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)
    optimizer = optim.AdamW(params_list, lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)


    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
        pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler

def fetch_optimizer_alt(args):
    """ Create the optimizer and learning rate scheduler (added argument list)

    Args:
        args (argparse.Namespace): model parameters
        model (torch.nn.Module): model to be trained

    Returns:
        Tuple[torch.optim.AdamW, torch.optim.lr_scheduler.OneCycleLR]: create optimizer
            and learning rate scheduler for the model
    """
    modules_ori = [torch.nn.Linear(2,2),torch.nn.Linear(2,2),torch.nn.Linear(2,2),
                    torch.nn.Linear(2,2),torch.nn.Linear(2,2),torch.nn.Linear(2,2)]

    modules_new = [torch.nn.Linear(2,2),torch.nn.Linear(2,2)]
    params_list = []
    for module in modules_ori:
        params_list.append(dict(params=module.parameters(), lr=args.lr))
    for module in modules_new:
        # TODO: this seems to do the same as adjust_learning_rate(), why duplicate this?
        params_list.append(dict(params=module.parameters(), lr=args.lr * 2.5))
    #optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)
    optimizer = optim.AdamW(params_list, lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)

    # use argument list for scheduler !!! makes adjust_learning_rate redundant !!!
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, ([args.lr]*6)+([args.lr*2.5]*2), args.num_steps+100,
        pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler

def adjust_learning_rate(optimizer, scheduler):
    """ adjust optimizer learning rates individually for the parameter groups

    Args:
        optimizer (torch.optim.Optimizer): parameter optimizer
        scheduler (torch.optim.lr_scheduler.OneCycleLR): learning rate updater
    """

    # use the only the first component of the latest learning rate
    # the array returned by get_last_lr has six identical learning rates for the parameter groups
    # i.e.: scheduler.get_last_lr() = np.ones((len(param_groups)))*lr
    # where the six groups are: 
    # [model.cnet, model.fnet, model.update_block, model.guidance, model.cost_agg1, model.cost_agg2]
    lr = scheduler.get_last_lr()[0]
    nums = len(optimizer.param_groups)

    # print(len(scheduler.get_last_lr()))
    # print("before:")
    # for index in range(nums):
    #     print(f"opt   {index}: {optimizer.param_groups[index]['lr']}")
    #     print(f"sched {index}: {scheduler.get_last_lr()[index]}")

    # for first nums-2 groups, set learning rate to lr
    # learninig rate does not change: lr -> lr
    # they are: [model.cnet, model.fnet, model.update_block, model.guidance]
    for index in range(0, nums-2):
        optimizer.param_groups[index]['lr'] = lr

    # for last two groups, set learning rate to lr*2.5
    # learning rate changes: lr -> 2.5*lr
    # they are: [model.cost_agg1, model.cost_agg2]
    # this means that the learning rate for 1d cost aggregation 
    # is chosen as 2.5 times higher than for the rest of the parameters
    for index in range(nums-2, nums):
        optimizer.param_groups[index]['lr'] = lr * 2.5

    # print("after:")
    # for index in range(nums):
    #     print(f"opt   {index}: {optimizer.param_groups[index]['lr']}")
    #     print(f"sched {index}: {scheduler.get_last_lr()[index]}")

class Argsclass:
    pass

if __name__ == "__main__":

    """
    this shows that except for the first-batch lr value,
    all other lr values are the same when using
    """

    args = Argsclass()
    args.lr = 0.0004
    args.wdecay = 0.0001
    args.epsilon = 1e-8
    args.num_steps = 50000

    optimizer1, scheduler1 = fetch_optimizer(args)
    optimizer2, scheduler2 = fetch_optimizer_alt(args)
    
    ngroups = len(optimizer1.param_groups)

    o1_lr = np.zeros((args.num_steps, ngroups))
    o2_lr = np.zeros((args.num_steps, ngroups))

    for i in range(args.num_steps):
        
        for group in range(ngroups):
            
            #assert optimizer1.param_groups[index]['lr'] == optimizer2.param_groups[index]['lr']
            
            o1_lr[i][group] = optimizer1.param_groups[group]['lr']
            o2_lr[i][group] = optimizer2.param_groups[group]['lr']

        optimizer1.step()
        scheduler1.step()
        adjust_learning_rate(optimizer1, scheduler1)
        optimizer2.step()
        scheduler2.step()

    print(f"max error: {np.max(np.abs((o1_lr[:,:]-o2_lr[:,:])))}")
    print(f"avg error: {np.average(np.abs((o1_lr[:,:]-o2_lr[:,:])))}")

    print("First step does not have adjust_learning_rate applied and thus is not identical:")
    print(f"{o1_lr[0]}\n{o2_lr[0]}")

    # All other steps are almost identical
    print(f"max error after first step: {np.max(np.abs((o1_lr[1:]-o2_lr[1:])))}")

    for group in range(ngroups):
        plt.plot(o1_lr[:, group], c="r", linewidth=4)
        plt.plot(o2_lr[:, group], c="b", linewidth=1)
    plt.savefig("fig_lr.png")

    plt.cla()
    for group in range(ngroups):
        plt.plot(o1_lr[1:, group]-o2_lr[1:, group])
    plt.savefig("fig_error.png")

    print(o2_lr[:4])