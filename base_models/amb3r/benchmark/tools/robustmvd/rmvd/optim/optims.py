# References:
#   https://github.com/HengyiWang/amb3r/blob/main/benchmark/tools/robustmvd/rmvd/optim/optims.py

import torch

from .registry import register_optimizer, register_scheduler


@register_optimizer
def adam(model, lr, **_):
    lr_base = 1e-4
    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.Adam(params, lr=lr)
    return optim
    

@register_scheduler
def flownet_scheduler(optimizer, **_):
    lr_intervals = [300000, 400000, 500000]
    gamma = 0.5
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_intervals, gamma=gamma)
    return scheduler

# def setup_late_fusion_optimization(data):
#     model_uncert = data['model_uncert']
#     model_unet = data['model_unet']
#     model_pred = data['model_pred']

#     lr_base = CONFIG['lr_base']

#     params = [p for p in model_uncert.parameters() if p.requires_grad] + [p for p in model_unet.parameters() if p.requires_grad] + [p for p in model_pred.parameters() if p.requires_grad]
#     print("Params to optimize: {}".format(sum(p.numel() for p in params)))
#     optimizer = torch.optim.Adam(params, lr=lr_base)

#     data['optimizer'] = optimizer


# def setup_mvsnet_optimization(data):
#     model = data['model']
#     lr_base = CONFIG['lr_base']

#     params = [p for p in model.parameters() if p.requires_grad]
#     print("Params to optimize: {}".format(sum(p.numel() for p in params)))
#     optimizer = torch.optim.RMSprop(params, lr=lr_base, alpha=0.9)

#     data['optimizer'] = optimizer

# def setup_mvsnet_lr(data):
#     optimizer = data['optimizer']
#     finished_iterations = data['finished_iterations']

#     # each 10k decay with factor 0.9 -> gamma = 0.9**(1/10000)
#     gamma = 0.9999894640039382

#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma)
#     for _ in range(finished_iterations):
#         scheduler.step()

#     data['scheduler'] = scheduler
