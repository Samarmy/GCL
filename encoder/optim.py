import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.nn.parameter import Parameter
import numpy as np
import math

def exclude_bias_and_norm(p):
    return p.ndim == 1

class LARS(optim.Optimizer):
    def __init__(
        self,
        params,
        lr=0,
        weight_decay=1e-6,
        momentum=0.9,
        eta=0.001,
        weight_decay_filter=exclude_bias_and_norm,
        lars_adaptation_filter=exclude_bias_and_norm,
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            eta=eta,
            weight_decay_filter=weight_decay_filter,
            lars_adaptation_filter=lars_adaptation_filter,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                dp = p.grad

                if dp is None:
                    continue

                if g["weight_decay_filter"] is None or not g["weight_decay_filter"](p):
                    dp = dp.add(p, alpha=g["weight_decay"])

                if g["lars_adaptation_filter"] is None or not g[
                    "lars_adaptation_filter"
                ](p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(
                        param_norm > 0.0,
                        torch.where(
                            update_norm > 0, (g["eta"] * param_norm / update_norm), one
                        ),
                        one,
                    )
                    dp = dp.mul(q)

                param_state = self.state[p]
                if "mu" not in param_state:
                    param_state["mu"] = torch.zeros_like(p)
                mu = param_state["mu"]
                mu.mul_(g["momentum"]).add_(dp)

                p.add_(mu, alpha=-g["lr"])


class LARC(object):
    """
    :class:`LARC` is a pytorch implementation of both the scaling and clipping variants of LARC,
    in which the ratio between gradient and parameter magnitudes is used to calculate an adaptive 
    local learning rate for each individual parameter. The algorithm is designed to improve
    convergence of large batch training.
     
    See https://arxiv.org/abs/1708.03888 for calculation of the local learning rate.

    In practice it modifies the gradients of parameters as a proxy for modifying the learning rate
    of the parameters. This design allows it to be used as a wrapper around any torch.optim Optimizer.

    ```
    model = ...
    optim = torch.optim.Adam(model.parameters(), lr=...)
    optim = LARC(optim)
    ```

    It can even be used in conjunction with apex.fp16_utils.FP16_optimizer.

    ```
    model = ...
    optim = torch.optim.Adam(model.parameters(), lr=...)
    optim = LARC(optim)
    optim = apex.fp16_utils.FP16_Optimizer(optim)
    ```

    Args:
        optimizer: Pytorch optimizer to wrap and modify learning rate for.
        trust_coefficient: Trust coefficient for calculating the lr. See https://arxiv.org/abs/1708.03888
        clip: Decides between clipping or scaling mode of LARC. If `clip=True` the learning rate is set to `min(optimizer_lr, local_lr)` for each parameter. If `clip=False` the learning rate is set to `local_lr*optimizer_lr`.
        eps: epsilon kludge to help with numerical stability while calculating adaptive_lr
    """

    def __init__(self, optimizer, trust_coefficient=0.02, clip=True, eps=1e-8):
        self.optim = optimizer
        self.trust_coefficient = trust_coefficient
        self.eps = eps
        self.clip = clip

    def __getstate__(self):
        return self.optim.__getstate__()

    def __setstate__(self, state):
        self.optim.__setstate__(state)

    @property
    def state(self):
        return self.optim.state

    def __repr__(self):
        return self.optim.__repr__()

    @property
    def param_groups(self):
        return self.optim.param_groups

    @param_groups.setter
    def param_groups(self, value):
        self.optim.param_groups = value
    
    def state_dict(self):
        return self.optim.state_dict()

    def load_state_dict(self, state_dict):
        self.optim.load_state_dict(state_dict)

    def zero_grad(self):
        self.optim.zero_grad()

    def add_param_group(self, param_group):
        self.optim.add_param_group( param_group)

    def step(self):
        with torch.no_grad():
            weight_decays = []
            for group in self.optim.param_groups:
                # absorb weight decay control from optimizer
                weight_decay = group['weight_decay'] if 'weight_decay' in group else 0
                weight_decays.append(weight_decay)
                group['weight_decay'] = 0
                for p in group['params']:
                    if p.grad is None:
                        continue
                    param_norm = torch.norm(p.data)
                    grad_norm = torch.norm(p.grad.data)

                    if param_norm != 0 and grad_norm != 0:
                        # calculate adaptive lr + weight decay
                        adaptive_lr = self.trust_coefficient * (param_norm) / (grad_norm + param_norm * weight_decay + self.eps)

                        # clip learning rate for LARC
                        if self.clip:
                            # calculation of adaptive_lr so that when multiplied by lr it equals `min(adaptive_lr, lr)`
                            adaptive_lr = min(adaptive_lr/group['lr'], 1)

                        p.grad.data += weight_decay * p.data
                        p.grad.data *= adaptive_lr

        self.optim.step()
        # return weight decay control to optimizer
        for i, group in enumerate(self.optim.param_groups):
            group['weight_decay'] = weight_decays[i]

def SwaVOptimizer(model, train_loader):
    optimizer = optimizer = torch.optim.SGD(
        model.parameters(),
        lr=4.8,
        momentum=0.9,
        weight_decay=0.000001,
    )

    warmup_lr_schedule = np.linspace(0.3, 4.8, len(train_loader) * 10)
    iters = np.arange(len(train_loader) * (100 - 10))
    cosine_lr_schedule = np.array([0.0048 + 0.5 * (4.8 - 0.0048) * (1 + \
                         math.cos(math.pi * t / (len(train_loader) * (100 - 10)))) for t in iters])
    lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
    return optimizer, lr_schedule


def adjust_learning_rate(args, optimizer, loader, step):
    max_steps = args.epochs * len(loader)
    warmup_steps = 10 * len(loader)
    base_lr = args.base_lr * args.batch_size / 256
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr

# def VICRegOptimizer(model, train_loader):
#     optimizer = LARS(
#         model.parameters(),
#         lr=0.3,
#         weight_decay=1e-6,
#         momentum=0.9,
#         eta=0.001,
#         weight_decay_filter=exclude_bias_and_norm,
#         lars_adaptation_filter=exclude_bias_and_norm,
#     )

#     warmup_lr_schedule = np.linspace(0.3, 4.8, len(train_loader) * 10)
#     iters = np.arange(len(train_loader) * (100 - 10))
#     cosine_lr_schedule = np.array([0.0048 + 0.5 * (4.8 - 0.0048) * (1 + \
#                          math.cos(math.pi * t / (len(train_loader) * (100 - 10)))) for t in iters])
#     lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
#     return optimizer, adjust_learning_rate

def VICRegOptimizer(model, train_loader):
    optimizer = LARS(
        model.parameters(),
        lr=0,
        weight_decay=1e-6,
        weight_decay_filter=exclude_bias_and_norm,
        lars_adaptation_filter=exclude_bias_and_norm,
    )

    return optimizer, adjust_learning_rate

def BYOLOptimizer(model, train_loader):
    optimizer = LARS(
        model.parameters(),
        lr=0,
        weight_decay=1e-6,
        weight_decay_filter=exclude_bias_and_norm,
        lars_adaptation_filter=exclude_bias_and_norm,
    )

    return optimizer, adjust_learning_rate

