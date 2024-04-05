import torch
import torchvision
from torch import nn
import argparse
import os
import time
import json
from typing import Callable, Dict, Optional, Tuple, Type, Union
import copy

# from lightly.loss import VICRegLoss, BarlowTwinsLoss
import torch.distributed as dist
import torch.nn.functional as F

## The projection head is the same as the Barlow Twins one
from lightly.models.modules import BarlowTwinsProjectionHead
from lightly.transforms.vicreg_transform import VICRegTransform

from distributed import init_distributed_mode
from optim import SwaVOptimizer, BYOLOptimizer, VICRegOptimizer
from datasets import FastBYOL
import resnet

def get_arguments():
    parser = argparse.ArgumentParser(description="Contrastive Learning", add_help=False)

    #Checkpoints
    parser.add_argument("--data-folder", type=str, default="/Work1/Focus100/train",
                        help='Path to the experiment folder, where all logs/checkpoints will be stored')
    parser.add_argument("--exp-file", type=str, default="./checkpoints/barlow_twins.pth",
                        help='Path to the experiment folder, where all logs/checkpoints will be stored')
    parser.add_argument("--log-freq-time", type=int, default=60,
                        help='Print logs to the stats.txt file every [log-freq-time] seconds')
    
    # Training
    parser.add_argument("--epochs", type=int, default=100,
                        help='Number of epochs')
    
    parser.add_argument("--batch-size", type=int, default=256,
                        help='Effective batch size (per worker batch size is [batch-size] / world-size)')
    parser.add_argument("--num-workers", type=int, default=10)
    # parser.add_argument("--base-lr", type=float, default=4.8,
    #                     help='Base learning rate, effective learning  after warmup is [base-lr] * [batch-size] / 256')
    parser.add_argument("--base-lr", type=float, default=0.3,
                        help='Base learning rate, effective learning after warmup is [base-lr] * [batch-size] / 256')
    parser.add_argument("--nmb-gen", type=int, default=0, help="number of generative images")
    parser.add_argument("--flip-aug", action="store_true",  help="flip augmentation for real image") 
    parser.add_argument("--color-aug", action="store_true",  help="color augmentation for real image") 
    

    # Distributed
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist-url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')

    return parser

# class BarlowTwinsIterative(nn.Module):
#     def __init__(self, backbone):
#         super().__init__()
#         self.backbone = backbone
#         self.projection_head = BarlowTwinsProjectionHead(2048, 8192, 8192)

#         # enable gather_distributed to gather features from all gpus
#         # before calculating the loss
#         self.criterion = BarlowTwinsLoss(gather_distributed=True)

#     def forward(self, x, y):
#         x = self.backbone(x).flatten(start_dim=1)
#         x = self.projection_head(x)

#         for ind, y_ in enumerate(y):
#             y_ = self.backbone(y_).flatten(start_dim=1)
#             y_ = self.projection_head(y_)
#             if (ind == 0):
#                 loss = self.criterion(x, y_) 
#             else:
#                 loss = loss + self.criterion(x, y_) 
#         return loss/(y.shape[0])

class VICReg(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.projection_head = BarlowTwinsProjectionHead(2048, 8192, 8192)

        # enable gather_distributed to gather features from all gpus
        # before calculating the loss
        # self.criterion = VICRegLoss(gather_distributed=True)

    def forward(self, x, y):
        x = self.backbone(x).flatten(start_dim=1)
        x = self.projection_head(x)

        num_views = y.shape[0]
        batch_size = y.shape[1]

        y = y.reshape(num_views*batch_size, y.shape[2], y.shape[3], y.shape[4])

        y = self.backbone(y).flatten(start_dim=1)
        y = self.projection_head(y)
        y = y.reshape(num_views, batch_size, -1)

        for ind in range(num_views):
            if (ind == 0):
                loss = VICRegLoss(x, y[ind]) 
            else:
                loss = loss + VICRegLoss(x, y[ind]) 
        return loss/num_views


def main(args):
    torch.backends.cudnn.benchmark = True
    init_distributed_mode(args)
    print(args)
    gpu = torch.device(args.device)

    backbone = resnet.__dict__["resnet50"](
            zero_init_residual=True
        )[0]
    
    model = VICReg(backbone).cuda(gpu)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    dataset = FastBYOL(root_dir=args.data_folder, nmb_gen=args.nmb_gen, flip_aug=args.flip_aug, color_aug=args.color_aug)
    print("Dataset length is " + str(len(dataset)))

    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
    assert args.batch_size % args.world_size == 0
    per_device_batch_size = args.batch_size // args.world_size
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=per_device_batch_size,
        pin_memory=True,
        num_workers=args.num_workers,
        sampler=sampler,
    )
    
    optimizer, adjust_learning_rate = VICRegOptimizer(model, dataloader)

    scaler = torch.cuda.amp.GradScaler()
    if os.path.isfile(args.exp_file):
        if args.rank == 0:
            print("Resuming From Checkpoint")
        ckpt = torch.load(args.exp_file, map_location="cpu")
        start_epoch = ckpt["epoch"]
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        # scaler.load_state_dict(ckpt["scaler"])
    else:
        start_epoch = 0

    start_time = last_logging = time.time()
    
    print("Starting Training")
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        for step, batch in enumerate(dataloader, start=epoch * len(dataloader)):
            x, y = batch
            x, y = x.to(gpu, non_blocking=True), y.to(gpu, non_blocking=True).permute(1, 0, 2, 3, 4)

            lr = adjust_learning_rate(args, optimizer, dataloader, step)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                loss = model(x, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            current_time = time.time()
            if args.rank == 0 and current_time - last_logging > args.log_freq_time:         
                stats = dict(
                    epoch=epoch,
                    step=step,
                    loss=loss.item(),
                    time=int(current_time - start_time),
                    lr=lr,
                )
                print(json.dumps(stats))
                last_logging = current_time
        if args.rank == 0: 
            state = dict(
                epoch=epoch + 1,
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                scaler=scaler.state_dict(),
            )
            torch.save(state, args.exp_file)
            torch.save(model.module.backbone.state_dict(), args.exp_file.replace(".pth", "_resnet50.pth"))
            # torch.save(state, args.exp_file.replace(".pth", "_" + str(epoch+1) + ".pth"))


def batch_all_gather(x):
    x_list = FullGatherLayer.apply(x)
    return torch.cat(x_list, dim=0)


class FullGatherLayer(torch.autograd.Function):
    """
    Gather tensors from all process and support backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def VICRegLoss(x, y):
    repr_loss = F.mse_loss(x, y)

    x = torch.cat(FullGatherLayer.apply(x), dim=0)
    y = torch.cat(FullGatherLayer.apply(y), dim=0)
    x = x - x.mean(dim=0)
    y = y - y.mean(dim=0)

    std_x = torch.sqrt(x.var(dim=0) + 0.0001)
    std_y = torch.sqrt(y.var(dim=0) + 0.0001)
    std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

    cov_x = (x.T @ x) / (768 - 1)
    cov_y = (y.T @ y) / (768 - 1)
    cov_loss = off_diagonal(cov_x).pow_(2).sum().div(
        8192
    ) + off_diagonal(cov_y).pow_(2).sum().div(8192)

    loss = (
        25.0 * repr_loss
        + 25.0 * std_loss
        + 1.0 * cov_loss
    )
    return loss
    


def handle_sigusr1(signum, frame):
    os.system(f'scontrol requeue {os.environ["SLURM_JOB_ID"]}')
    exit()


def handle_sigterm(signum, frame):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Contrastive learning training script', parents=[get_arguments()])
    args = parser.parse_args()
    main(args)