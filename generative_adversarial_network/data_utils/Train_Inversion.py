# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from pathlib import Path
import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F
from torch import nn, optim
import torch.distributed as dist
import torchvision.datasets as datasets

from distributed import init_distributed_mode

import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], ".."))

import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import torch

import data_utils.utils as data_utils
import inference.utils as inference_utils
import BigGAN_PyTorch.utils as biggan_utils
from data_utils.datasets_common import pil_loader
import torchvision.transforms as transforms
import time
from glob import glob

import torch
import torchvision
from glob import glob
import utils
import os
from PIL import Image
from data_utils.resnet import resnet50

def get_model(exp_name, root_path, backbone, device="cuda"):
    parser = biggan_utils.prepare_parser()
    parser = biggan_utils.add_sample_parser(parser)
    parser = inference_utils.add_backbone_parser(parser)

    args = ["--experiment_name", exp_name]
    args += ["--base_root", root_path]
    args += ["--model_backbone", backbone]

    config = vars(parser.parse_args(args=args))

    # Load model and overwrite configuration parameters if stored in the model
    config = biggan_utils.update_config_roots(config, change_weight_folder=False)
    generator, config = inference_utils.load_model_inference(config, device=device)
    biggan_utils.count_parameters(generator)
    generator.eval()

    return generator


def get_arguments():
    parser = argparse.ArgumentParser(description="Pretrain a resnet model with VICReg", add_help=False)

    # Checkpoints
    parser.add_argument("--exp-dir", type=Path, default="./inversion",
                        help='Path to the experiment folder, where all logs/checkpoints will be stored')
    parser.add_argument("--log-freq-time", type=int, default=2,
                        help='Print logs to the stats.txt file every [log-freq-time] seconds')

    # Optim
    parser.add_argument("--epochs", type=int, default=100,
                        help='Number of epochs')
    parser.add_argument("--batch-size", type=int, default=2048,
                        help='Effective batch size (per worker batch size is [batch-size] / world-size)')
    parser.add_argument("--lr", type=float, default=0.003,
                        help='Base learning rate, effective learning after warmup is [base-lr] * [batch-size] / 256')

    # Running
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')

    # Distributed
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist-url', default='env://',
                        help='url used to set up distributed training')

    return parser


def main(args):
    torch.backends.cudnn.benchmark = True
    init_distributed_mode(args)
    print(args)
    gpu = torch.device(args.device)

    if args.rank == 0:
        args.exp_dir.mkdir(parents=True, exist_ok=True)
        stats_file = open(args.exp_dir / "stats.txt", "a", buffering=1)
        print(" ".join(sys.argv))
        print(" ".join(sys.argv), file=stats_file)


    dataset = utils.get_dataset_images(
        256,
        data_path="/Work1/imagenet",
        longtail=False,
        split="train",
        test_part=False,
        which_dataset="imagenet",
        instance_json="",
        stuff_json="",
    )
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
    assert args.batch_size % args.world_size == 0
    per_device_batch_size = args.batch_size // args.world_size
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=per_device_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=sampler,
    )
    
    norm_mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda(gpu)
    norm_std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda(gpu)
    
    model = resnet50(
            pretrained=False, classifier_run=True
        )
    # inversion.fc=nn.Sequential([nn.Linear(2048,119),nn.BatchNorm1d(1024), nn.ReLU(True), nn.Linear(1024,119)])
    model.fc=nn.Linear(2048,119)
    

    model = model.cuda(gpu)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    
    device = "cuda"
    exp_name = "%s_%s_%s_res%i%s" % (
            "icgan",
            "biggan",
            "imagenet",
            256,
            "",
        )
    generator = get_model(
            exp_name, "/Work2/Watch_This/ICGAN/ic_gan/pretrained_models_path", "biggan", device=device
        )

    # generator = torch.nn.DataParallel(generator)
    generator.cuda(gpu)
    generator.eval()
    
    net = utils.load_pretrained_feature_extractor(
        '/Work2/Watch_This/ICGAN/ic_gan/pretrained_models_path/swav_800ep_pretrain.pth.tar',
        "selfsupervised",
        "resnet50",
    )
    # net = torch.nn.DataParallel(net)
    net.cuda(gpu)
    net.eval()
    
    mse_loss = torch.nn.MSELoss()

    if (args.exp_dir / "inversion.pth").is_file():
        if args.rank == 0:
            print("resuming from checkpoint")
        ckpt = torch.load(args.exp_dir / "inversion.pth", map_location="cpu")
        start_epoch = ckpt["epoch"]
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
    else:
        start_epoch = 0

    start_time = last_logging = time.time()
    scaler = torch.cuda.amp.GradScaler()
    
    if args.rank == 0:
        val_img = next(iter(dataset))[0].unsqueeze(0).cuda(gpu, non_blocking=True)
        val_img = val_img * 0.5 + 0.5
        val_img = (val_img - norm_mean) / norm_std
        val_img = torch.nn.functional.upsample(val_img, 224, mode="bicubic")
        with torch.no_grad():
            val_feat, _ = net(val_img)
            val_feat /= torch.linalg.norm(val_feat, dim=1, keepdims=True)
    
    
    total_num_iterations = len(loader)    
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        for step, (x, _, image_id, path) in enumerate(loader, start=epoch * len(loader)):
            x = x.cuda(gpu, non_blocking=True)
            x_tf = x
            x_tf = x_tf * 0.5 + 0.5
            x_tf = (x_tf - norm_mean) / norm_std
            x_tf = torch.nn.functional.upsample(x_tf, 224, mode="bicubic")
            
            
            with torch.no_grad():
                x_feat, _ = net(x_tf)
                x_feat /= torch.linalg.norm(x_feat, dim=1, keepdims=True)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                z, _ = model(x_tf)
                z_2 = (z / torch.linalg.norm(z, dim=1, keepdims=True))
                x_img = generator(z_2, None, x_feat)
                loss = mse_loss(x, x_img)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            current_time = time.time()
            if args.rank == 0 and current_time - last_logging > args.log_freq_time:
                stats = dict(
                    done=str(round(((step/total_num_iterations)*100.0),2)) + "%",
                    epoch=epoch,
                    step=step,
                    loss=loss.item(),
                    time=int(current_time - start_time),
                )
                print(json.dumps(stats))
                print(json.dumps(stats), file=stats_file)
                last_logging = current_time
                
        if args.rank == 0:
            state = dict(
                epoch=epoch + 1,
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
            )
            torch.save(state, args.exp_dir / "inversion.pth")
#             with torch.no_grad():
#                 val_z, _ = model(val_img)
#                 val_z /= torch.linalg.norm(val_z, dim=1, keepdims=True)
#                 val_gen_img = generator(val_z, None, val_feat)* 0.5 + 0.5
#                 sample_img_name = "img_" + str(epoch) + ".png"
#                 print("val_img[0]", val_img[0].shape)
#                 print("val_gen_img[0]", val_gen_img[0].shape)
#                 val_sample_img = torch.clamp(torch.cat((val_img[0], torch.nn.functional.upsample(val_gen_img, 224, mode="bicubic")[0]), dim=1), 0, 1)
#                 torchvision.transforms.functional.to_pil_image(val_sample_img).save(args.exp_dir / sample_img_name)
            
            
    if args.rank == 0:
        torch.save(model.module.backbone.state_dict(), args.exp_dir / "resnet50.pth")



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


def handle_sigusr1(signum, frame):
    os.system(f'scontrol requeue {os.environ["SLURM_JOB_ID"]}')
    exit()


def handle_sigterm(signum, frame):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Inversion training script', parents=[get_arguments()])
    args = parser.parse_args()
    main(args)