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
import torch.nn as nn
import timm
from metrics import metric_utils
import torchvision.transforms as T
import torch.nn.functional as F
from PIL import Image

splitty = 2
# torch.cuda.set_device(splitty)
# os.environ["CUDA_VISIBLE_DEVICES"]=str(splitty)

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

exp_name = "%s_%s_%s_res%i%s" % (
        "icgan",
        "biggan",
        "imagenet",
        256,
        "",
    )

device = "cuda"

vgg16_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/vgg16.pkl'
vgg16 = metric_utils.get_feature_detector(vgg16_url, device=device)
vgg16.cuda()
vgg16.eval()

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

generator.cuda()
generator.eval()

dataset = utils.get_dataset_images(
        256,
        data_path="/Work1/imagenet",
        longtail=False,
        split="train",
        test_part=False,
        which_dataset="imagenet",
        instance_json="",
        stuff_json="",
        get_encodings=True,
    )

generator1 = torch.Generator().manual_seed(42)
dataset = torch.utils.data.random_split(dataset, [0.25, 0.25, 0.25, 0.25], generator=generator1)[splitty]

loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=43,
    num_workers=12,
    pin_memory=True,
    drop_last=False,
)

gpu = torch.device("cuda")

for ind, data in enumerate(loader):
    if (((ind/len(loader))*100.0) < 22.0):
        print("skipping")
        continue
        
    with torch.no_grad():
#         start_time = time.time()
        # data2 = data[0].cuda(gpu, non_blocking=True)
        # x_tf1 = data2 * 0.5 + 0.5
        # x_tf2 = (x_tf1 - norm_mean) / norm_std
        # x_tf3 = torch.nn.functional.upsample(x_tf2, 224, mode="bicubic")
        # x_feat1, _ = net(x_tf3)
        # x_feat2 =  x_feat1/torch.linalg.norm(x_feat1, dim=1, keepdims=True)
        
        path = data[3]
        data0 = data[0].cuda(gpu, non_blocking=True)
        x_feat2 =  data[4].cuda(gpu, non_blocking=True)
        
        x_img_orig = generator(torch.zeros((data0.shape[0], 119)).cuda(), None, x_feat2)

        target_features = vgg16(F.interpolate((torch.clamp((data0 * 0.5 + 0.5), 0, 1)*255), size=(224, 224)), resize_images=False, return_lpips=True)

    scaler = torch.cuda.amp.GradScaler()
    num_steps = 100
    initial_learning_rate = 0.1
    z2 = torch.tensor(torch.zeros((data0.shape[0], 119)), dtype=torch.float32, device=gpu, requires_grad=True)
    
    optimizer = torch.optim.Adam([z2], betas=(0.9, 0.999), lr=initial_learning_rate)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=20)

    with torch.no_grad():
        original_synth_image = generator(z2, None, x_feat2).cpu()
    
    for step in range(num_steps):          
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            x_img = generator(z2, None, x_feat2)
            synth_images = (x_img + 1) * (255/2)
            synth_features = vgg16(F.interpolate(synth_images, size=(224, 224)), resize_images=False, return_lpips=True)
            lpips_loss = (target_features - synth_features).square().sum(dim=1).sum()
#             mse_loss = l2_criterion(target_img, synth_images)
            loss = lpips_loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
    with torch.no_grad():    
        # x_img = generator(z2, None, x_feat2)
        x_img = x_img.detach().cpu()
        
        for ind3 in range(x_img.shape[0]):
            folder = path[ind3].replace("imagenet", "ICGAN_Inversion").replace(".JPEG", "")
            os.makedirs(folder, exist_ok=True)
            file_name = folder.split("/")[-1]
            if (torch.all(original_synth_image[ind3] == x_img[ind3])):
                raise Exception("Inversion is Not Occuring")

            original_image = Image.open(path[ind3]).convert("RGB") 
            original_image.save(folder + "/" + file_name + "_R.JPEG")
            
            _gen_img = torchvision.transforms.functional.to_pil_image(torch.clamp((x_img[ind3] * 0.5 + 0.5), 0, 1))
            _gen_img.save(folder + "/" + file_name + "_0_G.JPEG")
            
            _original_synth_image = torchvision.transforms.functional.to_pil_image(torch.clamp((original_synth_image[ind3] * 0.5 + 0.5), 0, 1))
            _original_synth_image.save(folder + "/" + file_name + "_B.JPEG")

    del scaler
    del z2
    del optimizer
    del scheduler
            
    #         break
    print(str((ind/len(loader))*100.0) + " " + folder)
    torch.save(((ind/len(loader))*100.0), "Encoder_Inversion" + str(splitty) + "_Checkpoint.pt")