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

def get_data(root_path, model, resolution, which_dataset, visualize_instance_images):
    data_path = os.path.join(root_path, "stored_instances")
    if model == "cc_icgan":
        feature_extractor = "classification"
    else:
        feature_extractor = "selfsupervised"
    filename = "%s_res%i_rn50_%s_kmeans_k1000_instance_features.npy" % (
        which_dataset,
        resolution,
        feature_extractor,
    )
    # Load conditioning instances from files
    data = np.load(os.path.join(data_path, filename), allow_pickle=True).item()

    transform_list = None
    if visualize_instance_images:
        # Transformation used for ImageNet images.
        transform_list = transforms.Compose(
            [data_utils.CenterCropLongEdge(), transforms.Resize(resolution)]
        )
    return data, transform_list


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


def get_conditionings(test_config, generator, data):
    # Obtain noise vectors
    z = torch.empty(
        5 * 5,
        generator.z_dim if "biggan" == "stylegan2" else generator.dim_z,
    ).normal_(mean=0, std=1.0)

    # Subsampling some instances from the 1000 k-means centers file
    if 5 > 1:
        total_idxs = np.random.choice(
            range(1000), 5, replace=False
        )

    # Obtain features, labels and ground truth image paths
    all_feats, all_img_paths, all_labels = [], [], []
    for counter in range(5):
        # Index in 1000 k-means centers file
        if None is not None:
            idx = None
        else:
            idx = total_idxs[counter]
        # Image paths to visualize ground-truth instance
        if False:
            all_img_paths.append(data["image_path"][idx])
        # Instance features
        all_feats.append(
            torch.FloatTensor(data["instance_features"][idx : idx + 1]).repeat(
                5, 1
            )
        )
        # Obtain labels
        if None is not None:
            # Swap label for a manually specified one
            label_int = None
        else:
            # Use the label associated to the instance feature
            label_int = int(data["labels"][idx])
        # Format labels according to the backbone
        labels = None
        if "biggan" == "stylegan2":
            dim_labels = 1000
            labels = torch.eye(dim_labels)[torch.LongTensor([label_int])].repeat(
                5, 1
            )
        else:
            if "icgan" == "cc_icgan":
                labels = torch.LongTensor([label_int]).repeat(
                    5
                )
        all_labels.append(labels)
    # Concatenate all conditionings
    all_feats = torch.cat(all_feats)
    if all_labels[0] is not None:
        all_labels = torch.cat(all_labels)
    else:
        all_labels = None
    return z, all_feats, all_labels, all_img_paths

net = utils.load_pretrained_feature_extractor(
        '/Work2/Watch_This/ICGAN/ic_gan/pretrained_models_path/swav_800ep_pretrain.pth.tar',
        "selfsupervised",
        "resnet50",
    )
# net = torch.nn.DataParallel(net)
net.to("cuda")
net.eval()

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
generator.to("cuda")
generator.eval()

files = glob("/Work1/imagenet/train/*")

for file in files:
    os.makedirs(file.replace("imagenet", "ICGAN"), exist_ok=True)
    
files = glob("/Work1/imagenet/train/*/*")

for file in files:
    os.makedirs(file.replace("imagenet", "ICGAN").replace(".JPEG", ""), exist_ok=True)
    
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

generator1 = torch.Generator().manual_seed(42)
dataset, _, _, _ = torch.utils.data.random_split(dataset, [0.25, 0.25, 0.25, 0.25], generator=generator1)

kwargs = {
    "num_workers": 14,
    "pin_memory": True,
    "drop_last": False,
#     "persistent_workers": True,
}

train_loader = utils.get_dataloader(
    dataset, 120, shuffle=False, **kwargs
)

norm_mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
norm_std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()

for i, (x, _, image_id, path) in enumerate(train_loader):
    with torch.no_grad():
#         if (i < 2400): 
#             print(str((i/len(train_loader))*100.0) + "% Done, Epoch: " + str(i) + "/" + str(len(train_loader))) 
#             continue
    
        x_tf = x.cuda()
        x_tf = x_tf * 0.5 + 0.5
        x_tf = (x_tf - norm_mean) / norm_std
        x_tf = torch.nn.functional.upsample(x_tf, 224, mode="bicubic")

        x_feat, _ = net(x_tf)
        x_feat /= torch.linalg.norm(x_feat, dim=1, keepdims=True)
        
        for s in range(0, 10):

                
            z = torch.normal(0.0, 1.0, (x_tf.shape[0], 119)).cuda() 

            x_img = generator(z, None, x_feat)

            x = x.cpu()
    #         x_feat = x_feat.cpu()
            x_img = x_img.cpu()

            for j in range(x_tf.shape[0]):
                folder = path[j].replace("imagenet", "ICGAN10").replace(".JPEG", "")
                os.makedirs(folder, exist_ok=True)
                file_name = folder.split("/")[-1]
                test = folder + "/" + file_name + ".JPEG"
                if s == 0:
                    real_img = torch.clamp((x[j] * 0.5 + 0.5), 0, 1)
                    torchvision.transforms.functional.to_pil_image(real_img).save(folder + "/" + file_name + "_R.JPEG")
                gen_img = torch.clamp((x_img[j] * 0.5 + 0.5), 0, 1)
                torchvision.transforms.functional.to_pil_image(gen_img).save(folder + "/" + file_name + "_" + str(s) + "_G.JPEG")
    #             torch.save(x_feat[j], folder + "/" + file_name + "_F.pt")
        
    print(str((i/len(train_loader))*100.0) + "% Done, Epoch: " + str(i) + "/" + str(len(train_loader))) 
