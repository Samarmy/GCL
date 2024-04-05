import random
from PIL import ImageFilter, Image
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import glob
import torch
from torch.utils.data import Dataset

class PILRandomGaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image. Take the radius and probability of
    application as the parameter.
    This transform was used in SimCLR - https://arxiv.org/abs/2002.05709
    """

    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = np.random.rand() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )

def get_color_distortion(s=1.0):
    # s is the strength of color distortion.
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort

class SwaG(Dataset):
    def __init__(self, root_dir="/Work1/SwaG224_100/train", num_aug_imgs=10, size_crops=[224, 192, 96], nmb_crops=[2, 2, 6], min_scale_crops=[0.14,0., 0.05], max_scale_crops=[1., 1., 0.14]):
        assert len(size_crops) == 3
        assert len(size_crops) == len(nmb_crops)
        assert len(min_scale_crops) == len(nmb_crops)
        assert len(max_scale_crops) == len(nmb_crops)
        self.root_dir = root_dir  + "/*/*/*_R.JPEG"
        self.files = glob.glob(self.root_dir)
        self.num_aug_imgs = [("_aug" + str(x) + ".JPEG") for x in list(range(num_aug_imgs))]
        self.nmb_crops = nmb_crops

        color_transform = [get_color_distortion(), PILRandomGaussianBlur()]
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        randomresizedcrop0 = transforms.RandomResizedCrop(
                size_crops[0],
                scale=(min_scale_crops[0], max_scale_crops[0]),
            )

        randomresizedcrop2 = transforms.RandomResizedCrop(
                size_crops[2],
                scale=(min_scale_crops[2], max_scale_crops[2]),
            )
        
        self.trans0 = transforms.Compose([
                randomresizedcrop0,
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Compose(color_transform),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        
        self.trans1 = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        self.trans2 = transforms.Compose([
                randomresizedcrop2,
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Compose(color_transform),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
        ])
        
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        prime_img_name = self.files[idx]
        prime_image = Image.open(prime_img_name).convert("RGB")

        multi_crops = []

        if (self.nmb_crops[0] > 0):
            multi_crops = multi_crops + [self.trans0(prime_image) for r in range(self.nmb_crops[0])]

        if (self.nmb_crops[1] > 0):
            multi_crops = multi_crops +  [self.trans1(Image.open(self.files[idx].replace("_R.JPEG", x)).convert("RGB")) for x in random.sample(self.num_aug_imgs,self.nmb_crops[1])]

        if (self.nmb_crops[2] > 0):
            multi_crops = multi_crops + [self.trans2(prime_image) for r in range(self.nmb_crops[2])]
                
        return multi_crops

class FastBYOL(Dataset):
    def __init__(self, root_dir="/Work1/Focus100/train", num_aug_imgs=10, size_crops=224, nmb_gen=0, min_scale_crops=0.14, max_scale_crops=1., flip_aug=False, color_aug=False):
        self.root_dir = root_dir  + "/*/*/*_R.JPEG"
        self.files = glob.glob(self.root_dir)
        self.num_aug_imgs = [("_aug" + str(x) + ".JPEG") for x in list(range(num_aug_imgs))]
        self.nmb_gen = nmb_gen

        color_transform = [get_color_distortion(), PILRandomGaussianBlur()]
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        randomresizedcrop = transforms.RandomResizedCrop(
                size_crops,
                scale=(min_scale_crops, max_scale_crops),
            )


        trans_real = [randomresizedcrop,]
        if (flip_aug):
            print("Using flip augmentations")
            trans_real = trans_real + [transforms.RandomHorizontalFlip(p=0.5),]
        
        if (color_aug):
            print("Using color augmentations")
            trans_real = trans_real + [transforms.Compose(color_transform),]

        trans_real = trans_real + [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]

        self.trans_real = transforms.Compose(trans_real)

        self.trans_gen = transforms.Compose([
                randomresizedcrop,
                # transforms.RandomHorizontalFlip(p=0.5),
                # transforms.Compose(color_transform),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
        ])
        
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.files[idx]
        img = Image.open(img_name).convert("RGB")

        img_list = []

        if (self.nmb_gen == 0):
            img_list = torch.stack([self.trans_real(img),])
        else:
            img_list = torch.stack([self.trans_gen(Image.open(self.files[idx].replace("_R.JPEG", x)).convert("RGB")) for x in random.sample(self.num_aug_imgs,self.nmb_gen)])
                
        return self.trans_real(img), img_list

# class Focus(Dataset):
#     def __init__(self, root_dir="/Work1/Focus100/train", num_aug_imgs=10, size_crops=[224, 224], nmb_crops=[4, 4], min_scale_crops=[0.14,0.14], max_scale_crops=[1., 1.]):
#         assert len(size_crops) == len(nmb_crops)
#         assert len(min_scale_crops) == len(nmb_crops)
#         assert len(max_scale_crops) == len(nmb_crops)
#         self.root_dir = root_dir  + "/*/*/*_R.JPEG"
#         self.files = glob.glob(self.root_dir)
#         self.num_aug_imgs = [("_aug" + str(x) + ".JPEG") for x in list(range(num_aug_imgs))]
#         self.nmb_crops = nmb_crops
#         self.nmb_crops_generative = [x//2 for x in nmb_crops[1:]]

#         color_transform = [get_color_distortion(), PILRandomGaussianBlur()]
#         mean = [0.485, 0.456, 0.406]
#         std = [0.229, 0.224, 0.225]
#         randomresizedcrop = transforms.RandomResizedCrop(
#                 size_crops[0],
#                 scale=(min_scale_crops[0], max_scale_crops[0]),
#             )
        
#         trans_prime = transforms.Compose([
#                 randomresizedcrop,
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean=mean, std=std)
#             ])
#         trans_gen = transforms.Compose([
#                 transforms.CenterCrop(224),
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean=mean, std=std)
#             ])
#         self.trans_prime = trans_prime
#         self.trans_gen = trans_gen
        
#     def __len__(self):
#         return len(self.files)
    
#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()

#         sampled_generative_imgs = random.sample(self.num_aug_imgs,self.nmb_crops[1])
        
#         prime_img_name = self.files[idx]
#         prime_image = Image.open(prime_img_name).convert("RGB")
#         norm_crops = [self.trans_prime(prime_image) for r in range(self.nmb_crops[0])]
                
#         gen_images = [self.trans_gen(Image.open(self.files[idx].replace("_R.JPEG", x)).convert("RGB")) for x in sampled_generative_imgs]
        
#         multi_crops = torch.stack(norm_crops + gen_images)
        
#         return multi_crops
