from glob import glob
import torch
from diffusers.utils import load_image
from PIL import Image
import shutil
import argparse
import os

from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
import torch
from torchvision import transforms


transform = transforms.Compose([
    transforms.Resize(512),
    transforms.CenterCrop((512, 512)),
])

parser = argparse.ArgumentParser(
        description="Diffusion Augmenter"
    )

# parser.add_argument("--data-dir", type=Path, help="path to dataset")

parser.add_argument(
        "--split",
        default=0,
        type=int,
        choices=(0, 1, 2, 3),
    )

parser.add_argument(
        "--batch-size", default=14, type=int, metavar="N", help="mini-batch size"
    )

parser.add_argument(
        "--strength",
        default=0.5,
        type=float,
        help="Diffusion image mix strength",
    )

parser.add_argument(
        "--num_inference_steps",
        default=10,
        type=int,
        help="Diffusion number of inference steps",
    )

args = parser.parse_args()

# pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
pipe = AutoPipelineForImage2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
pipe.to("cuda")

splitty = args.split
img_strs = sorted(glob("/Work1/imagenet100_iti/train/*/*.txt"))
if splitty == 0:
    img_strs = img_strs[:(len(img_strs)//4)]
elif splitty == 1:
    img_strs = img_strs[(len(img_strs)//4):((len(img_strs)//4)*2)]
elif splitty == 2:   
    img_strs = img_strs[((len(img_strs)//4)*2):((len(img_strs)//4)*3)]
else:
    img_strs = img_strs[((len(img_strs)//4)*3):]

chunks = [img_strs[x:x+args.batch_size] for x in range(0, len(img_strs), args.batch_size)]

for progress, files in enumerate(chunks):
    if ((progress/len(chunks))*100.0) < 50.0:
        continue
    texts = [open(file, "r").read() + ", 8k" for file in files]
    pics = [transform(Image.open(file.replace("_iti", "").replace(".txt", ".JPEG")).convert('RGB')) for file in files]
    images = pipe(texts, image=pics, num_inference_steps=args.num_inference_steps, strength=args.strength, guidance_scale=0.0).images
    for ind, output_img in enumerate(images):
        file_name = files[ind].replace("imagenet100", "imagenet100_" + str(int(10*args.strength))).replace(".txt", "_G.JPEG")
        os.makedirs("/".join(file_name.split("/")[:-1]), exist_ok=True)
        output_img.save(file_name)
        shutil.copyfile(file_name.replace("imagenet100_" + str(int(10*args.strength)) + "_iti", "imagenet100").replace("_G.JPEG", ".JPEG"), file_name.replace("_G.JPEG", "_R.JPEG"))

    if progress%10==0:
        print(args.strength, (progress/len(chunks))*100.0 )






# from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

# model_id = "stabilityai/stable-diffusion-2-1"
# pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
# pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
# pipe = pipe.to("cuda")

# from diffusers import DiffusionPipeline

# pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
# pipe.to("cuda")

# from diffusers import StableDiffusionPipeline

# model_id = "runwayml/stable-diffusion-v1-5"
# pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
# pipe = pipe.to("cuda")

# from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler

# model_id = "stabilityai/stable-diffusion-2"

# # Use the Euler scheduler here instead
# scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
# pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
# pipe = pipe.to("cuda")

# from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler

# model_id = "stabilityai/stable-diffusion-2-base"

# # Use the Euler scheduler here instead
# scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
# pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
# pipe = pipe.to("cuda")