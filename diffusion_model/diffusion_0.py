from glob import glob
import torch
from diffusers.utils import load_image
from PIL import Image
import shutil

from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
import torch

# pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
pipe = AutoPipelineForImage2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
pipe.to("cuda")

splitty = 0
img_strs = sorted(glob("/Work1/imagenet_iti/train/*/*.txt"))
if splitty == 0:
    img_strs = img_strs[:(len(img_strs)//4)]
elif splitty == 1:
    img_strs = img_strs[(len(img_strs)//4):((len(img_strs)//4)*2)]
elif splitty == 2:   
    img_strs = img_strs[((len(img_strs)//4)*2):((len(img_strs)//4)*3)]
else:
    img_strs = img_strs[((len(img_strs)//4)*3):]

batch_size = 16
chunks = [img_strs[x:x+batch_size] for x in range(0, len(img_strs), batch_size)]

for progress, files in enumerate(chunks):
    if ((progress/len(chunks))*100.0) < 70.78773075297103:
        continue
    texts = [open(file, "r").read() for file in files]
    pics = [Image.open(file.replace("_iti", "").replace(".txt", ".JPEG")).convert('RGB').resize((512, 512)) for file in files]
    images = pipe(texts, image=pics, num_inference_steps=10, strength=0.5, guidance_scale=0.0).images
    for ind, output_img in enumerate(images):
        output_img.save(files[ind].replace(".txt", "_G.JPEG"))
        shutil.copyfile(files[ind].replace("_iti", "").replace(".txt", ".JPEG"), files[ind].replace(".txt", "_R.JPEG"))

    if progress%10==0:
        print((progress/len(chunks))*100.0)






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