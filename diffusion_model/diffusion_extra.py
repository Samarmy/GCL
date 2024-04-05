from glob import glob
import torch
from diffusers.utils import load_image
from PIL import Image
import shutil
import os

from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
import torch
strength = 0.1
for strength in [0.1, 0.2, 0.3, 0.4, 0.5]:
    # pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
    pipe = AutoPipelineForImage2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
    pipe.to("cuda")
    
    files = ["/Work1/imagenet100_iti/train/n01773157/n01773157_10524.txt",]
    files = ["/Work1/imagenet100_iti/train/n02012849/n02012849_673.txt",]
    files = ["/Work1/imagenet100_iti/train/n01740131/n01740131_19944.txt",]
    
    texts = [open(file, "r").read() for file in files]
    pics = [Image.open(file.replace("_iti", "").replace(".txt", ".JPEG")).convert('RGB').resize((512, 512)) for file in files]
    images = pipe(texts, image=pics, num_inference_steps=10, strength=strength, guidance_scale=0.0).images
    for ind, output_img in enumerate(images):
        file_name = files[ind].replace("imagenet100", "imagenet100_" + str(int(10*strength))).replace(".txt", ".JPEG")
        os.makedirs("/".join(file_name.split("/")[:-1]), exist_ok=True)
        output_img.save(file_name)
        
        print(file_name)
        # shutil.copyfile(files[ind].replace("_iti", "").replace(".txt", ".JPEG"), files[ind].replace(".txt", "_R.JPEG"))
    
    print("DONE")






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