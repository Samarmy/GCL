from diffusers import AutoPipelineForImage2Image
from diffusers.utils import load_image
import torch
from PIL import Image

pipe = AutoPipelineForImage2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
pipe.to("cuda")

# init_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png").resize((512, 512))
init_image = Image.open("/Work1/imagenet/train/n01498041/n01498041_28.JPEG")

prompt = "fish with blue spots on its body and a long tail"

image = pipe(prompt, image=init_image, num_inference_steps=10, strength=0.3, guidance_scale=0.0).images[0]

image.save("delete.png")