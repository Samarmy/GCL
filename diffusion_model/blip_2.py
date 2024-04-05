import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from glob import glob
import re
import os

araf_pattern = re.compile('(?:[a-zA-Z]|[\'///-])*araf(?:[a-zA-Z]|[\'///-])*([,;:])*( +)|( +)(?:[a-zA-Z]|[\'///-])*araf(?:[a-zA-Z]|[\'///-])*([,;:])*')


processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda")

splitty = 2
imgs = sorted(glob("/Work1/imagenet/train/*/*.JPEG"))
if splitty == 0:
    imgs = imgs[:(len(imgs)//4)]
elif splitty == 1:
    imgs = imgs[(len(imgs)//4):((len(imgs)//4)*2)]
elif splitty == 2:   
    imgs = imgs[((len(imgs)//4)*2):((len(imgs)//4)*3)]
else:
    imgs = imgs[((len(imgs)//4)*3):]

batch_size = 330
chunks = [imgs[x:x+batch_size] for x in range(0, len(imgs), batch_size)]



def remove_items(test_list, item): 
  
    # using list comprehension to perform the task 
    res = [i for i in test_list if i != item] 
    return res 

breaker = False
for progress, files in enumerate(chunks):
    if ((progress/len(chunks))*100.0) < 18.0:
        continue
    pics = [Image.open(file) for file in files]

    inputs = processor(pics, return_tensors="pt").to("cuda")
    out = model.generate(**inputs)
    output_strs = []
    
    for o1 in out:
        o1 = araf_pattern.sub('', processor.decode(o1, skip_special_tokens=True))
        output_strs.append(re.sub(' +', ' ', o1).strip())
        if "araf" in output_strs[-1]:
            print(output_strs[-1])
            breaker = True
    if breaker == True:
        break
    output_strs = remove_items(output_strs, "") 
    if(len(files) != len(output_strs)):
        print("PARSING ERROR")
        print("PARSING ERROR")
        print("PARSING ERROR")
        print(len(files), len(output_strs))
        break
    
    for ind, output_str in enumerate(output_strs):
        file_name = files[ind].replace(".JPEG", ".txt").replace("imagenet", "imagenet_iti")
        os.makedirs("/".join(file_name.split("/")[:-1]), exist_ok=True)
        f = open(file_name, "w")
        f.write(output_str)
        f.close()

    if progress%1==0:
        print((progress/len(chunks))*100.0)

# raw_image1 = Image.open("/Work1/imagenet100/train/n01531178/n01531178_108.JPEG").convert('RGB')
# raw_image2 = Image.open("/Work1/imagenet100/train/n01443537/n01443537_16.JPEG").convert('RGB')

# conditional image captioning
# text = "a photography of"
# inputs = processor(raw_image, text, return_tensors="pt").to("cuda")

# out = model.generate(**inputs)
# print(processor.decode(out[0], skip_special_tokens=True))

# unconditional image captioning
    