# torchrun --nproc_per_node=4 swav_original.py
# python evaluate.py --data-dir /Work1/imagenet100/ --pretrained /Work2/Watch_This/Lightly/checkpoints/SwaV_Original_resnet50.pth --exp-dir SwaV_Original --lr-head 0.02
# torchrun --nproc_per_node=4 swav.py
# python evaluate.py --data-dir /Work1/imagenet100/ --pretrained /Work2/Watch_This/Lightly/checkpoints/SwaV_resnet50.pth --exp-dir SwaV --lr-head 0.02

# torchrun --nproc_per_node=4 swag.py
# python evaluate.py --data-dir /Work1/imagenet100/ --pretrained /Work2/Watch_This/Lightly/checkpoints/SwaG_resnet50.pth --exp-dir SwaG --lr-head 0.02

# torchrun --nproc_per_node=4 swag.py --exp-file SwaG_2_4_6 --nmb_crops 2 4 6
# python evaluate.py --data-dir /Work1/imagenet100/ --pretrained /Work2/Watch_This/Lightly/checkpoints/SwaG_2_4_6_resnet50.pth --exp-dir SwaG_2_4_6 --lr-head 0.02

# torchrun --nproc_per_node=4 swag.py --exp-file SwaG_2_4_6 --nmb_crops 2 4 6
# python evaluate.py --data-dir /Work1/imagenet100/ --pretrained /Work2/Watch_This/Lightly/checkpoints/SwaG_2_4_6_resnet50.pth --exp-dir SwaG_2_4_6 --lr-head 0.02

# torchrun --nproc_per_node=4 swag.py --exp-file SwaG_2_6_6 --nmb_crops 2 6 6
# python evaluate.py --data-dir /Work1/imagenet100/ --pretrained /Work2/Watch_This/Lightly/checkpoints/SwaG_2_6_6_resnet50.pth --exp-dir SwaG_2_6_6 --lr-head 0.02

# torchrun --nproc_per_node=4 swag.py --exp-file /Work2/Watch_This/Lightly/checkpoints/SwaG_2_2_6.pth --nmb_crops 2 2 6
# python evaluate.py --data-dir /Work1/imagenet100/ --pretrained /Work2/Watch_This/Lightly/checkpoints/SwaG_2_2_6_resnet50.pth --exp-dir SwaG_2_2_6 --lr-head 0.02

# torchrun --nproc_per_node=4 swag.py --exp-file /Work2/Watch_This/Lightly/checkpoints/SwaG_2_4_6.pth --nmb_crops 2 4 6
# python evaluate.py --data-dir /Work1/imagenet100/ --pretrained /Work2/Watch_This/Lightly/checkpoints/SwaG_2_4_6_resnet50.pth --exp-dir SwaG_2_4_6 --lr-head 0.02

# torchrun --nproc_per_node=4 swag.py --exp-file /Work2/Watch_This/Lightly/checkpoints/SwaG_2_6_0.pth --nmb_crops 2 6 0
# python evaluate.py --data-dir /Work1/imagenet100/ --pretrained /Work2/Watch_This/Lightly/checkpoints/SwaG_2_6_0_resnet50.pth --exp-dir SwaG_2_6_0 --lr-head 0.02

# torchrun --nproc_per_node=4 swag.py --exp-file /Work2/Watch_This/Lightly/checkpoints/SwaG_1_7_0.pth --nmb_crops 1 7 0
# python evaluate.py --data-dir /Work1/imagenet100/ --pretrained /Work2/Watch_This/Lightly/checkpoints/SwaG_1_7_0_resnet50.pth --exp-dir SwaG_1_7_0 --lr-head 0.02

# torchrun --nproc_per_node=4 swag.py --exp-file /Work2/Watch_This/Lightly/checkpoints/SwaG_2_6_2.pth --nmb_crops 2 6 2
# python evaluate.py --data-dir /Work1/imagenet100/ --pretrained /Work2/Watch_This/Lightly/checkpoints/SwaG_2_6_2_resnet50.pth --exp-dir SwaG_2_6_2 --lr-head 0.02

# torchrun --nproc_per_node=4 barlow_twins.py --exp-file /Work2/Watch_This/Lightly/checkpoints/barlow_twins_1_color_flip.pth --nmb_gen 1 --flip_aug --color_aug
# python evaluate.py --data-dir /Work1/imagenet100/ --pretrained /Work2/Watch_This/Lightly/checkpoints/barlow_twins_1_color_flip_resnet50.pth --exp-dir barlow_twins_1_color_flip --lr-head 0.02

# torchrun --nproc_per_node=4 barlow_twins.py --exp-file /Work2/Watch_This/Lightly/checkpoints/barlow_twins_1_flip.pth --nmb_gen 1 --flip_aug 
# python evaluate.py --data-dir /Work1/imagenet100/ --pretrained /Work2/Watch_This/Lightly/checkpoints/barlow_twins_1_flip_resnet50.pth --exp-dir barlow_twins_1_flip --lr-head 0.02

### BARLOW TWINS

# torchrun --nproc_per_node=4 barlow_twins.py --exp-file /Work2/Watch_This/Lightly/model_exp/checkpoints/barlow_twins_0.pth --nmb-gen 0 --flip-aug --color-aug --batch-size 768
# python evaluate.py --data-dir /Work1/imagenet100/ --pretrained /Work2/Watch_This/Lightly/model_exp/checkpoints/barlow_twins_0_resnet50.pth --exp-dir model_exp/barlow_twins_0 --lr-head 0.02

# torchrun --nproc_per_node=4 barlow_twins.py --exp-file /Work2/Watch_This/Lightly/model_exp/checkpoints/barlow_twins_1.pth --nmb-gen 1 --batch-size 768
# python evaluate.py --data-dir /Work1/imagenet100/ --pretrained /Work2/Watch_This/Lightly/model_exp/checkpoints/barlow_twins_1_resnet50.pth --exp-dir model_exp/barlow_twins_1 --lr-head 0.02

# torchrun --nproc_per_node=4 barlow_twins.py --exp-file /Work2/Watch_This/Lightly/model_exp/checkpoints/barlow_twins_2.pth --nmb-gen 2 --batch-size 512
# python evaluate.py --data-dir /Work1/imagenet100/ --pretrained /Work2/Watch_This/Lightly/model_exp/checkpoints/barlow_twins_2_resnet50.pth --exp-dir model_exp/barlow_twins_2 --lr-head 0.02

# torchrun --nproc_per_node=4 barlow_twins.py --exp-file /Work2/Watch_This/Lightly/model_exp/checkpoints/barlow_twins_3.pth --nmb-gen 3 --batch-size 384
# python evaluate.py --data-dir /Work1/imagenet100/ --pretrained /Work2/Watch_This/Lightly/model_exp/checkpoints/barlow_twins_3_resnet50.pth --exp-dir model_exp/barlow_twins_3 --lr-head 0.02

# torchrun --nproc_per_node=4 barlow_twins.py --exp-file /Work2/Watch_This/Lightly/model_exp/checkpoints/barlow_twins_4.pth --nmb-gen 4 --batch-size 307
# python evaluate.py --data-dir /Work1/imagenet100/ --pretrained /Work2/Watch_This/Lightly/model_exp/checkpoints/barlow_twins_4_resnet50.pth --exp-dir model_exp/barlow_twins_4 --lr-head 0.02

# torchrun --nproc_per_node=4 barlow_twins.py --exp-file /Work2/Watch_This/Lightly/model_exp/checkpoints/barlow_twins_5.pth --nmb-gen 5 --batch-size 256
# python evaluate.py --data-dir /Work1/imagenet100/ --pretrained /Work2/Watch_This/Lightly/model_exp/checkpoints/barlow_twins_5_resnet50.pth --exp-dir model_exp/barlow_twins_5 --lr-head 0.02

### VICReg

# torchrun --nproc_per_node=4 vicreg.py --exp-file /Work2/Watch_This/Lightly/model_exp/checkpoints/vicreg_0.pth --nmb-gen 0 --flip-aug --color-aug --batch-size 768
python evaluate.py --data-dir /Work1/imagenet100/ --pretrained /Work2/Watch_This/Lightly/model_exp/checkpoints/vicreg_0_resnet50.pth --exp-dir model_exp/vicreg_0 --lr-head 0.02

torchrun --nproc_per_node=4 vicreg.py --exp-file /Work2/Watch_This/Lightly/model_exp/checkpoints/vicreg_1.pth --nmb-gen 1 --batch-size 768
python evaluate.py --data-dir /Work1/imagenet100/ --pretrained /Work2/Watch_This/Lightly/model_exp/checkpoints/vicreg_1_resnet50.pth --exp-dir model_exp/vicreg_1 --lr-head 0.02

torchrun --nproc_per_node=4 vicreg.py --exp-file /Work2/Watch_This/Lightly/model_exp/checkpoints/vicreg_2.pth --nmb-gen 2 --batch-size 512
python evaluate.py --data-dir /Work1/imagenet100/ --pretrained /Work2/Watch_This/Lightly/model_exp/checkpoints/vicreg_2_resnet50.pth --exp-dir model_exp/vicreg_2 --lr-head 0.02

torchrun --nproc_per_node=4 vicreg.py --exp-file /Work2/Watch_This/Lightly/model_exp/checkpoints/vicreg_3.pth --nmb-gen 3 --batch-size 384
python evaluate.py --data-dir /Work1/imagenet100/ --pretrained /Work2/Watch_This/Lightly/model_exp/checkpoints/vicreg_3_resnet50.pth --exp-dir model_exp/vicreg_3 --lr-head 0.02

# torchrun --nproc_per_node=4 vicreg.py --exp-file /Work2/Watch_This/Lightly/model_exp/checkpoints/vicreg_4.pth --nmb-gen 4 --batch-size 307
# python evaluate.py --data-dir /Work1/imagenet100/ --pretrained /Work2/Watch_This/Lightly/model_exp/checkpoints/vicreg_4_resnet50.pth --exp-dir model_exp/vicreg_4 --lr-head 0.02

# torchrun --nproc_per_node=4 vicreg.py --exp-file /Work2/Watch_This/Lightly/model_exp/checkpoints/vicreg_5.pth --nmb-gen 5 --batch-size 256
# python evaluate.py --data-dir /Work1/imagenet100/ --pretrained /Work2/Watch_This/Lightly/model_exp/checkpoints/vicreg_5_resnet50.pth --exp-dir model_exp/vicreg_5 --lr-head 0.02

### SwaV

# torchrun --nproc_per_node=4 swav.py --exp-file /Work2/Watch_This/Lightly/model_exp/checkpoints/swav_0.pth --nmb-gen 0 --flip-aug --color-aug --batch-size 768
# python evaluate.py --data-dir /Work1/imagenet100/ --pretrained /Work2/Watch_This/Lightly/model_exp/checkpoints/swav_0_resnet50.pth --exp-dir model_exp/swav_0 --lr-head 0.02

# torchrun --nproc_per_node=4 swav.py --exp-file /Work2/Watch_This/Lightly/model_exp/checkpoints/swav_1.pth --nmb-gen 1 --batch-size 768
# python evaluate.py --data-dir /Work1/imagenet100/ --pretrained /Work2/Watch_This/Lightly/model_exp/checkpoints/swav_1_resnet50.pth --exp-dir model_exp/swav_1 --lr-head 0.02

# torchrun --nproc_per_node=4 swav.py --exp-file /Work2/Watch_This/Lightly/model_exp/checkpoints/swav_2.pth --nmb-gen 2 --batch-size 512
# python evaluate.py --data-dir /Work1/imagenet100/ --pretrained /Work2/Watch_This/Lightly/model_exp/checkpoints/swav_2_resnet50.pth --exp-dir model_exp/swav_2 --lr-head 0.02

# torchrun --nproc_per_node=4 swav.py --exp-file /Work2/Watch_This/Lightly/model_exp/checkpoints/swav_3.pth --nmb-gen 3 --batch-size 384
# python evaluate.py --data-dir /Work1/imagenet100/ --pretrained /Work2/Watch_This/Lightly/model_exp/checkpoints/swav_3_resnet50.pth --exp-dir model_exp/swav_3 --lr-head 0.02

# torchrun --nproc_per_node=4 swav.py --exp-file /Work2/Watch_This/Lightly/model_exp/checkpoints/swav_4.pth --nmb-gen 4 --batch-size 307
# python evaluate.py --data-dir /Work1/imagenet100/ --pretrained /Work2/Watch_This/Lightly/model_exp/checkpoints/swav_4_resnet50.pth --exp-dir model_exp/swav_4 --lr-head 0.02

# torchrun --nproc_per_node=4 swav.py --exp-file /Work2/Watch_This/Lightly/model_exp/checkpoints/swav_5.pth --nmb-gen 5 --batch-size 256
# python evaluate.py --data-dir /Work1/imagenet100/ --pretrained /Work2/Watch_This/Lightly/model_exp/checkpoints/swav_5_resnet50.pth --exp-dir model_exp/swav_5 --lr-head 0.02


### BYOL

# torchrun --nproc_per_node=4 byol.py --exp-file /Work2/Watch_This/Lightly/model_exp/checkpoints/byol_0.pth --nmb-gen 0 --flip-aug --color-aug --batch-size 768
# python evaluate.py --data-dir /Work1/imagenet100/ --pretrained /Work2/Watch_This/Lightly/model_exp/checkpoints/byol_0_resnet50.pth --exp-dir model_exp/byol_0 --lr-head 0.02

# torchrun --nproc_per_node=4 byol.py --exp-file /Work2/Watch_This/Lightly/model_exp/checkpoints/byol_1.pth --nmb-gen 1 --batch-size 768
# python evaluate.py --data-dir /Work1/imagenet100/ --pretrained /Work2/Watch_This/Lightly/model_exp/checkpoints/byol_1_resnet50.pth --exp-dir model_exp/byol_1 --lr-head 0.02

# torchrun --nproc_per_node=4 byol.py --exp-file /Work2/Watch_This/Lightly/model_exp/checkpoints/byol_2.pth --nmb-gen 2 --batch-size 512
# python evaluate.py --data-dir /Work1/imagenet100/ --pretrained /Work2/Watch_This/Lightly/model_exp/checkpoints/byol_2_resnet50.pth --exp-dir model_exp/byol_2 --lr-head 0.02

# torchrun --nproc_per_node=4 byol.py --exp-file /Work2/Watch_This/Lightly/model_exp/checkpoints/byol_3.pth --nmb-gen 3 --batch-size 384
# python evaluate.py --data-dir /Work1/imagenet100/ --pretrained /Work2/Watch_This/Lightly/model_exp/checkpoints/byol_3_resnet50.pth --exp-dir model_exp/byol_3 --lr-head 0.02

# torchrun --nproc_per_node=4 byol.py --exp-file /Work2/Watch_This/Lightly/model_exp/checkpoints/byol_4.pth --nmb-gen 4 --batch-size 307
# python evaluate.py --data-dir /Work1/imagenet100/ --pretrained /Work2/Watch_This/Lightly/model_exp/checkpoints/byol_4_resnet50.pth --exp-dir model_exp/byol_4 --lr-head 0.02

# torchrun --nproc_per_node=4 byol.py --exp-file /Work2/Watch_This/Lightly/model_exp/checkpoints/byol_5.pth --nmb-gen 5 --batch-size 256
# python evaluate.py --data-dir /Work1/imagenet100/ --pretrained /Work2/Watch_This/Lightly/model_exp/checkpoints/byol_5_resnet50.pth --exp-dir model_exp/byol_5 --lr-head 0.02


### SimCLR

# torchrun --nproc_per_node=4 simclr.py --exp-file /Work2/Watch_This/Lightly/model_exp/checkpoints/simclr_0.pth --nmb-gen 0 --flip-aug --color-aug --batch-size 768
# python evaluate.py --data-dir /Work1/imagenet100/ --pretrained /Work2/Watch_This/Lightly/model_exp/checkpoints/simclr_0_resnet50.pth --exp-dir model_exp/simclr_0 --lr-head 0.02

# torchrun --nproc_per_node=4 simclr.py --exp-file /Work2/Watch_This/Lightly/model_exp/checkpoints/simclr_1.pth --nmb-gen 1 --batch-size 768
# python evaluate.py --data-dir /Work1/imagenet100/ --pretrained /Work2/Watch_This/Lightly/model_exp/checkpoints/simclr_1_resnet50.pth --exp-dir model_exp/simclr_1 --lr-head 0.02

# torchrun --nproc_per_node=4 simclr.py --exp-file /Work2/Watch_This/Lightly/model_exp/checkpoints/simclr_2.pth --nmb-gen 2 --batch-size 512
# python evaluate.py --data-dir /Work1/imagenet100/ --pretrained /Work2/Watch_This/Lightly/model_exp/checkpoints/simclr_2_resnet50.pth --exp-dir model_exp/simclr_2 --lr-head 0.02

# torchrun --nproc_per_node=4 simclr.py --exp-file /Work2/Watch_This/Lightly/model_exp/checkpoints/simclr_3.pth --nmb-gen 3 --batch-size 384
# python evaluate.py --data-dir /Work1/imagenet100/ --pretrained /Work2/Watch_This/Lightly/model_exp/checkpoints/simclr_3_resnet50.pth --exp-dir model_exp/simclr_3 --lr-head 0.02

# torchrun --nproc_per_node=4 simclr.py --exp-file /Work2/Watch_This/Lightly/model_exp/checkpoints/simclr_4.pth --nmb-gen 4 --batch-size 307
# python evaluate.py --data-dir /Work1/imagenet100/ --pretrained /Work2/Watch_This/Lightly/model_exp/checkpoints/simclr_4_resnet50.pth --exp-dir model_exp/simclr_4 --lr-head 0.02

# torchrun --nproc_per_node=4 simclr.py --exp-file /Work2/Watch_This/Lightly/model_exp/checkpoints/simclr_5.pth --nmb-gen 5 --batch-size 256
# python evaluate.py --data-dir /Work1/imagenet100/ --pretrained /Work2/Watch_This/Lightly/model_exp/checkpoints/simclr_5_resnet50.pth --exp-dir model_exp/simclr_5 --lr-head 0.02




