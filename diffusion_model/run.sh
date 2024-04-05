# CUDA_VISIBLE_DEVICES=0 python blip_0.py
# CUDA_VISIBLE_DEVICES=1 python blip_1.py
# CUDA_VISIBLE_DEVICES=2 python blip_2.py
# CUDA_VISIBLE_DEVICES=3 python blip_3.py

# CUDA_VISIBLE_DEVICES=0 python diffusion_0.py
# CUDA_VISIBLE_DEVICES=1 python diffusion_1.py
# CUDA_VISIBLE_DEVICES=2 python diffusion_2.py
# CUDA_VISIBLE_DEVICES=3 python diffusion_3.py


./run0.sh &
./run1.sh &
./run2.sh &
./run3.sh 