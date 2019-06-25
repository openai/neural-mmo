#Sets CUDA flags for GPUs to use. These
#are the two not plugged into the display
#on my 3-card machine. Also, the GPUs should
#be comparably fast, but can be different models.
CUDA_VISIBLE_DEVICES=0,2 python Forge.py
