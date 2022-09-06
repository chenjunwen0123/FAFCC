##################################################
# Training Config
##################################################
import os

GPU = '0'                   # GPU
workers = 4                 # number of Dataloader workers
epochs = 160                # number of epochs
batch_size = 24             # batch size
learning_rate = 1e-3        # initial learning rate

##################################################
# Model Config
##################################################
image_size = (448, 448)     # size of training images
net = 'inception_mixed_6e'  # feature extractor
num_attentions = 32         # number of attention maps
beta = 5e-2                 # param for update feature centers

##################################################
# Dataset/Path Config
##################################################
tag = 'car'                # 'aircraft', 'bird', 'car', or 'dog'

# saving directory of .ckpt models
save_dir = '/home/CompCars/data/ckpt/'
model_name = 'model_wsdan.ckpt'
log_name = 'train_fpn_w_c.log'

# checkpoint model for resume training
# ckpt = False
ckpt = save_dir + model_name

##################################################
# Eval Config
##################################################
visualize = True
eval_ckpt = save_dir + model_name
eval_savepath = '/home/CompCars/data/visualize/fpnAndW/'