# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


#choose the dataset you want to extract features from
DATAPATH_IMAGENET='/mnt/lustre/share/images'
DATAPATH_PLACES='/mnt/lustre/dongrunmin/data'
DATAPATH_PASCAL='/mnt/lustre/dongrunmin/data/VOCdevkit/VOC2007'

#you could choose which layer to extract feature representations. (max conv 13 under our model -- vgg16 as backbone)
conv=13
#ouput path
EXP=./exp/conv${conv}
mkdir -p $EXP
#please set the args --pretrained to your local path of deepercluster model.


srun -p AD -n1 --gres=gpu:1 --ntasks-per-node=1 \
  python -u extract_feature.py\
  --conv ${conv} \
  --pretrained='set_this_to_your_model_saved_local_path/checkpoint.pth.tar' \
  --sobel true --dump_path $EXP --data_path $DATAPATH_PASCAL --batch_size 16 --lr 0.02 --wd 0.00001 --nepochs 60