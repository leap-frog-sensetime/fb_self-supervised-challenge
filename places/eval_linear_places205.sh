# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


DATAPATH_IMAGENET='/mnt/lustre/share/images'
DATAPATH_PLACES='/mnt/lustre/dongrunmin/data'
DATAPATH_PASCAL='/mnt/lustre/dongrunmin/data/VOCdevkit/VOC2007'

# Places205 dataset
EXP='./exp/'
mkdir -p $EXP
srun -p AD -n4 --gres=gpu:4 --ntasks-per-node=4 \
  python eval_linear.py --conv 12 --pretrained your_local_model_path/checkpoint.pth.tar \
  --sobel true --dump_path $EXP --data_path $DATAPATH_PLACES --batch_size 64 --lr 0.01 --wd 0.00001 --nepochs 10 \
  --workers 4  


