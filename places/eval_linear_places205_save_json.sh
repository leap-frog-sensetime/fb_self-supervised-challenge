DATAPATH_IMAGENET='/mnt/lustre/share/images'
DATAPATH_PLACES='/mnt/lustre/dongrunmin/data'
DATAPATH_PASCAL='/mnt/lustre/dongrunmin/data/VOCdevkit/VOC2007'

#choose and change your dataset path.
# Places205 dataset
EXP='./exp'
mkdir -p $EXP

EXP_JSON='./submit_json/'
mkdir -p $EXP_JSON


srun -p AD -n1 --gres=gpu:1 --ntasks-per-node=1 \
  python eval_linear.py --conv 12 --pretrained ./your_local_model_path/checkpoint.pth.tar \
  --checkpoint ./trained_places_model_local_path/places205_linear.pth.tar \
  --is_save_json true --json_save_path $EXP_JSON \
  --json_save_name places_json_preds_drop24_rand4_conv12.json \
  --sobel true --dump_path $EXP --data_path $DATAPATH_PLACES --batch_size 64 --lr 0.01 --wd 0.00001 --nepochs 10
