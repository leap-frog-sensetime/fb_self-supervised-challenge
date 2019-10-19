#!/bin/sh

output_path=./your_own_training
mkdir -p ${output_path}

srun python tools/svm/train_svm_kfold.py \
  --data_file path_to_your_local_extracted_trainvalSplit_features_voc07/features.npy \
  --targets_data_file path_to_your_local_extracted_trainvalSplit_labels_voc07/labels.npy \
  --costs_list "1.0,10.0" \
  --output_path ${output_path} \
  2>&1|tee ${output_path}/your_own_training.log
