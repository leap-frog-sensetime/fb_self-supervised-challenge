#!/bin/sh

test_data_file='path_to_your_local_extracted_testSplit_features_voc07/features.npy'
test_targets_data_file='path_to_your_local_extracted_testSplit_labels_voc07/labels.npy'
output_path=./trained_svm/
test_log_name=test_svm_voc07.log


srun -p AD python tools/svm/test_svm.py \
  --data_file=${test_data_file} \
  --targets_data_file=${test_targets_data_file} \
  --json_targets ./test_targets.json \
  --costs_list "1.0,10.0" \
  --output_path=${output_path} \
  --generate_json 1 \
  2>&1|tee ${output_path}/${test_log_name}