#set your trained svm model path.
output_path=./trained_low_shot_svm

test_data_file='path_to_your_local_extracted_testSplit_features_voc07/features.npy'
test_targets_data_file='path_to_your_local_extracted_testSplit_label_voc07/labels.npy'

test_log_name=test_low_shot_svm.log

srun python tools/svm/test_svm_low_shot.py \
  --data_file=${test_data_file} \
  --targets_data_file=${test_targets_data_file} \
  --json_targets ./test_targets.json \
  --costs_list "1.0,10.0" \
  --output_path=${output_path} \
  --generate_json 1 \
  --k_values "1,2,4,8,16,32,64,96" \
  --sample_inds "0,1,2,3,4"\
  2>&1|tee ${output_path}/${test_log_name}

srun python tools/svm/aggregate_low_shot_svm_stats.py \
  --output_path=${output_path} \
  --k_values "1,2,4,8,16,32,64,96" \
  --sample_inds "0,1,2,3,4"