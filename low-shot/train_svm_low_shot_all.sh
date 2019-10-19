#!/bin/sh

#you could change different low-shot sampling features through change this 'train_targets_path'
train_targets_path=./sampled_targets

train_data_file='path_to_your_local_extracted_trainvalSplit_features_voc07/features.npy'
train_targets_data_file=${train_targets_path}/targets_sample1_k1.npy
#k_values "1,2,4,8,16,32,64,96" input this and sample 1-5 to train_target_data_file.

output_path=./trained_low_shot_svm_new
mkdir -p ${output_path}

train_log_name=train_low_shot_log.log

srun -p AD python tools/svm/train_svm_low_shot.py \
  --data_file=${train_data_file}  \
  --targets_data_file=${train_targets_data_file} \
  --costs_list "1.0,10.0" \
  --output_path=${output_path} \
  2>&1|tee ${output_path}/${train_log_name}

train_targets_data_file=${train_targets_path}/targets_sample1_k2.npy
#k_values "1,2,4,8,16,32,64,96" input this and sample 1-5 to train_target_data_file.

srun -p AD python tools/svm/train_svm_low_shot.py \
  --data_file=${train_data_file}  \
  --targets_data_file=${train_targets_data_file} \
  --costs_list "1.0,10.0" \
  --output_path=${output_path} \
  2>&1|tee ${output_path}/${train_log_name}

train_targets_data_file=${train_targets_path}/targets_sample1_k4.npy
#k_values "1,2,4,8,16,32,64,96" input this and sample 1-5 to train_target_data_file.

srun -p AD python tools/svm/train_svm_low_shot.py \
  --data_file=${train_data_file}  \
  --targets_data_file=${train_targets_data_file} \
  --costs_list "1.0,10.0" \
  --output_path=${output_path} \
  2>&1|tee ${output_path}/${train_log_name}

train_targets_data_file=${train_targets_path}/targets_sample1_k8.npy
#k_values "1,2,4,8,16,32,64,96" input this and sample 1-5 to train_target_data_file.

srun -p AD python tools/svm/train_svm_low_shot.py \
  --data_file=${train_data_file}  \
  --targets_data_file=${train_targets_data_file} \
  --costs_list "1.0,10.0" \
  --output_path=${output_path} \
  2>&1|tee ${output_path}/${train_log_name}

train_targets_data_file=${train_targets_path}/targets_sample1_k16.npy
#k_values "1,2,4,8,16,32,64,96" input this and sample 1-5 to train_target_data_file.

srun -p AD python tools/svm/train_svm_low_shot.py \
  --data_file=${train_data_file}  \
  --targets_data_file=${train_targets_data_file} \
  --costs_list "1.0,10.0" \
  --output_path=${output_path} \
  2>&1|tee ${output_path}/${train_log_name}

train_targets_data_file=${train_targets_path}/targets_sample1_k32.npy
#k_values "1,2,4,8,16,32,64,96" input this and sample 1-5 to train_target_data_file.

srun -p AD python tools/svm/train_svm_low_shot.py \
  --data_file=${train_data_file}  \
  --targets_data_file=${train_targets_data_file} \
  --costs_list "1.0,10.0" \
  --output_path=${output_path} \
  2>&1|tee ${output_path}/${train_log_name}

train_targets_data_file=${train_targets_path}/targets_sample1_k64.npy
#k_values "1,2,4,8,16,32,64,96" input this and sample 1-5 to train_target_data_file.

srun -p AD python tools/svm/train_svm_low_shot.py \
  --data_file=${train_data_file}  \
  --targets_data_file=${train_targets_data_file} \
  --costs_list "1.0,10.0" \
  --output_path=${output_path} \
  2>&1|tee ${output_path}/${train_log_name}

train_targets_data_file=${train_targets_path}/targets_sample1_k96.npy
#k_values "1,2,4,8,16,32,64,96" input this and sample 1-5 to train_target_data_file.

srun -p AD python tools/svm/train_svm_low_shot.py \
  --data_file=${train_data_file}  \
  --targets_data_file=${train_targets_data_file} \
  --costs_list "1.0,10.0" \
  --output_path=${output_path} \
  2>&1|tee ${output_path}/${train_log_name}

train_targets_data_file=${train_targets_path}/targets_sample2_k1.npy
#k_values "1,2,4,8,16,32,64,96" input this and sample 1-5 to train_target_data_file.

srun -p AD python tools/svm/train_svm_low_shot.py \
  --data_file=${train_data_file}  \
  --targets_data_file=${train_targets_data_file} \
  --costs_list "1.0,10.0" \
  --output_path=${output_path} \
  2>&1|tee ${output_path}/${train_log_name}

train_targets_data_file=${train_targets_path}/targets_sample2_k2.npy
#k_values "1,2,4,8,16,32,64,96" input this and sample 1-5 to train_target_data_file.

srun -p AD python tools/svm/train_svm_low_shot.py \
  --data_file=${train_data_file}  \
  --targets_data_file=${train_targets_data_file} \
  --costs_list "1.0,10.0" \
  --output_path=${output_path} \
  2>&1|tee ${output_path}/${train_log_name}

train_targets_data_file=${train_targets_path}/targets_sample2_k4.npy
#k_values "1,2,4,8,16,32,64,96" input this and sample 1-5 to train_target_data_file.

srun -p AD python tools/svm/train_svm_low_shot.py \
  --data_file=${train_data_file}  \
  --targets_data_file=${train_targets_data_file} \
  --costs_list "1.0,10.0" \
  --output_path=${output_path} \
  2>&1|tee ${output_path}/${train_log_name}

train_targets_data_file=${train_targets_path}/targets_sample2_k8.npy
#k_values "1,2,4,8,16,32,64,96" input this and sample 1-5 to train_target_data_file.

srun -p AD python tools/svm/train_svm_low_shot.py \
  --data_file=${train_data_file}  \
  --targets_data_file=${train_targets_data_file} \
  --costs_list "1.0,10.0" \
  --output_path=${output_path} \
  2>&1|tee ${output_path}/${train_log_name}

train_targets_data_file=${train_targets_path}/targets_sample2_k16.npy
#k_values "1,2,4,8,16,32,64,96" input this and sample 1-5 to train_target_data_file.

srun -p AD python tools/svm/train_svm_low_shot.py \
  --data_file=${train_data_file}  \
  --targets_data_file=${train_targets_data_file} \
  --costs_list "1.0,10.0" \
  --output_path=${output_path} \
  2>&1|tee ${output_path}/${train_log_name}

train_targets_data_file=${train_targets_path}/targets_sample2_k32.npy
#k_values "1,2,4,8,16,32,64,96" input this and sample 1-5 to train_target_data_file.

srun -p AD python tools/svm/train_svm_low_shot.py \
  --data_file=${train_data_file}  \
  --targets_data_file=${train_targets_data_file} \
  --costs_list "1.0,10.0" \
  --output_path=${output_path} \
  2>&1|tee ${output_path}/${train_log_name}

train_targets_data_file=${train_targets_path}/targets_sample2_k64.npy
#k_values "1,2,4,8,16,32,64,96" input this and sample 1-5 to train_target_data_file.

srun -p AD python tools/svm/train_svm_low_shot.py \
  --data_file=${train_data_file}  \
  --targets_data_file=${train_targets_data_file} \
  --costs_list "1.0,10.0" \
  --output_path=${output_path} \
  2>&1|tee ${output_path}/${train_log_name}

train_targets_data_file=${train_targets_path}/targets_sample2_k96.npy
#k_values "1,2,4,8,16,32,64,96" input this and sample 1-5 to train_target_data_file.

srun -p AD python tools/svm/train_svm_low_shot.py \
  --data_file=${train_data_file}  \
  --targets_data_file=${train_targets_data_file} \
  --costs_list "1.0,10.0" \
  --output_path=${output_path} \
  2>&1|tee ${output_path}/${train_log_name}

train_targets_data_file=${train_targets_path}/targets_sample3_k1.npy
#k_values "1,2,4,8,16,32,64,96" input this and sample 1-5 to train_target_data_file.

srun -p AD python tools/svm/train_svm_low_shot.py \
  --data_file=${train_data_file}  \
  --targets_data_file=${train_targets_data_file} \
  --costs_list "1.0,10.0" \
  --output_path=${output_path} \
  2>&1|tee ${output_path}/${train_log_name}

train_targets_data_file=${train_targets_path}/targets_sample3_k2.npy
#k_values "1,2,4,8,16,32,64,96" input this and sample 1-5 to train_target_data_file.

srun -p AD python tools/svm/train_svm_low_shot.py \
  --data_file=${train_data_file}  \
  --targets_data_file=${train_targets_data_file} \
  --costs_list "1.0,10.0" \
  --output_path=${output_path} \
  2>&1|tee ${output_path}/${train_log_name}

train_targets_data_file=${train_targets_path}/targets_sample3_k4.npy
#k_values "1,2,4,8,16,32,64,96" input this and sample 1-5 to train_target_data_file.

srun -p AD python tools/svm/train_svm_low_shot.py \
  --data_file=${train_data_file}  \
  --targets_data_file=${train_targets_data_file} \
  --costs_list "1.0,10.0" \
  --output_path=${output_path} \
  2>&1|tee ${output_path}/${train_log_name}

train_targets_data_file=${train_targets_path}/targets_sample3_k8.npy
#k_values "1,2,4,8,16,32,64,96" input this and sample 1-5 to train_target_data_file.

srun -p AD python tools/svm/train_svm_low_shot.py \
  --data_file=${train_data_file}  \
  --targets_data_file=${train_targets_data_file} \
  --costs_list "1.0,10.0" \
  --output_path=${output_path} \
  2>&1|tee ${output_path}/${train_log_name}

train_targets_data_file=${train_targets_path}/targets_sample3_k16.npy
#k_values "1,2,4,8,16,32,64,96" input this and sample 1-5 to train_target_data_file.

srun -p AD python tools/svm/train_svm_low_shot.py \
  --data_file=${train_data_file}  \
  --targets_data_file=${train_targets_data_file} \
  --costs_list "1.0,10.0" \
  --output_path=${output_path} \
  2>&1|tee ${output_path}/${train_log_name}

train_targets_data_file=${train_targets_path}/targets_sample3_k32.npy
#k_values "1,2,4,8,16,32,64,96" input this and sample 1-5 to train_target_data_file.

srun -p AD python tools/svm/train_svm_low_shot.py \
  --data_file=${train_data_file}  \
  --targets_data_file=${train_targets_data_file} \
  --costs_list "1.0,10.0" \
  --output_path=${output_path} \
  2>&1|tee ${output_path}/${train_log_name}

train_targets_data_file=${train_targets_path}/targets_sample3_k64.npy
#k_values "1,2,4,8,16,32,64,96" input this and sample 1-5 to train_target_data_file.

srun -p AD python tools/svm/train_svm_low_shot.py \
  --data_file=${train_data_file}  \
  --targets_data_file=${train_targets_data_file} \
  --costs_list "1.0,10.0" \
  --output_path=${output_path} \
  2>&1|tee ${output_path}/${train_log_name}

train_targets_data_file=${train_targets_path}/targets_sample3_k96.npy
#k_values "1,2,4,8,16,32,64,96" input this and sample 1-5 to train_target_data_file.

srun -p AD python tools/svm/train_svm_low_shot.py \
  --data_file=${train_data_file}  \
  --targets_data_file=${train_targets_data_file} \
  --costs_list "1.0,10.0" \
  --output_path=${output_path} \
  2>&1|tee ${output_path}/${train_log_name}

train_targets_data_file=${train_targets_path}/targets_sample4_k1.npy
#k_values "1,2,4,8,16,32,64,96" input this and sample 1-5 to train_target_data_file.

srun -p AD python tools/svm/train_svm_low_shot.py \
  --data_file=${train_data_file}  \
  --targets_data_file=${train_targets_data_file} \
  --costs_list "1.0,10.0" \
  --output_path=${output_path} \
  2>&1|tee ${output_path}/${train_log_name}

train_targets_data_file=${train_targets_path}/targets_sample4_k2.npy
#k_values "1,2,4,8,16,32,64,96" input this and sample 1-5 to train_target_data_file.

srun -p AD python tools/svm/train_svm_low_shot.py \
  --data_file=${train_data_file}  \
  --targets_data_file=${train_targets_data_file} \
  --costs_list "1.0,10.0" \
  --output_path=${output_path} \
  2>&1|tee ${output_path}/${train_log_name}

train_targets_data_file=${train_targets_path}/targets_sample4_k4.npy
#k_values "1,2,4,8,16,32,64,96" input this and sample 1-5 to train_target_data_file.

srun -p AD python tools/svm/train_svm_low_shot.py \
  --data_file=${train_data_file}  \
  --targets_data_file=${train_targets_data_file} \
  --costs_list "1.0,10.0" \
  --output_path=${output_path} \
  2>&1|tee ${output_path}/${train_log_name}

train_targets_data_file=${train_targets_path}/targets_sample4_k8.npy
#k_values "1,2,4,8,16,32,64,96" input this and sample 1-5 to train_target_data_file.

srun -p AD python tools/svm/train_svm_low_shot.py \
  --data_file=${train_data_file}  \
  --targets_data_file=${train_targets_data_file} \
  --costs_list "1.0,10.0" \
  --output_path=${output_path} \
  2>&1|tee ${output_path}/${train_log_name}

train_targets_data_file=${train_targets_path}/targets_sample4_k16.npy
#k_values "1,2,4,8,16,32,64,96" input this and sample 1-5 to train_target_data_file.

srun -p AD python tools/svm/train_svm_low_shot.py \
  --data_file=${train_data_file}  \
  --targets_data_file=${train_targets_data_file} \
  --costs_list "1.0,10.0" \
  --output_path=${output_path} \
  2>&1|tee ${output_path}/${train_log_name}

train_targets_data_file=${train_targets_path}/targets_sample4_k32.npy
#k_values "1,2,4,8,16,32,64,96" input this and sample 1-5 to train_target_data_file.

srun -p AD python tools/svm/train_svm_low_shot.py \
  --data_file=${train_data_file}  \
  --targets_data_file=${train_targets_data_file} \
  --costs_list "1.0,10.0" \
  --output_path=${output_path} \
  2>&1|tee ${output_path}/${train_log_name}

train_targets_data_file=${train_targets_path}/targets_sample4_k64.npy
#k_values "1,2,4,8,16,32,64,96" input this and sample 1-5 to train_target_data_file.

srun -p AD python tools/svm/train_svm_low_shot.py \
  --data_file=${train_data_file}  \
  --targets_data_file=${train_targets_data_file} \
  --costs_list "1.0,10.0" \
  --output_path=${output_path} \
  2>&1|tee ${output_path}/${train_log_name}

train_targets_data_file=${train_targets_path}/targets_sample4_k96.npy
#k_values "1,2,4,8,16,32,64,96" input this and sample 1-5 to train_target_data_file.

srun -p AD python tools/svm/train_svm_low_shot.py \
  --data_file=${train_data_file}  \
  --targets_data_file=${train_targets_data_file} \
  --costs_list "1.0,10.0" \
  --output_path=${output_path} \
  2>&1|tee ${output_path}/${train_log_name}

train_targets_data_file=${train_targets_path}/targets_sample5_k1.npy
#k_values "1,2,4,8,16,32,64,96" input this and sample 1-5 to train_target_data_file.

srun -p AD python tools/svm/train_svm_low_shot.py \
  --data_file=${train_data_file}  \
  --targets_data_file=${train_targets_data_file} \
  --costs_list "1.0,10.0" \
  --output_path=${output_path} \
  2>&1|tee ${output_path}/${train_log_name}

train_targets_data_file=${train_targets_path}/targets_sample5_k2.npy
#k_values "1,2,4,8,16,32,64,96" input this and sample 1-5 to train_target_data_file.

srun -p AD python tools/svm/train_svm_low_shot.py \
  --data_file=${train_data_file}  \
  --targets_data_file=${train_targets_data_file} \
  --costs_list "1.0,10.0" \
  --output_path=${output_path} \
  2>&1|tee ${output_path}/${train_log_name}

train_targets_data_file=${train_targets_path}/targets_sample5_k4.npy
#k_values "1,2,4,8,16,32,64,96" input this and sample 1-5 to train_target_data_file.

srun -p AD python tools/svm/train_svm_low_shot.py \
  --data_file=${train_data_file}  \
  --targets_data_file=${train_targets_data_file} \
  --costs_list "1.0,10.0" \
  --output_path=${output_path} \
  2>&1|tee ${output_path}/${train_log_name}

train_targets_data_file=${train_targets_path}/targets_sample5_k8.npy
#k_values "1,2,4,8,16,32,64,96" input this and sample 1-5 to train_target_data_file.

srun -p AD python tools/svm/train_svm_low_shot.py \
  --data_file=${train_data_file}  \
  --targets_data_file=${train_targets_data_file} \
  --costs_list "1.0,10.0" \
  --output_path=${output_path} \
  2>&1|tee ${output_path}/${train_log_name}

train_targets_data_file=${train_targets_path}/targets_sample5_k16.npy
#k_values "1,2,4,8,16,32,64,96" input this and sample 1-5 to train_target_data_file.

srun -p AD python tools/svm/train_svm_low_shot.py \
  --data_file=${train_data_file}  \
  --targets_data_file=${train_targets_data_file} \
  --costs_list "1.0,10.0" \
  --output_path=${output_path} \
  2>&1|tee ${output_path}/${train_log_name}

train_targets_data_file=${train_targets_path}/targets_sample5_k32.npy
#k_values "1,2,4,8,16,32,64,96" input this and sample 1-5 to train_target_data_file.

srun -p AD python tools/svm/train_svm_low_shot.py \
  --data_file=${train_data_file}  \
  --targets_data_file=${train_targets_data_file} \
  --costs_list "1.0,10.0" \
  --output_path=${output_path} \
  2>&1|tee ${output_path}/${train_log_name}

train_targets_data_file=${train_targets_path}/targets_sample5_k64.npy
#k_values "1,2,4,8,16,32,64,96" input this and sample 1-5 to train_target_data_file.

srun -p AD python tools/svm/train_svm_low_shot.py \
  --data_file=${train_data_file}  \
  --targets_data_file=${train_targets_data_file} \
  --costs_list "1.0,10.0" \
  --output_path=${output_path} \
  2>&1|tee ${output_path}/${train_log_name}

train_targets_data_file=${train_targets_path}/targets_sample5_k96.npy
#k_values "1,2,4,8,16,32,64,96" input this and sample 1-5 to train_target_data_file.

srun -p AD python tools/svm/train_svm_low_shot.py \
  --data_file=${train_data_file}  \
  --targets_data_file=${train_targets_data_file} \
  --costs_list "1.0,10.0" \
  --output_path=${output_path} \
  2>&1|tee ${output_path}/${train_log_name}