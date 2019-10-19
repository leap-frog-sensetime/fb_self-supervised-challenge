#!/bin/sh

output_path=./sampled_targets_new
mkdir -p ${output_path}

srun python tools/create_voc_low_shot_samples.py \
    --targets_data_file path_to_your_local_extracted_trainvalSplit_labels_voc07/targets.npy \
    --output_path ${output_path} \
    --k_values "1,2,4,8,16,32,64,96" \
    --num_samples 5
