#!/bin/bash
model=$1
seed=$2
output_dir=$3

CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 python -W ignore -m heareval.predictions.runner ${output_dir}/todo_audioset/hear_configs.${model}_r1_${seed}/* --random_seed ${seed}
