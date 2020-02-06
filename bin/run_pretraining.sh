#!/usr/bin/env bash
#
#SBATCH --job-name=sb_pretraining
#SBATCH --output=sparse_bert_pretraining.txt
#SBATCH --time=0-01:00

python run_pretraining.py \
    --output_dir=/mnt/nfs/scratch1/srongali/sparsification/pretrained_ckpts/sparse-bert \
    --model_type=sp-bert \
    --model_name_or_path=bert-base-uncased \
    --do_train \
    --do_lower_case \
    --train_data_file=/mnt/nfs/scratch1/abhyuday/data/cl/wiki.shuf.sample.txt \
    --do_eval \
    --eval_data_file=/mnt/nfs/scratch1/abhyuday/data/cl/wiki.shuf.sample.txt \
    --evaluate_during_training \
    --logging_steps=25000 \
    --save_steps=25000 \
    --mlm \
    --num_train_epochs=10 \
    --save_total_limit=10 \
    --per_gpu_train_batch_size=3 \
    --per_gpu_eval_batch_size=6