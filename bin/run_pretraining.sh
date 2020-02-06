#!/usr/bin/env bash
#
#SBATCH --job-name=sb_pretraining

#SBATCH -o /home/srongali/Projects/sparsification/logs/sparse_bert_pretraining_%A_%a.out
#SBATCH -e /home/srongali/Projects/sparsification/logs/sparse_bert_pretraining_%A_%a.err

#SBATCH --time=5-00:00

python run_pretraining.py \
    --output_dir=/mnt/nfs/scratch1/srongali/sparsification/pretrained_ckpts/config_${SLURM_ARRAY_TASK_ID} \
    --sparse_config=${SLURM_ARRAY_TASK_ID}\
    --model_name_or_path=bert-base-uncased \
    --do_train \
    --do_lower_case \
    --train_data_file=/mnt/nfs/scratch1/srongali/wikitext/wiki.shuf.train.txt \
    --do_eval \
    --eval_data_file=/mnt/nfs/scratch1/srongali/wikitext/wiki.shuf.valid.txt \
    --evaluate_during_training \
    --logging_steps=25000 \
    --save_steps=25000 \
    --mlm \
    --num_train_epochs=10 \
    --save_total_limit=10 \
    --per_gpu_train_batch_size=3 \
    --per_gpu_eval_batch_size=6