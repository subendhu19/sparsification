#!/usr/bin/env bash

python create_configs.py

sbatch --array=1-2 --partition=titanx-long --mem 80000 --gres=gpu:4 bin/run_snli_exp.sh