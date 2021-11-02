#!/bin/bash
#SBATCH --partition=gpu-L --gres=gpu:1 --constraint="rtx8000"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=7-00:00:00

python test_t0.py \
--input_file data/nli/gender/
