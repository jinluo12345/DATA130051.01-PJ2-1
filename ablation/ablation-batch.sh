#!/usr/bin/env bash

#SBATCH -p fnlp-4090d
#SBATCH --job-name=lzjjin-cs-batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=12G
#SBATCH --output=/remote-home1/lzjjin/project/fudan-course/DATA130051.01/PJ2/PJ2-1/log/output-batch.log
#SBATCH --error=/remote-home1/lzjjin/project/fudan-course/DATA130051.01/PJ2/PJ2-1/log/error-batch.log
source ~/.bashrc
conda init
conda activate flow
which python

work_dir=/remote-home1/lzjjin/project/fudan-course/DATA130051.01/PJ2/PJ2-1/
cd ${work_dir}
export PYTHONPATH="${work_dir}"

# Exps
export HF_ENDPOINT=https://hf-mirror.com
#--arch microsoft/resnet-18 microsoft/resnet-101 pytorch/alexnet \

/remote-home1/lzjjin/anaconda3/envs/flow/bin/python ablation.py \
  --batch_size 256 128 64 32 \
  --arch microsoft/resnet-18 \
  --pretrained y n\
  --lr 1e-3  \
  --epochs 300 \
  --plot_dir plot/batch \
  --save_dir save \
