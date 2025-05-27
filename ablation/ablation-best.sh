#!/usr/bin/env bash

#SBATCH -p fnlp-4090d
#SBATCH --job-name=lzjjin-cs-best
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=12G
#SBATCH --output=/remote-home1/lzjjin/project/fudan-course/DATA130051.01/PJ2/PJ2-1/log/output-best.log
#SBATCH --error=/remote-home1/lzjjin/project/fudan-course/DATA130051.01/PJ2/PJ2-1/log/error-best.log
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
  --batch_size 32 \
  --arch facebook/convnext-base-224  \
  --pretrained y \
  --lr 1e-3  \
  --epochs 500 \
  --plot_dir plot/best \
  --save_dir save \
