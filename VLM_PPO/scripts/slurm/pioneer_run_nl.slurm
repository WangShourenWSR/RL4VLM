#!/bin/bash
#SBATCH --job-name=numberline
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --account=xxh584_csds600
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=48:00:00

# 可选：激活你的 Conda 环境
source ~/.bashrc
conda activate vrenv

# 重要：不要手动设置 CUDA_VISIBLE_DEVICES！让 Slurm 自动设置
export TOKENIZERS_PARALLELISM=false

# 使用 accelerate 启动多进程训练（进程数由 config_zero2.yaml 决定）
accelerate launch --config_file config_zero2.yaml --main_process_port 29488 ../main.py \
    --env-name gym_cards/NumberLine-v0 \
    --init-lr 1e-5 \
    --end-lr 1e-9 \
    --lr_max_steps 25 \
    --eval-num-per-episode 200 \
    --num-env-steps 15000 \
    --num-steps 512 \
    --grad-accum-steps 128 \
    --max-new-tokens 256 \
    --thought-prob-coef 0.5 \
    --use-gae \
    --seed 1 \
    --temperature 0.2 \
    --ppo-epoch 4 \
    --mini-batch-size 1 \
    --model-path /your_sft_checkpoint_for_numberline \
    --use-lora \
    --train-vision all
    # --use-wandb \
    # --wandb-project you_wandb_proj \
    # --wandb-run you_wandb_run