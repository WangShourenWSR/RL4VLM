#!/bin/bash
#SBATCH --job-name=rl4vlm
#SBATCH -o sfai_sp_spmv_reward_dqn.o
#SBATCH -A xxh584_csds600
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p markov_gpu
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4gb
#SBATCH --time=16:00:00    # Days-HH:MM:SS

source ~/miniconda3/etc/profile.d/conda.sh
# source home/sxw992/miniconda3/bin/activate

conda activate sfai
cd /home/sxw992/Github/SFAI
python src/train_self_play_dqn.py --task_name sp_spmv_reward_dqn --multi_input_policy False --sticky_action_mode False --stickiness 0.0 --default_a2c False --log2file False --network_graph_tensorboard False