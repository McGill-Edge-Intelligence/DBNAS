#!/bin/bash
#SBATCH --gres=gpu:v100l:1
#SBATCH --cpus-per-task=2
#SBATCH --time=4:00:00
#SBATCH --account=xxxxx
#SBATCH --job-name=block1_cell2_sequential
#SBATCH --output=logs/%x-%j.out
#SBATCH --mem=32000M

# Start training
wandb login xxxxxxx
python searching/DNA_studentbertsequential.py
echo 'Block1_Cell2_sequential finished!'
