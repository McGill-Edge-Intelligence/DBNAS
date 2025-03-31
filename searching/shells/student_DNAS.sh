#!/bin/bash
#SBATCH --gres=gpu:v100l:1
#SBATCH --cpus-per-task=2
#SBATCH --time=12:00:00
#SBATCH --account=xxxxx
#SBATCH --job-name=BERT_DNAS
#SBATCH --output=logs/%x-%j.out
#SBATCH --mem=32000M

# Start training
wandb login xxxxxxx
python searching/DNA_studentbertdiff.py
echo 'DNAS cell search completed!'
