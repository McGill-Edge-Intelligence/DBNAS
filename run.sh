#!/bin/bash
#SBATCH --nodes 3                 # Request 3 nodes 
#SBATCH --gres=gpu:3              # Number of GPU(s) per node
#SBATCH --cpus-per-task=2         # CPU cores/threads
#SBATCH --mem=10G                 # memory
#SBATCH --time=3:00:00            # A time limit of zero requests that no time limit be imposed. Acceptable time formats include "minutes", "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds".
#SBATCH --job-name=nas_block_1_cell_1         
#SBATCH --output=logs/%N-%j.out
#SBATCH --mail-user=hang.zhang3@mail.mcgill.ca
#SBATCH --mail-type=ALL
#SBATCH --account=rrg-wjgross

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# confirm gpu available
nvidia-smi
wandb login 38955c7425a0a87291433d688a4d7ff14bde509a 

if [ $# -ne 1 ] 
then 
    echo "Usage $0 <filename>"
else
    # run the command
    source ../py377/bin/activate
    python $1
fi