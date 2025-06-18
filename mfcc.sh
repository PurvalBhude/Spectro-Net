#!/bin/bash
#SBATCH --partition=Brain3080
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=168:00:00
#SBATCH --output=log/%x/%j/logs.out
#SBATCH --error=log/%x/%j/errors.err

export WANDB_DIR=/Brain/private/p25bhude/wandb/
export WANDB_MODE=online
export WANDB_API_KEY=45b695422df2b3e1e26f4d5d462efc9b7b8b2fb8

export MPLCONFIGDIR=$PWD/matplotlib_cache
mkdir -p $MPLCONFIGDIR

source venv/bin/activate
srun python3 mfcc.py