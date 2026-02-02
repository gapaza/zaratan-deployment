#!/bin/bash
#SBATCH --job-name=gen_dataset
#SBATCH --output=logs/gen_dataset_%j.out
#SBATCH --error=logs/gen_dataset_%j.err
#SBATCH --time=10:00:00
#SBATCH --account=fuge-prj-jrl
#SBATCH --cpus-per-task=128
#SBATCH --mem=256G

# Load modules
module load openblas/0.3.23/gcc/11.3.0/x86_64
module load python/3.10.10/gcc/11.3.0/cuda/12.3.0/linux-rhel8-x86_64

# Change directory
cd /home/gapaza/scratch/repos/zaratan-deployment

# Run the training script
python3 -m main --num-procs 120 --samples 1000 --save-dir /home/gapaza/scratch/datasets/thermoelastic2dv101