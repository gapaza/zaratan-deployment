import os
import argparse
from hpc.optimize import optimize




if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate a dataset using slurm.'
    )
    parser.add_argument(
        '--stage-dir',
        type=str,
        default='/home/gapaza/scratch/datasets/thermoelastic2dv1',
        help='Directory to save the dataset (default: /Users/gapaza/repos/datasets/thermoelastic2dv1)'
    )
    args = parser.parse_args()

    # All files in the staging directory. Each will be a unique slurm job.
    staging_dir = args.stage_dir
    staging_files = [f for f in os.listdir(staging_dir) if f.endswith('.pkl')]

    # The optimize
























