import os
import argparse
from generator import Generator

NELX = 64
NELY = 64

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate a dataset using multiple processes with configurable parameters.'
    )
    parser.add_argument(
        '--num-procs',
        type=int,
        default=5,
        help='Number of processes to use (default: 5)'
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=5,
        help='Number of samples to generate (default: 5)'
    )
    parser.add_argument(
        '--save-dir',
        type=str,
        default='/Users/gapaza/repos/datasets/thermoelastic2dv1',
        help='Directory to save the dataset (default: /Users/gapaza/repos/datasets/thermoelastic2dv1)'
    )
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    gen = Generator(NELX, NELY, args.save_dir)
    gen.run_mp(
        me_dataset='training',
        th_dataset='training',
        sample_size=args.samples,
        num_processes=args.num_procs
    )