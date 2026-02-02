import config
import os
import argparse
from generator import Generator
import time

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
        default='/home/gapaza/scratch/datasets/thermoelastic2dv000',
        help='Directory to save the dataset (default: /home/gapaza/scratch/datasets/thermoelastic2dv000)'
    )

    parser.add_argument(
        '--el-dataset',
        type=str,
        default='training',
        help='Which elastic dataset to use (default: training)'
    )
    parser.add_argument(
        '--th-dataset',
        type=str,
        default='training',
        help='Which thermal dataset to use (default: training)'
    )

    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)


    # used_procs = config.num_cpus
    # if used_procs is None:
    #     used_procs = args.num_procs

    start_time = time.time()
    gen = Generator(NELX, NELY, args.save_dir)
    gen.run_mp(
        me_dataset=args.el_dataset,
        th_dataset=args.th_dataset,
        sample_size=args.samples,
        num_processes=args.num_procs
    )
    end_time = time.time()
    print(f'Elapsed time: {end_time - start_time} seconds')