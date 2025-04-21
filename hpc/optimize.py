import string
import os
import numpy as np
import pickle
from copy import deepcopy
import random
import argparse

# 5 designs - 86.73 sec previously
# 9 designs - 121.0 seconds
# 12 designs - 133.9 seconds

from thermoelastic2d.v0 import base_conditions
from thermoelastic2d.model.fea_model import FeaModel

ZERO_START = False
ONE_START = False
RAND_START = True
NOISY_START = False



def optimize(config_path):

    # Load the config file
    with open(config_path, 'rb') as f:
        config = pickle.load(f)

    datapoint = {}
    conditions = base_conditions
    boundary_dict = dict(conditions)
    for key, value in config.items():
        if key in boundary_dict:
            boundary_dict[key] = value

    # Pure structural
    me_conditions = deepcopy(boundary_dict)
    me_conditions['weight'] = 1.0
    starting_point = get_seed_design(boundary_dict)
    me_results = FeaModel(plot=False, eval_only=False).run(me_conditions, x_init=starting_point)
    me_conditions['optimization'] = me_results
    datapoint['elastic'] = me_conditions

    # Pure thermal
    th_conditions = deepcopy(boundary_dict)
    th_conditions['weight'] = 0.0
    starting_point = get_seed_design(boundary_dict)
    th_results = FeaModel(plot=False, eval_only=False).run(th_conditions, x_init=starting_point)
    th_conditions['optimization'] = th_results
    datapoint['thermal'] = th_conditions

    # Multi-physics
    mp_conditions = deepcopy(boundary_dict)
    mp_conditions['weight'] = 0.5
    starting_point = get_seed_design(boundary_dict)
    mp_results = FeaModel(plot=False, eval_only=False).run(mp_conditions, x_init=starting_point)
    mp_conditions['optimization'] = mp_results
    datapoint['thermoelastic'] = mp_conditions

    save_path = config['save_path']
    with open(save_path, 'wb') as f:
        pickle.dump(datapoint, f)




def get_seed_design(config):
    starting_point = np.ones((64, 64))
    if ZERO_START is True:
        starting_point = starting_point * 0.05
    elif ONE_START is True:
        starting_point = starting_point * 0.95
    elif RAND_START is True:
        points = np.linspace(0.05, 1.0, num=10)
        random_point = random.choice(points)
        starting_point = starting_point * random_point
    elif NOISY_START is True:
        starting_point = starting_point * 0.5
        starting_point += np.random.normal(0, 0.1, starting_point.shape)
    else:
        starting_point = starting_point * config['volfrac']
    starting_point = np.clip(starting_point, 0.0, 1.0)
    return starting_point



if __name__ == "__main__":

    # Parse the first argument as the config path
    parser = argparse.ArgumentParser(description="Optimize the design.")
    parser.add_argument('config_path', type=str, help='Path to the configuration file')
    args = parser.parse_args()

    # Call the optimize function with the parsed config path
    optimize(args.config_path)
