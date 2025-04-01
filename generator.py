import string
import os
import numpy as np
import pickle
from copy import deepcopy
import multiprocessing

from thermoelastic2d.conditions.thermoelastic_enum import ThermoelasticEnumeration
from thermoelastic2d.v0 import base_conditions
from thermoelastic2d.model.fea_model import FeaModel



def evaluate(config, design):
    conditions = base_conditions
    boundary_dict = dict(conditions)
    for key, value in config.items():
        if key in boundary_dict:
            boundary_dict[key] = value

    results = FeaModel(plot=False, eval_only=True).run(boundary_dict, x_init=design)
    return results



def optimize(config):
    datapoint = {}

    conditions = base_conditions
    boundary_dict = dict(conditions)
    for key, value in config.items():
        if key in boundary_dict:
            boundary_dict[key] = value

    # Pure structural
    me_conditions = deepcopy(boundary_dict)
    me_conditions['weight'] = 1.0
    starting_point = np.ones((64, 64)) * boundary_dict['volfrac']
    me_results = FeaModel(plot=False, eval_only=False).run(me_conditions, x_init=starting_point)
    me_conditions['optimization'] = me_results
    datapoint['elastic'] = me_conditions

    # Pure thermal
    th_conditions = deepcopy(boundary_dict)
    th_conditions['weight'] = 0.0
    starting_point = np.ones((64, 64)) * boundary_dict['volfrac']
    th_results = FeaModel(plot=False, eval_only=False).run(th_conditions, x_init=starting_point)
    th_conditions['optimization'] = th_results
    datapoint['thermal'] = th_conditions

    # Multi-physics
    mp_conditions = deepcopy(boundary_dict)
    mp_conditions['weight'] = 0.5
    starting_point = np.ones((64, 64)) * boundary_dict['volfrac']
    mp_results = FeaModel(plot=False, eval_only=False).run(mp_conditions, x_init=starting_point)
    mp_conditions['optimization'] = mp_results
    datapoint['thermoelastic'] = mp_conditions

    save_path = config['save_path']
    with open(save_path, 'wb') as f:
        pickle.dump(datapoint, f)


class Generator:


    def __init__(self, nelx, nely, save_dir):
        self.nelx = nelx
        self.nely = nely
        self.enumerator = ThermoelasticEnumeration(nelx, nely)
        self.save_dir = save_dir

    def get_initial_design(self, condition):
        return condition['volfrac'] * np.ones((self.nelx, self.nely))

    def salt_string(self):
        return ''.join(np.random.choice(list(string.ascii_lowercase), 12))



    def run(self, me_dataset='training', th_dataset='training', sample_size=1000):
        conditions = self.enumerator.sample_conditions(me_dataset, th_dataset, sample_size=sample_size)

        for condition in conditions:
            file_name = self.salt_string() + '.pkl'
            file_path = os.path.join(self.save_dir, file_name)
            condition['save_path'] = file_path
            optimize(condition)

    def run_mp(self, me_dataset='training', th_dataset='training', sample_size=1000, num_processes=4):
        conditions = self.enumerator.sample_conditions(me_dataset, th_dataset, sample_size=sample_size)

        # Prepare conditions with save_path for each case
        for condition in conditions:
            file_name = self.salt_string() + '.pkl'
            file_path = os.path.join(self.save_dir, file_name)
            condition['save_path'] = file_path

        # Use a multiprocessing pool to limit the number of concurrent processes
        with multiprocessing.Pool(processes=num_processes) as pool:
            pool.map(optimize, conditions)















if __name__ == '__main__':
    save_dir = '/Users/gapaza/repos/datasets/thermoelastic2dv1'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    gen = Generator(64, 64, save_dir)
    # gen.run()
    gen.run_mp(
        me_dataset='training',
        th_dataset='training',
        sample_size=5,
        num_processes=5
    )

