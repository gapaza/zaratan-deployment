import string
import os
import numpy as np
import pickle
from copy import deepcopy
from tqdm import tqdm
import multiprocessing
import random
import config as cfg
# multiprocessing.set_start_method('fork', force=True)
# 5 designs - 86.73 sec previously
# 9 designs - 121.0 seconds
# 12 designs - 133.9 seconds

from thermoelastic2d.conditions.thermoelastic_enum import ThermoelasticEnumeration
from thermoelastic2d.v0 import base_conditions
from thermoelastic2d.model.fea_model import FeaModel


SEED_DESIGN_TYPES = [
    # 'ZERO_START',
    # 'ONE_START',
    'RAND_START',
    'NOISY_START',
    'DEFAULT'
]

# Manual Control
ZERO_START = False
ONE_START = False
RAND_START = False
NOISY_START = False


CONST_SEED_VF = 0.5
def get_seed_design_constant(config):
    starting_point = np.ones((64, 64))
    starting_point = starting_point * CONST_SEED_VF
    return starting_point


def get_seed_design_nominal(config):
    starting_point = np.ones((64, 64))
    starting_point = starting_point * config['volfrac']
    return starting_point


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

def get_seed_design_random(config):
    starting_point = np.ones((64, 64))
    s_type = random.choice([
        'DEFAULT',
        'NOISY_START',
    ])
    points = np.linspace(0.05, 1.0, num=10)
    random_point = random.choice(points)
    starting_point = starting_point * random_point
    if s_type == 'NOISY_START':
        starting_point += np.random.normal(0, 0.1, starting_point.shape)
    starting_point = np.clip(starting_point, 0.0, 1.0)
    return starting_point



def evaluate(config, design):
    conditions = base_conditions
    boundary_dict = dict(conditions)
    for key, value in config.items():
        if key in boundary_dict:
            boundary_dict[key] = value

    results = FeaModel(plot=False, eval_only=True).run(boundary_dict, x_init=design)
    return results


def reoptimize(inputs):
    config_path, save_path = inputs

    # Here, config is the dictionary created in the optimize function
    # We need to re-optimize the elastic, thermal, and thermoelastic cases
    datapoint = {}

    # load config with pickle
    with open(config_path, 'rb') as f:
        config = pickle.load(f)
    
    me_conditions = config['elastic']
    th_conditions = config['thermal']
    mp_conditions = config['thermoelastic']

    # --- Nominal Updates ---
    nom_updates = 350

    # Re-optimize elastic
    me_conditions['max_iterations'] = nom_updates
    starting_point = get_seed_design_constant(me_conditions)
    me_results = FeaModel(plot=False, eval_only=False).run(me_conditions, x_init=starting_point)
    if me_results['iterations'] < 20:
        print('EL Iterations less than 20, exiting process')
        return
    me_conditions['optimization'] = me_results
    datapoint['elastic'] = me_conditions

    # Re-optimize thermal
    th_conditions['max_iterations'] = nom_updates
    starting_point = get_seed_design_constant(th_conditions)
    th_results = FeaModel(plot=False, eval_only=False).run(th_conditions, x_init=starting_point)
    if th_results['iterations'] < 20:
        print('TH Iterations less than 20, exiting process')
        return
    th_conditions['optimization'] = th_results
    datapoint['thermal'] = th_conditions

    # Re-optimize thermoelastic
    mp_conditions['max_iterations'] = nom_updates
    starting_point = get_seed_design_constant(mp_conditions)
    mp_results = FeaModel(plot=False, eval_only=False).run(mp_conditions, x_init=starting_point)
    if mp_results['iterations'] < 20:
        print('MP Iterations less than 20, exiting process')
        return
    mp_conditions['optimization'] = mp_results
    datapoint['thermoelastic'] = mp_conditions

    with open(save_path, 'wb') as f:
        pickle.dump(datapoint, f)

    return



def optimize(config):
    datapoint = {}

    conditions = base_conditions
    boundary_dict = dict(conditions)
    for key, value in config.items():
        if key in boundary_dict:
            boundary_dict[key] = value

    # --- Nominal Updates ---
    nom_updates = 350

    # Pure structural
    me_conditions = deepcopy(boundary_dict)
    me_conditions['max_iterations'] = nom_updates
    me_conditions['weight'] = 1.0
    # starting_point = get_seed_design(boundary_dict)
    # starting_point = get_seed_design_random(boundary_dict)
    starting_point = get_seed_design_constant(boundary_dict)
    me_results = FeaModel(plot=False, eval_only=False).run(me_conditions, x_init=starting_point)
    if me_results['iterations'] < 20:
        print('EL Iterations less than 20, exiting process')
        return
    me_conditions['optimization'] = me_results
    datapoint['elastic'] = me_conditions

    # Pure thermal
    th_conditions = deepcopy(boundary_dict)
    th_conditions['max_iterations'] = nom_updates
    th_conditions['weight'] = 0.0
    # starting_point = get_seed_design(boundary_dict)
    # starting_point = get_seed_design_random(boundary_dict)
    starting_point = get_seed_design_constant(boundary_dict)
    th_results = FeaModel(plot=False, eval_only=False).run(th_conditions, x_init=starting_point)
    if th_results['iterations'] < 20:
        print('TH Iterations less than 20, exiting process')
        return
    th_conditions['optimization'] = th_results
    datapoint['thermal'] = th_conditions

    # Multi-physics
    mp_conditions = deepcopy(boundary_dict)
    mp_conditions['max_iterations'] = nom_updates
    mp_conditions['weight'] = 0.5
    # starting_point = get_seed_design(boundary_dict)
    # starting_point = get_seed_design_random(boundary_dict)
    starting_point = get_seed_design_constant(boundary_dict)
    mp_results = FeaModel(plot=False, eval_only=False).run(mp_conditions, x_init=starting_point)
    if mp_results['iterations'] < 20:
        print('MP Iterations less than 20, exiting process')
        return
    mp_conditions['optimization'] = mp_results
    datapoint['thermoelastic'] = mp_conditions


    # # --- Mix-and-Match ---
    # mnm_updates = 80
    # mnm_samples = 1

    # # -------------- Optimized TH --> EL ----------------
    # # Gradient updates to better learn EL updates for TH-like designs
    # th_design_steps = th_results['design_steps']
    # n_th_design_steps = len(th_design_steps)
    # th_samp_steps = np.linspace(0, n_th_design_steps - 1, num=mnm_samples+1).astype(int)
    # th_samp_steps = th_samp_steps[1:]
    # for idx in range(len(th_samp_steps)):
    #     th_samp_step = th_samp_steps[idx]  # -1
    #     mnm_starting_point = th_design_steps[th_samp_step]
    #     mnm_conditions_th_el = deepcopy(boundary_dict)
    #     mnm_conditions_th_el['weight'] = 1.0  # structural
    #     mnm_conditions_th_el['max_iterations'] = mnm_updates
    #     mnm_results_th_el = FeaModel(plot=False, eval_only=False).run(mnm_conditions_th_el, x_init=mnm_starting_point)
    #     mnm_results_th_el['strain_energy_field'] = me_results['strain_energy_field']
    #     mnm_results_th_el['von_mises_stress_field'] = me_results['von_mises_stress_field']
    #     mnm_results_th_el['temperature_field'] = me_results['temperature_field']
    #     mnm_conditions_th_el['optimization'] = mnm_results_th_el
    #     datapoint[f'mnm_th_to_el_{idx}'] = mnm_conditions_th_el


    # # ---------------- Optimized EL --> TH ----------------
    # # Gradient updates to better learn TH updates for EL-like designs
    # el_design_steps = me_results['design_steps']
    # n_el_design_steps = len(el_design_steps)
    # el_samp_steps = np.linspace(0, n_el_design_steps - 1, num=mnm_samples+1).astype(int)
    # el_samp_steps = el_samp_steps[1:]
    # for idx in range(len(el_samp_steps)):
    #     el_samp_step = el_samp_steps[idx]  # -1
    #     mnm_starting_point = el_design_steps[el_samp_step]
    #     mnm_conditions_el_th = deepcopy(boundary_dict)
    #     mnm_conditions_el_th['weight'] = 0.0  # thermal
    #     mnm_conditions_el_th['max_iterations'] = mnm_updates
    #     mnm_results_el_th = FeaModel(plot=False, eval_only=False).run(mnm_conditions_el_th, x_init=mnm_starting_point)
    #     mnm_results_el_th['strain_energy_field'] = th_results['strain_energy_field']
    #     mnm_results_el_th['von_mises_stress_field'] = th_results['von_mises_stress_field']
    #     mnm_results_el_th['temperature_field'] = th_results['temperature_field']
    #     mnm_conditions_el_th['optimization'] = mnm_results_el_th
    #     datapoint[f'mnm_el_to_th_{idx}'] = mnm_conditions_el_th


    # # --- Volume Fraction Move ---
    # vfm_updates = 20
    # vmf_low_coeff = 0.75
    # vmf_high_coeff = 1.25

    # # ---------------- Optimized EL --> EL Low VF ----------------
    # el_design_steps = me_results['design_steps']
    # vfm_starting_point = el_design_steps[-1]
    # vfm_conditions_el_el = deepcopy(boundary_dict)
    # vfm_conditions_el_el['weight'] = 1.0  # structural
    # vfm_conditions_el_el['max_iterations'] = vfm_updates
    # vfm_conditions_el_el['volfrac'] = vfm_conditions_el_el['volfrac'] * vmf_low_coeff
    # vfm_results_el_el = FeaModel(plot=False, eval_only=False).run(vfm_conditions_el_el, x_init=vfm_starting_point)
    # vfm_results_el_el['strain_energy_field'] = me_results['strain_energy_field']
    # vfm_results_el_el['von_mises_stress_field'] = me_results['von_mises_stress_field']
    # vfm_results_el_el['temperature_field'] = me_results['temperature_field']
    # vfm_conditions_el_el['optimization'] = vfm_results_el_el
    # datapoint['vfm_el_el_0'] = vfm_conditions_el_el

    # # ---------------- Optimized EL --> EL High VF ----------------
    # el_design_steps = me_results['design_steps']
    # vfm_starting_point = el_design_steps[-1]
    # vfm_conditions_el_el = deepcopy(boundary_dict)
    # vfm_conditions_el_el['weight'] = 1.0  # structural
    # vfm_conditions_el_el['max_iterations'] = vfm_updates
    # vfm_conditions_el_el['volfrac'] = vfm_conditions_el_el['volfrac'] * vmf_high_coeff
    # vfm_results_el_el = FeaModel(plot=False, eval_only=False).run(vfm_conditions_el_el, x_init=vfm_starting_point)
    # vfm_results_el_el['strain_energy_field'] = me_results['strain_energy_field']
    # vfm_results_el_el['von_mises_stress_field'] = me_results['von_mises_stress_field']
    # vfm_results_el_el['temperature_field'] = me_results['temperature_field']
    # vfm_conditions_el_el['optimization'] = vfm_results_el_el
    # datapoint['vfm_el_el_1'] = vfm_conditions_el_el

    # # ---------------- Optimized TH --> TH Low VF ----------------
    # th_design_steps = th_results['design_steps']
    # vfm_starting_point = th_design_steps[-1]
    # vfm_conditions_th_th = deepcopy(boundary_dict)
    # vfm_conditions_th_th['weight'] = 0.0  # thermal
    # vfm_conditions_th_th['max_iterations'] = vfm_updates
    # vfm_conditions_th_th['volfrac'] = vfm_conditions_th_th['volfrac'] * vmf_low_coeff
    # vfm_results_th_th = FeaModel(plot=False, eval_only=False).run(vfm_conditions_th_th, x_init=vfm_starting_point)
    # vfm_results_th_th['strain_energy_field'] = th_results['strain_energy_field']
    # vfm_results_th_th['von_mises_stress_field'] = th_results['von_mises_stress_field']
    # vfm_results_th_th['temperature_field'] = th_results['temperature_field']
    # vfm_conditions_th_th['optimization'] = vfm_results_th_th
    # datapoint['vfm_th_th_0'] = vfm_conditions_th_th

    # # ---------------- Optimized TH --> TH High VF ----------------
    # th_design_steps = th_results['design_steps']
    # vfm_starting_point = th_design_steps[-1]
    # vfm_conditions_th_th = deepcopy(boundary_dict)
    # vfm_conditions_th_th['weight'] = 0.0  # thermal
    # vfm_conditions_th_th['max_iterations'] = vfm_updates
    # vfm_conditions_th_th['volfrac'] = vfm_conditions_th_th['volfrac'] * vmf_high_coeff
    # vfm_results_th_th = FeaModel(plot=False, eval_only=False).run(vfm_conditions_th_th, x_init=vfm_starting_point)
    # vfm_results_th_th['strain_energy_field'] = th_results['strain_energy_field']
    # vfm_results_th_th['von_mises_stress_field'] = th_results['von_mises_stress_field']
    # vfm_results_th_th['temperature_field'] = th_results['temperature_field']
    # vfm_conditions_th_th['optimization'] = vfm_results_th_th
    # datapoint['vfm_th_th_1'] = vfm_conditions_th_th


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

        ss_scaled = 10000 
        conditions = self.enumerator.sample_conditions(me_dataset, th_dataset, sample_size=ss_scaled)
        conditions = random.sample(conditions, sample_size)

        # Prepare conditions with save_path for each case
        for condition in conditions:
            file_name = self.salt_string() + '.pkl'
            file_path = os.path.join(self.save_dir, file_name)
            condition['save_path'] = file_path

        # # Use a multiprocessing pool to limit the number of concurrent processes
        # with multiprocessing.Pool(processes=num_processes) as pool:
        #     pool.map(optimize, conditions)

        with multiprocessing.Pool(processes=num_processes) as pool:
            # imap_unordered yields results as soon as they're ready.
            results = list(tqdm(pool.imap_unordered(optimize, conditions), total=len(conditions)))




class Regenerator:

    def __init__(self, source_path, target_path):
        self.source_path = source_path
        self.target_path = target_path

        if not os.path.exists(self.target_path):
            os.makedirs(self.target_path)

        # Load all .pkl files from source_path
        self.files = [f for f in os.listdir(source_path) if f.endswith('.pkl')]

        self.parameters = []
        for file in self.files:
            source_file_path = os.path.join(self.source_path, file)
            target_file_path = os.path.join(self.target_path, file)
            self.parameters.append((source_file_path, target_file_path))

        

    def run_mp(self, num_processes=4):



        with multiprocessing.Pool(processes=num_processes) as pool:
            results = list(tqdm(pool.imap_unordered(reoptimize, self.parameters), total=len(self.parameters)))

            








if __name__ == '__main__':

    # ### Generator Testing
    # save_dir = '/Users/gapaza/repos/datasets/thermoelastic2dv1'
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)

    # gen = Generator(64, 64, save_dir)
    # # gen.run()
    # gen.run_mp(
    #     me_dataset='training',
    #     th_dataset='training',
    #     sample_size=5,
    #     num_processes=5
    # )


    ### Regenerator Testing
    source_path = '/home/gapaza/scratch/datasets/thermoelastic2dv009_test_V2'
    target_path = '/home/gapaza/scratch/datasets/thermoelastic2dv009_test_V3'
    reg = Regenerator(source_path, target_path)
    reg.run_mp(num_processes=50)

