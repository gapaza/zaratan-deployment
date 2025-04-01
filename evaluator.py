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


if __name__ == '__main__':
    f_path = '/Users/gapaza/repos/datasets/thermoelastic2d/abdpparobnfs.pkl'
    data = pickle.load(open(f_path, 'rb'))

    th_data = data['elastic']
    print('Keys:', th_data.keys())

    # a uniform density design
    print('\n-------------- Uniform density')
    design = np.ones((th_data['nelx'], th_data['nely'])) * 0.5
    results = evaluate(th_data, design)
    for key, value in results.items():
        print(key, value)

    # optimal design
    print('\n-------------- Optimal design')
    design = np.array(th_data['optimization']['design'])

    print('Opt keys:', th_data['optimization'].keys())
    design_steps = th_data['optimization']['design_steps']
    last_design_step = np.array(design_steps[-1])


    # Plot last design step and design
    import matplotlib.pyplot as plt
    from matplotlib import colors
    from matplotlib.figure import Figure
    fig, ax = plt.subplots(1, 2, figsize=(14, 7))
    ax[0].imshow(-design, cmap="gray", interpolation="none", norm=colors.Normalize(vmin=-1, vmax=0))
    ax[0].axis("off")
    ax[0].set_title('Optimal design')
    ax[1].imshow(-last_design_step, cmap="gray", interpolation="none", norm=colors.Normalize(vmin=-1, vmax=0))
    ax[1].axis("off")
    ax[1].set_title('Last design step')
    plt.tight_layout()
    plt.show()

    results = evaluate(th_data, last_design_step)
    for key, value in results.items():
        print(key, value, '---', th_data['optimization'][key])





