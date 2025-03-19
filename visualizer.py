from matplotlib import colors
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import pickle
import os

DP_PATH = '/Users/gapaza/repos/datasets/thermoelastic2d/atbsuohzgyqt.pkl'


def plot_conditions(
        heatsink_elements,
        fixed_elements,
        force_elements_x,
        force_elements_y,
        design,
        w,
        vms,
        sup_title='Design'
):
    # Plotting
    fig, ax = plt.subplots(2, 4, figsize=(8, 4))

    im1 = ax[0][0].imshow(fixed_elements, cmap='gray', interpolation='none')
    ax[0][0].axis("off")
    ax[0][0].set_title("Fixed")

    im2 = ax[0][1].imshow(force_elements_x, cmap='gray', interpolation='none')
    ax[0][1].axis("off")
    ax[0][1].set_title("Force X")

    im3 = ax[0][2].imshow(force_elements_y, cmap='gray', interpolation='none')
    ax[0][2].axis("off")
    ax[0][2].set_title("Force Y")

    im4 = ax[0][3].imshow(heatsink_elements, cmap='gray', interpolation='none')
    ax[0][3].axis("off")
    ax[0][3].set_title("Heatsink")

    im5 = ax[1][0].imshow(design, cmap='gray', interpolation='none', norm=colors.Normalize(vmin=0, vmax=1))
    ax[1][0].axis("off")
    ax[1][0].set_title("Design")

    im6 = ax[1][1].imshow(w, cmap='viridis', interpolation='none')
    ax[1][1].axis("off")
    ax[1][1].set_title("Strain Energy")
    fig.colorbar(im6, ax=ax[1][1])

    im7 = ax[1][2].imshow(vms, cmap='viridis', interpolation='none')
    ax[1][2].axis("off")
    ax[1][2].set_title("Von Mises Stress")
    fig.colorbar(im7, ax=ax[1][2])

    fig.suptitle(sup_title)

    plt.tight_layout()
    plt.show()




def plot_animation(conditions, title='design_history'):
    opt_results = conditions['optimization']
    design_steps = opt_results['design_steps']
    # print(len(design_steps))
    images = design_steps

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    image_display = ax.imshow(images[0] * -1, cmap='gray', interpolation='none', norm=colors.Normalize(vmin=-1, vmax=0))
    ax.set_title('Design History')

    def update(frame):
        image_display.set_array(images[frame] * -1)
        return image_display,

    ani = FuncAnimation(fig, update, frames=len(images), interval=100, blit=True)
    ani.save(os.path.join(f'{title}.gif'), writer='imagemagick', fps=5)




def get_boundary_tensors(conditions):
    nelx = conditions["nelx"]
    nely = conditions["nely"]
    volfrac = conditions["volfrac"]

    nel_xnodes = nelx + 1
    nel_ynodes = nely + 1

    heatsink_indices = np.array(conditions["heatsink_elements"])
    heatsink_elements = np.zeros(((nel_xnodes) * nel_ynodes,))
    heatsink_elements[heatsink_indices] = 1
    heatsink_elements = heatsink_elements.reshape((nel_xnodes, nel_ynodes)).T

    fixed_indices = np.array(conditions["fixed_elements"])
    fixed_elements = np.zeros((nel_xnodes * nel_ynodes,))
    fixed_elements[fixed_indices] = 1
    fixed_elements = fixed_elements.reshape((nel_xnodes, nel_ynodes)).T

    force_indices_x = np.array(conditions["force_elements_x"])
    force_elements_x = np.zeros((nel_xnodes * nel_ynodes,))
    if len(force_indices_x) > 0:
        force_elements_x[force_indices_x] = 1
    force_elements_x = force_elements_x.reshape((nel_xnodes, nel_ynodes)).T

    force_indices_y = np.array(conditions["force_elements_y"])
    force_elements_y = np.zeros((nel_xnodes * nel_ynodes,))
    if len(force_indices_y) > 0:
        force_elements_y[force_indices_y] = 1
    force_elements_y = force_elements_y.reshape((nel_xnodes, nel_ynodes)).T

    volfrac_tensor = np.ones((nel_xnodes, nel_ynodes)) * volfrac

    return heatsink_elements, fixed_elements, force_elements_x, force_elements_y, volfrac_tensor




def parse_th(conditions):
    heatsink_elements, fixed_elements, force_elements_x, force_elements_y, volfrac_tensor = get_boundary_tensors(conditions)
    force_elements_x = np.zeros_like(force_elements_x)
    force_elements_y = np.zeros_like(force_elements_y)
    fixed_elements = np.zeros_like(fixed_elements)

    opt_results = conditions['optimization']
    design = opt_results['design']
    vms = np.array(opt_results['von_mises_stress_field']).T
    w = np.array(opt_results['strain_energy_field']).T

    plot_conditions(
        heatsink_elements,
        fixed_elements,
        force_elements_x,
        force_elements_y,
        design,
        w,
        vms
    )


def parse_el(conditions):
    heatsink_elements, fixed_elements, force_elements_x, force_elements_y, volfrac_tensor = get_boundary_tensors(conditions)
    heatsink_elements = np.zeros_like(heatsink_elements)

    opt_results = conditions['optimization']
    design = opt_results['design']
    vms = np.array(opt_results['von_mises_stress_field']).T
    w = np.array(opt_results['strain_energy_field']).T

    plot_conditions(
        heatsink_elements,
        fixed_elements,
        force_elements_x,
        force_elements_y,
        design,
        w,
        vms
    )


def parse_mf(conditions):
    heatsink_elements, fixed_elements, force_elements_x, force_elements_y, volfrac_tensor = get_boundary_tensors(conditions)

    opt_results = conditions['optimization']
    design = opt_results['design']
    vms = np.array(opt_results['von_mises_stress_field']).T
    w = np.array(opt_results['strain_energy_field']).T

    plot_conditions(
        heatsink_elements,
        fixed_elements,
        force_elements_x,
        force_elements_y,
        design,
        w,
        vms
    )







# Load the datapoint
with open(DP_PATH, 'rb') as f:
    datapoint = pickle.load(f)

# Print keys of datapoint
print(datapoint.keys())


# --- ELASTIC ---
elastic_data = datapoint['elastic']
parse_el(elastic_data)
plot_animation(elastic_data, title='design_history_elastic')

# --- THERMAL ---
thermal_data = datapoint['thermal']
parse_th(thermal_data)
plot_animation(thermal_data, title='design_history_thermal')

# --- THERMOELASTIC ---
thermoelastic_data = datapoint['thermoelastic']
parse_mf(thermoelastic_data)
plot_animation(thermoelastic_data, title='design_history_thermoelastic')












































