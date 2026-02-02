from matplotlib import colors
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import pickle
import os
import cv2
from scipy import ndimage  # distance_transform_edt lives here


def plot_conditions(
        heatsink_elements,
        fixed_elements,
        force_elements_x,
        force_elements_y,
        design,
        w,
        vms,
        sup_title='Design',
        save_dir=None,
        temp=None
):
    # Plotting
    fig, ax = plt.subplots(2, 4, figsize=(8, 4))


    fixed_elements = signed_distance(fixed_elements)
    im1 = ax[0][0].imshow(fixed_elements, cmap='viridis', interpolation='none')
    ax[0][0].axis("off")
    ax[0][0].set_title("Fixed")

    force_elements_x = signed_distance(force_elements_x)
    im2 = ax[0][1].imshow(force_elements_x, cmap='viridis', interpolation='none')
    ax[0][1].axis("off")
    ax[0][1].set_title("Force X")

    force_elements_y = signed_distance(force_elements_y)
    im3 = ax[0][2].imshow(force_elements_y, cmap='viridis', interpolation='none')
    ax[0][2].axis("off")
    ax[0][2].set_title("Force Y")

    im4 = ax[0][3].imshow(heatsink_elements, cmap='gray', interpolation='none')
    ax[0][3].axis("off")
    ax[0][3].set_title("Heatsink")

    im5 = ax[1][0].imshow(-design, cmap='gray', interpolation='none', norm=colors.Normalize(vmin=-1, vmax=0))
    # im5 = ax[1][0].imshow(design, cmap='viridis', interpolation='none')
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

    if temp is not None:
        im8 = ax[1][3].imshow(temp, cmap='hot', interpolation='none')
        ax[1][3].axis("off")
        ax[1][3].set_title("Temperature")
        fig.colorbar(im8, ax=ax[1][3])

    fig.suptitle(sup_title)
    plt.tight_layout()

    if save_dir:
        plt.savefig(os.path.join(save_dir, f'{sup_title}.png'))
    else:
        plt.savefig(f'{sup_title}.png')




def plot_animation(conditions, title='design_history', save_dir=None):
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

    if save_dir is not None:
        ani.save(os.path.join(save_dir, f'{title}.gif'), writer='imagemagick', fps=5)
    else:
        ani.save(os.path.join(f'{title}.gif'), writer='imagemagick', fps=5)


    # PLOTTING ----------
    # plot every 10 design steps in a 2x5 grid
    num_plots = 10
    step_numbers = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    fig, ax = plt.subplots(2, 5, figsize=(15, 6))
    for i in range(num_plots):
        step = step_numbers[i]
        if step >= len(images):
            break
        row = i // 5
        col = i % 5
        ax[row, col].imshow(images[step] * -1, cmap='gray', interpolation='none', norm=colors.Normalize(vmin=-1, vmax=0))
        # ax[row, col].set_title(f'Step {step}')
        ax[row, col].axis('off')
    
    progression_title = title + '_progression'
    plt.suptitle('Design Progression')
    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, f'{progression_title}.png'))
    else:
        plt.savefig(f'{progression_title}.png')






    # PLOTTING ----------
    # 1. plot a before image before and update
    # 2. plot an after image after the update
    # 3. plot the update

    update_size = 5
    before_idx = 0

    before_image = images[before_idx]
    after_image = images[before_idx + update_size]
    update_image = after_image - before_image


    title_update = title + '_update'

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(before_image * -1, cmap='gray', interpolation='none', norm=colors.Normalize(vmin=-1, vmax=0))
    ax[0].set_title('Before')
    ax[0].axis('off')
    ax[1].imshow(after_image * -1, cmap='gray', interpolation='none', norm=colors.Normalize(vmin=-1, vmax=0))
    ax[1].set_title('After')
    ax[1].axis('off')
    ax_update = ax[2].imshow(update_image, cmap='RdBu', interpolation='none')
    ax[2].set_title('Update')
    ax[2].axis('off')
    fig.colorbar(ax_update, ax=ax[2])
    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, f'{title_update}.png'))
    else:
        plt.savefig(f'{title_update}.png')


    # PLOTTING ----------
    # now noise the update image to get a pure noise, noisy, and clean version
    noise = np.random.normal(0, 0.3, size=update_image.shape)
    noisy_image = 0.8 * update_image + 0.2 * noise
    clean_image = update_image

    title_denoise = title + '_denoise'

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    ax_0 = ax[0].imshow(noise, cmap='gray', interpolation='none')
    ax[0].set_title('Pure Noise')
    ax[0].axis('off')
    fig.colorbar(ax_0, ax=ax[0])

    ax_1 = ax[1].imshow(noisy_image, cmap='RdBu', interpolation='none')
    ax[1].set_title('Noisy Update')
    ax[1].axis('off')
    fig.colorbar(ax_1, ax=ax[1])

    ax_2 = ax[2].imshow(clean_image, cmap='RdBu', interpolation='none')
    ax[2].set_title('Clean Update')
    ax[2].axis('off')
    fig.colorbar(ax_2, ax=ax[2])

    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, f'{title_denoise}.png'))
    else:
        plt.savefig(f'{title_denoise}.png')








    




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



def signed_distance(mask_01, norm=True) -> np.ndarray:
    """
    Compute a pixel-centred signed Euclidean distance field (SDF).

    Parameters
    ----------
    mask_01 : np.ndarray, bool or int
        2-D array where non-zeros mark the surface (shape (64, 64) here).
    norm : bool, optional

    Returns
    -------
    sdf : np.ndarray, float32
        Signed distance array, same shape as the input.
        Positive = outside, negative = inside, 0 = surface.
    """
    # Ensure a binary array of 0/1
    mask = (mask_01 != 0)

    # Outside → distance to *nearest* non-zero (surface) pixel
    dist_out = ndimage.distance_transform_edt(~mask)

    # Inside  → distance to *nearest* zero pixel (again the surface)
    dist_in  = ndimage.distance_transform_edt(mask)

    # Positive outside, negative inside
    sdf = dist_out - dist_in

    # Explicitly force the surface itself to 0
    sdf[mask] = 0.0

    sdf = sdf.astype(np.float32)
    if norm is True:
        sdf = sdf / np.abs(sdf).max()
        # sdf = sdf / 91.0
    return sdf



def parse_th(conditions, save_dir=None, sup_title='thermal'):
    heatsink_elements, fixed_elements, force_elements_x, force_elements_y, volfrac_tensor = get_boundary_tensors(conditions)
    force_elements_x = np.zeros_like(force_elements_x)
    force_elements_y = np.zeros_like(force_elements_y)
    fixed_elements = np.zeros_like(fixed_elements)

    opt_results = conditions['optimization']
    design = opt_results['design']
    vms = np.array(opt_results['von_mises_stress_field']).T
    w = np.array(opt_results['strain_energy_field']).T
    temp = np.array(opt_results['temperature_field']).T

    # print('Temp shape:', temp.shape)


    plot_conditions(
        heatsink_elements,
        fixed_elements,
        force_elements_x,
        force_elements_y,
        design,
        w,
        vms,
        sup_title=sup_title,
        save_dir=save_dir,
        temp=temp
    )


def parse_el(conditions, save_dir=None, sup_title='elastic'):
    heatsink_elements, fixed_elements, force_elements_x, force_elements_y, volfrac_tensor = get_boundary_tensors(conditions)
    heatsink_elements = np.zeros_like(heatsink_elements)

    opt_results = conditions['optimization']
    design = opt_results['design']
    vms = np.array(opt_results['von_mises_stress_field']).T
    w = np.array(opt_results['strain_energy_field']).T

    # resize vms to 65x65 with interpolation


    plot_conditions(
        heatsink_elements,
        fixed_elements,
        force_elements_x,
        force_elements_y,
        design,
        w,
        vms,
        sup_title=sup_title,
        save_dir=save_dir
    )


def parse_mf(conditions, save_dir=None, sup_title='thermoelastic'):
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
        vms,
        sup_title=sup_title,
        save_dir=save_dir
    )











































