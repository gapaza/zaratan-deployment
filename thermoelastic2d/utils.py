"""Utility functions for the thermoelastic2d problem."""

# ruff: noqa: PLR0913, PLR0915
from __future__ import annotations

from matplotlib import colors
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


def get_res_bounds(x_res: npt.NDArray, y_res: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    """Generates the indices corresponding to the left, top, right, and bottom elements in the domain.

    Args:
        x_res: The number of elements in the x-direction
        y_res: The number of elements in the y-direction

    Returns:
        The indices corresponding to the left, top, right, and bottom elements in the domain

    """
    row_elements = x_res
    col_elements = y_res

    bottom_row_indices = np.arange(col_elements - 1, row_elements * col_elements, col_elements)
    right_col_indices = np.arange((row_elements - 1) * col_elements, row_elements * col_elements)
    top_row_indices = np.arange(0, row_elements * col_elements, col_elements)
    left_col_indices = np.arange(0, col_elements, 1)

    return left_col_indices, top_row_indices, right_col_indices, bottom_row_indices




def plot_multi_physics(
    design: npt.NDArray,
    strain_energy_field: npt.NDArray,
    von_mises_stress_field: npt.NDArray,
    temperature_field: npt.NDArray,
    conditions: dict[str, npt.NDArray],
    open_plot: bool = False,
) -> Figure:
    """Visualizes a single iteration of the topology optimization solver plotting: the current design variables, the strain energy field, the von mises stress field, the temperature field.

        Args:
            design (npt.NDArray): The current design variables.
            strain_energy_field (npt.NDArray): The strain energy field.
            von_mises_stress_field (npt.NDArray): The von mises stress field.
            temperature_field (npt.NDArray): The temperature field.
            conditions (Dict[str, Any]): Dictionary specifying boundary conditions. Expected keys include:
                - "heatsink_elements": Indices for fixed thermal degrees of freedom.
                - "fixed_elements": Indices for fixed mechanical degrees of freedom.
                - "force_elements_x": Indices for x-direction force elements.
                - "force_elements_y": Indices for y-direction force elements.

        Returns:
            fig (Figure): The figure generated.
    """

    fig, ax = plt.subplots(2, 4, figsize=(8, 5))

    # First row is for the boundary conditions
    nelx = conditions["nelx"]
    nely = conditions["nely"]
    heatsink_indices = np.array(conditions["heatsink_elements"])
    fixed_indices = np.array(conditions["fixed_elements"])
    force_indices_x = np.array(conditions["force_elements_x"])
    force_indices_y = np.array(conditions["force_elements_y"])

    heatsink_elements = np.zeros(((nelx+1) * (nely+1),))
    heatsink_elements[heatsink_indices] = 1

    fixed_elements = np.zeros(((nelx+1) * (nely+1),))
    fixed_elements[fixed_indices] = 1

    force_elements_x = np.zeros(((nelx+1) * (nely+1),))
    if len(force_indices_x) > 0:
        force_elements_x[force_indices_x] = 1

    force_elements_y = np.zeros(((nelx+1) * (nely+1),))
    if len(force_indices_y) > 0:
        force_elements_y[force_indices_y] = 1

    im1 = ax[0][0].imshow(fixed_elements.reshape((nelx+1, nely+1)).T, cmap='gray', interpolation='none')
    ax[0][0].axis("off")
    ax[0][0].set_title("Fixed")
    # fig.colorbar(im1, ax=ax[0][0])

    im2 = ax[0][1].imshow(force_elements_x.reshape((nelx+1, nely+1)).T, cmap='gray', interpolation='none')
    ax[0][1].axis("off")
    ax[0][1].set_title("Force X")
    # fig.colorbar(im2, ax=ax[0][1])

    im3 = ax[0][2].imshow(force_elements_y.reshape((nelx+1, nely+1)).T, cmap='gray', interpolation='none')
    ax[0][2].axis("off")
    ax[0][2].set_title("Force Y")
    # fig.colorbar(im3, ax=ax[0][2])

    im4 = ax[0][3].imshow(heatsink_elements.reshape((nelx+1, nely+1)).T, cmap='gray', interpolation='none')
    ax[0][3].axis("off")
    ax[0][3].set_title("Heatsink")
    # fig.colorbar(im4, ax=ax[0][3])

    # Second row is for the fields
    im5 = ax[1][0].imshow(design, cmap='gray', interpolation='none', norm=colors.Normalize(vmin=0, vmax=1))
    ax[1][0].axis("off")
    ax[1][0].set_title("Design")
    fig.colorbar(im5, ax=ax[1][0])

    im6 = ax[1][1].imshow(strain_energy_field.T, cmap='viridis', interpolation='none')
    ax[1][1].axis("off")
    ax[1][1].set_title("Strain Energy")
    # fig.colorbar(im6, ax=ax[1][1])

    im7 = ax[1][2].imshow(von_mises_stress_field.T, cmap='viridis', interpolation='none')
    ax[1][2].axis("off")
    ax[1][2].set_title("Von Mises Stress")
    # fig.colorbar(im7, ax=ax[1][2])

    im8 = ax[1][3].imshow(temperature_field.T, cmap='inferno', interpolation='none')
    ax[1][3].axis("off")
    ax[1][3].set_title("Temperature")
    # fig.colorbar(im8, ax=ax[1][3])

    plt.tight_layout()
    if open_plot is True:
        plt.show()
    return fig


