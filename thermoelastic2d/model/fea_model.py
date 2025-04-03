"""This module contains the Python implementation of the thermoelastic 2D problem."""

from __future__ import annotations

from math import ceil
from math import hypot
import time

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from scipy.signal import convolve2d

from thermoelastic2d.model.fem_matrix_builder import fe_melthm
from thermoelastic2d.model.fem_setup import fe_mthm_bc
from thermoelastic2d.model.mma_subroutine import MMAInputs
from thermoelastic2d.model.mma_subroutine import mmasub
from thermoelastic2d.model.stress_strain_calcs import calc_stress_strain_me, calc_stress_strain_th
from thermoelastic2d.utils import get_res_bounds
from thermoelastic2d.utils import plot_multi_physics

SECOND_ITERATION_THRESHOLD = 2
FIRST_ITERATION_THRESHOLD = 1
MIN_ITERATIONS = 10
MAX_ITERATIONS = 100
UPDATE_THRESHOLD = 0.01
PLOTTING_FREQ = 10



import dataclasses
import numpy.typing as npt

@dataclasses.dataclass
class OptiStep:
    """Optimization step."""

    obj_values: npt.NDArray
    step: int


class FeaModel:
    """Finite Element Analysis (FEA) model for coupled 2D thermoelastic topology optimization."""

    def __init__(self, plot: bool = False, eval_only: bool | None = False) -> None:
        """Instantiates a new model for the thermoelastic 2D problem.

        Args:
            plot: (bool, optional): If True, the updated design will be plotted at each iteration.
            eval_only: (bool, optional): If True, the model will only evaluate the design and return the objective values.
        """
        self.plot = plot
        self.eval_only = eval_only

    def get_initial_design(self, volume_fraction: float, nelx: int, nely: int) -> np.ndarray:
        """Generates the initial design variable field for the optimization process.

        Args:
            volume_fraction (float): The initial volume fraction for the material distribution.
            nelx (int): Number of elements in the x-direction.
            nely (int): Number of elements in the y-direction.

        Returns:
            np.ndarray: A 2D NumPy array of shape (nely, nelx) initialized with the given volume fraction.
        """
        return volume_fraction * np.ones((nely, nelx))

    def get_matricies(self, nu: float, e: float, k: float, alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Computes and returns the element matrices required for the structural-thermal analysis.

        Args:
            nu (float): Poisson's ratio.
            e (float): Young's modulus (modulus of elasticity).
            k (float): Thermal conductivity.
            alpha (float): Coefficient of thermal expansion.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
                - The stiffness matrix for mechanical analysis.
                - The thermal stiffness matrix.
                - The coupling matrix for thermal expansion effects.
        """
        return fe_melthm([nu, e, k, alpha])

    def get_filter(self, nelx: int, nely: int, rmin: float) -> tuple[coo_matrix, np.ndarray]:
        """Constructs a sensitivity filtering matrix to smoothen the design variables.

        The filter helps mitigate checkerboarding issues in topology optimization by averaging
        sensitivities over neighboring elements.

        Args:
            nelx (int): Number of elements in the x-direction.
            nely (int): Number of elements in the y-direction.
            rmin (float): Minimum filter radius.

        Returns:
            Tuple[csr_matrix, np.ndarray]: A tuple containing:
                - `h` (csr_matrix): A sparse matrix that represents the filtering operation.
                - `hs` (np.ndarray): A normalization factor for the filtering.
        """
        i_h = []
        j_h = []
        s_h = []

        for i1 in range(nelx):
            i2_min = max(i1 - (ceil(rmin) - 1), 0)
            for j1 in range(nely):
                e1 = i1 * nely + j1
                i2_max = min(i1 + (ceil(rmin) - 1), nelx - 1)
                for i2 in range(i2_min, i2_max + 1):
                    j2_min = max(j1 - (ceil(rmin) - 1), 0)
                    j2_max = min(j1 + (ceil(rmin) - 1), nely - 1)
                    for j2 in range(j2_min, j2_max + 1):
                        e2 = i2 * nely + j2
                        i_h.append(e1)
                        j_h.append(e2)
                        s_h.append(max(0, rmin - hypot(i1 - i2, j1 - j2)))

        h = coo_matrix((s_h, (i_h, j_h)), shape=(nelx * nely, nelx * nely)).tocsr()
        hs = np.array(h.sum(axis=1)).flatten()

        return h, hs

    def run(self, bcs: dict[str, any], x_init: np.ndarray | None = None) -> dict[str, any]:  # noqa: PLR0915
        """Run the optimization algorithm for the coupled structural-thermal problem.

        This method performs an iterative optimization procedure that adjusts the design
        variables for a coupled structural-thermal problem until convergence criteria are met.
        The algorithm utilizes finite element analysis (FEA), sensitivity analysis, and the Method
        of Moving Asymptotes (MMA) to update the design.

        Args:
            bcs (dict[str, any]): A dictionary containing boundary conditions and problem parameters.
                Expected keys include:
                    - 'nelx' (int): Number of elements along the x-direction.
                    - 'nely' (int): Number of elements along the y-direction.
                    - 'volfrac' (float): Target volume fraction.
                    - 'fixed_elements' (List[Any], optional): List of fixed elements.
                    - 'load_elements' (List[Any], optional): List of force elements.
                    - 'weight' (float, optional): Weighting factor between structural and thermal objectives.
            x_init (Optional[np.ndarray]): Initial design variable array. If None, the design is generated
                using the get_initial_design method.

        Returns:
            Dict[str, Any]: A dictionary containing the optimization results. The dictionary includes:
                - 'design' (np.ndarray): Final design layout.
                - 'bcs' (Dict[str, Any]): The input boundary conditions.
                - 'structural_compliance' (float): Structural cost component.
                - 'thermal_compliance' (float): Thermal cost component.
                - 'volume_fraction' (float): Volume fraction error.
                - 'opti_steps' (List[OptiStep]): List of optimization steps.
                - 'strain_energy_field' (np.ndarray): Strain energy density field of the initial design.
                - 'von_mises_stress_field' (np.ndarray): Von Mises stress field of the initial design.

            If self.eval_only is True, returns a dictionary with keys 'structural_compliance', 'thermal_compliance', and 'volume_fraction' only.
        """
        # WEIGHTING
        w1 = bcs.get("weight", 0.5)
        w2 = 1.0 - w1

        nelx = bcs["nelx"]
        nely = bcs["nely"]
        volfrac = bcs["volfrac"]
        n = nely * nelx  # Total number of elements

        # OptiSteps records
        opti_steps = []
        design_steps = []
        displacement_steps = []
        temp_steps = []
        vms_steps = []
        w_steps = []

        # 1. Initial Design
        x = self.get_initial_design(volfrac, nelx, nely) if x_init is None else x_init

        # 2. Parameters
        penal = 3  # Penalty term
        rmin = bcs["rmin"]  # Filter's radius
        e = 1.0  # Modulus of elasticity
        emin = 1e-9  # Minimum modulus of elasticity
        e0 = 1.0  # Initial modulus of elasticity
        nu = 0.3  # Poisson's ratio
        k = 1.0  # Conductivity
        alpha = 5e-4  # Coefficient of thermal expansion (CTE)
        tref = 9.267e-4  # Reference Temperature
        change = 1.0  # Density change criterion
        m = 1  # Number of constraints (volume constraints)
        iterr = 0  # Number of iterations
        xmin = 1e-3  # Densities' Lower bound
        xmax = 1.0  # Densities' Upper bound
        low = xmin
        upp = xmax
        xold1 = x.reshape(n, 1)
        xold2 = x.reshape(n, 1)
        a0 = 1
        a = np.zeros((m, 1))
        c = 10000 * np.ones((m, 1))
        d = np.zeros((m, 1))
        stress_init = np.zeros((nely, nelx))
        strain_init = np.zeros((nely, nelx))

        # 3. Matrices
        ke, k_eth, c_ethm = self.get_matricies(nu, e, k, alpha)

        # 4. Filter
        h, hs = self.get_filter(nelx, nely, rmin)

        # 5. Optimization Loop
        change_evol = []
        obj = []

        while change > UPDATE_THRESHOLD or iterr < MIN_ITERATIONS:
            iterr += 1
            s_time = time.time()
            curr_time = time.time()

            # FE-ANALYSIS
            results = fe_mthm_bc(nely, nelx, penal, x, ke, k_eth, c_ethm, tref, bcs)
            km, kth, um, uth, fm, fth, d_cthm, fixeddofsm, alldofsm, freedofsm, fixeddofsth, alldofsth, freedofsth, fp = (
                results
            )
            t_forward = time.time() - curr_time
            curr_time = time.time()

            # --- Stress and Strain ---
            # Mechanical Stress and Strain / Thermal Stress and Strain
            if w1 == 0.0:
                w_th, vms_th = calc_stress_strain_th(uth, bcs, tref, alpha, e, nu)
                vms = vms_th
                w = w_th
            elif w1 == 1.0:
                w_me, vms_me = calc_stress_strain_me(um, bcs, penal, e0, emin, nu)
                vms = vms_me
                w = w_me
            else:
                w_th, vms_th = calc_stress_strain_th(uth, bcs, tref, alpha, e, nu)
                w_me, vms_me = calc_stress_strain_me(um, bcs, penal, e0, emin, nu)
                vms = vms_th + vms_me
                w = w_th + w_me

            # Plot design update
            if self.plot is True and (iterr % PLOTTING_FREQ == 0 or iterr == 1):
                t_init = np.reshape(uth, (nely + 1, nelx + 1))
                convm = np.array([[0.25, 0.25], [0.25, 0.25]])
                t_field = convolve2d(t_init, convm, mode='valid')
                plot_multi_physics(x, w, vms, t_field, bcs, open_plot=True)

            ndofm = 2 * (nely + 1) * (nelx + 1)

            # Flatten the force matrix once
            flam = fm.flatten()

            # --- Solve for free degrees of freedom ---
            km_ff = km[freedofsm, :][:, freedofsm].tocsc()
            flam_freedofs = flam[freedofsm]
            lamm_freedofs = -spsolve(km_ff, flam_freedofs)
            lamm = np.zeros(ndofm)
            lamm[freedofsm] = lamm_freedofs

            # --- Sensitivity analysis or second solve ---
            temp = lamm.T @ d_cthm - um.T @ d_cthm - fth.T
            temp = np.asarray(temp).flatten()
            kth_sparse = kth.tocsc()
            lamth = spsolve(kth_sparse, temp)

            # ------- END OPTIMIZED CODE ------- #

            # PREPARE SENSITIVITY ANALYSIS
            t_sensitivity = time.time() - curr_time
            curr_time = time.time()
            f0val = 0
            f0valm = 0
            f0valt = 0
            df0dx_mat = np.zeros((nely, nelx))

            df0dx_m = np.zeros((nely, nelx))
            df0dx_t = np.zeros((nely, nelx))

            xval = x.reshape(n, 1)
            # DEFINE CONSTRAINTS
            volconst = np.sum(x) / (volfrac * n) - 1  # shape ()
            fval = volconst  # Column vector of size (1xm)
            dfdx = np.ones((1, n)) / (volfrac * n)  # shape (1, n)

            # CALCULATE SENSITIVITIES
            for elx in range(nelx):
                for ely in range(nely):
                    n1 = (nely + 1) * elx + ely
                    n2 = (nely + 1) * (elx + 1) + ely
                    edof4 = np.array([n1 + 1, n2 + 1, n2, n1], dtype=int)
                    edof8 = np.array(
                        [2 * n1 + 2, 2 * n1 + 3, 2 * n2 + 2, 2 * n2 + 3, 2 * n2, 2 * n2 + 1, 2 * n1, 2 * n1 + 1], dtype=int
                    )

                    ume = um[edof8].flatten()
                    uthe = uth[edof4].flatten()
                    lamm[edof8].flatten()
                    lamthe = lamth[edof4].flatten()
                    x_p = x[ely, elx] ** penal
                    x_p_minus1 = penal * x[ely, elx] ** (penal - 1)

                    # Separate
                    f0valm += x_p * ume.T @ ke @ ume
                    f0valt += x_p * uthe.T @ k_eth @ uthe

                    df0dx_m[ely, elx] = -x_p_minus1 * ume.T @ ke @ ume
                    df0dx_t[ely, elx] = lamthe.T @ (x_p_minus1 * k_eth @ uthe)
                    df0dx_mat[ely, elx] = (df0dx_m[ely, elx] * w1) + (df0dx_t[ely, elx] * w2)  # Weighted sensitivity

            f0val = (f0valm * w1) + (f0valt * w2)

            if self.eval_only is True:
                vf_error = np.abs(np.mean(x) - volfrac)
                return {
                    "structural_compliance": f0valm,
                    "thermal_compliance": f0valt,
                    "volume_fraction": vf_error,
                }
            else:
                vf_error = np.abs(np.mean(x) - volfrac)
                obj_values = [f0valm, f0valt, vf_error]
                # opti_step = OptiStep(obj_values=obj_values, step=iterr)
                opti_steps.append(obj_values)
                design_steps.append(x.astype(np.float16))
                displacement_steps.append(um.astype(np.float16))
                vms_steps.append(vms.astype(np.float16))
                w_steps.append(w.astype(np.float16))
                temp_steps.append(uth.astype(np.float16))


            df0dx = df0dx_mat.reshape(nely * nelx, 1)
            df0dx = (h @ (xval * df0dx)) / hs[:, None] / np.maximum(1e-3, xval)  # Filtered sensitivity

            t_sensitivity_calc = time.time() - curr_time
            curr_time = time.time()

            # UPDATE DESIGN VARIABLES USING MMA
            upp_vec = np.ones((n,)) * upp
            low_vec = np.ones((n,)) * low
            mmainputs = MMAInputs(
                m=m,
                n=n,
                iterr=iterr,
                xval=xval[:, 0],  # selecting appropriate column
                xmin=xmin,
                xmax=xmax,
                xold1=xold1,
                xold2=xold2,
                df0dx=df0dx[:, 0],  # selecting appropriate column
                fval=fval,  # Constraint values
                dfdx=dfdx,
                low=low_vec,
                upp=upp_vec,
                a0=a0,
                a=a[0],
                c=c[0],
                d=d[0],
                f0val=f0val,
            )
            xmma = mmasub(mmainputs)
            t_mma = time.time() - curr_time

            # Store previous density fields
            if iterr > SECOND_ITERATION_THRESHOLD:
                xold2 = xold1
                xold1 = xval
            elif iterr > FIRST_ITERATION_THRESHOLD:
                xold1 = xval

            x = xmma.reshape(nely, nelx)

            # Print results
            change = np.max(np.abs(xmma - xold1))
            change_evol.append(change)
            obj.append(f0val)
            t_total = time.time() - s_time
            # print(
            #     f" It.: {iterr:4d} Obj.: {f0val:10.4f} Vol.: {np.sum(x) / (nelx * nely):6.3f} ch.: {change:6.3f} || t_forward:{t_forward:6.3f} + t_sensitivity:{t_sensitivity:6.3f} + t_sens_calc:{t_sensitivity_calc:6.3f} + t_mma: {t_mma:6.3f} = {t_total:6.3f}"
            # )

            if iterr > MAX_ITERATIONS:
                break

        # print("Optimization finished...")
        vf_error = np.abs(np.mean(x) - volfrac)

        result = {
            "design": x,
            "bcs": bcs,
            "structural_compliance": f0valm,
            "thermal_compliance": f0valt,
            "volume_fraction": vf_error,
            "opti_steps": opti_steps,
            "design_steps": design_steps,
            "displacement_steps": displacement_steps,
            "strain_energy_field": w,
            "von_mises_stress_field": vms,
            "von_mises_stress_field_steps": vms_steps,
            "strain_energy_field_steps": w_steps,
            "temp_steps": temp_steps,
        }
        return result


if __name__ == "__main__":
    nelx = 64
    nely = 64

    client = FeaModel(plot=True)

    lci, tri, rci, bri = get_res_bounds(nelx + 1, nely + 1)

    bcs = {
        "nelx": nelx,
        "nely": nely,
        "fixed_elements": [lci[21], lci[32], lci[43]],
        "force_elements_y": [bri[31]],
        "heatsink_elements": [lci[31], lci[32], lci[33]],
        "volfrac": 0.2,
        "rmin": 1.1,
        "weight": 1.0,  # 1.0 for pure structural, 0.0 for pure thermal
    }

    result = client.run(bcs)
