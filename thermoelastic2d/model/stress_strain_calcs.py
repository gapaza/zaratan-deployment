import numpy as np
from scipy.signal import convolve2d



def calc_stress_strain_th(uth, bcs, tref, alpha, e, nu):
    nelx = bcs["nelx"]
    nely = bcs["nely"]

    t_init = np.reshape(uth, (nely + 1, nelx + 1))
    convm = np.array([[0.25, 0.25], [0.25, 0.25]])
    t_field = convolve2d(t_init, convm, mode='valid')
    deltaT = t_field.flatten() - tref  # Captures increase in temperature
    thermal_strain_x = alpha * deltaT
    thermal_strain_y = alpha * deltaT
    sigma_x = e * (thermal_strain_x - nu * (thermal_strain_x + thermal_strain_y))
    sigma_y = e * (thermal_strain_y - nu * (thermal_strain_x + thermal_strain_y))
    von_mises_stress = np.sqrt(0.5 * ((sigma_x - sigma_y) ** 2 + sigma_x ** 2 + sigma_y ** 2))
    strain_energy_density = 0.5 * (sigma_x * thermal_strain_x + sigma_y * thermal_strain_y)
    vms_th = np.reshape(von_mises_stress, (nely, nelx))
    w_th = np.reshape(strain_energy_density, (nely, nelx))

    return w_th, vms_th




def calc_stress_strain_me(um, bcs, penal, e0, emin, nu):
    volfrac = bcs['volfrac']
    nelx = bcs["nelx"]
    nely = bcs["nely"]

    edofMat = np.zeros((nelx * nely, 8), dtype=int)
    for elx in range(nelx):
        for ely in range(nely):
            el = ely + elx * nely
            n1 = (nely + 1) * elx + ely
            n2 = (nely + 1) * (elx + 1) + ely
            edofMat[el, :] = np.array(
                [2 * n1 + 2, 2 * n1 + 3, 2 * n2 + 2, 2 * n2 + 3, 2 * n2, 2 * n2 + 1, 2 * n1, 2 * n1 + 1]
            )
    xstress = np.full((nely, nelx), volfrac)
    estress = + (xstress.flatten() ** penal) * (e0 - emin)
    b_ = 0.5 * np.array([[-1, 0, 1, 0, 1, 0, -1, 0], [0, -1, 0, -1, 0, 1, 0, 1], [-1, -1, -1, 1, 1, 1, 1, -1]])
    de = (1 / (1 - nu ** 2)) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])
    deb = de @ b_
    ue = um[edofMat]
    sig = (ue @ deb.T) * estress[:, np.newaxis]
    vms_me = np.sqrt(np.sum(sig ** 2, axis=1) - sig[:, 0] * sig[:, 1] + 2 * sig[:, 2] ** 2)
    vms_me = np.reshape(vms_me, (nely, nelx))
    eps = ue @ b_.T  # shape: (nely*nelx, 3)
    energy_density = 0.5 * (sig[:, 0] * eps[:, 0] + sig[:, 1] * eps[:, 1] + 2 * sig[:, 2] * eps[:, 2])
    w_me = np.reshape(energy_density, (nely, nelx))

    return w_me, vms_me









