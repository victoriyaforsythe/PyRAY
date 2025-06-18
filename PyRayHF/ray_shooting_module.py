# ray_shooting.py

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq


def constants():
    cp = 8.97866275
    g_p = 2.799249247e10
    return cp, g_p


def get_interp_refractive_indices(input_example, freq, mode='O'):
    cp, gp = constants()

    den = input_example['den']
    alt = input_example['alt']
    bmag = input_example['bmag']
    bpsi = input_example['bpsi']

    omega = 2 * np.pi * freq * 1e6
    omega_p2 = (cp * np.sqrt(np.abs(den))) ** 2

    X = omega_p2 / omega**2
    Y = gp * bmag / omega
    cos_b = np.cos(np.radians(bpsi))

    if mode.upper() == 'O':
        n2 = 1 - X / (1 - Y**2 * cos_b**2)
    elif mode.upper() == 'X':
        n2 = 1 - X / (1 + Y**2 * cos_b**2)
    else:
        raise ValueError("Mode must be 'O' or 'X'.")

    n2 = np.maximum(n2, 1e-6)
    n = np.sqrt(n2)
    ng = n * (1 + (X / (1 - X)))

    n_interp = interp1d(alt, n, kind='linear', bounds_error=False, fill_value="extrapolate")
    ng_interp = interp1d(alt, ng, kind='linear', bounds_error=False, fill_value="extrapolate")
    ne_interp = interp1d(alt, den, kind='linear', bounds_error=False, fill_value="extrapolate")

    return n_interp, ng_interp, ne_interp


def trace_oblique_ray_full(n_interp, z0, theta0_deg, dz=0.5, z_max=600.0):
    theta0 = np.radians(theta0_deg)
    z_list = [z0]
    x_list = [0.0]
    theta_list = [theta0]

    n0 = n_interp(z0)
    z = z0
    upward = True

    while True:
        n = n_interp(z)
        sin_theta = (n0 * np.sin(theta0)) / n

        if abs(sin_theta) >= 1.0:
            if upward:
                upward = False
                dz = -dz
                theta0 = np.arcsin(np.sign(sin_theta) * 1.0 - 1e-6)
                continue
            else:
                break

        theta = np.arcsin(sin_theta)
        dx = abs(dz) * np.tan(theta)

        x_new = x_list[-1] + dx
        z_new = z + dz

        if z_new < z0:
            break

        x_list.append(x_new)
        z_list.append(z_new)
        theta_list.append(theta)
        z = z_new

    return np.array(z_list), np.array(x_list), np.array(theta_list)


def ray_endpoint(n_interp, z0, theta_deg):
    z_path, x_path, _ = trace_oblique_ray_full(n_interp, z0=z0, theta0_deg=theta_deg)
    return x_path[-1]


def find_launch_angle(n_interp, z0, x_target, theta_bounds=(40.0, 70.0)):
    def objective(theta_deg):
        try:
            x_end = ray_endpoint(n_interp, z0, theta_deg)
            return x_end - x_target
        except:
            return np.inf

    return brentq(objective, theta_bounds[0], theta_bounds[1], xtol=5.0)
