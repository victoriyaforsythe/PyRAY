#!/usr/bin/env python
# --------------------------------------------------------
# Distribution statement A. Approved for public release.
# Distribution is unlimited.
# This work was supported by the Office of Naval Research.
# --------------------------------------------------------
"""This library contains components for PyRayHF software.

"""

import numpy as np
from PyRayHF import logger


def constants():
    """Define constants for virtual height calculation.

    Returns
    -------
    cp : float
        Constant that relates plasma frequency to plasma density in Hz m^-1.5.
    g_p : float
        g_p * B is the electron gyrofrequency in Hz/T.

    Notes
    -----
    This function provides constants used in virtual height calculations.

    """
    # Constant to convert density to frequency (MHz)
    cp = 8.97866275
    # Proton gyrofrequency constant (Hz/T)
    g_p = 2.799249247e10
    return cp, g_p


def den2freq(density):
    """Convert plasma density to plasma frequency.

    Parameters
    ----------
    density : float or array-like
        Plasma density in m^-3.

    Returns
    -------
    frequency : float or array-like
        Plasma frequency in Hz.

    """
    # Declaring constants
    cp, _ = constants()

    # Test for negative input
    if np.any(np.asarray(density) < 0):
        raise ValueError("Density must be non-negative")

    frequency = np.sqrt(density) * cp
    return frequency


def freq2den(frequency):
    """Convert plasma frequency to plasma density.

    Parameters
    ----------
    frequency : float or array-like
        Plasma frequency in Hz.

    Returns
    -------
    density : float or array-like
        Plasma density in m^-3.

    """
    # Declaring constants
    cp, _ = constants()
    density = (frequency / cp)**2
    return density


def find_X(n_e, f):
    """Calculate the square of the plasma freq over the square of the ion freq.

    Parameters
    ----------
    n_e : array-like
        Electron density in m^-3.
    f : array-like
        Ionosonde frequency in Hz.

    Returns
    -------
    X : float or array-like
        Ratio (f_N / f)^2.

    """
    X = (den2freq(n_e))**2 / f**(2)
    return X


def find_Y(f, b):
    """Calculate Y: the electron gyrofrequency to ionosonde frequency ratio.

    Parameters
    ----------
    f : array-like
        Ionosonde frequency in Hz.
    b : array-like
        Magnetic field magnitude in Tesla.

    Returns
    -------
    Y : array-like
        Electron gyrofrequency / ionosonde frequency.

    """
    _, g_p = constants()
    Y = g_p * b / f
    return Y


def find_mu_mup(X, Y, bpsi, mode):
    """Calculate the phase and group refractive indices (μ and μ′).

    Parameters
    ----------
    X : array-like
        Ratio of plasma and transmission frequencies squared.
    Y : array-like
        Ratio of electron gyrofrequency and transmission frequency.
    bpsi : array-like
        Angle ψ between wave vector and magnetic field in degrees.
    mode : str
        'O' for ordinary or 'X' for extraordinary wave mode.

    Returns
    -------
    mu : array-like
        Phase refractive index μ.
    mup : array-like
        Group refractive index μ′.

    """
    # Compute transverse and longitudinal components of Y
    YT = Y * np.sin(np.deg2rad(bpsi))
    YL = Y * np.cos(np.deg2rad(bpsi))

    # Compute 1-X
    Xm1 = 1.0 - X

    # Calculate alpha and beta as intermediate terms for refractive index
    alpha = 0.25 * YT**4 + YL**2 * Xm1**2
    beta = np.sqrt(alpha)

    # Set mode multiplier depending on propagation mode
    if mode == 'O':
        modeMult = 1.
    if mode == 'X':
        modeMult = -1.

    # Appleton-Hartree denominator and mu
    D = Xm1 - 0.5 * YT**2 + modeMult * beta

    # Select > 0 part
    under_sqrt = 1. - X * Xm1 / D
    under_sqrt[under_sqrt < 0] = np.nan
    mu = np.sqrt(under_sqrt)

    # Apply physical constraints on refractive index
    mu[np.where(mu < 0.)] = 0.
    mu[np.where(mu > 1.)] = np.nan

    # Derivatives with respect to X and Y
    dbetadX = -YL**2 * Xm1 / beta
    dDdX = -1. + modeMult * dbetadX

    dalphadY = ((YT**3 * np.sin(np.deg2rad(bpsi)))
                + (2. * YL * Xm1**2 * np.cos(np.deg2rad(bpsi))))
    dbetadY = 0.5 * dalphadY / beta
    dDdY = -YT * np.sin(np.deg2rad(bpsi)) + modeMult * dbetadY

    # Compute partial derivatives of mu for corrected index
    dmudY = (X * Xm1 * dDdY) / (2. * mu * D**2)
    dmudX = (1. / (2. * mu * D)) * (2. * X - 1. + X * Xm1 / D * dDdX)

    # Modified refractive index considering dispersion effects
    mup = mu - (2. * X * dmudX + Y * dmudY)

    return mu, mup


def find_vh(X, Y, bpsi, dh, alt_min, mode):
    """Calculate virtual height for given mode.

    Parameters
    ----------
    X : array-like
        Plasma to ionosonde frequency ratio squared.
    Y : array-like
        Electron gyrofrequency to ionosonde frequency ratio.
    bpsi : array-like
        Angle between wave vector and magnetic field (degrees).
    dh : array-like
        Vertical layer thickness in km.
    alt_min : float
        Minimum altitude in km.
    mode : str
        'O' or 'X' mode.

    Returns
    -------
    vh : array-like
        Virtual height in km.

    """
    # Find the phase refractive index μ and the group refractive index μ′ for
    # ordinary (O) and extraordinary (X) modes of signal propagation
    _, mup = find_mu_mup(X, Y, bpsi, mode)

    # Find virtual height as vertical integral through μ′
    vh = np.nansum(mup * dh, axis=1) + alt_min
    return vh


def smooth_nonuniform_grid(start, end, n_points, sharpness):
    """Generate smooth non-uniform grid from start to end.

    Parameters
    ----------
    start : float
    end : float
    n_points : int
    sharpness : float
        Controls how sharply resolution increases near `end`.

    Returns
    -------
    x : ndarray
        Non-uniformly spaced grid.

    """
    # Uniform grid [0, 1]
    u = np.linspace(0.0, 1.0, n_points)

    # Flip to make fine resolution near 'end'
    flipped_u = 1.0 - u

    factor = (np.exp(sharpness * flipped_u) - 1.0) / (np.exp(sharpness) - 1.0)
    x = 1. - (start + (end - start) * factor)
    return x


def regrid_to_nonuniform_grid(f, n_e, b, bpsi, aalt, npoints):
    """Regrid profile to smooth non-uniform vertical grid.

    Parameters
    ----------
    f : array-like
        Ionosonde frequency in Hz.
    n_e : array-like
        Electron density in m^-3.
    b : array-like
        Magnetic field magnitude.
    bpsi : array-like
        Angle to magnetic field vector in degrees.
    aalt : array-like
        Altitude profile in km.
    npoints : int
        Points in new vertical grid.

    Returns
    -------
    regridded : dict
        Dictionary with re-gridded arrays

    """
    # Create non-regular grid that has low resolution near zero and hight
    # resolution near one
    start = 0
    end = 1
    sharpness = 10.
    multiplier = smooth_nonuniform_grid(start, end, npoints, sharpness)

    N_grid = multiplier.size
    N_freq = f.size
    ind_grid = np.arange(0, N_grid, 1)

    # Limit input arrays to the fof2 of the ionosphere
    ind_max = np.argmax(n_e)
    n_e = n_e[0: ind_max]
    b = b[0: ind_max]
    bpsi = bpsi[0: ind_max]
    aalt = aalt[0: ind_max]

    # How close to the reflection height do we want to get
    dh = 1e-20

    # An array of critical height for the given ionosonde frequency
    # We subtract dh so that the critical height is not exactly reached
    critical_height = np.interp(f, den2freq(n_e), aalt) - dh

    # Make arrays 2-D
    multiplier_2d = np.full((N_freq, N_grid), multiplier)
    critical_height_2d = np.transpose(np.full((N_grid, N_freq),
                                              critical_height))
    new_alt_2d = multiplier_2d * (critical_height_2d - aalt[0]) + aalt[0]

    dh_2d = np.concatenate((np.diff(new_alt_2d, axis=1), np.full((N_freq, 1),
                                                                 dh)), axis=1)

    new_ind_2d = np.full((N_freq, N_grid), ind_grid)

    # Flattened array of new altitudes where we want to sample the density
    # profile, so we can apply 1-D Numpy interpolation, that is faster
    new_alt_1d = np.reshape(new_alt_2d, new_ind_2d.size)
    # Create arrays on the modified grid
    den_mod = np.reshape(np.interp(new_alt_1d, aalt, n_e), new_alt_2d.shape)
    bmag_mod = np.reshape(np.interp(new_alt_1d, aalt, b), new_alt_2d.shape)
    bpsi_mod = np.reshape(np.interp(new_alt_1d, aalt, bpsi), new_alt_2d.shape)
    ionosonde_freq_mod = np.transpose(np.full((N_grid, N_freq), f))

    # Create a dictionary to hold the new re-gridded arrays
    regridded = {'freq': ionosonde_freq_mod,
                 'den': den_mod,
                 'bmag': bmag_mod,
                 'bpsi': bpsi_mod,
                 'dist': dh_2d,
                 'alt': new_alt_2d,
                 'crit_height': critical_height_2d,
                 'ind': new_ind_2d}
    return regridded


def vertical_to_magnetic_angle(inclination_deg):
    """Calculate angle between vertical and magnetic field vector.

    Parameters
    ----------
    inclination_deg : float or ndarray
        Magnetic inclination in degrees (positive = downward).

    Returns
    -------
    vertical_angle : float or ndarray
        Angle between vertical and magnetic field in degrees.

    """
    vertical_angle = 90.0 - np.abs(inclination_deg)
    return vertical_angle


def vertical_forward_operator(freq, den, bmag, bpsi, alt, mode='O',
                              n_points=2000):
    """Calculate virtual height from ionosonde freq and ion profile.

    Parameters
    ----------
    freq : array-like
        Frequency in MHz.
    den : array-like
        Electron density in m^-3.
    bmag : array-like
        Magnetic field magnitude in Tesla.
    bpsi : array-like
        Angle to magnetic field vector.
    alt : array-like
        Altitude profile in km.
    mode : str
        'O' or 'X' propagation mode.
    n_points : int
        Number of vertical grid points.

    Returns
    -------
    vh : ndarray
        Virtual height in km.

    """
    # Check that input arrays have the same size
    if (den.shape != bmag.shape != bpsi.sahpe != alt.shape):
        logger.error("Error: freq, den, bmag, bpsi, alt should have same size")

    # Limit the ionosonde frequency array up tp the ionospheric critical
    # frequency foF2 and convert form MHz to Hz.
    foF2 = np.max(den2freq(den))

    # Index where ionosonde frequency is less then foF2 value
    ind = np.where((freq * 1e6) < foF2)

    # Select ionosonde frequency with this criteria
    freq_lim = freq[ind] * 1e6

    # Make empty array to collect virtual height of the same size as input
    # frequency array
    vh = np.zeros((freq.size)) + np.nan

    # Interpolate input arrays into a new stretched grid based on the
    # reflective height for each ionosonde frequency
    # Frequency needs to be converted to MHz from Hz
    regridded = regrid_to_nonuniform_grid(freq_lim,
                                          den,
                                          bmag,
                                          bpsi,
                                          alt,
                                          n_points)

    # Find the ratio of the square of the plasma frequency f_N to the square of
    # the ionosonde frequency f.
    aX = find_X(regridded['den'], regridded['freq'])

    # Find the ratio of electron gyrofrequency and the ionosonde frequency
    aY = find_Y(regridded['freq'], regridded['bmag'])

    # Find virtual height
    vh[ind] = find_vh(aX, aY, regridded['bpsi'], regridded['dist'],
                      np.min(alt), mode)
    return vh


def model_vh(F2, F1, E, f_in, alt, b_mag, b_psi):
    """
    Compute vertical virtual height using a modeled electron density profile
    and perform ray tracing.

    Parameters
    ----------
    F2 : dict
        Dictionary of F2 layer parameters. Must include:
        - 'Nm': peak electron density (NmF2)
        - 'hm': peak height (hmF2)
        - 'B_bot': thickness of the bottomside of the F2 layer
    F1 : dict
        Dictionary of F1 layer parameters. Must include:
        - 'P': shape factor or profile parameter
    E : dict
        Dictionary of E layer parameters. Must include:
        - 'hm': peak height of the E layer
    f_in : ndarray
        Input frequency [MHz].
    alt : ndarray
        1D array of altitudes [km].
    b_mag : ndarray
        1D array of magnetic field magnitudes [nT].
    b_psi : ndarray
        1D array of magnetic field dip angles [rad].

    Returns
    -------
    vh_O : ndarray
        Virtual height trace (O-mode) [km].
    EDP : ndarray
        Reconstructed electron density profile [m^-3].

    """
    # Using PyIRI formalizm update the F1 layer parameters, in case F2
    # parameters have changed
    (NmF1,
     foF1,
     hmF1,
     B_F1_bot) = PyIRI.edp_update.derive_dependent_F1_parameters(F1['P'],
                                                                 F2['Nm'],
                                                                 F2['hm'],
                                                                 F2['B_bot'],
                                                                 E['hm'])

    # Update F1 with derived values
    F1['Nm'] = NmF1
    F1['hm'] = hmF1
    F1['fo'] = foF1
    F1['B_bot'] = B_F1_bot

    # Reconstruct electron density profile
    EDP = PyIRI.edp_update.reconstruct_density_from_parameters_1level(F2,
                                                                      F1,
                                                                      E,
                                                                      alt)
    EDP = EDP[0, :, 0]

    # Set ray-tracing parameters
    mode = 'O'
    n_points = 200

    # Run vertical raytracing using PyRayHF
    vh_O = PyRayHF.library.vertical_forward_operator(f_in, edp,
                                                     b_mag, b_psi,
                                                     alt, mode, n_points)
    return vh_O, EDP
