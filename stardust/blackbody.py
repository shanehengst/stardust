# -*- coding: utf-8 -*-
"""
stardust.blackbody
==================
Planck blackbody functions.
"""
import numpy as np
from .constants import h, c, kb


def Blam(T, w):
    """
    Planck spectral radiance B_λ(T, λ).

    Parameters
    ----------
    T : float        – Temperature [K]
    w : array-like   – Wavelength [m]

    Returns
    -------
    B : ndarray  [W m⁻² sr⁻¹ m⁻¹]
    """
    w = np.asarray(w, dtype=float)
    x = h * c / (w * kb * T)
    # Guard overflow for cold grains / short wavelengths (x >> 1 → B ≈ 0)
    return (2 * h * c**2 / w**5) * np.where(
        x > 700, np.exp(-x), 1.0 / (np.exp(x) - 1.0)
    )


def Planck_nu(A, T, w):
    """
    Planck function as flux density [Jy], normalised to amplitude A.

    Parameters
    ----------
    A : float        – Amplitude (normalisation scale factor)
    T : float        – Temperature [K]
    w : array-like   – Wavelength [µm]

    Returns
    -------
    F_nu : ndarray  [Jy, normalised]
    """
    w = np.asarray(w, dtype=float)
    L   = (2 * h * c**2 / w**5) * (np.exp(h * c / (w * kb * T)) - 1.0)**-1
    F_nu = L * (w * 1e-6)**2 / c
    return A * F_nu / np.max(F_nu)
