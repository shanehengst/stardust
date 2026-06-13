# -*- coding: utf-8 -*-
"""
stardust.disc
=============
Converts pre-computed 3-D orbital distributions into optical-depth maps,
with optional iterative collisional attenuation.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d

from .constants import pi, Me, au


def compute_time_evolved_disc_from_3d(be_values, pd_grt, sgs, m_o, q, timesteps,
                                      rbins_bm, hd_bs, Age, Ms, mu, rho,
                                      use_collisions=False):
    """
    Compute time-evolved debris disc distribution from pre-computed orbital data.

    Parameters
    ----------
    be_values      : ndarray – β values
    pd_grt         : ndarray – shape (n_r, n_s, n_t), normalised probability density
    sgs            : ndarray – grain sizes [µm]
    m_o            : float   – mass injected per timestep [M_Earth]
    q              : float   – grain-size power-law index
    timesteps      : int
    rbins_bm       : ndarray – radial bin mid-points [au]
    hd_bs          : float   – bin width [au]
    Age            : float   – system age [yr]
    Ms             : float   – stellar mass [M_sun]
    mu             : float   – standard gravitational parameter [au³ yr⁻²]
    rho            : float   – grain density [g cm⁻³]
    use_collisions : bool    – include iterative collisional attenuation

    Returns
    -------
    OD_sr   : ndarray (n_r, n_s)  – optical depth per size per bin
    OD_r    : ndarray (n_r,)      – total optical depth vs radius
    CS_s    : ndarray (n_r, n_s)  – grain number per size per bin
    n_sr    : ndarray (n_s,)      – grain number per size (radially integrated)
    m_final : float               – remaining disc mass [M_Earth]
    """
    be_bound = np.asarray(be_values) < 0.5

    dMs   = (sgs * 1e-4)**-q * rho * (4/3) * pi * (sgs * 1e-4)**3
    dNc   = (m_o * Me * 1e3) / np.sum(dMs)
    Ns    = dNc * (sgs * 1e-4)**-q
    CSs   = pi * (sgs * 1e-6)**2
    NsCSs = Ns * CSs

    rdiff     = pi * ((au * (rbins_bm + hd_bs/2))**2 - (au * (rbins_bm - hd_bs/2))**2)
    dt        = Age / timesteps
    P_approx  = rbins_bm**(3/2) * Ms**(-0.5)
    P_cross   = 2 * pi * np.sqrt(rbins_bm**3 / mu)

    # Age weights: grains injected at timestep a contribute (timesteps - a) times
    ages   = np.arange(timesteps, dtype=float)
    base_w = timesteps - ages

    OD_sr = np.einsum('ijk,k,j->ij', pd_grt, base_w, NsCSs) / rdiff[:, None]
    OD_r  = np.sum(OD_sr, axis=1)

    if use_collisions:
        max_iter  = 10
        tol       = 1e-6
        OD_r_prev = OD_r.copy()

        for _ in range(max_iter):
            inv_t_eff = np.zeros((len(rbins_bm), len(sgs)))
            if np.any(be_bound):
                inv_t_eff[:, be_bound]  = (4 * pi * OD_r[:, None]) / P_approx[:, None]
            if np.any(~be_bound):
                inv_t_eff[:, ~be_bound] = (4 * pi * OD_r[:, None]) / P_cross[:, None]

            h_sa     = np.log(2) * dt * np.einsum('rsa,rs->sa', pd_grt, inv_t_eff)
            h_sa     = np.clip(h_sa, 0, None)
            Hcum_sa  = np.cumsum(h_sa, axis=1)
            surv_sa  = np.exp(-Hcum_sa)

            OD_sr = np.einsum('rsa,sa,s->rs', pd_grt, surv_sa, NsCSs) / rdiff[:, None]
            OD_r  = np.sum(OD_sr, axis=1)

            rel_change = np.max(np.abs(OD_r - OD_r_prev) / (OD_r_prev + 1e-30))
            if rel_change < tol:
                break
            OD_r_prev = OD_r.copy()

    # Final radial smoothing
    for s_idx in range(OD_sr.shape[1]):
        col = OD_sr[:, s_idx]
        if np.any(col > 0):
            OD_sr[:, s_idx] = gaussian_filter1d(col, sigma=3.0)
    OD_sr = np.clip(OD_sr, 0, None)

    OD_r    = np.sum(OD_sr, axis=1)
    CS_sr   = OD_sr * rdiff[:, None]
    CS_s    = CS_sr / CSs
    n_sr    = np.sum(CS_sr, axis=0) / CSs
    m_final = np.sum(n_sr * rho * (4/3) * pi * (sgs * 1e-4)**3 / (Me * 1e3))

    return OD_sr, OD_r, CS_s, n_sr, m_final
