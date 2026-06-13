# -*- coding: utf-8 -*-
"""
stardust.orbital
================
Keplerian orbit integration and time-evolved grain probability distributions.

All functions are stateless – simulation globals (rbins_bm, Age, Ms, …) are
passed as explicit keyword arguments so each function can be tested and used
independently of the main driver script.
"""
import math
import numpy as np
from scipy.integrate import solve_ivp
from scipy.ndimage import gaussian_filter1d

from .constants import pi


# ─────────────────────────────────────────────────────────────────────────────
def orbit(a, b, e, P, dt):
    """
    Keplerian orbit integration.

    Parameters
    ----------
    a  : float  – Semi-major axis [au]
    b  : float  – Semi-minor axis [au]
    e  : float  – Eccentricity
    P  : float  – Orbital period [yr]
    dt : float  – Timestep [yr]

    Returns
    -------
    [rd, tht, ti]  – Radial distances [au], angles [rad], times [yr]
    """
    tht = []
    rd  = []
    N   = math.ceil(P / dt)
    ti  = np.linspace(0, P, N)
    th  = 0.0

    for t in ti:
        rth = a * (1 - e**2) / (1 + e * math.cos(th))
        thn = th + (a * b * 2 * pi) / (P * rth**2) * dt
        rd.append(rth)
        tht.append(thn)
        th = thn

    return [rd, tht, ti]


# ─────────────────────────────────────────────────────────────────────────────
def prob_grain_rp_prc_time_evolved_3d(be_values, rpi, sgs, timesteps,
                                      rbins_bm, rbins_bins, hd_bs, hd_max,
                                      Age, Ms, mu, cs, f_gauss):
    """
    Time-evolved grain radial probability distribution (radiation pressure + PR drag).

    Returns a 3-D array of shape (n_rbins, n_be, timesteps).

    Parameters
    ----------
    be_values  : ndarray  – β values for bound grains
    rpi        : float    – Parent-belt launch radius [au]
    sgs        : ndarray  – Grain size grid [µm]
    timesteps  : int      – Number of time steps
    rbins_bm   : ndarray  – Radial bin mid-points [au]
    rbins_bins : ndarray  – Radial bin edges [au]
    hd_bs      : float    – Bin width [au]
    hd_max     : float    – Maximum radial extent [au]
    Age        : float    – System age [yr]
    Ms         : float    – Stellar mass [M_sun]
    mu         : float    – Standard gravitational parameter [au³ yr⁻²]
    cs         : float    – Speed of light [au yr⁻¹]
    f_gauss    : callable – Belt surface-density profile f(r)
    """
    g_prob_all = np.zeros((len(rbins_bm), len(be_values), timesteps))
    t_eval     = np.linspace(0, Age, timesteps)

    for i, be in enumerate(be_values):

        if be < 0.5:   # ── Bound grain: PR drag shrinks orbit ───────────────

            e      = be / (1 - be)
            ai     = rpi / (1 - e)
            C      = ai * e**(-4/5) * (1 - e**2)
            A_val  = -5.0 * be * mu / (2.0 * cs * C**2)

            def dedt_local(t, e_arr):
                if e_arr[0] <= 0:
                    return [0.0]
                return [A_val * (1.0 - e_arr[0]**2)**1.5 / e_arr[0]**0.6]

            sol  = solve_ivp(dedt_local, (0, Age), [e], t_eval=t_eval,
                             method='RK45', rtol=1e-8, atol=1e-10)
            e_t  = sol.y[0]
            e_t[e_t < 0] = 0

            for t_idx in range(len(sol.t)):
                ecc = e_t[t_idx]
                if ecc <= 0:
                    break

                a_curr  = C / (ecc**(-4/5) * (1 - ecc**2))
                rp_curr = C / (ecc**(-4/5) * (1 + ecc))
                ra_curr = 2 * a_curr - rp_curr
                P_curr  = a_curr**(3/2) * Ms**(-0.5)
                b_curr  = a_curr * np.sqrt(1 - ecc**2)

                if ra_curr - rp_curr > 1 and rp_curr > 1:

                    if ra_curr > hd_max:
                        rat   = hd_max
                    else:
                        idx_ra = np.abs(rbins_bins - ra_curr).argmin()
                        rat    = rbins_bins[idx_ra]

                    idx_rp  = np.abs(rbins_bins - rp_curr).argmin()
                    rpt     = rbins_bins[idx_rp]

                    r_steps = np.int64((rat - rpt) / hd_bs + 1)
                    rad     = np.linspace(rpt, rat, r_steps)

                    thi  = (a_curr * (1 - ecc**2) / rad - 1) / ecc
                    thi  = np.clip(thi, -1, 1)
                    th1  = np.arccos(thi[:-1])
                    th2  = np.arccos(thi[1:])
                    dth  = th2 - th1
                    dt_i = np.multiply(dth, P_curr * rad[:-1]**2 / (a_curr * b_curr * 2 * pi))
                    dt_i /= np.sum(dt_i)

                    rad_centers = rad[:-1]
                    if len(rad_centers) >= 2:
                        g_time_rt = np.interp(rbins_bm, rad_centers, dt_i,
                                              left=0.0, right=0.0)
                        g_time_rt = np.maximum(g_time_rt, 0.0)
                    else:
                        g_time_rt = np.zeros(len(rbins_bm))
                        idx_closest = np.abs(rbins_bm - rad_centers[0]).argmin()
                        g_time_rt[idx_closest] = dt_i[0]

                    g_time_rt = gaussian_filter1d(g_time_rt.astype(float), sigma=2.0)
                    g_time_rt = np.clip(g_time_rt, 0, None)
                    if np.sum(g_time_rt) > 0:
                        g_time_rt /= np.sum(g_time_rt)
                    g_prob_all[:, i, t_idx] = g_time_rt

                elif ra_curr - rp_curr <= 1 and rp_curr > 1:
                    # Circularised – delta function at pericentre
                    g_time_rt = np.zeros(len(rbins_bm))
                    idx_rp    = np.abs(rbins_bins - rp_curr).argmin()
                    g_time_rt[idx_rp] = 1.0
                    g_time_rt = gaussian_filter1d(g_time_rt, sigma=2.0)
                    if np.sum(g_time_rt) > 0:
                        g_time_rt /= np.sum(g_time_rt)
                    g_prob_all[:, i, t_idx] = g_time_rt

        else:   # ── Blow-out grain: escapes radially ──────────────────────────

            Vk     = np.sqrt(mu / rpi)
            ro     = max(1, math.floor(rpi))
            nsteps = np.int64((hd_max - ro) / hd_bs + 1)
            rvalues = np.linspace(ro, hd_max, nsteps)
            rvalz   = np.zeros(max(0, ro - 1))

            if f_gauss(rpi) > 0.001:
                Vsr = Vk * np.sqrt(np.maximum(
                    2 * (be - 0.5) + 2 * (1 - be) * rpi / np.maximum(rvalues, 1e-10),
                    1e-10))
                nsr = 1.0 / (rvalues**2 * np.maximum(Vsr, 1e-10))
                if np.sum(nsr) > 0:
                    nsr /= np.sum(nsr)
                nr = np.concatenate((rvalz, nsr), axis=None)
            else:
                nr = np.zeros(int(hd_max))

            r_age = np.sqrt(mu * (2 * be - 1) / rpi) * Age
            if r_age > 0:
                g_prob_all[:len(nr), i, :] = nr[:, np.newaxis] * (hd_max / r_age)

    return g_prob_all


# ─────────────────────────────────────────────────────────────────────────────
def prob_grain_gravity_only_time_evolved_3d(rpi, sgs, timesteps, rbins_bm, Ms):
    """
    Time-evolved grain probability distribution for gravity only (no RP, no PR).

    Returns a 3-D array of shape (n_rbins, n_sgs, timesteps).
    Grains remain on circular orbits at rpi – distribution is a Gaussian-smoothed
    delta-function, constant across all timesteps.

    Parameters
    ----------
    rpi      : float    – Birth radius [au]
    sgs      : ndarray  – Grain size grid [µm]
    timesteps: int
    rbins_bm : ndarray  – Radial bin mid-points [au]
    Ms       : float    – Stellar mass [M_sun]
    """
    g_prob_all = np.zeros((len(rbins_bm), len(sgs), timesteps))

    g_time_rt          = np.zeros(len(rbins_bm))
    idx_rpi            = np.abs(rbins_bm - rpi).argmin()
    g_time_rt[idx_rpi] = 1.0
    g_time_rt          = gaussian_filter1d(g_time_rt.astype(float), sigma=2.0)
    g_time_rt          = np.clip(g_time_rt, 0, None)
    if np.sum(g_time_rt) > 0:
        g_time_rt /= np.sum(g_time_rt)

    for t_idx in range(timesteps):
        for grain_idx in range(len(sgs)):
            g_prob_all[:, grain_idx, t_idx] = g_time_rt

    return g_prob_all


# ─────────────────────────────────────────────────────────────────────────────
def prob_grain_gravity_rp_time_evolved_3d(be_values, rpi, sgs, timesteps,
                                          rbins_bm, Ms):
    """
    Time-evolved grain probability distribution for gravity + RP (no PR drag).

    Returns a 3-D array of shape (n_rbins, n_be, timesteps).
    Distribution is steady-state (same for all timesteps) since orbits don't
    evolve without PR drag.

    Parameters
    ----------
    be_values : ndarray  – β values
    rpi       : float    – Birth radius [au]
    sgs       : ndarray  – Grain size grid [µm]
    timesteps : int
    rbins_bm  : ndarray  – Radial bin mid-points [au]
    Ms        : float    – Stellar mass [M_sun]
    """
    g_prob_all = np.zeros((len(rbins_bm), len(be_values), timesteps))

    for i, be in enumerate(be_values):
        if be < 0.5:
            e  = be / (1 - be)
            a  = rpi / (1 - e)
            b  = a * np.sqrt(1 - e**2)
            ra = a * (1 + e)
            P  = a**(3/2) * Ms**(-0.5)

            rp_idx = np.abs(rbins_bm - rpi).argmin()
            ra_idx = np.abs(rbins_bm - ra).argmin()

            if rp_idx <= ra_idx:
                rad_segment = rbins_bm[rp_idx:ra_idx + 1]
                if len(rad_segment) > 1:
                    thi  = (a * (1 - e**2) / rad_segment - 1) / e
                    thi  = np.clip(thi, -1, 1)
                    th1  = np.arccos(thi[:-1])
                    th2  = np.arccos(thi[1:])
                    dth  = th2 - th1
                    dt_i = dth * P * rad_segment[:-1]**2 / (a * b * 2 * np.pi)

                    if np.sum(dt_i) > 0:
                        dt_i /= np.sum(dt_i)
                        dt_i  = gaussian_filter1d(dt_i.astype(float), sigma=2.0)
                        dt_i  = np.clip(dt_i, 0, None)
                        if np.sum(dt_i) > 0:
                            dt_i /= np.sum(dt_i)
                        for t_idx in range(timesteps):
                            g_prob_all[rp_idx:rp_idx + len(dt_i), i, t_idx] = dt_i

        # be >= 0.5: blow-out, leave as zero
    return g_prob_all
