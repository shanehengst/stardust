# -*- coding: utf-8 -*-
"""
stardust.radiative
==================
Mie-theory grain property grids (radpressure) and the disc SED builder (DustyMM).
"""
import os
import numpy as np
import pandas as pd
import miepython as mpy
from os import path
from scipy import interpolate
from scipy.integrate import trapezoid as trapz

from .constants import pi, R_s, au, Me, pc, c
from .blackbody import Blam
from .utils import _make_interp2d


# ─────────────────────────────────────────────────────────────────────────────
def radpressure(Ls, Ms, Rs, Ts, rho, composition,
                rbins_bm, fn, fk, fun_spek):
    """
    Build (or load) Mie-theory grain-temperature and efficiency grids.

    Checks the current working directory for cached CSV files; creates them
    on the first run (this may take several minutes).

    Parameters
    ----------
    Ls          : float    – Stellar luminosity [L_sun]
    Ms          : float    – Stellar mass [M_sun]
    Rs          : float    – Stellar radius [R_sun]
    Ts          : float    – Stellar effective temperature [K]
    rho         : float    – Grain bulk density [g cm⁻³]
    composition : str      – Base name of the .lnk optical-constants file
    rbins_bm    : ndarray  – Radial bin mid-points [au]
    fn          : callable – Interpolator n(λ) from the .lnk file
    fk          : callable – Interpolator k(λ) from the .lnk file
    fun_spek    : callable – Log-interpolator of the stellar spectrum

    Returns
    -------
    [f_Tg, f_Qpr, f_Qabs, f_Qsca, sblow, beta]
    """
    n_gs   = 100
    srgs   = np.geomspace(0.01, 3000, n_gs)
    wrange = np.geomspace(0.01, 3000, 101)

    # Stellar spectrum proxy
    f_10spek = fun_spek(np.log10(wrange))
    flux_sa  = 10**f_10spek
    BB_star  = Blam(Ts, wrange * 1e-6)
    BStar    = np.max(BB_star) * (flux_sa / np.max(flux_sa))

    # Cache file names
    Tg_file   = f'df_Tg_{composition}_Ts{Ts}_ns_{n_gs}_master.csv'
    Qpr_file  = f'df_Qpr_{composition}_Ts_{Ts}_rho{rho}_ns_{n_gs}_master.csv'
    Qabs_file = f'df_Qabs_{composition}_Ts_{Ts}_rho{rho}_ns_{n_gs}_master.csv'
    Qsca_file = f'df_Qsca_{composition}_Ts_{Ts}_rho{rho}_ns_{n_gs}_master.csv'

    if all(path.exists(f) for f in (Tg_file, Qpr_file, Qabs_file, Qsca_file)):
        print('All temperature and optical constants files were discovered.')
        df_Tg   = pd.read_csv(Tg_file)
        df_Qpr  = pd.read_csv(Qpr_file)
        df_Qabs = pd.read_csv(Qabs_file)
        df_Qsca = pd.read_csv(Qsca_file)
    else:
        print(f'One or more cached files not found – building grids for {composition}. '
              f'This may take several minutes...')
        gtr  = np.geomspace(2.5, 1500, 101)
        fact = 0.5 * Rs * R_s / au
        nv   = fn(wrange)
        kv   = fk(wrange)

        df_Tg   = pd.DataFrame({'R': rbins_bm})
        df_Qabs = pd.DataFrame({'Wavelength (um)': wrange})
        df_Qsca = pd.DataFrame({'Wavelength (um)': wrange})
        df_Qpr  = pd.DataFrame(columns=['A'], index=range(1))

        for s in srgs:
            s_col  = str(round(s, 4))
            x      = 2 * pi * s / wrange
            qext, qsca, qback, g = mpy.mie(nv - 1.0j * kv, x)
            qabs   = qext - qsca
            qpr    = qabs + qsca * (1 - g)
            star   = trapz(BStar * qabs, wrange)

            df_Qpr[s_col]  = trapz(qpr * BStar, wrange) / trapz(BStar, wrange)
            df_Qabs[s_col] = qabs
            df_Qsca[s_col] = qsca

            r_val = []
            for gt in gtr:
                BBFluxg = Blam(gt, wrange * 1e-6)
                dust    = trapz(BBFluxg * qabs, wrange)
                r_val.append(fact * np.sqrt(star / dust))

            frgt         = interpolate.interp1d(r_val, gtr, kind='linear',
                                                fill_value='extrapolate')
            df_Tg[s_col] = frgt(rbins_bm)

        df_Tg.to_csv(Tg_file,   index=False)
        df_Qpr.to_csv(Qpr_file,  index=False)
        df_Qabs.to_csv(Qabs_file, index=False)
        df_Qsca.to_csv(Qsca_file, index=False)
        print(f'Grids saved.')

    print('Temperature Grid space:');  print(df_Tg)
    print('Qabs values:');             print(df_Qabs)
    print('Qsca values:');             print(df_Qsca)
    print('Qpr values:');              print(df_Qpr)

    # Build interpolators (RegularGridInterpolator replaces removed interp2d)
    tg_rbins = df_Tg['R'].to_numpy()
    f_Tg   = _make_interp2d(srgs, tg_rbins,  df_Tg.iloc[:, 1:].to_numpy())
    f_Qabs = _make_interp2d(srgs, wrange,    df_Qabs.iloc[:, 1:].to_numpy())
    f_Qsca = _make_interp2d(srgs, wrange,    df_Qsca.iloc[:, 1:].to_numpy())
    f_Qpr  = interpolate.interp1d(srgs, df_Qpr.iloc[0, 1:].to_numpy(), kind='linear')

    # Blow-out size
    beta     = [0.574 * Ls * float(df_Qpr[str(round(s, 4))][0]) / (Ms * rho * s)
                for s in srgs]
    s_range  = np.geomspace(0.01, 3000, 3000)
    b_fn     = interpolate.interp1d(srgs, beta, kind='linear', fill_value='extrapolate')
    b_interp = b_fn(s_range)
    idx      = np.argwhere(np.diff(np.sign(0.5 * np.ones(len(s_range)) - b_interp))).flatten()
    sblow    = [s_range[i] for i in idx]

    return [f_Tg, f_Qpr, f_Qabs, f_Qsca, sblow, beta]


# ─────────────────────────────────────────────────────────────────────────────
def DustyMM(smin, dfrac, q, rm, rw, rin, rout, f_sd,
            s_gs, wr, rbins_bm, rho, Ls, Ms, dpc,
            Tg_sr, f_BBgT, Qabs_sw, Qsca_sw, Qpr_s, flux_nu):
    """
    Compute the disc SED (thermal emission) for one parameter set.

    Parameters
    ----------
    smin, dfrac, q  : disc parameters (min size [µm], mass [M_Earth], index)
    rm, rw, rin, rout : belt geometry [au]
    f_sd            : callable – spatial distribution f_sd(s, r)
    s_gs            : ndarray  – grain size grid [µm]
    wr              : ndarray  – wavelength grid [µm]
    rbins_bm        : ndarray  – radial bin mid-points [au]
    rho             : float    – grain density [g cm⁻³]
    Ls, Ms          : float    – stellar luminosity [L_sun], mass [M_sun]
    dpc             : float    – distance [pc]
    Tg_sr           : callable – grain temperature T_g(s, r)
    f_BBgT          : callable – blackbody matrix BB(T, λ)
    Qabs_sw         : callable – Q_abs(s, λ)
    Qsca_sw         : callable – Q_sca(s, λ)
    Qpr_s           : callable – Q_pr(s)
    flux_nu         : ndarray  – stellar photosphere flux [Jy]

    Returns
    -------
    SED_total : ndarray  – Star + disc SED [Jy]
    flux_b    : ndarray  – Disc emission only [Jy]
    MPerb     : float    – Fraction of disc mass in bound grains [%]
    """
    sm_mask  = s_gs >= smin
    DiscMass = dfrac * Me * 1e3   # grams

    dMs  = (s_gs[sm_mask] * 1e-4)**-q * rho * (4/3) * pi * (s_gs[sm_mask] * 1e-4)**3
    dNc  = DiscMass / np.sum(dMs)

    dM_radial_b = 0.0
    flux_b      = np.zeros(len(wr))

    for s in s_gs[sm_mask]:
        dNb     = f_sd(s, rbins_bm).flatten()
        dNb_sum = np.sum(dNb)
        if dNb_sum <= 0:
            continue

        dNb  = (dNb / dNb_sum) * dNc * (s * 1e-4)**(1 - q)
        dNbr = np.trim_zeros(dNb)

        dM_radial_b += np.sum(dNb * rho * (4/3) * pi * (s * 1e-4)**3)

        glocID              = dNb / dNb
        glocID[np.isnan(glocID)] = 0
        gloc                = np.trim_zeros(rbins_bm * glocID)

        GTg    = Tg_sr(s, gloc).flatten()
        BBfns  = f_BBgT(GTg, wr)
        mBBfns = np.multiply(BBfns, Qabs_sw(s, wr))

        # No [::-1] reversal needed: _make_interp2d evaluates columns in the
        # same order as GTg (ascending r / descending T), matching dNbr and gloc
        # directly.  The old interp2d sorted x internally, requiring reversal.
        Flam_all = np.multiply(mBBfns, dNbr * 2 * pi * gloc)
        Flam_sum = np.sum(Flam_all, axis=1)
        Flam     = 4 * pi * Flam_sum * (s * 1e-6)**2
        Fnu      = 1e26 * Flam * (wr * 1e-6)**2 / c

        flux_b += Fnu

    flux_b /= (dpc * pc)**2
    MPerb    = round((dM_radial_b / DiscMass) * 100, 3)
    SED_total = np.add(flux_b, flux_nu)

    return SED_total, flux_b, MPerb
