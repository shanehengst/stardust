#!/usr/bin/env python3
#!/usr/bin/env bash
# -*- coding: utf-8 -*-
"""
    Created on Thur 9 Jan 2021
    Latest Version: Sat 13 June 2026
    
    @author: Shane
    
    1-Dimensional model of debris discs
    Grains affected by radiation pressure (using beta = Frad/Fgrav) and PR Effect
    Included is a simple collisional model dependent on the disc's optical depth
    Optical Constants for grain: miepython
    """

#libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import LogNorm
import math
import time
from astropy.modeling.models import BlackBody as BBody
from astropy import units as u
from scipy import interpolate

#Constants (kg-m-s)
#Universal Contants
G = 6.673705*10**-11        #Gravitational constant
c = 299792458           #Speed of Light
h = 6.62607004*10**-34      #Planck's constant
kb = 1.38064852*10**-23     #boltzman constant

#Sun values (Note in Solar units L=M=R=1)
L_s = 3.845*10**26           #luminosity
M_s = 1.9885*10**30          #mass
R_s = 6.957*10**8             #radius
T_s = 5770                #surface temperature (K)

#Solar System Units
au = 149597870700           #astronomical unit defined by IAU [m] (https://cneos.jpl.nasa.gov/glossary/au.html)
Me = 5.972*10**24           #Earth Mass [kg]
pc = 3.086*10**16           #parsec [m]

#Mathematical constants
pi = 3.1415926535       #pi (10 d.p.)

#initial time
t0 = time.time()

#Format tick labels
formatter = FuncFormatter(lambda y, _: '{:.16g}'.format(y))

#-----------------------------------------------------------------#
# NOTE: All functions (orbit, Blam, Planck_nu, prob_grain_*, compute_time_evolved_disc_from_3d,
#       DustyMM, radpressure) are defined in the stardust/ package and imported below.
#       To modify a function, edit the corresponding file in stardust/:
#         orbital.py    – orbit, prob_grain_*
#         blackbody.py  – Blam, Planck_nu
#         disc.py       – compute_time_evolved_disc_from_3d
#         radiative.py  – radpressure, DustyMM
#         utils.py      – _make_interp2d
#-----------------------------------------------------------------#


# ═══════════════════════════════════════════════════════════════════════════════
# ── Import from stardust/ package (overrides the function definitions above) ──
# ── Each function now receives simulation globals as explicit parameters,     ──
# ── making them independently importable and testable.                        ──
# ═══════════════════════════════════════════════════════════════════════════════
import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from stardust import *           # constants, Blam, Planck_nu, all orbit/disc/radiative fns
from stardust.utils import _make_interp2d


## Wavelength Space##
wr = np.geomspace(0.1,3000,3000) # Wavelength space in microns

###Host Star properties in Solar units##
Ls = 1        #Luminosity of Star [Solar Luminosity]
Ms = 1       #Mass of Star [Solar Masses]
Rs = 1          #Radius of Star [Solar Radius]
Ts = 5772       #star temperature [K]
Age = 100e6         #Age of System [years]
dpc = 10            #distance to star system [pc]
mu = 4*pi**2*Ms      #Standard graviational parameter of the Star: [au]^3/[yr]^2

###Star Flux values: Blackbody + Spectrum (if available)
###Blackbody Stellar spectrum (default)
bb = BBody(temperature=Ts*u.K)
d = dpc*pc   #distance to Star [m]
Rad = Rs*R_s   #Star Radius: Sun [m]
As = pi*(Rad/d)**2 #Amplitude of Star's blackbody planck fn (i.e. for this case: cross-section of Sun) and convert to Jansky (as BBody units are given in: erg / (cm2 Hz s sr))
flux_s = As*Blam(Ts,wr*10**-6)*10**26*((wr*10**-6)**2)/(c)
#
###Stellar Spectrum##
##Import Model Atmosphere values
##f_spek = np.loadtxt('HD13895_BTGen_8800.txt', dtype=[('wave', float),('fspek', float)])
f_spek = np.loadtxt('BTGen-5770.txt', dtype=[('wave', float),('fspek', float)])

#Convert, Extrapolate and Calibrate with Blackbody Spectrum of similar temperature
wave = f_spek["wave"]*0.0001 #Convert from Angstroms to Microns
fspek = f_spek["fspek"]

#Extrapolate/Interplate F(wave)
fun_spek = interpolate.interp1d(np.log10(wave),np.log10(fspek),kind = 'linear', fill_value = 'extrapolate')

#Convert flux (wave) to flux (frequency)
sc_flux = 11.542  #taken from StellarAtmFit function
f_10spek = fun_spek(np.log10(wr))
flux_nu = (10**f_10spek)*((wr*10**-6)**2)/(c)  #unscaled flux (frequency)
flux_nu = sc_flux*flux_nu/np.max(flux_nu) #scaled photosphere flux



#-----------------------------------------------------------------#
##Radial distance parameter space##
##distances are in au##
hd_bs = 1      #size of bin: recommened for debris discs: 1 au
hd_min = 0      #minimum radial distance
hd_max = 500  #maximum radial distance
hd_minm = hd_min + hd_bs/2   #minimum mid space  value + hd_bs/2
hd_maxm = hd_max - hd_bs/2  #maximum mid space value - hd_bs/2
print(f'Maximum distance of grain used in model: {hd_max} au')  #note smallest grains may go beyond this distance (depending on r_min)
hd_steps = np.int64((hd_maxm-hd_minm)/hd_bs + 1)
rbins_bm = np.linspace(hd_minm, hd_maxm,hd_steps)

#bin space for histogram (binning radial values in orbit def)
bin_steps = np.int64(hd_max/hd_bs + 1)
rbins_bins = np.linspace(hd_min,hd_max,bin_steps)



#-----------------------------------------------------------------#

##Grain properties##
#set grain size regime
n_gs = 100  #number of grain sizes
s_gs = np.geomspace(0.01,3000,n_gs) #Set sizes
#set wavelength regime
wrange = np.geomspace(0.01,3000,101)    #wavelength range used for this calculation | note this is different from the user defined for determining overall SED

##Grain Belt Properties##
rho = 3.3 #grain density
###Grain Optical Constants
composition = 'silicate_d03'

#read in nk (optical constants file): 'text' file Col1: Wavelength (um) range: ~10^-5 to ~10^5 / Col2: n Col3: k
nk = pd.read_csv(composition+'.lnk', header = None, delim_whitespace=True)

#extract column information
nkw = nk.iloc[:,0].to_numpy() #if given as a function of wavelength
n = nk.iloc[:,1].to_numpy() #n values optical constants for dirty ice
k = nk.iloc[:,2].to_numpy() #k values optical constants for dirty ice

#create functions for n & k
fn = interpolate.interp1d(nkw,n,fill_value = 'extrapolate')
fk = interpolate.interp1d(nkw,k,fill_value = 'extrapolate')
nv = fn(wr) #n optical constant as a function of the wavelength space
kv = fk(wr) #k optical constant as a function of the wavelength space
#Determine Temperature,Qpr,Qabs,Qsca profiles
#Checks if temperate profile and mie theory files are present for host star and grain composition & density. If not, creates new files.
[Tg_sr,Qpr_s,Qabs_sw,Qsca_sw,sblowA,beta] = radpressure(Ls,Ms,Rs,Ts,rho,composition,
                                                         rbins_bm,fn,fk,fun_spek)

#-----------------------------------------------------------------#
#Create Blackbody Profiles for temperature range
gtr = np.geomspace(2.5,1500,500)
df_BBprofiles = pd.DataFrame({'Wavelength (um)': wr})
#Define column name: Wavelength
df_wave = {'Wavelength (um)': wr}
# Compute all columns first and store in df_wave
for gt in gtr:
    df_wave[gt] = Blam(gt, wr * 10**-6)

# Create the DataFrame in one go
df_BBprofiles = pd.DataFrame(df_wave)


#Interpolate 2D grid space for grain size regime and radial distance values
f_BBgT = _make_interp2d(gtr, wr, df_BBprofiles.iloc[:,1:].to_numpy())  #f_BBgT (grain temperature, wavelength range)

#-----------------------------------------------------------------#
##Creating Density grid for grains released from 1au##
##To then be fitted any grain release at any distance
thr = np.linspace(0,pi,36000)  #angle space for radial function


#-----------------------------------------------------------------#

#Constant Input parameters for Outer belt
q = 3.5              #Exponent of the grain size distribution [-] | q = 3.5 steady state collisional cascade
dfrac = 10           #Mass of grains [Earth mass x 10**-3]
smin_r = 0.01        #Minimum grain size [microns]
smax_r = 3000        #Maximum grain size [microns]
rin = 50             #Radial inner limit [au]
rout = 150           #Radial outer limit [au]
rm = 100             #Gaussian mean distance [au]
drm = 0.1            #Fractional width (Delta rm)
rw = rm*drm/2.35482  #Gaussian width [au]
m_im = 1e-3          #Initial mass of grains released at each timestep [Earth mass]

#Solar System Units
au = 149597870700           #astronomical unit defined by IAU [m] (https://cneos.jpl.nasa.gov/glossary/au.html)
Me = 5.972*10**24           #Earth Mass [kg]
pc = 3.086*10**16           #parsec [m]
cs = 63241.077              #speed of light [au/yr]

##Toggle for collision treatment in time-evolved disc model##
# Options: True (with collisions) or False (no collisions)
USE_COLLISIONS = True

##Beta Values for scenario
be_s = 0.574 * Ls * Qpr_s(s_gs) / (Ms * rho * s_gs)
be_mask = be_s > 0

r_steps = np.int64((rout-rin)/hd_bs + 1) #steps will be same size as rbins_bm values
r_belt = np.linspace(rin, rout, r_steps) #radial parametric space for belt

# Surface (area) density calculation
gauss_mask = (rbins_bm >= rin) & (rbins_bm <= rout)
ddgauss = np.where(gauss_mask, np.exp(-0.5 * ((rbins_bm - rm) / rw) ** 2), 0)
ddgauss /= np.sum(ddgauss)  # Normalise

f_gauss = interpolate.interp1d(rbins_bm, ddgauss)

timesteps = 100
prob_densities_time = np.zeros((len(rbins_bm), len(be_s), timesteps))
print("Computing 3D time-evolved orbital distributions... this may take several minutes depending on your set up with the number of timesteps, radial distriubiton of belt, and maximum size of radial distance.")
t_start_p3d = time.time()
for idx, r in enumerate(rbins_bm):
    if f_gauss(r) != 0:
        prob_3d = prob_grain_rp_prc_time_evolved_3d(
            be_s[be_mask], r, s_gs, timesteps,
            rbins_bm, rbins_bins, hd_bs, hd_max,
            Age, Ms, mu, cs, f_gauss)
        prob_densities_time[:, be_mask, :] += f_gauss(r) * prob_3d
t_end_p3d = time.time()
print(f'Orbital evolution (all belt radii): {t_end_p3d - t_start_p3d:.2f} s ({(t_end_p3d - t_start_p3d)/60:.2f} min)')

print("Computing optical depths from 3D distributions...")
t_start_ctd = time.time()
OD_sr, OD_r, CS_s, n_sr, m_final = compute_time_evolved_disc_from_3d(
    be_s[be_mask], prob_densities_time, s_gs, m_im, q, timesteps,
    rbins_bm, hd_bs, Age, Ms, mu, rho, use_collisions=USE_COLLISIONS)
t_end_ctd = time.time()
print(f'Collision-groomed distribution: {t_end_ctd - t_start_ctd:.2f} s ({(t_end_ctd - t_start_ctd)/60:.2f} min)')

f_sd = _make_interp2d(s_gs, rbins_bm, CS_s)

#Call on RT codes
SED_total,SED_disc, MPerb = DustyMM(
    smin_r, m_final, q, rm, rw, rin, rout, f_sd,
    s_gs, wr, rbins_bm, rho, Ls, Ms, dpc,
    Tg_sr, f_BBgT, Qabs_sw, Qsca_sw, Qpr_s, flux_nu)


plt.clf()
fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (7,4.8))

ax.plot(wr,flux_nu*1000, 'g--',linewidth = 1, label = 'Photosphere', zorder = 1)
ax.plot(wr,SED_disc*1000, 'r-.', label = 'Belt', zorder = 2)
ax.plot(wr,SED_total*1000, 'k', linewidth = 1, label = 'Star+Belt', zorder = 3)

ax.set_xlim([0.3, 3000])
ax.set_ylim([1, 20000])
ax.set_xscale('log')
ax.set_yscale('log')
ax.tick_params(axis='x', labelsize = 12)
ax.tick_params(axis='y', labelsize = 12)
ax.xaxis.set_major_formatter(formatter)
ax.yaxis.set_major_formatter(formatter)
ax.set_xlabel('Wavelength [$\mu$m]')
ax.set_ylabel('Flux Density [mJy]')

ax.legend(loc = 'lower left')
plt.savefig('SED_Star+disc_thermal.pdf')


#-----------------------------------------------------------------#
##Heat maps: grain probability distribution & optical depth##
#-----------------------------------------------------------------#
#Helper: build cell edges for pcolormesh from bin centres
def _edges_log(centres):
    centres = np.asarray(centres, dtype=float)
    logc = np.log10(centres)
    mid = (logc[:-1] + logc[1:]) / 2
    first = logc[0] - (mid[0] - logc[0])
    last = logc[-1] + (logc[-1] - mid[-1])
    return 10**np.concatenate(([first], mid, [last]))

def _edges_lin(centres):
    centres = np.asarray(centres, dtype=float)
    mid = (centres[:-1] + centres[1:]) / 2
    first = centres[0] - (mid[0] - centres[0])
    last = centres[-1] + (centres[-1] - mid[-1])
    return np.concatenate(([first], mid, [last]))

s_edges = _edges_log(s_gs)          #grain-size cell edges [micron]
r_edges = _edges_lin(rbins_bm)      #radial cell edges [au]

#Collapse the time axis of the 3D orbital cube -> (radius, grain size)
prob_map = np.sum(prob_densities_time, axis=2)   #shape (n_r, n_s)

#-- Figure: grain probability distribution heat map --
plt.clf()
fig, ax = plt.subplots(figsize=(7, 5))
prob_pos = prob_map[prob_map > 0]
if prob_pos.size > 0:
    pnorm = LogNorm(vmin=prob_pos.min(), vmax=prob_map.max())
else:
    pnorm = None
cmap = plt.cm.inferno.copy()
cmap.set_bad(cmap(0.0))     #empty/zero cells shown at the lowest colour
prob_masked = np.ma.masked_where(prob_map <= 0, prob_map)
#grain size as a function of radial distance -> radius on x, grain size on y
pc = ax.pcolormesh(r_edges, s_edges, prob_masked.T, cmap=cmap, norm=pnorm, shading='auto', edgecolors='none', linewidth=0, rasterized=True)
ax.set_yscale('log')
ax.set_xlabel('Stellar distance [au]', fontsize=13)
ax.set_ylabel('Grain size [$\\mu$m]', fontsize=13)
ax.set_title('Grain probability distribution', fontsize=12)
cb = fig.colorbar(pc, ax=ax)
cb.set_label('Relative probability density')
fig.tight_layout()
plt.savefig('HeatMap_Probability_'+composition+'.pdf')

#-- Figure: optical depth heat map --
plt.clf()
fig, ax = plt.subplots(figsize=(7, 5))
OD_pos = OD_sr[OD_sr > 0]
if OD_pos.size > 0:
    onorm = LogNorm(vmin=OD_pos.min(), vmax=OD_sr.max())
else:
    onorm = None
cmap2 = plt.cm.viridis.copy()
cmap2.set_bad(cmap2(0.0))    #empty/zero cells shown at the lowest colour
OD_masked = np.ma.masked_where(OD_sr <= 0, OD_sr)
#grain size as a function of radial distance -> radius on x, grain size on y
pc = ax.pcolormesh(r_edges, s_edges, OD_masked.T, cmap=cmap2, norm=onorm, shading='auto', edgecolors='none', linewidth=0, rasterized=True)
ax.set_yscale('log')
ax.set_xlabel('Stellar distance [au]', fontsize=13)
ax.set_ylabel('Grain size [$\\mu$m]', fontsize=13)
ax.set_title('Optical depth', fontsize=12)
cb = fig.colorbar(pc, ax=ax)
cb.set_label(r'Optical depth $\tau(s, r)$')
fig.tight_layout()
plt.savefig('HeatMap_OpticalDepth_'+composition+'.pdf')

print('Saved: SED_Star+disc_thermal.pdf')
print('Saved: HeatMap_Probability_'+composition+'.pdf')
print('Saved: HeatMap_OpticalDepth_'+composition+'.pdf')
