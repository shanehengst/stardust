# Stardust

A Python code for modelling the thermal emission and spectral energy distribution (SED) of debris discs around main-sequence stars. It includes grain dynamics (radiation pressure + Poynting-Robertson drag), size distribution evolution, and Mie theory calculations for dust opacity and scattering.

## Overview

`Stardust_base_code.py` demonstrates the full modelling workflow:

1. **Stellar setup** - blackbody grid, radiation-pressure beta values via Mie theory
2. **Orbital dynamics** - time-evolved grain radial distributions under PR drag and radiation pressure
3. **Disc structure** - size-weighted surface density from collisional/dynamical evolution
4. **Thermal emission** - grain temperature grids, Mie absorption efficiencies, synthetic SED

The `stardust/` package contains all the underlying functions as importable modules.

## Default model configuration

Out of the box, `Stardust_base_code.py` is configured for a **Sun-like (G2V) host star** with a fiducial debris belt. These values are set near the top of the driver script and can be edited freely.

### Host star (default: the Sun, in Solar units)

| Property | Symbol | Value |
|---|---|---|
| Luminosity | `Ls` | 1 Lsun |
| Mass | `Ms` | 1 Msun |
| Radius | `Rs` | 1 Rsun |
| Effective temperature | `Ts` | 5772 K |
| System age | `Age` | 100 Myr |
| Distance | `dpc` | 10 pc |

The stellar spectrum defaults to a blackbody at `Ts`, optionally replaced/calibrated by a BT-NextGen (AGSS2009) solar-type photosphere model (`BTGen-5770.txt`).

### Fiducial debris belt

| Property | Symbol | Value |
|---|---|---|
| Inner radius | `rin` | 50 au |
| Outer radius | `rout` | 150 au |
| Gaussian mean radius | `rm` | 100 au |
| Fractional width | `dfrac` (dr/r) | 0.1 |
| Radial grid bin size | `hd_bs` | 1 au |
| Maximum modelled distance | `hd_max` | 500 au |
| Mass released per timestep | `m_im` | 1e-3 Mearth |
| Number of timesteps | `timesteps` | 100 |

The surface density follows a Gaussian ring centred on `rm` with width `rw = rm*dfrac/2.35482`.

### Grain size distribution & composition

| Property | Symbol | Value |
|---|---|---|
| Minimum grain size | `smin_r` | 0.01 um |
| Maximum grain size | `smax_r` | 3000 um |
| Number of grain sizes | `n_gs` | 100 (log-spaced) |
| Size-distribution exponent | `q` | 3.5 (steady-state collisional cascade, dn/ds ~ s^-q) |
| Grain density | `rho` | 3.3 g/cm^3 |
| Composition | `composition` | `silicate_d03` (astronomical silicate, Draine 2003) |

Collisional grooming of the size distribution can be toggled with `USE_COLLISIONS` (default `True`).

## Package Structure

```
stardust/
├── __init__.py      # Re-exports all public symbols
├── constants.py     # Physical constants (G, c, h, kb, au, ...)
├── blackbody.py     # Planck functions (Blam, Planck_nu)
├── utils.py         # RegularGridInterpolator wrapper (_make_interp2d)
├── orbital.py       # Keplerian orbit integration, PR drag time evolution
├── disc.py          # Disc surface density from size-distribution evolution
└── radiative.py     # Mie grids (radpressure), synthetic SED (DustyMM)
```

## Requirements

- Python >= 3.10
- numpy
- scipy >= 1.11
- matplotlib
- miepython >= 3.0
- astropy
- pandas

Install dependencies:

```bash
pip install -r requirements.txt
```

## Input data files

The following files must be present in the working directory before running:

| File | Description | Source |
|---|---|---|
| `BTGen-5770.txt` | BT-NextGen (AGSS2009) solar-type (Teff = 5770 K) photosphere spectrum | [SVO Theoretical Spectra][svo] |
| `silicate_d03.lnk` | Astronomical silicate optical constants (Draine 2003) | [optool `lnk_data`][optool] |

[svo]: http://svo2.cab.inta-csic.es/theory/newov2/index.php
[optool]: https://github.com/cdominik/optool/tree/master/lnk_data

Cached Mie / temperature grid CSV files (`df_Tg_*.csv`) are generated automatically on first run and reused thereafter. They are excluded from version control.

### Data references

- **Stellar photosphere:** BT-NextGen model atmospheres using the AGSS2009 solar
  abundances (Allard, Homeier & Freytag 2012; Asplund et al. 2009). Models were
  obtained from the [SVO Theoretical Spectra web service][svo]
  (http://svo2.cab.inta-csic.es/theory/newov2/index.php).
- **Grain optical constants:** the `.lnk` (n, k) files, including
  `silicate_d03.lnk` (astronomical silicate; Draine 2003), are distributed with
  the [optool package][optool]
  (https://github.com/cdominik/optool/tree/master/lnk_data); see the references
  therein for the original laboratory/theoretical sources of each material.

## Running

```bash
python Stardust_base_code.py
```

On first run the Mie grid computation may take several minutes. Subsequent runs load the cached CSV files and complete much faster.

### Outputs

| File | Description |
|---|---|
| `SED_Star+disc_thermal.pdf` | Synthetic SED (photosphere, belt, star+belt) |
| `HeatMap_Probability_silicate_d03.pdf` | Grain size vs. stellar distance probability density |
| `HeatMap_OpticalDepth_silicate_d03.pdf` | Grain size vs. stellar distance geometric optical depth |

## Citation

If you use this code in your research please cite (to be submitted):

```bibtex
@ARTICLE{Hengst_stardust,
       author = {{Hengst}, Shane and {Pearce}, Tim and {Marshall}, Jonty and {Sommer}, Max and {Horner}, Jonti and {Marsden}, Stephen},
        title = "{Stardust: a debris disc thermal emission and radiative-transfer modelling code}",
      journal = {to be submitted},
         year = 2026,
}
```

The methodology underlying this code is described in the appendix of Marshall, Hengst et al. (2025):

```bibtex
@ARTICLE{2025MNRAS.541...71M,
       author = {{Marshall}, J.~P. and {Hengst}, S. and {Trejo-Cruz}, A. and {del Burgo}, C. and {Milli}, J. and {Booth}, M. and {Augereau}, J.~C. and {Choquet}, E. and {Morales}, F.~Y. and {Th{\'e}bault}, P. and {Kemper}, F. and {Faramaz-Gorka}, V. and {Bryden}, G.},
        title = "{ALMA millimetre-wavelength imaging of HD 138965: new constraints on the debris dust composition and presence of planetary companions}",
      journal = {\mnras},
     keywords = {planet-disc interactions, circumstellar matter, stars: individual: HD 138965, radio continuum: planetary systems, Earth and Planetary Astrophysics},
         year = 2025,
        month = jul,
       volume = {541},
       number = {1},
        pages = {71-84},
          doi = {10.1093/mnras/staf984},
archivePrefix = {arXiv},
       eprint = {2506.11726},
 primaryClass = {astro-ph.EP},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2025MNRAS.541...71M},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

## Contact

Questions, feedback, and collaboration are warmly welcomed. If you have any
questions about the code, run into issues, or would like to discuss its use or
potential extensions, please feel free to get in touch with **Shane Hengst** —
it would be appreciated to hear from you.

## License

**Attribution (citation-required) license.**

You are free to use, modify, and distribute this code for any purpose. The only
condition is:

- **Cite the author.** Any published or presented work (papers, posters, talks,
  theses, reports, or derived data) that makes use of the code must cite it and
  acknowledge the author (see the Citation section above).

See the [LICENSE](LICENSE) file for the full terms.
