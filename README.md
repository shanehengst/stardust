# Stardust

A Python code for modelling the thermal emission and spectral energy distribution (SED) of debris discs around main-sequence stars. It includes grain dynamics (PR drag, radiation pressure, Poynting–Robertson drag), size distribution evolution, and Mie theory calculations for dust opacity and scattering.

## Overview

`Stardust_base_code.py` demonstrates the full modelling workflow:

1. **Stellar setup** — blackbody grid, radiation-pressure beta values via Mie theory
2. **Orbital dynamics** — time-evolved grain radial distributions under PR drag and radiation pressure
3. **Disc structure** — size-weighted surface density from collisional/dynamical evolution
4. **Thermal emission** — grain temperature grids, Mie absorption efficiencies, synthetic SED

The `stardust/` package contains all the underlying functions as importable modules.

## Package Structure

```
stardust/
├── __init__.py      # Re-exports all public symbols
├── constants.py     # Physical constants (G, c, h, kb, au, …)
├── blackbody.py     # Planck functions (Blam, Planck_nu)
├── utils.py         # RegularGridInterpolator wrapper (_make_interp2d)
├── orbital.py       # Keplerian orbit integration, PR drag time evolution
├── disc.py          # Disc surface density from size-distribution evolution
└── radiative.py     # Mie grids (radpressure), synthetic SED (DustyMM)
```

## Requirements

- Python ≥ 3.10
- numpy
- scipy ≥ 1.11
- matplotlib
- miepython ≥ 3.0
- astropy
- pandas

Install dependencies:

```bash
pip install -r requirements.txt
```

## Input data files

The following files must be present in the working directory before running:

| File | Description |
|---|---|
| `BTGen-5770.txt` | BT-Settl stellar photosphere spectrum |
| `BTNextGen_SolarPhotosphere.txt` | BT-NextGen solar photosphere spectrum |
| `silicate_d03.lnk` | Silicate optical constants (Draine 2003) |

Cached Mie / temperature grid CSV files (`df_Tg_*.csv`) are generated automatically on first run and reused thereafter. They are excluded from version control.

## Running

```bash
python Stardust_base_code.py
```

On first run the Mie grid computation may take several minutes. Subsequent runs load the cached CSV files and complete much faster.

## Citation

If you use this code in your research please cite the relevant papers describing the underlying physical model.

## License

MIT
