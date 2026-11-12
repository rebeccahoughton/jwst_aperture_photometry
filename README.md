# PHANGS-JWST Aperture Photometry
 
A Python pipeline for performing aperture photometry on HST, JWST NIRCam, and JWST MIRI imaging data from the [PHANGS-JWST survey](https://sites.google.com/view/phangs/home). The pipeline handles the full workflow from loading mosaics to producing calibrated photometric catalogs, including background subtraction, source detection, optimal aperture selection, and aperture corrections.

**IMPORTANT: THIS MODULE IS STILL IN DEVELOPMENT. Please expect bugs. It is being updated regularly and currently does not have the QA infrastructure to know that the output catalogs are reliable.** 
 
---
 
## Overview
 
This module takes JWST stage 3 mosaic products (typically anchored mosaics in FITS format) and produces source catalogs with aperture photometry in AB magnitudes. The pipeline is controlled via TOML configuration files, making it straightforward to apply consistently across many galaxies and filters.
 
**Key capabilities:**
 
- 2D background estimation and subtraction using `photutils.Background2D`
- Source detection with `IRAFStarFinder`, `DAOStarFinder` [*in development*], or `find_peaks`
- Optimal aperture radius selection via curve-of-growth analysis [*in development*]
- Circular aperture photometry with local sky annulus background subtraction
- Aperture corrections using CRDS calibration files or the empirical factors from Rodriguez et al. (2025)
- Flux conversion from Jy/sr (or MJy/sr) to AB magnitudes
- Catalog output in FITS or CSV format
- Quality-assurance (QA) plots [*in development*]
- Residual image generation with stars subtracted [*in development*]

## Repository Structure
 
```
jwst_aperture_photometry/          
├── pyproject.toml
├── README.md
├── LICENSE
├── src/
│   └── rphot/
│       ├── __init__.py        # API + __version__
│       ├── io.py              # finding, opening, and reading data files
│       ├── units.py           # flux to mag, image units to flux units, etc.
│       ├── background.py      # 2D background estimation + subtraction
│       ├── detect.py          # source finder (currently IRAF and peaks supported)
│       ├── catalog.py         # catalog matching
│       ├── apcorr.py          # aperture correction
│       ├── photometry.py      # core photometry computation
│       ├── residuals.py       # residual image creation
│       └── main.py            # run the full pipeline
├── phangs_phot.py             # legacy code from before modularization
├── tests/
│   ├── tbc.py                 # tests
│   ├── ....
│   └── ....
├── config/
│   ├── config.toml            # photometry parameters (targets, filters, pipeline steps)
│   └── local.toml             # Local paths (data directory, output directory, CRDS directory)
├── scripts/                   # helpful scripts for development testing
├── catalogs/                  # *DO NOT USE* - left from initial testing
└── examples/                  # notebooks
```

---
 
## Installation
 
### Prerequisites
 
Python 3.9+ is recommended. Install the required packages with pip:
 
```bash
pip install numpy scipy matplotlib astropy photutils astroquery
```
 
Or with conda:
 
```bash
conda install numpy scipy matplotlib astropy photutils
conda install -c conda-forge astroquery
```


### Clone the repository
 
```bash
git clone https://github.com/rebeccahoughton/jwst_aperture_photometry.git
cd jwst_aperture_photometry
```

---
 
## Configuration
 
All pipeline settings are controlled by two TOML files in the `config/` directory. You can create new config and local files for your science goal and track them in your .gitignore.
 
### `config/config.toml` — photometry parameters
 
This file specifies which galaxies and filters to process, which pipeline steps to run, and the parameters for each step. 

### `config/local.toml` — local paths
 
```toml
jwst_dir  = "/path/to/jwst/data/"
out_dir   = "/path/to/output/catalogs/"
crds_dir  = "/path/to/crds/apcorr/files/"
```

---
 
## Usage
 
### Running the full pipeline
 
Edit the two config files to point at your data and set your parameters, then runfrom the terminal in jwst_aperture_photometry/src as:
 
```bash
python -m rphot.main
```
 
This loops over all galaxies and filters defined in `config.toml`, running whichever steps are listed in `steps`.


## Aperture Corrections
 
Three methods are supported, set via `apcorr_method` in the config:
 
- **`crds`** — Uses the official JWST Calibration Reference Data System (CRDS) `apcorr` files for NIRCam. Derives the aperture radius, sky annulus radii, and correction factor for a specified encircled energy fraction (default: 80%).
- **`cluster`** — Uses empirical correction factors for star clusters from Rodriguez et al. (2025), based on Deger et al. (2022). Valid for a fixed aperture radius of 4 pixels. Corrections are provided in Vega magnitudes and converted to AB internally via SVO filter zero points.
- **`psf`** - Calculates the aperture correction from the image PSF
- **Note that aperture corrections for custom radii are not currently available, as the curve of growth radii still needs some improvement**.


---
 
## Output Catalogs
 
Catalogs are written to `out_dir` as `{galaxy}_jwst_{band}_cat.fits` (or other filetype, such as `.csv`, specified in `config`). Key columns will depend on the source finder method used. 

---

## Dependencies
 
| Package | Purpose |
|---|---|
| `numpy` | Array operations |
| `scipy` | Spatial matching (KD-tree) |
| `matplotlib` | Plotting and QA figures |
| `astropy` | FITS I/O, WCS, tables, units |
| `photutils` | Background estimation, source detection, aperture photometry |
| `astroquery` | SVO filter zero points for aperture corrections |
 
---
 