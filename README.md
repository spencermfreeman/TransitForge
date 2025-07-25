# TransitForge (STEPUP Image Analysis Pipeline - V3)

<p align="center">
  <img src="static/Pitt_Astrophysics_Logo.png" width="110" />
  <img src="static/Pitt_Logo.png" height="120" />
  <img src="static/STEPUP_logo_pure.png" width="110" />
</p>


## Pipeline Overview
#### An image pipeline for the calibration and processing of FITS images, equipped with differential photometric techniques. Dependencies include Numpy/Matplotlib, Astropy/Photutils, and Pandas (details in requirements.txt).

- Primary Goal: Production of exoplanet transit light curves.
- Development: Ongoing work on a graphical user interface (GUI) and existing features, decrease overhead computation.
- Target Data: The pipeline aims to support analysis of existing TESS data with libraries such as Astropy, the Photutils sub-package, and personal/group observational data.
- Functionality: Designed to work similarly to AstroImageJ, but in a more condensed and streamlined manner, tailored to STEPUP operations.
  
## GUI Preview

<p align="center">
  <img src="./static/file_io.png" width="700" />
</p>
<p align="center">
  <img src="./static/photometry.png" width="700" />
</p>
<p align="center">
  <img src="./static/zoom_window.png" width="500" />
</p>
<p align="center">
  <img src="./static/plotting.png" width="700" />
</p>

## Data and Output
#### Median of flat and bias frames are computed and mapped to a color plot as shown below:

<p align="center">
  <img src="./static/master_frames.png" width="700" />
</p>

#### Comparison of uncalibrated and calibrated images (AstroImageJ window):

<p align="center">
  <img src="./static/comparison.png" width="700" />
</p>

#### Current Adjustments: These remove some background noise, apply lens/CCD corrections, reduce vignetting, and mitigate haze.

<p align="center">
  <img src="./static/Qatar-5.png" width="700" />
</p>
