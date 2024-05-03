This is a copy of TOPS made by Hallvar Haugdal: https://github.com/hallvar-h/TOPS/

This is a version made in my master's thesis, adapted to easily study the Nordic 45 (N45) model - a dynamic model for the Nordic synchronous area.
This version has laid the groundwork for further research on N45, with a script that easily adapts for different time scenarios.

While doing this thesis, the HYGOV implimentations was suspected to be faulty. Therefore, a simplified HYGOV implementation is added to the core code.

The complimentary script with ENTSO-E Transparency platform can be [found here](https://github.com/eirissa/ENTSO-E-Data-for-TOPS).

# Original readme:

# TOPS (**T**iny **O**pen **P**ower System **S**imulator)
**Note**: This repository was previously called DynPSSimPy.


This is a package for performing dynamic power system simulations in Python. The aim is to provide a simple and lightweight tool which is easy to install, run and modify, to be used by researchers and in education. Performance is not the main priority. The only dependencies are numpy, scipy, pandas and matplotlib (the core functionality only uses numpy and scipy).

The package is being developed as part of ongoing research, and thus contains experimental features. Use at your own risk!

Some features:
- Newton-Rhapson power flow
- Dynamic time domain simulation (RMS/phasor approximation)
- Linearization, eigenvalue analysis/modal analysis

# Installation
The package can be installed using pip, as follows:

`pip install tops`

# Citing
If you use this code for your research, please cite [this paper](https://arxiv.org/abs/2101.02937).

# Example notebooks
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/hallvar-h/TOPS/HEAD?filepath=examples%2Fnotebooks)

# Contact
[Hallvar Haugdal](mailto:hallvhau@gmail.com)
