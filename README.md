# DynPSSimPy
This is a package for performing dynamic power system simulations in Python. The aim is to provide a simple and lightweight tool which is easy to install, run and modify, to be used by researchers and in education. Performance is not the main priority. The only dependencies are numpy, scipy, pandas and matplotlib (the core functionality only uses numpy and scipy).

The package is being developed as part of ongoing research, and thus contains experimental features. Use at your own risk!

Some features:
- Newton-Rhapson power flow
- Dynamic time domain simulation (RMS/phasor approximation)
- Linearization, eigenvalue analysis/modal analysis

# Installation
The package can be installed using pip, as follows:

`pip install git+https://github.com/hallvar-h/dynpssimpy`

# Citing
If you use this code for your research, please cite [this paper](https://arxiv.org/abs/2101.02937).

# Example notebooks
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/hallvar-h/DynPSSimPy/HEAD?filepath=examples%2Fnotebooks)
