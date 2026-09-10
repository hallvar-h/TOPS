# TOPS (**T**iny **O**pen **P**ower System **S**imulator)

[![Docs](https://github.com/hallvar-h/TOPS/actions/workflows/docs.yml/badge.svg)](https://github.com/hallvar-h/TOPS/actions/workflows/docs.yml)
[![Tests](https://github.com/hallvar-h/TOPS/actions/workflows/python-app.yml/badge.svg?branch=main)](https://github.com/hallvar-h/TOPS/actions/workflows/python-app.yml)

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

# Contributing and releases
Pull request titles must follow the [Conventional Commits](https://www.conventionalcommits.org/) format. Use `fix: ...` for bug fixes, `feat: ...` for features, and a `!` such as `feat!: ...` for breaking changes. Pull requests should be squash-merged using the pull request title as the commit message.

[Release Please](https://github.com/googleapis/release-please) maintains a release pull request on `main`. That pull request updates `CHANGELOG.md` and the package version in `pyproject.toml`. Because Release Please uses the repository `GITHUB_TOKEN`, run the **Tests** workflow manually against the generated release branch before merging it. Merging the release pull request creates a version tag and GitHub release, builds and validates the distributions, and starts the protected `pypi` environment deployment.

# Citing
If you use this code for your research, please cite [this paper](https://ieeexplore.ieee.org/document/9494770).

# Example notebooks
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/hallvar-h/TOPS/HEAD?filepath=examples%2Fnotebooks)

# Contact
[Hallvar Haugdal](mailto:hallvhau@gmail.com)
