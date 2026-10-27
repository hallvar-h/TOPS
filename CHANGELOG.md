# Changelog

## [0.4.0](https://github.com/hallvar-h/TOPS/compare/v0.3.0...v0.4.0) (2026-10-19)


### ⚠ BREAKING CHANGES

* int_par is removed; former internal parameters are now inputs set via set_input. PowerSystemModel raises on invalid model input. Requires Python >= 3.10.

### Features

* Access dynamic models by name, e.g. ps.gen.GEN ([36b7e72](https://github.com/hallvar-h/TOPS/commit/36b7e72597eb562c1a4ce6a9c33a984754b5b0df))
* Add event system with events defined in the model data ([36b7e72](https://github.com/hallvar-h/TOPS/commit/36b7e72597eb562c1a4ce6a9c33a984754b5b0df))
* Add Simulator class and constant-power loads ([36b7e72](https://github.com/hallvar-h/TOPS/commit/36b7e72597eb562c1a4ce6a9c33a984754b5b0df))
* Allow model data tables to be given as pandas DataFrames ([36b7e72](https://github.com/hallvar-h/TOPS/commit/36b7e72597eb562c1a4ce6a9c33a984754b5b0df))
* Initialize models one unit at a time ([36b7e72](https://github.com/hallvar-h/TOPS/commit/36b7e72597eb562c1a4ce6a9c33a984754b5b0df))
* Refactor model setup, add event system and simulator ([36b7e72](https://github.com/hallvar-h/TOPS/commit/36b7e72597eb562c1a4ce6a9c33a984754b5b0df))


### Bug Fixes

* Line can be disconnected more than once ([36b7e72](https://github.com/hallvar-h/TOPS/commit/36b7e72597eb562c1a4ce6a9c33a984754b5b0df))
* Solver and line current bugs ([36b7e72](https://github.com/hallvar-h/TOPS/commit/36b7e72597eb562c1a4ce6a9c33a984754b5b0df))
* state_idx_global shifted on repeated init_dyn_sim ([36b7e72](https://github.com/hallvar-h/TOPS/commit/36b7e72597eb562c1a4ce6a9c33a984754b5b0df))

## [0.3.0](https://github.com/hallvar-h/TOPS/compare/v0.2.0...v0.3.0) (2026-09-11)


### Features

* PR Title Validation ([4efbfcb](https://github.com/hallvar-h/TOPS/commit/4efbfcb3870b48f7ded0f49ed3dae2f488694a79))


### Bug Fixes

* update Line.I_to to use i_to instead of i_from ([a08bca3](https://github.com/hallvar-h/TOPS/commit/a08bca336903a6cfa62b62c8cca2be6ba33d1668))
* update Line.I_to to use i_to instead of i_from ([2a5429a](https://github.com/hallvar-h/TOPS/commit/2a5429a75ce9fe0f29fcda318f87638ef1faaa4d))
