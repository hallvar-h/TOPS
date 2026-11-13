# Changelog

## [0.8.0](https://github.com/hallvar-h/TOPS/compare/v0.7.0...v0.8.0) (2026-11-13)


### Features

* add governor set_reference event for TGOV1, HYGOV and IEESGO ([d52dd9e](https://github.com/hallvar-h/TOPS/commit/d52dd9e084c2bb1a481f7f63f98fe80ae030f3d4))
* add governor set_reference event for TGOV1, HYGOV and IEESGO ([b01e9db](https://github.com/hallvar-h/TOPS/commit/b01e9dbc1984e44b23badd6f96a81249e0c45bc0))

## [0.7.0](https://github.com/hallvar-h/TOPS/compare/v0.6.1...v0.7.0) (2026-11-12)


### Features

* add ResultKeeper.calculate to evaluate outputs after the simulation ([09abe7f](https://github.com/hallvar-h/TOPS/commit/09abe7f05fac98a42f43e2dcb0c233f3373e78fe))
* allow events to be given as event functions ([0b29592](https://github.com/hallvar-h/TOPS/commit/0b2959219361209fe003a864926c2604cb9adee0))
* give events as event functions, and declare outputs in output_list ([ad86d2c](https://github.com/hallvar-h/TOPS/commit/ad86d2c10a39c24c8bcc04c3a4a3ae20fd6e9bf9))
* result keeper calculate ([9abed2f](https://github.com/hallvar-h/TOPS/commit/9abed2f01db909d228a408bf0da25359bd6acd45))


### Bug Fixes

* initialize loads of zero power to zero admittance ([333ff8f](https://github.com/hallvar-h/TOPS/commit/333ff8fabeebdd0774693741ad05d9d68805e570))
* initialize loads of zero power to zero admittance ([3e3d17c](https://github.com/hallvar-h/TOPS/commit/3e3d17c0945e706a27afac2f904c7894af07b3f5))
* replay event functions in ResultKeeper.calculate ([f7bde9b](https://github.com/hallvar-h/TOPS/commit/f7bde9b428e51640009397045df1f737b146f497))
* respect HYGOV g_min/g_max parameters ([7b02a46](https://github.com/hallvar-h/TOPS/commit/7b02a4640cdb6fd0f7e48f80c5ca0ed70ab9b4fd))
* respect HYGOV g_min/g_max parameters ([f02020a](https://github.com/hallvar-h/TOPS/commit/f02020a8a428097162a2d8d30ac8207d9ba04780))


### Documentation

* add example of ResultKeeper.calculate with events replayed ([461431f](https://github.com/hallvar-h/TOPS/commit/461431fd6c387f29fb72e93679a3057155d552a1))

## [0.6.1](https://github.com/hallvar-h/TOPS/compare/v0.6.0...v0.6.1) (2026-11-05)


### Bug Fixes

* choose the generator with the largest output as slack bus and sl… ([ca276a3](https://github.com/hallvar-h/TOPS/commit/ca276a3b423b6da428b90ba5f1654032604a0a03))
* choose the generator with the largest output as slack bus and slack unit ([195241a](https://github.com/hallvar-h/TOPS/commit/195241ab522ca7054027028a80d4ab410e357653))
* make v0 optional and keyword-only in EulerDAE ([3a73bda](https://github.com/hallvar-h/TOPS/commit/3a73bdaca4f5ecdf2e088cd5f16ce892caa3243b))
* make v0 optional and keyword-only in EulerDAE ([a1ff4f0](https://github.com/hallvar-h/TOPS/commit/a1ff4f07da17a44148325e98136e61a17b2f5f3d))

## [0.6.0](https://github.com/hallvar-h/TOPS/compare/v0.5.0...v0.6.0) (2026-11-02)


### Features

* add Switch model with open/close events ([4b38bd2](https://github.com/hallvar-h/TOPS/commit/4b38bd2c8b4215bf785d3b84c74e5ec547ab9a33))
* add Switch model with open/close events ([f368c8b](https://github.com/hallvar-h/TOPS/commit/f368c8bede50a3171304bbd0742261980f1d135b))

## [0.5.0](https://github.com/hallvar-h/TOPS/compare/v0.4.0...v0.5.0) (2026-11-02)


### ⚠ BREAKING CHANGES

* Explicit blocks, declared inputs/outputs and largest-P slack bus ([#23](https://github.com/hallvar-h/TOPS/issues/23))

### Features

* **dynamic:** find buses without admittance ([11af429](https://github.com/hallvar-h/TOPS/commit/11af429dfd63d82292e31836837550f4b68ab960))
* **dynamic:** pin dead buses in the ConstPowerLoad solve ([509d792](https://github.com/hallvar-h/TOPS/commit/509d792a5cd40a6fa494f848a17f176ab6c5cb2d))
* **dynamic:** pin dead buses to 0 V in the linear solve ([ec208b3](https://github.com/hallvar-h/TOPS/commit/ec208b3cc1e029a9a9403f690a5f0a6aa4a9ac62))
* **dynamic:** refuse a load flow that connected=False cuts off from the slack ([2cb9a18](https://github.com/hallvar-h/TOPS/commit/2cb9a182a395f119cc92be1f7873cc239959bbb7))
* **lines:** optional connected column ([cda7d2d](https://github.com/hallvar-h/TOPS/commit/cda7d2d034b3e2a93341ec249bb041ad64a8b6d4))
* **loads,dynamic:** set_power load events and dead-bus handling ([b4c8b2a](https://github.com/hallvar-h/TOPS/commit/b4c8b2a9ed68b43a44a6b65f0152be7c51daa8ce))
* **loads:** set_power event on ConstPowerLoad ([845c081](https://github.com/hallvar-h/TOPS/commit/845c0813e0f6513683c7810d61abdc00503ec262))
* **loads:** set_power event on Load ([ed07ab3](https://github.com/hallvar-h/TOPS/commit/ed07ab32ae1f6a0b5d557d1b11adbc9ed7764fd8))
* **realtime:** headless RealTimeSimulator ([f0c1945](https://github.com/hallvar-h/TOPS/commit/f0c1945cea5ff287e78b19bd6cdc2e7cf816de08))
* **realtime:** on_frame hook and divergence guard ([3f91fa7](https://github.com/hallvar-h/TOPS/commit/3f91fa720b2edc34a4b1d9b83552da60abb8e1bc))
* **realtime:** pause, resume, set_speed, stop and a thread runner ([85a1016](https://github.com/hallvar-h/TOPS/commit/85a1016aedae669d86739fb84869112224829217))
* **realtime:** RealTimeSimulator with an absolute schedule and max_lag reset ([e6dea76](https://github.com/hallvar-h/TOPS/commit/e6dea769d8813b2546c73d5c5c918ac8bd24d004))
* **realtime:** thread-safe submit queue drained before each frame ([4abd7fe](https://github.com/hallvar-h/TOPS/commit/4abd7fe77ce4ace243da395cf91ca74034c1c133))
* **simulator:** fs parameter and timer on InterfacerDirect ([e762883](https://github.com/hallvar-h/TOPS/commit/e76288343e5f08922e41a5b3ce4f692385753e2d))
* **simulator:** optional kwargs column on EventSchedule ([b631c76](https://github.com/hallvar-h/TOPS/commit/b631c76c9287e7827a54bdff02c27d1276f1a098))
* **trafos:** optional connected column ([ed6d3a2](https://github.com/hallvar-h/TOPS/commit/ed6d3a2168fad25bdd4f5487d0d437e26128275a))
* **trafos:** TrafoEvents connect and disconnect ([35a1655](https://github.com/hallvar-h/TOPS/commit/35a16551f2588060a5255aef5d38c6e7e9bfed02))
* **trafos:** transformer events and an optional connected column ([#26](https://github.com/hallvar-h/TOPS/issues/26)) ([0a8eb3f](https://github.com/hallvar-h/TOPS/commit/0a8eb3f0d2651362143ed6eba4d7b5dae7b1624e))


### Bug Fixes

* **pll:** hold the PLL1 angle on a bus without voltage ([3f0ae80](https://github.com/hallvar-h/TOPS/commit/3f0ae80a3517111cd2ae898b2a3b89f2a8ffb6b4))
* **simulator:** InterfacerDirect sets its name before connect ([b2f9371](https://github.com/hallvar-h/TOPS/commit/b2f9371d04ee8106f01460131c1714a96ccb08d5))
* **trafos:** give DynTrafo integer indices for its empty constant admittance ([e1d6442](https://github.com/hallvar-h/TOPS/commit/e1d64423cee57f95c22dfce3a1f435c5f7d0fa5d))


### Documentation

* add About page, complete home page and cross-link pages ([50932c7](https://github.com/hallvar-h/TOPS/commit/50932c7776ad543b0c758b103a7bf59ec1909fc8))
* add AGENTS.md with agent guidelines and minimal ruff config ([50932c7](https://github.com/hallvar-h/TOPS/commit/50932c7776ad543b0c758b103a7bf59ec1909fc8))
* add documentation site published to GitHub Pages ([#25](https://github.com/hallvar-h/TOPS/issues/25)) ([50932c7](https://github.com/hallvar-h/TOPS/commit/50932c7776ad543b0c758b103a7bf59ec1909fc8))
* add Included test systems page ([50932c7](https://github.com/hallvar-h/TOPS/commit/50932c7776ad543b0c758b103a7bf59ec1909fc8))
* add Install page ([50932c7](https://github.com/hallvar-h/TOPS/commit/50932c7776ad543b0c758b103a7bf59ec1909fc8))
* add mkdocs.yml skeleton and placeholder home page ([50932c7](https://github.com/hallvar-h/TOPS/commit/50932c7776ad543b0c758b103a7bf59ec1909fc8))
* add plot snippet helper for executed code blocks ([50932c7](https://github.com/hallvar-h/TOPS/commit/50932c7776ad543b0c758b103a7bf59ec1909fc8))
* add Quickstart with executed short-circuit simulation ([50932c7](https://github.com/hallvar-h/TOPS/commit/50932c7776ad543b0c758b103a7bf59ec1909fc8))
* add SINTEF colour theme ([50932c7](https://github.com/hallvar-h/TOPS/commit/50932c7776ad543b0c758b103a7bf59ec1909fc8))
* **api:** add API reference skeleton (mkdocstrings) ([50932c7](https://github.com/hallvar-h/TOPS/commit/50932c7776ad543b0c758b103a7bf59ec1909fc8))
* credit Hallvar Haugdal as main author, NTNU origin and SINTEF ([50932c7](https://github.com/hallvar-h/TOPS/commit/50932c7776ad543b0c758b103a7bf59ec1909fc8))
* **examples:** line trip, transformer trip and load change ([4c6b33e](https://github.com/hallvar-h/TOPS/commit/4c6b33e4780c7dd39fce94da5912ebbdf0c099ae))
* **guide:** add Dynamic models catalogue ([50932c7](https://github.com/hallvar-h/TOPS/commit/50932c7776ad543b0c758b103a7bf59ec1909fc8))
* **guide:** add Events page ([50932c7](https://github.com/hallvar-h/TOPS/commit/50932c7776ad543b0c758b103a7bf59ec1909fc8))
* **guide:** add Examples page ([50932c7](https://github.com/hallvar-h/TOPS/commit/50932c7776ad543b0c758b103a7bf59ec1909fc8))
* **guide:** add Modal analysis page ([50932c7](https://github.com/hallvar-h/TOPS/commit/50932c7776ad543b0c758b103a7bf59ec1909fc8))
* **guide:** add Model format page ([50932c7](https://github.com/hallvar-h/TOPS/commit/50932c7776ad543b0c758b103a7bf59ec1909fc8))
* **guide:** add Running a simulation page ([50932c7](https://github.com/hallvar-h/TOPS/commit/50932c7776ad543b0c758b103a7bf59ec1909fc8))
* **guide:** add User-defined models page ([50932c7](https://github.com/hallvar-h/TOPS/commit/50932c7776ad543b0c758b103a7bf59ec1909fc8))
* **guide:** connected column and transformer connect events ([8e16fbe](https://github.com/hallvar-h/TOPS/commit/8e16fbe84372fc540db3551bb6c4e158bff45f04))
* **guide:** kwargs column, set_power and dead buses ([c4fb9da](https://github.com/hallvar-h/TOPS/commit/c4fb9daab5fbbbe1079ac8a012d512b2076752d1))
* **guide:** real-time simulation guide, API page and events example ([d12de1f](https://github.com/hallvar-h/TOPS/commit/d12de1fdfc87960d017fca06f72670e58d798a1a))
* link documentation site from README ([50932c7](https://github.com/hallvar-h/TOPS/commit/50932c7776ad543b0c758b103a7bf59ec1909fc8))


### Code Refactoring

* Explicit blocks, declared inputs/outputs and largest-P slack bus ([#23](https://github.com/hallvar-h/TOPS/issues/23)) ([4b2bc63](https://github.com/hallvar-h/TOPS/commit/4b2bc63fd93721a2135bf7dbba6df8fcf1890638))

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
