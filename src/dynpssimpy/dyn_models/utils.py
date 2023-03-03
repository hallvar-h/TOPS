import numpy as np
import inspect
import functools
import dynpssimpy.utility_functions as dps_uf
from scipy.optimize import least_squares


def determine_connections(dyn_mdls, order_by='input'):
    mdl_connections = {}

    def add_entry(container_dict, key_1, key_2, entry):
        if not key_1 in container_dict.keys():
            container_dict[key_1] = {}
        if not key_2 in container_dict[key_1].keys():
            container_dict[key_1][key_2] = []
        container_dict[key_1][key_2].append(entry)

    for container_key, mdls in dyn_mdls.items():
        for mdl_key, mdl in mdls.items():
            if hasattr(mdl, 'connections'):
                connections = mdl.connections().copy()
                for conn in connections:
                    if 'source' in conn.keys():
                        source_container_key = conn['source']['container']
                        dest_mdl = mdl
                        dest_mdl_key = mdl_key
                        dest_container_key = container_key

                        source_container = dyn_mdls[conn['source']['container']]
                        for source_mdl_key, source_mdl in source_container.items():
                            if conn['source']['mdl'] == '*' or conn['source']['mdl'] == source_mdl_key:
                                source_idx, mask = dps_uf.lookup_strings(
                                    conn['source']['id'], source_mdl.par['name'], return_mask=True
                                )
                                dest_idx = np.where(mask)[0]
                                if len(source_idx) > 0:
                                    if order_by == 'input':
                                        add_entry(
                                            mdl_connections, dest_mdl, conn['input'],
                                            {'container': source_container_key, 'mdl': source_mdl_key,
                                             'source_idx': source_idx, 'dest_idx': dest_idx, 'output': conn['output']}
                                        )
                                    elif order_by == 'output':
                                        add_entry(
                                            mdl_connections, source_mdl, conn['output'],
                                            {'container': dest_container_key, 'mdl': dest_mdl_key,
                                             'source_idx': source_idx, 'dest_idx': dest_idx, 'input': conn['input']}
                                        )

                    if 'destination' in conn.keys():
                        source_mdl_key = mdl_key
                        source_mdl = mdl
                        source_container_key = container_key
                        dest_container_key = conn['destination']['container']
                        dest_container = dyn_mdls[dest_container_key]
                        for dest_mdl_key, dest_mdl in dest_container.items():
                            if conn['destination']['mdl'] == '*' or conn['destination']['mdl'] == dest_mdl_key:
                                dest_idx, mask = dps_uf.lookup_strings(
                                    conn['destination']['id'], dest_mdl.par['name'], return_mask=True
                                )
                                source_idx = np.where(mask)[0]
                                if len(dest_idx) > 0:
                                    if order_by == 'input':
                                        add_entry(
                                            mdl_connections, dest_mdl, conn['input'],
                                            {'container': source_container_key, 'mdl': source_mdl_key,
                                             'source_idx': source_idx, 'dest_idx': dest_idx, 'output': conn['output']}
                                        )
                                    elif order_by == 'output':
                                        add_entry(
                                            mdl_connections, source_mdl, conn['output'],
                                            {'container': dest_container_key, 'mdl': dest_mdl_key,
                                             'source_idx': source_idx, 'dest_idx': dest_idx, 'input': conn['input']}
                                        )

    return mdl_connections


def get_submodules(mdl):
    attributes = inspect.getmembers(mdl)
    attributes = [a for a in attributes if not (a[0].startswith('__') and a[0].endswith('__'))]
    output = [mdl]
    for attr, typ in attributes:
        if issubclass(type(typ), DAEModel):
            submodule = getattr(mdl, attr)
            [output.append(sm) for sm in get_submodules(submodule)]
    return output


def output(f):
    @functools.wraps(f)
    def wrap(self, *args):
        if self._store_output:
            if not self._output_ready[f.__name__]:
                # print('Output not ready, calculating output')
                self._output_values[f.__name__] = f(self, *args)
                self._output_ready[f.__name__] = True
            return self._output_values[f.__name__]
        else:
            return f(self, *args)
    return wrap


class DynMdl:
    def get_states(self, x):
        return x[self.idx].view(dtype=self.dtypes)

    def set_input(self, input_name, value, idx=None):
        if idx is not None:
            self.input[input_name][idx] = value
        else:
            self.input[input_name] = value


class DAEModel:
    """Base class for dynamic models"""
    def __init__(self, par=None, sys_par=None, first_state_idx=0, n_units=None, **kwargs):
        # type(self)._ids = count(0)
        # self.id = next(type(self)._ids)

        if n_units is not None:
            name = self.__class__.__name__
            # print(name, n_units, np.array((name,)*n_units, dtype=[('name', '<U32')]))
            # print([name] * n_units)
            self.par = dps_uf.structured_array_from_list(names=['name'], entries=[(name,)]*n_units)

        else:
            self.par = par if par is not None else self.parse_input_data(**kwargs)
        self.sys_par = sys_par

        self.n_units = len(self.par)
        # self.output_value = np.zeros(self.n_units)
        # self._input_value = np.zeros(self.n_units)
        self.n_states = len(self.state_list())
        self.idx = slice(first_state_idx, first_state_idx + self.n_states*self.n_units)  # This will be shifted from outside

        self.state_desc = np.vstack([
            np.repeat(self.par['name'], self.n_states),
            np.tile(self.state_list(), self.n_units)]
        ).T

        self.dtypes = [(state, float) for state in self.state_list()]
        self.state_idx = np.zeros((self.n_units,), dtype=[(state, int) for state in self.state_list()])
        self.state_idx_global = np.zeros((self.n_units,), dtype=[(state, int) for state in self.state_list()])
        # self.state_idx_global = np.zeros((self.n_units,), dtype=[(state, int) for state in self.state_list()])
        for i, state in enumerate(self.state_list()):
            idx = np.where(self.state_desc[:, 1] == state)[0]
            self.state_idx[state] = idx
            self.state_idx_global[state] = idx + first_state_idx
            # mdl.state_idx_global[state] = idx

        self.add_blocks()
        self.update_block_names()

        self._output_ready = np.zeros(1, dtype=[(var, bool) for var in self.output_list()])
        self._output_values = np.zeros(self.n_units, dtype=[(var, float) for var in self.output_list()])
        self._input_values = np.zeros(self.n_units, dtype=[(var, float) for var in self.input_list()])
        self._store_output = False
        [self.disconnect_input(inp) for inp in self.input_list()]

        self.int_par = np.zeros(self.n_units, dtype=[(var, float) for var in self.int_par_list()])

    def state_list(self):
        """
        Returns list of states for dynamic model.
        Should be overwritten (if model has one or more states)
        """
        return []

    def input_list(self):
        """
        Returns list of inputs for dynamic model.
        Should be overwritten (if model has one or more inputs)
        """
        return ['input']

    def output_list(self):
        """
        Returns list of outputs for dynamic model.
        Should be overwritten (if model has one or more outputs)
        """
        return ['output']

    def int_par_list(self):
        return []

    def reset_outputs(self):
        self._output_ready[:] = False
        # for key in self._output_ready.keys():
        #     self._output_ready[key] = False

    def set_input(self, input_name, value, idx=None):
        if idx is not None:
            self._input_values[input_name][idx] = value
        else:
            self._input_values[input_name] = value

    def disconnect_input(self, input_name):
        def input(self, x, v):
            return self._input_values[input_name]
        setattr(type(self), input_name, input)

    def add_blocks(self):
        """Sub-modules can be specified by overwriting this function"""
        pass

    def update_block_names(self):
        """Update names of modules and sub-modules"""
        # names = [mdl.par['name'] for mdl in get_submodules(mdl_0)[1:]]
        # np.array(names)[:, 1]

        new_names = []
        for mdl in get_submodules(self)[1:]:
            # print(mdl.par['name'])
            concat_str = np.core.defchararray.add
            new_names.append(concat_str(self.par['name'], concat_str('/', mdl.par['name'])))

        unique, counts = np.unique(np.array(new_names), return_counts=True, axis=0)

        for x in unique:
            number = 0
            for i in range(0, len(new_names)):
                if np.array_equal(new_names[i], x):
                    number += 1

                    if number >= 2:
                        # print(number, new_names[i])
                        new_names[i] = concat_str(new_names[i], f'-{number}')

        for mdl, names in zip(get_submodules(self)[1:], new_names):

            if mdl.n_states > 0:
                mdl.state_desc = np.vstack([
                    np.repeat(names, mdl.n_states),
                    np.tile(mdl.state_list(), mdl.n_units)]
                ).T
            names = np.array(names, dtype=[('name', names.dtype)])
            old_par = dps_uf.remove_recarray_field(mdl.par, 'name')
            mdl.par = dps_uf.combine_recarrays(names, old_par)

    def parse_input_data(self, **kwargs):
        """Read parameters from kwargs"""
        # p = dm.par
        # kwargs = dict(K_a=p['K'], T=p['T_a'])
        names = []
        data = []
        for arg, val in kwargs.items():
            names.append(arg)
            data.append(val)

        if not 'name' in names:
            names.append('name')
            data.append(np.array([self.__class__.__name__]*len(list(kwargs.values())[0])))

        entries = [entry for entry in zip(*data)]
        col_dtypes = [np.array(col).dtype for col in data]
        dtypes = [(name_, dtype_) for name_, dtype_ in zip(names, col_dtypes)]

        return np.array(entries, dtype=dtypes)

    def local_view(self, x):
        return x[self.idx].view(dtype=self.dtypes)

    def output(self, x, v):
        pass

    # def state_derivatives(self, dx, x, v):
    #     pass


# class DAEModel(ODEModel):
#     def disconnect_input(self, input_name):
#         def input(self, x, v):
#             return self._input_values[input_name]
#         setattr(type(self), input_name, input)


def auto_init(mdl, x0, v0, output_0):
    submodules = get_submodules(mdl)
    n_states_all = len(x0)
    
    # Find states belonging to model:
    state_idx = []
    state_idx_local = []
    n_states = 0
    start_idx = 0
    for submodule in submodules:
        state_list = submodule.state_list()
        if len(state_list) > 0:
            state_idx.append(submodule.idx)
            n_states_mdl = submodule.idx.stop - submodule.idx.start
            n_states += n_states_mdl
            state_idx_local.append(slice(start_idx, start_idx + n_states_mdl))
            start_idx += n_states_mdl
    
    init_val = 1
    n_int_par = len(mdl.int_par_list())
    n_units = mdl.n_units
    n_sol = n_states + n_int_par*n_units
    int_par_idx = slice(n_states, None)
    x_test = np.ones(n_sol)

    def ode_fun_mdl(x_test):
        int_par = x_test[int_par_idx]
        mdl.int_par[:] = int_par
        x_all_test = np.zeros_like(x0)
        for idx, idx_local in zip(state_idx, state_idx_local):
            x_all_test[idx] = x_test[idx_local]
        
        dx_all = np.zeros(n_states_all)
        for submodule in submodules:
            submodule.reset_outputs()
            submodule._store_output = True

        dx_all = np.zeros(n_states_all)
        for submodule in submodules:
            if hasattr(submodule, 'state_derivatives'):
                submodule.state_derivatives(dx_all, x_all_test, v0)

        for submodule in submodules:
            submodule._store_output = False

        dx_mdl = []
        for idx in state_idx:
            dx_mdl.append(dx_all[idx])

        output_err = mdl.output(x_all_test, v0) - output_0
        return np.concatenate(dx_mdl + [output_err])

    x_test = np.concatenate([x0[:n_states], mdl.int_par['bias']])
    x_test[:] = 1

    err_best = 1e6
    for init_conditions in [np.ones(n_sol), np.zeros(n_sol), np.random.randn(n_sol)]:
        try:
            sol = least_squares(ode_fun_mdl, init_conditions)
            err = np.linalg.norm(ode_fun_mdl(sol['x']))
            if err < err_best:
                x_sol_best = sol['x']
                err_best = err
        except ValueError:
            continue

    x_sol_all = x0.copy()
    for idx, idx_local in zip(state_idx, state_idx_local):
        x_sol_all[idx] = x_sol_best[idx_local]

    
    mdl.int_par[:] = x_sol_best[int_par_idx]
        
    assert np.linalg.norm(mdl.output(x_sol_all, v0) - output_0) < 1e-6
    # assert max(abs(ps.state_derivatives(0, x_sol_all, ps.v_0))) < 1e-6
    x0[:] = x_sol_all