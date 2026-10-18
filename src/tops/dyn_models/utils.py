import numpy as np
import inspect
import functools
import warnings
from typing import NamedTuple
import tops.utility_functions as dps_uf
from scipy.optimize import least_squares


class _Endpoint(NamedTuple):
    """One end of a connection: a model, where it lives, and which of its units are involved."""
    container: str
    key: str
    mdl: object
    idx: np.ndarray


def determine_connections(dyn_mdls, order_by='input'):
    """Resolve the connections declared by each model's ``connections()`` method.

    A connection is declared on one model (the local side) and points to models in another
    container (the remote side), either as a ``'source'`` (remote output -> local input) or as a
    ``'destination'`` (local output -> remote input). The ``'id'`` list of the connection is aligned
    with the local model's units and is looked up among the remote models' names.

    With ``order_by='input'`` the result is ``{dest_mdl: {input: [entry, ...]}}`` where each entry
    names the source model and output. With ``order_by='output'`` it is
    ``{source_mdl: {output: [entry, ...]}}`` where each entry names the destination model and input.
    """
    if order_by not in ('input', 'output'):
        raise ValueError(f"order_by must be 'input' or 'output', got {order_by!r}")

    mdl_connections = {}
    for container_key, mdls in dyn_mdls.items():
        for mdl_key, mdl in mdls.items():
            if not hasattr(mdl, 'connections'):
                continue
            for conn in mdl.connections():
                for source, dest in _resolve_connection(dyn_mdls, container_key, mdl_key, mdl, conn):
                    if order_by == 'input':
                        key_mdl, key_var = dest.mdl, conn['input']
                        entry = {'container': source.container, 'mdl': source.key, 'output': conn['output']}
                    else:
                        key_mdl, key_var = source.mdl, conn['output']
                        entry = {'container': dest.container, 'mdl': dest.key, 'input': conn['input']}
                    entry.update(source_idx=source.idx, dest_idx=dest.idx)
                    mdl_connections.setdefault(key_mdl, {}).setdefault(key_var, []).append(entry)

    return mdl_connections


def _resolve_connection(dyn_mdls, container_key, mdl_key, mdl, conn):
    """Yield ``(source, dest)`` endpoint pairs for each remote model matched by ``conn``."""
    for direction in ('source', 'destination'):
        if direction not in conn:
            continue
        spec = conn[direction]
        for remote_key, remote_mdl in dyn_mdls[spec['container']].items():
            if spec['mdl'] not in ('*', remote_key):
                continue
            remote_idx, mask = dps_uf.lookup_strings(spec['id'], remote_mdl.par['name'], return_mask=True)
            if len(remote_idx) == 0:
                continue

            local = _Endpoint(container_key, mdl_key, mdl, np.where(mask)[0])
            remote = _Endpoint(spec['container'], remote_key, remote_mdl, remote_idx)
            yield (remote, local) if direction == 'source' else (local, remote)


def get_submodules(mdl):
    attributes = inspect.getmembers(mdl)
    attributes = [a for a in attributes if not (a[0].startswith('__') and a[0].endswith('__'))]
    output = [mdl]
    for attr, typ in attributes:
        if issubclass(type(typ), DAEModel):
            submodule = getattr(mdl, attr)
            output.extend(get_submodules(submodule))
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
    events_class = None  # Subclass of tops.events.ModelEvents, defining the events of the model

    def __init__(self, par=None, sys_par=None, first_state_idx=0, n_units=None, **kwargs):
        # type(self)._ids = count(0)
        # self.id = next(type(self)._ids)
        if hasattr(self, 'int_par_list'):
            raise TypeError(
                f'{type(self).__name__} defines int_par_list(), but int_par has been removed. '
                f'Add the parameters to input_list() instead, and set them in _input_values.'
            )

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

        self.events = self.events_class(self) if self.events_class is not None else None

    @property
    def int_par(self):
        raise AttributeError(
            f'{type(self).__name__}.int_par has been removed. Former internal parameters (e.g. bias) '
            f"are now inputs: read with mdl.bias(x, v) and set with mdl.set_input('bias', value)."
        )

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
            concat_str = np.char.add
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



def summation(*funcs):
    '''Creates a new function which sums the output of functions and constants/
    constant arrays specified as arguments
    '''
    callables = [func for func in funcs if callable(func)]
    constants = [func for func in funcs if not callable(func)]
    def summation(*args):
        return sum([func(*args) for func in callables] + constants)
    
    return summation


# class DAEModel(ODEModel):
#     def disconnect_input(self, input_name):
#         def input(self, x, v):
#             return self._input_values[input_name]
#         setattr(type(self), input_name, input)


def auto_init(mdl, x0, v0, output_0, free_inputs=(), tol=1e-6):
    """Numerically initialize ``mdl`` (including its submodules) to a steady state.

    Solves for the states and the given ``free_inputs`` such that all state derivatives are zero
    and ``mdl.output`` equals ``output_0``. Each unit is solved separately, since the units of a
    model are independent (all model equations are elementwise over units). This keeps the number
    of free variables per solve at ``n_states_per_unit + len(free_inputs)``, regardless of
    ``n_units``. The residual is still evaluated for all units (the models are vectorized), but only
    the entries of the unit being solved are used.

    ``free_inputs`` are inputs of ``mdl`` that are not connected to other models (typically a
    'bias' input). Their solved values are stored in ``mdl._input_values``.
    """
    submodules = get_submodules(mdl)
    state_submodules = [sm for sm in submodules if sm.n_states > 0]

    # Global state indices of each unit, shape (n_units, n_states_per_unit)
    unit_state_idx = np.hstack([
        np.column_stack([sm.state_idx_global[state] for state in sm.state_list()])
        for sm in state_submodules
    ]) if state_submodules else np.zeros((mdl.n_units, 0), dtype=int)
    n_states_unit = unit_state_idx.shape[1]
    n_free = n_states_unit + len(free_inputs)

    x_all = x0.copy()

    def residual(z, i):
        x_all[unit_state_idx[i]] = z[:n_states_unit]
        for input_name, val in zip(free_inputs, z[n_states_unit:]):
            mdl._input_values[input_name][i] = val

        for submodule in submodules:
            submodule.reset_outputs()
            submodule._store_output = True

        dx_all = np.zeros_like(x0)
        with np.errstate(all='ignore'):  # Other units may be evaluated at meaningless points
            for submodule in state_submodules:
                submodule.state_derivatives(dx_all, x_all, v0)
            output_err = mdl.output(x_all, v0)[i] - output_0[i]

        for submodule in submodules:
            submodule._store_output = False

        return np.append(dx_all[unit_state_idx[i]], output_err)

    rng = np.random.default_rng(0)
    for i in range(mdl.n_units):
        z_best, err_best = None, np.inf
        for z_init in [np.ones(n_free), np.zeros(n_free), rng.standard_normal(n_free)]:
            try:
                sol = least_squares(residual, z_init, args=(i,))
            except ValueError:
                continue
            err = np.linalg.norm(residual(sol.x, i))
            if err < err_best:
                z_best, err_best = sol.x, err
            if err_best < tol:
                break

        if z_best is None:
            warnings.warn(f'Initialization of {mdl.par["name"][i]} ({mdl.__class__.__name__}) failed.')
            continue

        residual(z_best, i)  # Leave x_all and inputs at the best solution
        if err_best > tol:
            warnings.warn(
                f'{mdl.par["name"][i]} ({mdl.__class__.__name__}) was potentially not initialized '
                f'properly (error: {err_best:.2e}).'
            )

    x0[:] = x_all