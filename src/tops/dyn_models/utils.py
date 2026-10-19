import numpy as np
import contextlib
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
    """Return ``mdl`` followed by all its blocks (recursively), in the order they were created."""
    output = [mdl]
    for block in mdl.blocks.values():
        output.extend(get_submodules(block))
    return output


class AlgebraicLoopError(RuntimeError):
    """Raised when an output depends on itself without passing through a state."""


def output(f):
    """Decorator for output methods of models.

    While outputs are stored (within ``store_outputs``), each output is computed once and the stored
    value is returned on later calls, so that an output used by several blocks is only computed once
    per evaluation.

    An output that depends on itself without passing through a state (an algebraic loop) raises
    AlgebraicLoopError. Loops through a state are fine, since outputs of states do not depend on
    inputs.
    """
    @functools.wraps(f)
    def wrap(self, *args):
        # Keyed by f (not by name), so that an overriding output can call super().output
        if self._store_output and f in self._stored_outputs:
            return self._stored_outputs[f]
        if f in self._outputs_in_progress:
            raise AlgebraicLoopError(
                f'{type(self).__name__}.{f.__name__} ({", ".join(self.par["name"][:3])}) depends on '
                f'itself without passing through a state (algebraic loop).')
        self._outputs_in_progress.add(f)
        try:
            value = f(self, *args)
        finally:
            self._outputs_in_progress.discard(f)
        if self._store_output:
            value = self._stored_outputs[f] = np.copy(value)  # Copy, since value may be a view of x
        return value
    return wrap


@contextlib.contextmanager
def store_outputs(mdls):
    """Within the context, each output of the models is computed once and then stored (see output).
    Should be entered for each evaluation, since stored outputs do not depend on (x, v)."""
    for mdl in mdls:
        mdl._stored_outputs.clear()
        mdl._store_output = True
    try:
        yield
    finally:
        for mdl in mdls:
            mdl._store_output = False
            mdl._stored_outputs.clear()


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

        self._stored_outputs = {}  # See output and store_outputs
        self._outputs_in_progress = set()
        self._input_values = np.zeros(self.n_units, dtype=[(var, float) for var in self.input_list()])
        self._store_output = False
        for input_name in self.input_list():
            self.disconnect_input(input_name)

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
        self._stored_outputs.clear()

    def set_input(self, input_name, value, idx=None):
        """Set the value of an input. Units where the input is connected to another model's output
        are not affected, since their value is given by the connection."""
        if idx is not None:
            self._input_values[input_name][idx] = value
        else:
            self._input_values[input_name] = value

    def disconnect_input(self, input_name):
        """Make the input read its value from _input_values (set with set_input), replacing any
        connection. Set on the instance, so that other instances and the class are not affected."""
        setattr(self, input_name, functools.partial(self._read_input, input_name))

    def _read_input(self, input_name, x, v):
        return self._input_values[input_name]

    def __setattr__(self, name, value):
        # Blocks (sub-models) are registered when assigned to an attribute, e.g. self.gain = Gain(...)
        blocks = self.__dict__.get('_blocks')
        if isinstance(value, DAEModel):
            if blocks is None:
                blocks = self.__dict__['_blocks'] = {}
            for other_name, block in blocks.items():
                if block is value and other_name != name:
                    raise ValueError(
                        f'{type(self).__name__}: block is already assigned to self.{other_name}, '
                        f'cannot also assign it to self.{name}.')
            blocks[name] = value
        elif blocks is not None:
            blocks.pop(name, None)
        object.__setattr__(self, name, value)

    @property
    def blocks(self):
        """Blocks (sub-models) of the model, by attribute name, in the order they were created."""
        return self.__dict__.get('_blocks', {})

    def add_blocks(self):
        """Create blocks (sub-models) by assigning them to attributes, e.g. self.gain = Gain(...).
        Should be overwritten (if model has blocks)."""
        pass

    def update_block_names(self):
        """Name the blocks (recursively) after their attribute, e.g. 'AVR1/gain' or 'VSC1/pll/filter'."""
        for attr, block in self.blocks.items():
            names = np.char.add(self.par['name'].astype(str), f'/{attr}')
            other_par = dps_uf.remove_recarray_field(block.par, 'name')
            block.par = dps_uf.combine_recarrays(np.array(names, dtype=[('name', names.dtype)]), other_par)
            if block.n_states > 0:
                block.state_desc = np.vstack([
                    np.repeat(block.par['name'], block.n_states),
                    np.tile(block.state_list(), block.n_units)]
                ).T
            block.update_block_names()

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

        dx_all = np.zeros_like(x0)
        # Other units may be evaluated at meaningless points, hence errstate
        with store_outputs(submodules), np.errstate(all='ignore'):
            for submodule in state_submodules:
                submodule.state_derivatives(dx_all, x_all, v0)
            output_err = mdl.output(x_all, v0)[i] - output_0[i]

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