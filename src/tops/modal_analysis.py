"""Numerical linearisation and eigenvalue (modal) analysis of a PowerSystemModel."""

import contextlib
import numpy as np
import tops.dyn_models.utils as mdl_utils
import tops.utility_functions as dps_uf


class PowerSystemModelLinearization:
    """Linearised model around an operating point, with eigenvalues and eigenvectors.

    The state-space model is

    .. math:: \\dot{\\Delta x} = A \\Delta x + B \\Delta u, \\quad \\Delta y = C \\Delta x + D \\Delta u

    where the inputs u and outputs y are inputs and outputs of the dynamic models (see :meth:`linearize`).

    :param ps: Initialised PowerSystemModel (``ps.init_dyn_sim()`` has been called).
    :param eps: Relative step of the central differences. The step for a variable with value z is
        ``eps*max(1, |z|)``.
    """

    def __init__(self, ps, eps=1e-6):
        self.ps = ps
        self.eps = eps
        self.linearization_ready = False
        self.eigenvalues_ready = False

        self.n = ps.n_states
        self.t0 = 0
        self.x0 = ps.x0
        self.v0 = ps.v0
        self.inputs = []
        self.outputs = []
        self.input_desc = []
        self.output_desc = []
        self.a = np.zeros((self.n,) * 2)
        self.b = np.zeros((self.n, 0))
        self.c = np.zeros((0, self.n))
        self.d = np.zeros((0, 0))

    def linearize(self, inputs=(), outputs=(), x0=None, t0=0, get_eigs=False):
        """Linearise the model around the operating point ``x0``.

        Inputs and outputs are given as functions, either alone (all units) or as ``(function, units)``, where units
        are given by name or by integer index, as a single value or a list. For example::

            lin.linearize(
                inputs=[(ps.avr.SEXS.v_setp, 'AVR1'), ps.gov.TGOV1.bias],
                outputs=[(ps.gen.GEN.speed, ['G1', 'G3']), lambda x, v: abs(v[3])],
            )

        - An input is an input of a model (in its ``input_list()``), also of a block, e.g.
          ``ps.avr.SEXS.tg_red.input``. Take it after ``ps.init_dyn_sim()``, which replaces connected inputs.
        - An output is any function ``f(x, v)``, e.g. ``ps.gen.GEN.speed`` or a lambda. Units can be selected by
          name only for functions of a model; otherwise, use the indices of the elements that the function returns.

        Each selected unit gives one input (column of B and D) or one output (row of C and D). An input is perturbed
        by adding a small value to the value it would otherwise have, so connected inputs (e.g. ``P_m`` of a
        generator with a governor) can be used, and the input of a unit can be thought of as an added signal.

        :param inputs: List of inputs.
        :param outputs: List of outputs.
        :param x0: Operating point (default: ``ps.x0``).
        :param t0: Time at which the model is evaluated.
        :param get_eigs: Also compute the eigenvalues (see :meth:`eigenvalue_decomposition`).
        :return: The matrices ``(A, B, C, D)``.
        """
        ps = self.ps
        self.n = ps.n_states
        self.t0 = t0
        self.x0 = np.array(ps.x0 if x0 is None else x0, dtype=float)
        self.eigenvalues_ready = False
        self.v0 = ps.solve_algebraic(t0, self.x0, v_0=ps.v0)
        self.inputs = self._resolve_inputs(inputs)
        self.outputs = self._resolve_outputs(outputs)
        self.input_desc = [label for *_, label in self.inputs]
        self.output_desc = [label for *_, label in self.outputs]

        y0 = self._eval_outputs(self.x0, self.v0)
        dtype_y = complex if np.iscomplexobj(y0) else float

        # A and C: perturb the states
        self.a = np.zeros((self.n, self.n))
        self.c = np.zeros((len(self.outputs), self.n), dtype=dtype_y)
        for j in range(self.n):
            h = self.eps * max(1, abs(self.x0[j]))
            dx, dy = [], []
            for sign in (1, -1):
                x = self.x0.copy()
                x[j] += sign * h
                v = ps.solve_algebraic(t0, x, v_0=self.v0)
                dx.append(ps.state_derivatives(t0, x, v))
                dy.append(self._eval_outputs(x, v))
            self.a[:, j] = (dx[0] - dx[1]) / (2 * h)
            self.c[:, j] = (dy[0] - dy[1]) / (2 * h)

        # B and D: perturb the inputs
        self.b = np.zeros((self.n, len(self.inputs)))
        self.d = np.zeros((len(self.outputs), len(self.inputs)), dtype=dtype_y)
        for j, (mdl, name, unit, _) in enumerate(self.inputs):
            u0 = getattr(mdl, name)(self.x0, self.v0)[unit]
            h = self.eps * max(1, abs(u0))
            dx, dy = [], []
            for sign in (1, -1):
                with _offset_input(mdl, name, unit, sign * h):
                    v = ps.solve_algebraic(t0, self.x0, v_0=self.v0)
                    dx.append(ps.state_derivatives(t0, self.x0, v))
                    dy.append(self._eval_outputs(self.x0, v))
            self.b[:, j] = (dx[0] - dx[1]) / (2 * h)
            self.d[:, j] = (dy[0] - dy[1]) / (2 * h)

        self.linearization_ready = True
        if get_eigs:
            self.eigenvalue_decomposition()
        return self.a, self.b, self.c, self.d

    def _find_input(self, fun):
        """The model and name of the input that is the function fun, or (None, None)."""
        for mdl in self.ps.dyn_mdls:
            for name in mdl.input_list():
                if mdl.__dict__.get(name) is fun:
                    return mdl, name
        return None, None

    def _resolve_inputs(self, specs):
        """Turn inputs into a list of ``(mdl, name, unit index, label)``, one per unit."""
        resolved = []
        for fun, units in map(_split_spec, specs):
            mdl, name = self._find_input(fun)
            if mdl is None:
                raise ValueError(
                    f"{fun!r} is not an input of a model of the PowerSystemModel. Inputs are taken as e.g. "
                    f"ps.avr.SEXS.v_setp, after ps.init_dyn_sim() (which replaces connected inputs)."
                )
            for unit in _unit_idx(units, mdl.n_units, mdl.par["name"]):
                resolved.append((mdl, name, unit, f"{mdl.par['name'][unit]}.{name}"))
        return resolved

    def _resolve_outputs(self, specs):
        """Turn outputs into a list of ``(function, element index, label)``, one per element."""
        resolved = []
        for fun, units in map(_split_spec, specs):
            mdl, name = self._find_input(fun)
            if mdl is not None:
                # Looked up at each call, so that the output follows when the input is perturbed
                fun = lambda x, v, mdl=mdl, name=name: getattr(mdl, name)(x, v)  # noqa: E731
            else:
                mdl = getattr(fun, "__self__", None)
                name = getattr(fun, "__name__", "output")
            names = mdl.par["name"] if isinstance(mdl, mdl_utils.DAEModel) else None
            n = np.size(fun(self.x0, self.v0))
            for i in _unit_idx(units, n, names):
                label = f"{names[i]}.{name}" if names is not None else f"{name}[{i}]"
                resolved.append((fun, i, label))
        return resolved

    def _eval_outputs(self, x, v):
        with mdl_utils.store_outputs(self.ps.dyn_mdls):
            return np.array([np.atleast_1d(fun(x, v))[i] for fun, i, _ in self.outputs])

    def eigenvalue_decomposition(self):
        """Compute eigenvalues, right and left eigenvectors, frequencies and damping ratios of A.

        The model is linearised first (without inputs and outputs) if :meth:`linearize` has not been called.

        :return: The eigenvalues.
        """
        if not self.linearization_ready:
            self.linearize()

        self.eigs, self.rev = np.linalg.eig(self.a)  # Right eigenvectors as columns
        self.lev = np.linalg.inv(self.rev)  # Left eigenvectors as rows, scaled such that lev @ rev = I
        self.freq = self.eigs.imag / (2 * np.pi)
        abs_eigs = abs(self.eigs)
        self.damping = np.divide(-self.eigs.real, abs_eigs, out=np.full(self.n, np.nan), where=abs_eigs > 0)
        self.eigenvalues_ready = True
        return self.eigs

    def _require_eigs(self):
        if not self.eigenvalues_ready:
            self.eigenvalue_decomposition()

    def residues(self, mode_idx):
        """Residues of the transfer functions from the inputs to the outputs, for one mode.

        :param mode_idx: Index of the mode.
        :return: Array with one row per output and one column per input.
        """
        self._require_eigs()
        return np.outer(self.c @ self.rev[:, mode_idx], self.lev[mode_idx, :] @ self.b)

    def participation_factors(self):
        """Participation factors, ``p[k, i]`` being the participation of state k in mode i.

        :return: Complex array with one row per state and one column per mode. Use ``abs`` for magnitudes.
        """
        self._require_eigs()
        return self.rev * self.lev.T

    def get_mode_idx(self, mode_type=("em", "non_conj"), damp_threshold=1, freq_range=(0.1, 3), sorted=True):
        """Indices of the modes that meet the given criteria.

        :param mode_type: One or more of ``'em'`` (electromechanical: frequency within ``freq_range``) and
            ``'non_conj'`` (only the eigenvalue with non-negative imaginary part of each complex pair).
        :param damp_threshold: Only modes with damping ratio below this value are kept.
        :param freq_range: Frequency range [Hz] of the ``'em'`` criterion.
        :param sorted: Sort from least to most damped.
        :return: Array of mode indices.
        """
        self._require_eigs()
        if isinstance(mode_type, str):
            mode_type = [mode_type]

        keep = self.damping < damp_threshold
        abs_freq = abs(self.freq)
        if "em" in mode_type:
            keep &= (freq_range[0] < abs_freq) & (abs_freq < freq_range[1])
        if "non_conj" in mode_type:
            keep &= self.eigs.imag >= 0

        idx = np.flatnonzero(keep)
        if sorted:
            idx = idx[np.argsort(self.damping[idx])]
        return idx

    def get_dominant_mode(self, freq_range=(0.1, 3)):
        """Index of the least damped electromechanical mode, or None if there is none.

        :param freq_range: Frequency range [Hz] of electromechanical modes.
        """
        idx = self.get_mode_idx(("em", "non_conj"), damp_threshold=np.inf, freq_range=freq_range)
        return idx[0] if len(idx) else None


def _split_spec(spec):
    """Split an input or output, given as function or (function, units), into (function, units)."""
    if callable(spec):
        return spec, None
    if isinstance(spec, tuple) and len(spec) == 2 and callable(spec[0]):
        return spec
    raise TypeError(f"Inputs and outputs must be a function or (function, units), got {spec!r}.")


def _unit_idx(units, n, names=None):
    """Integer indices of units given by name or index (single value or list), or all n units if None."""
    if units is None:
        return np.arange(n)
    units = np.atleast_1d(units)
    if units.dtype.kind in "iu":
        if np.any((units < -n) | (units >= n)):
            raise IndexError(f"Unit index out of range for {n} units: {units.tolist()}.")
        return units % n
    if names is None:
        raise TypeError(f"Units can be selected by name only for functions of a model, got {units.tolist()}.")
    idx, found = dps_uf.lookup_strings(list(units), names, return_mask=True)
    if not all(found):
        missing = [str(u) for u, f in zip(units, found) if not f]
        raise KeyError(f"Units {missing} not found among {list(names)}.")
    return idx


@contextlib.contextmanager
def _offset_input(mdl, name, unit, delta):
    """Within the context, add delta to an input of one unit, on top of its value from set_input or a connection."""
    offset = np.zeros(mdl.n_units)
    offset[unit] = delta
    had_own = name in mdl.__dict__
    original = getattr(mdl, name)
    setattr(mdl, name, lambda x, v: original(x, v) + offset)
    try:
        yield
    finally:
        if had_own:
            setattr(mdl, name, original)
        else:
            delattr(mdl, name)
