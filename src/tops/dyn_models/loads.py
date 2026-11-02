import numpy as np
from tops.dyn_models.utils import DAEModel
from tops.dyn_models.blocks import TimeConstant
from tops.events import ModelEvents, AdmittanceChange


def read_power(column, value):
    """A power in MW or Mvar given to set_power, as a finite float."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = np.nan
    if not np.isfinite(number):
        raise ValueError(f'{column} must be a finite number, got {value!r}.')
    return number


def float_power_columns(par):
    """The load table with float P and Q, as set_power must not truncate the integer columns a table may have."""
    if par['P'].dtype.kind == 'f' and par['Q'].dtype.kind == 'f':
        return par
    dtype = [(name, float if name in ('P', 'Q') else par.dtype[name]) for name in par.dtype.names]
    return par.astype(dtype)


def write_power(mdl, load_idx, P, Q):
    """Write the power of a load, in MW and Mvar (None keeps it), to the table of its model.

    Nothing is changed if a value is refused, or if the bus of the load is not in the reduced system.
    Returns False if the power is as it was.
    """
    if mdl.bus_idx_red['terminal'][load_idx] < 0:
        raise ValueError(f'Bus of load {mdl.par["name"][load_idx]} is not in the reduced system.')
    p = mdl.par['P'][load_idx] if P is None else read_power('P', P)
    q = mdl.par['Q'][load_idx] if Q is None else read_power('Q', Q)
    if p == mdl.par['P'][load_idx] and q == mdl.par['Q'][load_idx]:
        return False

    mdl.par = mdl.data = float_power_columns(mdl.par)
    mdl.par['P'][load_idx] = p
    mdl.par['Q'][load_idx] = q
    return True


class LoadEvents(ModelEvents):
    actions = ('set_power',)

    def set_power(self, load_name, P=None, Q=None):
        idx = self._unit_idx(load_name)
        change = self.mdl.power_change(idx, P, Q)
        return [] if change is None else [change]


class Load(DAEModel):
    events_class = LoadEvents

    def __init__(self, data, sys_par, **kwargs):
        super().__init__(data, sys_par, **kwargs)
        self.data = data
        self.par = data
        self.n_units = len(data)

        self.bus_idx = np.array(np.zeros(self.n_units), dtype=[(key, int) for key in self.bus_ref_spec().keys()])
        self.bus_idx_red = np.array(np.zeros(self.n_units), dtype=[(key, int) for key in self.bus_ref_spec().keys()])
        self.sys_par = sys_par  # {'s_n': 0, 'f_n': 50, 'bus_v_n': None}

    def bus_ref_spec(self):
        return {'terminal': self.par['bus']}

    def reduced_system(self):
        return self.par['bus']

    def load_flow_pq(self):
        return self.bus_idx['terminal'], self.par['P'], self.par['Q']

    def init_from_load_flow(self, x_0, v_0, S):
        self.v_0 = v_0[self.bus_idx['terminal']]
        s_load = (self.par['P'] + 1j * self.par['Q']) / self.sys_par['s_n']
        z_load = np.conj(abs(self.v_0) ** 2 / s_load)
        self.y_load = 1/z_load

        V_n = self.sys_par['bus_v_n'][self.bus_idx['terminal']]
        self.I_n = self.sys_par['s_n']/(np.sqrt(3)*V_n)

    def dyn_const_adm(self):
        return self.y_load, (self.bus_idx_red['terminal'],)*2

    def power_change(self, load_idx, P=None, Q=None):
        """Set the power of a load in MW and Mvar (None keeps it), at the voltage of the load flow.

        Returns the change in the reduced admittance matrix, or None if the power is as it was.
        """
        if not write_power(self, load_idx, P, Q):
            return None
        bus = self.bus_idx_red['terminal'][load_idx]
        s_load = (self.par['P'][load_idx] + 1j*self.par['Q'][load_idx])/self.sys_par['s_n']
        y_load = np.conj(s_load)/abs(self.v_0[load_idx])**2
        change = y_load - self.y_load[load_idx]
        self.y_load[load_idx] = y_load
        return AdmittanceChange(np.array([change]), np.array([bus]), np.array([bus]))

    def i(self, x, v):
        return v[self.bus_idx_red['terminal']]*self.y_load
    
    def I(self, x, v):
        return self.i(x, v)*self.I_n
    
    def s(self, x, v):
        return v[self.bus_idx_red['terminal']]*np.conj(self.i(x, v))

    def p(self, x, v):
        # p.u. system base
        return self.s(x, v).real

    def q(self, x, v):
        # p.u. system base
        return self.s(x, v).imag
    
    def P(self, x, v):
        # MW
        return self.s(x, v).real*self.sys_par['s_n']

    def Q(self, x, v):
        # MVA
        return self.s(x, v).imag*self.sys_par['s_n']


class DynamicLoad(DAEModel):
    def __init__(self, data, sys_par, **kwargs):
        super().__init__(data, sys_par, **kwargs)
        self.data = data
        self.par = data
        self.n_units = len(data)

        self.bus_idx = np.array(np.zeros(self.n_units), dtype=[(key, int) for key in self.bus_ref_spec().keys()])
        self.bus_idx_red = np.array(np.zeros(self.n_units), dtype=[(key, int) for key in self.bus_ref_spec().keys()])
        self.sys_par = sys_par  # {'s_n': 0, 'f_n': 50, 'bus_v_n': None}

    def input_list(self):
        return ['g_setp', 'b_setp']
    
    def bus_ref_spec(self):
        return {'terminal': self.par['bus']}

    def reduced_system(self):
        return self.par['bus']

    def load_flow_pq(self):
        return self.bus_idx['terminal'], self.par['P'], self.par['Q']

    def init_from_load_flow(self, x_0, v_0, S):
        self.v_0 = v_0[self.bus_idx['terminal']]
        s_load = (self.par['P'] + 1j * self.par['Q']) / self.sys_par['s_n']
        z_load = np.conj(abs(self.v_0) ** 2 / s_load)
        y_load = 1/z_load
        self._input_values['g_setp'] = y_load.real
        self._input_values['b_setp'] = y_load.imag

        V_n = self.sys_par['bus_v_n'][self.bus_idx['terminal']]
        self.I_n = self.sys_par['s_n']/(np.sqrt(3)*V_n)

    def g_load(self, x, v):
        return self.g_setp(x, v)

    def b_load(self, x, v):
        return self.b_setp(x, v)

    def y_load(self, x, v):
        return self.g_load(x, v) + 1j*self.b_load(x, v)

    def dyn_var_adm(self, x, v):
        return self.y_load(x, v), (self.bus_idx_red['terminal'],)*2

    def i(self, x, v):
        return v[self.bus_idx_red['terminal']]*self.y_load(x, v)
    
    def I(self, x, v):
        return self.i(x, v)*self.I_n
    
    def s(self, x, v):
        return v[self.bus_idx_red['terminal']]*np.conj(self.i(x, v))

    def p(self, x, v):
        # p.u. system base
        return self.s(x, v).real

    def q(self, x, v):
        # p.u. system base
        return self.s(x, v).imag
    
    def P(self, x, v):
        # MW
        return self.s(x, v).real*self.sys_par['s_n']

    def Q(self, x, v):
        # MVA
        return self.s(x, v).imag*self.sys_par['s_n']


class DynamicLoadFiltered(DynamicLoad):
    """Dynamic load where the input is filtered using a low pass filter. 
    
    The load is an admittance which is determined by the output of two low pass filters,
    one for G (conductance) and one for B (susceptance).
    """

    def add_blocks(self):
        p = self.par
        self.lpf_g = TimeConstant(T=p['T_g'])
        self.lpf_g.input = lambda x, v: self.g_setp(x, v)

        self.lpf_b = TimeConstant(T=p['T_b'])
        self.lpf_b.input = lambda x, v: self.b_setp(x, v)

    def g_load(self, x, v):
        return self.lpf_g.output(x, v)

    def b_load(self, x, v):
        return self.lpf_b.output(x, v)

    def init_from_load_flow(self, x_0, v_0, S):
        self.v_0 = v_0[self.bus_idx['terminal']]
        s_load = (self.par['P'] + 1j * self.par['Q']) / self.sys_par['s_n']
        z_load = np.conj(abs(self.v_0) ** 2 / s_load)
        y_load = 1/z_load
        self._input_values['g_setp'] = y_load.real
        self._input_values['b_setp'] = y_load.imag

        self.lpf_g.initialize(x_0, v_0, y_load.real)
        self.lpf_b.initialize(x_0, v_0, y_load.imag)

        V_n = self.sys_par['bus_v_n'][self.bus_idx['terminal']]
        self.I_n = self.sys_par['s_n']/(np.sqrt(3)*V_n)


class ConstPowerLoad(DAEModel):
    events_class = LoadEvents

    def __init__(self, data, sys_par, **kwargs):
        super().__init__(data, sys_par, **kwargs)
        self.data = data
        self.par = data
        self.n_units = len(data)

        self.bus_idx = np.array(np.zeros(self.n_units), dtype=[(key, int) for key in self.bus_ref_spec().keys()])
        self.bus_idx_red = np.array(np.zeros(self.n_units), dtype=[(key, int) for key in self.bus_ref_spec().keys()])
        self.sys_par = sys_par  # {'s_n': 0, 'f_n': 50, 'bus_v_n': None}

    def bus_ref_spec(self):
        return {'terminal': self.par['bus']}

    def reduced_system(self):
        return self.par['bus']

    def load_flow_pq(self):
        return self.bus_idx['terminal'], self.par['P'], self.par['Q']

    def power_change(self, load_idx, P=None, Q=None):
        """Set the power of a load in MW and Mvar (None keeps it). The power is read at every step, so there is no
        change of the admittance matrix to return."""
        write_power(self, load_idx, P, Q)

    def apparent_power_injections(self, x, v):
        s_inj = -(self.par['P'] + 1j*self.par['Q'])/self.sys_par['s_n']
        return self.bus_idx_red['terminal'], s_inj
