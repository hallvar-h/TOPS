import numpy as np
from dynpssimpy.dyn_models.utils import DAEModel


class Load(DAEModel):
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

    def dyn_const_adm(self):
        return self.y_load, (self.bus_idx_red['terminal'],)*2

    def i(self, x, v):
        return v[self.bus_idx_red['terminal']]*self.y_load
    
    def s(self, x, v):
        return v[self.bus_idx_red['terminal']]*np.conj(self.i(x, v))

    def p(self, x, v):
        return self.s(x, v).real

    def q(self, x, v):
        return self.s(x, v).imag


class DynamicLoad(DAEModel):
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

    def dyn_var_adm(self):
        return self.y_load, (self.bus_idx_red['terminal'],)*2

    def i(self, x, v):
        return v[self.bus_idx_red['terminal']]*self.y_load
    
    def s(self, x, v):
        return v[self.bus_idx_red['terminal']]*np.conj(self.i(x, v))

    def p(self, x, v):
        return self.s(x, v).real

    def q(self, x, v):
        return self.s(x, v).imag

