import numpy as np
from dynpssimpy.dyn_models.utils import DAEModel
from dynpssimpy.dyn_models.blocks import *
from .pll import PLL1


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


class ConstCurrentLoad(DAEModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.bus_idx = np.array(np.zeros(self.n_units), dtype=[(key, int) for key in self.bus_ref_spec().keys()])
        self.bus_idx_red = np.array(np.zeros(self.n_units), dtype=[(key, int) for key in self.bus_ref_spec().keys()])

    def bus_ref_spec(self):
        return {'terminal': self.par['bus']}

    def add_blocks(self):
        p = self.par
        self.pll = PLL1(T_filter=self.par['T_pll'], bus=p['bus'])

        self.I_inj = lambda x, v: (self.Id_setp(x, v) - 1j*self.Iq_setp(x, v))*np.exp(1j*self.pll.output(x, v))

    def input_list(self):
        return ['Id_setp', 'Iq_setp']

#    def P(self, x, v):
#        v_n = self.sys_par['bus_v_n'][self.bus_idx_red['terminal']]
#        V = abs(v[self.bus_idx_red['terminal']])*v_n
#        return np.sqrt(3)*V*self.I_d(x, v)
#
#    def Q(self, x, v):
#        v_n = self.sys_par['bus_v_n'][self.bus_idx_red['terminal']]
#        V = abs(v[self.bus_idx_red['terminal']])*v_n
#        return np.sqrt(3)*V*self.I_q(x, v)

    def load_flow_pq(self):
        return self.bus_idx['terminal'], self.par['P'], self.par['Q']

    def init_from_load_flow(self, x_0, v_0, S):
        # self._input_values['P_setp'] = self.par['P']
        # self._input_values['Q_setp'] = self.par['Q']

        v_n = self.sys_par['bus_v_n'][self.bus_idx_red['terminal']]

        V_0 = v_0[self.bus_idx_red['terminal']]*v_n

        I_d_0 = self.par['P']/(abs(V_0)*np.sqrt(3))
        I_q_0 = self.par['Q']/(abs(V_0)*np.sqrt(3))
        self._input_values['Id_setp'] = I_d_0
        self._input_values['Iq_setp'] = I_q_0

    def current_injections(self, x, v):
        i_n = self.sys_par['s_n'] / (np.sqrt(3) * self.sys_par['bus_v_n'])
        # self.P(x, v)
        return self.bus_idx_red['terminal'], self.I_inj(x, v)/i_n[self.bus_idx_red['terminal']]