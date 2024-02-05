import numpy as np
from tops.dyn_models.utils import DAEModel


class Trafo(DAEModel):
    def __init__(self, data, sys_par, **kwargs):
        super().__init__(data, sys_par, **kwargs)
        self.data = data
        self.par = data
        self.n_units = len(data)
        self.connected = np.ones(self.n_units, dtype=bool)

        self.bus_idx = np.array(np.zeros(self.n_units), dtype=[(key, int) for key in self.bus_ref_spec().keys()])
        self.bus_idx_red = np.array(np.zeros(self.n_units), dtype=[(key, int) for key in self.bus_ref_spec().keys()])
        self.sys_par = sys_par  # {'s_n': 0, 'f_n': 50, 'bus_v_n': None}

    def bus_ref_spec(self):
        return {'from_bus': self.par['from_bus'], 'to_bus': self.par['to_bus']}

    def init_extras(self):
        self.idx_from = self.bus_idx_red['from_bus']
        self.idx_to = self.bus_idx_red['to_bus']
        n_bus = self.sys_par['n_bus']

        # Trafos:
        n_elements = len(self.par)
        self.v_to_i = np.zeros((n_elements, n_bus), dtype=complex)
        self.v_to_i_rev = np.zeros((n_elements, n_bus), dtype=complex)
        
        for i in range(self.n_units):
            # This might not be correct for phase shifting transformers (conj in the right place?)
            shunt_from = self.ratio_from_0[i] * np.conj(self.ratio_from_0[i]) * self.admittance[i]
            shunt_to = self.ratio_to_0[i] * np.conj(self.ratio_to_0[i]) * self.admittance[i]
            adm_from_to = self.ratio_from_0[i] * np.conj(self.ratio_to_0[i]) * self.admittance[i]
            adm_to_from = np.conj(self.ratio_from_0[i]) * self.ratio_to_0[i] * self.admittance[i]

            self.v_to_i[i, [self.idx_from[i], self.idx_to[i]]] = [shunt_from, -adm_from_to]
            self.v_to_i_rev[i, [self.idx_to[i], self.idx_from[i]]] = [shunt_to, -adm_to_from]
    
    def load_flow_adm(self):
        z_n = self.sys_par['bus_v_n'] ** 2 / self.sys_par['s_n']

        data = self.data
        self.admittance = np.zeros(self.n_units, dtype=complex)
        self.ratio_from_0 = np.zeros(self.n_units, dtype=complex)
        self.ratio_to_0 = np.zeros(self.n_units, dtype=complex)

        self.idx_from = idx_from = self.bus_idx['from_bus']  # dps_uf.lookup_strings(trafo['from_bus'], buses['name'])
        self.idx_to = idx_to = self.bus_idx['to_bus']  # dps_uf.lookup_strings(trafo['to_bus'], buses['name'])

        for i, trafo in enumerate(self.par):

            ratio_from = (trafo['ratio_from'] if not np.isnan(trafo['ratio_from']) else 1) if 'ratio_from' in trafo.dtype.names else 1
            ratio_to = (trafo['ratio_to'] if not np.isnan(trafo['ratio_to']) else 1) if 'ratio_to' in trafo.dtype.names else 1

            V_n_from = trafo['V_n_from'] if 'V_n_from' in trafo.dtype.names and trafo['V_n_from'] > 0 else self.sys_par['bus_v_n'][idx_from[i]]
            Z_base_trafo = V_n_from ** 2/(trafo['S_n'] if 'S_n' in trafo.dtype.names else self.sys_par['s_n'])  # <= Could also have used _to instead of _from
            impedance = (trafo['R']+1j*trafo['X'])*Z_base_trafo/z_n[idx_from[i]]
            n_par = trafo['N_par'] if 'N_par' in trafo.dtype.names else 1
            admittance = n_par/impedance

            self.admittance[i] = admittance
            self.ratio_from_0[i] = ratio_from
            self.ratio_to_0[i] = ratio_to

        rows = np.array([idx_from, idx_to, idx_from, idx_to])
        cols = np.array([idx_from, idx_to, idx_to, idx_from])
        data = np.array([
            self.ratio_from_0*np.conj(self.ratio_from_0)*self.admittance,
            self.ratio_to_0*np.conj(self.ratio_to_0)*self.admittance,
            -self.ratio_from_0*np.conj(self.ratio_to_0)*self.admittance,
            -np.conj(self.ratio_from_0)*self.ratio_to_0*self.admittance
        ])

        # self.init_extras()
        self.rows = rows
        self.cols = cols

        self.init_extras()

        return data, (self.rows, self.cols)

    def dyn_const_adm(self):
        return self.load_flow_adm()

    def i_from(self, x, v):
        v_full = v
        return self.v_to_i.dot(v_full)*self.connected

    def i_to(self, x, v):
        v_full = v
        return self.v_to_i_rev.dot(v_full)*self.connected

    def s_from(self, x, v):
        v_full = v
        return v_full[self.idx_from]*np.conj(self.i_from(x, v))

    def s_to(self, x, v):
        v_full = v
        return v_full[self.idx_to]*np.conj(self.i_to(x, v))

    def p_from(self, x, v):
        return self.s_from(x, v).real

    def p_to(self, x, v):
        return self.s_to(x, v).real

    def q_from(self, x, v):
        return self.s_from(x, v).imag

    def q_to(self, x, v):
        return self.s_to(x, v).imag

    def s_line(self, x, v):
        return self.s_from(x, v) + self.s_to(x, v)

    def p_line(self, x, v):
        return self.s_line(x, v).real

    def q_line(self, x, v):
        return self.s_line(x, v).imag

    def p_loss_tot(self, x, v):
        return np.sum(np.abs(self.p_line(x, v)))


    
class DynTrafo(Trafo):
    def input_list(self):
        return ['ratio_from', 'ratio_to']
    
    def load_flow_adm(self):
        z_n = self.sys_par['bus_v_n'] ** 2 / self.sys_par['s_n']

        data = self.data
        self.admittance = np.zeros(self.n_units, dtype=complex)
        self.ratio_from_0 = self.par['ratio_from']
        self.ratio_to_0 = self.par['ratio_to']

        self.idx_from = idx_from = self.bus_idx['from_bus']
        self.idx_to = idx_to = self.bus_idx['to_bus']

        for i, trafo in enumerate(self.par):

            V_n_from = trafo['V_n_from'] if 'V_n_from' in trafo.dtype.names and trafo['V_n_from'] > 0 else self.sys_par['bus_v_n'][idx_from[i]]
            Z_base_trafo = V_n_from ** 2/(trafo['S_n'] if 'S_n' in trafo.dtype.names else self.sys_par['s_n'])  # <= Could also have used _to instead of _from
            impedance = (trafo['R']+1j*trafo['X'])*Z_base_trafo/z_n[idx_from[i]]
            n_par = trafo['N_par'] if 'N_par' in trafo.dtype.names else 1
            admittance = n_par/impedance
            self.admittance[i] = admittance

        rows = np.array([idx_from, idx_to, idx_from, idx_to])
        cols = np.array([idx_from, idx_to, idx_to, idx_from])
        data = np.array([
            self.ratio_from_0*np.conj(self.ratio_from_0)*self.admittance,
            self.ratio_to_0*np.conj(self.ratio_to_0)*self.admittance,
            -self.ratio_from_0*np.conj(self.ratio_to_0)*self.admittance,
            -np.conj(self.ratio_from_0)*self.ratio_to_0*self.admittance
        ])

        self.rows = rows
        self.cols = cols
    
        return data, (self.rows, self.cols)

    def init_from_load_flow(self, x_0, v_0, S):
        self._input_values['ratio_from'] = self.ratio_from_0
        self._input_values['ratio_to'] = self.ratio_to_0

    def dyn_const_adm(self):
        return np.ones(0), (np.ones(0),)*2

    def dyn_var_adm(self, x, v):
        ratio_from = self.ratio_from(x, v)
        ratio_to = self.ratio_to(x, v)
        data = np.array([
            ratio_from              *   np.conj(ratio_from) *   self.admittance,
            ratio_to                *   np.conj(ratio_to)   *   self.admittance,
            -ratio_from             *   np.conj(ratio_to)   *   self.admittance,
            -np.conj(ratio_from)    *   ratio_to            *   self.admittance
        ])

        return data, (self.rows, self.cols)
    
    def i_from(self, x, v):
        v_full = v
        v_from = v_full[self.idx_from]
        v_to = v_full[self.idx_to]

        shunt_from = self.ratio_from(x, v)*np.conj(self.ratio_from(x, v))*self.admittance
        adm_from_to = self.ratio_from(x, v)*np.conj(self.ratio_to(x, v))*self.admittance

        return v_from*shunt_from - v_to*adm_from_to

    def i_to(self, x, v):
        v_full = v
        v_from = v_full[self.idx_from]
        v_to = v_full[self.idx_to]
        
        shunt_to = self.ratio_to(x, v) * np.conj(self.ratio_to(x, v)) * self.admittance
        adm_to_from = np.conj(self.ratio_from(x, v)) * self.ratio_to(x, v) * self.admittance

        return v_to*shunt_to - v_from*adm_to_from