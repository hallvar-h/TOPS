import numpy as np
from dynpssimpy.dyn_models.utils import DAEModel


class Trafo(DAEModel):
    def __init__(self, data, sys_par, **kwargs):
        super().__init__(data, sys_par, **kwargs)
        self.data = data
        self.par = data
        self.n_units = len(data)

        self.bus_idx = np.array(np.zeros(self.n_units), dtype=[(key, int) for key in self.bus_ref_spec().keys()])
        self.bus_idx_red = np.array(np.zeros(self.n_units), dtype=[(key, int) for key in self.bus_ref_spec().keys()])
        self.sys_par = sys_par  # {'s_n': 0, 'f_n': 50, 'bus_v_n': None}

    def bus_ref_spec(self):
        return {'from_bus': self.par['from_bus'], 'to_bus': self.par['to_bus']}

    def init_extras(self):
        # This is copied from lines:Line model, is not correct yet.
        self.idx_from = self.bus_idx_red['from_bus']
        self.idx_to = self.bus_idx_red['to_bus']
        n_bus = self.sys_par['n_bus']

        self.v_to_i = np.zeros((self.n_units, n_bus), dtype=complex)
        self.v_to_i_rev = np.zeros((self.n_units, n_bus), dtype=complex)
        self.from_mat = np.zeros((self.n_units, n_bus), dtype=complex)
        self.to_mat = np.zeros((self.n_units, n_bus), dtype=complex)
        for i, row in enumerate(self.par):
            self.v_to_i[i, [self.idx_from[i], self.idx_to[i]]] = [self.admittance[i] + self.shunt[i]/2, -self.admittance[i]]
            self.v_to_i_rev[i, [self.idx_to[i], self.idx_from[i]]] = [self.admittance[i] + self.shunt[i]/2, -self.admittance[i]]
            self.from_mat[i, self.idx_from[i]] = 1
            self.to_mat[i, self.idx_to[i]] = 1
    
    def load_flow_adm(self):
        # print('hei')
        # buses = self.ref['buses']
        # z_n = buses.z_n
        # print(self.sys_par['bus_v_n'])
        z_n = self.sys_par['bus_v_n'] ** 2 / self.sys_par['s_n']

        data = self.data
        # self.idx_from = dps_uf.lookup_strings(data['from_bus'], buses.data['name'])
        # self.idx_to = dps_uf.lookup_strings(data['to_bus'], buses.data['name'])
        self.admittance = np.zeros(self.n_units, dtype=complex)
        self.ratio_from = np.zeros(self.n_units, dtype=complex)
        self.ratio_to = np.zeros(self.n_units, dtype=complex)

        idx_from = self.bus_idx['from_bus']  # dps_uf.lookup_strings(trafo['from_bus'], buses['name'])
        idx_to = self.bus_idx['to_bus']  # dps_uf.lookup_strings(trafo['to_bus'], buses['name'])

        for i, trafo in enumerate(self.par):

            ratio_from = (trafo['ratio_from'] if not np.isnan(trafo['ratio_from']) else 1) if 'ratio_from' in trafo.dtype.names else 1
            ratio_to = (trafo['ratio_to'] if not np.isnan(trafo['ratio_to']) else 1) if 'ratio_to' in trafo.dtype.names else 1

            V_n_from = trafo['V_n_from'] if 'V_n_from' in trafo.dtype.names and trafo['V_n_from'] > 0 else self.sys_par['bus_v_n'][idx_from[i]]
            Z_base_trafo = V_n_from ** 2/(trafo['S_n'] if 'S_n' in trafo.dtype.names else self.sys_par['s_n'])  # <= Could also have used _to instead of _from
            impedance = (trafo['R']+1j*trafo['X'])*Z_base_trafo/z_n[idx_from[i]]
            n_par = trafo['N_par'] if 'N_par' in trafo.dtype.names else 1
            admittance = n_par/impedance

            self.admittance[i] = admittance
            self.ratio_from[i] = ratio_from
            self.ratio_to[i] = ratio_to

        rows = np.array([idx_from, idx_to, idx_from, idx_to])
        cols = np.array([idx_from, idx_to, idx_to, idx_from])
        data = np.array([
            self.ratio_from*np.conj(self.ratio_from)*self.admittance,
            self.ratio_to*np.conj(self.ratio_to)*self.admittance,
            -self.ratio_from*np.conj(self.ratio_to)*self.admittance,
            -np.conj(self.ratio_from)*self.ratio_to*self.admittance
        ])

        # self.init_extras()


        return data, (rows, cols)
