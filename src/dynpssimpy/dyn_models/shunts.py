import numpy as np
from dynpssimpy.dyn_models.utils import DAEModel


class Shunt(DAEModel):
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

    def load_flow_adm(self):
        s_shunt = -1j * self.par['Q'] / self.sys_par['s_n']
        z = np.conj(abs(1) ** 2 / s_shunt)
        y_shunt = 1 / z
        return y_shunt, (self.bus_idx['terminal'],) * 2
    #
    # def dyn_const_adm_(self):
    #     return self.load_flow_adm()
