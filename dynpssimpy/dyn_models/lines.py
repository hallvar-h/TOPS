import numpy as np
from dynpssimpy.dyn_models.utils import DAEModel
from dynpssimpy.utility_functions import lookup_strings
from scipy.sparse import lil_matrix


class Line(DAEModel):
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

    def event(self, ps, line_name, event_name):
        line_idx = lookup_strings(line_name, ps.lines['Line'].par['name'])

        if event_name in ['connect', 'disconnect']:

            if event_name == 'connect':
                sign = 1
                self.connected[line_idx] = True
            elif event_name == 'disconnect':
                sign = -1
                self.connected[line_idx] = False

            idx_from = self.bus_idx_red['from_bus'][line_idx]
            idx_to = self.bus_idx_red['to_bus'][line_idx]

            admittance = self.admittance[line_idx]
            shunt = self.shunt[line_idx]


            buses_in_red_sys = idx_from in ps.bus_idx_red and idx_to in ps.bus_idx_red
            data = np.array([admittance + shunt/2,
                             admittance + shunt/2,
                             -admittance,
                             -admittance])

            if buses_in_red_sys:
                rows_red = np.array([idx_from, idx_to, idx_from, idx_to])
                cols_red = np.array([idx_from, idx_to, idx_to, idx_from])
                y_line_red = lil_matrix((ps.n_bus_red,) * 2, dtype=complex)
                y_line_red[rows_red, cols_red] = data
                ps.y_bus_red += y_line_red*sign

            else:
                print('Line buses are not in reduced system, line event failed.')

    def load_flow_adm(self):
        # print('hei')
        # buses = self.ref['buses']
        # z_n = buses.z_n
        # print(self.sys_par['bus_v_n'])
        z_n = self.sys_par['bus_v_n'] ** 2 / self.sys_par['s_n']

        data = self.data
        # self.idx_from = dps_uf.lookup_strings(data['from_bus'], buses.data['name'])
        # self.idx_to = dps_uf.lookup_strings(data['to_bus'], buses.data['name'])
        self.shunt = np.zeros(self.n_units, dtype=complex)
        self.admittance = np.zeros(self.n_units, dtype=complex)
        lengths = data['length'] if 'length' in data.dtype.names else np.ones(self.n)
        for i, line in enumerate(data):
            idx_from = self.bus_idx['from_bus'][i]
            idx_to = self.bus_idx['to_bus'][i]
            length = lengths[i]
            if 'unit' not in line.dtype.names or line['unit'] in ['p.u.', 'pu', 'pu/km']:
                if 'S_n' in line.dtype.names and 'V_n' in line.dtype.names and line['S_n'] != 0 and line['V_n'] != 0:
                    # If impedance given in p.u./km
                    impedance = (line['R'] + 1j * line['X']) * length * line['V_n'] ** 2 / line['S_n'] / \
                                z_n[idx_from]
                    shunt = 1j * line['B'] * length * 1 / (
                            line['V_n'] ** 2 / line['S_n'] / z_n[idx_from])
                else:
                    # Per unit of system base and bus nominal voltage
                    impedance = (line['R'] + 1j * line['X']) * length
                    shunt = 1j * line['B'] * length
            elif line['unit'] in ['PF', 'pf', 'PowerFactory', 'powerfactory']:
                # Given in ohm/km, but with capacitance in micro-Siemens
                impedance = (line['R'] + 1j * line['X']) * length / z_n[idx_from]
                shunt = 1j * line['B'] * length * z_n[idx_from] * 1e-6
            elif line['unit'] in ['Ohm', 'ohm']:
                # Given in Ohm/km
                impedance = (line['R'] + 1j * line['X']) * length / z_n[idx_from]
                shunt = 1j * line['B'] * length * z_n[idx_from]
            admittance = 1 / impedance
            self.admittance[i] = admittance
            self.shunt[i] = shunt

        idx_from = self.bus_idx['from_bus']
        idx_to = self.bus_idx['to_bus']

        rows = np.array([idx_from, idx_to, idx_from, idx_to])
        cols = np.array([idx_from, idx_to, idx_to, idx_from])
        data = np.array(
            [self.admittance + self.shunt/2, self.admittance + self.shunt/2, -self.admittance, -self.admittance])

        return data, (rows, cols)


    #     s_shunt = -1j * self.par['Q'] / self.sys_par['s_n']
    #     z = np.conj(abs(1) ** 2 / s_shunt)
    #     y_shunt = 1 / z
    #     return y_shunt, (self.bus_idx_red['terminal'],) * 2
    #
    # def dyn_const_adm_(self):
    #     return self.load_flow_adm()
