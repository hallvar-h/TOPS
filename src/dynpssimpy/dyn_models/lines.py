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
        self.sys_par = sys_par  # {'s_n': 0, 'f_n': 50, n_bus: None, 'bus_v_n': None}

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

    def init_extras(self):
        self.idx_from = self.bus_idx_red['from_bus']
        self.idx_to = self.bus_idx_red['to_bus']
        n_bus = self.sys_par['n_bus']

        self.v_to_i = np.zeros((self.n_units, n_bus), dtype=complex)
        self.v_to_i_rev = np.zeros((self.n_units, n_bus), dtype=complex)
        for i in range(self.n_units):
            self.v_to_i[i, [self.idx_from[i], self.idx_to[i]]] = [self.admittance[i] + self.shunt[i]/2, -self.admittance[i]]
            self.v_to_i_rev[i, [self.idx_to[i], self.idx_from[i]]] = [self.admittance[i] + self.shunt[i]/2, -self.admittance[i]]
    
    def load_flow_adm(self):
        z_n = self.sys_par['bus_v_n'] ** 2 / self.sys_par['s_n']

        data = self.data
        self.shunt = np.zeros(self.n_units, dtype=complex)
        self.admittance = np.zeros(self.n_units, dtype=complex)
        lengths = data['length'] if 'length' in data.dtype.names else np.ones(self.n_units)
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
        data = np.array([
            self.admittance + self.shunt/2,
            self.admittance + self.shunt/2,
            -self.admittance,
            -self.admittance
        ])

        self.init_extras()

        return data, (rows, cols)

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
