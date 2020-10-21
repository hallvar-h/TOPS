from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from scipy.sparse import lil_matrix
import dynpssimpy.modal_analysis as dps_mdl
import dynpssimpy.utility_functions as dps_uf
import dynpssimpy.dyn_models.avr as avr_lib
import dynpssimpy.dyn_models.gov as gov_lib
import dynpssimpy.dyn_models.pss as pss_lib


class PowerSystemModel:
    def __init__(self, model):

        # Load flow parameters
        self.pf_max_it = 10
        self.tol = 1e-8

        # Get model data
        for td in ['buses', 'lines', 'loads', 'generators', 'transformers', 'shunts']:
            # print(td)
            if td in model:
                if isinstance(model[td], pd.core.frame.DataFrame):
                    # Import dataframe
                    setattr(self, td, model[td])
                else:
                    # Make DataFrame from list
                    setattr(self, td, pd.DataFrame(model[td][1:], columns=model[td][0]))
            else:
                setattr(self, td, pd.DataFrame())

        for td in ['gov', 'avr', 'pss']:
            setattr(self, td, dict())
            if td in model:
                for key in model[td].keys():
                    getattr(self, td)[key] = pd.DataFrame(model[td][key][1:], columns=model[td][key][0])

        self.branches = self.lines

        if 'slack_bus' in model:
            self.slack_bus = model['slack_bus']
        else:
            self.slack_bus = None

        self.n_bus = len(self.buses)
        self.n_gen = len(self.generators)

        # Base for pu-system (determined by transformers)
        self.f = model['f']
        self.s_n = model['base_mva']  # For all buses
        self.v_n = np.array(self.buses['V_n'])  # Assuming nominal bus voltages are according to transformer ratios
        self.z_n = self.v_n ** 2 / self.s_n
        self.i_n = self.s_n / (np.sqrt(3) * self.v_n)

        # self.S_n = self.s_n
        # self.I_n = self.i_n
        # self.V_n = self.v_n
        # self.z_n = self.z_n

        self.e = np.empty((0, 0))

        # Load flow
        self.v_g_setp = np.array(self.generators['V'], dtype=float)

        self.v_g = np.empty(self.n_gen, dtype=complex)
        self.i_inj = np.empty(self.n_gen, dtype=complex)
        self.i_g = np.empty(self.n_gen, dtype=complex)
        self.angle = np.empty(self.n_gen, dtype=complex)
        self.v_pss = np.zeros(self.n_gen, dtype=float)
        self.v_aux = np.zeros(self.n_gen, dtype=float)

        self.gen_bus_idx = np.array([self.buses[self.buses['name'] == gen['bus']].index[0] for i, gen in self.generators.iterrows()])

        # Remove duplicate buses
        _, idx = np.unique(self.gen_bus_idx, return_index=True)
        self.gen_bus_idx_unique = self.gen_bus_idx[np.sort(idx)]

        self.n_gen_bus = len(self.gen_bus_idx_unique)

        # Get gen nom. voltage. If given as zero in input data, select bus nom. voltage.
        self.V_n_gen = np.zeros(len(self.generators))
        self.S_n_gen = np.zeros(len(self.generators))
        self.P_n_gen = np.zeros(len(self.generators))
        for i, gen in self.generators.iterrows():
            if 'V_n' in gen and gen['V_n'] > 0:
                self.V_n_gen[i] = gen['V_n']
            else:
                self.V_n_gen[i] = self.v_n[self.gen_bus_idx[i]]

            if 'S_n' in gen and gen['S_n'] > 0:
                self.S_n_gen[i] = gen['S_n']
            else:
                self.S_n_gen[i] = self.s_n

            if 'PF_n' in gen:
                self.P_n_gen[i] = gen['S_n']*gen['PF_n']
            else:
                self.P_n_gen[i] = self.S_n_gen[i]

        self.I_n_gen = self.S_n_gen / (np.sqrt(3) * self.V_n_gen)
        self.Z_n_gen = self.V_n_gen ** 2 / self.S_n_gen

        self.n_par = np.array(self.generators['N_par']) if 'N_par' in self.generators else np.ones(self.n_gen)

        self.reduced_bus_idx = self.gen_bus_idx_unique
        self.y_bus = np.empty((self.n_bus, self.n_bus))  # Can not be built before power flow, since load admittances depend on it.
        self.y_bus_lf = np.empty((self.n_bus, self.n_bus))
        self.y_bus_lf[:] = np.nan
        self.y_bus_red = np.empty((self.n_gen, self.n_gen))
        self.y_bus_red_inv = np.empty((self.n_gen, self.n_gen))
        self.y_bus_red_mod = np.zeros_like(self.y_bus_red)

        self.time = 0.0

    def get_bus_idx(self, names):
        # Get index of bus with given bus names
        if isinstance(names, str):
            names = [names]

        bus_idx = -np.ones(len(names), dtype=int)
        for i, name in enumerate(names):
            idx = np.where(self.buses['name'] == name)[0]
            if len(idx) > 0:
                bus_idx[i] = idx
        return bus_idx
        # return pd.concat([self.buses[self.buses['name'] == name] for bus in names])

    def get_bus_idx_red(self, names):
        # sorter = np.argsort(self.reduced_bus_idx)
        # return sorter[np.searchsorted(self.reduced_bus_idx, self.get_bus_idx(names), sorter=sorter)]
        idx_full = self.get_bus_idx(names)
        idx_red = -np.ones(len(names), dtype=int)
        for i, idx in enumerate(idx_full):
            if idx >= 0:
                idx_red_search = np.where(self.reduced_bus_idx == idx)[0]
                if len(idx_red_search) > 0:
                    idx_red[i] = idx_red_search

        return idx_red

    def kron_reduction(self, y_bus, keep_buses):
        remove_buses = list(set(range(self.n_bus)) - set(keep_buses))
        y_rr = y_bus[np.ix_(remove_buses, remove_buses)]
        y_rk = y_bus[np.ix_(remove_buses, keep_buses)]
        y_kk = y_bus[np.ix_(keep_buses, keep_buses)]

        # Build matrix for mapping back to full system (v_full = self.red_to_full.dot(self.v_red)
        self.red_to_full = np.zeros((self.n_bus, self.n_bus_red), dtype=complex)
        remove_buses = list(set(range(self.n_bus)) - set(self.reduced_bus_idx))
        self.red_to_full[remove_buses, :] = - np.linalg.inv(y_rr).dot(y_rk)
        self.red_to_full[self.reduced_bus_idx] = np.eye(self.n_bus_red)

        return y_kk - y_rk.T.dot(np.linalg.inv(y_rr)).dot(y_rk)

    def read_admittance_data(self, element_type, element):
        buses = self.buses

        if element_type == 'line':
            line = element
            idx_from = buses[buses['name'] == line['from_bus']].index[0]
            idx_to = buses[buses['name'] == line['to_bus']].index[0]
            if line['unit'] in ['p.u.', 'pu', 'pu/km']:
                if 'S_n' in line and 'V_n' in line and line['S_n'] != 0 and line['V_n'] != 0:
                    # If impedance given in p.u./km
                    impedance = (line['R'] + 1j * line['X']) * line['length'] * line['V_n'] ** 2 / line['S_n'] / \
                                self.z_n[idx_from]
                    shunt = 1j * line['B'] * line['length'] * 1 / (
                            line['V_n'] ** 2 / line['S_n'] / self.z_n[idx_from])
                else:
                    # Per unit of system base and bus nominal voltage
                    impedance = (line['R'] + 1j * line['X']) * line['length']
                    shunt = 1j * line['B'] * line['length']
            elif line['unit'] in ['PF', 'pf', 'PowerFactory', 'powerfactory']:
                # Given in ohm/km, but with capacitance in micro-Siemens
                impedance = (line['R'] + 1j * line['X']) * line['length'] / self.z_n[idx_from]
                shunt = 1j * line['B'] * line['length'] * self.z_n[idx_from] * 1e-6
            elif line['unit'] in ['Ohm', 'ohm']:
                # Given in Ohm/km
                impedance = (line['R'] + 1j * line['X']) * line['length'] / self.z_n[idx_from]
                shunt = 1j * line['B'] * line['length'] * self.z_n[idx_from]
            admittance = 1/impedance
            return idx_from, idx_to, admittance, shunt

        elif element_type == 'transformer':
            trafo = element
            idx_from = buses[buses['name'] == trafo['from_bus']].index
            idx_to = buses[buses['name'] == trafo['to_bus']].index
            ratio_from = (trafo['ratio_from'] if not np.isnan(trafo['ratio_from']) else 1) if 'ratio_from' in trafo else 1
            ratio_to = (trafo['ratio_to'] if not np.isnan(trafo['ratio_to']) else 1) if 'ratio_to' in trafo else 1

            V_n_from = trafo['V_n_from'] if trafo['V_n_from'] else self.v_n[idx_from]
            Z_base_trafo = V_n_from ** 2 / trafo['S_n']  # <= Could also have used _to instead of _from
            impedance = (trafo['R'] + 1j * trafo['X']) * Z_base_trafo / self.z_n[idx_from]
            n_par = trafo['N_par'] if 'N_par' in trafo else 1
            admittance = n_par / impedance

            return idx_from, idx_to, admittance, ratio_from, ratio_to

    def build_y_branch(self):
        # Build matrices for easy computation of branch (line and trafo) currents.
        # E.g. i_lines = self.v_to_i_lines*v_bus (full system, not reduced).

        # Lines:
        n_elements = len(self.lines)
        self.v_to_i_lines = np.zeros((n_elements, self.n_bus), dtype=complex)
        self.v_to_i_lines_rev = np.zeros((n_elements, self.n_bus), dtype=complex)
        self.lines_from_mat = np.zeros((n_elements, self.n_bus), dtype=complex)
        self.lines_to_mat = np.zeros((n_elements, self.n_bus), dtype=complex)
        for i, element in self.lines.iterrows():
            idx_from, idx_to, admittance, shunt = self.read_admittance_data('line', element)
            self.v_to_i_lines[i, [idx_from, idx_to]] = [admittance + shunt/2, -admittance]
            self.v_to_i_lines_rev[i, [idx_to, idx_from]] = [admittance + shunt/2, -admittance]
            self.lines_from_mat[i, idx_from] = 1
            self.lines_to_mat[i, idx_to] = 1

        # Trafos:
        n_elements = len(self.transformers)
        self.v_to_i_trafos = np.zeros((n_elements, self.n_bus), dtype=complex)
        self.v_to_i_trafos_rev = np.zeros((n_elements, self.n_bus), dtype=complex)
        self.trafos_from_mat = np.zeros((n_elements, self.n_bus), dtype=complex)
        self.trafos_to_mat = np.zeros((n_elements, self.n_bus), dtype=complex)
        for i, element in self.transformers.iterrows():
            idx_from, idx_to, admittance, ratio_from, ratio_to = self.read_admittance_data('transformer', element)

            # This might not be correct for phase shifting transformers (conj in the right place?)
            shunt_from = ratio_from * np.conj(ratio_from) * admittance
            shunt_to = ratio_to * np.conj(ratio_to) * admittance
            adm_from_to = ratio_from * np.conj(ratio_to) * admittance
            adm_to_from = np.conj(ratio_from) * ratio_to * admittance

            self.v_to_i_trafos[i, [idx_from, idx_to]] = [shunt_from, -adm_from_to]
            self.v_to_i_trafos_rev[i, [idx_to, idx_from]] = [shunt_to, -adm_to_from]
            self.trafos_from_mat[i, idx_from] = 1
            self.trafos_to_mat[i, idx_to] = 1

    def build_y_bus(self, type='dyn', y_ext=np.empty((0, 0))):
        # Build bus admittance matrix.
        # If type=='dyn', generator admittances and load admittances are included in the admittance matrix.
        # Used for dynamic simulation.
        # If not type=='dyn', generator admittances and load admittances are not included.
        # Used for power flow calculation.

        n_bus = len(self.buses)

        y_branch = np.zeros((n_bus, n_bus), dtype=complex)  # Branches = Trafos + Lines
        buses = self.buses

        y_lines = np.zeros((n_bus, n_bus), dtype=complex)

        for i, line in self.lines.iterrows():
            idx_from, idx_to, admittance, shunt = self.read_admittance_data('line', line)
            rows = np.array([idx_from, idx_to, idx_from, idx_to])
            cols = np.array([idx_from, idx_to, idx_to, idx_from])
            data = np.array(
                [admittance + shunt / 2, admittance + shunt / 2, -admittance, -admittance])
            y_lines[rows, cols] += data

        y_branch += y_lines

        y_trafo = np.zeros((n_bus, n_bus), dtype=complex)
        for i, trafo in self.transformers.iterrows():

            idx_from, idx_to, admittance, ratio_from, ratio_to = self.read_admittance_data('transformer', trafo)

            rows = np.array([idx_from, idx_to, idx_from, idx_to])
            cols = np.array([idx_from, idx_to, idx_to, idx_from])
            data = np.array([
                ratio_from*np.conj(ratio_from)*admittance,
                ratio_to*np.conj(ratio_to)*admittance,
                -ratio_from*np.conj(ratio_to)*admittance,
                -np.conj(ratio_from)*ratio_to*admittance
            ])
            y_trafo[rows, cols] += data

        y_branch += y_trafo

        y_gen = np.zeros((n_bus, n_bus), dtype=complex)
        for i, gen in self.generators.iterrows():
            # Generator impedance on system base
            idx_bus = buses[buses['name'] == gen['bus']].index
            V_n = gen['V_n'] if gen['V_n'] else self.v_n[idx_bus]
            impedance = 1j * gen['X_d_st'] * V_n**2/gen['S_n']/self.z_n[self.gen_bus_idx[i]]
            y_gen[idx_bus, idx_bus] += self.n_par[i] / impedance

        y_load = np.zeros((n_bus, n_bus), dtype=complex)
        if type == 'dyn':
            for i, load in self.loads.iterrows():
                s_load = (load['P'] + 1j * load['Q']) / self.s_n
                if load['model'] == 'Z' and abs(s_load) > 0:
                    idx_bus = buses[buses['name'] == load['bus']].index
                    z = np.conj(abs(self.v_0[idx_bus])**2/s_load)
                    y_load[idx_bus, idx_bus] += 1/z

        y_shunt = np.zeros((n_bus, n_bus), dtype=complex)
        for i, shunt in self.shunts.iterrows():
            # if shunt['model'] == 'Z':
            idx_bus = buses[buses['name'] == shunt['bus']].index
            s_shunt = -1j*shunt['Q']/self.s_n

            z = np.conj(abs(1)**2/s_shunt)
            y_shunt[idx_bus, idx_bus] += 1/z

        self.y_gen = y_gen
        self.y_branch = y_branch
        self.y_load = y_load
        self.y_shunt = y_shunt
        self.y_ext = y_ext

        Y = y_branch.copy()
        Y += y_shunt
        if type == 'dyn':
            Y += y_gen + y_load
        if y_ext.shape[0] > 0:
            Y += y_ext
        return Y

    def build_y_bus_red(self, keep_extra_buses=[]):
        # Builds the admittance matrix of the reduced system by applying Kron reduction. By default, all buses other
        # generator buses are eliminated. Additional buses to include in the reduced system can be specified
        # in "keep_extra_buses" (list of bus names).

        # If extra buses are specified before , store these. To ensure that the reduced admittance matrix has the same
        # dimension if rebuilt (by i.e. by network_event()-function.
        if len(keep_extra_buses) > 0:
            keep_extra_buses_idx = [self.buses[self.buses.name == name].index[0] for name in keep_extra_buses]
            self.reduced_bus_idx = np.concatenate([self.gen_bus_idx, np.array(keep_extra_buses_idx, dtype=int)])

            # Remove duplicate buses
            _, idx = np.unique(self.reduced_bus_idx, return_index=True)
            self.reduced_bus_idx = self.reduced_bus_idx[np.sort(idx)]

        self.n_bus_red = len(self.reduced_bus_idx)
        self.y_bus_red = self.kron_reduction(self.y_bus, self.reduced_bus_idx)  # np.empty((self.n_gen, self.n_gen))
        self.y_bus_red_mod = np.zeros_like(self.y_bus_red)

        # self.n_bus_red = self.y_bus_red.shape[0]
        self.gen_bus_idx_red = self.get_bus_idx_red(self.buses.iloc[self.gen_bus_idx]['name'])

        self.build_y_branch()

    def power_flow(self, print_output=True):
        #  Build admittance matrix if not defined
        if np.any(np.isnan(self.y_bus_lf)):
            self.y_bus_lf = self.build_y_bus(type='lf')

        y_bus = self.y_bus_lf

        n_bus = self.n_bus
        n_gen = self.n_gen
        n_gen_bus = self.n_gen_bus

        # Determine bus types (if not given)
        bus_type = np.empty(self.n_bus, dtype='<U2')
        p_sum_bus = np.empty(self.n_bus)
        p_sum_loads_bus = np.empty(self.n_bus)
        q_sum_bus = np.empty(self.n_bus)
        q_sum_loads_bus = np.empty(self.n_bus)

        for i, bus in self.buses.iterrows():
            gen_idx = self.generators['bus'] == bus['name']
            p_sum_loads_bus[i] = (sum(self.loads[self.loads['bus'] == bus['name']]['P']) / self.s_n) if len(self.loads) > 0 else 0
            p_sum_bus[i] = p_sum_loads_bus[i] - sum(self.generators[gen_idx]['P']*self.n_par[gen_idx]) / self.s_n
            q_sum_loads_bus[i] = (sum(self.loads[self.loads['bus'] == bus['name']]['Q']) / self.s_n) if len(self.loads) > 0 else 0
            q_sum_bus[i] = q_sum_loads_bus[i]  # - 0*(sum(self.shunts[self.shunts['bus'] == bus['name']]['Q'])) / self.s_n
            # print(p_sum_bus, q_sum_bus)
            if any(bus['name'] == self.generators['bus']):
                bus_type[i] = 'PV'
            else:
                bus_type[i] = 'PQ'

        if self.slack_bus:
            bus_type[self.buses['name'] == self.slack_bus] = 'SL'
        else:
            bus_type[np.argmax(bus_type == 'PV')] = 'SL'  # Set the first generator as slack bus

        pv_idx = np.where(bus_type == 'PV')[0]
        pq_idx = np.where(bus_type == 'PQ')[0]
        pvpq_idx = np.concatenate([pv_idx, pq_idx])
        sl_idx = np.where(bus_type == 'SL')
        self.slack_bus = self.buses.iloc[sl_idx].name.tolist()[0]

        # Map x to angles and voltages
        idx_phi = range(self.n_bus - 1)
        idx_v = range(self.n_bus - 1, self.n_bus - 1 + len(pq_idx))

        def x_to_v(x):
            phi = np.zeros(self.n_bus)
            phi[pvpq_idx] = x[idx_phi]
            v = v_0.copy()
            v[pq_idx] = x[idx_v]
            v_ph = v * np.exp(1j * phi)
            return v_ph

        # Initial guess: Flat start
        phi_0 = np.zeros(self.n_bus)
        v_0 = np.ones(self.n_bus)  # self.v_0
        v_0[self.gen_bus_idx] = self.v_g_setp

        x0 = np.zeros(2 * (n_bus - 1) - (n_gen_bus - 1))
        x0[idx_phi] = phi_0[pvpq_idx]
        x0[idx_v] = v_0[pq_idx]

        x = x0.copy()

        def pf_equations(x):
            v_ph = x_to_v(x)
            S_calc = v_ph * np.conj(y_bus.dot(v_ph))
            # S_err = p_sum_bus + 1j*q_sum_bus + S_calc
            p_err = p_sum_bus + S_calc.real
            q_err = q_sum_bus + S_calc.imag

            return np.concatenate([p_err[pvpq_idx], q_err[pq_idx]])

        converged = False
        i = 0
        x = x0.copy()
        err = pf_equations(x)
        err_norm = max(abs(err))
        # print(err_norm)

        # Based on PYPOWER code!
        while (not converged and i < self.pf_max_it):
            i = i + 1
            J = dps_uf.jacobian_num(pf_equations, x)

            # Update step
            dx = np.linalg.solve(J, err)
            x -= dx

            err = pf_equations(x)
            err_norm = max(abs(err))
            # print(err_norm)

            if self.tol > err_norm:
                converged = True
                if print_output:
                    print('Power flow converged.')
            if i == self.pf_max_it and print_output:
                print('Power flow did not converge in {} iterations.'.format(self.pf_max_it))

        # soln = newton(pf_equations, x, full_output=True, maxiter=20000, disp=True)
        # pf_equations(soln.root)

        self.v_0 = x_to_v(x)
        self.s_0 = self.v_0 * np.conj(y_bus.dot(self.v_0))
        self.p_sum_loads_bus = p_sum_loads_bus
        self.q_sum_loads_bus = q_sum_loads_bus

    def init_dyn_sim(self):

        # Generator parameters, p.u. on system base (lower case letters) and p.u. on generator base (upper case letters)
        for par in ['X_d', 'X_q', 'X_d_t', 'X_q_t', 'X_d_st', 'X_q_st']:
            setattr(self, par, np.array(self.generators[par]))
            setattr(self, par.lower(), np.array(self.generators[par]) * self.Z_n_gen / self.z_n[self.gen_bus_idx])

        for par in ['T_d0_t', 'T_q0_t', 'T_d0_st', 'T_q0_st']:
            setattr(self, par, np.array(self.generators[par]))

        self.H = np.array(self.generators['H'])
        if 'PF_n' in self.generators:
            self.H /= np.array(self.generators['PF_n'])
            # self.H /= self.generators['PF_n']

        # # Build bus admittance matrices
        # self.y_bus = self.build_y_bus()
        # self.build_y_bus_red()

        # State variables:
        self.state_desc = np.empty((0, 2))

        def add_states(start_idx, data, states):
            states = np.array(states)
            names = data['name'].to_numpy()
            state_desc_mdl = np.vstack([np.tile(names, len(states)), np.repeat(states, len(data))]).T
            # state_desc_mdl = np.vstack([np.repeat(names, len(states)), np.tile(states, len(data))]).T

            mdl = dps_uf.DynamicModel()
            mdl.idx = start_idx + np.arange(len(state_desc_mdl), dtype=int)
            mdl.states = dict(zip(states, [mdl.idx[state_desc_mdl[:, 1] == state] for state in states]))
            mdl.par = data.to_dict(orient='list')

            for key in mdl.par.keys():
                mdl.par[key] = np.array(mdl.par[key])  # Convert data to np.arrays
            return mdl, state_desc_mdl

        # Generators
        data = self.generators
        states = np.array(['speed', 'angle', 'e_q_t', 'e_d_t', 'e_q_st', 'e_d_st'])
        self.gen_mdl, state_desc_mdl = add_states(len(self.state_desc), data, states)
        self.state_desc = np.vstack([self.state_desc, state_desc_mdl])
        [setattr(self, var + '_idx', self.gen_mdl.states[var]) for var in states]  # Define self.speed_idx, self.angle_idx and so on

        self.gov_mdls = dict()
        self.avr_mdls = dict()
        self.pss_mdls = dict()

        for input_data, container, library in zip([self.gov, self.pss, self.avr],
                                                  [self.gov_mdls, self.pss_mdls, self.avr_mdls],
                                                  [gov_lib, pss_lib, avr_lib]):

            for key in input_data.keys():
                data = input_data[key]
                start_idx = len(self.state_desc)
                mdl = getattr(library, key)()
                state_list = mdl.state_list
                names = data['name'].to_numpy()
                n_units = len(data)
                n_states = len(state_list)

                state_desc_mdl = np.vstack([np.tile(names, n_states), np.repeat(state_list, n_units)]).T

                mdl.idx = start_idx + np.arange(len(state_desc_mdl),
                                                dtype=int)  # Indices of all states belonging to model
                mdl.par = data.to_dict(orient='list')  # Model parameters
                mdl.active = np.ones(len(data), dtype=bool)
                mdl.int_par = dict.fromkeys(mdl.int_par_list, np.zeros(len(data)))
                mdl.gen_idx = np.array(
                    [self.generators[self.generators['name'] == name].index.tolist()[0] for name in data['gen']])
                mdl.state_idx = dict(
                    zip(state_list, [np.arange(n_units * i, n_units * (i + 1)) for i in range(n_states)]))

                for key_ in mdl.par.keys():
                    mdl.par[key_] = np.array(mdl.par[key_])  # Convert data to np.arrays

                container[key] = mdl
                self.state_desc = np.vstack([self.state_desc, state_desc_mdl])

        self.n_states = self.state_desc.shape[0]  # self.n_gen_states * self.n_gen
        self.state_desc_der = self.state_desc.copy()
        self.state_desc_der[:, 1] = np.char.add(np.array(self.n_states * ['d_']), self.state_desc[:, 1])
        self.x0 = np.zeros(self.n_states)
        self.e_q_0 = np.zeros(self.n_gen)
        self.e_q = np.zeros(self.n_gen)

        # Build reduced system
        self.y_bus = self.build_y_bus()  # np.empty((self.n_bus, self.n_bus))
        self.build_y_bus_red()

        # Choose first generator at slack bus as slack generator
        self.slack_generator = (self.generators[self.generators['bus'] == self.slack_bus]).index[0]
        self.p_m_setp = np.array(self.generators['P'])/self.s_n
        sum_load_sl = sum(self.loads[self.loads['bus'] == self.slack_bus]['P'])/self.s_n if len(self.loads) > 0 else 0  # Sum loads at slack bus

        sl_gen_idx = np.array(self.generators['bus'] == self.slack_bus)
        # The case with multiple generators at the slack bus where one or more are n_par in parallel has not been tested.
        sum_gen_sl = sum((self.p_m_setp[sl_gen_idx]*self.n_par[sl_gen_idx])[1:])  # Sum other generation at slack bus (not slack gen)
        self.p_m_setp[self.slack_generator] = self.s_0[self.get_bus_idx([self.slack_bus])].real - sum_gen_sl + sum_load_sl
        # self.p_m_setp = self.s_0[self.gen_bus_idx].real + self.p_sum_loads_bus[self.gen_bus_idx]
        self.p_m_setp[self.slack_generator] /= self.n_par[self.slack_generator]

        # Distribute reactive power equally among generators on the same bus
        self.n_gen_per_bus = np.array([sum(self.gen_bus_idx == idx) for idx in np.arange(self.n_bus)])  # Not counting identical generators (given by N_par-parameter)!
        self.q_g = (self.s_0.imag + self.q_sum_loads_bus)[self.gen_bus_idx]/self.n_gen_per_bus[self.gen_bus_idx]

        self.q_g /= self.n_par

        # From Load Flow
        # self.v_g_setp = abs(self.v_0[self.gen_bus_idx])
        self.v_g = self.v_0[self.gen_bus_idx]
        # self.s_g = (self.s_0 + self.p_sum_loads_bus + 1j*self.q_sum_loads_bus)[self.gen_bus_idx]
        self.s_g = (self.p_m_setp + 1j*self.q_g)
        self.i_g = np.conj(self.s_g/self.v_g)

        # Alternative 1
        # self.e_t = self.v_g + self.i_g * 1j * self.x_d_t
        # self.e_q_t = abs(self.e_t)
        # self.angle = np.angle(self.e_t)

        # Get rotor angle
        # self.e_q_tmp = self.v_g + 1j * self.x_q * self.i_g
        self.I_g = self.i_g*self.i_n[self.gen_bus_idx]/self.I_n_gen
        self.e_q_tmp = self.v_g + 1j * self.X_q * self.I_g
        # self.e_q_tmp = self.v_g + 1j * self.x_q * self.i_g  # Equivalent with above line (due to same nominal voltage).
        self.angle = np.angle(self.e_q_tmp)
        self.speed = np.zeros_like(self.angle)

        self.d = np.exp(1j*(self.angle - np.pi/2))
        self.q = np.exp(1j*self.angle)

        self.i_g_dq = self.i_g * np.exp(1j * (np.pi / 2 - self.angle))
        self.i_d = self.i_g_dq.real
        self.i_q = self.i_g_dq.imag  # q-axis leading d-axis

        self.v_g_dq = self.v_g*np.exp(1j*(np.pi/2 - self.angle))
        self.v_d = self.v_g_dq.real
        self.v_q = self.v_g_dq.imag

        self.e_q_t = self.v_q + self.x_d_t*self.i_d
        self.e_d_t = self.v_d - self.x_q_t*self.i_q
        self.e_t = self.e_q_t*self.q + self.e_d_t*self.d

        self.e_q_st = self.v_q + self.x_d_st * self.i_d
        self.e_d_st = self.v_d - self.x_q_st * self.i_q
        self.e_st = self.e_q_st * self.q + self.e_d_st * self.d

        self.e_q = self.e_q_t + self.i_d * (self.x_d - self.x_d_t)
        self.e = self.e_q * np.exp(1j * self.angle)
        self.e_q_0 = self.e_q.copy()

        # self.x0 = np.zeros((self.n_gen_states + self.n_gov_states + self.n_avr_states)* self.n_gen)
        self.x0[self.angle_idx] = np.angle(self.e_q_tmp)
        self.x0[self.e_q_t_idx] = self.e_q_t
        self.x0[self.e_d_t_idx] = self.e_d_t
        self.x0[self.e_q_st_idx] = self.e_q_st
        self.x0[self.e_d_st_idx] = self.e_d_st

        self.p_m = self.p_m_setp.copy()
        self.p_e = self.e_q_st * self.i_q + self.e_d_st*self.i_d - (self.x_d_st - self.x_q_st) * self.i_d * self.i_q
        self.P_m = self.p_m*self.s_n/self.P_n_gen
        self.P_e = self.p_e*self.s_n/self.P_n_gen

        # AVR
        # self.dyn_mdls = {**self.avr_mdls, **self.gov_mdls, **self.pss_mdls}
        for key in self.avr_mdls.keys():
            dm = self.avr_mdls[key]
            if hasattr(dm, 'initialize'):
                self.x0[dm.idx] = dm.initialize(self.e_q_0[dm.gen_idx])

        # GOV
        for key in self.gov_mdls.keys():
            dm = self.gov_mdls[key]
            if hasattr(dm, 'initialize'):
                self.x0[dm.idx] = dm.initialize(self.P_m[dm.gen_idx])

        for key in self.pss_mdls.keys():
            dm = self.pss_mdls[key]
            if hasattr(dm, 'initialize'):
                pass

    def ode_fun(self, t, x):

        self.time = t

        self.speed = x[self.speed_idx]
        self.angle = x[self.angle_idx]
        self.e_q_t = x[self.e_q_t_idx]
        self.e_d_t = x[self.e_d_t_idx]
        self.e_q_st = x[self.e_q_st_idx]
        self.e_d_st = x[self.e_d_st_idx]

        self.d = np.exp(1j * (self.angle - np.pi / 2))
        self.q = np.exp(1j * self.angle)

        self.e_st = self.e_q_st * self.q + self.e_d_st * self.d
        self.e_t = self.e_q_t*self.q + self.e_d_t*self.d
        self.e = self.e_q * self.q

        # Interfacing generators with system
        self.i_inj_d = np.zeros(self.n_bus_red, dtype=complex)
        self.i_inj_q = np.zeros(self.n_bus_red, dtype=complex)
        np.add.at(self.i_inj_d, self.gen_bus_idx_red, self.e_q_st / (1j * self.x_d_st) * self.q * self.n_par)
        np.add.at(self.i_inj_q, self.gen_bus_idx_red, self.e_d_st / (1j * self.x_q_st) * self.d * self.n_par)

        self.i_inj = self.i_inj_d + self.i_inj_q

        self.v_red = np.linalg.solve(self.y_bus_red + self.y_bus_red_mod, self.i_inj)
        self.v_g = self.v_red[self.gen_bus_idx_red]
        self.v_g_dq = self.v_g * np.exp(1j * (np.pi / 2 - self.angle))
        self.v_d = self.v_g_dq.real
        self.v_q = self.v_g_dq.imag

        self.i_g = (self.e_st - self.v_g)/(1j*self.x_d_st)
        self.i_g_dq = self.i_g * np.exp(1j * (np.pi / 2 - self.angle))
        self.i_d = self.i_g_dq.real
        self.i_q = self.i_g_dq.imag

        self.I_d = self.i_d*self.i_n[self.gen_bus_idx]/self.I_n_gen
        self.I_q = self.i_q*self.i_n[self.gen_bus_idx]/self.I_n_gen

        self.e_q_tmp = self.v_g + 1j * self.x_q * self.i_g

        self.p_e = self.e_q_st * self.i_q + self.e_d_st * self.i_d  # - (self.x_d_st - self.x_q_st) * self.i_d * self.i_q
        # self.P_e = self.e_q_st * self.I_q + self.e_d_st * self.I_d - (self.X_d_st - self.X_q_st) * self.I_d * self.I_q
        self.P_e = self.p_e*self.s_n/self.P_n_gen

        # Get updated dynamic model equations
        dx = np.zeros(self.n_states)

        # Controls
        # GOV
        self.speed_dev = x[self.speed_idx]
        for key in self.gov_mdls.keys():
            dm = self.gov_mdls[key]
            input = -self.speed_dev[dm.gen_idx]
            dx_loc, output = dm.update(x[dm.idx], input)
            dx[dm.idx] = dx_loc

            # dm.apply(self, output)
            active_mdls = dm.active
            active_gov_gen_idx = dm.gen_idx[active_mdls]
            self.P_m[active_gov_gen_idx] = output[active_mdls]
            self.p_m[active_gov_gen_idx] = (self.P_m[active_gov_gen_idx] * self.P_n_gen[active_gov_gen_idx] / self.s_n)

        # PSS
        for key in self.pss_mdls.keys():
            dm = self.pss_mdls[key]
            input = self.speed[dm.gen_idx]
            dx_loc, output = dm.update(x[dm.idx], input)
            dx[dm.idx] = dx_loc

            # dm.apply(self, output)
            active_mdls = dm.active
            self.v_pss[dm.gen_idx[active_mdls]] = output[active_mdls]

        # AVR
        self.v_g_dev = self.v_g_setp - abs(self.v_g)  # Used for validating AVR
        for key in self.avr_mdls.keys():
            dm = self.avr_mdls[key]
            input = self.v_g_setp[dm.gen_idx] - abs(self.v_g[dm.gen_idx]) + self.v_pss[dm.gen_idx]
            dx_loc, output = dm.update(x[dm.idx], input)
            dx[dm.idx] = dx_loc

            # dm.apply(self, output)
            active_mdls = dm.active
            self.e_q[dm.gen_idx[active_mdls ]] = output[active_mdls]

        # Generators
        self.p_m = self.P_m * self.P_n_gen/self.s_n
        self.t_m = self.p_m / (1 + self.speed)
        self.T_m = self.P_m / (1 + self.speed)
        for i, dm in enumerate([self.gen_mdl]):
            dx[dm.idx] = np.concatenate([
                1/(2*self.H)*(self.T_m - self.P_e - dm.par['D'] * x[dm.states['speed']]),
                x[dm.states['speed']]*2*np.pi*self.f,
                1/(dm.par['T_d0_t'])*(self.e_q + self.v_aux - x[dm.states['e_q_t']] - self.I_d * (dm.par['X_d'] - dm.par['X_d_t'])),
                1/(dm.par['T_q0_t'])*(-x[dm.states['e_d_t']] + self.I_q * (dm.par['X_q'] - dm.par['X_q_t'])),
                1/(dm.par['T_d0_st']) * (x[dm.states['e_q_t']] - x[dm.states['e_q_st']] - self.I_d * (dm.par['X_d_t'] - dm.par['X_d_st'])),
                1/(dm.par['T_q0_st']) * (x[dm.states['e_d_t']] - x[dm.states['e_d_st']] + self.I_q * (dm.par['X_q_t'] - dm.par['X_q_st'])),
            ])  # .T.flatten()

        return dx

    def network_event(self, element_type, name, action):
        # Simulate disconnection/connection of element by modifying admittance matrix
        if not element_type[-1] == 's':
            # line => lines
            # transformer => transformers
            element_type += 's'

        df = getattr(self, element_type)
        if element_type == 'lines':

            line = df[df['name'] == name]

            idx_from, idx_to, admittance, shunt = self.read_admittance_data('line', line.iloc[0])
            rows = np.array([idx_from, idx_to, idx_from, idx_to])
            cols = np.array([idx_from, idx_to, idx_to, idx_from])
            data = np.array([admittance + shunt / 2, admittance + shunt / 2, -admittance, -admittance])

            rebuild_red = not(idx_from in self.reduced_bus_idx and idx_to in self.reduced_bus_idx)

            if action == 'connect':
                sign = 1
            elif action == 'disconnect':
                sign = -1


            y_line = lil_matrix((self.n_bus,) * 2, dtype=complex)
            y_line[rows, cols] = data
            self.y_bus += sign*y_line
            # self.v_to_i_lines_rev[line.index, [idx_to, idx_from]] += sign*np.array([admittance + shunt / 2, -admittance])
            if rebuild_red:
                self.build_y_bus_red()
            else:
                idx_from_red = np.where(self.reduced_bus_idx == idx_from)[0][0]
                idx_to_red = np.where(self.reduced_bus_idx == idx_to)[0][0]
                rows_red = np.array([idx_from_red, idx_to_red, idx_from_red, idx_to_red])
                cols_red = np.array([idx_from_red, idx_to_red, idx_to_red, idx_from_red])
                y_line_red = lil_matrix((self.n_bus_red,) * 2, dtype=complex)
                y_line_red[rows_red, cols_red] = data
                self.y_bus_red += sign*y_line_red

    def apply_inputs(self, input_desc, u):
        # Function to make it easy to apply the same control action as in the linearized model.
        # Go through all variables to be changed, and set to zero. Neccessary to be able to have multiple inputs
        # controlling the same variables.
        for i, (inp_, u_) in enumerate(zip(input_desc, u)):
            for inp__ in inp_:
                var = getattr(self, inp__[0])
                index = inp__[1]
                var[index] *= 0

        for i, (inp_, u_) in enumerate(zip(input_desc, u)):
            for inp__ in inp_:
                var = getattr(self, inp__[0])
                index = inp__[1]
                gain = inp__[2] if len(inp__) == 3 else 1
                var[index] += u_ * gain

    def linearize(self, **kwargs):
        # Linearize model (modal analysis)
        lin = dps_mdl.PowerSystemModelLinearization(self)
        lin.linearize(**kwargs)
        return lin


if __name__ == '__main__':
    from scipy.integrate import RK23, RK45, solve_ivp
    import importlib

    # Load model
    import ps_models.k2a as model_data

    importlib.reload(model_data)
    model = model_data.load()
    # model['loads'] = pd.DataFrame(columns=['name', 'bus', 'P', 'Q', 'model'],
    #                               data=[['L1', 'B1', 1998, 0, 'Z']])

    eq = PowerSystemModel(model=model)
    eq.pf_max_it = 100
    eq.power_flow()
    eq.init_dyn_sim()

    t_end = 5
    x0 = eq.x0.copy()
    x0[eq.angle_idx[0]] += 1
    np.max(eq.ode_fun(0, eq.x0))

    sol = RK23(eq.ode_fun, 0, x0, t_end, max_step=20e-3)

    t = 0
    result_dict = defaultdict(list)

    while t < t_end:
        print(t)

        # Simulate next step
        result = sol.step()
        x = sol.y
        t = sol.t

        # Store result
        result_dict['Global', 't'].append(sol.t)
        [result_dict[tuple(desc)].append(state) for desc, state in zip(eq.state_desc, x)]

    index = pd.MultiIndex.from_tuples(result_dict)
    result = pd.DataFrame(result_dict, columns=index)

    fig, ax = plt.subplots(2)
    ax[0].plot(result[('Global', 't')], result.xs(key='speed', axis='columns', level=1))
    ax[1].plot(result[('Global', 't')], result.xs(key='angle', axis='columns', level=1))
