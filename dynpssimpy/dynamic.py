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
import dynpssimpy.dyn_models.gen as gen_lib
import importlib
from scipy import sparse as sp
from scipy.sparse import linalg as sp_linalg
from scipy.sparse import diags as sp_diags
from scipy.integrate import RK45

[importlib.reload(lib) for lib in [gen_lib, gov_lib, avr_lib, pss_lib]]


class PowerSystemModel:
    def __init__(self, model):

        self.use_numba = False

        # Load flow parameters
        self.pf_max_it = 10
        self.tol = 1e-8

        # Get static data: Convert lists to np.arrays
        for td in ['buses', 'lines', 'loads', 'transformers', 'shunts']:
            if td in model and len(model[td]) > 0:
                header = model[td][0]
                # Get dtype for each column
                data = model[td][1:]
                data_T = list(map(list, zip(*data)))
                col_dtypes = [np.array(col).dtype for col in data_T]

                entries = [tuple(entry) for entry in model[td][1:]]
                dtypes = [(name_, dtype_) for name_, dtype_ in zip(header, col_dtypes)]
                setattr(self, td, np.array(entries, dtype=dtypes))
            else:
                setattr(self, td, np.empty(0))

        # Get dynamic data: Convert dicts with lists to dicts with np.arrays
        for td in ['gov', 'avr', 'pss', 'generators']:
            setattr(self, td, dict())
            if td in model and len(model[td]) > 0:
                for key in model[td].keys():
                    # Get dtype for each column
                    header = model[td][key][0]
                    data = model[td][key][1:]
                    data_T = list(map(list, zip(*data)))
                    col_dtypes = [np.array(col).dtype for col in data_T]

                    entries = [tuple(entry) for entry in data]
                    dtypes = [(name_, dtype_) for name_, dtype_ in zip(header, col_dtypes)]
                    getattr(self, td)[key] = np.array(entries, dtype=dtypes)

        # Add some potentially missing fields
        for key in self.generators.keys():
            for req_attr, default in zip(['PF_n', 'N_par'], [1, 1]):
                if not req_attr in self.generators[key].dtype.names:
                    new_field = np.ones(len(self.generators[key]), dtype=[(req_attr, float)])
                    new_field[req_attr] *= default
                    self.generators[key] = dps_uf.combine_recarrays(self.generators[key], new_field)

        # Base for pu-system
        self.f = model['f']
        self.s_n = model['base_mva']  # For all buses
        self.v_n = np.array(self.buses['V_n'])
        self.z_n = self.v_n ** 2 / self.s_n
        self.i_n = self.s_n / (np.sqrt(3) * self.v_n)
        if 'slack_bus' in model:
            self.slack_bus = model['slack_bus']
        else:
            self.slack_bus = None
        self.n_bus = len(self.buses)

        # If nominal gen. voltage or apparent power is zero, change to system base
        # (Needs to be done after nominal bus voltages have been loaded (self.v_n)
        for key in self.generators.keys():
            fix_idx = self.generators[key]['V_n'] == 0
            gen_bus_idx = dps_uf.lookup_strings(self.generators[key]['bus'], self.buses['name'])
            self.generators[key]['V_n'][fix_idx] = self.buses['V_n'][gen_bus_idx][fix_idx]

            fix_idx = self.generators[key]['S_n'] == 0
            self.generators[key]['S_n'][fix_idx] = self.s_n

        # This is for backwards compatibilty only. Should have only self.gen (dict) in the future.
        self.gen = self.generators
        self.generators = self.gen['GEN']

        # Load flow data (concatenate data from all generator types)
        self.n_gen = sum([len(gen) for gen in self.gen.values()])
        self.v_g_setp = np.concatenate([gen['V'] for gen in self.gen.values()])
        self.P_g_setp = np.concatenate([gen['P'] for gen in self.gen.values()])
        self.n_par = np.concatenate([gen['N_par'] for gen in self.gen.values()])
        self.gen_bus_names = np.concatenate([gen['bus'] for gen in self.gen.values()])
        self.gen_pf_map_idx = []
        for key, gen in self.gen.items():
            [self.gen_pf_map_idx.append((key, i)) for i in range(len(gen))]
        self.gen_pf_map_idx = np.array(self.gen_pf_map_idx)
        self.gen_bus_idx = dps_uf.lookup_strings(self.gen_bus_names, self.buses['name'])

        # Remove duplicate buses to get number of unique buses with generators
        _, idx = np.unique(self.gen_bus_idx, return_index=True)
        self.gen_bus_idx_unique = self.gen_bus_idx[np.sort(idx)]
        self.n_gen_bus = len(self.gen_bus_idx_unique)
        self.reduced_bus_idx = self.gen_bus_idx_unique

        # Initialize bus admittance matrices
        self.y_bus = np.empty((self.n_bus, self.n_bus), dtype=complex)
        self.y_bus_lf = np.empty((self.n_bus, self.n_bus), dtype=complex)
        self.y_bus_lf[:] = np.nan
        self.y_bus_red_full = np.empty((self.n_gen, self.n_gen), dtype=complex)
        # self.y_bus_red_inv = np.empty((self.n_gen, self.n_gen))

        # Initialize dynamic variables
        self.v_g = np.empty(self.n_gen, dtype=complex)
        self.i_inj = np.empty(self.n_gen, dtype=complex)
        self.time = 0.0

    def get_bus_idx(self, names):
        # Get index of buses with given bus names in full system
        if isinstance(names, str):
            names = [names]

        bus_idx = -np.ones(len(names), dtype=int)
        for i, name in enumerate(names):
            idx = np.where(self.buses['name'] == name)[0]
            if len(idx) > 0:
                bus_idx[i] = idx
        return bus_idx

    def get_bus_idx_red(self, names):
        # Get index of buses with given bus names in reduced system
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
        # Reads data on specified element and returns admittances (series and shunt)
        # and indices in the admittance matrix where the admittances should be added.
        # Currently only for lines and transformers (other elements are treated directly
        # when building admittance matrices.
        buses = self.buses
        if element_type == 'line':
            line = element
            idx_from = dps_uf.lookup_strings(line['from_bus'], buses['name'])
            idx_to = dps_uf.lookup_strings(line['to_bus'], buses['name'])
            if line['unit'] in ['p.u.', 'pu', 'pu/km']:
                if 'S_n' in line.dtype.names and 'V_n' in line.dtype.names and line['S_n'] != 0 and line['V_n'] != 0:
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
            idx_from = dps_uf.lookup_strings(trafo['from_bus'], buses['name'])
            idx_to = dps_uf.lookup_strings(trafo['to_bus'], buses['name'])
            ratio_from = (trafo['ratio_from'] if not np.isnan(trafo['ratio_from']) else 1) if 'ratio_from' in trafo.dtype.names else 1
            ratio_to = (trafo['ratio_to'] if not np.isnan(trafo['ratio_to']) else 1) if 'ratio_to' in trafo.dtype.names else 1

            V_n_from = trafo['V_n_from'] if trafo['V_n_from'] else self.v_n[idx_from]
            Z_base_trafo = V_n_from ** 2 / trafo['S_n']  # <= Could also have used _to instead of _from
            impedance = (trafo['R'] + 1j * trafo['X']) * Z_base_trafo / self.z_n[idx_from]
            n_par = trafo['N_par'] if 'N_par' in trafo.dtype.names else 1
            admittance = n_par / impedance

            return idx_from, idx_to, admittance, ratio_from, ratio_to

    def build_y_branch(self):
        # Build matrices for easy computation of branch (line and trafo) currents.
        # E.g. i_lines = self.v_to_i_lines*v_bus (full system, not reduced).
        # NB: Is not necessarily updated when y_bus is updated.

        # Lines:
        n_elements = len(self.lines)
        self.v_to_i_lines = np.zeros((n_elements, self.n_bus), dtype=complex)
        self.v_to_i_lines_rev = np.zeros((n_elements, self.n_bus), dtype=complex)
        self.lines_from_mat = np.zeros((n_elements, self.n_bus), dtype=complex)
        self.lines_to_mat = np.zeros((n_elements, self.n_bus), dtype=complex)
        for i, element in enumerate(self.lines):
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
        for i, element in enumerate(self.transformers):
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

        n_bus = self.n_bus
        y_branch = np.zeros((n_bus, n_bus), dtype=complex)  # Branches = Trafos + Lines
        buses = self.buses

        y_lines = np.zeros((n_bus, n_bus), dtype=complex)

        for i, line in enumerate(self.lines):
            idx_from, idx_to, admittance, shunt = self.read_admittance_data('line', line)
            rows = np.array([idx_from, idx_to, idx_from, idx_to])
            cols = np.array([idx_from, idx_to, idx_to, idx_from])
            data = np.array(
                [admittance + shunt / 2, admittance + shunt / 2, -admittance, -admittance])
            y_lines[rows, cols] += data

        y_branch += y_lines

        y_trafo = np.zeros((n_bus, n_bus), dtype=complex)
        for i, trafo in enumerate(self.transformers):

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
        for key, data in self.gen.items():
            for gen in data:
                idx_bus = dps_uf.lookup_strings(gen['bus'], buses['name'])
                V_n = gen['V_n']
                impedance = 1j * gen['X_d_st'] * V_n ** 2 / gen['S_n'] / self.z_n[idx_bus]
                y_gen[idx_bus, idx_bus] += gen['N_par'] / impedance

        y_load = np.zeros((n_bus, n_bus), dtype=complex)
        if type == 'dyn':
            for i, load in enumerate(self.loads):
                s_load = (load['P'] + 1j * load['Q']) / self.s_n
                if load['model'] == 'Z' and abs(s_load) > 0:
                    # idx_bus = buses[buses['name'] == load['bus']].index
                    idx_bus = dps_uf.lookup_strings(load['bus'], buses['name'])
                    z = np.conj(abs(self.v_0[idx_bus])**2/s_load)
                    y_load[idx_bus, idx_bus] += 1/z

        y_shunt = np.zeros((n_bus, n_bus), dtype=complex)
        for i, shunt in enumerate(self.shunts):
            # if shunt['model'] == 'Z':
            # idx_bus = buses[buses['name'] == shunt['bus']].index
            idx_bus = dps_uf.lookup_strings(shunt['bus'], buses['name'])
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
            keep_extra_buses_idx = dps_uf.lookup_strings(keep_extra_buses, self.buses['name'])

            self.reduced_bus_idx = np.concatenate([self.gen_bus_idx, np.array(keep_extra_buses_idx, dtype=int)])

            # Remove duplicate buses
            _, idx = np.unique(self.reduced_bus_idx, return_index=True)
            self.reduced_bus_idx = self.reduced_bus_idx[np.sort(idx)]

        self.n_bus_red = len(self.reduced_bus_idx)
        self.y_bus_red_full = self.kron_reduction(self.y_bus, self.reduced_bus_idx)  # np.empty((self.n_gen, self.n_gen))

        self.gen_bus_idx_red = self.get_bus_idx_red(self.buses[self.gen_bus_idx]['name'])

        self.y_bus_red = sp.csr_matrix(self.y_bus_red_full)
        self.y_bus_red_mod = self.y_bus_red*0

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

        # Determine bus types and active and reactive load and generation
        for i, bus in enumerate(self.buses):
            gen_idx = self.gen_bus_idx == i
            sum_gen = sum(self.P_g_setp[gen_idx]*self.n_par[gen_idx])

            p_sum_loads_bus[i] = (sum(self.loads[self.loads['bus'] == bus['name']]['P']) / self.s_n) if len(self.loads) > 0 else 0
            p_sum_bus[i] = p_sum_loads_bus[i] - sum_gen / self.s_n
            q_sum_loads_bus[i] = (sum(self.loads[self.loads['bus'] == bus['name']]['Q']) / self.s_n) if len(self.loads) > 0 else 0
            q_sum_bus[i] = q_sum_loads_bus[i]  # - (sum(self.shunts[self.shunts['bus'] == bus['name']]['Q'])) / self.s_n
            if any(i == self.gen_bus_idx):
                bus_type[i] = 'PV'
            else:
                bus_type[i] = 'PQ'

        if self.slack_bus:
            bus_type[self.buses['name'] == self.slack_bus] = 'SL'
        else:
            bus_type[np.argmax(bus_type == 'PV')] = 'SL'  # Set the first generator as slack bus

        # Indices of PV, PQ, PV+PQ and SL-buses
        pv_idx = np.where(bus_type == 'PV')[0]
        pq_idx = np.where(bus_type == 'PQ')[0]
        pvpq_idx = np.concatenate([pv_idx, pq_idx])
        sl_idx = np.where(bus_type == 'SL')
        self.slack_bus = self.buses[sl_idx]['name'][0]

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
        v_0 = np.ones(self.n_bus)
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

        while not converged and i < self.pf_max_it:
            i = i + 1

            # Numerical jacobian
            J = dps_uf.jacobian_num(pf_equations, x)

            # Update step
            dx = np.linalg.solve(J, err)
            x -= dx

            err = pf_equations(x)
            err_norm = max(abs(err))

            if self.tol > err_norm:
                converged = True
                if print_output:
                    print('Power flow converged.')
            if i == self.pf_max_it and print_output:
                print('Power flow did not converge in {} iterations.'.format(self.pf_max_it))

        self.v_0 = x_to_v(x)
        self.s_0 = self.v_0 * np.conj(y_bus.dot(self.v_0))
        self.p_sum_loads_bus = p_sum_loads_bus
        self.q_sum_loads_bus = q_sum_loads_bus

    def init_dyn_sim(self):

        # Build reduced system
        self.y_bus = self.build_y_bus()
        self.build_y_bus_red()
        self.v_red = self.v_0[self.reduced_bus_idx]
        self.i_inj = np.zeros(len(self.reduced_bus_idx), dtype=complex)
        self.s_inj = np.zeros(len(self.reduced_bus_idx), dtype=complex)

        # State variables:
        self.state_desc = np.empty((0, 2))

        # Containers for dynamic models
        self.gen_mdls = dict()
        self.gov_mdls = dict()
        self.avr_mdls = dict()
        self.pss_mdls = dict()

        for i, (input_data, container, library) in enumerate(zip([self.gen, self.gov, self.pss, self.avr],
                                                                 [self.gen_mdls, self.gov_mdls, self.pss_mdls, self.avr_mdls],
                                                                 [gen_lib, gov_lib, pss_lib, avr_lib])):

            for key in input_data.keys():
                data = input_data[key].copy()
                start_idx = len(self.state_desc)
                mdl = getattr(library, key)()
                state_list = mdl.state_list
                names = data['name']
                n_units = len(data)
                n_states = len(state_list)

                # state_desc_mdl = np.vstack([np.tile(names, n_states), np.repeat(state_list, n_units)]).T
                state_desc_mdl = np.vstack([np.repeat(names, n_states), np.tile(state_list, n_units)]).T

                mdl.n_units = n_units
                # Global indices of model states in global state vector.
                # (x_global[mdl.idx] = x_local)
                mdl.idx = slice(start_idx, start_idx + len(state_desc_mdl))  # Indices of all states belonging to model
                # dtypes-attribute is defined to allow view to be created easily (with named fields)
                # Allows syntax x['speed'] instead of x[state_idx['speed']]
                # within differential equation functions of dyn. models.
                mdl.dtypes = [(state, np.float) for state in state_list]
                mdl.shape = (n_states, n_units)
                mdl.par = data
                mdl.int_par = np.zeros(n_units, [(par, np.float) for par in mdl.int_par_list])
                mdl.input = np.zeros(n_units, [(field, np.float) for field in mdl.input_list])
                mdl.output = np.zeros(n_units, [(field, np.float) for field in mdl.output_list])

                # JIT-compilation using Numba
                compile_these = ['_update', '_current_injections']
                if self.use_numba:
                    from numba import jit  # Is this OK?
                    for fun in compile_these:
                        if hasattr(mdl, fun):
                            setattr(mdl, fun[1:], jit()(getattr(mdl, fun)))
                else:
                    for fun in compile_these:
                        if hasattr(mdl, fun):
                            setattr(mdl, fun[1:], getattr(mdl, fun))

                if i == 0:  # Do this for generators only
                    mdl.bus_idx = dps_uf.lookup_strings(data['bus'], self.buses['name'])
                    mdl.bus_idx_red = self.get_bus_idx_red(data['bus'])
                    mdl.int_par['f'] = self.f

                else:  # Do this for control models only
                    mdl.active = np.ones(len(data), dtype=bool)

                    # Determine which generators are controlled (model and indices).
                    # This information is stored as a dict with keys corresponding to generator types,
                    # and values with lists: [mask, idx], where:
                    # mask (np.array, bool): Boolean mask to select controls that act on the particular generator type
                    # idx (np.array, int): Points to which generators of this type are controlled
                    mdl.gen_idx = dict()
                    for gen_key in self.gen.keys():
                        lookup, mask = dps_uf.lookup_strings(data['gen'], self.gen[gen_key]['name'], return_mask=True)
                        if len(lookup) > 0:
                            mdl.gen_idx[gen_key] = [mask, lookup]

                # Local indices of states (starting on zero, ending on number of states of model)
                mdl.state_idx = np.zeros((n_units,), dtype=[(state, int) for state in state_list])
                for state in state_list:
                    mdl.state_idx[state] = np.where(state_desc_mdl[:, 1] == state)[0]

                container[key] = mdl
                self.state_desc = np.vstack([self.state_desc, state_desc_mdl])

        self.n_states = self.state_desc.shape[0]
        self.state_desc_der = self.state_desc.copy()
        self.state_desc_der[:, 1] = np.char.add(np.array(self.n_states * ['d_']), self.state_desc[:, 1])

        # Choose first generator at slack bus as slack generator
        for gen_key, gen_mdl in self.gen_mdls.items():
            lookup = dps_uf.lookup_strings(self.slack_bus, gen_mdl.par['bus'])
            if not np.isnan(lookup) > 0:
                self.slack_generator_id = (gen_key, lookup)
                break

        self.slack_generator = np.where((self.gen_pf_map_idx == self.slack_generator_id).all(axis=1))[0][0]

        self.p_g_setp = self.P_g_setp / self.s_n  # System base
        sum_load_sl = sum(self.loads[self.loads['bus'] == self.slack_bus]['P'])/self.s_n if len(self.loads) > 0 else 0  # Sum loads at slack bus
        sl_gen_idx = np.array(self.gen_bus_names == self.slack_bus)

        # The case with multiple generators at the slack bus where one or more are n_par in parallel has not been tested.
        sum_gen_sl = sum((self.p_g_setp[sl_gen_idx] * self.n_par[sl_gen_idx])[1:])  # Sum other generation at slack bus (not slack gen)
        self.p_g_setp[self.slack_generator] = self.s_0[self.get_bus_idx([self.slack_bus])].real - sum_gen_sl + sum_load_sl
        self.p_g_setp[self.slack_generator] /= self.n_par[self.slack_generator]

        # Distribute reactive power equally among generators on the same bus
        self.n_gen_per_bus = np.array([sum(self.gen_bus_idx == idx) for idx in np.arange(self.n_bus)])  # Not counting identical generators (given by N_par-parameter)!
        self.q_g = (self.s_0.imag + self.q_sum_loads_bus)[self.gen_bus_idx]/self.n_gen_per_bus[self.gen_bus_idx]
        self.q_g /= self.n_par

        # From Load Flow
        self.v_g = self.v_0[self.gen_bus_idx]
        self.s_g = (self.p_g_setp + 1j * self.q_g)

        # Initialize global state vector
        self.x0 = np.zeros(self.n_states)

        # Generators
        for key, dm in self.gen_mdls.items():
            if hasattr(dm, 'initialize'):
                # Get data from power flow solution
                dm_pf_map_idx = self.gen_pf_map_idx[:, 0] == key
                v_g = self.v_g[dm_pf_map_idx]
                s_g = self.s_g[dm_pf_map_idx]*self.s_n/dm.par['S_n']
                # Convert complex quantities to float
                dm.input['V_t_abs'] = np.abs(v_g)
                dm.input['V_t_angle'] = np.angle(v_g)
                dm.output['P_e'] = s_g.real
                dm.output['Q'] = s_g.imag
                dm.initialize(
                    self.x0[dm.idx].view(dtype=dm.dtypes),
                    dm.input, dm.output, dm.par, dm.int_par)

        # self.dyn_mdls = {**self.avr_mdls, **self.gov_mdls, **self.pss_mdls}

        # AVR
        for key, dm in self.avr_mdls.items():
            if hasattr(dm, 'initialize'):
                for gen_key, (mask, idx) in dm.gen_idx.items():
                    gen_mdl = self.gen_mdls[gen_key]
                    dm.output['E_f'][mask] = gen_mdl.input['E_f'][idx]

                # for gen_key in dm.gen_mdl_list:
                #     e_q_0[dm.gen_mdl_3 == gen_key] = self.gen_mdls[gen_key].e_q[dm.gen_idx_2[gen_key]]

                dm.initialize(
                    self.x0[dm.idx].view(dtype=dm.dtypes),
                    dm.input, dm.output, dm.par, dm.int_par)

        # GOV
        for key, dm in self.gov_mdls.items():
            if hasattr(dm, 'initialize'):
                for gen_key, (mask, idx) in dm.gen_idx.items():
                    # mask: Boolean mask to map controls to generators
                    # idx: Points to which generators are controlled
                    gen_mdl = self.gen_mdls[gen_key]
                    dm.output['P_m'][mask] = gen_mdl.input['P_m'][idx]


                # for gen_key in dm.gen_mdl_list:
                #     P_m_0[dm.gen_mdl_3 == gen_key] = self.gen_mdls[gen_key].P_m[dm.gen_idx_2[gen_key]]

                dm.initialize(
                    self.x0[dm.idx].view(dtype=dm.dtypes),
                    dm.input, dm.output, dm.par, dm.int_par)

        # PSS
        for key, dm in self.pss_mdls.items():
            if hasattr(dm, 'initialize'):
                pass

    def ode_fun(self, t, x):

        self.time = t

        # Interfacing models  with system (current injections)
        self.i_inj_d = np.zeros(self.n_bus_red, dtype=complex)
        self.i_inj_q = np.zeros(self.n_bus_red, dtype=complex)
        for key, dm in self.gen_mdls.items():
            I_n = dm.par['S_n'] / (np.sqrt(3) * dm.par['V_n'])
            i_inj_d_mdl, i_inj_q_mdl = dm.current_injections(
                x[dm.idx].view(dtype=dm.dtypes),
                dm.par)*I_n/self.i_n[dm.bus_idx]
            np.add.at(self.i_inj_d, dm.bus_idx_red, i_inj_d_mdl)
            np.add.at(self.i_inj_q, dm.bus_idx_red, i_inj_q_mdl)

        self.i_inj[:] = self.i_inj_d + self.i_inj_q

        # Solve network equations
        # self.v_red = sp_linalg.spsolve(self.y_bus_red + self.y_bus_red_mod, self.i_inj)
        v_red = self.v_red.copy()
        self.i_inj = self.i_inj_d + self.i_inj_q

        tol = 1e-10
        max_it = 10
        error = 10 * tol
        it = 0

        # v_red *= 0.1
        y_bus = self.y_bus_red + self.y_bus_red_mod
        t_tot = time.time()
        t_spsolve_cum0 = 0
        while error > tol and it < max_it:
            s_v2_diag = np.conj(sp_diags(self.s_inj/v_red**2))
            A = y_bus + s_v2_diag
            b = np.conj(self.s_inj/v_red) + self.i_inj + s_v2_diag*v_red
            v_red = sp_linalg.spsolve(A, b)
            error = np.linalg.norm(y_bus.dot(v_red) - np.conj(self.s_inj/v_red) - self.i_inj)
            it += 1

        if error > tol:
            print('Warning: Solution of algebraic equations did not converge.')

        self.v_red[:] = v_red

        self.v_g = self.v_red[self.gen_bus_idx_red]

        # Calculate state derivatives
        dx = np.zeros(self.n_states)

        # GOV
        # self.speed_dev = x[self.gen_mdls['GEN'].state_idx['speed']]
        for key, dm in self.gov_mdls.items():
            input = np.zeros(dm.n_units, dtype=float)
            for gen_key, (mask, idx) in dm.gen_idx.items():
                # mask: Boolean mask to map controls to generators
                # idx: Points to which generators are controlled
                gen_mdl = self.gen_mdls[gen_key]
                x_loc = x[gen_mdl.idx]
                input[mask] = -x_loc[gen_mdl.state_idx['speed'][idx]]
                dm.input['speed_dev'][mask] = -x_loc[gen_mdl.state_idx['speed'][idx]]

            dm.update(
                dx[dm.idx].view(dtype=dm.dtypes),
                x[dm.idx].view(dtype=dm.dtypes),
                dm.input, dm.output, dm.par, dm.int_par)

            for gen_key, (mask, idx) in dm.gen_idx.items():
                gen_mdl = self.gen_mdls[gen_key]
                gen_mdl.input['P_m'][idx[dm.active[mask]]] = dm.output['P_m'][dm.active & mask]

            # self.P_m[active_gov_gen_idx] = output[active_mdls]
            # self.p_m[active_gov_gen_idx] = (self.P_m[active_gov_gen_idx] * self.P_n_gen[active_gov_gen_idx] / self.s_n)

        # PSS
        for key, dm in self.pss_mdls.items():
            input = np.zeros(dm.n_units, dtype=float)
            for gen_key, (mask, idx) in dm.gen_idx.items():
                gen_mdl = self.gen_mdls[gen_key]
                x_loc = x[gen_mdl.idx]
                input[mask] = x_loc[gen_mdl.state_idx['speed'][idx]]
                dm.input['speed'][mask] = x_loc[gen_mdl.state_idx['speed'][idx]]

            dm.update(
                dx[dm.idx].view(dtype=dm.dtypes),
                x[dm.idx].view(dtype=dm.dtypes),
                dm.input, dm.output, dm.par, dm.int_par)

            for gen_key, (mask, idx) in dm.gen_idx.items():
                gen_mdl = self.gen_mdls[gen_key]
                gen_mdl.input['v_pss'][idx[dm.active[mask]]] = dm.output['v_pss'][dm.active & mask]

        # AVR
        for key, dm in self.avr_mdls.items():
            input = np.zeros(dm.n_units, dtype=float)
            for gen_key, (mask, idx) in dm.gen_idx.items():
                gen_mdl = self.gen_mdls[gen_key]
                v_g_setp = gen_mdl.par['V'][idx]
                v_g = self.v_red[gen_mdl.bus_idx_red][idx]

                dm.input['v_dev'][mask] = v_g_setp - abs(v_g)
                dm.input['v_pss'][mask] = gen_mdl.input['v_pss'][idx]

            dm.update(
                dx[dm.idx].view(dtype=dm.dtypes),
                x[dm.idx].view(dtype=dm.dtypes),
                dm.input, dm.output, dm.par, dm.int_par)
            for gen_key, (mask, idx) in dm.gen_idx.items():
                gen_mdl = self.gen_mdls[gen_key]
                gen_mdl.input['E_f'][idx[dm.active[mask]]] = dm.output['E_f'][dm.active & mask]

        # Generators
        for key, dm in self.gen_mdls.items():
            # Potential p.u. issue here? v_red in system base, V_t in generator base.
            v_g = self.v_red[dm.bus_idx_red]
            dm.input['V_t_abs'] = np.abs(v_g)
            dm.input['V_t_angle'] = np.angle(v_g)
            dm.update(
                dx[dm.idx].view(dtype=dm.dtypes),
                x[dm.idx].view(dtype=dm.dtypes),
                dm.input, dm.output, dm.par, dm.int_par)

        return dx

    def network_event(self, element_type, name, action):
        # Simulate disconnection/connection of element by modifying admittance matrix
        if not element_type[-1] == 's':
            # line => lines
            # transformer => transformers
            element_type += 's'

        df = getattr(self, element_type)
        if element_type == 'lines':


            line = df[dps_uf.lookup_strings(name, df['name'])]

            idx_from, idx_to, admittance, shunt = self.read_admittance_data('line', line)
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
        # NB: Experimental
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

    ps = PowerSystemModel(model=model)
    ps.pf_max_it = 100
    ps.power_flow()
    ps.init_dyn_sim()

    t_end = 5
    x0 = ps.x0.copy()
    x0[ps.gen_mdls['GEN'].state_idx['angle'][0]] += 1
    np.max(ps.ode_fun(0, ps.x0))

    sol = RK45(ps.ode_fun, 0, x0, t_end, max_step=10e-3)

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
        [result_dict[tuple(desc)].append(state) for desc, state in zip(ps.state_desc, x)]

    index = pd.MultiIndex.from_tuples(result_dict)
    result = pd.DataFrame(result_dict, columns=index)

    fig, ax = plt.subplots(2)
    ax[0].plot(result[('Global', 't')], result.xs(key='speed', axis='columns', level=1))
    ax[1].plot(result[('Global', 't')], result.xs(key='angle', axis='columns', level=1))

    plt.show()