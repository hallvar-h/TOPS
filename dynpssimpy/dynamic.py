from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time


def jacobian_num(f, x, eps=1e-10, **params):

    J = np.zeros([len(x), len(x)], dtype=np.float)

    for i in range(len(x)):
        x1 = x.copy()
        x2 = x.copy()

        x1[i] += eps
        x2[i] -= eps

        f1 = f(x1, **params)
        f2 = f(x2, **params)

        J[:, i] = (f1 - f2) / (2 * eps)

    return J


class DynamicModel:
    def __init__(self):
        pass


class PowerSystemModel:
    def __init__(self, model):

        # Load flow parameters
        self.pf_max_it = 10
        self.tol = 1e-8

        # Get model data
        for td in ['buses', 'lines', 'loads', 'generators', 'transformers', 'shunts']:
            # print(td)
            if td in model:
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

        # Simulation variables (updated every simulation step)

        self.n_gen_states = 6
        self.speed_idx = 0 + self.n_gen_states * np.arange(self.n_gen)
        self.angle_idx = 1 + self.n_gen_states * np.arange(self.n_gen)
        self.e_q_t_idx = 2 + self.n_gen_states * np.arange(self.n_gen)
        self.e_d_t_idx = 3 + self.n_gen_states * np.arange(self.n_gen)
        self.e_q_st_idx = 4 + self.n_gen_states * np.arange(self.n_gen)
        self.e_d_st_idx = 5 + self.n_gen_states * np.arange(self.n_gen)


        self.n_gov_states = 1
        self.gov_state_idx = self.n_gen_states*self.n_gen + self.n_gov_states * np.arange(self.n_gen)

        self.n_avr_states = 2
        self.avr_state_x_idx = 0 + (self.n_gov_states + self.n_gen_states)*self.n_gen + self.n_avr_states * np.arange(self.n_gen)
        self.avr_state_ef_idx = 1 + (self.n_gov_states + self.n_gen_states)*self.n_gen + self.n_avr_states * np.arange(self.n_gen)

        self.v_g = np.empty(self.n_gen, dtype=complex)
        self.i_inj = np.empty(self.n_gen, dtype=complex)
        self.i_g = np.empty(self.n_gen, dtype=complex)
        self.angle = np.empty(self.n_gen, dtype=complex)
        self.v_pss = np.zeros(self.n_gen, dtype=float)
        self.v_aux = np.zeros(self.n_gen, dtype=float)

        self.gen_bus_idx = [self.buses[self.buses['name'] == gen['bus']].index[0] for i, gen in self.generators.iterrows()]
        self.gen_bus_idx_unique = np.unique(self.gen_bus_idx)
        self.n_gen_bus = len(self.gen_bus_idx_unique)

        # Get gen nom. voltage. If given as zero in input data, select bus nom. voltage.
        self.V_n_gen = np.zeros(len(self.generators))
        self.S_n_gen = np.zeros(len(self.generators))
        for i, gen in self.generators.iterrows():
            self.V_n_gen[i] = gen['V_n'] if gen['V_n'] else self.v_n[self.gen_bus_idx[i]]
            self.S_n_gen[i] = gen['S_n'] if gen['S_n'] else self.s_n
        self.I_n_gen = self.S_n_gen / (np.sqrt(3) * self.V_n_gen)
        self.Z_n_gen = self.V_n_gen ** 2 / self.S_n_gen

        # Generator parameters, p.u. on system base (lower case letters) and p.u. on generator base (upper case letters)
        for par in ['X_d', 'X_q', 'X_d_t', 'X_q_t', 'X_d_st', 'X_q_st']:
            setattr(self, par, np.array(self.generators[par]))
            setattr(self, par.lower(), np.array(self.generators[par])*self.Z_n_gen/self.z_n[self.gen_bus_idx])

        for par in ['T_d0_t', 'T_q0_t', 'T_d0_st', 'T_q0_st']:
            setattr(self, par, np.array(self.generators[par]))

        self.n_par = np.array(self.generators['N_par']) if 'N_par' in self.generators else np.ones(self.n_gen)

        self.reduced_bus_idx = np.empty(0)
        self.y_bus = np.empty((self.n_bus, self.n_bus))
        self.y_bus_red = np.empty((self.n_gen, self.n_gen))
        self.y_bus_red_inv = np.empty((self.n_gen, self.n_gen))
        self.y_bus_red_mod = np.zeros_like(self.y_bus_red)

    def get_bus_idx(self, names):
        return pd.concat([self.buses[self.buses['name'] == bus] for bus in names])

    def get_bus_idx_red(self, names):
        # TODO: Currently returns zero for buses not in reduced system. Could return None? Or -1?
        sorter = np.argsort(self.reduced_bus_idx)
        return sorter[np.searchsorted(self.reduced_bus_idx, self.get_bus_idx(names).index, sorter=sorter)]

    def kron_reduction(self, y_bus, keep_buses):
        remove_buses = list(set(range(self.n_bus)) - set(keep_buses))
        y_rr = y_bus[np.ix_(remove_buses, remove_buses)]
        y_rk = y_bus[np.ix_(remove_buses, keep_buses)]
        y_kk = y_bus[np.ix_(keep_buses, keep_buses)]

        return y_kk - y_rk.T.dot(np.linalg.inv(y_rr)).dot(y_rk)

    def build_y_bus(self, type='dyn', y_ext=np.empty((0, 0))):
        n_bus = len(self.buses)

        Y_branch = np.zeros((n_bus, n_bus), dtype=complex)  # Branches = Trafos + Lines
        buses = self.buses
        for i, line in self.lines.iterrows():
            idx_from = buses[buses['name'] == line['from_bus']].index
            idx_to = buses[buses['name'] == line['to_bus']].index
            if line['unit'] in ['p.u.', 'pu', 'pu/km']:
                if 'S_n' in line and 'V_n' in line:
                    if line['S_n'] != 0 and line['V_n'] != 0:
                        # If impedance given in p.u./km
                        impedance = (line['R'] + 1j*line['X'])*line['length']*line['V_n']**2/line['S_n']/self.z_n[idx_from]
                        shunt = 1j*line['B']*line['length']*1/(line['V_n']**2/line['S_n']/self.z_n[idx_from])
                    else:
                        # Per unit of system base and bus nominal voltage
                        impedance = (line['R'] + 1j * line['X']) * line['length']
                        shunt = 1j * line['B'] * line['length']
                else:
                    # TODO: Same as above.. Messy programming. Tidy up.
                    # Per unit of system base and bus nominal voltage
                    impedance = (line['R'] + 1j * line['X']) * line['length']
                    shunt = 1j * line['B'] * line['length']
            elif line['unit'] in ['PF', 'pf', 'PowerFactory', 'powerfactory']:
                # Given in ohm/km, but with capacitance in micro-Siemens
                impedance = (line['R'] + 1j * line['X']) * line['length'] / self.z_n[idx_from]
                shunt = 1j * line['B'] * line['length'] * self.z_n[idx_from]*1e-6
            elif line['unit'] in ['Ohm', 'ohm']:
                # Given in Ohm/km
                impedance = (line['R'] + 1j * line['X']) * line['length']/self.z_n[idx_from]
                shunt = 1j*line['B'] * line['length'] * self.z_n[idx_from]

            Y_branch[idx_from, idx_from] += 1 / impedance + shunt/2
            Y_branch[idx_to, idx_to] += 1 / impedance + shunt/2
            Y_branch[idx_from, idx_to] -= 1 / impedance
            Y_branch[idx_to, idx_from] -= 1 / impedance

        for i, trafo in self.transformers.iterrows():

            idx_from = buses[buses['name'] == trafo['from_bus']].index
            idx_to = buses[buses['name'] == trafo['to_bus']].index
            ratio_from = (trafo['ratio_from'] if not np.isnan(trafo['ratio_from']) else 1) if 'ratio_from' in trafo else 1
            ratio_to = (trafo['ratio_to'] if not np.isnan(trafo['ratio_to']) else 1) if 'ratio_to' in trafo else 1

            V_n_from = trafo['V_n_from'] if trafo['V_n_from'] else self.v_n[idx_from]
            Z_base_trafo = V_n_from**2/trafo['S_n']  # <= Could also have used _to instead of _from
            impedance = (trafo['R'] + 1j*trafo['X'])*Z_base_trafo/self.z_n[idx_from]
            n_par = trafo['N_par'] if 'N_par' in trafo else 1
            admittance = n_par/impedance
            Y_branch[idx_from, idx_from] += ratio_from*np.conj(ratio_from)*admittance
            Y_branch[idx_to, idx_to] += ratio_to*np.conj(ratio_to)*admittance
            Y_branch[idx_from, idx_to] -= ratio_from*np.conj(ratio_to)*admittance
            Y_branch[idx_to, idx_from] -= np.conj(ratio_from)*ratio_to*admittance
            # Y_branch[idx_from, idx_from] += ratio*np.conj(ratio)*admittance
            # Y_branch[idx_to, idx_to] += admittance
            # Y_branch[idx_from, idx_to] -= ratio*admittance
            # Y_branch[idx_to, idx_from] -= np.conj(ratio)*admittance

        Y_gen = np.zeros((n_bus, n_bus), dtype=complex)
        for i, gen in self.generators.iterrows():
            # Generator impedance on system base
            idx_bus = buses[buses['name'] == gen['bus']].index
            V_n = gen['V_n'] if gen['V_n'] else self.v_n[idx_bus]
            impedance = 1j * gen['X_d_st'] * V_n**2/gen['S_n']/self.z_n[self.gen_bus_idx[i]]
            Y_gen[idx_bus, idx_bus] += self.n_par[i] / impedance

        Y_load = np.zeros((n_bus, n_bus), dtype=complex)
        if type=='dyn':
            for i, load in self.loads.iterrows():
                s_load = (load['P'] + 1j * load['Q']) / self.s_n
                if load['model'] == 'Z' and abs(s_load) > 0:
                    idx_bus = buses[buses['name'] == load['bus']].index
                    z = np.conj(abs(self.v_0[idx_bus])**2/s_load)
                    Y_load[idx_bus, idx_bus] += 1/z

        Y_shunt = np.zeros((n_bus, n_bus), dtype=complex)
        for i, shunt in self.shunts.iterrows():
            if shunt['model'] == 'Z':
                idx_bus = buses[buses['name'] == shunt['bus']].index
                s_shunt = -1j*shunt['Q']/self.s_n

                z = np.conj(abs(1)**2/s_shunt)
                Y_shunt[idx_bus, idx_bus] += 1/z

        self.y_gen = Y_gen
        self.y_branch = Y_branch
        self.y_load = Y_load
        self.y_shunt = Y_shunt
        self.y_ext = y_ext

        Y = Y_branch.copy()
        Y += Y_shunt
        if type == 'dyn':
            Y += Y_gen + Y_load
        if y_ext.shape[0] > 0:
            Y += y_ext
        return Y

    def build_y_bus_red(self, keep_extra_buses=[]):
        keep_extra_buses_idx = [self.buses[self.buses.name == name].index[0] for name in keep_extra_buses]
        self.reduced_bus_idx = np.concatenate([self.gen_bus_idx, np.array(keep_extra_buses_idx, dtype=int)])

        # Remove duplicate buses
        _, idx = np.unique(self.reduced_bus_idx, return_index=True)
        self.reduced_bus_idx = self.reduced_bus_idx[np.sort(idx)]

        self.y_bus_red = self.kron_reduction(self.y_bus, self.reduced_bus_idx)  # np.empty((self.n_gen, self.n_gen))
        self.y_bus_red_mod = np.zeros_like(self.y_bus_red)

        self.n_bus_red = self.y_bus_red.shape[0]
        self.gen_bus_idx_red = self.get_bus_idx_red(self.buses.iloc[self.gen_bus_idx]['name'])

    def power_flow(self):

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

        # Build admittance matrix
        y_bus = self.build_y_bus(type='lf')
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
            J = jacobian_num(pf_equations, x)

            # Update step
            dx = np.linalg.solve(J, err)
            x -= dx

            err = pf_equations(x)
            err_norm = max(abs(err))
            print(err_norm)

            if self.tol > err_norm:
                converged = True
                print('Power flow converged.')
            if i == self.pf_max_it:
                print('Power flow did not converge in {} iterations.'.format(self.pf_max_it))

        # soln = newton(pf_equations, x, full_output=True, maxiter=20000, disp=True)
        # pf_equations(soln.root)

        self.v_0 = x_to_v(x)
        self.s_0 = self.v_0 * np.conj(y_bus.dot(self.v_0))
        self.p_sum_loads_bus = p_sum_loads_bus
        self.q_sum_loads_bus = q_sum_loads_bus

    def init_dyn_sim(self):

        # Build bus admittance matrices
        self.y_bus = self.build_y_bus()
        self.build_y_bus_red()

        # State variables:
        self.state_desc = np.empty((0, 2))
        self.gen_mdls = []
        for i, data in self.generators.iterrows():
            states = ['speed', 'angle', 'e_q_t', 'e_d_t', 'e_q_st', 'e_d_st']
            state_desc = np.vstack([[data['name'], ] * len(states), states]).T
            self.state_desc = np.vstack([self.state_desc, state_desc])

            mdl = DynamicModel()
            mdl.idx = np.where(self.state_desc[:, 0] == data['name'])[0]
            mdl.states = dict(zip(self.state_desc[mdl.idx, 1], mdl.idx))
            mdl.par = data
            self.gen_mdls.append(mdl)

        self.gov_mdls = []
        for key in self.gov.keys():
            for i, data in self.gov[key].iterrows():
                states = ['x_1', 'x_2']
                state_desc = np.vstack([[data['name'], ] * len(states), states]).T
                self.state_desc = np.vstack([self.state_desc, state_desc])

                mdl = DynamicModel()
                mdl.name = key
                mdl.idx = np.where(self.state_desc[:, 0] == data['name'])[0]
                mdl.states = dict(zip(self.state_desc[mdl.idx, 1], mdl.idx))
                mdl.par = data
                mdl.int_par = dict(x_1_bias=0)
                mdl.gen_idx = self.generators[self.generators['name'] == data['gen']].index.tolist()[0]
                mdl.active = True
                self.gov_mdls.append(mdl)

        self.avr_mdls = []
        for key in self.avr.keys():
            for i, data in self.avr[key].iterrows():
                states = ['x', 'e_f']
                state_desc = np.vstack([[data['name'], ] * len(states), states]).T
                self.state_desc = np.vstack([self.state_desc, state_desc])

                mdl = DynamicModel()
                mdl.name = key
                mdl.idx = np.where(self.state_desc[:, 0] == data['name'])[0]
                mdl.states = dict(zip(self.state_desc[mdl.idx, 1], mdl.idx))
                mdl.par = data
                mdl.int_par = dict(x_bias=0)  # This parameter is necessary to give consistent initial conditions. Without
                # it there would have to be a steady state deviation in the voltage at t=0, since there is no integrator.
                mdl.gen_idx = self.generators[self.generators['name'] == data['gen']].index.tolist()[0]
                mdl.active = True
                self.avr_mdls.append(mdl)

        self.pss_mdls = []
        for key in self.pss.keys():
            for i, data in self.pss[key].iterrows():
                states = ['x_1', 'x_2', 'x_3']
                state_desc = np.vstack([[data['name'], ] * len(states), states]).T
                self.state_desc = np.vstack([self.state_desc, state_desc])

                mdl = DynamicModel()
                mdl.name = key
                mdl.idx = np.where(self.state_desc[:, 0] == data['name'])[0]
                mdl.states = dict(zip(self.state_desc[mdl.idx, 1], mdl.idx))
                mdl.par = data
                mdl.gen_idx = self.generators[self.generators['name'] == data['gen']].index.tolist()[0]
                mdl.active = True
                self.pss_mdls.append(mdl)

        self.n_states = self.state_desc.shape[0]  # self.n_gen_states * self.n_gen
        self.state_desc_der = self.state_desc.copy()
        self.state_desc_der[:, 1] = np.char.add(np.array(self.n_states * ['d_']), self.state_desc[:, 1])
        self.x0 = np.zeros(self.n_states)
        self.e_q_0 = np.zeros(self.n_gen)
        self.e_q = np.zeros(self.n_gen)

        self.y_bus = self.build_y_bus()  # np.empty((self.n_bus, self.n_bus))
        self.build_y_bus_red()

        # Choose first generator at slack bus as slack generator
        self.slack_generator = (self.generators[self.generators['bus'] == self.slack_bus]).index[0]
        self.p_m_setp = np.array(self.generators['P'])/self.s_n
        sum_load_sl = sum(self.loads[self.loads['bus'] == self.slack_bus]['P'])/self.s_n if len(self.loads) > 0 else 0  # Sum loads at slack bus

        sl_gen_idx = np.array(self.generators['bus'] == self.slack_bus)
        # The case with multiple generators at the slack bus where one or more are n_par in parallel has not been tested.
        sum_gen_sl = sum(self.p_m_setp[sl_gen_idx]*self.n_par[sl_gen_idx][1:])  # Sum other generation at slack bus (not slack gen)
        self.p_m_setp[self.slack_generator] = self.s_0[self.get_bus_idx([self.slack_bus]).index].real - sum_gen_sl + sum_load_sl
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
        self.P_m = self.p_m*self.s_n/self.S_n_gen
        self.P_e = self.p_e*self.s_n/self.S_n_gen

        # Control model initial conditions
        # GOV
        for i, dm in enumerate(self.gov_mdls):
            v_2 = np.minimum(np.maximum(self.P_m[dm.gen_idx], dm.par['V_min']), dm.par['V_max'])
            v_1 = v_2
            v_3 = v_2

            dm.int_par['x_1_bias'] = dm.par['R']*v_1
            self.x0[dm.idx] = [
                v_2,
                dm.par['T_2']*v_2 - dm.par['T_3']*v_3,
            ]

        # AVR
        for i, dm in enumerate(self.avr_mdls):
            bias = -dm.par['T_b'] / dm.par['K'] * self.e_q_0[dm.gen_idx]
            dm.int_par['x_bias'] = bias
            self.x0[dm.idx] = [
                bias,
                self.e_q_0[dm.gen_idx]
            ]

    def ode_fun(self, t, x):

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
        for idx, gen in self.generators.iterrows():
            self.i_inj_d[self.gen_bus_idx_red[idx]] += self.e_q_st[idx] / (1j * self.x_d_st[idx])*self.q[idx]*self.n_par[idx]
            self.i_inj_q[self.gen_bus_idx_red[idx]] += self.e_d_st[idx] / (1j * self.x_q_st[idx])*self.d[idx]*self.n_par[idx]
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

        self.p_e = self.e_q_st * self.i_q + self.e_d_st * self.i_d - (self.x_d_st - self.x_q_st) * self.i_d * self.i_q
        self.P_e = self.e_q_st * self.I_q + self.e_d_st * self.I_d - (self.X_d_st - self.X_q_st) * self.I_d * self.I_q

        # Get updated dynamic model equations
        dx = np.zeros(self.n_states)

        # Controls
        # GOV
        self.speed_dev = x[self.speed_idx]
        for i, dm in enumerate(self.gov_mdls):
            speed_dev = -self.speed_dev[dm.gen_idx]
            # delta_p_m = 1 / dm.par['R'] * (1 / dm.par['T_2'] * (dm.par['T_1'] * (-speed_dev) - x[dm.states['x']]))
            v_1 = 1/dm.par['R']*(speed_dev + dm.int_par['x_1_bias'])
            v_2 = np.minimum(np.maximum(x[dm.states['x_1']], dm.par['V_min']), dm.par['V_max'])
            v_3 = dm.par['T_2']/dm.par['T_3']*v_2 - 1/dm.par['T_3']*x[dm.states['x_2']]
            delta_p_m = v_3 - dm.par['D_t']*speed_dev

            if dm.active:
                self.P_m[dm.gen_idx] = delta_p_m
                self.p_m[dm.gen_idx] = self.P_m[dm.gen_idx]*self.S_n_gen[dm.gen_idx]/self.s_n

            dx[dm.idx] = [
                1/dm.par['T_1']*(v_1 - v_2),
                v_3 - v_2
            ]

            # Lims on state variable x_1 (clamping)
            # print([dm.states['x_1']])
            if x[dm.states['x_1']] <= dm.par['V_min'] and dx[dm.states['x_1']] < 0:
                dx[dm.states['x_1']] *= 0

            if x[dm.states['x_1']] >= dm.par['V_max'] and dx[dm.states['x_1']] > 0:
                dx[dm.states['x_1']] *= 0

        # PSS
        for i, dm in enumerate(self.pss_mdls):
            u = self.speed[dm.gen_idx]
            v_1 = (dm.par['K']*u - x[dm.states['x_1']])/dm.par['T']
            v_2 = 1/dm.par['T_3']*(dm.par['T_1']*v_1 - x[dm.states['x_2']])
            v_3 = 1/dm.par['T_4']*(dm.par['T_2']*v_2 - x[dm.states['x_3']])
            if dm.active:
                self.v_pss[dm.gen_idx] = np.minimum(np.maximum(v_3, -dm.par['H_lim']), dm.par['H_lim'])

            dx[dm.idx] = [
                v_1,
                (dm.par['T_1']/dm.par['T_3'] - 1)*v_1 - 1/dm.par['T_3']*x[dm.states['x_2']],
                (dm.par['T_2']/dm.par['T_4'] - 1)*v_2 - 1/dm.par['T_4']*x[dm.states['x_3']],
            ]

        # AVR
        self.v_g_dev = self.v_g_setp - abs(self.v_g)  # Used for validating AVR
        for i, dm in enumerate(self.avr_mdls):
            if dm.active:
                self.e_q[dm.gen_idx] = np.minimum(np.maximum(x[dm.states['e_f']], dm.par['E_min']), dm.par['E_max'])
            # self.e_q[dm.gen_idx] = x[dm.states['e_f']]
            v_in = self.v_g_setp[dm.gen_idx] - abs(self.v_g[dm.gen_idx]) + self.v_pss[dm.gen_idx]

            dx[dm.idx] = [
                (dm.par['T_a']/dm.par['T_b'] - 1)*(v_in) - 1/dm.par['T_b']*(x[dm.states['x']] - dm.int_par['x_bias']),
                1/dm.par['T_e']*(dm.par['K']/dm.par['T_b']*(dm.par['T_a']*v_in - x[dm.states['x']]) - x[dm.states['e_f']])  # + self.v_aux[dm.gen_idx])
            ]

            # Lims on state variable e_f (clamping)
            if x[dm.states['e_f']] <= dm.par['E_min'] and dx[dm.states['e_f']] < 0:
                    dx[dm.states['e_f']] *= 0

            if x[dm.states['e_f']] >= dm.par['E_max'] and dx[dm.states['e_f']] > 0:
                    dx[dm.states['e_f']] *= 0


        # Generators
        t1 = time.time()
        self.p_m = self.P_m * self.S_n_gen/self.s_n
        self.t_m = self.p_m / (1 + self.speed)
        self.T_m = self.P_m / (1 + self.speed)
        for i, dm in enumerate(self.gen_mdls):
            dx[dm.idx] = [
                1/(2*dm.par['H'])*(self.T_m[i] - self.P_e[i] - dm.par['D'] * x[dm.states['speed']]),
                x[dm.states['speed']]*2*np.pi*self.f,
                1/(dm.par['T_d0_t'])*(self.e_q[i] + self.v_aux[i] - x[dm.states['e_q_t']] - self.I_d[i] * (self.X_d[i] - self.X_d_t[i])),
                1/(dm.par['T_q0_t'])*(-x[dm.states['e_d_t']] + self.I_q[i] * (self.X_q[i] - self.X_q_t[i])),
                1/(dm.par['T_d0_st']) * (x[dm.states['e_q_t']] - x[dm.states['e_q_st']] - self.I_d[i] * (self.X_d_t[i] - self.X_d_st[i])),
                1/(dm.par['T_q0_st']) * (x[dm.states['e_d_t']] - x[dm.states['e_d_st']] + self.I_q[i] * (self.X_q_t[i] - self.X_q_st[i])),
            ]

        return dx

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
