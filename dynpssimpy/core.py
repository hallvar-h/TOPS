from collections import defaultdict
from scipy.integrate import RK23, RK45, solve_ivp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
from scipy.sparse import csr_matrix
import matplotlib.cm as cm


def phasor(vec, start=0j, ax=None, **kwargs):

    if not ax:
        fig, ax = plt.subplots(1, subplot_kw=dict(aspect=1))
    return ax.annotate('',
                       xy=(vec.real + start.real, vec.imag + start.imag),
                       xytext=(start.real, start.imag),
                       arrowprops=dict(arrowstyle='->', **kwargs),
                       annotation_clip=False)


def plot_mode_shape(mode_shape, ax=None, normalize=False, xy0=np.empty(0), linewidth=2, auto_lim=False, colors=cm.get_cmap('Set1')):

    if not ax:
        ax = plt.subplot(111, projection='polar')
    if auto_lim:
        ax.set_rlim(0, max(abs(mode_shape)))

    if xy0.shape == (0,):
        xy0 = np.zeros_like(mode_shape)
    ax.axes.get_xaxis().set_major_formatter(NullFormatter())
    ax.axes.get_yaxis().set_major_formatter(NullFormatter())
    ax.grid(color=[0.85, 0.85, 0.85])
    # f_txt = ax.set_xlabel('f={0:.2f}'.format(f), color=cluster_color_list(), weight='bold', family='Times New Roman', )

    if normalize:
        mode_shape_max = mode_shape[np.argmax(np.abs(mode_shape))]
        if abs(mode_shape_max) > 0:
            mode_shape = mode_shape * np.exp(-1j * np.angle(mode_shape_max)) / np.abs(mode_shape_max)

    pl = []
    for i, (vec, xy0_) in enumerate(zip(mode_shape, xy0)):
        pl.append(ax.annotate("",
                              xy=(np.angle(vec), np.abs(vec)),
                              xytext=(np.angle(xy0_), np.abs(xy0_)),
                              arrowprops=dict(arrowstyle="->",
                                              #linewidth=linewidth,
                                              #linestyle=style_,
                                              color=colors(i),
                                              )))  # , headwidth=1, headlength = 1))

    return pl


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

        # Load model data
        for type_data in ['buses', 'lines', 'loads', 'generators', 'transformers', 'shunts', 'gov', 'avr', 'pss']:
            if type_data in model:
                setattr(self, type_data, model[type_data])
            else:
                setattr(self, type_data, pd.DataFrame())

        # self.buses = model['buses']
        # self.branches = model['lines']
        self.branches = self.lines
        # self.loads = model['loads']
        # self.generators = model['generators']
        # self.transformers = model['transformers']
        # self.shunts = model['shunts']
        # self.gov = pd.DataFrame()  # columns=['name', 'gen', 'R', 'T_1', 'T_2'], data=[])
        # self.avr = pd.DataFrame()  # columns=['name', 'gen', 'K', 'T_a', 'T_b', 'T_e', 'E_min', 'E_max'], data=[])
        # self.pss = pd.DataFrame()
        # self.events = model['events']

        if 'slack_bus' in model:
            self.slack_bus = model['slack_bus']
        else:
            self.slack_bus = None

        self.n_bus = len(self.buses)
        self.n_gen = len(self.generators)

        # Base for pu-system (determined by transformers)
        self.f = model['f']
        self.s_n = model['base_mva']  # For all buses
        self.S_n = self.s_n
        # Could make function for this, using graph algorithms etc.
        self.v_n = np.array(self.buses['V_n'])  # Assuming nominal bus voltages are according to transformer ratios
        self.z_n = self.v_n ** 2 / self.s_n
        self.i_n = self.s_n / (np.sqrt(3) * self.v_n)

        self.I_n = self.i_n
        self.V_n = self.v_n
        self.Z_n = self.z_n

        self.e = np.empty((0, 0))

        # Load flow
        # self.v_0 = np.ones(self.n_bus)
        # self.v_g_setp = self.generators['V']
        self.v_g_setp = np.array(self.generators['V'], dtype=float)

        # Simulation variables (updated every simulation step)
        # self.event_flags = np.zeros(len(self.events), dtype=bool)

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

        # Get p.u. values of reactances
        # Get gen nom. voltage. If given as zero in input data, select bus nom. voltage.
        gen_V_n = np.zeros(len(self.generators))
        for i, gen in self.generators.iterrows():
            gen_V_n[i] = gen['V_n'] if gen['V_n'] else self.V_n[self.gen_bus_idx[i]]

        self.x_d = np.array(self.generators['X_d']) * np.array(gen_V_n) ** 2 / np.array(
            self.generators['S_n']) / self.Z_n[self.gen_bus_idx]
        self.x_q = np.array(self.generators['X_q']) * np.array(gen_V_n) ** 2 / np.array(
            self.generators['S_n']) / self.Z_n[self.gen_bus_idx]
        self.x_d_t = np.array(self.generators['X_d_t']) * np.array(gen_V_n) ** 2 / np.array(
            self.generators['S_n']) / self.Z_n[self.gen_bus_idx]
        self.x_q_t = np.array(self.generators['X_q_t']) * np.array(gen_V_n) ** 2 / np.array(
            self.generators['S_n']) / self.Z_n[self.gen_bus_idx]
        self.x_d_st = np.array(self.generators['X_d_st']) * np.array(gen_V_n) ** 2 / np.array(
            self.generators['S_n']) / self.Z_n[self.gen_bus_idx]
        self.x_q_st = np.array(self.generators['X_q_st']) * np.array(gen_V_n) ** 2 / np.array(
            self.generators['S_n']) / self.Z_n[self.gen_bus_idx]
        self.T_d0_t = np.array(self.generators['T_d0_t'])
        self.T_q0_t = np.array(self.generators['T_q0_t'])
        self.T_d0_st = np.array(self.generators['T_d0_t'])
        self.T_q0_st = np.array(self.generators['T_q0_t'])

        self.reduced_bus_idx = np.empty(0)
        self.y_bus = np.empty((self.n_bus, self.n_bus))  # self.build_Y_bus() # np.empty((self.n_bus, self.n_bus))
        self.y_bus_red = np.empty((self.n_gen, self.n_gen))  # self.kron_reduction(self.Y_bus, self.gen_bus_idx)  # np.empty((self.n_gen, self.n_gen))
        self.y_bus_red_inv = np.empty((self.n_gen, self.n_gen))  # np.linalg.inv(self.Y_bus_red)  # np.empty((self.n_gen, self.n_gen))
        self.y_bus_red_mod = np.zeros_like(self.y_bus_red)

        # self.build_Y_bus()
        # self.kron_reduction()
        self.event_flag = False

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

        Y_branch = np.zeros((n_bus, n_bus), dtype=complex)
        buses = self.buses
        for i, branch in self.branches.iterrows():
            idx_from = buses[buses['name'] == branch['from_bus']].index
            idx_to = buses[buses['name'] == branch['to_bus']].index
            if branch['S_n'] == 0 and branch['V_n'] == 0:
                # Per unit of system base and bus nominal voltage
                impedance = (branch['R'] + 1j * branch['X']) * branch['length']
                shunt = 1j * branch['B'] * branch['length']
            else:
                if branch['unit'] in ['p.u.', 'pu']:
                    # If impedance given in p.u./km
                    impedance = (branch['R'] + 1j*branch['X'])*branch['length']*branch['V_n']**2/branch['S_n']/self.Z_n[idx_from]
                    shunt = 1j*branch['B']*branch['length']*1/(branch['V_n']**2/branch['S_n']/self.Z_n[idx_from])
                elif branch['unit'] in ['PF', 'pf', 'PowerFactory', 'powerfactory']:
                    impedance = (branch['R'] + 1j * branch['X']) * branch['length'] / self.Z_n[idx_from]
                    shunt = 1j * branch['B'] * branch['length'] * self.Z_n[idx_from]*1e-6
                else:  # branch['unit'] in ['Ohm', 'ohm']:
                    # If impedance given in Ohm/km
                    impedance = (branch['R'] + 1j * branch['X']) * branch['length']/self.Z_n[idx_from]
                    shunt = 1j*branch['B'] * branch['length'] * self.Z_n[idx_from]
            Y_branch[idx_from, idx_from] += 1 / impedance + shunt/2
            Y_branch[idx_to, idx_to] += 1 / impedance + shunt/2
            Y_branch[idx_from, idx_to] -= 1 / impedance
            Y_branch[idx_to, idx_from] -= 1 / impedance

        for i, trafo in self.transformers.iterrows():

            idx_from = buses[buses['name'] == trafo['from_bus']].index
            idx_to = buses[buses['name'] == trafo['to_bus']].index
            ratio = (trafo['ratio'] if not np.isnan(trafo['ratio']) else 1) if 'ratio' in trafo else 1

            V_n_from = trafo['V_n_from'] if trafo['V_n_from'] else self.V_n[idx_from]
            Z_base_trafo = V_n_from**2/trafo['S_n']  # <= Could also have used _to instead of _from
            impedance = (trafo['R'] + 1j*trafo['X'])*Z_base_trafo/self.Z_n[idx_from]
            admittance = 1/impedance
            Y_branch[idx_from, idx_from] += admittance
            Y_branch[idx_to, idx_to] += ratio*np.conj(ratio)*admittance
            Y_branch[idx_from, idx_to] -= np.conj(ratio)*admittance
            Y_branch[idx_to, idx_from] -= ratio*admittance
            # Y_branch[idx_from, idx_from] += ratio*np.conj(ratio)*admittance
            # Y_branch[idx_to, idx_to] += admittance
            # Y_branch[idx_from, idx_to] -= ratio*admittance
            # Y_branch[idx_to, idx_from] -= np.conj(ratio)*admittance

        Y_gen = np.zeros((n_bus, n_bus), dtype=complex)
        for i, gen in self.generators.iterrows():
            idx_bus = buses[buses['name'] == gen['bus']].index
            V_n = gen['V_n'] if gen['V_n'] else self.V_n[idx_bus]
            impedance = 1j * gen['X_d_st'] * V_n**2/gen['S_n']/self.Z_n[self.gen_bus_idx[i]]
            Y_gen[idx_bus, idx_bus] += 1 / impedance

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
        # self.y_bus_red_inv = np.linalg.inv(self.y_bus_red)  # np.empty((self.n_gen, self.n_gen))

        # self.y_bus_red = self.kron_reduction(self.y_bus, self.gen_bus_idx)  # np.empty((self.n_gen, self.n_gen))
        # self.y_bus_red_inv = np.linalg.inv(self.y_bus_red)  # np.empty((self.n_gen, self.n_gen))

        self.n_bus_red = self.y_bus_red.shape[0]
        # self.gen_bus_idx_red = np.arange(self.n_gen_bus)
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
            # p_sum_bus =
            p_sum_loads_bus[i] = (sum(self.loads[self.loads['bus'] == bus['name']]['P'])) / self.s_n
            p_sum_bus[i] = (sum(self.loads[self.loads['bus'] == bus['name']]['P'])
                            - sum(self.generators[self.generators['bus'] == bus['name']]['P'])) / self.s_n
            q_sum_loads_bus[i] = (sum(self.loads[self.loads['bus'] == bus['name']]['Q'])) / self.s_n
            q_sum_bus[i] = (sum(self.loads[self.loads['bus'] == bus['name']]['Q'])
                            - 0*sum(self.shunts[self.shunts['bus'] == bus['name']]['Q'])) / self.s_n
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

        # soln = newton(pf_equations, x, full_output=True, maxiter=10000)

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
        for i, data in self.gov.iterrows():
            states = ['x']
            state_desc = np.vstack([[data['name'], ] * len(states), states]).T
            self.state_desc = np.vstack([self.state_desc, state_desc])

            mdl = DynamicModel()
            mdl.idx = np.where(self.state_desc[:, 0] == data['name'])[0]
            mdl.states = dict(zip(self.state_desc[mdl.idx, 1], mdl.idx))
            mdl.par = data
            mdl.gen_idx = self.generators[self.generators['name'] == data['gen']].index.tolist()[0]
            self.gov_mdls.append(mdl)

        self.avr_mdls = []
        for i, data in self.avr.iterrows():
            states = ['x', 'e_f']
            state_desc = np.vstack([[data['name'], ] * len(states), states]).T
            self.state_desc = np.vstack([self.state_desc, state_desc])

            mdl = DynamicModel()
            mdl.idx = np.where(self.state_desc[:, 0] == data['name'])[0]
            mdl.states = dict(zip(self.state_desc[mdl.idx, 1], mdl.idx))
            mdl.par = data
            mdl.int_par = dict(x_bias=0)  # This parameter is necessary to give consistent initial conditions. Without
            # it there would have to be a steady state deviation in the voltage at t=0, since there is no integrator.
            mdl.gen_idx = self.generators[self.generators['name'] == data['gen']].index.tolist()[0]
            self.avr_mdls.append(mdl)

        self.pss_mdls = []
        for i, data in self.pss.iterrows():
            states = ['x_1', 'x_2', 'x_3']
            state_desc = np.vstack([[data['name'], ] * len(states), states]).T
            self.state_desc = np.vstack([self.state_desc, state_desc])

            mdl = DynamicModel()
            mdl.idx = np.where(self.state_desc[:, 0] == data['name'])[0]
            mdl.states = dict(zip(self.state_desc[mdl.idx, 1], mdl.idx))
            mdl.par = data
            mdl.gen_idx = self.generators[self.generators['name'] == data['gen']].index.tolist()[0]
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
        self.p_m_setp = np.array(self.generators['P'])/self.S_n
        sum_load_sl = sum(self.loads[self.loads['bus'] == self.slack_bus]['P'])/self.S_n  # Sum loads at slack bus
        sum_gen_sl = sum(self.p_m_setp[np.array(self.generators['bus'] == self.slack_bus)][1:])  # Sum other generation at slack bus (not slack gen)
        self.p_m_setp[self.slack_generator] = self.s_0[self.get_bus_idx([self.slack_bus]).index].real - sum_gen_sl + sum_load_sl
        # self.p_m_setp = self.s_0[self.gen_bus_idx].real + self.p_sum_loads_bus[self.gen_bus_idx]

        # Distribute reactive power equally among generators on the same bus
        self.n_gen_per_bus = np.array([sum(self.gen_bus_idx == idx) for idx in np.arange(self.n_bus)])
        self.q_g = (self.s_0.imag + self.q_sum_loads_bus)[self.gen_bus_idx]/self.n_gen_per_bus[self.gen_bus_idx]

        # From Load Flow
        # self.v_g_setp = abs(self.v_0[self.gen_bus_idx])
        self.v_g = self.v_0[self.gen_bus_idx]
        # self.s_g = (self.s_0 + self.p_sum_loads_bus + 1j*self.q_sum_loads_bus)[self.gen_bus_idx]
        self.s_g = self.p_m_setp + 1j*self.q_g
        self.i_g = np.conj(self.s_g/self.v_g)

        # Alternative 1
        # self.e_t = self.v_g + self.i_g * 1j * self.x_d_t
        # self.e_q_t = abs(self.e_t)
        # self.angle = np.angle(self.e_t)

        # Get rotor angle
        self.e_q_tmp = self.v_g + 1j * self.x_q * self.i_g
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

        # self.e_q_st = self.e_q_t - (self.x_d_t - self.x_d_st) * self.i_d
        # self.e_d_st = self.e_d_t + (self.x_q_t - self.x_q_st) * self.i_q

        self.e_q_st = self.v_q + self.x_d_st * self.i_d
        self.e_d_st = self.v_d - self.x_q_st * self.i_q
        self.e_st = self.e_q_st * self.q + self.e_d_st * self.d

        # self.e_t = self.v_g + (self.i_d*1j*self.x_d_t + 1j*self.i_q*1j*self.x_q_t)*self.d
        # self.e_q_t = np.abs(self.e_t)
        # self.e_q_t = self.v_q + self.x_d_t * self.i_d
        # self.e_t = self.e_q_t * np.exp(1j * self.angle)

        # np.angle(self.q)
        # np.angle(self.v_g + self.i_g * 1j * self.x_d_t)
        # self.e_q_t = np.abs(self.e_t)

        #E_q_t = self.v_q + self.generators['X_q']*self.I_d + self.generators['R']*self.I_q

        self.e_q = self.e_q_t + self.i_d * (self.x_d - self.x_d_t)
        self.e = self.e_q * np.exp(1j * self.angle)
        self.e_q_0 = self.e_q.copy()

        # self.x0 = np.zeros((self.n_gen_states + self.n_gov_states + self.n_avr_states)* self.n_gen)
        self.x0[self.angle_idx] = np.angle(self.e_q_tmp)
        self.x0[self.e_q_t_idx] = self.e_q_t
        self.x0[self.e_d_t_idx] = self.e_d_t
        self.x0[self.e_q_st_idx] = self.e_q_st
        self.x0[self.e_d_st_idx] = self.e_d_st
        # self.x0[self.avr_state_x_idx] *= 0
        # self.x0[self.avr_state_ef_idx] *= 0

        # AVR
        for i, dm in enumerate(self.avr_mdls):
            bias = -dm.par['T_b'] / dm.par['K'] * self.e_q_0[dm.gen_idx]
            dm.int_par['x_bias'] = bias
            self.x0[dm.idx] = [
                bias,
                self.e_q_0[dm.gen_idx]
            ]

        self.p_m = self.p_m_setp.copy()
        self.p_e = self.e_q_st * self.i_q + self.e_d_st*self.i_d - (self.x_d_st - self.x_q_st) * self.i_d * self.i_q
        # E = np.ones(len(model['generators']), dtype=complex)
        # E[0:2] *= np.exp(1j*np.pi/12)

    def ode_fun(self, t, x):

        self.speed = x[self.speed_idx]
        self.angle = x[self.angle_idx]
        self.e_q_t = x[self.e_q_t_idx]
        self.e_d_t = x[self.e_d_t_idx]
        self.e_q_st = x[self.e_q_st_idx]
        self.e_d_st = x[self.e_d_st_idx]
        # self.gov_state = x[self.gov_state_idx]
        # gov_R = self.generators['gov_R']
        # gov_T_1 = self.generators['gov_T_1']
        # gov_T_2 = self.generators['gov_T_2']

        self.d = np.exp(1j * (self.angle - np.pi / 2))
        self.q = np.exp(1j * self.angle)

        self.e_st = self.e_q_st * self.q + self.e_d_st * self.d
        self.e_t = self.e_q_t*self.q + self.e_d_t*self.d
        self.e = self.e_q * self.q


        # Interfacing generators with system
        self.i_inj_d = np.zeros(self.n_bus_red, dtype=complex)
        self.i_inj_q = np.zeros(self.n_bus_red, dtype=complex)
        for idx in range(self.n_gen):
            self.i_inj_d[self.gen_bus_idx_red[idx]] += self.e_q_st[idx] / (1j * self.x_d_st[idx])*self.q[idx]
            self.i_inj_q[self.gen_bus_idx_red[idx]] += self.e_d_st[idx] / (1j * self.x_q_st[idx])*self.d[idx]

        self.i_inj = self.i_inj_d + self.i_inj_q


        self.v_red = np.linalg.solve(self.y_bus_red + self.y_bus_red_mod, self.i_inj)
        self.v_g = self.v_red[self.gen_bus_idx_red]
        # self.v_g = np.linalg.solve(self.y_bus_red + self.y_bus_red_mod, self.i_inj)
        self.v_g_dq = self.v_g * np.exp(1j * (np.pi / 2 - self.angle))
        self.v_d = self.v_g_dq.real
        self.v_q = self.v_g_dq.imag

        # self.i_q = -(self.e_d_st - self.v_d) / self.x_q_st
        # self.i_d = (self.e_q_st - self.v_q) / self.x_d_st
        # self.i_g_dq = self.i_d + 1j * self.i_q
        # self.i_g = self.i_g_dq * np.exp(1j * -(np.pi / 2 - self.angle))

        self.i_g = (self.e_st - self.v_g)/(1j*self.x_d_st)
        self.i_g_dq = self.i_g * np.exp(1j * (np.pi / 2 - self.angle))
        self.i_d = self.i_g_dq.real
        self.i_q = self.i_g_dq.imag

        self.e_q_tmp = self.v_g + 1j * self.x_q * self.i_g

        #self.p_e = self.e_q_t * self.i_q + self.e_d_t*self.i_d - (self.x_d_t - self.x_q_t) * self.i_d * self.i_q
        self.p_e = self.e_q_st * self.i_q + self.e_d_st * self.i_d - (self.x_d_st - self.x_q_st) * self.i_d * self.i_q

        # Governor
        self.speed_dev = -x[self.speed_idx]

        # Get updated dynamic model equations
        # TODO: Could be written without loops to make it faster?
        dx = np.zeros(self.n_states)

        # Controls

        # GOV
        for i, dm in enumerate(self.gov_mdls):
            speed_dev = self.speed_dev[dm.gen_idx]
            delta_p_m = 1 / dm.par['R'] * (1/dm.par['T_2'] * (dm.par['T_1'] * speed_dev - x[dm.states['x']]))
            self.p_m[dm.gen_idx] = self.p_m_setp[dm.gen_idx] + delta_p_m

            dx[dm.idx] = [
                (dm.par['T_1']/dm.par['T_2'] - 1)*speed_dev - 1/dm.par['T_2']*x[dm.states['x']],
            ]

        # PSS
        for i, dm in enumerate(self.pss_mdls):
            u = self.speed[dm.gen_idx]
            v_1 = (dm.par['K']*u - x[dm.states['x_1']])/dm.par['T']
            v_2 = 1/dm.par['T_3']*(dm.par['T_1']*v_1 - x[dm.states['x_2']])
            v_3 = 1/dm.par['T_4']*(dm.par['T_2']*v_2 - x[dm.states['x_3']])
            self.v_pss[dm.gen_idx] = np.minimum(np.maximum(v_3, -dm.par['H_lim']), dm.par['H_lim'])

            dx[dm.idx] = [
                v_1,
                (dm.par['T_1']/dm.par['T_3'] - 1)*v_1 - 1/dm.par['T_3']*x[dm.states['x_2']],
                (dm.par['T_2']/dm.par['T_4'] - 1)*v_2 - 1/dm.par['T_4']*x[dm.states['x_3']],
            ]

        # AVR
        self.v_g_dev = self.v_g_setp - abs(self.v_g)  # Used for validating AVR
        for i, dm in enumerate(self.avr_mdls):
            self.e_q[dm.gen_idx] = np.minimum(np.maximum(x[dm.states['e_f']], dm.par['E_min']), dm.par['E_max'])
            # self.e_q[dm.gen_idx] = x[dm.states['e_f']]
            v_in = self.v_g_setp[dm.gen_idx] - abs(self.v_g[dm.gen_idx]) + self.v_pss[dm.gen_idx]

            # lim_switch = np.minimum(np.maximum(x[dm.states['e_f']], dm.par['E_min']), dm.par['E_max'])

            dx[dm.idx] = [
                (dm.par['T_a']/dm.par['T_b'] - 1)*(v_in) - 1/dm.par['T_b']*(x[dm.states['x']] - dm.int_par['x_bias']),
                1/dm.par['T_e']*(dm.par['K']/dm.par['T_b']*(dm.par['T_a']*v_in - x[dm.states['x']]) - x[dm.states['e_f']])  # + self.v_aux[dm.gen_idx])
            ]

            # Lims:
            if x[dm.states['e_f']] <= dm.par['E_min'] and dx[dm.states['e_f']] < 0:
                    dx[dm.states['e_f']] *= 0

            if x[dm.states['e_f']] >= dm.par['E_max'] and dx[dm.states['e_f']] > 0:
                    dx[dm.states['e_f']] *= 0


        # Generators
        self.t_m = self.p_m / (1 + self.speed)
        for i, dm in enumerate(self.gen_mdls):
            dx[dm.idx] = [
                1/(2*dm.par['H'])*(self.t_m[i] - self.p_e[i] - dm.par['D'] * x[dm.states['speed']]),
                x[dm.states['speed']]*2*np.pi*50,
                1/(dm.par['T_d0_t'])*(self.e_q[i] + self.v_aux[i] - x[dm.states['e_q_t']] - self.i_d[i] * (self.x_d[i] - self.x_d_t[i])),
                1/(dm.par['T_q0_t'])*(-x[dm.states['e_d_t']] + self.i_q[i] * (self.x_q[i] - self.x_q_t[i])),
                1/(dm.par['T_d0_st']) * (x[dm.states['e_q_t']] - x[dm.states['e_q_st']] - self.i_d[i] * (self.x_d_t[i] - self.x_d_st[i])),
                1/(dm.par['T_q0_st']) * (x[dm.states['e_d_t']] - x[dm.states['e_d_st']] + self.i_q[i] * (self.x_q_t[i] - self.x_q_st[i])),
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


class PowerSystemModelLinearization:
    def __init__(self, eq):
        self.eq = eq
        self.eps = 1e-10
        pass

    def linearize(self, eq=None):
        if eq:
            self.eq = eq
        self.a = jacobian_num(lambda x: self.eq.ode_fun(0, x), self.eq.x0, eps=self.eps)
        self.n = self.a.shape[0]
        self.eigs, evs = np.linalg.eig(self.a)
        # self.lev = np.conj(evs).T
        self.rev = evs
        self.lev = np.linalg.inv(self.rev)
        self.damping = -self.eigs.real / abs(self.eigs)
        self.freq = self.eigs.imag / (2 * np.pi)

    def linearize_inputs(self, inputs):
        eq = self.eq
        b = np.zeros((len(eq.x0), 0))
        for inp in inputs:
            var = getattr(eq, inp[0])
            index = inp[1]
            if not index:
                index = range(len(var))

            if len(inp) == 3:
                if inp[2] == 'Complex' or inp[2] == 'imag':
                    mod = 1j
            else:
                mod = 1

            for i in index:
                var_0 = var[i]
                var[i] = var_0 + self.eps*mod
                f_1 = eq.ode_fun(0, eq.x0)
                var[i] = var_0 - self.eps*mod
                f_2 = eq.ode_fun(0, eq.x0)
                var[i] = var_0
                b = np.hstack([b, ((f_1 - f_2) / (2 * self.eps))[:, None]])
        # self.b = b
        return b

    def linearize_inputs_v2(self, input_desc):
        eq = self.eq
        eps = self.eps
        b = np.zeros((len(eq.x0), len(input_desc)))
        for i, inp_ in enumerate(input_desc):
            b_tmp = np.zeros(len(eq.x0))
            for inp__ in inp_:
                var = getattr(eq, inp__[0])
                index = inp__[1]
                gain = inp__[2] if len(inp__) == 3 else 1

                var_0 = var[index]
                var[index] = var_0 + eps * gain
                f_1 = eq.ode_fun(0, eq.x0)
                var[index] = var_0 - eps * gain
                f_2 = eq.ode_fun(0, eq.x0)
                var[index] = var_0
                b_tmp += ((f_1 - f_2) / (2 * eps))
            b[:, i] = b_tmp  # np.hstack([b, ((f_1 - f_2) / (2 * eps))[:, None]])
        # self.b = b
        return b

    def plot_eigs(self):
        eigs = self.eigs
        fig, ax = plt.subplots(1)
        sc = ax.scatter(eigs.real, eigs.imag)
        ax.axvline(0, color='k', linewidth=0.5)
        ax.axhline(0, color='k', linewidth=0.5)
        ax.grid(True)

        annot = ax.annotate("", xy=(0, 0), xytext=(20, 20), textcoords="offset points",
                            bbox=dict(boxstyle="round", fc="w"),
                            arrowprops=dict(arrowstyle="->"))
        annot.set_visible(False)

        def update_annot(ind):

            pos = sc.get_offsets()[ind["ind"][0]]
            annot.xy = pos
            text = '{:.2f} Hz\n{:.2f}%'.format(pos[1] / (2 * np.pi), -100 * pos[0] / np.sqrt(sum(pos ** 2)))
            annot.set_text(text)
            annot.get_bbox_patch().set_facecolor('C0')
            annot.get_bbox_patch().set_alpha(0.4)

        def hover(event):
            vis = annot.get_visible()
            if event.inaxes == ax:
                cont, ind = sc.contains(event)
                if cont:
                    update_annot(ind)
                    annot.set_visible(True)
                    fig.canvas.draw_idle()
                else:
                    if vis:
                        annot.set_visible(False)
                        fig.canvas.draw_idle()

        fig.canvas.mpl_connect("motion_notify_event", hover)
        plt.show()



if __name__ == '__main__':

    from cases.sm_load_simple import sm_load_simple

    model = sm_load_simple()
    model['loads']['Q'] = 200
    # model['lines']['X'] *= 2

    #
    from cases.k2a import k2a
    model = k2a()

    model['generators']['gov_R'] = 0.05
    model['generators']['gov_T_1'] = 1.0
    model['generators']['gov_T_2'] = 10.0
    model['generators']['avr_K'] = 100
    model['generators']['avr_T_a'] = 2.0
    model['generators']['avr_T_b'] = 10.0
    model['generators']['avr_T_e'] = 0.1
    model['generators']['avr_E_min'] = -3
    model['generators']['avr_E_max'] = 3
    model['generators']['D'] = 1 * 0
    #
    # model['generators']['T_d0_st'] = 0.05
    # model['generators']['T_q0_st'] = 0.05
    # model['generators']['X_d_t'] = 0.3
    # model['generators']['X_q_t'] = 0.3
    # model['generators']['X_d_st'] = 0.2
    # model['generators']['X_q_st'] = 0.2

    eq = PowerSystemModel(model=model)
    eq.power_flow()
    eq.init_dyn_sim()
    eq.build_y_bus_red(eq.loads['bus'])
    load_bus_red_idx = eq.get_bus_idx_red(eq.loads['bus'])



    fig, ax_all, = plt.subplots(1, eq.n_gen, subplot_kw=dict(aspect=1), squeeze=False)
    ax_all = ax_all[0, :]
    for i, ax in enumerate(ax_all):
        ax.set_xlim([-1.5, 1.5])
        ax.set_ylim([-1.5, 1.5])

        phasor(eq.e_q_tmp[i], ax=ax, color='C2')
        phasor(eq.e_t[i], ax=ax, color='C2')
        phasor(eq.e_q_t[i]*eq.q[i], ax=ax, color='C2')
        phasor(eq.e_d_t[i]*eq.d[i], ax=ax, color='C2')
        phasor(eq.e_q_st[i]*eq.q[i], ax=ax, color='C2')
        phasor(eq.e_d_st[i]*eq.d[i], ax=ax, color='C2')
        phasor(eq.e_st[i], ax=ax, color='C2')

        phasor(eq.i_g[i], ax=ax, color='C3')
        phasor(eq.i_q[i] * eq.q[i], ax=ax, color='C3')
        phasor(eq.i_d[i] * eq.d[i], ax=ax, color='C3')

        phasor(eq.v_g[i], ax=ax, color='C0')
        phasor(eq.v_q[i] * eq.q[i], ax=ax, color='C0')
        phasor(eq.v_d[i] * eq.d[i], ax=ax, color='C0')

        phasor(eq.i_d[i] * eq.x_d_t[i] * eq.q[i], start=eq.v_g[i], ax=ax, color='C3')
        phasor(eq.i_q[i] * eq.x_q_t[i] * -eq.d[i], start=eq.v_g[i] + eq.i_d[i] * eq.x_d_t[i] * eq.q[i], ax=ax, color='C3')

        phasor(eq.i_d[i] * eq.x_d_st[i] * eq.q[i], start=eq.v_g[i], ax=ax, color='C3')
        phasor(eq.i_q[i] * eq.x_q_st[i] * -eq.d[i], start=eq.v_g[i] + eq.i_d[i] * eq.x_d_st[i] * eq.q[i], ax=ax, color='C3')
    print(eq.ode_fun(0, eq.x0))

    # if False:
    # for i, event in eq.events[(~eq.event_flags) & (eq.events['time'] < t)].iterrows():
    #     if event['type'] == 'short_circuit':
    #         bus_idx = eq.buses[eq.buses['name'] == event['object']].index
    #         event_data = csr_matrix(([1/event['impedance']], (bus_idx, bus_idx)), shape=(eq.n_bus, eq.n_bus))
    x0 = eq.x0
    # x0[1] = 1e-3
    t_end = 20
    # sol = RK45(eq.ode_fun, 0, x0, t_end, max_step=5e-3)
    from scipy.integrate import BDF, LSODA
    sol = RK23(eq.ode_fun, 0, x0, t_end, max_step=20e-3)
    # sol = BDF(eq.ode_fun, 0, x0, t_end, max_step=20e-3)

    t = 0
    t_prev = t
    result_dict = defaultdict(list)
    monitored_variables = ['speed', 'angle', 't_m', 'p_e', 'e_q', 'e_q_t', 'e_d_t', 'e_q_st', 'e_d_st', 'v_g',
                           'voltage_dev', 'v_pss']

    np.random.seed(0)
    load_switching_freq = 50

    while t < t_end:
        print(t)

        # Simulate next step
        result = sol.step()
        t_prev = t
        t = sol.t

        # if t > 1 and t <= 1.1:
        #     print('Event!')
        #     eq.y_bus_red_mod[5, 5] = 1e6
        #     # eq.v_0[eq.gen_bus_idx[0]] = 1.1j
        # else:
        #     eq.y_bus_red_mod[5, 5] = 0
        #     pass

        if (t%(1/load_switching_freq)) < (t_prev%(1/load_switching_freq)):
            # eq.y_bus_red_mod[4, 4] += np.random.randn(1)*0.1
            # eq.y_bus_red_mod[5, 5] += np.random.randn(1)*0.1
            eq.y_bus_red_mod[(load_bus_red_idx,) * 2] += np.random.randn(len(load_bus_red_idx))*eq.loads['P'] / eq.S_n * 0.01

        # Store result variables
        result_dict['Global', 't'].append(sol.t)
        for var in monitored_variables:
            [result_dict[('G' + str(i), var)].append(var_) for i, var_ in enumerate(getattr(eq, var))]
        # [result_dict[('G' + str(i), 'speed')].append(speed) for i, speed in enumerate(eq.speed)]

    index = pd.MultiIndex.from_tuples(result_dict)
    result = pd.DataFrame(result_dict, columns=index)

    plot_var = monitored_variables
    fig, ax = plt.subplots(len(plot_var), 1, sharex=True)
    result.xs_0 = lambda q: result.xs((q), axis='columns', level=0)
    result.xs_1 = lambda q: result.xs((q), axis='columns', level=1)
    for ax_, var in zip(ax, plot_var):
        ax_.plot(result[('Global', 't')], result.xs_1(var))
        ax_.set_ylabel(var)


    # plot_quantities = ['speed', 'angle', 'P_e', 'P_m', 'E', 'E_q_t', 'v_g', 'voltage_dev']
    # fig, ax = plt.subplots(len(plot_quantities), sharex=True)
    # for ax_, q in zip(ax[[0, 1, 2, 2, 3, 4, 5, 6]], plot_quantities):
    #     ax_.plot(result[('Global', 't')], result.xs((q), axis='columns', level=1))
    #     ax_.set_ylabel(q)
    #
    # angles = np.array(result.xs(('angle'), axis='columns', level=1))
    # ax[-1].plot(result['Global', 't'], (angles.T - np.mean(angles, axis=1)).T)

# if False:
    # Modal analysis
    lin = PowerSystemModelLinearization(eq)
    lin.linearize()

    inputs = [
        # ['v_pss', None],
        # ['p_m_setp', [0, 1, 2]],
        ['y_bus_red_mod', [(idx,)*2 for idx in load_bus_red_idx]]
    ]
    b = lin.linearize_inputs(inputs)
    # plt.imshow(b)

    a = lin.a
    lin.plot_eigs()

    em_idx = np.where(lin.eigs.imag/(2*np.pi) > 0.1)[0]
    fig, ax = plt.subplots(1, len(em_idx), subplot_kw=dict(projection='polar'))
    for ax_, mode_shape, eig in zip(ax, lin.rev[np.ix_(eq.speed_idx, em_idx)].T, lin.eigs[em_idx]):
        plot_mode_shape(mode_shape, ax=ax_, normalize=True)
        ax_.set_xlabel('{:.2f} Hz \n {:.1f}%'.format(eig.imag/(2*np.pi), -100*eig.real/(abs(eig))))

