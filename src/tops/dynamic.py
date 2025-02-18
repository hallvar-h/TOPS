import numpy as np

import tops.utility_functions as dps_uf
import tops.dyn_models as mdl_lib
import scipy.sparse as sp
from scipy.sparse import linalg as sp_linalg
from scipy.sparse import diags as sp_diags
from scipy.optimize import fsolve

import json
import os

import importlib
import warnings
importlib.reload(mdl_lib)


class PowerSystemModel:
    def __init__(self, model, user_mdl_lib=None):
        self.user_mdl_lib = user_mdl_lib
        file_is_json = isinstance(model, str) and model[-5:] == '.json'
        if file_is_json:
            try:
                # Try to open specified path directly
                with open(model) as f:
                    data = f.read()
                model_data = json.loads(data)
            except IOError:
                try:
                    # Try to see if the specified model is available in the ps_models folder of the module
                    current_folder = os.path.dirname(os.path.abspath(__file__))
                    model_file_path = os.path.join(current_folder, 'ps_models', model)
                    with open(model_file_path) as f:
                        data = f.read()
                    model_data = json.loads(data)
                except IOError:
                    print('Power System Model Data File not found. Aborting.')
                    return

        elif isinstance(model, dict):
            model_data = model
        
        model = model_data.copy()
        self.model = model

        self.perform_kron_reduction = False
        self.pf_max_it = 10
        self.pf_tol = 1e-8

        self.s_n = model['base_mva']
        self.f_n = model['f']
        self.slack_bus = model['slack_bus'] if 'slack_bus' in model else None
        self.buses = dps_uf.structured_array_from_list(model['buses'][0], model['buses'][1:])
        self.n_bus = len(self.buses)

        self.y_bus_lf = None
        self.power_flow_ready = False
        
        self.setup_ready = False
        self.initialization_ready = False

        if 'transformers' in model:
            model['trafos'] = model['transformers']
            model.pop('transformers', None)

        if 'generators' in model:
            model['gen'] = model['generators']
            model.pop('generators', None)

        mdl_data = ['lines', 'trafos', 'loads', 'shunts']
        default_mdls = ['Line', 'Trafo', 'Load', 'Shunt']

        for key, default_mdl in zip(mdl_data, default_mdls):
            # print(key, default_mdl)
            if key in model and isinstance(model[key], list) and len(model[key]) > 1:
                model[key] = {default_mdl: model[key]}

        
        self.sys_data = {
            's_n': self.s_n,
            'f_n': self.f_n,
            'n_bus': self.n_bus,
            'bus_v_n': self.buses['V_n'],
            'bus_names': self.buses['name'],
            'red_to_full': None
        }
        self.dyn_mdls = []
        self.dyn_mdls_dict = {}

        self.add_model_data(model)

    def add_model_data(self, model_data):
        for key, val in model_data.items():
            if isinstance(val, dict):
                category_key = key
                category = val
                for mdl_key, mdl_data_raw in category.items():
                    if hasattr(self.user_mdl_lib, category_key) and hasattr(getattr(self.user_mdl_lib, category_key), mdl_key):
                        # print('User model: {}, {}'.format(category_key, mdl_key))
                        mdl_class = getattr(getattr(self.user_mdl_lib, category_key), mdl_key)
                    elif hasattr(mdl_lib, category_key) and hasattr(getattr(mdl_lib, category_key), mdl_key):
                        # print('Standard model: {}, {}'.format(category_key, mdl_key))
                        mdl_class = getattr(getattr(mdl_lib, category_key), mdl_key)

                    else:
                        print('Model {}:{} not found in model library.'.format(category_key, mdl_key))
                        continue

                    mdl_data = dps_uf.structured_array_from_list(mdl_data_raw[0], mdl_data_raw[1:])
                    mdl = mdl_class(mdl_data, self.sys_data)
                    if hasattr(self, category_key):
                        getattr(self, category_key).update({mdl_key: mdl})
                        self.dyn_mdls_dict[category_key].update({mdl_key: mdl})
                    else:
                        setattr(self, category_key, {mdl_key: mdl})
                        self.dyn_mdls_dict[category_key] = {mdl_key: mdl}

                    [self.dyn_mdls.append(item) for item in mdl_lib.utils.get_submodules(mdl)]  # [::-1]

    def setup(self):
        self.mdl_instructions = {key: list() for key in [
            'initialize',
            'state_derivatives',
            'connections',
            'bus_references',
            'bus_ref_spec',
            'reduced_system',
            'load_flow_pq',
            'load_flow_pv',
            'load_flow_adm',
            'dyn_const_adm',
            'dyn_var_adm',
            'init_from_load_flow',
            'current_injections',
            'apparent_power_injections',
            # 'init_mdl', 'lf_adm', 'dyn_const_adm', 'dyn_var_adm',
            # '_current_injections', 'state_derivatives',
            # 'ref'
        ]}

        for mdl in self.dyn_mdls:
            for key, fun_list in self.mdl_instructions.items():
                if hasattr(mdl, key):
                    fun_list.append(mdl)

        if self.perform_kron_reduction:
            buses_red = []
            for mdl in self.mdl_instructions['reduced_system']:
                [buses_red.append(bus) for bus in mdl.reduced_system()]

            bus_idx_red = dps_uf.lookup_strings(buses_red, self.buses['name'])
        else:
            bus_idx_red = np.arange(self.n_bus)

        # Remove duplicate buses
        bus_idx_red_sort, idx = np.unique(bus_idx_red, return_index=True)
        self.bus_idx_red = bus_idx_red_sort
        self.n_bus_red = len(self.bus_idx_red)

        for mdl in self.mdl_instructions['bus_ref_spec']:
            for identifier, bus_names in mdl.bus_ref_spec().items():
                bus_idx = dps_uf.lookup_strings(bus_names, self.buses['name'])
                mdl.bus_idx[identifier] = bus_idx
                lookup, mask  = dps_uf.lookup_strings(bus_idx, self.bus_idx_red, return_mask=True)
                mdl.bus_idx_red[identifier][mask] = lookup
                mdl.bus_idx_red[identifier][~mask] = -99999

        self.setup_ready = True

    def build_y_bus_lf(self):

        y_lf = np.zeros((self.n_bus,) * 2, dtype=complex)
        for mdl in self.mdl_instructions['load_flow_adm']:
            data, (row_idx, col_idx) = mdl.load_flow_adm()
            sp_mat = sp.csr_matrix((data.flatten(), (row_idx.flatten(), col_idx.flatten())),
                                   shape=(self.n_bus,) * 2)
            y_lf += sp_mat.todense()

        self.y_bus_lf = y_lf
        return y_lf

    def build_y_bus_dyn(self):

        y_dyn = np.zeros((self.n_bus,) * 2, dtype=complex)
        for mdl in self.mdl_instructions['dyn_const_adm']:
            data, (row_idx, col_idx) = mdl.dyn_const_adm()
            sp_mat = sp.csr_matrix((data.flatten(), (row_idx.flatten(), col_idx.flatten())), shape=(self.n_bus,) * 2)
            y_dyn += sp_mat.todense()

        self.y_bus_dyn = y_dyn
        return y_dyn


    def power_flow(self, print_output=False):

        if not self.setup_ready:
            self.setup()

        if self.y_bus_lf is None:
            self.build_y_bus_lf()

        bus_type = np.array(['PQ'] * self.n_bus, dtype='<U2')

        p_pq = np.zeros(self.n_bus)
        q_pq = np.zeros(self.n_bus)

        for mdl in self.mdl_instructions['load_flow_pq']:
            bus_idx, p, q = mdl.load_flow_pq()
            np.add.at(p_pq, bus_idx, p / self.s_n)
            np.add.at(q_pq, bus_idx, q / self.s_n)

        p_pv = np.zeros(self.n_bus)
        v_pv = np.ones(self.n_bus)

        if not self.slack_bus:
            bus_idx = self.mdl_instructions['load_flow_pv'][0].load_flow_pv()[0]
            sl_idx = bus_idx[0]
            self.slack_bus = self.buses[sl_idx]['name']
        else:
            sl_idx = dps_uf.lookup_strings(self.slack_bus, self.buses['name'])

        for mdl in self.mdl_instructions['load_flow_pv']:
            bus_idx, p, v = mdl.load_flow_pv()
            if sl_idx in bus_idx:
                unit_idx = np.argmax(bus_idx == sl_idx)
                slack_unit = (mdl, unit_idx)
                self.slack_unit = slack_unit
            bus_type[bus_idx] = 'PV'
            np.add.at(p_pv, bus_idx, p / self.s_n)
            v_pv[bus_idx] = v
            # mdl.lf_idx = slice(k_lf, mdl.n_units)
            # k_lf += mdl.n_units

        bus_type[sl_idx] = 'SL'

        phi_0 = np.zeros(self.n_bus)
        self.v_0, self.s_0, converged = dps_uf.newton_rhapson_power_flow(self.y_bus_lf, v_pv, p_pv + p_pq, q_pq, bus_type,
                                                              self.pf_tol, self.pf_max_it)
        self.v0 = self.v_0
        self.v_prev = self.v0.copy()
        self.it_prev = 0

        pv_units_per_bus = np.zeros(self.n_bus, dtype=int)
        for mdl in self.mdl_instructions['load_flow_pv']:
            bus_idx = mdl.load_flow_pv()[0]
            np.add.at(pv_units_per_bus, bus_idx, 1)

        a = self.s_0.imag + q_pq
        b = pv_units_per_bus
        q_per_pv_unit = np.divide(a, b, out=np.zeros_like(a), where=b != 0)

        self.load_flow_soln = {}
        for mdl in self.mdl_instructions['load_flow_pv']:
            bus_idx, p_setp, v = mdl.load_flow_pv()
            q = q_per_pv_unit[bus_idx]
            p = -p_setp / self.s_n
            if sl_idx in bus_idx:
                unit_idx = np.argmax(bus_idx == sl_idx)
                p[unit_idx] = self.s_0[sl_idx].real + -p_setp[unit_idx] / self.s_n + p_pv[sl_idx] + p_pq[sl_idx]
                # p[unit_idx] /= mdl.par['n_par'][unit_idx]

            self.load_flow_soln[mdl] = (p + 1j * q) * self.s_n
        
        if converged:
            self.power_flow_ready = True

    def kron_reduction(self, y_bus, keep_buses):
        remove_buses = list(set(range(self.n_bus)) - set(keep_buses))
        y_rr = y_bus[np.ix_(remove_buses, remove_buses)]
        y_rk = y_bus[np.ix_(remove_buses, keep_buses)]
        y_kk = y_bus[np.ix_(keep_buses, keep_buses)]

        # Build matrix for mapping back to full system (v_full = self.red_to_full.dot(self.v_red)
        self.red_to_full = np.zeros((self.n_bus, self.n_bus_red), dtype=complex)
        self.red_to_full[remove_buses, :] = - np.linalg.inv(y_rr).dot(y_rk)
        self.red_to_full[self.bus_idx_red] = np.eye(self.n_bus_red)

        return y_kk - y_rk.T.dot(np.linalg.inv(y_rr)).dot(y_rk)
    
    def define_state_vector(self):
        self.state_desc = np.empty((0, 2))
        self.n_states = 0
        for mdl in self.dyn_mdls:
            mdl.idx = slice(self.n_states, mdl.idx.stop - mdl.idx.start + self.n_states)
            for field in mdl.state_idx_global.dtype.names:
                mdl.state_idx_global[field] += mdl.idx.start
            self.n_states += mdl.n_states * mdl.n_units
            self.state_desc = np.vstack([self.state_desc, mdl.state_desc])

        self.state_desc_der = self.state_desc.copy()
        self.state_desc_der[:, 1] = np.char.add(np.array(self.n_states * ['d_']), self.state_desc[:, 1])

    def init_dyn_sim(self):
        if not self.power_flow_ready:
            self.power_flow()

        self.define_state_vector()
        self.x_0 = np.zeros(self.n_states)
        self.x0 = self.x_0

        for mdl in self.mdl_instructions['init_from_load_flow']:
            mdl_lf_soln = self.load_flow_soln[mdl] if mdl in self.load_flow_soln else None
            mdl.init_from_load_flow(self.x_0, self.v_0, mdl_lf_soln)

        # Build reduced system
        self.y_bus_dyn = self.build_y_bus_dyn()
        self.y_bus_red_full = self.kron_reduction(self.y_bus_dyn, self.bus_idx_red)
        self.y_bus_red = sp.csr_matrix(self.y_bus_red_full)
        self.y_bus_red_mod = sp.csr_matrix(self.y_bus_red_full)*0

        # for mdl in self.dyn_mdls:
        #     mdl.sys_par['red_to_full'] = self.red_to_full

        self.mdl_connections = mdl_lib.utils.determine_connections(self.dyn_mdls_dict)
        for mdl, connections in self.mdl_connections.items():
            for input_key, conn in connections.items():
                init_vals = mdl._input_values[input_key].copy()
                def new_input_fun(x, v, conn=conn, mdl=mdl, init_vals=init_vals):
                    input = init_vals
                    for c in conn:
                        source_fun = getattr(self.dyn_mdls_dict[c['container']][c['mdl']], c['output'])
                        input[c['dest_idx']] = source_fun(x, v)[c['source_idx']]
                    return input
                setattr(mdl, input_key, new_input_fun)
        
        # Initialize state vector
        self.mdl_connections_by_source = mdl_lib.utils.determine_connections(self.dyn_mdls_dict, order_by='output')
        for mdl, connections in self.mdl_connections_by_source.items():
            if hasattr(mdl, 'init_from_connections'):
                output_values = np.zeros(mdl.n_units, [(field, float) for field in mdl.output_list()])
                for output_key, conn in connections.items():
                    init_val = np.zeros(mdl.n_units)
                    for c in conn:
                        input_fun = getattr(self.dyn_mdls_dict[c['container']][c['mdl']], c['input'])
                        input_fun = lambda x, v:  self.dyn_mdls_dict[c['container']][c['mdl']]._input_values[c['input']]
                        np.add.at(init_val, c['source_idx'], input_fun(None, None)[c['dest_idx']])
                    output_values[output_key] = init_val

                mdl.init_from_connections(self.x_0, self.v_0, output_values)

    
        self.initialization_ready = True

    def state_derivatives(self, t, x, v_red):

        for mdl in self.dyn_mdls:
            mdl.reset_outputs()
            mdl._store_output = True

        dx = np.zeros(self.n_states)
        for mdl in self.mdl_instructions['state_derivatives']:
            mdl.state_derivatives(dx, x, v_red)

        for mdl in self.dyn_mdls:
            mdl._store_output = False

        return dx

    def solve_algebraic(self, t, x, v_0=None):
        '''
        Solves algebraic equations given states
        :param t:
        :param x:
        :return:
        '''
        i_inj = np.zeros(self.n_bus_red, dtype=complex)
        for mdl in self.mdl_instructions['current_injections']:
            bus_idx_red, i_inj_mdl = mdl.current_injections(x, None)
            np.add.at(i_inj, bus_idx_red, i_inj_mdl)

        y_var = np.zeros((self.n_bus,) * 2, dtype=complex)
        for mdl in self.mdl_instructions['dyn_var_adm']:
            data, (row_idx, col_idx) = mdl.dyn_var_adm(x, None)
            sp_mat = sp.csr_matrix((data.flatten(), (row_idx.flatten(), col_idx.flatten())), shape=(self.n_bus,) * 2)
            y_var += sp_mat.todense()
        y_var = sp.csr_matrix(y_var)

        if len(self.mdl_instructions['apparent_power_injections']) == 0:
            return sp_linalg.spsolve(self.y_bus_red + y_var + self.y_bus_red_mod, i_inj)

        y_bus = self.y_bus_red + y_var + self.y_bus_red_mod
        s_inj = np.zeros(self.n_bus_red, dtype=complex)
        for mdl in self.mdl_instructions['apparent_power_injections']:
            bus_idx_red, s_inj_mdl = mdl.apparent_power_injections(x, None)
            np.add.at(s_inj, bus_idx_red, s_inj_mdl)
    
        v_abs_idx = slice(self.n_bus)
        v_ang_idx = slice(self.n_bus, 2*self.n_bus)
        
        def f(x):
            v = x[v_abs_idx]*np.exp(1j*x[v_ang_idx])
            f_complex = y_bus.dot(v)*np.conj(v) - i_inj*np.conj(v) - np.conj(s_inj)
            return np.concatenate([f_complex.real, f_complex.imag])
        
        # v_0 = None
        v_0 = v_0 if v_0 is not None else np.ones_like(self.v0)
        x_alg = np.concatenate([abs(v_0), np.angle(v_0)])

        with warnings.catch_warnings():
            # Cause all warnings to always be triggered.
            warnings.filterwarnings('error')
            try:
                x_alg = fsolve(f, x_alg, xtol=1e-10)
            except RuntimeWarning:
                x_alg *= np.nan
                print('Warning: Power flow did not converge.')
                # raise Warning('''Singular jacobian when solving
                    # algebraic equations''') 



        # Implementation of Newton's method. Turned out a bit slower than fsolve
        #tol = 1e-6
        #max_it = 2000
        #it = 0
        # x_alg[v_ang_idx] %= 2*np.pi

          
        # if False:    
            # print((np.angle(i_inj) - np.angle(i_inj)[0])[:4])
            error = np.linalg.norm(f(x_alg))
            # while error > tol and it < max_it:
            #     A = sp.csr_matrix(dps_uf.jacobian_num(f, x_alg))
            #     # A += np.random.randn(2*self.n_bus, 2*self.n_bus)
            #     # if False:  # np.linalg.cond(A) < 1e-10:
            #         # raise Exception('''Singular jacobian when solving
            #             # algebraic equations''') 
            #     b = - f(x_alg)
            #     try:
            #         dx = sp_linalg.spsolve(A, b)
            #     except sp.linalg.MatrixRankWarning:
            #         # warnings.filterwarnings('default')
            #         raise Warning('''Singular jacobian when solving
            #             algebraic equations''') 

            #     x_alg += dx

            #     # x_alg[v_ang_idx] %= 2*np.pi
            #     # x_alg[v_abs_idx][x_alg[v_abs_idx] < 0.7] = 0.7
            #     # x_alg[v_abs_idx][x_alg[v_abs_idx] > 1.3] = 1.3


            #     error = np.linalg.norm(f(x_alg))
            #     it += 1
            # #     # print(f"{it} \t {error:.6f}")
            
            # if error > tol:
                # raise Warning('''Solution of algebraic equations did not converge
                    # due to apparent power injections.''')
                # print('''Warning: ''')
                # x_alg *= np.nan
                # self.it_prev = it

        v = x_alg[v_abs_idx]*np.exp(1j*x_alg[v_ang_idx])
        return v


    def ode_fun(self, t, x, v_0=None):
        '''
        Can be integrated with any ODE-integration method (e.g. Euler, Runge-Kutta etc.)
        :param t:
        :param x:
        :return:
        '''
        v_red = self.solve_algebraic(t, x, v_0=v_0)

        return self.state_derivatives(t, x, v_red)


# if __name__ == '__main__':

#     from collections import defaultdict
#     import time
#     import tops.solvers as dps_sol
#     import sys

#     # Load model
#     import tops.ps_models.k2a as model_data

#     importlib.reload(model_data)
#     model = model_data.load()

#     # Power system model
#     ps = dps.PowerSystemModel(model=model)
#     ps.init_dyn_sim()
#     print(max(abs(ps.state_derivatives(0, ps.x_0, ps.v_0))))

#     t_end = 10
#     x_0 = ps.x_0.copy()

#     # Solver
#     sol = dps_sol.ModifiedEulerDAE(ps.state_derivatives, ps.solve_algebraic, 0, x_0, t_end, max_step=5e-3)

#     # Initialize simulation
#     t = 0
#     result_dict = defaultdict(list)
#     t_0 = time.time()

#     sc_bus_idx = ps.gen['GEN'].bus_idx_red['terminal'][0]

#     # Run simulation
#     while t < t_end:
#         sys.stdout.write("\r%d%%" % (t / (t_end) * 100))

#         # Short circuit
#         if t >= 1 and t <= 1.05:
#             ps.y_bus_red_mod[(sc_bus_idx,) * 2] = 1e6
#         else:
#             ps.y_bus_red_mod[(sc_bus_idx,) * 2] = 0

#         # Simulate next step
#         result = sol.step()
#         x = sol.y
#         t = sol.t

#         dx = ps.ode_fun(0, ps.x_0)

#         # Store result
#         result_dict['Global', 't'].append(sol.t)
#         [result_dict[tuple(desc)].append(state) for desc, state in zip(ps.state_desc, x)]
#         [result_dict[tuple(desc)].append(state) for desc, state in zip(ps.state_desc_der, dx)]

#     print('Simulation completed in {:.2f} seconds.'.format(time.time() - t_0))

#     plt.figure()
#     state = 'speed'
#     for i, gen in enumerate(ps.gen['GEN'].par['name']):
#         plt.plot(result_dict[('Global', 't')], result_dict[(gen, f'{state}')], color=f'C{i}', alpha=0.5)
#     plt.show()
