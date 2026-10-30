import numpy as np

import tops.utility_functions as dps_uf
import tops.dyn_models as mdl_lib
from tops.model_data import load_model_data, find_model_class, read_table
from tops.events import AdmittanceChange, BusEvents, DeadBusWarning
import tops.power_flow as pf
import scipy.sparse as sp
from scipy.sparse import linalg as sp_linalg
from scipy.sparse import csgraph
from scipy.optimize import fsolve

import warnings


class PowerSystemModel:
    def __init__(self, model, user_mdl_lib=None):
        self.user_mdl_lib = user_mdl_lib
        model = load_model_data(model)
        self.model = model

        self.perform_kron_reduction = False
        self.pf_max_it = 10
        self.pf_tol = 1e-8
        self.pin_dead_buses = True  # solve dead buses (see dead_buses) as 0 V, with a DeadBusWarning
        self.dead_bus_rtol = 1e-12  # a bus is dead when its admittance is below this fraction of its initial value

        self.s_n = model['base_mva']
        self.f_n = model['f']
        self.slack_bus = model['slack_bus'] if 'slack_bus' in model else None
        self.buses = read_table(model['buses'])
        self.n_bus = len(self.buses)

        self.y_bus_lf = None
        self.power_flow_ready = False
        
        self.setup_ready = False
        self.initialization_ready = False

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

        self.bus_events = None

    def add_model_data(self, model_data):
        for key, val in model_data.items():
            if not isinstance(val, dict):
                continue
            category_key = key
            category = val
            for mdl_key, mdl_data_raw in category.items():
                mdl_class = find_model_class(category_key, mdl_key, self.user_mdl_lib)
                if mdl_class is None:
                    continue

                mdl_data = read_table(mdl_data_raw)
                mdl = mdl_class(mdl_data, self.sys_data)
                if not hasattr(self, category_key):
                    setattr(self, category_key, dps_uf.dotdict())
                    self.dyn_mdls_dict[category_key] = {}
                getattr(self, category_key).update({mdl_key: mdl})
                self.dyn_mdls_dict[category_key].update({mdl_key: mdl})
                self.dyn_mdls.extend(mdl_lib.utils.get_submodules(mdl))

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
                buses_red.extend(mdl.reduced_system())

            bus_idx_red = dps_uf.lookup_strings(buses_red, self.buses['name'])
        else:
            bus_idx_red = np.arange(self.n_bus)

        self.algebraic_equations_are_linear = len(
            self.mdl_instructions['apparent_power_injections']) == 0
        
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
            np.add.at(y_lf, (row_idx.flatten(), col_idx.flatten()), data.flatten())

        self.y_bus_lf = y_lf
        return y_lf

    def build_y_bus_dyn(self):

        y_dyn = np.zeros((self.n_bus,) * 2, dtype=complex)
        for mdl in self.mdl_instructions['dyn_const_adm']:
            data, (row_idx, col_idx) = mdl.dyn_const_adm()
            np.add.at(y_dyn, (row_idx.flatten(), col_idx.flatten()), data.flatten())

        self.y_bus_dyn = y_dyn
        return y_dyn


    def power_flow(self, print_output=False):

        if not self.setup_ready:
            self.setup()

        if self.y_bus_lf is None:
            self.build_y_bus_lf()

        pq_mdls = self.mdl_instructions['load_flow_pq']
        pv_mdls = self.mdl_instructions['load_flow_pv']

        if not self.slack_bus:
            sl_idx = pf.default_slack_bus_idx(pv_mdls)
            self.slack_bus = self.buses[sl_idx]['name']
        else:
            sl_idx = dps_uf.lookup_strings(self.slack_bus, self.buses['name'])

        self._check_islands(sl_idx)

        result = pf.solve_power_flow(self.y_bus_lf, pq_mdls, pv_mdls, self.s_n, sl_idx, self.pf_tol, self.pf_max_it)

        self.v_0 = result.v
        self.v0 = self.v_0
        self.s_0 = result.s
        if result.slack_unit is not None:
            self.slack_unit = result.slack_unit
        self.load_flow_soln = result.unit_injections

        if result.converged:
            self.power_flow_ready = True

    def _check_islands(self, slack_idx):
        """Raise if branches that are out of service ('connected' column) cut buses off from the slack bus."""
        def reachable(only_connected):
            rows, cols = [], []
            for mdl in self.dyn_mdls:
                if hasattr(mdl, 'connected') and 'from_bus' in mdl.bus_idx.dtype.names:
                    keep = mdl.connected if only_connected else slice(None)
                    rows.extend(mdl.bus_idx['from_bus'][keep])
                    cols.extend(mdl.bus_idx['to_bus'][keep])
            graph = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(self.n_bus,) * 2)
            _, labels = csgraph.connected_components(graph, directed=False)
            return labels == labels[slack_idx]

        cut_off = reachable(only_connected=False) & ~reachable(only_connected=True)
        if cut_off.any():
            raise ValueError(f"Branches with connected=False cut the buses {list(self.buses['name'][cut_off])} off "
                             f"from the slack bus {self.buses['name'][slack_idx]}, so the load flow cannot be solved.")

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
                # Computed from local indices (not +=), so that repeated calls give the same result
                mdl.state_idx_global[field] = mdl.state_idx[field] + mdl.idx.start
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
            if not hasattr(mdl, 'init_from_connections'):
                continue
            output_values = np.zeros(mdl.n_units, [(field, float) for field in mdl.output_list()])
            for output_key, conn in connections.items():
                init_val = np.zeros(mdl.n_units)
                for c in conn:
                    input_fun = lambda x, v:  self.dyn_mdls_dict[c['container']][c['mdl']]._input_values[c['input']]
                    np.add.at(init_val, c['source_idx'], input_fun(None, None)[c['dest_idx']])
                output_values[output_key] = init_val

            mdl.init_from_connections(self.x_0, self.v_0, output_values)

        self.bus_events = BusEvents(self.buses['name'], self.bus_idx_red)
        self._row_admittance_0 = self._row_admittance(self.x0)
        self._dead_idx = np.zeros(0, dtype=int)
        self._dead_dirty = True

        self.initialization_ready = True

    def get_events(self, category, mdl_key=None):
        """Return the events object (see tops.events) of a model, or of the buses if category is 'buses'."""
        if category == 'buses':
            return self.bus_events
        mdl = self.dyn_mdls_dict.get(category, {}).get(mdl_key)
        if mdl is None:
            raise KeyError(f'Event target {category}:{mdl_key} not found.')
        if mdl.events is None:
            raise ValueError(f'Model {category}:{mdl_key} has no events.')
        return mdl.events

    def apply_event(self, category, mdl_key, unit, action, **kwargs):
        """Apply an event immediately, e.g. apply_event('lines', 'Line', 'L1-2', 'disconnect')."""
        for effect in self.get_events(category, mdl_key).apply(unit, action, **kwargs):
            self.apply_effect(effect)
        self._dead_dirty = True  # also for events without effect, such as those of a DynTrafo

    def apply_effect(self, effect):
        self._dead_dirty = True
        if isinstance(effect, AdmittanceChange):
            self.y_bus_red = self.y_bus_red + sp.csr_matrix(
                (effect.data, (effect.rows, effect.cols)), shape=(self.n_bus_red,) * 2)
        else:
            raise TypeError(f'Unknown event effect: {type(effect).__name__}')

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

    def _y_var(self, x):
        """Admittances that change with the states, as a matrix."""
        y_var = sp.csr_matrix((self.n_bus,) * 2, dtype=complex)
        for mdl in self.mdl_instructions['dyn_var_adm']:
            data, (row_idx, col_idx) = mdl.dyn_var_adm(x, None)
            y_var += sp.csr_matrix((data.flatten(), (row_idx.flatten(), col_idx.flatten())), shape=(self.n_bus,) * 2)
        return y_var

    def _row_admittance(self, x):
        """Sum of the magnitudes of the entries of each row of the admittance matrix at the states x."""
        y = self.y_bus_red + self._y_var(x) + self.y_bus_red_mod
        return np.asarray(abs(y).sum(axis=1)).ravel()

    def _dead_bus_idx(self, x):
        # Not zero but a tiny fraction of what the row was: stamps taken out one by one leave rounding behind
        return np.flatnonzero(self._row_admittance(x) <= self.dead_bus_rtol*self._row_admittance_0)

    def dead_buses(self, x=None):
        """Names of the buses that have lost all their admittance since the initialisation, e.g. a bay node whose
        switches are all open. The variable admittances are taken at the states x (default: the initial states)."""
        x = self.x0 if x is None else x
        return self.buses['name'][self.bus_idx_red[self._dead_bus_idx(x)]]

    def _dead_buses_now(self, x):
        """Indices of the dead buses. They are looked for again only after an event, the only thing that can change
        them, and a DeadBusWarning is given when they are not the same as before."""
        if self._dead_dirty:
            dead = self._dead_bus_idx(x)
            if not np.array_equal(dead, self._dead_idx):
                self._dead_idx = dead
                self._live = np.ones(self.n_bus_red)
                self._live[dead] = 0
                if len(dead):
                    names = list(self.buses['name'][self.bus_idx_red[dead]])
                    warnings.warn(DeadBusWarning(f'Buses {names} have no admittance and are pinned to 0 V.'))
            self._dead_dirty = False
        return self._dead_idx

    def _pin_dead_buses(self, y_bus):
        """The admittance matrix without the rows and columns of the dead buses, which get a one on the diagonal."""
        live = sp.diags(self._live)
        return live @ y_bus @ live + sp.diags(1 - self._live)

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

        y_var = self._y_var(x)
        if self.algebraic_equations_are_linear:
            y_bus = self.y_bus_red + y_var + self.y_bus_red_mod
            if self.pin_dead_buses and len(self._dead_buses_now(x)):
                y_bus, i_inj = self._pin_dead_buses(y_bus), i_inj*self._live
            return sp_linalg.spsolve(y_bus, i_inj)

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

        converged = True
        with warnings.catch_warnings():
            # Turn fsolve's RuntimeWarning into an exception, so non-convergence can be detected
            warnings.filterwarnings('error')
            try:
                x_alg = fsolve(f, x_alg, xtol=1e-10)
            except RuntimeWarning:
                x_alg *= np.nan
                converged = False

        if not converged:
            # Warn outside the catch_warnings block, where warnings are turned into errors
            warnings.warn('Solution of algebraic equations did not converge.', RuntimeWarning)

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

