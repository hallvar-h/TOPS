import sys
from collections import defaultdict
import matplotlib.pyplot as plt
import time
import dynpssimpy.dynamic as dps
import dynpssimpy.solvers as dps_sol
import importlib
importlib.reload(dps)

if __name__ == '__main__':

    # Load model
    import dynpssimpy.ps_models.k2a as model_data
    importlib.reload(model_data)
    model = model_data.load()
    # model['gov'] = {}
    # model['gov']['TGOV1'] = model['gov']['TGOV1'][:2]
    model['gov'] = {'HYGOV': [
        ['name',    'gen',  'R',    'r',    'T_f',  'T_r',  'T_g',  'A_t',  'T_w',  'q_nl'],  # ,   'G_min',    'G_max']
        ['HYGOV1',  'G1',   0.05,   0.5,    0.1,    10,     1,      1,      5,      0.1],  # ,      0.1,        0.9]
        ['HYGOV2',  'G2',   0.05,   0.5,    0.1,    10,     1,      1,      5,      0.1],  # ,      0.1,        0.9]
        ['HYGOV3',  'G3',   0.05,   0.5,    0.1,    10,     1,      1,      5,      0.1],  # ,      0.1,        0.9]
        ['HYGOV4',  'G4',   0.05,   0.5,    0.1,    10,     1,      1,      5,      0.1],  # ,      0.1,        0.9]
    ]}

    # Power system model
    ps = dps.PowerSystemModel(model=model)
    ps.init_dyn_sim()
    
    # ps.gov['TGOV1'].output(x0, v0)
    print(max(abs(ps.state_derivatives(0, ps.x_0, ps.v_0))))
    x0, v0 = x_0, v_0 = x, v = ps.x0, ps.v_0

    # self = ps.gov['HYGOV']
    self = ps.gov['HYGOV']
    output_0 = ps.gen['GEN']._input_values['P_m']
    
    from dynpssimpy.dyn_models.utils import get_submodules
    submodules = get_submodules(self)
    submodules
    self.input(x, v)
    
    
    # Find states belonging to model:
    state_idx = []
    state_idx_local = []
    n_states = 0
    start_idx = 0
    for submodule in submodules:
        state_list = submodule.state_list()
        if len(state_list) > 0:
            state_idx.append(submodule.idx)
            n_states_mdl = submodule.idx.stop - submodule.idx.start
            n_states += n_states_mdl
            state_idx_local.append(slice(start_idx, start_idx + n_states_mdl))
            start_idx += n_states_mdl
    
    init_val = 1
    import numpy as np
    n_int_par = len(self.int_par_list())
    n_units = self.n_units
    n_sol = n_states + n_int_par*n_units
    int_par_idx = slice(n_states, None)
    x_test = np.ones(n_sol)

    def ode_fun_mdl(x_test):
        int_par = x_test[int_par_idx]
        self.int_par[:] = int_par
        x_all_test = np.zeros_like(x)
        for idx, idx_local in zip(state_idx, state_idx_local):
            x_all_test[idx] = x_test[idx_local]
        
        self.int_par
        dx_all = ps.state_derivatives(0, x_all_test, v)
        dx_mdl = []
        for idx in state_idx:
            dx_mdl.append(dx_all[idx])

        output_err = self.output(x_all_test, v) - output_0
        self.input(x_all_test, v)
        
        
        return np.concatenate(dx_mdl + [output_err])

    x_test = np.concatenate([ps.x0[:n_states], self.int_par['bias']])
    
    x_test[:] = 1
    ode_fun_mdl(x_test)
    from scipy.optimize import least_squares
    sol = least_squares(ode_fun_mdl, np.ones(n_sol))
    x_sol = sol['x']
    ode_fun_mdl(x_sol)
    x_sol_all = x.copy()
    for idx, idx_local in zip(state_idx, state_idx_local):
        x_sol_all[idx] = x_sol[idx_local]
    self.output(x_sol_all, v)
    output_0

    # ps.x0[:n_states]
    x_0 = x_sol_all
    print(max(abs(ps.state_derivatives(0, x_0, ps.v_0))))

    self.output(x_0, v0)

    
    # def ode_fun(x_test):

    t_end = 10
    # x_0 = ps.x_0.copy()

    # Solver
    sol = dps_sol.ModifiedEulerDAE(ps.state_derivatives, ps.solve_algebraic, 0, x_0, t_end, max_step=5e-3)

    # Initialize simulation
    t = 0
    res = defaultdict(list)
    t_0 = time.time()

    sc_bus_idx = ps.gen['GEN'].bus_idx_red['terminal'][0]

    # Run simulation
    while t < t_end:
        sys.stdout.write("\r%d%%" % (t/(t_end)*100))

        # Short circuit
        if t >= 1 and t <= 1.05:
            ps.y_bus_red_mod[(sc_bus_idx,) * 2] = 1e6
        else:
            ps.y_bus_red_mod[(sc_bus_idx,) * 2] = 0

        # Simulate next step
        result = sol.step()
        x = sol.y
        v = sol.v
        t = sol.t

        dx = ps.ode_fun(0, ps.x_0)

        # Store result
        res['t'].append(t)
        res['gen_speed'].append(ps.gen['GEN'].speed(x, v).copy())

    print('Simulation completed in {:.2f} seconds.'.format(time.time() - t_0))

    plt.figure()
    plt.plot(res['t'], res['gen_speed'])
    plt.xlabel('Time [s]')
    plt.ylabel('Gen. speed')
    plt.show()