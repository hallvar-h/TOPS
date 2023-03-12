import sys
from collections import defaultdict
import matplotlib.pyplot as plt
import time
import dynpssimpy.dynamic as dps
import dynpssimpy.solvers as dps_sol
import importlib
importlib.reload(dps)
from scipy.optimize import least_squares


from dynpssimpy.dyn_models.utils import get_submodules, auto_init
import numpy as np


def auto_init(mdl, x0, v0, output_0):
    submodules = get_submodules(mdl)
    n_units = mdl.n_units
    n_states_all = len(x0)

    included_states = np.zeros(n_states_all, dtype=bool)
    for submodule in submodules:
        included_states[submodule.idx] = True

    n_states = sum(included_states)
    n_int_par = len(mdl.int_par_list())
    
    n_int_par = len(mdl.int_par_list())
    n_units = mdl.n_units
    n_sol = n_states + n_int_par*n_units
    state_idx = slice(n_states)
    int_par_idx = slice(n_states, None)
    x_test = np.ones(n_sol)

    def ode_fun_mdl(x_test):
        int_par = x_test[int_par_idx]
        mdl.int_par[:] = int_par
        x_all_test = np.zeros_like(x0)

        x_all_test[included_states] = x_test[state_idx]
        
        for submodule in submodules:
            submodule.reset_outputs()
            submodule._store_output = True

        dx_all = np.zeros(n_states_all)
        for submodule in submodules:
            if hasattr(submodule, 'state_derivatives'):
                submodule.state_derivatives(dx_all, x_all_test, v0)

        for submodule in submodules:
            submodule._store_output = False

        dx_mdl = dx_all[included_states]

        output_err = mdl.output(x_all_test, v0) - output_0
        return np.concatenate([dx_mdl] + [output_err])

    x_test = np.concatenate([x0[:n_states], mdl.int_par['bias']])
    x_test[:] = 1

    x_sol_best = np.ones(n_sol)
    err_best = 1e6
    for init_conditions in [np.ones(n_sol), np.zeros(n_sol), np.random.randn(n_sol)]:
        try:
            sol = least_squares(ode_fun_mdl, init_conditions)
            err = np.linalg.norm(ode_fun_mdl(sol['x']))
            if err < err_best:
                x_sol_best = sol['x']
                err_best = err
        except ValueError:
            continue

    x_sol_all = x0.copy()
    x_sol_all[included_states] = x_sol_best[state_idx]

    mdl.int_par[:] = x_sol_best[int_par_idx]
        
    if np.linalg.norm(mdl.output(x_sol_all, v0) - output_0) > 1e-6:
        print(f'Model {mdl.__class__} was potentially not initialized properly (error: {err}).')
    # assert max(abs(ps.state_derivatives(0, x_sol_all, ps.v_0))) < 1e-6
    x0[:] = x_sol_all
    x0


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

    # mdl = ps.gov['HYGOV']
    mdl = ps.gov['HYGOV']
    output_0 = ps.gen['GEN']._input_values['P_m']
    
    auto_init(mdl, x0, v0, output_0)    
    print(max(abs(ps.state_derivatives(0, ps.x_0, ps.v_0))))
if False:

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