import sys
from collections import defaultdict
import matplotlib.pyplot as plt
import time
import dynpssimpy.dynamic as dps
import dynpssimpy.solvers as dps_sol
import importlib
importlib.reload(dps)


from dynpssimpy.dyn_models.utils import get_submodules, auto_init
import numpy as np


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

    t_end = 20
    # x_0 = ps.x_0.copy()

    # Solver
    sol = dps_sol.ModifiedEulerDAE(ps.state_derivatives, ps.solve_algebraic, 0, x_0, t_end, max_step=5e-3)

    # Initialize simulation
    t = 0
    res = defaultdict(list)
    t_0 = time.time()

    load_bus_idx = ps.loads['Load'].bus_idx_red['terminal'][0]
    y_load_0 = ps.loads['Load'].y_load.real[0]

    # Run simulation
    while t < t_end:
        sys.stdout.write("\r%d%%" % (t/(t_end)*100))

        # Short circuit
        if t > 1:
            ps.y_bus_red_mod[(load_bus_idx,) * 2] = y_load_0*0.1

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