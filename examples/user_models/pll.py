import sys
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import time
import dynpssimpy.dynamic as dps
import dynpssimpy.solvers as dps_sol
import importlib
importlib.reload(dps)
import pandas as pd

if __name__ == '__main__':

    # Load model
    import dynpssimpy.ps_models.k2a as model_data
    importlib.reload(model_data)
    model = model_data.load()
    
    model['pll'] = {'PLL1':[
        ['name', 'T_filter', 'bus'],
        *[[f'PLL{i}', 0.1, bus[0]] for i, bus in enumerate(model['buses'][1:])],
    ]}

    import examples.user_models.user_lib as user_lib

    # Power system model
    ps = dps.PowerSystemModel(model=model, user_mdl_lib=user_lib)
    ps.init_dyn_sim()

    print(max(abs(ps.state_derivatives(0, ps.x_0, ps.v_0))))

    t_end = 10
    x_0 = ps.x_0.copy()

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
        t = sol.t
        v = sol.v

        res['t'].append(sol.t)
        res['PLL_freq_est'].append(ps.pll['PLL1'].freq_est(x, v).copy())
        res['gen_speed'].append(ps.gen['GEN'].speed(x, v).copy())

    print('Simulation completed in {:.2f} seconds.'.format(time.time() - t_0))

    fig, ax = plt.subplots(4, sharex=True)
    for i, (pll_freq_est, gen_speed) in enumerate(zip(np.array(res['PLL_freq_est']).T, np.array(res['gen_speed']).T)):
        ax[i].plot(res['t'], pll_freq_est, color=f'C{i}', linestyle='--', label=f'PLL freq est Bus {i+1}')
        ax[i].plot(res['t'], gen_speed, color=f'C{i}', label=f'Gen {i+1} speed')
        ax[i].set_ylabel(f'Bus {i}')
        ax[i].legend()
    ax[-1].set_xlabel('Time [s]')
    plt.show()