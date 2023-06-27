import sys
from collections import defaultdict
import matplotlib.pyplot as plt
import time
import dynpssimpy.dynamic as dps
import dynpssimpy.solvers as dps_sol
import dynpssimpy.utility_functions as dps_uf
import importlib
importlib.reload(dps)
import importlib
import numpy as np



if __name__ == '__main__':

    # Load model
    import dynpssimpy.ps_models.k2a as model_data
    importlib.reload(model_data)
    model = model_data.load()

    # model['pll'] = {
    #     'PLL1':[
    #        ['name',        'T_filter',   'bus'],
    #         *[[f'PLL{i}',   0.1,        bus[0]] for i, bus in enumerate(model['buses'][1:])],
    #     ],
    #     'PLL2':[
    #         ['name',        'K_p', 'K_i',   'bus'],
    #         *[[f'PLL{i}',   10,    1,    bus[0]] for i, bus in enumerate(model['buses'][1:])],
    #         ]
    # }


    # Power system model
    ps = dps.PowerSystemModel(model=model)
    ps.setup()
    ps.build_y_bus_lf()
    
    ps.add_model_data({'pll': {
        'PLL1': [
             ['name',        'T_filter',     'bus'   ],
            *[[f'PLL{i}',    0.01,            bus_name  ] for i, bus_name in enumerate(ps.buses['name'])],
        ],
        'PLL2': [
             ['name',        'K_p',  'K_i',  'bus'   ],
            *[[f'PLL{i}',    100,     100,      bus_name  ] for i, bus_name in enumerate(ps.buses['name'])],
        ]
    }})
    ps.init_dyn_sim()
    print(max(abs(ps.ode_fun(0, ps.x_0))))

    x0 = ps.x_0
    v0 = ps.v_0

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

        dx = ps.ode_fun(0, ps.x_0)

        # Store result
        res['t'].append(sol.t)
        res['gen_speed'].append(ps.gen['GEN'].speed(x, v).copy())
        res['v_angle'].append(np.angle(sol.v).copy())
        res['PLL1'].append(ps.pll['PLL1'].output(x, v).copy())
        res['PLL2'].append(ps.pll['PLL2'].output(x, v).copy())

        res['PLL1_freq'].append(ps.pll['PLL1'].freq_est(x, v).copy())
        res['PLL2_freq'].append(ps.pll['PLL2'].freq_est(x, v).copy())

    print('Simulation completed in {:.2f} seconds.'.format(time.time() - t_0))

    # plt.figure()
    # plt.plot(res['t'], res['PLL'])
    # plt.plot(res['t'], res['v_angle'])
    # plt.show()

    n_plt = 4
    fig, ax = plt.subplots(n_plt, sharex=True)
    for i, (pll_1, pll_2, v_angle) in enumerate(zip(np.array(res['PLL1']).T, np.array(res['PLL2']).T, np.array(res['v_angle']).T)):
        if i >= n_plt:
            break 
        ax[i].plot(res['t'], pll_1, color=f'C{i}', linestyle='--', label=f'PLL 1 angle est Bus {i+1}')
        ax[i].plot(res['t'], pll_2, color=f'C{i}', linestyle=':', label=f'PLL 2 angle est Bus {i+1}')
        ax[i].plot(res['t'], v_angle, color=f'C{i}', label=f'Voltage {i+1} angle')
        ax[i].set_ylabel(f'Bus {i}')
        ax[i].legend()
    ax[-1].set_xlabel('Time [s]')
    # plt.show()

    
    # freq_diff = np.diff(np.array(res['v_angle']).T, n=1)
    n_plt = 4
    fig, ax = plt.subplots(n_plt, sharex=True)
    for i, (pll_1, pll_2) in enumerate(zip(np.array(res['PLL1_freq']).T, np.array(res['PLL2_freq']).T)):  # , freq_diff)):
        if i >= n_plt:
            break

        if i in ps.gen['GEN'].bus_idx_red['terminal']:
            ax[i].plot(res['t'], np.array(res['gen_speed'])[:, i], color=f'C{i}', linestyle='-.', label=f'Gen {i+1} el. speed')

        ax[i].plot(res['t'], pll_1, color=f'C{i}', linestyle='--', label=f'PLL 1 freq est Bus {i+1}')
        ax[i].plot(res['t'], pll_2, color=f'C{i}', linestyle=':', label=f'PLL 2 freq est Bus {i+1}')
        # ax[i].plot(res['t'][:len(freq_diff_)], freq_diff_, color=f'C{i}', label=f'Diff {i+1}')  # This one was not accurate
        ax[i].set_ylabel(f'Bus {i}')
        ax[i].legend()
    ax[-1].set_xlabel('Time [s]')
    plt.show()
