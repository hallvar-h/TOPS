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
    # model['avr']['SEXS_mod'] = model['avr']['SEXS']
    # del(model['avr']['SEXS'])
    model['pll'] = {'PLL1':[
        ['name', 'T_filter', 'bus'],
        *[[f'PLL{i}', 0.1, bus[0]] for i, bus in enumerate(model['buses'][1:])],
    ]}

    # model['pll'] = {'VSC': [
    #     ['name', 'T_pll', 'bus'],
    #     *[[f'PLL{i}', 0.1, bus[0]] for i, bus in enumerate(model['buses'][1:])],
    # ]}
    # model['avr'] = {}
    # model['pss'] = {}
    # model['gov'] = {}

    import examples.user_models.user_lib as user_lib

    # Power system model
    ps = dps.PowerSystemModel(model=model, user_mdl_lib=user_lib)

# if False:
    ps.init_dyn_sim()

    print(max(abs(ps.state_derivatives(0, ps.x_0, ps.v_0))))

    t_end = 10
    x_0 = ps.x_0.copy()

    # Solver
    sol = dps_sol.ModifiedEulerDAE(ps.state_derivatives, ps.solve_algebraic, 0, x_0, t_end, max_step=5e-3)

    # Initialize simulation
    t = 0
    result_dict = defaultdict(list)
    t_0 = time.time()

    event_flag = True

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
        result_dict['Global', 't'].append(sol.t)
        [result_dict[tuple(desc)].append(state) for desc, state in zip(ps.state_desc, x)]
        [result_dict[tuple(desc)].append(state) for desc, state in zip(ps.state_desc_der, dx)]

        [result_dict[('v', f'{i}')].append(v) for i, v in enumerate(ps.red_to_full.dot(v))]

        [result_dict[(f'PLL{i}', 'output')].append(output) for i, output in enumerate(ps.pll['PLL1'].output(x, v).copy())]
        [result_dict[(f'PLL{i}', 'freq_est')].append(output) for i, output in enumerate(ps.pll['PLL1'].freq_est(x, v).copy())]
        # result_dict[('PLL1', 'output')].append(ps.pll['PLL1'].output(x, v))

    print('Simulation completed in {:.2f} seconds.'.format(time.time() - t_0))

    index = pd.MultiIndex.from_tuples(result_dict)
    result = pd.DataFrame(result_dict, columns=index)

    plt.figure()
    state = 'speed'
    for i, gen in enumerate(ps.gen['GEN'].par['name']):
        plt.plot(result_dict[('Global', 't')], result_dict[(gen, f'{state}')], color=f'C{i}', alpha=0.5)


    plt.figure()
    for i in range(ps.n_bus):
        plt.plot(result_dict[('Global', 't')], result_dict[(f'PLL{i}', 'output')], '-', color=f'C{i}')
        plt.plot(result_dict[('Global', 't')], np.angle(result_dict[('v', f'{i}')]), '--', color=f'C{i}')

    # state = 'angle'
    # for i, gen in enumerate(ps.gen['GEN'].par['name']):
    #     plt.plot(result_dict[('Global', 't')], result[(gen, f'{state}')], color=f'C{i}', alpha=0.5)
    # plt.show()

    # plt.figure()
    fig, ax = plt.subplots(4)
    for i in range(4):
        ax[i].plot(result_dict[('Global', 't')], result_dict[(f'PLL{i}', 'freq_est')], '--', color=f'C{i}')
        # plt.plot(result_dict[('Global', 't')], np.angle(result_dict[('v', f'{i}')]), '-', color=f'C{i}')

    state = 'speed'
    for i, gen in enumerate(ps.gen['GEN'].par['name']):
        ax[i].plot(result_dict[('Global', 't')], result[(gen, f'{state}')], color=f'C{i}')
    plt.show()