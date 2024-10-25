from collections import defaultdict
import pandas as pd
import sys
from collections import defaultdict
import matplotlib.pyplot as plt
import time
import tops.dynamic as dps
import tops.solvers as dps_sol
import importlib
importlib.reload(dps)
import numpy as np

if __name__ == '__main__':

    # Load model
    import inertia.Basecase as model_data
    importlib.reload(model_data)
    model = model_data.load()
    model['loads'] = {'DynamicLoad': model['loads']}

    # Power system model
    ps = dps.PowerSystemModel(model=model)
    ps.init_dyn_sim()
    print(max(abs(ps.state_derivatives(0, ps.x_0, ps.v_0))))

    t_end = 20
    x_0 = ps.x_0.copy()

    # Solver
    sol = dps_sol.ModifiedEulerDAE(ps.state_derivatives, ps.solve_algebraic, 0, x_0, t_end, max_step=5e-3)

    # Initialize simulation
    t = 0
    res = defaultdict(list)
    t_0 = time.time()

    sc_bus_idx = ps.gen['GEN'].bus_idx_red['terminal'][0]


    result_dict = defaultdict(list)
    # Run simulation
    while t < t_end:
        sys.stdout.write("\r%d%%" % (t/(t_end)*100))

        # if 1 <= t:
        #     ps.loads['DynamicLoad'].set_input('g_setp', 1.3, 0)
        if 10 <= t:
            #ps.loads['DynamicLoad'].set_input('g_setp', 0, 2)
            ps.loads['DynamicLoad'].par['P'][1] = 0
            
        # Simulate next step
        result = sol.step()
        x = sol.y
        v = sol.v
        t = sol.t

        dx = ps.ode_fun(0, ps.x_0)

        # Store result
        res['t'].append(t)
        res['gen_speed'].append(ps.gen['GEN'].speed(x, v).copy())
        res['v'].append(v.copy())
        res['gen_I'].append(ps.gen['GEN'].I(x, v).copy())
        res['load_I'].append(ps.loads['DynamicLoad'].I(x, v).copy())
        res['load_P'].append(ps.loads['DynamicLoad'].P(x, v).copy())
        res['load_Q'].append(ps.loads['DynamicLoad'].Q(x, v).copy())
        res['angle'].append(ps.gen['GEN'].angle(x, v).copy())
        result_dict['Global', 't'].append(sol.t)
        [result_dict[tuple(desc)].append(state) for desc, state in zip(ps.state_desc, x)]

    index = pd.MultiIndex.from_tuples(result_dict)
    result = pd.DataFrame(result_dict, columns=index)
    print('Simulation completed in {:.2f} seconds.'.format(time.time() - t_0))






    #Average of simulated:
    list_frequency_sim = list()
    # speed_results = result.xs(key='speed', axis='columns', level=1).drop(columns=['Virtual gen']) if add_virtual_gen \
    #            else result.xs(key='speed', axis='columns', level=1)
    speed_results = result.xs(key='speed', axis='columns', level=1)
    list_frequency_sim = speed_results.mean(axis=1)



    #Plotting the results
    timestamps = result[('Global', 't')]
    fig, ax = plt.subplots(1)
    fig.suptitle('Generator speed deviation', fontsize=20)
    ax.plot(timestamps, list_frequency_sim * ps.model['f'], label='Simulated')
    ax.plot(timestamps, np.linspace(-0.1, -0.1, num=len(timestamps)), linestyle="dashed", color='k')
    ax.plot(timestamps, np.linspace( 0.1,  0.1, num=len(timestamps)), linestyle="dashed", color='k')
    ax.set_ylabel('Deviation (Hz)', fontsize=15)
    ax.set_xlabel('Time (s)', fontsize=15)
    ax.set_xlim(0, t_end)
    ax.legend()
    plt.show()

    # fig = plt.figure()
    # plt.plot(res['t'], [i*50 for i in res['gen_speed']], label = ps.gen['GEN'].par['name'])
    # plt.xlabel('Time [s]')
    # plt.ylabel('Frequency deviations')
    # plt.legend()
    # fig = plt.figure()
    # plt.plot(res['t'], [50 - i*50 for i in res['gen_speed']], label = ps.gen['GEN'].par['name'])
    # plt.xlabel('Time [s]')
    # plt.ylabel('Frequency')
    # plt.legend()
    # plt.show()

    # fig = plt.figure()
    # plt.plot(res['t'], np.abs(res['v']), label = ps.buses['name'])
    # plt.xlabel('Time [s]')
    # plt.ylabel('Bus voltage')
    # plt.legend()
    # fig = plt.figure()
    # # Note: Geneartor current is higher than load current due to transformers
    # plt.plot(res['t'], np.abs(res['gen_I']))
    # plt.xlabel('Time [s]')
    # plt.ylabel('Generator current [A]')

    # fig = plt.figure()
    # plt.plot(res['t'], np.abs(res['load_I']))
    # plt.xlabel('Time [s]')
    # plt.ylabel('Load current [A]')
    
    # fig = plt.figure()
    # plt.plot(res['t'], res['load_P'], label = ps.loads['DynamicLoad'].par['name'])
    # plt.xlabel('Time [s]')
    # plt.ylabel('MW')
    # plt.legend()
    # fig = plt.figure()
    # plt.plot(res['t'], res['load_Q'], label = ps.loads['DynamicLoad'].par['name'])
    # plt.xlabel('Time [s]')
    # plt.ylabel('MVA')
    # plt.legend()
    # plt.show()
    