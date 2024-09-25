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
    import my_k2a as model_data
    importlib.reload(model_data)
    model = model_data.load()
    model['loads'] = {'DynamicLoad': model['loads']}

    # Power system model
    ps = dps.PowerSystemModel(model=model)
    ps.init_dyn_sim()
    print(max(abs(ps.state_derivatives(0, ps.x_0, ps.v_0))))

    t_end = 50
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

        if 1 <= t:
            ps.loads['DynamicLoad'].set_input('g_setp', 1.6, 1)
        # if 10 <= t:
        #     ps.loads['DynamicLoad'].set_input('b_setp', -0.2, 1)

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
        

    print('Simulation completed in {:.2f} seconds.'.format(time.time() - t_0))

    # fig = plt.figure()
    # plt.plot(res['t'], np.abs(res['v']))
    # plt.xlabel('Time [s]')
    # plt.ylabel('Bus voltage')

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
    # plt.plot(res['t'], np.abs(res['load_P']))
    # plt.xlabel('Time [s]')
    # plt.ylabel('MW')

    # fig = plt.figure()
    # plt.plot(res['t'], np.abs(res['load_Q']))
    # plt.xlabel('Time [s]')
    # plt.ylabel('MVA')
    
    # plt.show()
    
    plt.figure()
    #plt.plot(res['t'], res['gen_speed'],label = ps.gen['GEN'].par['name'])
    plt.plot(res['t'],50+50*np.mean((res['gen_speed']),axis = 1), label = 'f')
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency')
    plt.grid()
    plt.legend()
    plt.show()