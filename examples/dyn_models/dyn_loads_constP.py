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
    import tops.ps_models.k2a as model_data
    importlib.reload(model_data)
    model = model_data.load()
    model['loads'] = {'AlmostConstPLoad': [
        model['loads'][0] + ['T'],
        *[row + [0.5] for row in model['loads'][1:]]
    ]}

    # Power system model
    ps = dps.PowerSystemModel(model=model)
    ps.init_dyn_sim()
    print(max(abs(ps.state_derivatives(0, ps.x_0, ps.v_0))))

    ps.loads['AlmostConstPLoad'].v_abs_filtered.output(ps.x0, ps.v0)
    ps.loads['AlmostConstPLoad'].P_setp(ps.x0, ps.v0)
    ps.loads['AlmostConstPLoad'].Q_setp(ps.x0, ps.v0)

    t_end = 20
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
            ps.loads['AlmostConstPLoad'].set_input('P_setp', 1000, 0)
        if 10 <= t:
            ps.loads['AlmostConstPLoad'].set_input('Q_setp', +120, 1)

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

        res['p_load'].append(ps.loads['AlmostConstPLoad'].p(x, v).copy())
        res['P_setp'].append(ps.loads['AlmostConstPLoad'].P_setp(x, v).copy())
        res['q_load'].append(ps.loads['AlmostConstPLoad'].q(x, v).copy())
        res['Q_setp'].append(ps.loads['AlmostConstPLoad'].Q_setp(x, v).copy())

    print('Simulation completed in {:.2f} seconds.'.format(time.time() - t_0))

    fig = plt.figure()
    

    fig, ax = plt.subplots(3, sharex=True)
    ax[0].plot(res['t'], np.abs(res['v']))
    ax[0].set_xlabel('Time [s]')
    ax[0].set_ylabel('Bus voltage')
    
    ax[1].plot(res['t'], np.array(np.abs(res['p_load'])*ps.sys_data['s_n']))
    ax[1].plot(res['t'], np.array(np.abs(res['P_setp'])))
    ax[1].set_xlabel('Time [s]')
    ax[1].set_ylabel('Active power')

    ax[2].plot(res['t'], np.array(np.abs(res['q_load'])*ps.sys_data['s_n']))
    ax[2].plot(res['t'], np.array(np.abs(res['Q_setp'])))
    ax[2].set_xlabel('Time [s]')
    ax[2].set_ylabel('Reactive power')
    
    plt.show()
    