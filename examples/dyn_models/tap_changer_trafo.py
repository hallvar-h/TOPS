import sys
from collections import defaultdict
import matplotlib.pyplot as plt
import time
import dynpssimpy.dynamic as dps
import dynpssimpy.solvers as dps_sol
import importlib
importlib.reload(dps)
import numpy as np

if __name__ == '__main__':

    # Load model
    import dynpssimpy.ps_models.k2a as model_data
    importlib.reload(model_data)
    model = model_data.load()
    model['trafos'] = {'DynTrafo': [
        model['transformers'][0] + ['ratio_from', 'ratio_to'],
        *[row + [1, 1] for row in model['transformers'][1:]],
    ]}

    print(model['trafos'])
    model.pop('transformers')

    # Power system model
    ps = dps.PowerSystemModel(model=model)
    ps.init_dyn_sim()
    print(max(abs(ps.state_derivatives(0, ps.x_0, ps.v_0))))

    t_end = 30
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

        if 1 <= t < 10:
            ps.trafos['DynTrafo'].set_input('ratio_from', 0.99, 0)
        elif 10 <= t < 20:
            ps.trafos['DynTrafo'].set_input('ratio_from', 0.98, 0)
        elif 20 <= t < 30:
            ps.trafos['DynTrafo'].set_input('ratio_from', 0.97, 0)

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
        res['trafo_current_from'].append(ps.trafos['DynTrafo'].i_from(x, v).copy())
        res['trafo_current_to'].append(ps.trafos['DynTrafo'].i_to(x, v).copy())

    print('Simulation completed in {:.2f} seconds.'.format(time.time() - t_0))

    plt.figure()
    fig, ax = plt.subplots(2, sharex=True)
    ax[0].plot(res['t'], np.abs(res['v']))
    ax[0].set_xlabel('Time [s]')
    ax[0].set_ylabel('Bus voltage')

    ax[1].plot(res['t'], np.abs(res['trafo_current_from']), color='C0')
    ax[1].plot(res['t'], np.abs(res['trafo_current_to']), color='C1', linestyle=':')
    ax[1].set_xlabel('Time [s]')
    ax[1].set_ylabel('Trafo current')
    
    plt.show()
    