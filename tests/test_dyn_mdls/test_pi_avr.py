import sys
from collections import defaultdict
import matplotlib.pyplot as plt
import time
import dynpssimpy.dynamic as dps
import dynpssimpy.solvers as dps_sol
import importlib
import numpy as np
importlib.reload(dps)

if __name__ == '__main__':

    # Load model
    import dynpssimpy.ps_models.k2a as model_data
    importlib.reload(model_data)
    model = model_data.load()
    
    model['avr'] = {'SEXS_PI': [
        ['name',            'gen',      'K',    'T_a',  'T_b',  'T_e',  'E_min',    'E_max',    'K_p',  'K_i',  'T_ext'],
        *[[f'AVR{i+1}',     gen[0],     100,    1.0,    1.0,   0.1,    -3,         3,          2,      1,      0.01] for i, gen in enumerate(model['generators']['GEN'][1:])]
    ]}

    # Power system model
    ps = dps.PowerSystemModel(model=model)
    ps.init_dyn_sim()
    print(max(abs(ps.state_derivatives(0, ps.x_0, ps.v_0))))

    t_end = 3
    x_0 = ps.x_0.copy()

    # Solver
    sol = dps_sol.ModifiedEulerDAE(ps.state_derivatives, ps.solve_algebraic, 0, x_0, t_end, max_step=5e-3)

    # Initialize simulation
    t = 0
    res = defaultdict(list)
    t_0 = time.time()

    sc_bus_idx = ps.gen['GEN'].bus_idx_red['terminal'][0]
    v_setp = 1.04  # 1.01*np.ones(ps.gen['GEN'].n_units)

    # Run simulation
    while t < t_end:
        sys.stdout.write("\r%d%%" % (t/(t_end)*100))

        # Apply new voltage set-points
        if t >= 1:
            ps.gen['GEN'].par['V'][0] = v_setp

        # Simulate next step
        result = sol.step()
        x = sol.y
        t = sol.t

        dx = ps.ode_fun(0, ps.x_0)

        # Store result
        res['t'].append(sol.t)
        res['v'].append(sol.v.copy())
        res['v_setp'].append()
        res['v_setp_lag']

    print('Simulation completed in {:.2f} seconds.'.format(time.time() - t_0))

    # plt.figure()
    # state = 'speed'
    # for i, gen in enumerate(ps.gen['GEN'].par['name']):
        # plt.plot(result_dict[('Global', 't')], result_dict[(gen, f'{state}')], color=f'C{i}', alpha=0.5)
    # plt.show()
    fig, ax = plt.subplots(1)
    ax.plot(res['t'], np.abs(res['v'])[:, 0])
    ax.axhline(v_setp, linestyle='--')
    plt.show()