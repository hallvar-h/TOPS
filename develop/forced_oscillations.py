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
    import dynpssimpy.ps_models.sm_load as model_data
    importlib.reload(model_data)
    model = model_data.load()

    K_p = 5
    K_i = 2

    r = 1/K_p       # Used to be 0.1
    T_r = K_p/K_i   # Used to be 10
    
    model['gov'] = {
        # 'TGOV1': [model['gov']['TGOV1'][0], *model['gov']['TGOV1'][2:]],
        'HYGOV': [
            ['name',    'gen',  'R',    'r',    'T_f',  'T_r',  'T_g',  'A_t',  'T_w',  'q_nl',     'D_turb' ,      'G_min',    'V_elm',    'G_max',    'P_N',  'backlash'],
            ['HYGOV1',  'G1',   0.04,   r,      0.2,    T_r,     0.2,      1,      1.28,      0.01,      0.01,          0,          0.15,       1,          0,      0.5/100],
    ]}
    # model['pss'] = {
    #     'STAB1': [
    #         ['name',    'gen',  'K',    'T',    'T_1',  'T_2',  'T_3',  'T_4',  'H_lim'],
    #         ['PSS1',    'G1',   50,     10.0,   0.5,    0.5,    0.05,   0.05,   0.03],
    #     ]
    # }
    # model['avr'] = {
    #     'SEXS': [
    #         ['name', 'gen', 'K', 'T_a', 'T_b', 'T_e', 'E_min', 'E_max'],
    #         ['AVR1', 'G1', 200, 9.0, 10.0, 0.01, -10, 10],
    #     ]
    # }

    # Power system model
    ps = dps.PowerSystemModel(model=model)
    ps.gen['GEN'].par
    ps.lines['Line'].par['length'] *= 1
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

    initial_bias = ps.gov['HYGOV'].int_par['bias'].copy()

    # Run simulation
    while t < t_end:
        sys.stdout.write("\r%d%%" % (t/(t_end)*100))

        # Short circuit
        if t >= 1 and t <= 1.05:
            ps.y_bus_red_mod[(sc_bus_idx,) * 2] = 1e-1
        # else:
            # ps.y_bus_red_mod[(sc_bus_idx,) * 2] = 0
        # ps.gov['HYGOV'].int_par['bias'] = initial_bias + np.sin(t)*0.01

        # Simulate next step
        result = sol.step()
        x = sol.y
        v = sol.v
        t = sol.t

        dx = ps.ode_fun(0, ps.x_0)

        # Store result
        res['t'].append(t)
        res['gen_speed'].append(ps.gen['GEN'].speed(x, v).copy())

        res['backlash_in'].append(ps.gov['HYGOV'].backlash.input(x, v).copy())
        res['backlash_out'].append(ps.gov['HYGOV'].backlash.output(x, v).copy())

    print('Simulation completed in {:.2f} seconds.'.format(time.time() - t_0))


    fig, ax = plt.subplots(3)
    ax[0].plot(res['t'], res['gen_speed'])
    ax[0].set_xlabel('Time [s]')
    ax[0].set_ylabel('Gen. speed')
    ax[1].plot(res['t'], res['backlash_in'])
    ax[1].plot(res['t'], res['backlash_out'])
    # ax[0].xlabel('Time [s]')
    # ax[0].ylabel('Deadband in/out')
    ax[2].plot(res['backlash_in'], res['backlash_out'])
    
    
    plt.show()