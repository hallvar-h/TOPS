import sys
from collections import defaultdict
import matplotlib.pyplot as plt
import time
import dynpssimpy.dynamic as dps
import dynpssimpy.solvers as dps_sol
import importlib
importlib.reload(dps)
import importlib

if __name__ == '__main__':

    # Load model
    import dynpssimpy.ps_models.k2a as model_data
    importlib.reload(model_data)
    model = model_data.load()

    model['vsc'] = {'VSC': [
        ['name',    'T_pll',    'T_i',  'bus',  'P_K_p',    'P_K_i',    'Q_K_p',    'Q_K_i',    'P_setp',   'Q_setp',   ],
        ['VSC1',    0.1,        1,      'B8',   0.1,        0.1,        0.1,        0.1,        100,          100],
    ]}

    # Power system model
    ps = dps.PowerSystemModel(model=model)
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

    # Run simulation
    while t < t_end:
        sys.stdout.write("\r%d%%" % (t/(t_end)*100))

        if t > 1:
            ps.vsc['VSC'].set_input('P_setp', 500)
        if t > 5:
            ps.vsc['VSC'].set_input('Q_setp', 200)

        # Simulate next step
        result = sol.step()
        x = sol.y
        t = sol.t
        v = sol.v

        dx = ps.ode_fun(0, ps.x_0)

        # Store result
        res['t'].append(sol.t)
        res['gen_speed'].append(ps.gen['GEN'].speed(x, v).copy())
        res['VSC_P'].append(ps.vsc['VSC'].P(x, v).copy())
        res['VSC_P_setp'].append(ps.vsc['VSC'].P_setp(x, v).copy())

        res['VSC_Q'].append(ps.vsc['VSC'].Q(x, v).copy())
        res['VSC_Q_setp'].append(ps.vsc['VSC'].Q_setp(x, v).copy())

    print('Simulation completed in {:.2f} seconds.'.format(time.time() - t_0))

    # plt.figure()
    # plt.plot(res['t'], res['gen_speed'])
    # plt.show()

    fig, ax = plt.subplots(2)
    ax[0].plot(res['t'], res['VSC_P'])
    ax[0].plot(res['t'], res['VSC_P_setp'])

    ax[1].plot(res['t'], res['VSC_Q'])
    ax[1].plot(res['t'], res['VSC_Q_setp'])
    plt.show()