import sys
from collections import defaultdict
import matplotlib.pyplot as plt
import time
import tops.dynamic as dps
import tops.solvers as dps_sol
import importlib
importlib.reload(dps)

if __name__ == '__main__':

    # Load model
    import tops.ps_models.k2a as model_data
    importlib.reload(model_data)
    model = model_data.load()

    # Power system model
    ps = dps.PowerSystemModel(model=model)
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

    event_flag = True

    # Run simulation
    while t < t_end:
        sys.stdout.write("\r%d%%" % (t/(t_end)*100))

        # Short circuit
        if t > 1 and event_flag:
            event_flag = False
            ps.lines['Line'].event(ps, ps.lines['Line'].par['name'][0], 'disconnect')

        # Simulate next step
        result = sol.step()
        x = sol.y
        v = sol.v
        t = sol.t

        dx = ps.ode_fun(0, ps.x_0)

        # Store result
        res['t'].append(t)
        res['gen_speed'].append(ps.gen['GEN'].speed(x, v).copy())

    print('Simulation completed in {:.2f} seconds.'.format(time.time() - t_0))

    plt.figure()
    plt.plot(res['t'], res['gen_speed'])
    plt.xlabel('Time [s]')
    plt.ylabel('Gen. speed')
    plt.show()