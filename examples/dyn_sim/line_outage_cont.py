import sys
from collections import defaultdict
import matplotlib.pyplot as plt
import time
import tops.dynamic as dps
import tops.solvers as dps_sol
import importlib
import numpy as np
importlib.reload(dps)

if __name__ == '__main__':

    # Load model
    import tops.ps_models.ieee39 as model_data
    importlib.reload(model_data)
    model = model_data.load()

    # Power system model

    res_all = dict()
    ps = dps.PowerSystemModel(model=model)

    for line_name in ps.lines["Line"].par["name"]:
        print(line_name)
        ps.init_dyn_sim()

        print(max(abs(ps.state_derivatives(0, ps.x_0, ps.v_0))))

        t_end = 5
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

            # Line outage
            if t > 1 and event_flag:
                event_flag = False
                ps.lines['Line'].event(ps, line_name, 'disconnect')

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

        print('Simulation completed in {:.2f} seconds.'.format(time.time() - t_0))
        res_all[line_name] = res

    plt.figure()
    for line_name, res in res_all.items():
        line_name

        for i, gen_spd in enumerate(np.array(res['gen_speed']).T):
            plt.plot(res['t'], gen_spd, color=f"C{i}")
        # res
        # res["gen_speed"][]
        # plt.plot(res['t'], res['gen_speed'])
        plt.xlabel('Time [s]')
        plt.ylabel('Gen. speed')
    plt.show()