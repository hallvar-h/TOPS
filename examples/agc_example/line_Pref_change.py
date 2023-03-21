import sys
from collections import defaultdict
import matplotlib.pyplot as plt
import time
import dynpssimpy.dynamic as dps
import dynpssimpy.solvers as dps_sol
import importlib
importlib.reload(dps)

if __name__ == '__main__':

    # Load model
    try:
        import k2a_agc as model_data
        import user_lib
    except:
        import examples.agc_example.k2a_agc as model_data
        import examples.agc_example.user_lib as user_lib

    model = model_data.load()
    # model['agc'] = {}

    # Power system model
    ps = dps.PowerSystemModel(model=model, user_mdl_lib=user_lib)
    ps.init_dyn_sim()
    print(max(abs(ps.state_derivatives(0, ps.x_0, ps.v_0))))
    t_end = 10
    x_0 = ps.x_0.copy()

if False:
    # Solver
    sol = dps_sol.ModifiedEulerDAE(ps.state_derivatives, ps.solve_algebraic, 0, x_0, t_end, max_step=5e-3)

    # Initialize simulation
    t = 0
    res = defaultdict(list)
    t_0 = time.time()

    load_bus_idx = ps.loads['Load'].bus_idx_red['terminal'][0]
    y_load_0 = ps.loads['Load'].y_load.real[0]

    # Run simulation
    while t < t_end:
        sys.stdout.write("\r%d%%" % (t/(t_end)*100))

        # Short circuit
        if t > 1:
            ps.acg
            # ps.y_bus_red_mod[(load_bus_idx,) * 2] = y_load_0*0.5
            

        # Simulate next step
        result = sol.step()
        x = sol.y
        v = sol.v
        t = sol.t

        dx = ps.ode_fun(0, ps.x_0)

        # Store result
        res['t'].append(t)
        res['gen_speed'].append(ps.gen['GEN'].speed(x, v).copy())
        res['gen_angle'].append(ps.gen['GEN'].angle(x, v).copy())
        res['gov_output'].append(ps.gov['HYGOV'].output(x, v).copy())

    print('Simulation completed in {:.2f} seconds.'.format(time.time() - t_0))

    plt.figure()
    plt.plot(res['t'], res['gen_angle'])
    plt.xlabel('Time [s]')
    plt.ylabel('Gen. angle')
    plt.show()

    plt.figure()
    plt.plot(res['t'], res['gov_output'])
    plt.xlabel('Time [s]')
    plt.ylabel('Gov. output')
    plt.show()