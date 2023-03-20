import sys
from collections import defaultdict
import matplotlib.pyplot as plt
import time
import dynpssimpy.dynamic as dps
import dynpssimpy.solvers as dps_sol
import dynpssimpy.modal_analysis as dps_ma
import importlib
importlib.reload(dps)

if __name__ == '__main__':

    # Load model
    # import dynpssimpy.ps_models.ieee39 as model_data
    try:
        import n44_new as model_data
    except ModuleNotFoundError:
        import develop.n44_model.n44_new as model_data

    # importlib.reload(model_data)
    model = model_data.load()
    # model['avr'] = {}
    # model['gov'] = {}
    # model['pss'] = {}
    # model['gov'].pop('HYGOV')

    # Power system model
    ps = dps.PowerSystemModel(model=model)
    # ps.gen['GEN'].par['H'] *= 2
    # ps.loads['Load'].par['P'] *= 0.8
    # ps.gen['GEN'].par['P'] *= 0.8
    ps.init_dyn_sim()
    # overload_idx = [True]
    # while np.any(overload_idx):
    #     overload_idx = abs(ps.load_flow_soln[ps.gen['GEN']])/ps.gen['GEN'].par['S_n'] > 1
    #     ps.gen['GEN'].par['S_n'][overload_idx] *= 1.1

    # Generator overload:
    print(abs(ps.load_flow_soln[ps.gen['GEN']])/ps.gen['GEN'].par['S_n'])

    ps_lin = dps_ma.PowerSystemModelLinearization(ps)
    ps_lin.linearize()
    ps_lin.eigenvalue_decomposition()
    eigs = ps_lin.eigs
    import numpy as np
    idx_unstable = np.argwhere(ps_lin.damping < 0).flatten()
    
    plt.scatter(eigs.real, eigs.imag)
    plt.scatter(eigs[idx_unstable].real, eigs[idx_unstable].imag, color='r', marker='X')
    # plt.xlim(-2, 2)
    plt.grid(True)
    plt.show()
if False:
    # rev_unstable = ps_lin.rev[idx_unstable, :]
    # plt.plot(rev_unstable.T)
    # max_obs_idx = np.argmax(abs(rev_unstable[0]))
    # ps.state_desc[max_obs_idx]
    # plt.show()
    # import numpy as np
    # state_idx = np.where(abs(ps.state_derivatives(0, ps.x_0, ps.v_0)) > 0.01)
    # ps.state_desc[state_idx]
    # plt.plot(ps.state_derivatives(0, ps.x_0, ps.v_0))
    # plt.show()

    t_end = 5
    x_0 = ps.x_0.copy()

    # Solver
    sol = dps_sol.ModifiedEulerDAE(ps.state_derivatives, ps.solve_algebraic, 0, x_0, t_end, max_step=2.5e-3)

    # Initialize simulation
    t = 0
    res = defaultdict(list)
    t_0 = time.time()

    sc_bus_idx = ps.gen['GEN'].bus_idx_red['terminal'][0]
    load_bus_idx = ps.loads['Load'].bus_idx_red['terminal'][0]
    y_load_0 = ps.loads['Load'].y_load.real[0]

    # Run simulation
    while t < t_end:
        sys.stdout.write("\r%d%%" % (t/(t_end)*100))

        # if t > 1:
            # ps.y_bus_red_mod[(load_bus_idx,) * 2] = y_load_0*0.1
        # Short circuit
        if t >= 1 and t <= 1.05:
            ps.y_bus_red_mod[(sc_bus_idx,) * 2] = 1e6
        else:
            ps.y_bus_red_mod[(sc_bus_idx,) * 2] = 0

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