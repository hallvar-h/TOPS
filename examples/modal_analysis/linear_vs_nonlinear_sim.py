import sys
import tops.dynamic as dps
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import tops.solvers as dps_sol
import tops.modal_analysis as dps_lin
from scipy.signal import dlsim, StateSpace


if __name__ == '__main__':

    # Load model
    import tops.ps_models.k2a as model_data
    model = model_data.load()

    # Power system model
    ps = dps.PowerSystemModel(model=model)
    ps.init_dyn_sim()
    np.max(abs(ps.ode_fun(0.0, ps.x0)))
    t_end = 2
    dt = 1e-3

    x0 = ps.x0.copy()
    sol = dps_sol.ModifiedEulerDAE(ps.state_derivatives, ps.solve_algebraic, 0, x0, t_end, max_step=dt)

    # Linear model
    ps_lin = dps_lin.PowerSystemModelLinearization(ps)
    ps_lin.linearize()

    def input_fun_dist(ps, eps):
        ps.y_bus_red_mod[0, 0] += eps
    b_dist = ps_lin.linearize_inputs_v3([input_fun_dist])

    b = np.hstack([b_dist])

    output_fun = lambda t, x, v: v[1]
    c = ps_lin.linearize_outputs_v4([output_fun])
    d = np.zeros((1, b.shape[1]))
    y0 = output_fun(0, ps.x0, ps.v0)
    delta_x_lin = np.zeros_like(ps.x0)
    x0_lin = ps.x0.copy()

    x_init_test = ps.x0.copy()
    delta_x_init_test = np.zeros_like(x_init_test)
    delta_x_init_test[0] += 1e-6
    x_init_test += delta_x_init_test

    sys_ss = StateSpace(ps_lin.a, b, c, d)
    sys_ss_d = sys_ss.to_discrete(dt=dt)
    
    assert np.linalg.norm(ps.ode_fun(0, x_init_test) - ps_lin.a.dot(delta_x_init_test)) < 1e-9

    # Initialize simulation
    t = 0
    t_lin = 0
    result_dict = defaultdict(list)
    result_dict_lin = defaultdict(list)
    t_0 = time.time()

    dist_magn = 1e-2

    # Run simulation
    while t < t_end:
        sys.stdout.write("\r%d%%" % (t/(t_end)*100))

        # Short circuit
        if 1 < t < 1.1:
            ps.y_bus_red_mod[0, 0] = dist_magn
            u = dist_magn
        else:
            ps.y_bus_red_mod *= 0
            u = 0

        # Simulate next step
        result = sol.step()
        x = sol.y
        v = sol.v
        t = sol.t

        # Simulate linear step
        solution = dlsim(sys_ss_d, u=np.vstack([u, u]), x0=delta_x_lin)
        delta_x_lin = solution[2][-1, :]
        t_lin += dt

        x_lin = x0_lin + delta_x_lin

        # Store result
        result_dict['Global', 't'].append(sol.t)
        [result_dict[tuple(desc)].append(state) for desc, state in zip(ps.state_desc, x)]
        result_dict['Non-linear output'].append(output_fun(t, x, v))

        result_dict_lin['Global', 't'].append(t_lin)
        [result_dict_lin[tuple(desc)].append(state) for desc, state in zip(ps.state_desc, x_lin)]
        result_dict_lin['Linear output'].append(y0 + c.dot(delta_x_lin))

    print('Simulation completed in {:.2f} seconds.'.format(time.time() - t_0))

    # Convert dict to pandas dataframe
    index = pd.MultiIndex.from_tuples(result_dict)
    result = pd.DataFrame(result_dict, columns=index)
    result_lin = pd.DataFrame(result_dict_lin, columns=index)

    # Plot angle and speed
    fig, ax = plt.subplots(2)
    for result_, style, color in zip([result, result_lin], ['-', '--'], ['C0', 'C1']):
        ax[0].plot(result_[('Global', 't')], result_.xs(key='speed', axis='columns', level=1), linestyle=style, color=color)
        ax[1].plot(result_[('Global', 't')], result_.xs(key='angle', axis='columns', level=1), linestyle=style, color=color)
    plt.show()

    fig, ax = plt.subplots(1)
    ax.plot(result_dict[('Global', 't')], result_dict['Non-linear output'], 'C0')
    ax.plot(result_dict_lin[('Global', 't')], result_dict_lin['Linear output'], 'C1')
    plt.show()