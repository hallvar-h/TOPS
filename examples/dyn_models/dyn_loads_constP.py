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
    # model['loads'] = {'DynamicLoad': model['loads']}

    # Power system model
    ps = dps.PowerSystemModel(model=model)
    ps.init_dyn_sim()
    print(max(abs(ps.state_derivatives(0, ps.x_0, ps.v_0))))

    ps.solve_algebraic(0, ps.x0)
    

    ##########################
    from scipy.sparse import linalg as sp_linalg
    from scipy.sparse import diags as sp_diags
    self = ps
    v = self.v0.copy()
    x = self.x0
    self.s_inj = np.zeros_like(v, dtype=complex)
    self.s_inj += 1*(np.random.randn(len(v)) + 1j*np.random.randn(len(v)))

    i_inj = np.zeros(self.n_bus_red, dtype=complex)
    for mdl in self.mdl_instructions['current_injections']:
        bus_idx_red, i_inj_mdl = mdl.current_injections(x, None)
        np.add.at(i_inj, bus_idx_red, i_inj_mdl)

    s_inj = np.zeros(self.n_bus_red, dtype=complex)
    for mdl in self.mdl_instructions['apparent_power_injections']:
        bus_idx_red, s_inj_mdl = mdl.apparent_power_injections(x, None)
        np.add.at(s_inj, bus_idx_red, s_inj_mdl)


    y_bus = self.y_bus_red

    v_sol = sp_linalg.spsolve(self.y_bus_red, i_inj)

    tol = 1e-10
    max_it = 100
    error = 10 * tol
    it = 0
    
    i_mag = abs(i_inj)
    i_ang = np.angle(i_inj)

    v_mag = abs(v_sol)
    v_ang = np.angle(v_sol)

    pq_vec = np.concatenate([s_inj.real, s_inj.imag])
    i_vec = np.concatenate([i_mag*v_mag*np.cos(i_ang - v_ang), i_mag*v_mag*np.sin(i_ang - v_ang)])

    v0 = v_sol
    n_bus = len(v0)
    x0 = np.zeros(2*n_bus)
    v_abs_idx = slice(n_bus)
    v_ang_idx = slice(n_bus, 2*n_bus)
    x0[v_abs_idx] = 1
    def f(x):
        v = x[v_abs_idx]*np.exp(1j*x[v_ang_idx])
        f_complex = y_bus.dot(v)*v - i_inj*v - s_inj
        return np.concatenate([f_complex.real, f_complex.imag])

    # x0 = np.concatenate([abs(v0), np.angle(v0)])
    f(x0)
    from tops.utility_functions import jacobian_num


    
    x = x0.copy()
    error = np.linalg.norm(f(x))
    # t_spsolve_cum0 = 0
    from scipy.sparse import csr_matrix
    while error > tol and it < max_it:
        A = csr_matrix(jacobian_num(f, x))
        b = - f(x)
        dx = sp_linalg.spsolve(A, b)
        x += dx

        error = np.linalg.norm(f(x))
        it += 1
        print(it)

    if error > tol:
        print('Warning: Solution of algebraic equations did not converge.')

    # return v_red
    ##########################

    ps.solve_algebraic

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
            ps.loads['DynamicLoad'].set_input('g_setp', 1.3, 0)
        if 10 <= t:
            ps.loads['DynamicLoad'].set_input('b_setp', -0.2, 1)

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
        res['load_P'].append(ps.loads['DynamicLoad'].P(x, v).copy())
        res['load_Q'].append(ps.loads['DynamicLoad'].Q(x, v).copy())

    print('Simulation completed in {:.2f} seconds.'.format(time.time() - t_0))

    fig = plt.figure()
    plt.plot(res['t'], np.abs(res['v']))
    plt.xlabel('Time [s]')
    plt.ylabel('Bus voltage')

    
    fig = plt.figure()
    plt.plot(res['t'], np.abs(res['load_P']))
    plt.xlabel('Time [s]')
    plt.ylabel('MW')

    fig = plt.figure()
    plt.plot(res['t'], np.abs(res['load_Q']))
    plt.xlabel('Time [s]')
    plt.ylabel('MVA')
    
    plt.show()
    