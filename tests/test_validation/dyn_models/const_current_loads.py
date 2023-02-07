import sys
from collections import defaultdict
import matplotlib.pyplot as plt
import time
import dynpssimpy.dynamic as dps
import dynpssimpy.solvers as dps_sol
import importlib
import numpy as np


if __name__ == '__main__':

    # Load model
    import dynpssimpy.ps_models.k2a as model_data
    importlib.reload(model_data)
    model = model_data.load()


    ps = dps.PowerSystemModel(model=model)
    ps.init_dyn_sim()
    i_1 =ps.loads['Load'].i(ps.x0, ps.v_0)
    

    model['loads'] = {'ConstCurrentLoadPLL': [
        model['loads'][0] + ['T_pll'],
        *[l + [0.01] for l in model['loads'][1:]]
    ]}

    # Power system model
    ps = dps.PowerSystemModel(model=model)
    ps.init_dyn_sim()
    print(max(abs(ps.state_derivatives(0, ps.x_0, ps.v_0))))

    v_0_test = ps.solve_algebraic(0, ps.x0)
    print(v_0_test - ps.v_0)
    # self = ps.loads['ConstCurrentLoad']
    # self.i_n = self.sys_par['s_n'] / (np.sqrt(3) * self.sys_par['bus_v_n'])

    # i_2 = self.I_inj(ps.x0, ps.v_0)/self.i_n[self.bus_idx_red['terminal']]
    # print(i_1)
    # print(i_2)

    # print(np.linalg.norm(i_1 - i_2))
    # print(ps.buses['V_n'])

    t_end = 10  
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

        # Short circuit
        # if t >= 1 and t <= 1.05:
        #     ps.y_bus_red_mod[(sc_bus_idx,) * 2] = 1e6
        # else:
        #     ps.y_bus_red_mod[(sc_bus_idx,) * 2] = 0
        if t >= 1:
            ps.loads['ConstCurrentLoadPLL'].set_input('Id_setp', 2.5, 0)
        if t >= 5:
            ps.loads['ConstCurrentLoadPLL'].set_input('Iq_setp', 0.4, 1)

        # Simulate next step
        result = sol.step()
        x = sol.y
        v = sol.v
        t = sol.t

        dx = ps.ode_fun(0, ps.x_0)

        for mdl in ps.dyn_mdls:
            mdl.reset_outputs()


        # Store result
        res['t'].append(t)
        res['gen_speed'].append(ps.gen['GEN'].speed(x, v).copy())
        res['V_angle'].append(ps.loads['ConstCurrentLoadPLL'].pll.filter.input(x, v).copy())
        res['PLL_angle'].append(ps.loads['ConstCurrentLoadPLL'].pll.output(x, v).copy())
        res['I_inj'].append(ps.loads['ConstCurrentLoadPLL'].I_inj(x, v).copy())
        res['Id_setp'].append(ps.loads['ConstCurrentLoadPLL'].Id_setp(x, v).copy())
        res['Iq_setp'].append(ps.loads['ConstCurrentLoadPLL'].Iq_setp(x, v).copy())
        res['Id'].append(ps.loads['ConstCurrentLoadPLL'].I_d(x, v).copy())
        res['Iq'].append(ps.loads['ConstCurrentLoadPLL'].I_q(x, v).copy())

    print('Simulation completed in {:.2f} seconds.'.format(time.time() - t_0))

    plt.figure()
    plt.plot(res['t'], res['V_angle'], color='g')
    plt.plot(res['t'], res['PLL_angle'], color='b')
    plt.xlabel('Time [s]')
    plt.ylabel('Id')
    plt.show()

    plt.figure()
    plt.plot(res['t'], res['Id_setp'], color='g')
    plt.plot(res['t'], np.array(res['Id']), color='b')
    plt.xlabel('Time [s]')
    # plt.ylabel('Id')
    plt.show()
    
    plt.figure()
    plt.plot(res['t'], res['Iq_setp'], color='g')
    plt.plot(res['t'], np.array(res['Iq']), color='b')
    plt.xlabel('Time [s]')
    # plt.ylabel('Id')
    plt.show()
    