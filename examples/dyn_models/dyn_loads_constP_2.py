import sys
from collections import defaultdict
import matplotlib.pyplot as plt
import time
import tops.dynamic as dps
import tops.solvers as dps_sol
import importlib
importlib.reload(dps)
import numpy as np

from tops.dyn_models.utils import DAEModel


class ConstPowerLoad(DAEModel):
    def __init__(self, data, sys_par, **kwargs):
        super().__init__(data, sys_par, **kwargs)
        self.data = data
        self.par = data
        self.n_units = len(data)

        self.bus_idx = np.array(np.zeros(self.n_units), dtype=[(key, int) for key in self.bus_ref_spec().keys()])
        self.bus_idx_red = np.array(np.zeros(self.n_units), dtype=[(key, int) for key in self.bus_ref_spec().keys()])
        self.sys_par = sys_par  # {'s_n': 0, 'f_n': 50, 'bus_v_n': None}

    def bus_ref_spec(self):
        return {'terminal': self.par['bus']}

    def reduced_system(self):
        return self.par['bus']

    def load_flow_pq(self):
        return self.bus_idx['terminal'], self.par['P'], self.par['Q']

    def apparent_power_injections(self, x, v):
        s_inj = -(self.par['P'] + 1j*self.par['Q'])/self.sys_par['s_n']
        return self.bus_idx_red['terminal'], s_inj
    
    def i(self, x, v):
        return v[self.bus_idx_red['terminal']]*self.y_load
    
    def s(self, x, v):
        return v[self.bus_idx_red['terminal']]*np.conj(self.i(x, v))

    def p(self, x, v):
        # p.u. system base
        return self.s(x, v).real

    def q(self, x, v):
        # p.u. system base
        return self.s(x, v).imag
    
    def P(self, x, v):
        # MW
        return self.s(x, v).real*self.sys_par['s_n']

    def Q(self, x, v):
        # MVA
        return self.s(x, v).imag*self.sys_par['s_n']



if __name__ == '__main__':

    # Load model
    import tops.ps_models.k2a as model_data
    importlib.reload(model_data)
    model = model_data.load()
    model['loads'] = {'ConstPowerLoad': model['loads']}

    user_mdl_lib = type('', (), {'loads': type('', (), {'ConstPowerLoad': ConstPowerLoad})})
    hasattr(getattr(user_mdl_lib, 'loads'), 'ConstPowerLoad')

    # Power system model
    ps = dps.PowerSystemModel(model=model, user_mdl_lib=user_mdl_lib)
    ps.init_dyn_sim()
    print(max(abs(ps.state_derivatives(0, ps.x_0, ps.v_0))))

    ps.solve_algebraic(0, ps.x0)

    t_end = 3
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
            ps.loads['ConstPowerLoad'].par['P'][0] = 1000
        if 10 <= t:
            ps.loads['ConstPowerLoad'].par['P'][0] = 967

        # Simulate next step
        result = sol.step()
        x = sol.y
        v = sol.v
        t = sol.t

        dx = ps.ode_fun(0, ps.x_0)

        # Store result
        res['t'].append(t)
        res['gen_angle'].append(ps.gen['GEN'].angle(x, v).copy())
        res['v'].append(v.copy())
        # res['load_P'].append(ps.loads['ConstPowerLoad'].P(x, v).copy())
        # res['load_Q'].append(ps.loads['ConstPowerLoad'].Q(x, v).copy())
        res['iterations'].append(ps.it_prev)

    print('Simulation completed in {:.2f} seconds.'.format(time.time() - t_0))

    fig = plt.figure()
    plt.plot(res['t'], np.abs(res['gen_angle']))
    plt.xlabel('Time [s]')
    plt.ylabel('Gen angle')
    
    fig = plt.figure()
    plt.plot(res['t'], np.abs(res['v']))
    plt.xlabel('Time [s]')
    plt.ylabel('Bus voltage magnitude')

    fig = plt.figure()
    plt.plot(res['t'], np.angle(np.array(res['v'])), color='g')
    plt.xlabel('Time [s]')
    plt.ylabel('Bus voltage angle')

    fig = plt.figure()
    plt.plot(res['t'], res['iterations'])

    
    # fig = plt.figure()
    # plt.plot(res['t'], np.abs(res['load_P']))
    # plt.xlabel('Time [s]')
    # plt.ylabel('MW')

    # fig = plt.figure()
    # plt.plot(res['t'], np.abs(res['load_Q']))
    # plt.xlabel('Time [s]')
    # plt.ylabel('MVA')
    
    plt.show()
    