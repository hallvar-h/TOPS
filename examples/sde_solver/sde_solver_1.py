import sys
from collections import defaultdict
import matplotlib.pyplot as plt
import time
import dynpssimpy.dynamic as dps
from dynpssimpy.solvers_sde import EulerDAE_SDE
import numpy as np

if __name__ == '__main__':

    # Load model
    import dynpssimpy.ps_models.k2a as model_data
    model = model_data.load()
    model['loads'] = {'DynamicLoadFiltered':  [# model['loads']}
        model['loads'][0] + ['T_g', 'T_b'],
        *[row + [0.1, 0.1] for row in model['loads'][1:]]
    ]}

    # Power system model
    ps = dps.PowerSystemModel(model=model)
    ps.init_dyn_sim()

    load_state_idx_g = ps.loads['DynamicLoadFiltered'].lpf_g.state_idx_global['x']
    load_state_idx_b = ps.loads['DynamicLoadFiltered'].lpf_b.state_idx_global['x']

    t_end = 10
    # Solver
    sol = EulerDAE_SDE(ps.state_derivatives, ps.solve_algebraic, 0, ps.x_0, t_end, max_step=5e-3, dim_w = 2)

    def b_func(t, x, v):
        mat = np.zeros((len(sol.x), sol.dim_w))
        mat[load_state_idx_g[0], 0] = 0.1
        mat[load_state_idx_b[0], 0] = 0.1
        mat[load_state_idx_g[1], 1] = 0.1
        mat[load_state_idx_b[1], 1] = 0.1
        return mat

    # plt.imshow(b_func(*[None]*3))
    # plt.show()

    sol.b = b_func
        
    # Initialize simulation
    t = 0
    res = defaultdict(list)
    t_0 = time.time()

    # Run simulation
    while t < t_end:
        sys.stdout.write("\r%d%%" % (t/(t_end)*100))

        # Simulate next step
        result = sol.step()
        x = sol.y
        v = sol.v
        dw = sol.dw  # random variable
        t = sol.t

        res['time'].append(t)
        res['p_load'].append(ps.loads['DynamicLoadFiltered'].p(x, v))
        res['g_load'].append(ps.loads['DynamicLoadFiltered'].g_load(x, v).copy())
        res['gen_angle'].append(ps.gen['GEN'].angle(x, v).copy())
        res['dw'].append(sol.dw)

    
    fig, ax = plt.subplots(3, sharex=True)
    ax[0].plot(res['time'], res['gen_angle'])
    ax[0].set_ylabel('Gen angle')
    ax[0].set_xlabel('Time [s]')
    ax[1].plot(res['time'], sol.dt*np.cumsum(np.array(res['dw']), axis=0))
    ax[1].set_ylabel('W')
    ax[2].plot(res['time'], res['p_load'])
    ax[2].set_ylabel('P Load [p.u.]')
    plt.show()