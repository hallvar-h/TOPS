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

    gen_speed_state_idx = ps.gen['GEN'].state_idx_global['speed']

    t_end = 30
    # Solver
    sol = EulerDAE_SDE(ps.state_derivatives, ps.solve_algebraic, 0, ps.x_0, t_end, max_step=5e-3, dim_w = 4)

    def b_func(t, x, v):
        mat = np.zeros((len(sol.x), sol.dim_w))
        for i, idx in enumerate(gen_speed_state_idx):
            mat[idx, i] = 1e-3
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
        res['gen_speed'].append(ps.gen['GEN'].speed(x, v).copy())
        res['dw'].append(sol.dw)

    
    fig, ax = plt.subplots(3, sharex=True)
    ax[0].plot(res['time'], res['gen_speed'])
    ax[0].set_ylabel('Gen speed')
    ax[0].set_xlabel('Time [s]')
    ax[1].plot(res['time'], sol.dt*np.cumsum(np.array(res['dw']), axis=0))
    ax[1].set_ylabel('W')
    ax[2].plot(res['time'], res['p_load'])
    ax[2].set_ylabel('P Load [p.u.]')
    plt.show()
    
    gen_angle_np = np.array(res['gen_angle']).T
    gen_speed_np = np.array(res['gen_speed']).T
    
    angle_speed_mat = np.vstack([gen_angle_np, gen_speed_np])

    cov_mat = np.cov(angle_speed_mat)
    cov_mat_angle = np.cov(gen_angle_np[[0, 2], :])
    print(np.linalg.det(cov_mat_angle))

    np.linalg.inv(cov_mat_angle)