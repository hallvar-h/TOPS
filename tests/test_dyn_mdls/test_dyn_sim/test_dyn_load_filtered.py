import sys
from collections import defaultdict
import matplotlib.pyplot as plt
import time
import tops.dynamic as dps
import tops.solvers as dps_sol
import numpy as np

if __name__ == '__main__':

    # Load model
    import tops.ps_models.k2a as model_data
    model = model_data.load()
    model['loads'] = {'DynamicLoadFiltered':  [# model['loads']}
        model['loads'][0] + ['T_g', 'T_b'],
        *[row + [0.2, 0.2] for row in model['loads'][1:]]
    ]}

    # Power system model
    ps = dps.PowerSystemModel(model=model)
    ps.init_dyn_sim()

    t_end = 2
    # Solver
    sol = dps_sol.ModifiedEulerDAE(ps.state_derivatives, ps.solve_algebraic, 0, ps.x_0, t_end, max_step=5e-3)

    # Initialize simulation
    t = 0
    res = defaultdict(list)
    t_0 = time.time()

    # Run simulation
    while t < t_end:
        sys.stdout.write("\r%d%%" % (t/(t_end)*100))

        # Short circuit
        if t >= 1:
            ps.loads['DynamicLoadFiltered'].set_input('g_setp', [1, 2])
            ps.loads['DynamicLoadFiltered'].set_input('b_setp', [-0.1, -0.1])

        # Simulate next step
        result = sol.step()
        x = sol.y
        v = sol.v
        t = sol.t

        res['time'].append(t)
        res['load_P'].append(ps.loads['DynamicLoadFiltered'].p(x, v)[0])

    assert np.linalg.norm(ps.loads['DynamicLoadFiltered'].y_load(x, v).real - [1, 2]) < 1e-2
    assert np.linalg.norm(ps.loads['DynamicLoadFiltered'].y_load(x, v).imag - [-0.1, -0.1]) < 1e-2