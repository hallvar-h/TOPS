import sys
from collections import defaultdict
import matplotlib.pyplot as plt
import time
import dynpssimpy.dynamic as dps
import dynpssimpy.solvers as dps_sol
import importlib
importlib.reload(dps)

import time
import numpy as np



def state_derivatives_mod(self, t, x, v_red):

    for mdl in self.dyn_mdls:
        mdl.reset_outputs()
        mdl._store_output = True

    dx = np.zeros(self.n_states)
    for mdl in self.mdl_instructions['state_derivatives'][0:1]:
        mdl.state_derivatives(dx, x, v_red)

    for mdl in self.dyn_mdls:
        mdl._store_output = False

    return dx

if __name__ == '__main__':

    import sys
    sys.path.append(r'C:\Users\hallvarh\Coding\dynpssimpy\develop_untracked\neweps_n44_model')

    # Load model
    import dynpssimpy.ps_models.ieee39 as model_data
    # import n44_new as model_data
    importlib.reload(model_data)
    model = model_data.load()

    # Power system model
    ps = dps.PowerSystemModel(model=model)
    ps.init_dyn_sim()
    print(max(abs(ps.state_derivatives(0, ps.x_0, ps.v_0))))


    state_derivatives_mod(ps, 0, ps.x_0, ps.v_0)
    x = ps.x_0
    v = ps.v_0

    self = ps
    
    n_fev = 1000
    t_0 = time.time()
    for _ in range(n_fev):
        _ = ps.state_derivatives(0, ps.x_0, ps.v_0)
    print(time.time() - t_0)


    t_0 = time.time()
    for _ in range(n_fev):
        _ = state_derivatives_mod(ps, 0, ps.x_0, ps.v_0)
    print(time.time() - t_0)