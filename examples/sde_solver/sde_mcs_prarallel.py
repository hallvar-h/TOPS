import sys
from multiprocessing import Pool
from collections import defaultdict
import matplotlib.pyplot as plt
import time
import itertools
import tops.dynamic as dps
from tops.solvers_sde import EulerDAE_SDE
import numpy as np
import tops.ps_models.k2a as model_data
import pandas as pd

def simulate_sys(sim_i, R, H):
    model = model_data.load()

    # Change the droop of the machines. There is probably an easier way of doing this.
    R_idx = model['gov']['TGOV1'][0].index('R')
    for gov in model['gov']['TGOV1'][1:]:
        gov[R_idx] = R
    
    H_idx = model['generators']['GEN'][0].index('H')
    for gen in model['generators']['GEN'][1:]:
        gen[H_idx] = H

    # Load model
    model['loads'] = {'DynamicLoadFiltered':  [# model['loads']}
        model['loads'][0] + ['T_g', 'T_b'],
        *[row + [0.1, 0.1] for row in model['loads'][1:]]
    ]}

    # Power system model
    ps = dps.PowerSystemModel(model=model)
    ps.init_dyn_sim()

    load_state_idx_g = ps.loads['DynamicLoadFiltered'].lpf_g.state_idx_global['x']
    load_state_idx_b = ps.loads['DynamicLoadFiltered'].lpf_b.state_idx_global['x']

    t_end = 1800
    t_tops = 5e-3
    t_save = 0.02
    # Solver
    sol = EulerDAE_SDE(ps.state_derivatives, ps.solve_algebraic, 0, ps.x_0, t_end, max_step=t_tops, dim_w = 2)

    def b_func(t, x, v):
        mat = np.zeros((len(sol.x), sol.dim_w))
        mat[load_state_idx_g[0], 0] = 0.1
        mat[load_state_idx_b[0], 0] = 0.1
        mat[load_state_idx_g[1], 1] = 0.1
        mat[load_state_idx_b[1], 1] = 0.1
        return mat

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
        speeds = ps.gen['GEN'].speed(x, v).copy()
        # Create separate dictonary entries for each generator in the system.
        [res['w%d' % i].append(speed) for i, speed in enumerate(speeds)]
    
    
    k = int(t_save/t_tops) # I assume that TOPS give a constant sampling rate

    # Here I write the results to a parquet file. This will slow everything down. However, if we keep
    # everything in memory, we may run out of memory. Another nice thing with writing to file is that
    # if the code crashes results will have been saved and we can later continue from where it crashed.
    pd.DataFrame(data=res).iloc[::k, :].to_parquet("data/res_sim_%d_R_%.2f_H_%d.parquet" % (sim_i, R, H))

n_sim = 10
sim_i = np.arange(n_sim)
r_vals = np.arange(start=0.02, step=0.01, stop=0.03)
h_vals = np.arange(start=3, step=1, stop=4)

items = list(itertools.product(sim_i, r_vals, h_vals))

# Number of processes
n_processes = 10

if __name__ == '__main__':
    start = time.time()
    with Pool(n_processes) as pool:
        pool.starmap(simulate_sys, items)
    end = time.time()
    cases = len(items)
    print("Ran ", cases, " cases in ", (end-start), "seconds")

