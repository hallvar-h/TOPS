import sys
from collections import defaultdict
import matplotlib.pyplot as plt
import time
import dynpssimpy.dynamic as dps
import dynpssimpy.solvers as dps_sol
import importlib
importlib.reload(dps)

def test_dyn_load_filtered():
    import dynpssimpy.ps_models.k2a as model_data
    importlib.reload(model_data)
    model = model_data.load()
    model['loads'] = {'DynamicLoadFiltered':  [# model['loads']}
        model['loads'][0] + ['T_g', 'T_b'],
        *[row + [1, 1] for row in model['loads'][1:]]
    ]}

    # Power system model
    ps = dps.PowerSystemModel(model=model)
    ps.loads['DynamicLoadFiltered']
    ps.init_dyn_sim()
    assert max(abs(ps.ode_fun(0, ps.x0))) < 1e-10
    

if __name__ == '__main__':
    test_dyn_load_filtered()