import sys
from collections import defaultdict
import matplotlib.pyplot as plt
import time
import tops.dynamic as dps
from tops.simulator import Simulator


if __name__ == '__main__':

    # Load model
    import tops.ps_models.ieee39 as model_data
    model = model_data.load()

    # Disconnect line L16-19 at 1 s, and reconnect it at 1.2 s
    model['events'] = [
        ['t',   'category', 'model', 'unit',   'action'],
        [1,     'lines',    'Line',  'L16-19', 'disconnect'],
        [1.2,   'lines',    'Line',  'L16-19', 'connect'],
    ]

    # Power system model
    ps = dps.PowerSystemModel(model=model)
    ps.init_dyn_sim()

    sim = Simulator(ps, dt=5e-3, t_end=10)
    res = []
    sim.interface_functions["res_store"] = lambda sim: res.append(ps.gen["GEN"].speed(sim.sol.x, sim.sol.v).copy())
    sim.main_loop()

    plt.plot(res)
    plt.show()