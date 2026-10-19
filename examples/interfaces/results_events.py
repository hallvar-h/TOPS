import tops.dynamic as dps
from tops.simulator import Simulator, ResultKeeper
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np


if __name__ == '__main__':

    import tops.ps_models.ieee39 as model_data

    model = model_data.load()

    ps = dps.PowerSystemModel(model=model)
    ps.init_dyn_sim()

    events = [
        ['t', 'category', 'model', 'unit', 'action'],
        [1, 'lines', 'Line', 'L16-19', 'disconnect'],
        [1.2, 'lines', 'Line', 'L16-19', 'connect'],
    ]

    sim = Simulator(ps, dt=5e-3, t_end=10, events=events)
    res_keeper = ResultKeeper(sim,
        gen_speed=ps.gen["GEN"].speed,
        p_line_from=ps.lines["Line"].p_from)

    sim.interface_functions['ResultKeeper'] = res_keeper.update
    sim.main_loop()
    
    print('Done')
    df = res_keeper.get_dataframe()
    len(res_keeper.t)
    plt.plot(res_keeper.t, res_keeper.store["gen_speed"])
    plt.show()

    plt.plot(res_keeper.t, res_keeper.store["p_line_from"])
    plt.show()


    df[('G1', 'speed')].plot()
    plt.show()