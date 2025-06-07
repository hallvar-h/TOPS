import tops.dynamic as dps
from tops.simulator import Simulator, Events
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np


class ResultKeeper:
    def __init__(self, sim, **kwargs):
        self.t = []
        self.x = []
        self.sim = sim
        self.store = defaultdict(list)
        self.spec = dict(**kwargs)

    def update(self, sim):
        self.t.append(sim.sol.t)
        self.x.append(sim.sol.x.copy())
        # self.store
        for key, val in self.spec.items():
            self.store[key].append(val(sim.sol.x, sim.sol.v).copy())
        

    def get_dataframe(self):
        df = pd.DataFrame(columns=self.sim.ps.state_desc, data=self.x, index=self.t)
        return df


if __name__ == '__main__':

    import tops.ps_models.ieee39 as model_data

    model = model_data.load()

    ps = dps.PowerSystemModel(model=model)
    ps.init_dyn_sim()

    # ps.ode_fun(0, ps.x0)
    sim = Simulator(ps, dt=5e-3, t_end=10)
    res_keeper = ResultKeeper(sim,
        gen_speed=ps.gen["GEN"].speed,
        p_line_from=ps.lines["Line"].p_from)
    events = Events(sim, [
        (1, ('line', 'L16-19', 'disconnect')),
        (1.2, ('line', 'L16-19', 'connect')),
    ])

    sim.interface_functions['ResultKeeper'] = res_keeper.update
    sim.interface_functions['Events'] = events.update
    sim.main_loop()
    
    print('Done')
    df = res_keeper.get_dataframe()
    len(res_keeper.t)
    plt.plot(res_keeper.t, res_keeper.store["gen_speed"])
    plt.show()

    plt.plot(res_keeper.t, res_keeper.store["p_line_from"])
    plt.show()
    
    self = res_keeper
    index = pd.MultiIndex.from_tuples([tuple(row) for row in sim.ps.state_desc], names=['Model', 'state'])

    df = pd.DataFrame(columns=index, data=self.x, index=self.t)

    df[('G1', 'speed')].plot()
    plt.show()