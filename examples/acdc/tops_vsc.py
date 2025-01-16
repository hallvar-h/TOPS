import tops.dynamic as dps
from tops.simulator import Simulator
import pandas as pd
import matplotlib.pyplot as plt
from utils import Events, ResultKeeper


if __name__ == '__main__':

    import json
    with open('examples/acdc/model.json', 'r') as file:
        model = json.load(file)

    del model['circuits']

    ps = dps.PowerSystemModel(model=model)
    ps.init_dyn_sim()

    sim = Simulator(ps, dt=5e-3, t_end=10)
    res_keeper = ResultKeeper(sim)
    events = Events(sim, [
        (1, ('line', 'L1-2', 'disconnect')),
        (1.2, ('line', 'L1-2', 'connect')),
    ])

    sim.interface_functions['ResultKeeper'] = res_keeper.update
    sim.interface_functions['Events'] = events.update
    sim.main_loop()
    

    print('Done')
    df = res_keeper.get_dataframe()
    self = res_keeper

    index = pd.MultiIndex.from_tuples([tuple(row) for row in sim.ps.state_desc], names=['Model', 'state'])

    df = pd.DataFrame(columns=index, data=self.x, index=self.t)

    df[('IB', 'speed')].plot()
    plt.show()