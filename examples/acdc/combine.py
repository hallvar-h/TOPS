import tops.dynamic as dps
from tops.simulator import Simulator as TOPSSimulator
import pandas as pd
import matplotlib.pyplot as plt
from utils import Events as TOPSEvents, ResultKeeper as TOPSResultKeeper

from tops_circuits.circuit import CircuitModel
import numpy as np
import matplotlib.pyplot as plt
from tops_circuits.sim import Simulator, ResultKeeper, ProgressBar
from tops_circuits.models import PIEquivalent
    




if __name__ == '__main__':

    # Load data
    import json
    with open('examples/acdc/model.json', 'r') as file:
        model = json.load(file)

    circuit_model_data = model['circuits']
    del model['circuits']

    # Initialize
    ps = dps.PowerSystemModel(model=model)
    ps.init_dyn_sim()

    cm = CircuitModel(circuit_model_data)
    # cm.start_state_idx = ps.n_states
    cm.initialize()

    # Configure
    # def update_inputs(sim):
        # sim.model.mdls['CS'].set_input('i_set', (sim.t>0.0005), 0)
    cm.mdls['CS'].input['i_set'] = lambda t, x, y: t>0.0005


    tops_sim = TOPSSimulator(ps, dt=5e-3, t_end=10)
    dir(tops_sim)
    res_keeper = TOPSResultKeeper(tops_sim)
    events = TOPSEvents(tops_sim, [
        (1, ('line', 'L1-2', 'disconnect')),
        (1.2, ('line', 'L1-2', 'connect')),
    ])

    tops_sim.interface_functions['ResultKeeper'] = res_keeper.update
    tops_sim.interface_functions['Events'] = events.update
    tops_sim.main_loop()

    sim = Simulator(cm, dt=1e-8)
    # sim.update_functions.append(update_inputs)
    res = ResultKeeper(sim) 
    pb = ProgressBar(sim)

    import time
    t_0 = time.time()
    sim.simulate(t_end=0.002)
    print(f'Simulation time: {time.time() - t_0:.2f} seconds.')

    

    print('Done')
    df = res_keeper.get_dataframe()
    self = res_keeper

    index = pd.MultiIndex.from_tuples([tuple(row) for row in tops_sim.ps.state_desc], names=['Model', 'state'])

    df = pd.DataFrame(columns=index, data=self.x, index=self.t)

    df[('IB', 'speed')].plot()
    plt.show()

    t, x, y, u = res.get_arrays()
    fig, ax = plt.subplots(4, sharex=True)
    ax[0].plot(t, x)
    ax[1].plot(t, y)
    ax[2].plot(t, y[:, [0, -1]])
    ax[3].plot(t, u)

    plt.show()
