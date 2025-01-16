from tops_circuits.circuit import CircuitModel
import numpy as np
import matplotlib.pyplot as plt
from tops_circuits.sim import Simulator, ResultKeeper, ProgressBar
from tops_circuits.models import PIEquivalent
    


if __name__ == '__main__':

    import json
    with open('examples/acdc/model.json', 'r') as file:
        model = json.load(file)
    circuit_model_data = model['circuits']

    cm = CircuitModel(circuit_model_data)
    cm.initialize()

    # def update_inputs(sim):
        # sim.model.mdls['CS'].set_input('i_set', (sim.t>0.0005), 0)
    cm.mdls['CS'].input['i_set'] = lambda t, x, y: np.array([t>0.0005])

    sim = Simulator(cm, dt=1e-8)
    # sim.update_functions.append(update_inputs)
    res = ResultKeeper(sim) 
    pb = ProgressBar(sim)

    import time
    t_0 = time.time()
    sim.simulate(t_end=0.002)
    print(f'Simulation time: {time.time() - t_0:.2f} seconds.')

    t, x, y, u = res.get_arrays()
    fig, ax = plt.subplots(4, sharex=True)
    ax[0].plot(t, x)
    ax[1].plot(t, y)
    ax[2].plot(t, y[:, [0, -1]])
    ax[3].plot(t, u)

    plt.show()
