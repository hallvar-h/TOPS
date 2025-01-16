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
from tops.utility_functions import lookup_strings
    

class PowerSystemModelACDC(dps.PowerSystemModel):
    def __init__(self, model, user_mdl_lib=None):
        super().__init__(model, user_mdl_lib)
        self.circuit_model = CircuitModel(self.model['circuits'])

    def init_dyn_sim(self):
        super().init_dyn_sim()
        self.circuit_model.initialize()
        # self.circuit_model.mdls['CS'].input['i_set'](self.x0, np.zeros(self.circuit_model.n_alg))
        self.circuit_model.mdls['CS'].input['i_set'] = lambda t, x, _: self.vsc['VSC'].I_d(x, None)
        lookup_strings(self.vsc['VSC'].par['DCbus'], self.circuit_model.bus_names)
        
        
    def define_state_vector(self):
        super().define_state_vector()
        self.circuit_model.start_state_idx = self.n_states
        self.circuit_model.define_variables()
        self.n_states_cm = self.circuit_model.n_states - self.n_states
        self.n_states = self.circuit_model.n_states
        for d in self.circuit_model.state_desc:
            new_state_desc = [f'{d[0]}: {d[1]}', d[2]]
            new_state_desc_der = [f'd_{d[0]}: {d[1]}', d[2]]
            self.state_desc = np.vstack([self.state_desc, new_state_desc])
            self.state_desc_der = np.vstack(
                [self.state_desc_der, new_state_desc_der])
            
    def state_derivatives(self, t, x, v_red):
        dx = super().state_derivatives(t, x, v_red)
        circuit_model = self.circuit_model
        y = np.zeros(circuit_model.n_alg)
        u = circuit_model.generate_input(t, x, y)
        y = circuit_model.lu.solve(
            -circuit_model.M_x.dot(x) - circuit_model.M_u.dot(u))
        dx_circuit = self.circuit_model.dxdt(t, x, y)
        dx[-self.n_states_cm:] = dx_circuit[-self.n_states_cm:]
        return dx


if __name__ == '__main__':

    # Load data
    import json
    with open('examples/acdc/model.json', 'r') as file:
        model = json.load(file)

    circuit_model_data = model['circuits']
    # del model['circuits']

    # Initialize
    ps = PowerSystemModelACDC(model=model)
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
    sim.simulate(t_end=0.0005)  # 0.002)
    print(f'Simulation time: {time.time() - t_0:.2f} seconds.')

    

    # print('Done')
    # df = res_keeper.get_dataframe()
    # self = res_keeper

    # index = pd.MultiIndex.from_tuples([tuple(row) for row in tops_sim.ps.state_desc], names=['Model', 'state'])

    # df = pd.DataFrame(columns=index, data=self.x, index=self.t)

    # df[('IB', 'speed')].plot()
    # plt.show()

    # t, x, y, u = res.get_arrays()
    # fig, ax = plt.subplots(4, sharex=True)
    # ax[0].plot(t, x)
    # ax[1].plot(t, y)
    # ax[2].plot(t, y[:, [0, -1]])
    # ax[3].plot(t, u)

    # plt.show()
