import tops.dynamic as dps
from tops.simulator import Simulator
import pandas as pd
import matplotlib.pyplot as plt


class ResultKeeper:
    def __init__(self, sim):
        self.t = []
        self.x = []
        self.sim = sim

    def update(self, sim):
        self.t.append(sim.sol.t)
        self.x.append(sim.sol.x.copy())

    def get_dataframe(self):
        df = pd.DataFrame(columns=self.sim.ps.state_desc, data=self.x, index=self.t)
        return df


class Events:
    def __init__(self, sim, data):
        self.data = data
        self.next_event_time = None
        self.next_event_data = None

    def update(self, sim):
        if len(self.data) == 0 and self.next_event_time is None:
            # Could be stopped
            return
        
        if self.next_event_time is None:
            self.next_event_time, self.next_event_data = self.data.pop(0)

        if self.next_event_time <= sim.sol.t:
            if self.next_event_data[0] == 'line':
                sim.ps.lines['Line'].event(sim.ps, self.next_event_data[1], self.next_event_data[2])
            self.next_event_time = None


if __name__ == '__main__':

    import tops.ps_models.ieee39 as model_data

    model = model_data.load()

    ps = dps.PowerSystemModel(model=model)
    ps.init_dyn_sim()

    # ps.ode_fun(0, ps.x0)
    sim = Simulator(ps, dt=5e-3, t_end=10)
    res_keeper = ResultKeeper(sim)
    events = Events(sim, [
        (1, ('line', 'L16-19', 'disconnect')),
        (1.2, ('line', 'L16-19', 'connect')),
    ])

    sim.interface_functions['ResultKeeper'] = res_keeper.update
    sim.interface_functions['Events'] = events.update
    sim.main_loop()
    

    print('Done')
    df = res_keeper.get_dataframe()
    self = res_keeper

    index = pd.MultiIndex.from_tuples([tuple(row) for row in sim.ps.state_desc], names=['Model', 'state'])

    df = pd.DataFrame(columns=index, data=self.x, index=self.t)

    df[('G1', 'speed')].plot()
    plt.show()