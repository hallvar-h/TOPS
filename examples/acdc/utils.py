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