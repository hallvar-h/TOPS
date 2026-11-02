import tops.solvers as dps_sol
import threading
import numpy as np
from collections import defaultdict
import pandas as pd

from tops.model_data import read_table


class Simulator:
    def __init__(self, ps, dt=5e-3, solver=dps_sol.ModifiedEulerDAE, t_end=np.inf, log_fun=None, ode_fun=None,
                 events=None):
        """
        :param ps: Initialized PowerSystemModel.
        :param events: Events to apply during the simulation, as a table (list with header row, or DataFrame)
            with columns t, category, model, unit, action. See EventSchedule.
        """

        self._stopped = False
        self.n_markers = 20
        self.t_end = t_end
        self.dt = dt
        self.log_fun = log_fun
        self.log = callable(self.log_fun)
        self.ps = ps
        self.events = EventSchedule(events)
        self.events.check(self.ps)

        if callable(ode_fun):
            self.ode_fun = ode_fun
        else:
            self.ode_fun = self.ps.ode_fun

        self.sol = solver(self.ps.state_derivatives, self.ps.solve_algebraic, 0, self.ps.x0, self.ps.v0, max_step=self.dt, first_step=self.dt)

        self.new_data_cv = threading.Condition()
        self.new_data_ready = False
        self.x = self.sol.y
        self.t = self.sol.t

        self.interface_functions = dict()
        self.interface_functions_lock = threading.Lock()
        self.interface_timers = dict()


    def stopped(self):
        return self._stopped

    def make_simulation_step(self):
        # Simulate next step
            with self.new_data_cv:
                self.events.apply_due(self.ps, self.sol.t)
                self.sol.step()

                with self.interface_functions_lock:
                    for key, fun in self.interface_functions.items():
                        fun(self)

                self.new_data_ready = True
                self.new_data_cv.notify()
    
    def main_loop(self):
        t = self.sol.t
        while not self.stopped() and t < self.t_end:
            self.make_simulation_step()
            t = self.sol.t


class EventSchedule:
    """Events to be applied at given times.

    Events are given as a table (list with header row, or DataFrame) with columns t, category, model, unit, action,
    e.g. [0.1, 'lines', 'Line', 'L1-2', 'disconnect'] or [0.1, 'buses', '', 'B1', 'short_circuit']. The model column
    is ignored for category 'buses'. An optional column kwargs holds a dict of arguments for the action, or None,
    e.g. [0.1, 'buses', '', 'B1', 'short_circuit', {'y': 100}]; they are kept in the list kwargs, one per event.

    Used by the Simulator, or directly in a custom simulation loop:

        schedule = EventSchedule(events)
        schedule.check(ps)
        while t < t_end:
            schedule.apply_due(ps, t)
            ...
    """
    fields = ('t', 'category', 'model', 'unit', 'action')

    def __init__(self, events=None):
        table = read_table(events) if events is not None else None
        if table is None or len(table) == 0:
            self.events = []
            self.kwargs = []
        else:
            missing = [field for field in self.fields if field not in table.dtype.names]
            if missing:
                raise ValueError(f'Event table is missing fields: {missing}')
            order = np.argsort(table['t'], kind='stable')
            self.events = [tuple(table[field][i].item() for field in self.fields) for i in order]
            kwargs = table['kwargs'] if 'kwargs' in table.dtype.names else [None]*len(table)
            self.kwargs = [self._read_kwargs(kwargs[i]) for i in order]
        self.reset()

    @staticmethod
    def _read_kwargs(value):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return {}
        if isinstance(value, dict):
            return dict(value)
        raise ValueError(f"The 'kwargs' of an event must be a dict or None, got {value!r}.")

    def reset(self):
        self._next = 0

    def check(self, ps):
        """Check that all events exist in the (initialized) power system model."""
        for _, category, mdl_key, unit, action in self.events:
            ps.get_events(category, mdl_key).check(unit, action)

    def pop_due(self, t):
        """Return events with time <= t that have not been returned before."""
        due = []
        while self._next < len(self.events) and self.events[self._next][0] <= t:
            due.append(self.events[self._next])
            self._next += 1
        return due

    def apply_due(self, ps, t):
        """Apply events with time <= t that have not been applied before."""
        first = self._next
        for (_, category, mdl_key, unit, action), kwargs in zip(self.pop_due(t), self.kwargs[first:self._next]):
            ps.apply_event(category, mdl_key, unit, action, **kwargs)


class Events:
    """Events given as a list of (t, ('line', line_name, action)).

    Kept for backwards compatibility. Events can instead be given to the Simulator (Simulator(ps, ..., events=...)).
    """
    def __init__(self, sim, data):
        self.data = sorted(data, key=lambda event: event[0])
        self._next = 0

    def update(self, sim):
        while self._next < len(self.data) and self.data[self._next][0] <= sim.sol.t:
            _, (category, name, action) = self.data[self._next]
            if category != 'line':
                raise ValueError(f'Events only supports line events, not "{category}". Use Simulator(ps, ..., events=...) instead.')
            sim.ps.apply_event('lines', 'Line', name, action)
            self._next += 1


class InterfacerDirect:
    def __init__(self, rts=None, name='InterfacerDirect', fs=None):
        """:param fs: Updates per second of simulated time, or None to update at every step."""
        self.interface_name = name
        self.fs = fs
        self._timer = 0

        if rts:
            self.connect(rts)

    def connect(self, rts):
        rts.interface_functions[self.interface_name] = self.interface_fun

    def interface_fun(self, rts):
        if self.fs:
            if rts.sol.t < self._timer:
                return
            else:
                self._timer += 1 / self.fs

        self.update(self.read_input_signal(rts))
        self.apply_ctrl_signal(rts, self.generate_ctrl_signal())

    def read_input_signal(self, rts):
        # Read input from RealTimeSimulator
        pass

    def apply_ctrl_signal(self, rts, ctrl_signal):
        # Apply control in RealTimeSimulator, generated by "generate_ctrl_signal"
        pass

    def generate_ctrl_signal(self):
        # Generate control signal from internal states
        pass

    def update(self, input):
        # Update internal states from whatever is returned by "read_input_signal"
        pass


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