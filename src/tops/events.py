"""Events: discrete changes to the power system during simulation (line switching, short circuits, etc.).

Events of a model are defined in a ModelEvents subclass, which is set as the model's events_class. Event methods may
modify the model they belong to, but other parts of the system are only affected through the effects they return
(e.g. AdmittanceChange), which are applied by PowerSystemModel.apply_effect.
"""
from typing import NamedTuple

import numpy as np


class AdmittanceChange(NamedTuple):
    """Change in the reduced bus admittance matrix. Rows and columns are indices in the reduced system."""
    data: np.ndarray
    rows: np.ndarray
    cols: np.ndarray


class ModelEvents:
    """Base class for the events of a model (or of the buses).

    Event methods take the unit name (and possibly keyword arguments), and return a list of effects. Only methods
    listed in `actions` can be triggered by name (e.g. from the event table in the model definition).
    """
    actions = ()

    def __init__(self, mdl):
        self.mdl = mdl

    def unit_names(self):
        return self.mdl.par['name']

    def apply(self, unit, action, **kwargs):
        self.check(unit, action)
        return getattr(self, action)(unit, **kwargs)

    def check(self, unit, action):
        if action not in self.actions:
            raise ValueError(f'{type(self).__name__} has no event "{action}". Available events: {self.actions}')
        self._unit_idx(unit)

    def _unit_idx(self, unit):
        idx = np.where(self.unit_names() == unit)[0]
        if len(idx) == 0:
            raise KeyError(f'{type(self).__name__}: unit "{unit}" not found.')
        return idx[0]


class BusEvents(ModelEvents):
    """Events at buses: short circuits through a fixed fault admittance."""
    actions = ('short_circuit', 'clear_short_circuit')

    def __init__(self, bus_names, bus_idx_red):
        super().__init__(mdl=None)
        self.bus_names = bus_names
        self.bus_idx_red = bus_idx_red
        self.y_fault = np.zeros(len(bus_names), dtype=complex)

    def unit_names(self):
        return self.bus_names

    def _bus_idx_red(self, bus):
        idx = np.where(self.bus_idx_red == self._unit_idx(bus))[0]
        if len(idx) == 0:
            raise ValueError(f'Bus {bus} is not in the reduced system, short circuit not possible.')
        return idx

    def short_circuit(self, bus, y=1e6):
        idx = self._unit_idx(bus)
        if self.y_fault[idx] != 0:
            return []
        idx_red = self._bus_idx_red(bus)
        self.y_fault[idx] = y
        return [AdmittanceChange(np.array([y]), idx_red, idx_red)]

    def clear_short_circuit(self, bus):
        idx = self._unit_idx(bus)
        if self.y_fault[idx] == 0:
            return []
        idx_red = self._bus_idx_red(bus)
        y = self.y_fault[idx]
        self.y_fault[idx] = 0
        return [AdmittanceChange(np.array([-y]), idx_red, idx_red)]


class EventSchedule:
    """Events to be applied at given times, from a table with fields t, category, model, unit, action.

    The model field is ignored for category 'buses'.
    """
    fields = ('t', 'category', 'model', 'unit', 'action')

    def __init__(self, table=None):
        if table is None or len(table) == 0:
            self.events = []
        else:
            missing = [field for field in self.fields if field not in table.dtype.names]
            if missing:
                raise ValueError(f'Event table is missing fields: {missing}')
            order = np.argsort(table['t'], kind='stable')
            self.events = [tuple(table[field][i].item() for field in self.fields) for i in order]
        self.reset()

    def reset(self):
        self._next = 0

    def pop_due(self, t):
        """Return events with time <= t that have not been returned before."""
        due = []
        while self._next < len(self.events) and self.events[self._next][0] <= t:
            due.append(self.events[self._next])
            self._next += 1
        return due
