"""Switches (breakers, disconnectors): ideal connections between two buses, modelled as a very high admittance."""

import numpy as np
from tops.dyn_models.utils import DAEModel, read_connected
from tops.events import ModelEvents, AdmittanceChange


class SwitchEvents(ModelEvents):
    actions = ("open", "close")

    def close(self, switch_name):
        idx = self._unit_idx(switch_name)
        if self.mdl.connected[idx]:
            return []
        self.mdl.connected[idx] = True
        return [self.mdl.admittance_change(idx, sign=1)]

    def open(self, switch_name):
        idx = self._unit_idx(switch_name)
        if not self.mdl.connected[idx]:
            return []
        self.mdl.connected[idx] = False
        return [self.mdl.admittance_change(idx, sign=-1)]


class Switch(DAEModel):
    """Switch between two buses.

    A closed switch is a series admittance ``y`` (pu on system base, default :attr:`y_closed`), large enough that the
    voltage across it is negligible. An open switch is removed from the network.

    Parameters (table columns): ``name``, ``from_bus``, ``to_bus``, and optionally ``closed`` (initial state, boolean
    or 1/0, default closed) and ``y`` (admittance when closed, pu on system base). The state is kept in ``connected``,
    like for lines and transformers.
    """

    events_class = SwitchEvents
    y_closed = 1e5

    def __init__(self, data, sys_par, **kwargs):
        super().__init__(data, sys_par, **kwargs)
        self.par = data
        self.n_units = len(data)
        names = data.dtype.names
        self.connected = read_connected(data, column="closed")  # True when the switch is closed
        self.admittance = data["y"].astype(complex) if "y" in names else np.full(self.n_units, self.y_closed, complex)

        self.bus_idx = np.array(np.zeros(self.n_units), dtype=[(key, int) for key in self.bus_ref_spec().keys()])
        self.bus_idx_red = np.array(np.zeros(self.n_units), dtype=[(key, int) for key in self.bus_ref_spec().keys()])
        self.sys_par = sys_par

    def bus_ref_spec(self):
        return {"from_bus": self.par["from_bus"], "to_bus": self.par["to_bus"]}

    def admittance_change(self, idx, sign):
        """Change in the reduced admittance matrix when closing (sign=1) or opening (sign=-1) a switch."""
        idx_from = self.bus_idx_red["from_bus"][idx]
        idx_to = self.bus_idx_red["to_bus"][idx]
        if idx_from < 0 or idx_to < 0:
            raise ValueError(f"Buses of switch {self.par['name'][idx]} are not in the reduced system.")

        y = self.admittance[idx]
        data = sign * np.array([y, y, -y, -y])
        rows = np.array([idx_from, idx_to, idx_from, idx_to])
        cols = np.array([idx_from, idx_to, idx_to, idx_from])
        return AdmittanceChange(data, rows, cols)

    def load_flow_adm(self):
        V_n_from = self.sys_par["bus_v_n"][self.bus_idx["from_bus"]]
        self.I_n = self.sys_par["s_n"] / (np.sqrt(3) * V_n_from)

        idx_from = self.bus_idx["from_bus"]
        idx_to = self.bus_idx["to_bus"]
        y = self.admittance * self.connected
        rows = np.array([idx_from, idx_to, idx_from, idx_to])
        cols = np.array([idx_from, idx_to, idx_to, idx_from])
        data = np.array([y, y, -y, -y])
        return data, (rows, cols)

    def dyn_const_adm(self):
        return self.load_flow_adm()

    def i_from(self, x, v):
        # Current in p.u., flowing from from_bus into the switch
        idx_from = self.bus_idx_red["from_bus"]
        idx_to = self.bus_idx_red["to_bus"]
        return self.admittance * (v[idx_from] - v[idx_to]) * self.connected

    def i_to(self, x, v):
        return -self.i_from(x, v)

    def I_from(self, x, v):
        # Current in kA
        return self.i_from(x, v) * self.I_n

    def s_from(self, x, v):
        return v[self.bus_idx_red["from_bus"]] * np.conj(self.i_from(x, v))

    def s_to(self, x, v):
        return v[self.bus_idx_red["to_bus"]] * np.conj(self.i_to(x, v))

    def p_from(self, x, v):
        return self.s_from(x, v).real

    def q_from(self, x, v):
        return self.s_from(x, v).imag

    def p_to(self, x, v):
        return self.s_to(x, v).real

    def q_to(self, x, v):
        return self.s_to(x, v).imag
