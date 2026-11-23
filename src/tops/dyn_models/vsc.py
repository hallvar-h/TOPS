import numpy as np
from tops.dyn_models.utils import DAEModel
from tops.dyn_models.blocks import PIRegulator, TimeConstant
from .pll import PLL1


class VSC(DAEModel):
    """Grid-following voltage source converter, modelled as a controlled current source.

    The active and reactive power injected at the terminal bus are measured and controlled by PI
    regulators. The outputs of the regulators, filtered by the time constant ``T_i``, give the
    d- and q-axis currents, which are aligned with the voltage angle estimated by a PLL
    (``PLL1``).

    Quantities are in physical units, not per unit: power in MW and Mvar, current in kA and
    voltage in kV. The gains of the regulators are thus in kA/MW (kA/Mvar).

    Parameters (columns in the model dict):

    - ``bus``: Terminal bus.
    - ``T_pll``: Time constant of the PLL angle filter [s].
    - ``T_i``: Time constant of the current response [s].
    - ``P_K_p``, ``P_K_i``: Proportional and integral gain of the active power regulator.
    - ``Q_K_p``, ``Q_K_i``: Proportional and integral gain of the reactive power regulator.
    - ``P_setp``, ``Q_setp``: Initial active [MW] and reactive [Mvar] power setpoints. Positive
      values are injected into the grid. The setpoints are inputs, which can be changed during
      the simulation with ``set_input``.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.bus_idx = np.array(np.zeros(self.n_units), dtype=[(key, int) for key in self.bus_ref_spec().keys()])
        self.bus_idx_red = np.array(np.zeros(self.n_units), dtype=[(key, int) for key in self.bus_ref_spec().keys()])

    def bus_ref_spec(self):
        return {"terminal": self.par["bus"]}

    def add_blocks(self):
        p = self.par
        self.pll = PLL1(T_filter=self.par["T_pll"], bus=p["bus"])

        self.pi_p = PIRegulator(K_p=p["P_K_p"], K_i=p["P_K_i"])
        self.pi_p.input = lambda x, v: self.P_setp(x, v) - self.P(x, v)

        self.pi_q = PIRegulator(K_p=p["Q_K_p"], K_i=p["Q_K_i"])
        self.pi_q.input = lambda x, v: self.Q_setp(x, v) - self.Q(x, v)

        self.lag_p = TimeConstant(T=p["T_i"])
        self.lag_p.input = self.pi_p.output
        self.lag_q = TimeConstant(T=p["T_i"])
        self.lag_q.input = self.pi_q.output

        self.I_d = self.lag_p.output
        self.I_q = self.lag_q.output

    def I_inj(self, x, v):
        return (self.I_d(x, v) - 1j * self.I_q(x, v)) * np.exp(1j * self.pll.output(x, v))

    def input_list(self):
        return ["P_setp", "Q_setp"]

    def S(self, x, v):
        """Complex power [MVA] injected into the terminal bus."""
        v_n = self.sys_par["bus_v_n"][self.bus_idx_red["terminal"]]
        V = v[self.bus_idx_red["terminal"]] * v_n
        return np.sqrt(3) * V * np.conj(self.I_inj(x, v))

    def P(self, x, v):
        """Active power [MW] injected into the terminal bus."""
        return self.S(x, v).real

    def Q(self, x, v):
        """Reactive power [Mvar] injected into the terminal bus."""
        return self.S(x, v).imag

    def load_flow_pq(self):
        return self.bus_idx["terminal"], -self.par["P_setp"], -self.par["Q_setp"]

    def init_from_load_flow(self, x_0, v_0, S):
        self._input_values["P_setp"] = self.par["P_setp"]
        self._input_values["Q_setp"] = self.par["Q_setp"]

        v_n = self.sys_par["bus_v_n"][self.bus_idx_red["terminal"]]

        V_0 = v_0[self.bus_idx_red["terminal"]] * v_n

        I_d_0 = self.par["P_setp"] / (abs(V_0) * np.sqrt(3))
        I_q_0 = self.par["Q_setp"] / (abs(V_0) * np.sqrt(3))

        self.pi_p.initialize(x_0, v_0, self.lag_p.initialize(x_0, v_0, I_d_0))

        self.pi_q.initialize(x_0, v_0, self.lag_q.initialize(x_0, v_0, I_q_0))

    def current_injections(self, x, v):
        i_n = self.sys_par["s_n"] / (np.sqrt(3) * self.sys_par["bus_v_n"])
        return self.bus_idx_red["terminal"], self.I_inj(x, v) / i_n[self.bus_idx_red["terminal"]]
