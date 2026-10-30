"""Power flow: collects injections from models, solves, and distributes the solution back to the units."""
from dataclasses import dataclass, field

import numpy as np

from tops.utility_functions import newton_rhapson_power_flow


@dataclass
class PowerFlowResult:
    v: np.ndarray  # Complex bus voltages (p.u.)
    s: np.ndarray  # Complex bus power injections (p.u.)
    converged: bool
    slack_unit: tuple = None  # (model, unit index) of the unit at the slack bus, if any
    unit_injections: dict = field(default_factory=dict)  # model -> complex power per unit (MVA)


def default_slack_bus_idx(pv_mdls):
    """Index of the bus of the PV unit with the largest P setpoint, used as slack bus if none is specified.

    Ties are resolved in favour of the first unit found.
    """
    bus_idx, p_setp = zip(*(mdl.load_flow_pv()[:2] for mdl in pv_mdls))
    bus_idx = np.concatenate(bus_idx)
    p_setp = np.concatenate(p_setp)
    return bus_idx[np.argmax(p_setp)]


def solve_power_flow(y_bus, pq_mdls, pv_mdls, s_n, sl_idx, tol, max_it):
    """Solve the power flow.

    :param y_bus: Bus admittance matrix for the power flow.
    :param pq_mdls: Models with load_flow_pq(), returning (bus_idx, P, Q) in MW/MVAr.
    :param pv_mdls: Models with load_flow_pv(), returning (bus_idx, P setpoint, V setpoint).
    :param s_n: System base power (MVA).
    :param sl_idx: Index of the slack bus.
    """
    n_bus = y_bus.shape[0]
    bus_type = np.array(['PQ'] * n_bus, dtype='<U2')

    p_pq = np.zeros(n_bus)
    q_pq = np.zeros(n_bus)
    for mdl in pq_mdls:
        bus_idx, p, q = mdl.load_flow_pq()
        np.add.at(p_pq, bus_idx, p / s_n)
        np.add.at(q_pq, bus_idx, q / s_n)

    p_pv = np.zeros(n_bus)
    v_pv = np.ones(n_bus)
    # The slack unit is the unit at the slack bus with the largest P setpoint (first one found on ties)
    slack_unit = None
    p_slack_unit = -np.inf
    for mdl in pv_mdls:
        bus_idx, p, v = mdl.load_flow_pv()
        at_slack = np.flatnonzero(bus_idx == sl_idx)
        if len(at_slack) > 0:
            unit_idx = at_slack[np.argmax(p[at_slack])]
            if p[unit_idx] > p_slack_unit:
                slack_unit = (mdl, unit_idx)
                p_slack_unit = p[unit_idx]
        bus_type[bus_idx] = 'PV'
        np.add.at(p_pv, bus_idx, p / s_n)
        v_pv[bus_idx] = v

    bus_type[sl_idx] = 'SL'

    v_sol, s_sol, converged = newton_rhapson_power_flow(y_bus, v_pv, p_pv + p_pq, q_pq, bus_type, tol, max_it)

    # Reactive power at each PV bus is shared equally between the units at that bus
    pv_units_per_bus = np.zeros(n_bus, dtype=int)
    for mdl in pv_mdls:
        np.add.at(pv_units_per_bus, mdl.load_flow_pv()[0], 1)

    q_bus = s_sol.imag + q_pq
    q_per_pv_unit = np.divide(q_bus, pv_units_per_bus, out=np.zeros_like(q_bus), where=pv_units_per_bus != 0)

    unit_injections = {}
    for mdl in pv_mdls:
        bus_idx, p_setp, v = mdl.load_flow_pv()
        q = q_per_pv_unit[bus_idx]
        p = -p_setp / s_n
        if slack_unit is not None and slack_unit[0] is mdl:
            # The slack unit takes the power balance at the slack bus
            unit_idx = slack_unit[1]
            p[unit_idx] = s_sol[sl_idx].real - p_setp[unit_idx] / s_n + p_pv[sl_idx] + p_pq[sl_idx]

        unit_injections[mdl] = (p + 1j * q) * s_n

    return PowerFlowResult(v_sol, s_sol, converged, slack_unit, unit_injections)
