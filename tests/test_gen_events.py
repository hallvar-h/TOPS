"""Generators out of service: the connect and disconnect events, and the connected column."""

import numpy as np
import pytest

import tops.dynamic as dps
import tops.ps_models.k2a as k2a
from tops.solvers import ModifiedEulerDAE


def k2a_model(offline=()):
    """The two-area system, with the generators `offline` out of service from the start, and the load in their area
    (L1 for G1 and G2, L2 for G3 and G4) lowered by their production so that the load flow balances."""
    model = k2a.load()
    header, *rows = model["generators"]["GEN"]
    model["generators"]["GEN"] = [header + ["connected"]] + [row + [row[0] not in offline] for row in rows]
    loads = model["loads"]
    load_row = {"G1": 1, "G2": 1, "G3": 2, "G4": 2}
    for row in rows:
        if row[0] in offline:
            loads[load_row[row[0]]][loads[0].index("P")] -= row[header.index("P")]
    return model


def solver(ps, dt=10e-3):
    return ModifiedEulerDAE(ps.state_derivatives, ps.solve_algebraic, 0, ps.x0.copy(), v0=ps.v0.copy(), dt=dt)


def run(sol, until):
    """Step the solver until `until` (s)."""
    while sol.t < until - 1e-9:
        sol.step()


def test_a_disconnected_generator_gives_no_power_and_comes_back():
    ps = dps.PowerSystemModel(model=k2a_model())
    ps.init_dyn_sim()
    gen = ps.gen["GEN"]
    p_0 = gen.P_e(ps.x0, ps.v0)
    assert ps.get_events("gen", "GEN").actions == ("connect", "disconnect")

    sol = solver(ps)
    run(sol, 1)
    ps.apply_event("gen", "GEN", "G2", "disconnect")
    run(sol, 20)
    p = gen.P_e(sol.x, sol.v)
    assert p[1] == pytest.approx(0, abs=1e-9)
    assert gen.Q_e(sol.x, sol.v)[1] == pytest.approx(0, abs=1e-9)
    # The others take over (the loads, of constant impedance, also fall with the voltage and frequency)
    assert (p[[0, 2, 3]] - p_0[[0, 2, 3]] > 100).all()
    # It is kept in step with its bus, at the frequency of the others
    speed = gen.speed(sol.x, sol.v)
    assert speed[1] == pytest.approx(speed[0], abs=1e-4)

    # Connected again, it takes its share back
    ps.apply_event("gen", "GEN", "G2", "connect")
    run(sol, 60)
    np.testing.assert_allclose(gen.P_e(sol.x, sol.v), p_0, rtol=0.02)


def test_a_generator_can_start_out_of_service():
    ps = dps.PowerSystemModel(model=k2a_model(offline=["G2"]))
    ps.init_dyn_sim()
    gen = ps.gen["GEN"]
    assert list(gen.connected) == [True, False, True, True]
    assert np.abs(ps.state_derivatives(0, ps.x0, ps.v0)).max() < 1e-10
    assert gen.P_e(ps.x0, ps.v0)[1] == 0
    # It runs at no load, excited to its voltage setpoint
    assert gen.v_t_abs(ps.x0, ps.v0)[1] == pytest.approx(gen.par["V"][1])

    # A load step changes the frequency; the synchronizer keeps the generator in step, so it connects smoothly
    ps.apply_event("loads", "Load", "L1", "set_power", P=300.0)
    sol = solver(ps)
    run(sol, 20)
    assert abs(gen.speed(sol.x, sol.v)[0]) > 1e-4  # The frequency has moved
    p_before = gen.P_e(sol.x, sol.v)
    ps.apply_event("gen", "GEN", "G2", "connect")
    sol.step()
    assert gen.connected[1]
    assert abs(gen.P_e(sol.x, sol.v)[1]) < 20  # MW, of 900 MVA
    assert np.abs(gen.P_e(sol.x, sol.v) - p_before)[[0, 2, 3]].max() < 20


def test_slack_unit_with_generators_out_of_service():
    # G1 out of service: the slack unit G3 keeps its index in the full table
    ps = dps.PowerSystemModel(model=k2a_model(offline=["G1"]))
    ps.init_dyn_sim()
    assert ps.slack_unit[1] == 2
    assert np.abs(ps.state_derivatives(0, ps.x0, ps.v0)).max() < 1e-10

    # G3, the only generator at the slack bus B3, out of service: the default slack bus is used instead
    ps = dps.PowerSystemModel(model=k2a_model(offline=["G3"]))
    with pytest.warns(UserWarning, match="slack bus B3"):
        ps.init_dyn_sim()
    assert ps.power_flow_ready
    assert ps.slack_bus != "B3"
    assert ps.gen["GEN"].par["name"][ps.slack_unit[1]] != "G3"
    assert np.abs(ps.state_derivatives(0, ps.x0, ps.v0)).max() < 1e-10


def test_events_twice_and_unknown_generators():
    ps = dps.PowerSystemModel(model=k2a_model())
    ps.init_dyn_sim()
    y_0 = ps.y_bus_red.copy()
    ps.apply_event("gen", "GEN", "G2", "connect")  # already connected: nothing happens
    assert abs(ps.y_bus_red - y_0).max() == 0
    ps.apply_event("gen", "GEN", "G2", "disconnect")
    ps.apply_event("gen", "GEN", "G2", "disconnect")
    ps.apply_event(ps.gen.GEN.events.connect, "G2")
    assert abs(ps.y_bus_red - y_0).max() < 1e-12
    with pytest.raises(KeyError, match="G9"):
        ps.apply_event("gen", "GEN", "G9", "disconnect")
