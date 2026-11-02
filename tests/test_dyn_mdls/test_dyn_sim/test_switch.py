import numpy as np
import tops.dynamic as dps
from tops.simulator import Simulator


def k2a_model():
    """k2a, without shunt capacitance on L7-8-1, so that the line carries nothing when open at one end."""
    import tops.ps_models.k2a as model_data

    model = model_data.load()
    lines = model["lines"]
    lines[[row[0] for row in lines].index("L7-8-1")][lines[0].index("B")] = 0
    return model


def k2a_with_switch():
    """k2a_model, with line L7-8-1 connected to bus 7 through switch S7."""
    model = k2a_model()
    model["buses"].append(["B7S", 230])
    lines = model["lines"]
    lines[[row[0] for row in lines].index("L7-8-1")][1] = "B7S"
    model["switches"] = [
        ["name", "from_bus", "to_bus"],
        ["S7", "B7", "B7S"],
    ]
    return model


def simulate(model, events):
    ps = dps.PowerSystemModel(model=model)
    ps.init_dyn_sim()
    sim = Simulator(ps, dt=5e-3, t_end=2, events=events)
    speed = []
    sim.interface_functions["speed"] = lambda sim: speed.append(ps.gen["GEN"].speed(sim.sol.x, sim.sol.v).copy())
    sim.main_loop()
    return ps, sim, np.array(speed)


def test_switch_open_close():
    header = ["t", "category", "model", "unit", "action"]

    # Opening the switch is the same as disconnecting the line
    _, _, speed_ref = simulate(k2a_model(), [header, [0.5, "lines", "Line", "L7-8-1", "disconnect"]])
    ps, sim, speed = simulate(k2a_with_switch(), [header, [0.5, "switches", "Switch", "S7", "open"]])
    assert np.max(np.abs(speed_ref)) > 1e-4
    assert np.allclose(speed, speed_ref, atol=1e-6)
    assert ps.switches["Switch"].i_from(sim.sol.x, sim.sol.v) == 0

    # Closing it again reverses the change
    ps.apply_event("switches", "Switch", "S7", "close")
    assert abs(ps.switches["Switch"].i_from(sim.sol.x, sim.sol.v)[0]) > 0
