import numpy as np
import pytest
import tops.dynamic as dps


def k2a_with_switch(closed=1):
    """k2a, with line L7-8-1 connected to bus 7 through switch S7."""
    import tops.ps_models.k2a as model_data

    model = model_data.load()
    model["buses"].append(["B7S", 230])
    lines = model["lines"]
    lines[[row[0] for row in lines].index("L7-8-1")][1] = "B7S"
    model["switches"] = [
        ["name", "from_bus", "to_bus", "closed"],
        ["S7", "B7", "B7S", closed],
    ]
    return model


def test_switch_init():
    ps = dps.PowerSystemModel(model=k2a_with_switch())
    ps.init_dyn_sim()
    assert max(abs(ps.ode_fun(0, ps.x0))) < 1e-8

    # A closed switch gives the same operating point as connecting the line directly
    import tops.ps_models.k2a as model_data

    ps_ref = dps.PowerSystemModel(model=model_data.load())
    ps_ref.init_dyn_sim()
    assert np.allclose(ps.v0[: ps_ref.n_bus], ps_ref.v0, atol=1e-5)
    assert abs(ps.v0[-1] - ps.v0[ps.buses["name"] == "B7"][0]) < 1e-5

    # The switch carries the power of the line
    switch, line = ps.switches["Switch"], ps.lines["Line"]
    p_line = line.p_from(ps.x0, ps.v0)[line.par["name"] == "L7-8-1"]
    assert np.allclose(switch.p_to(ps.x0, ps.v0), -p_line, atol=1e-5)


def test_open_switch_init():
    ps = dps.PowerSystemModel(model=k2a_with_switch(closed=0))
    ps.init_dyn_sim()
    assert max(abs(ps.ode_fun(0, ps.x0))) < 1e-8
    assert ps.switches["Switch"].i_from(ps.x0, ps.v0) == 0


def test_open_switch_cutting_off_bus_raises():
    # Load L1 moved to a bus that is only connected through an open switch
    model = k2a_with_switch()
    model["buses"].append(["B7L", 230])
    model["switches"].append(["S7L", "B7", "B7L", 0])
    loads = model["loads"]
    loads[[row[0] for row in loads].index("L1")][1] = "B7L"
    ps = dps.PowerSystemModel(model=model)
    with pytest.raises(ValueError, match="B7L"):
        ps.power_flow()


if __name__ == "__main__":
    test_switch_init()
    test_open_switch_init()
    test_open_switch_cutting_off_bus_raises()
