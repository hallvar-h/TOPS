"""The set_reference event of the governors: the power a governor gives in steady state at rated speed."""

import numpy as np
import pytest

import tops.dynamic as dps
import tops.ps_models.sm_ib as sm_ib
from tops.solvers import ModifiedEulerDAE

GOVERNORS = {
    "TGOV1": [
        ["name", "gen", "R", "D_t", "V_min", "V_max", "T_1", "T_2", "T_3"],
        ["GOV1", "G1", 0.05, 0.02, 0, 1, 0.5, 2, 2],
    ],
    "HYGOV": [
        [
            "name",
            "gen",
            "R",
            "r",
            "T_f",
            "T_r",
            "T_g",
            "A_t",
            "T_w",
            "q_nl",
            "D_turb",
            "g_min",
            "V_elm",
            "g_max",
            "P_N",
        ],
        # A fast PI regulator (T_r), so the test settles in 120 s
        ["GOV1", "G1", 0.04, 0.1, 0.1, 2, 0.5, 1.2, 1, 0.08, 0.01, 0, 0.15, 1, 0],
    ],
    "IEESGO": [
        ["name", "gen", "T_1", "T_2", "K_1", "T_3", "T_4", "K_3", "T_6", "K_2", "T_5", "P_N", "P_min", "P_max"],
        ["GOV1", "G1", 0.3, 1.0, 10.0, 1.0, 0.1, 0.4, 0.2, 0.4, 1.0, 0.0, 0.0, 1.0],
    ],
    # The output scaled from P_N (MW) to the rating of the generator
    "IEESGO P_N": [
        ["name", "gen", "T_1", "T_2", "K_1", "T_3", "T_4", "K_3", "T_6", "K_2", "T_5", "P_N", "P_min", "P_max"],
        ["GOV1", "G1", 0.3, 1.0, 10.0, 1.0, 0.1, 0.4, 0.2, 0.4, 1.0, 1200.0, 0.0, 1.0],
    ],
}


def smib(governor):
    """The machine against an infinite bus, with the governor given, initialized."""
    model = sm_ib.load()
    model["gov"] = {governor.split()[0]: GOVERNORS[governor]}
    ps = dps.PowerSystemModel(model=model)
    ps.power_flow()
    ps.init_dyn_sim()
    return ps


def governor_of(ps, governor):
    return ps.gov[governor.split()[0]]


@pytest.mark.parametrize("governor", GOVERNORS)
def test_the_reference_of_the_initial_power_is_the_initial_bias(governor):
    ps = smib(governor)
    gov = governor_of(ps, governor)
    p_m = gov.output(ps.x0, ps.v0)
    assert np.allclose(gov.reference_bias(p_m), gov._input_values["bias"], rtol=1e-12)


@pytest.mark.parametrize("governor", GOVERNORS)
def test_the_machine_settles_at_the_new_reference(governor):
    ps = smib(governor)
    gov = governor_of(ps, governor)
    p_0 = gov.output(ps.x0, ps.v0)[0]
    assert ps.get_events("gov", type(gov).__name__).actions == ("set_reference",)

    ps.apply_event("gov", type(gov).__name__, "GOV1", "set_reference", P_m=p_0 + 0.1)
    sol = ModifiedEulerDAE(ps.state_derivatives, ps.solve_algebraic, 0, ps.x0.copy(), v0=ps.v0.copy(), dt=10e-3)
    while sol.t < 120:
        sol.step()
    gen = ps.gen["GEN"]
    # The infinite bus holds the speed, so the power is the reference
    assert abs(gen.speed(sol.x, sol.v)[0]) < 1e-6
    assert gov.output(sol.x, sol.v)[0] == pytest.approx(p_0 + 0.1, abs=1e-4)
    assert gen.P_e(sol.x, sol.v)[0] == pytest.approx(gen.par["S_n"][0] * (p_0 + 0.1), rel=2e-3)


def test_a_reference_must_be_a_number_of_a_known_governor():
    ps = smib("TGOV1")
    with pytest.raises(ValueError, match="finite number"):
        ps.apply_event("gov", "TGOV1", "GOV1", "set_reference", P_m="a lot")
    with pytest.raises(KeyError, match="GOV2"):
        ps.apply_event("gov", "TGOV1", "GOV2", "set_reference", P_m=0.5)
    assert ps.gov["TGOV1"]._input_values["bias"][0] == pytest.approx(0.05 * 600 / 900)
