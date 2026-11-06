import numpy as np
import pytest

import tops.dynamic as dps
import tops.modal_analysis as dps_mdl
import tops.ps_models.k2a as k2a


@pytest.fixture(scope="module")
def ps():
    ps = dps.PowerSystemModel(model=k2a.load())
    ps.init_dyn_sim()
    return ps


@pytest.fixture(scope="module")
def lin(ps):
    lin = dps_mdl.PowerSystemModelLinearization(ps)
    lin.linearize(
        inputs=[(ps.avr.SEXS.v_setp, "AVR1"), (ps.gen.GEN.P_m, "G1"), ps.gov.TGOV1.bias],
        outputs=[(ps.gen.GEN.speed, ["G1", "G3"]), (ps.avr.SEXS.v_setp, 0), ps.gen.GEN.p_e],
        get_eigs=True,
    )
    return lin


def test_electromechanical_modes(lin):
    idx = lin.get_mode_idx(["em", "non_conj"], damp_threshold=0.3)
    np.testing.assert_allclose(lin.freq[idx], [1.1337, 1.0937, 0.6352], atol=1e-4)
    np.testing.assert_allclose(lin.damping[idx], [0.1388, 0.1424, 0.2264], atol=1e-4)
    assert lin.get_dominant_mode() == idx[0]


def test_shapes_and_labels(lin, ps):
    n = ps.n_states
    assert lin.a.shape == (n, n)
    assert lin.b.shape == (n, 6)
    assert lin.c.shape == (7, n)
    assert lin.d.shape == (7, 6)
    assert lin.input_desc[:3] == ["AVR1.v_setp", "G1.P_m", "GOV1.bias"]
    assert lin.output_desc[:4] == ["G1.speed", "G3.speed", "AVR1.v_setp", "G1.p_e"]


def test_b_matches_set_input(lin, ps):
    # Independent check of an unconnected input (the bias of GOV1), by changing it with set_input
    gov = ps.gov["TGOV1"]
    bias_0 = gov._input_values["bias"].copy()
    h = 1e-6
    try:
        gov.set_input("bias", bias_0[0] + h, 0)
        f_1 = ps.ode_fun(0, ps.x0)
        gov.set_input("bias", bias_0[0] - h, 0)
        f_2 = ps.ode_fun(0, ps.x0)
    finally:
        gov.set_input("bias", bias_0)
    np.testing.assert_allclose(lin.b[:, 2], (f_1 - f_2) / (2 * h), atol=1e-6)
    assert abs(lin.b[:, 2]).max() > 0


def test_connected_input_gets_added_signal(lin, ps):
    # v_setp of AVR1 is connected to the generator, so set_input has no effect. The perturbation is added to the
    # connected value instead, which moves only the states of AVR1.
    nonzero = np.flatnonzero(abs(lin.b[:, 0]) > 1e-8)
    assert len(nonzero) > 0
    assert all(name.startswith("AVR1/") for name in ps.state_desc[nonzero, 0])


def test_connected_input_only_affects_its_unit(lin, ps):
    # P_m of G1 is connected to the governor; perturbing it moves only the speed of G1
    nonzero = np.flatnonzero(abs(lin.b[:, 1]) > 1e-8)
    assert nonzero.tolist() == [ps.gen["GEN"].state_idx_global["speed"][0]]


def test_c_and_d(lin, ps):
    speed_idx = ps.gen["GEN"].state_idx_global["speed"]
    expected = np.zeros(ps.n_states)
    expected[speed_idx[2]] = 1
    np.testing.assert_allclose(lin.c[1], expected, atol=1e-8)
    # The input v_setp, used as an output, passes straight through
    np.testing.assert_allclose(lin.d[2], [1, 0, 0, 0, 0, 0], atol=1e-8)
    # Generator speed has no direct feedthrough
    np.testing.assert_allclose(lin.d[:2], 0, atol=1e-8)


def test_residues_give_transfer_function(lin):
    s = 2j * np.pi * 0.6
    tf = lin.c @ np.linalg.solve(s * np.eye(lin.n) - lin.a, lin.b) + lin.d
    tf_res = sum(lin.residues(i) / (s - lin.eigs[i]) for i in range(lin.n)) + lin.d
    np.testing.assert_allclose(tf_res, tf, atol=1e-9)


def test_participation_factors_sum_to_one(lin):
    np.testing.assert_allclose(lin.participation_factors().sum(axis=0), 1, atol=1e-8)


def test_model_is_left_unchanged(ps):
    x0 = ps.x0.copy()
    f0 = ps.ode_fun(0, x0)
    lin = dps_mdl.PowerSystemModelLinearization(ps)
    lin.linearize(inputs=[ps.gen.GEN.P_m, ps.avr.SEXS.v_setp], outputs=[ps.gen.GEN.P_m])
    np.testing.assert_array_equal(ps.x0, x0)
    np.testing.assert_array_equal(ps.ode_fun(0, x0), f0)
    assert "P_m" in ps.gen["GEN"].__dict__  # The connection is restored, not removed


def test_outputs_without_inputs(ps):
    lin = dps_mdl.PowerSystemModelLinearization(ps)
    a, b, c, d = lin.linearize(outputs=[ps.gen.GEN.speed])
    assert b.shape == (ps.n_states, 0)
    assert c.shape == (4, ps.n_states)
    assert d.shape == (4, 0)


def test_eigenvalues_without_explicit_linearize(ps):
    lin = dps_mdl.PowerSystemModelLinearization(ps)
    assert len(lin.get_mode_idx(["em", "non_conj"], damp_threshold=0.3)) == 3


def test_relinearizing_resets_eigenvalues(ps):
    lin = dps_mdl.PowerSystemModelLinearization(ps)
    lin.linearize(get_eigs=True)
    lin.linearize()
    assert not lin.eigenvalues_ready


def test_undamped_mode_has_zero_damping(ps):
    lin = dps_mdl.PowerSystemModelLinearization(ps)
    lin.linearization_ready = True
    lin.n = 2
    lin.a = np.array([[0.0, 1.0], [-4.0, 0.0]])
    lin.eigenvalue_decomposition()
    np.testing.assert_allclose(lin.damping, 0)


def test_block_input(ps):
    lin = dps_mdl.PowerSystemModelLinearization(ps)
    lin.linearize(inputs=[(ps.avr.SEXS.tg_red.input, "AVR1/tg_red")])
    assert lin.input_desc == ["AVR1/tg_red.input"]
    nonzero = np.flatnonzero(abs(lin.b[:, 0]) > 1e-8)
    assert all(name.startswith("AVR1/") for name in ps.state_desc[nonzero, 0])


def test_function_outputs(ps):
    lin = dps_mdl.PowerSystemModelLinearization(ps)
    speed = ps.gen.GEN.speed
    lin.linearize(
        outputs=[
            lambda x, v: speed(x, v)[0] - speed(x, v)[2],  # Scalar
            (lambda x, v: abs(v), [0, 3]),
        ]
    )
    assert lin.output_desc == ["<lambda>[0]", "<lambda>[0]", "<lambda>[3]"]
    speed_idx = ps.gen.GEN.state_idx_global["speed"]
    expected = np.zeros(ps.n_states)
    expected[speed_idx[[0, 2]]] = [1, -1]
    np.testing.assert_allclose(lin.c[0], expected, atol=1e-8)
    assert lin.c.dtype == float


def test_input_taken_before_init_is_rejected():
    ps = dps.PowerSystemModel(model=k2a.load())
    v_setp = ps.avr.SEXS.v_setp  # Replaced by init_dyn_sim, since it is connected
    ps.init_dyn_sim()
    lin = dps_mdl.PowerSystemModelLinearization(ps)
    with pytest.raises(ValueError, match="init_dyn_sim"):
        lin.linearize(inputs=[v_setp])


@pytest.mark.parametrize(
    "get_spec, error",
    [
        (lambda ps: ps.gen.GEN.speed, ValueError),  # Not an input
        (lambda ps: (ps.avr.SEXS.v_setp, "G1"), KeyError),  # Unit names are those of the AVRs
        (lambda ps: (ps.avr.SEXS.v_setp, 4), IndexError),
        (lambda ps: ("avr", "SEXS", "v_setp"), TypeError),
    ],
)
def test_invalid_inputs(ps, get_spec, error):
    lin = dps_mdl.PowerSystemModelLinearization(ps)
    with pytest.raises(error):
        lin.linearize(inputs=[get_spec(ps)])


def test_unit_names_need_a_model(ps):
    lin = dps_mdl.PowerSystemModelLinearization(ps)
    with pytest.raises(TypeError):
        lin.linearize(outputs=[(lambda x, v: abs(v), "B1")])
