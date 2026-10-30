"""The set_power event of the constant-impedance Load."""
import numpy as np
import pytest

from k2a_variants import initialized, k2a_model, y_bus_red
from tops.events import AdmittanceChange
from tops.simulator import Simulator

LOAD = 'L1'
EVENT_COLUMNS = ['t', 'category', 'model', 'unit', 'action', 'kwargs']


def load_index(ps, name=LOAD):
    return np.where(ps.loads['Load'].par['name'] == name)[0][0]


def admittance(ps, p, q, i):
    """Admittance of a load of p MW and q Mvar at the voltage of the load flow: conj(S) / |v|^2."""
    v = ps.v_0[ps.loads['Load'].bus_idx['terminal'][i]]
    return np.conj((p + 1j*q)/ps.s_n)/abs(v)**2


def test_loads_have_a_set_power_event():
    ps = initialized(k2a_model())
    assert ps.get_events('loads', 'Load').actions == ('set_power',)


def test_set_power_returns_the_change_of_the_admittance_at_the_voltage_of_the_load_flow():
    ps = initialized(k2a_model())
    load = ps.loads['Load']
    i = load_index(ps)
    p, q = load.par['P'][i], load.par['Q'][i]
    before = y_bus_red(ps)

    effect, = load.events.set_power(LOAD, P=1000.5, Q=120.25)
    assert isinstance(effect, AdmittanceChange)
    bus = load.bus_idx_red['terminal'][i]
    assert list(effect.rows) == [bus] and list(effect.cols) == [bus]
    assert np.allclose(effect.data, admittance(ps, 1000.5, 120.25, i) - admittance(ps, p, q, i))
    assert np.array_equal(y_bus_red(ps), before)


def test_apply_event_changes_the_diagonal_of_the_load_bus_and_nothing_else():
    ps = initialized(k2a_model())
    i = load_index(ps)
    before = y_bus_red(ps)
    expected = admittance(ps, 1000.5, 120.25, i) - ps.loads['Load'].y_load[i]

    ps.apply_event('loads', 'Load', LOAD, 'set_power', P=1000.5, Q=120.25)

    bus = ps.loads['Load'].bus_idx_red['terminal'][i]
    difference = y_bus_red(ps) - before
    assert np.isclose(difference[bus, bus], expected)
    difference[bus, bus] = 0
    assert np.array_equal(difference, np.zeros_like(difference))


def test_the_power_is_kept_without_truncation_although_the_columns_are_integers():
    ps = initialized(k2a_model())
    load = ps.loads['Load']
    i = load_index(ps)
    assert load.par['P'].dtype.kind == 'i'

    ps.apply_event('loads', 'Load', LOAD, 'set_power', P=1000.5, Q=120.25)

    assert load.par['P'][i] == 1000.5 and load.par['Q'][i] == 120.25
    assert load.par['P'][1 - i] == 1767 and load.par['name'][i] == LOAD  # the other load is as it was
    assert load.load_flow_pq()[1][i] == 1000.5


def test_none_keeps_the_other_power():
    ps = initialized(k2a_model())
    load = ps.loads['Load']
    i = load_index(ps)

    ps.apply_event('loads', 'Load', LOAD, 'set_power', P=1000.0)
    assert load.par['P'][i] == 1000.0 and load.par['Q'][i] == 100
    ps.apply_event('loads', 'Load', LOAD, 'set_power', Q=50.0)
    assert load.par['P'][i] == 1000.0 and load.par['Q'][i] == 50.0


def test_repeating_a_power_changes_nothing_and_the_admittance_never_drifts():
    ps = initialized(k2a_model())
    load = ps.loads['Load']
    i = load_index(ps)

    assert load.events.set_power(LOAD, P=967, Q=100) == []
    for p in (1000.0, 900.0, 1100.0, 1000.0):
        ps.apply_event('loads', 'Load', LOAD, 'set_power', P=p)
    before = ps.y_bus_red.copy()
    assert load.events.set_power(LOAD, P=1000.0) == []

    assert np.isclose(load.y_load[i], admittance(ps, 1000.0, 100, i), rtol=1e-12, atol=0)
    assert (ps.y_bus_red != before).nnz == 0


def test_a_fresh_initialisation_starts_from_the_new_power():
    ps = initialized(k2a_model())
    i = load_index(ps)
    ps.apply_event('loads', 'Load', LOAD, 'set_power', P=1000.5)
    ps.init_dyn_sim()
    assert np.isclose(ps.loads['Load'].y_load[i], admittance(ps, 1000.5, 100, i), rtol=1e-12, atol=0)


@pytest.mark.parametrize('column, value', [('P', float('nan')), ('P', float('inf')), ('Q', 'a lot')])
def test_a_power_that_is_not_a_finite_number_is_refused_naming_it_and_nothing_is_changed(column, value):
    ps = initialized(k2a_model())
    load = ps.loads['Load']
    i = load_index(ps)
    other = 'Q' if column == 'P' else 'P'
    before = ps.y_bus_red.copy(), load.par[column][i], load.par[other][i], load.y_load[i]

    with pytest.raises(ValueError, match=column):
        ps.apply_event('loads', 'Load', LOAD, 'set_power', **{column: value, other: 1000.0})

    assert (ps.y_bus_red != before[0]).nnz == 0
    assert (load.par[column][i], load.par[other][i], load.y_load[i]) == before[1:]


def test_unknown_load_or_argument_raises():
    ps = initialized(k2a_model())
    with pytest.raises(KeyError):
        ps.apply_event('loads', 'Load', 'NoSuchLoad', 'set_power', P=1.0)
    with pytest.raises(TypeError, match='volts'):
        ps.apply_event('loads', 'Load', LOAD, 'set_power', volts=1.0)


def test_a_load_outside_the_reduced_system_raises_and_keeps_its_power():
    ps = initialized(k2a_model())
    load = ps.loads['Load']
    i = load_index(ps)
    load.bus_idx_red['terminal'][i] = -99999  # how setup marks a bus that Kron reduction removed
    with pytest.raises(ValueError, match='not in the reduced system'):
        load.events.set_power(LOAD, P=1000.5)
    assert load.par['P'][i] == 967


def test_the_schedule_can_change_a_power():
    ps = initialized(k2a_model())
    i = load_index(ps)
    events = [EVENT_COLUMNS, [0.1, 'loads', 'Load', LOAD, 'set_power', {'P': 1000.5}]]
    Simulator(ps, dt=5e-3, t_end=0.2, events=events).main_loop()
    assert ps.loads['Load'].par['P'][i] == 1000.5


def test_the_admittance_is_what_a_rebuild_stamps_at_the_same_operating_voltage():
    ps = initialized(k2a_model())
    load = ps.loads['Load']
    ps.apply_event('loads', 'Load', LOAD, 'set_power', P=1000.5, Q=120.25)
    after_event = load.y_load.copy()

    load.init_from_load_flow(ps.x0, ps.v_0, None)  # what initialisation does with the new P and Q, at the same voltage

    assert np.allclose(load.y_load, after_event, rtol=1e-12, atol=0)


def test_the_load_takes_the_new_power_at_the_operating_point():
    ps = initialized(k2a_model())
    load = ps.loads['Load']
    i = load_index(ps)
    assert np.isclose(load.P(ps.x0, ps.v_0)[i], 967) and np.isclose(load.Q(ps.x0, ps.v_0)[i], 100)

    ps.apply_event('loads', 'Load', LOAD, 'set_power', P=1000.5, Q=120.25)

    assert np.isclose(load.P(ps.x0, ps.v_0)[i], 1000.5, rtol=1e-12, atol=0)
    assert np.isclose(load.Q(ps.x0, ps.v_0)[i], 120.25, rtol=1e-12, atol=0)
    assert np.isclose(load.P(ps.x0, ps.v_0)[1 - i], 1767)


# A rebuilt model re-initialises the setpoints of the governors and the AVRs; a load changed live does not, and the
# droop of the governors leaves the frequency and the voltages offset. So the grid does not settle where a rebuilt
# model starts. What holds is that the load, at the voltage it settles at, draws about the power it was given.
@pytest.mark.slow
@pytest.mark.parametrize('p_new', [1017.0, 867.0])
def test_a_load_changed_live_settles_nearer_the_new_power_than_the_old_one(p_new):
    ps = initialized(k2a_model())
    i = load_index(ps)
    events = [EVENT_COLUMNS, [0.1, 'loads', 'Load', LOAD, 'set_power', {'P': p_new}]]
    sim = Simulator(ps, dt=5e-3, t_end=20, events=events)
    sim.main_loop()

    p_settled = ps.loads['Load'].P(sim.sol.x, sim.sol.v)[i]
    assert abs(p_settled - p_new) < abs(p_settled - 967)
    assert abs(p_settled - p_new) < 0.1*abs(p_new - 967) + 0.05*p_new  # within the effect of the voltage offset
