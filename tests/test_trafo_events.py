"""Connect and disconnect events of Trafo and DynTrafo."""
import numpy as np
import pytest

from k2a_variants import KINDS, TRAFO, initialized, trafo_index, trafo_model, y_bus_red
from tops.events import AdmittanceChange, ModelEvents
from tops.simulator import EventSchedule, Simulator

EVENT_COLUMNS = ['t', 'category', 'model', 'unit', 'action']
RATIOS = (0.95, 1.02)


def stamp(ps, kind):
    """What the load flow stamps for TRAFO, as a matrix over the reduced system (no Kron reduction here)."""
    trafo = ps.trafos[kind]
    i = trafo_index(ps, kind)
    data, (rows, cols) = trafo.load_flow_adm()
    matrix = np.zeros((ps.n_bus_red,) * 2, dtype=complex)
    np.add.at(matrix, (rows[:, i], cols[:, i]), data[:, i])
    return matrix


def trip(ps, kind):
    ps.apply_event('trafos', kind, TRAFO, 'disconnect')


def simulate(ps, events=None, t_end=0.5):
    sim = Simulator(ps, dt=5e-3, t_end=t_end, events=events)
    speed = []
    sim.interface_functions['speed'] = lambda sim: speed.append(ps.gen['GEN'].speed(sim.sol.x, sim.sol.v).copy())
    sim.main_loop()
    return sim, np.array(speed)


def test_transformers_have_connect_and_disconnect_events():
    for kind in KINDS:
        ps = initialized(trafo_model(kind))
        events = ps.get_events('trafos', kind)
        assert isinstance(events, ModelEvents)
        assert events.actions == ('connect', 'disconnect')


def test_trafo_events_return_an_admittance_change_and_leave_the_matrix_to_apply_effect():
    ps = initialized(trafo_model('Trafo'))
    trafo = ps.trafos['Trafo']
    before = ps.y_bus_red.copy()

    effects = trafo.events.disconnect(TRAFO)
    assert len(effects) == 1 and isinstance(effects[0], AdmittanceChange)
    assert not trafo.connected[trafo_index(ps, 'Trafo')]
    assert (ps.y_bus_red != before).nnz == 0

    # Disconnecting again has no effect, connecting reverses the change
    assert trafo.events.disconnect(TRAFO) == []
    reconnect, = trafo.events.connect(TRAFO)
    assert np.allclose(reconnect.data, -effects[0].data)
    assert trafo.events.connect(TRAFO) == []


@pytest.mark.parametrize('ratios', [(1, 1), RATIOS], ids=['unit ratios', 'tap ratios'])
def test_disconnect_takes_out_what_the_load_flow_stamps(ratios):
    ps = initialized(trafo_model('Trafo', ratios=ratios))
    before = y_bus_red(ps)
    expected = stamp(ps, 'Trafo')
    assert np.abs(expected).max() > 1

    trip(ps, 'Trafo')
    assert np.allclose(before - y_bus_red(ps), expected, atol=1e-10)


def test_a_repeated_event_changes_nothing():
    ps = initialized(trafo_model('Trafo'))
    trip(ps, 'Trafo')
    tripped = ps.y_bus_red.copy()
    trip(ps, 'Trafo')
    assert (ps.y_bus_red != tripped).nnz == 0

    ps.apply_event('trafos', 'Trafo', TRAFO, 'connect')
    connected = ps.y_bus_red.copy()
    ps.apply_event('trafos', 'Trafo', TRAFO, 'connect')
    assert (ps.y_bus_red != connected).nnz == 0


def test_connect_stamps_the_full_admittance_of_a_trafo_built_out_of_service():
    ps = initialized(trafo_model('Trafo', {TRAFO: False}, ratios=RATIOS))
    before = y_bus_red(ps)
    ps.apply_event('trafos', 'Trafo', TRAFO, 'connect')

    assert ps.trafos['Trafo'].connected.all()
    assert np.allclose(y_bus_red(ps) - before, stamp(ps, 'Trafo'), atol=1e-10)


def test_dyn_trafo_events_flip_the_flag_and_change_no_matrix():
    ps = initialized(trafo_model('DynTrafo'))
    trafo = ps.trafos['DynTrafo']
    before = ps.y_bus_red.copy()
    v_before = ps.solve_algebraic(0, ps.x0)

    assert trafo.events.disconnect(TRAFO) == []
    assert not trafo.connected[trafo_index(ps, 'DynTrafo')]
    assert trafo.events.disconnect(TRAFO) == []
    assert (ps.y_bus_red != before).nnz == 0
    assert not np.allclose(ps.solve_algebraic(0, ps.x0), v_before)

    assert trafo.events.connect(TRAFO) == []
    assert trafo.connected.all()
    assert np.allclose(ps.solve_algebraic(0, ps.x0), v_before)


def test_a_tripped_dyn_trafo_gives_the_network_of_a_tripped_trafo():
    trafo, dyn_trafo = (initialized(trafo_model(kind)) for kind in KINDS)
    trip(trafo, 'Trafo')
    trip(dyn_trafo, 'DynTrafo')
    assert np.allclose(trafo.solve_algebraic(0, trafo.x0), dyn_trafo.solve_algebraic(0, dyn_trafo.x0))


def test_a_dyn_trafo_comes_back_with_its_ratios_of_the_moment():
    ps = initialized(trafo_model('DynTrafo'))
    trafo = ps.trafos['DynTrafo']
    trafo.set_input('ratio_from', 0.95, trafo_index(ps, 'DynTrafo'))
    v_before = ps.solve_algebraic(0, ps.x0)
    assert not np.allclose(v_before, initialized(trafo_model('DynTrafo')).solve_algebraic(0, ps.x0))

    trip(ps, 'DynTrafo')
    ps.apply_event('trafos', 'DynTrafo', TRAFO, 'connect')
    assert np.allclose(ps.solve_algebraic(0, ps.x0), v_before)


def test_an_event_on_a_trafo_outside_the_reduced_system_raises_and_keeps_its_state():
    ps = initialized(trafo_model('Trafo'))
    trafo = ps.trafos['Trafo']
    i = trafo_index(ps, 'Trafo')
    trafo.bus_idx_red['from_bus'][i] = -99999  # how setup marks a bus that Kron reduction removed
    with pytest.raises(ValueError, match='not in the reduced system'):
        trafo.events.disconnect(TRAFO)
    assert trafo.connected[i]


@pytest.mark.parametrize('kind', KINDS)
def test_invalid_action_or_unit_raises(kind):
    ps = initialized(trafo_model(kind))
    with pytest.raises(ValueError, match='no event'):
        ps.trafos[kind].events.apply(TRAFO, 'explode')
    with pytest.raises(KeyError):
        ps.trafos[kind].events.apply('NoSuchTrafo', 'disconnect')


@pytest.mark.parametrize('kind', KINDS)
def test_the_schedule_accepts_trafo_events_and_checks_them(kind):
    ps = initialized(trafo_model(kind))
    EventSchedule([EVENT_COLUMNS, [0.1, 'trafos', kind, TRAFO, 'disconnect']]).check(ps)
    with pytest.raises(KeyError, match='NoSuchTrafo'):
        EventSchedule([EVENT_COLUMNS, [0.1, 'trafos', kind, 'NoSuchTrafo', 'disconnect']]).check(ps)


@pytest.mark.parametrize('kind', KINDS)
def test_a_trip_and_reconnect_at_the_same_instant_stay_at_the_equilibrium(kind):
    events = [EVENT_COLUMNS, [0.1, 'trafos', kind, TRAFO, 'disconnect'], [0.1, 'trafos', kind, TRAFO, 'connect']]
    ps = initialized(trafo_model(kind))
    x0 = ps.x0.copy()
    sim, _ = simulate(ps, events)
    assert np.allclose(sim.sol.x, x0, atol=1e-8)
    assert max(abs(ps.ode_fun(0, sim.sol.x))) < 1e-8


@pytest.mark.parametrize('kind', KINDS)
def test_a_trip_disturbs_the_system_and_the_reconnect_restores_the_network(kind):
    events = [EVENT_COLUMNS, [0.1, 'trafos', kind, TRAFO, 'disconnect'], [0.3, 'trafos', kind, TRAFO, 'connect']]
    ps = initialized(trafo_model(kind))
    before = y_bus_red(ps)
    _, speed = simulate(ps, events)

    assert np.ptp(speed) > 1e-6
    assert ps.trafos[kind].connected.all()
    assert np.allclose(y_bus_red(ps), before, atol=1e-10)


@pytest.mark.slow
@pytest.mark.parametrize('kind', KINDS)
def test_a_tripped_and_reconnected_trafo_settles_back_to_the_equilibrium(kind):
    events = [EVENT_COLUMNS, [0.1, 'trafos', kind, TRAFO, 'disconnect'], [0.3, 'trafos', kind, TRAFO, 'connect']]
    ps = initialized(trafo_model(kind))
    x0 = ps.x0.copy()
    sim, speed = simulate(ps, events, t_end=20)

    assert np.ptp(speed[:100]) > 1e-6
    # The machines have no angle reference and the governors leave a small speed offset, so all angles drift by the
    # same amount. What returns is the state of rest (derivatives -> 0) and the angles relative to each other.
    assert max(abs(ps.ode_fun(0, sim.sol.x))) < 2e-4
    angles = ps.state_desc[:, 1] == 'angle'
    shift = (sim.sol.x - x0)[angles]
    assert np.ptp(shift) < 1e-4
