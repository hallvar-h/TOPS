import numpy as np
import pytest

import tops.dynamic as dps
from tops.events import AdmittanceChange, BusEvents, EventSchedule
from tops.model_data import read_table
from tops.simulator import Simulator, Events


def k2a_model(events=None):
    import tops.ps_models.k2a as model_data
    model = model_data.load()
    if events is not None:
        model['events'] = events
    return model


def simulate(ps, t_end=0.5, events=None):
    sim = Simulator(ps, dt=5e-3, t_end=t_end)
    speed = []
    if events is not None:
        sim.interface_functions['Events'] = Events(sim, events).update
    sim.interface_functions['speed'] = lambda sim: speed.append(ps.gen['GEN'].speed(sim.sol.x, sim.sol.v).copy())
    sim.main_loop()
    return np.array(speed)


def test_line_events_only_modify_line_and_return_admittance_change():
    ps = dps.PowerSystemModel(model=k2a_model())
    ps.init_dyn_sim()
    line = ps.lines['Line']
    y_bus_red = ps.y_bus_red.copy()

    effects = line.events.disconnect('L7-8-1')
    assert not line.connected[line.par['name'] == 'L7-8-1'][0]
    assert len(effects) == 1 and isinstance(effects[0], AdmittanceChange)
    assert (ps.y_bus_red != y_bus_red).nnz == 0

    # Disconnecting again has no effect, connecting reverses the change
    assert line.events.disconnect('L7-8-1') == []
    reconnect = line.events.connect('L7-8-1')[0]
    assert np.allclose(reconnect.data, -effects[0].data)


def test_invalid_action_or_unit_raises():
    ps = dps.PowerSystemModel(model=k2a_model())
    ps.init_dyn_sim()
    with pytest.raises(ValueError, match='no event'):
        ps.lines['Line'].events.apply('L7-8-1', 'explode')
    with pytest.raises(KeyError):
        ps.lines['Line'].events.apply('NoSuchLine', 'disconnect')


def test_bus_events_use_reduced_indices():
    bus_events = BusEvents(np.array(['A', 'B', 'C']), bus_idx_red=np.array([0, 2]))
    effect, = bus_events.short_circuit('C', y=10)
    assert effect.rows == [1] and effect.cols == [1] and effect.data == [10]
    with pytest.raises(ValueError, match='not in the reduced system'):
        bus_events.short_circuit('B')


def test_schedule_is_sorted_and_returns_simultaneous_events_together():
    schedule = EventSchedule(read_table([
        ['t', 'category', 'model', 'unit', 'action'],
        [0.2, 'lines', 'Line', 'L1', 'connect'],
        [0.1, 'lines', 'Line', 'L1', 'disconnect'],
        [0.1, 'lines', 'Line', 'L2', 'disconnect'],
    ]))
    assert schedule.pop_due(0.05) == []
    assert [event[3] for event in schedule.pop_due(0.15)] == ['L1', 'L2']
    assert [event[4] for event in schedule.pop_due(1)] == ['connect']


def test_events_in_model_definition():
    events = [
        ['t', 'category', 'model', 'unit', 'action'],
        [0.1, 'buses', '', 'B8', 'short_circuit'],
        [0.15, 'buses', '', 'B8', 'clear_short_circuit'],
        [0.15, 'lines', 'Line', 'L7-8-1', 'disconnect'],
        [0.3, 'lines', 'Line', 'L7-8-1', 'connect'],
    ]
    ps = dps.PowerSystemModel(model=k2a_model(events))
    ps.init_dyn_sim()
    y_bus_red = ps.y_bus_red.copy()
    speed = simulate(ps)

    assert np.ptp(speed) > 0
    assert ps.lines['Line'].connected.all()
    assert np.allclose((ps.y_bus_red - y_bus_red).toarray(), 0)


def test_events_in_model_definition_match_events_class():
    line_events = [(0.1, ('line', 'L7-8-1', 'disconnect')), (0.3, ('line', 'L7-8-1', 'connect'))]
    table = [['t', 'category', 'model', 'unit', 'action']] + [[t, 'lines', 'Line', name, action] for t, (_, name, action) in line_events]

    ps_events_class = dps.PowerSystemModel(model=k2a_model())
    ps_events_class.init_dyn_sim()
    ps_model_events = dps.PowerSystemModel(model=k2a_model(table))
    ps_model_events.init_dyn_sim()

    assert np.allclose(simulate(ps_events_class, events=line_events), simulate(ps_model_events))


def test_invalid_event_in_model_definition_raises_at_init():
    events = [['t', 'category', 'model', 'unit', 'action'], [0.1, 'lines', 'Line', 'NoSuchLine', 'disconnect']]
    ps = dps.PowerSystemModel(model=k2a_model(events))
    with pytest.raises(KeyError, match='NoSuchLine'):
        ps.init_dyn_sim()
