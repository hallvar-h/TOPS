import numpy as np
import pandas as pd
import pytest

import tops.dynamic as dps
from tops.events import AdmittanceChange, BusEvents
from tops.simulator import Simulator, Events, EventSchedule

EVENT_COLUMNS = ['t', 'category', 'model', 'unit', 'action']


def k2a_model():
    import tops.ps_models.k2a as model_data
    return model_data.load()


def simulate(ps, t_end=0.5, events=None, legacy_events=None):
    sim = Simulator(ps, dt=5e-3, t_end=t_end, events=events)
    speed = []
    if legacy_events is not None:
        sim.interface_functions['Events'] = Events(sim, legacy_events).update
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
    schedule = EventSchedule([
        EVENT_COLUMNS,
        [0.2, 'lines', 'Line', 'L1', 'connect'],
        [0.1, 'lines', 'Line', 'L1', 'disconnect'],
        [0.1, 'lines', 'Line', 'L2', 'disconnect'],
    ])
    assert schedule.pop_due(0.05) == []
    assert [event[3] for event in schedule.pop_due(0.15)] == ['L1', 'L2']
    assert [event[4] for event in schedule.pop_due(1)] == ['connect']


def test_events_in_simulator():
    events = [
        EVENT_COLUMNS,
        [0.1, 'buses', '', 'B8', 'short_circuit'],
        [0.15, 'buses', '', 'B8', 'clear_short_circuit'],
        [0.15, 'lines', 'Line', 'L7-8-1', 'disconnect'],
        [0.3, 'lines', 'Line', 'L7-8-1', 'connect'],
    ]
    ps = dps.PowerSystemModel(model=k2a_model())
    ps.init_dyn_sim()
    y_bus_red = ps.y_bus_red.copy()
    speed = simulate(ps, events=events)

    assert np.ptp(speed) > 0
    assert ps.lines['Line'].connected.all()
    assert np.allclose((ps.y_bus_red - y_bus_red).toarray(), 0)


def test_same_system_simulated_with_different_events():
    ps = dps.PowerSystemModel(model=k2a_model())
    speeds = {}
    for line in ['L7-8-1', 'L8-9-1']:
        ps.init_dyn_sim()
        speeds[line] = simulate(ps, events=[EVENT_COLUMNS, [0.1, 'lines', 'Line', line, 'disconnect']])
        ps.apply_event('lines', 'Line', line, 'connect')
    assert not np.allclose(speeds['L7-8-1'], speeds['L8-9-1'])


def test_events_as_dataframe_match_list():
    rows = [[0.1, 'lines', 'Line', 'L7-8-1', 'disconnect'], [0.3, 'lines', 'Line', 'L7-8-1', 'connect']]
    results = []
    for events in [[EVENT_COLUMNS] + rows, pd.DataFrame(rows, columns=EVENT_COLUMNS)]:
        ps = dps.PowerSystemModel(model=k2a_model())
        ps.init_dyn_sim()
        results.append(simulate(ps, events=events))
    assert np.allclose(*results)


def test_events_in_simulator_match_events_class():
    line_events = [(0.1, ('line', 'L7-8-1', 'disconnect')), (0.3, ('line', 'L7-8-1', 'connect'))]
    table = [EVENT_COLUMNS] + [[t, 'lines', 'Line', name, action] for t, (_, name, action) in line_events]

    ps_events_class = dps.PowerSystemModel(model=k2a_model())
    ps_events_class.init_dyn_sim()
    ps_simulator_events = dps.PowerSystemModel(model=k2a_model())
    ps_simulator_events.init_dyn_sim()

    assert np.allclose(simulate(ps_events_class, legacy_events=line_events),
                       simulate(ps_simulator_events, events=table))


def test_invalid_event_raises_when_simulator_is_created():
    ps = dps.PowerSystemModel(model=k2a_model())
    ps.init_dyn_sim()
    with pytest.raises(KeyError, match='NoSuchLine'):
        Simulator(ps, events=[EVENT_COLUMNS, [0.1, 'lines', 'Line', 'NoSuchLine', 'disconnect']])


def test_events_in_model_data_raise():
    model = k2a_model()
    model['events'] = [EVENT_COLUMNS, [0.1, 'lines', 'Line', 'L7-8-1', 'disconnect']]
    with pytest.raises(ValueError, match='Simulator'):
        dps.PowerSystemModel(model=model)
