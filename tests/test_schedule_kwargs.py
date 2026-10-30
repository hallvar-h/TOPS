"""The optional 'kwargs' column of an event table: arguments for the action of the event."""
import pandas as pd
import pytest

import tops.dynamic as dps
from tops.simulator import EventSchedule, Simulator

COLUMNS = ['t', 'category', 'model', 'unit', 'action']
BUS = 'B8'


def k2a_ps():
    import tops.ps_models.k2a as model_data
    ps = dps.PowerSystemModel(model=model_data.load())
    ps.init_dyn_sim()
    return ps


def y_fault(ps, bus=BUS):
    return ps.bus_events.y_fault[list(ps.buses['name']).index(bus)]


def test_kwargs_reach_the_action_of_the_event():
    ps = k2a_ps()
    schedule = EventSchedule([[*COLUMNS, 'kwargs'], [0.1, 'buses', '', BUS, 'short_circuit', {'y': 100}]])
    schedule.apply_due(ps, 0.1)
    assert y_fault(ps) == 100


def test_a_row_without_kwargs_uses_the_defaults_of_the_action():
    ps = k2a_ps()
    schedule = EventSchedule([[*COLUMNS, 'kwargs'],
                              [0.1, 'buses', '', BUS, 'short_circuit', None],
                              [0.1, 'buses', '', 'B9', 'short_circuit', {'y': 100}]])
    schedule.apply_due(ps, 0.1)
    assert y_fault(ps) == 1e6
    assert y_fault(ps, 'B9') == 100


def test_without_the_column_nothing_is_passed():
    schedule = EventSchedule([COLUMNS, [0.1, 'buses', '', BUS, 'short_circuit']])
    assert schedule.kwargs == [{}]
    ps = k2a_ps()
    schedule.apply_due(ps, 0.1)
    assert y_fault(ps) == 1e6


def test_kwargs_stay_with_their_event_when_the_table_is_sorted():
    schedule = EventSchedule([[*COLUMNS, 'kwargs'],
                              [0.2, 'buses', '', 'B9', 'short_circuit', {'y': 9}],
                              [0.1, 'buses', '', BUS, 'short_circuit', {'y': 8}]])
    assert [event[3] for event in schedule.events] == [BUS, 'B9']
    assert schedule.kwargs == [{'y': 8}, {'y': 9}]


def test_a_data_frame_with_a_kwargs_column_is_accepted():
    frame = pd.DataFrame([[0.1, 'buses', '', BUS, 'short_circuit', {'y': 100}]], columns=[*COLUMNS, 'kwargs'])
    ps = k2a_ps()
    EventSchedule(frame).apply_due(ps, 0.1)
    assert y_fault(ps) == 100


def test_pop_due_still_returns_the_five_fields():
    schedule = EventSchedule([[*COLUMNS, 'kwargs'], [0.1, 'buses', '', BUS, 'short_circuit', {'y': 100}]])
    assert schedule.pop_due(0.1) == [(0.1, 'buses', '', BUS, 'short_circuit')]


@pytest.mark.parametrize('bad', ['y=100', 5])
def test_kwargs_that_are_not_a_dict_are_refused(bad):
    with pytest.raises(ValueError, match='kwargs'):
        EventSchedule([[*COLUMNS, 'kwargs'], [0.1, 'buses', '', BUS, 'short_circuit', bad]])


def test_arguments_the_action_does_not_take_fail_when_it_is_applied():
    ps = k2a_ps()
    schedule = EventSchedule([[*COLUMNS, 'kwargs'], [0.1, 'buses', '', BUS, 'short_circuit', {'volts': 1}]])
    schedule.check(ps)
    with pytest.raises(TypeError, match='volts'):
        schedule.apply_due(ps, 0.1)


def test_the_simulator_applies_the_kwargs_of_its_events():
    ps = k2a_ps()
    events = [[*COLUMNS, 'kwargs'],
              [0.1, 'buses', '', BUS, 'short_circuit', {'y': 100}],
              [0.2, 'buses', '', 'B9', 'short_circuit', {'y': 50}]]
    Simulator(ps, dt=5e-3, t_end=0.25, events=events).main_loop()
    assert y_fault(ps) == 100
    assert y_fault(ps, 'B9') == 50
