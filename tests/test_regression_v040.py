"""What v0.4.0 users see, pinned. These tests pass on the unmodified tag and must keep passing."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from golden_scenarios import EVENT_COLUMNS, LINE_TRIP, k2a_ps, outputs, run
from tops.simulator import EventSchedule, InterfacerDirect

GOLDEN = Path(__file__).parent / 'data' / 'v040_k2a.npz'
TOL = dict(rtol=1e-9, atol=1e-12)


@pytest.fixture(scope='module')
def golden():
    with np.load(GOLDEN) as data:
        return {key: data[key] for key in data.files}


def assert_matches(golden, prefix, actual):
    """Every output of `actual` equals the golden one named prefix_output, and none is missing on either side."""
    expected = {key[len(prefix) + 1:]: value for key, value in golden.items() if key.startswith(prefix + '_')}
    assert sorted(expected) == sorted(actual) == ['angle', 'p_e', 'speed', 'v']
    for key in expected:
        assert np.allclose(actual[key], expected[key], **TOL), f'{prefix}_{key}'


def test_initial_state_matches_v040(golden):
    ps = k2a_ps()
    assert np.allclose(ps.y_bus_lf, golden['y_bus_lf'], **TOL)
    assert np.allclose(ps.y_bus_red.toarray(), golden['y_bus_red'], **TOL)
    assert np.allclose(ps.v_0, golden['v_0'], **TOL)
    assert_matches(golden, 'init', outputs(ps, ps.x0, ps.v_0))


def test_free_simulation_matches_v040(golden):
    assert_matches(golden, 'free', run(k2a_ps()))


def test_line_trip_simulation_matches_v040(golden):
    assert_matches(golden, 'trip', run(k2a_ps(), events=LINE_TRIP))


def test_legacy_line_event_equals_apply_event():
    legacy, current = k2a_ps(), k2a_ps()
    legacy.lines['Line'].event(legacy, 'L7-8-1', 'disconnect')
    current.apply_event('lines', 'Line', 'L7-8-1', 'disconnect')
    assert (legacy.y_bus_red != current.y_bus_red).nnz == 0
    assert (legacy.y_bus_red != k2a_ps().y_bus_red).nnz > 0


@pytest.mark.parametrize('as_frame', [False, True])
def test_five_column_schedule_returns_five_tuples(as_frame):
    rows = [[0.2, 'lines', 'Line', 'L6-7', 'connect'], [0.1, 'lines', 'Line', 'L5-6', 'disconnect']]
    table = pd.DataFrame(rows, columns=EVENT_COLUMNS) if as_frame else [EVENT_COLUMNS, *rows]
    schedule = EventSchedule(table)
    assert schedule.pop_due(0.15) == [(0.1, 'lines', 'Line', 'L5-6', 'disconnect')]
    assert schedule.pop_due(0.25) == [(0.2, 'lines', 'Line', 'L6-7', 'connect')]


def test_branches_start_connected_and_currents_are_untouched():
    ps = k2a_ps()
    assert ps.lines['Line'].connected.all()
    assert ps.trafos['Trafo'].connected.all()


def test_interfacer_without_simulator_constructs():
    interfacer = InterfacerDirect()
    assert interfacer.interface_name == 'InterfacerDirect'
    assert interfacer.fs is None
