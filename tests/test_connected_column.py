"""The optional 'connected' column of the line and transformer tables."""
import numpy as np
import pandas as pd
import pytest

import tops.dynamic as dps

LINE = 'L7-8-1'


def k2a_model():
    import tops.ps_models.k2a as model_data
    return model_data.load()


def with_connected(model, category, connected):
    """Add a 'connected' column to a table; `connected` maps unit names to flags, the others are connected."""
    header, *rows = model[category]
    model[category] = [[*header, 'connected'], *[[*row, connected.get(row[0], True)] for row in rows]]
    return model


def without_unit(model, category, name):
    header, *rows = model[category]
    model[category] = [header, *[row for row in rows if row[0] != name]]
    return model


def initialized(model):
    ps = dps.PowerSystemModel(model=model)
    ps.init_dyn_sim()
    return ps


def y_bus_red(ps):
    return ps.y_bus_red.toarray()


def line_index(ps, name=LINE):
    return np.where(ps.lines['Line'].par['name'] == name)[0][0]


def tripped_after_init(name=LINE):
    ps = initialized(k2a_model())
    ps.apply_event('lines', 'Line', name, 'disconnect')
    return ps


def test_line_built_out_of_service_differs_from_a_tripped_line_only_by_the_loads():
    built = initialized(with_connected(k2a_model(), 'lines', {LINE: False}))
    tripped = tripped_after_init()

    assert not built.lines['Line'].connected[line_index(built)]
    assert built.lines['Line'].connected.sum() == len(built.lines['Line'].par) - 1

    # The branches are stamped alike. A constant-impedance load takes its admittance from the load flow,
    # which was solved without the line in one model and with it in the other.
    difference = y_bus_red(built) - y_bus_red(tripped)
    load_buses = built.loads['Load'].bus_idx_red['terminal']
    expected = np.zeros_like(difference)
    expected[load_buses, load_buses] = built.loads['Load'].y_load - tripped.loads['Load'].y_load
    assert np.abs(expected).max() > 1e-6
    assert np.allclose(difference, expected, atol=1e-10)


def test_line_built_out_of_service_is_solved_without_it():
    built = initialized(with_connected(k2a_model(), 'lines', {LINE: False}))
    removed = initialized(without_unit(k2a_model(), 'lines', LINE))

    assert np.allclose(built.v_0, removed.v_0)
    assert np.allclose(y_bus_red(built), y_bus_red(removed), atol=1e-10)
    # The load flow was solved without the line, so the simulation starts at an equilibrium
    assert max(abs(built.ode_fun(0, built.x0))) < 1e-8


def test_connecting_a_line_built_out_of_service_stamps_its_full_admittance():
    built = initialized(with_connected(k2a_model(), 'lines', {LINE: False}))
    before = y_bus_red(built)
    line_stamp = y_bus_red(initialized(k2a_model())) - y_bus_red(tripped_after_init())
    assert np.abs(line_stamp).max() > 1e-6

    built.apply_event('lines', 'Line', LINE, 'connect')

    assert np.allclose(y_bus_red(built) - before, line_stamp, atol=1e-10)
    assert built.lines['Line'].connected.all()


def test_current_of_a_line_built_out_of_service_is_zero():
    built = initialized(with_connected(k2a_model(), 'lines', {LINE: False}))
    line = built.lines['Line']
    i = line_index(built)
    assert line.i_from(built.x0, built.v_0)[i] == 0
    assert line.i_to(built.x0, built.v_0)[i] == 0
    assert abs(line.i_from(built.x0, built.v_0)[i - 1]) > 0


def test_all_connected_column_changes_nothing():
    plain = initialized(k2a_model())
    flagged = initialized(with_connected(k2a_model(), 'lines', {}))
    assert np.array_equal(y_bus_red(plain), y_bus_red(flagged))
    assert np.array_equal(plain.v_0, flagged.v_0)


@pytest.mark.parametrize('flags', [{LINE: 0}, {LINE: 0.0}], ids=['int', 'float'])
def test_zero_and_one_flags_are_accepted(flags):
    model = with_connected(k2a_model(), 'lines', flags)
    # Non-bool flags for the other rows too: 1 for connected
    model['lines'] = [model['lines'][0], *[[*row[:-1], int(row[-1])] for row in model['lines'][1:]]]
    built = initialized(model)
    assert not built.lines['Line'].connected[line_index(built)]
    assert built.lines['Line'].connected.sum() == len(built.lines['Line'].par) - 1


def test_a_data_frame_table_with_a_bool_column_is_accepted():
    model = with_connected(k2a_model(), 'lines', {LINE: False})
    model['lines'] = pd.DataFrame(model['lines'][1:], columns=model['lines'][0])
    built = initialized(model)
    assert not built.lines['Line'].connected[line_index(built)]


@pytest.mark.parametrize('flag', [float('nan'), 'yes', 2])
def test_invalid_flags_are_refused_naming_the_column(flag):
    model = with_connected(k2a_model(), 'lines', {LINE: flag})
    with pytest.raises(ValueError, match='connected'):
        dps.PowerSystemModel(model=model)


def test_init_dyn_sim_keeps_a_line_tripped_by_an_event_tripped():
    ps = initialized(k2a_model())
    ps.apply_event('lines', 'Line', LINE, 'disconnect')
    after_event = y_bus_red(ps)

    ps.init_dyn_sim()
    assert np.allclose(y_bus_red(ps), after_event, atol=1e-10)
    assert not ps.lines['Line'].connected[line_index(ps)]
