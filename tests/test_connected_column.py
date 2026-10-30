"""The optional 'connected' column of the line and transformer tables."""
import numpy as np
import pandas as pd
import pytest

import tops.dynamic as dps
from k2a_variants import (KINDS, TRAFO, initialized, k2a_model, removed_trafo_model, trafo_index, trafo_model,
                          with_connected, y_bus_red)

LINE = 'L7-8-1'


def without_unit(model, category, name):
    header, *rows = model[category]
    model[category] = [header, *[row for row in rows if row[0] != name]]
    return model


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


# Transformers
@pytest.mark.parametrize('kind', KINDS)
def test_trafo_built_out_of_service_is_solved_without_it(kind):
    built = initialized(trafo_model(kind, {TRAFO: False}))
    removed = initialized(removed_trafo_model(kind))
    trafo = built.trafos[kind]

    assert not trafo.connected[trafo_index(built, kind)]
    assert trafo.connected.sum() == len(trafo.par) - 1
    assert np.allclose(built.v_0, removed.v_0)
    assert np.allclose(y_bus_red(built), y_bus_red(removed), atol=1e-10)
    assert np.allclose(built.solve_algebraic(0, built.x0), removed.solve_algebraic(0, removed.x0))
    assert max(abs(built.ode_fun(0, built.x0))) < 1e-8


@pytest.mark.parametrize('kind', KINDS)
def test_trafo_out_of_service_keeps_its_admittance_for_a_later_connect(kind):
    built = initialized(trafo_model(kind, {TRAFO: False}))
    full = initialized(trafo_model(kind))
    i = trafo_index(built, kind)

    data, _ = built.trafos[kind].load_flow_adm()
    full_data, _ = full.trafos[kind].load_flow_adm()
    assert (data[:, i] == 0).all()
    assert np.array_equal(np.delete(data, i, axis=1), np.delete(full_data, i, axis=1))
    assert built.trafos[kind].admittance[i] != 0
    assert built.trafos[kind].ratio_from_0[i] != 0 and built.trafos[kind].ratio_to_0[i] != 0


@pytest.mark.parametrize('kind', KINDS)
def test_current_of_a_trafo_built_out_of_service_is_zero(kind):
    built = initialized(trafo_model(kind, {TRAFO: False}))
    trafo = built.trafos[kind]
    i = trafo_index(built, kind)
    for current in (trafo.i_from(built.x0, built.v_0), trafo.i_to(built.x0, built.v_0)):
        assert current[i] == 0
        assert (abs(np.delete(current, i)) > 0).all()


@pytest.mark.parametrize('kind', KINDS)
def test_all_connected_trafo_column_changes_nothing(kind):
    plain = initialized(trafo_model(kind))
    flagged = initialized(trafo_model(kind, {}))
    assert np.array_equal(y_bus_red(plain), y_bus_red(flagged))
    assert np.array_equal(plain.v_0, flagged.v_0)
    assert np.array_equal(plain.solve_algebraic(0, plain.x0), flagged.solve_algebraic(0, flagged.x0))


def test_trafo_zero_and_one_flags_and_data_frames_are_accepted():
    model = trafo_model('Trafo', {TRAFO: False})
    header, *rows = model['trafos']['Trafo']
    as_ints = [header, *[[*row[:-1], int(row[-1])] for row in rows]]
    model['trafos'] = {'Trafo': pd.DataFrame(as_ints[1:], columns=as_ints[0])}
    built = initialized(model)
    assert not built.trafos['Trafo'].connected[trafo_index(built, 'Trafo')]
    assert built.trafos['Trafo'].connected.sum() == len(built.trafos['Trafo'].par) - 1


@pytest.mark.parametrize('kind', KINDS)
@pytest.mark.parametrize('flag', [float('nan'), 'yes', 2])
def test_invalid_trafo_flags_are_refused_naming_the_column(kind, flag):
    with pytest.raises(ValueError, match='connected'):
        dps.PowerSystemModel(model=trafo_model(kind, {TRAFO: flag}))


# Islands. In K2A, T1 is the only link of B1, and T3 the only link of the slack bus B3.
def test_a_branch_out_of_service_that_cuts_a_bus_off_from_the_slack_is_refused():
    ps = dps.PowerSystemModel(model=with_connected(k2a_model(), 'transformers', {'T1': False}))
    with pytest.raises(ValueError, match=r"connected.*\bB1\b"):
        ps.init_dyn_sim()
    assert not ps.power_flow_ready


def test_every_bus_cut_off_is_named():
    ps = dps.PowerSystemModel(model=with_connected(k2a_model(), 'transformers', {'T3': False}))
    with pytest.raises(ValueError) as error:
        ps.power_flow()
    assert all(f'B{i}' in str(error.value) for i in [1, 2, 4, 5, 6, 7, 8, 9, 10, 11])
    assert 'B3' not in str(error.value).replace('slack bus B3', '')


def test_a_dyn_trafo_out_of_service_that_cuts_a_bus_off_is_refused():
    ps = dps.PowerSystemModel(model=trafo_model('DynTrafo', {'T1': False, TRAFO: False}))
    with pytest.raises(ValueError, match=r"connected.*\bB1\b"):
        ps.power_flow()


def test_a_branch_out_of_service_that_leaves_the_grid_connected_is_accepted():
    ps = initialized(with_connected(k2a_model(), 'lines', {LINE: False}))
    assert ps.power_flow_ready


def test_buses_without_any_branch_are_not_refused_when_everything_is_connected():
    ps = dps.PowerSystemModel(model=k2a_model())
    ps.setup()
    ps.buses = np.append(ps.buses, np.array([('B12', 20)], dtype=ps.buses.dtype))
    ps.n_bus += 1
    ps._check_islands(0)
