"""Buses whose row of the admittance matrix has vanished: a bay node whose switches are all open."""
import numpy as np
import pytest

from k2a_variants import bay_model, initialized, k2a_model


def trip(ps, *names, category='lines', model='Line'):
    for name in names:
        ps.apply_event(category, model, name, 'disconnect')


def test_a_model_without_floating_buses_has_no_dead_bus():
    assert len(initialized(k2a_model()).dead_buses()) == 0
    assert len(initialized(bay_model()).dead_buses()) == 0


def test_a_bay_node_is_dead_once_its_only_switch_is_open():
    ps = initialized(bay_model())
    trip(ps, 'LS')
    assert list(ps.dead_buses()) == ['B12']


def test_a_bay_node_is_not_dead_while_one_of_its_switches_is_closed():
    ps = initialized(bay_model(('LS1', 'LS2')))
    trip(ps, 'LS1')
    assert len(ps.dead_buses()) == 0
    trip(ps, 'LS2')
    assert list(ps.dead_buses()) == ['B12']


def test_rounding_left_by_the_stamps_is_not_admittance():
    # The row is the sum of three large stamps, taken out one by one: what is left is rounding, not admittance
    ps = initialized(bay_model(('LS1', 'LS2', 'LS3')))
    bus = list(ps.buses['name']).index('B12')
    trip(ps, 'LS1', 'LS2', 'LS3')

    residue = abs(ps.y_bus_red[bus]).sum()
    assert 0 < residue < 1e-9  # it is there (else this test proves nothing), against about 5e3 before
    assert list(ps.dead_buses()) == ['B12']


def test_a_small_shunt_keeps_a_bus_alive_next_to_large_switches():
    ps = initialized(bay_model(('LS1', 'LS2'), shunt_mvar=0.009))  # 1e-5 p.u. against about 1e3 p.u. per switch
    trip(ps, 'LS1', 'LS2')
    assert len(ps.dead_buses()) == 0


def test_a_bus_fed_only_through_a_dyn_trafo_is_alive_until_it_is_tripped():
    ps = initialized(bay_model(via='DynTrafo'))
    assert len(ps.dead_buses()) == 0
    trip(ps, 'TS', category='trafos', model='DynTrafo')
    assert list(ps.dead_buses()) == ['B12']


def test_a_bus_that_is_connected_again_is_no_longer_dead():
    ps = initialized(bay_model())
    trip(ps, 'LS')
    assert list(ps.dead_buses()) == ['B12']
    ps.apply_event('lines', 'Line', 'LS', 'connect')
    assert len(ps.dead_buses()) == 0


def test_dead_buses_do_not_change_the_solution_of_the_network_equations():
    ps = initialized(bay_model())
    before = ps.solve_algebraic(0, ps.x0)
    ps.dead_buses()
    assert np.array_equal(ps.solve_algebraic(0, ps.x0), before)
