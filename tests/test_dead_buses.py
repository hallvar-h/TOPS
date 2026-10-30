"""Buses whose row of the admittance matrix has vanished: a bay node whose switches are all open."""
import warnings

import numpy as np
import pytest
from scipy.sparse.linalg import MatrixRankWarning

import tops.dynamic as dps
from k2a_variants import bay_model, initialized, k2a_model
from tops.events import DeadBusWarning
from tops.simulator import Simulator

EVENT_COLUMNS = ['t', 'category', 'model', 'unit', 'action']


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


# Pinned to 0 V
def bus(ps, name):
    return list(ps.buses['name']).index(name)


def solve(ps):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return ps.solve_algebraic(0, ps.x0)


@pytest.mark.parametrize('switches', [('LS',), ('LS1', 'LS2', 'LS3')], ids=['exact zero', 'rounding left'])
def test_a_dead_bus_reads_zero_volts_and_the_others_do_not_notice(switches):
    ps = initialized(bay_model(switches))
    before = solve(ps)
    assert abs(before[bus(ps, 'B12')]) > 0.5  # the node follows B5 while a switch is closed

    trip(ps, *switches)
    after = solve(ps)

    assert after[bus(ps, 'B12')] == 0
    others = np.arange(len(after)) != bus(ps, 'B12')
    assert np.isfinite(after).all()
    assert np.allclose(after[others], before[others], atol=1e-9)  # the bay carried no current


def test_without_pinning_a_dead_bus_still_makes_the_matrix_singular():
    ps = initialized(bay_model())
    ps.pin_dead_buses = False
    trip(ps, 'LS')
    with pytest.warns(MatrixRankWarning):
        assert not np.isfinite(ps.solve_algebraic(0, ps.x0)).all()


def test_pinning_is_on_by_default_and_the_warning_is_a_runtime_warning():
    assert dps.PowerSystemModel(model=k2a_model()).pin_dead_buses is True
    assert issubclass(DeadBusWarning, RuntimeWarning)


def test_the_warning_names_the_dead_buses_and_comes_once_per_change_not_per_step():
    ps = initialized(bay_model())
    trip(ps, 'LS')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        for _ in range(5):
            ps.solve_algebraic(0, ps.x0)
    assert [type(w.message) for w in caught] == [DeadBusWarning]
    assert 'B12' in str(caught[0].message)

    with warnings.catch_warnings(record=True) as caught:  # the bus comes back, then is lost again: a change
        warnings.simplefilter('always')
        ps.apply_event('lines', 'Line', 'LS', 'connect')
        ps.solve_algebraic(0, ps.x0)
        trip(ps, 'LS')
        ps.solve_algebraic(0, ps.x0)
        ps.solve_algebraic(0, ps.x0)
    assert [type(w.message) for w in caught] == [DeadBusWarning]


def test_the_warning_can_be_silenced():
    ps = initialized(bay_model())
    trip(ps, 'LS')
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        warnings.filterwarnings('ignore', category=DeadBusWarning)
        assert np.isfinite(ps.solve_algebraic(0, ps.x0)).all()


def test_a_model_without_dead_buses_is_untouched_and_never_warns():
    pinned, free = initialized(k2a_model()), initialized(k2a_model())
    free.pin_dead_buses = False
    trip(pinned, 'L7-8-1')
    trip(free, 'L7-8-1')
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert np.array_equal(pinned.solve_algebraic(0, pinned.x0), free.solve_algebraic(0, free.x0))


def test_a_bus_that_comes_back_is_solved_again():
    ps = initialized(bay_model())
    trip(ps, 'LS')
    assert solve(ps)[bus(ps, 'B12')] == 0
    ps.apply_event('lines', 'Line', 'LS', 'connect')
    v = solve(ps)
    assert np.isclose(v[bus(ps, 'B12')], v[bus(ps, 'B5')])


def test_a_bus_fed_through_a_dyn_trafo_is_pinned_when_the_trafo_is_tripped():
    ps = initialized(bay_model(via='DynTrafo'))
    v = solve(ps)
    assert abs(v[bus(ps, 'B12')]) > 0.5

    trip(ps, 'TS', category='trafos', model='DynTrafo')
    assert solve(ps)[bus(ps, 'B12')] == 0


def test_a_simulation_goes_on_past_the_loss_of_a_bay_and_the_grid_does_not_notice():
    events = [EVENT_COLUMNS, [0.1, 'lines', 'Line', 'LS', 'disconnect']]
    plain, bay = initialized(k2a_model()), initialized(bay_model())
    speeds = []
    for ps, table in ((plain, None), (bay, events)):
        sim = Simulator(ps, dt=5e-3, t_end=0.5, events=table)
        speed = []
        sim.interface_functions['speed'] = lambda sim, ps=ps, speed=speed: speed.append(
            ps.gen['GEN'].speed(sim.sol.x, sim.sol.v).copy())
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            sim.main_loop()
        speeds.append(np.array(speed))
        assert [type(w.message) for w in caught] == ([] if ps is plain else [DeadBusWarning])

    assert np.isfinite(speeds[1]).all()
    assert np.allclose(speeds[0], speeds[1], atol=1e-9)
