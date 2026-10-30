"""A PLL on a bus that has lost its voltage holds what it had, instead of following an angle that is not there."""
import warnings

import numpy as np
import pytest

from k2a_variants import bay_model, initialized
from tops.simulator import Simulator

EVENT_COLUMNS = ['t', 'category', 'model', 'unit', 'action']


def pll_model(kind):
    """The bay model, with a PLL on the bay node B12 and on B5, which stays alive."""
    model = bay_model()
    if kind == 'PLL1':
        table = [['name', 'T_filter', 'bus'], ['PLL_B12', 0.01, 'B12'], ['PLL_B5', 0.01, 'B5']]
    else:
        table = [['name', 'K_p', 'K_i', 'bus'], ['PLL_B12', 100, 100, 'B12'], ['PLL_B5', 100, 100, 'B5']]
    model['pll'] = {kind: table}
    return model


def solve_after_trip(kind):
    ps = initialized(pll_model(kind))
    ps.apply_event('lines', 'Line', 'LS', 'disconnect')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return ps, ps.solve_algebraic(0, ps.x0)


@pytest.mark.parametrize('kind', ['PLL1', 'PLL2'])
def test_a_pll_on_a_live_bus_is_at_rest_and_follows_the_angle(kind):
    ps = initialized(pll_model(kind))
    pll = ps.pll[kind]
    assert np.allclose(pll.freq_est(ps.x0, ps.v_0), 0, atol=1e-9)
    assert np.allclose(pll.output(ps.x0, ps.v_0), np.angle(ps.v_0[[list(ps.buses['name']).index(b)
                                                                    for b in ('B12', 'B5')]]))


@pytest.mark.parametrize('kind', ['PLL1', 'PLL2'])
def test_a_pll_on_a_dead_bus_reports_no_frequency_deviation(kind):
    ps, v = solve_after_trip(kind)
    assert v[list(ps.buses['name']).index('B12')] == 0

    freq = ps.pll[kind].freq_est(ps.x0, v)
    assert np.isfinite(freq).all()
    assert abs(freq[0]) < 1e-9  # PLL_B12; a filter chasing the angle 0 of no voltage would give about -0.14 p.u.


def test_the_pll_of_a_live_bus_does_not_notice_the_dead_one():
    ps, v = solve_after_trip('PLL1')
    assert abs(ps.pll['PLL1'].freq_est(ps.x0, v)[1]) < 1e-9


@pytest.mark.parametrize('kind', ['PLL1', 'PLL2'])
def test_a_simulation_keeps_the_pll_of_a_dead_bus_where_it_was(kind):
    ps = initialized(pll_model(kind))
    events = [EVENT_COLUMNS, [0.1, 'lines', 'Line', 'LS', 'disconnect']]
    sim = Simulator(ps, dt=5e-3, t_end=0.5, events=events)
    freq, angle = [], []

    def record(sim):
        freq.append(ps.pll[kind].freq_est(sim.sol.x, sim.sol.v).copy())
        angle.append(ps.pll[kind].output(sim.sol.x, sim.sol.v).copy())

    sim.interface_functions['record'] = record
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        sim.main_loop()
    freq, angle = np.array(freq), np.array(angle)

    after = np.array([t >= 0.11 for t in np.arange(1, len(freq) + 1)*5e-3])
    assert np.isfinite(freq).all() and np.isfinite(angle).all()
    assert np.abs(freq[after, 0]).max() < 1e-6
    assert np.ptp(angle[after, 0]) < 1e-6
