import numpy as np

import tops.dynamic as dps
from tops.simulator import Simulator, Events, ResultKeeper


def test_simulator_with_results_and_events():
    import tops.ps_models.k2a as model_data
    ps = dps.PowerSystemModel(model=model_data.load())
    ps.init_dyn_sim()

    dt = 5e-3
    t_end = 0.5
    sim = Simulator(ps, dt=dt, t_end=t_end)
    res_keeper = ResultKeeper(sim, gen_speed=ps.gen['GEN'].speed)
    events = Events(sim, [
        (0.1, ('line', 'L7-8-1', 'disconnect')),
        (0.3, ('line', 'L7-8-1', 'connect')),
    ])
    disconnected_at = []

    def check_line(sim):
        if not ps.lines['Line'].connected[ps.lines['Line'].par['name'] == 'L7-8-1'][0]:
            disconnected_at.append(sim.sol.t)

    sim.interface_functions['Events'] = events.update
    sim.interface_functions['ResultKeeper'] = res_keeper.update
    sim.interface_functions['CheckLine'] = check_line
    sim.main_loop()

    n_steps = len(res_keeper.t)
    assert sim.sol.t >= t_end
    assert n_steps == np.ceil(t_end / dt - 1e-9)

    # Results are stored for every step, for both states and the requested outputs
    df = res_keeper.get_dataframe()
    assert df.shape == (n_steps, ps.n_states)
    assert np.array(res_keeper.store['gen_speed']).shape == (n_steps, 4)

    # The line was out of service between the two events, and the system responded
    assert min(disconnected_at) >= 0.1 and max(disconnected_at) < 0.3 + dt
    assert ps.lines['Line'].connected.all()
    assert np.ptp(np.array(res_keeper.store['gen_speed'])) > 0


def test_result_keeper_calculate_replays_events():
    import tops.ps_models.k2a as model_data
    ps = dps.PowerSystemModel(model=model_data.load())
    ps.init_dyn_sim()

    load_name = ps.loads['Load'].par['name'][0]
    events = [
        ['t', 'category', 'model', 'unit', 'action', 'kwargs'],
        [0.05, 'buses', '', 'B8', 'short_circuit', None],
        [0.1, 'buses', '', 'B8', 'clear_short_circuit', None],
        [0.1, 'lines', 'Line', 'L7-8-1', 'disconnect', None],
        [0.2, 'loads', 'Load', load_name, 'set_power', {'P': 1000}],
        [0.3, 'lines', 'Line', 'L7-8-1', 'connect', None],
    ]
    sim = Simulator(ps, dt=5e-3, t_end=0.4, events=events)
    outputs = dict(speed=ps.gen['GEN'].speed, p_line=ps.lines['Line'].p_from, i_load=ps.loads['Load'].i)
    res_keeper = ResultKeeper(sim, **outputs)
    sim.interface_functions['ResultKeeper'] = res_keeper.update
    sim.main_loop()

    connected_after = ps.lines['Line'].connected.copy()
    y_load_after = ps.loads['Load'].y_load.copy()
    y_bus_red_after = ps.y_bus_red

    # The same values as stored during the simulation, also when calculated twice
    for _ in range(2):
        for name, fun in outputs.items():
            assert np.allclose(res_keeper.calculate(fun), np.array(res_keeper.store[name]))

    # The line was out of service between the events, so its flow was zero there
    p_line = res_keeper.calculate(ps.lines['Line'].p_from)[:, ps.lines['Line'].par['name'] == 'L7-8-1'][:, 0]
    t = np.array(res_keeper.t)
    assert np.all(p_line[(t > 0.1 + 5e-3) & (t < 0.3)] == 0)
    assert np.all(p_line[t < 0.1] != 0)

    # The model is left as it was after the simulation
    assert np.array_equal(ps.lines['Line'].connected, connected_after)
    assert np.array_equal(ps.loads['Load'].y_load, y_load_after)
    assert ps.y_bus_red is y_bus_red_after


def test_result_keeper_calculate_replays_event_functions():
    import tops.ps_models.k2a as model_data
    ps = dps.PowerSystemModel(model=model_data.load())
    ps.init_dyn_sim()

    events = [
        ['t', 'event', 'unit', 'kwargs'],
        [0.05, ps.bus_events.short_circuit, 'B8', None],
        [0.1, ps.bus_events.clear_short_circuit, 'B8', None],
        [0.1, ps.lines.Line.events.disconnect, 'L7-8-1', None],
        [0.2, ps.loads.Load.events.set_power, ps.loads['Load'].par['name'][0], {'P': 1000}],
    ]
    sim = Simulator(ps, dt=5e-3, t_end=0.3, events=events)
    res_keeper = ResultKeeper(sim, p_line=ps.lines['Line'].p_from, i_load=ps.loads['Load'].i)
    sim.interface_functions['ResultKeeper'] = res_keeper.update
    sim.main_loop()

    for name in ['p_line', 'i_load']:
        fun = ps.lines['Line'].p_from if name == 'p_line' else ps.loads['Load'].i
        assert np.allclose(res_keeper.calculate(fun), np.array(res_keeper.store[name]))
