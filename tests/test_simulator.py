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
