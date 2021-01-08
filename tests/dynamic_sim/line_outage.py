import dynpssimpy.dynamic as dps
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import dynpssimpy.utility_functions as dps_uf
import importlib
import sys
import time


if __name__ == '__main__':
    importlib.reload(dps)

    # Load model
    import ps_models.k2a as model_data
    # import ps_models.ieee39 as model_data
    # import ps_models.sm_ib as model_data
    # import ps_models.sm_load as model_data
    model = model_data.load()

    ps = dps.PowerSystemModel(model=model)
    ps.pf_max_it = 100
    ps.power_flow()
    ps.init_dyn_sim()

    # Solver
    t_end = 10
    sol = dps_uf.ModifiedEuler(ps.ode_fun, 0, ps.x0, t_end, max_step=10e-3)

    t = 0
    result_dict = defaultdict(list)
    t_0 = time.time()

    event_flag = True
    while t < t_end:
        sys.stdout.write('\rt={:.2f}s'.format(t))

        # Simulate next step
        result = sol.step()
        x = sol.y
        t = sol.t

        if t > 1 and event_flag:
            event_flag = False
            ps.network_event('line', 'L7-8-1', 'disconnect')

        # Store result
        result_dict['Global', 't'].append(sol.t)
        [result_dict[tuple(desc)].append(state) for desc, state in zip(ps.state_desc, x)]

    print('\nSimulation completed in {:.2f} seconds.'.format(time.time() - t_0))

    index = pd.MultiIndex.from_tuples(result_dict)
    result = pd.DataFrame(result_dict, columns=index)

    fig, ax = plt.subplots(2)
    ax[0].plot(result[('Global', 't')], result.xs(key='speed', axis='columns', level=1))
    ax[1].plot(result[('Global', 't')], result.xs(key='angle', axis='columns', level=1))
    plt.show()
