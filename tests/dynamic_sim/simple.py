import dynpssimpy.dynamic as dps
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import time
import dynpssimpy.utility_functions as dps_uf
import sys


if __name__ == '__main__':

    # Load model
    import ps_models.k2a as model_data
    model = model_data.load()

    # Power system model
    ps = dps.PowerSystemModel(model=model)
    # ps.use_numba = True
    ps.power_flow()
    ps.init_dyn_sim()

    # Add small perturbation to initial angle of first generator
    x0 = ps.x0.copy()
    x0[ps.gen_mdls['GEN'].state_idx['angle'][0]] += 1

    # Solver
    t_end = 5
    sol = dps_uf.ModifiedEuler(ps.ode_fun, 0, x0, t_end, max_step=5e-3)

    # Initialize simulation
    t = 0
    result_dict = defaultdict(list)
    t_0 = time.time()

    # Run simulation
    print('Running dynamic simulation')
    while t < t_end:
        sys.stdout.write('\rt={:.2f}s'.format(t))

        # Simulate next step
        result = sol.step()
        x = sol.y
        t = sol.t

        # Store result
        result_dict['Global', 't'].append(sol.t)
        [result_dict[tuple(desc)].append(state) for desc, state in zip(ps.state_desc, x)]

    print('\nSimulation completed in {:.2f} seconds.'.format(time.time() - t_0))

    # Convert dict to pandas dataframe
    index = pd.MultiIndex.from_tuples(result_dict)
    result = pd.DataFrame(result_dict, columns=index)

    # Plot angle and speed
    fig, ax = plt.subplots(2)
    ax[0].plot(result[('Global', 't')], result.xs(key='speed', axis='columns', level=1))
    ax[1].plot(result[('Global', 't')], result.xs(key='angle', axis='columns', level=1))
    plt.show()
