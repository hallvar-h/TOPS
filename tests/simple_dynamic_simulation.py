import dynpssimpy.dynamic as dps
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import RK23, RK45, solve_ivp
import importlib


if __name__ == '__main__':

    # Load model
    import ps_models.k2a as model_data
    # import ps_models.ieee39 as model_data
    # import ps_models.sm_ib as model_data
    # import ps_models.sm_load as model_data
    model = model_data.load()

    eq = dps.PowerSystemModel(model=model)
    eq.pf_max_it = 100
    eq.power_flow()
    eq.init_dyn_sim()

    t_end = 5
    x0 = eq.x0.copy()
    x0[eq.angle_idx[0]] += 1
    np.max(eq.ode_fun(0, eq.x0))

    sol = RK23(eq.ode_fun, 0, x0, t_end, max_step=20e-3)

    t = 0
    result_dict = defaultdict(list)

    while t < t_end:
        print(t)

        # Simulate next step
        result = sol.step()
        x = sol.y
        t = sol.t

        # Store result
        result_dict['Global', 't'].append(sol.t)
        [result_dict[tuple(desc)].append(state) for desc, state in zip(eq.state_desc, x)]

    index = pd.MultiIndex.from_tuples(result_dict)
    result = pd.DataFrame(result_dict, columns=index)

    fig, ax = plt.subplots(2)
    ax[0].plot(result[('Global', 't')], result.xs(key='speed', axis='columns', level=1))
    ax[1].plot(result[('Global', 't')], result.xs(key='angle', axis='columns', level=1))
