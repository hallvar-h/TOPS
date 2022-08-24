from collections import defaultdict
import tests.validation.validation_functions as val_fun
import sys
sys.path.append(r'tests/validation')
import dynpssimpy.dynamic as dps
import importlib
from scipy.integrate import RK45
import pandas as pd
importlib.reload(val_fun)


if __name__ == '__main__':

    import dynpssimpy.ps_models.ieee39 as model_data
    model = model_data.load()

    [importlib.reload(mdl) for mdl in [model_data, dps, val_fun]]

    ps = dps.PowerSystemModel(model=model)

    ps.pf_max_it = 100
    # ps.setup()
    # ps.power_flow()
    ps.init_dyn_sim()

    t_end = 10
    max_step = 5e-3

    # PowerFactory result
    pf_res = val_fun.load_pf_res('tests/validation/ieee39/powerfactory_res.csv')

    x0 = ps.x0

    sol = RK45(ps.ode_fun, 0, x0, t_end, max_step=5e-3)

    t = 0
    result_dict = defaultdict(list)
    sc_bus_idx = ps.gen['GEN'].bus_idx_red['terminal'][0]

    print('Running dynamic simulation')
    while t < t_end:
        sys.stdout.write("\r%d%%" % (t / (t_end) * 100))

        # Simulate next step
        result = sol.step()
        t = sol.t

        if t >= 1 and t <= 1.05:
            # print('Event!')
            ps.y_bus_red_mod[(sc_bus_idx,)*2] = 1e6
        else:
            ps.y_bus_red_mod[(sc_bus_idx,)*2] = 0

        # Store result variables
        result_dict['Global', 't'].append(sol.t)
        [result_dict[tuple(desc)].append(state) for desc, state in zip(ps.state_desc, sol.y)]



    index = pd.MultiIndex.from_tuples(result_dict)
    result = pd.DataFrame(result_dict, columns=index)

    # Computing the error
    val_fun.generate_plots(ps, result, pf_res, max_step, choose_plots='expanded')

    print(val_fun.compute_error(ps, result, pf_res, max_step))