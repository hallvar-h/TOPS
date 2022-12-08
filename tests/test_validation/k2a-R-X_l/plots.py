from collections import defaultdict
import tests.test_validation.validation_functions as val_fun
import dynpssimpy.dynamic as dps
import importlib
from scipy.integrate import RK45
import pandas as pd

if __name__ == '__main__':

    import dynpssimpy.ps_models.k2a as model_data
    model = model_data.load()

    [importlib.reload(mdl) for mdl in [model_data, dps]]

    ps = dps.PowerSystemModel(model=model)
    ps.avr['SEXS']['T_a'] = 2
    ps.avr['SEXS']['T_e'] = 0.1
    ps.gov['TGOV1']['T_1'] = 0.5
    ps.gov['TGOV1']['T_2'] = 1
    ps.gov['TGOV1']['T_3'] = 2

    ps.power_flow()
    ps.init_dyn_sim()

    t_end = 10
    max_step = 5e-3

    # PowerFactory result
    pf_res = val_fun.load_pf_res('tests/validation/k2a-R-X_l/powerfactory_res.csv')

    # Python result
    x0 = ps.x0
    sol = RK45(ps.ode_fun, 0, x0, t_end, max_step=5e-3)

    t = 0
    result_dict = defaultdict(list)

    print('Running dynamic simulation')
    while t < t_end:
        # print(t)

        # Simulate next step
        result = sol.step()
        t = sol.t

        if t>=1 and t<=1.1:
            # print('Event!')
            ps.y_bus_red_mod[0, 0] = 1e6
        else:
            ps.y_bus_red_mod[0, 0] = 0

        # Store result variables
        result_dict['Global', 't'].append(sol.t)
        [result_dict[tuple(desc)].append(state) for desc, state in zip(ps.state_desc, sol.y)]

    index = pd.MultiIndex.from_tuples(result_dict)
    result = pd.DataFrame(result_dict, columns=index)

    # Computing the error
    val_fun.generate_plots(ps, result, pf_res, max_step, choose_plots='basic')

    print(val_fun.compute_error(ps, result, pf_res, max_step))