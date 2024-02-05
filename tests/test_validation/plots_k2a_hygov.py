from collections import defaultdict
import validation_functions as val_fun
import tops.dynamic as dps
import importlib
from scipy.integrate import RK45
import pandas as pd
import sys
import pathlib


if __name__ == '__main__':

    import tops.ps_models.k2a_val as model_data
    model = model_data.load()

    [importlib.reload(mdl) for mdl in [model_data, dps]]

    model['gov'] = {
        'TGOV1': [model['gov']['TGOV1'][0], *model['gov']['TGOV1'][2:]],
        'HYGOV': [
            ['name',    'gen',  'R',    'r',    'T_f',  'T_r',  'T_g',  'A_t',  'T_w',  'q_nl',     'D_turb' ,      'G_min',    'V_elm',    'G_max',    'P_N'   ],
            ['HYGOV1',  'G1',   0.04,   0.1,    0.1,    10,     0.5,      1,      1,      0.01,      0.01,          0,          0.15,       1,          0       ]
    ]}

    ps = dps.PowerSystemModel(model=model)
    ps.init_dyn_sim()

    t_end = 30
    max_step = 5e-3

    # PowerFactory result

    # os.chdir()
    file_path = pathlib.Path(__file__).parent
    pf_res = val_fun.load_pf_res(str(file_path) + '/k2a_hygov/powerfactory_res.csv')

    # Python result
    x0 = ps.x0
    sol = RK45(ps.ode_fun, 0, x0, t_end, max_step=5e-3)

    t = 0
    result_dict = defaultdict(list)
    load_bus_idx = ps.loads['Load'].bus_idx_red['terminal'][0]
    y_load_0 = ps.loads['Load'].y_load.real[0]


    print('Running dynamic simulation')
    while t < t_end:
        sys.stdout.write("\r%d%%" % (t / (t_end) * 100))

        # Simulate next step
        result = sol.step()
        t = sol.t

        if t > 1:
            ps.y_bus_red_mod[(load_bus_idx,) * 2] = y_load_0*0.1
        # if t>=1 and t<=1.1:
        #     # print('Event!')
        #     ps.y_bus_red_mod[(sc_bus_idx,)*2] = 1e6
        # else:
        #     ps.y_bus_red_mod[(sc_bus_idx,)*2] = 0

        # Store result variables
        result_dict['Global', 't'].append(sol.t)
        [result_dict[tuple(desc)].append(state) for desc, state in zip(ps.state_desc, sol.y)]

    index = pd.MultiIndex.from_tuples(result_dict)
    result = pd.DataFrame(result_dict, columns=index)

    # Computing the error
    val_fun.generate_plots(ps, result, pf_res, max_step, choose_plots='basic')

    print(val_fun.compute_error(ps, result, pf_res, max_step))